# HuggingFace 模型抓取：批次脚本与内存管理完全指南

> Worker2 实战经验 + 新 Agent 继承工作流
> 最后更新: 2026-03-31

---

## 目录

1. [批次脚本架构](#一批次脚本架构)
2. [内存管理实战](#二内存管理实战)
3. [新 Agent 继承工作流](#三新-agent-继承工作流)
4. [监控与运维](#四监控与运维)

---

## 一、批次脚本架构

### 1.1 多 Worker 并行架构

```
HuggingFace 模型抓取系统
├── Worker1 (idx 0-50000)
│   ├── 进程1: idx 0-3750
│   ├── 进程2: idx 3750-7500
│   └── ...
├── Worker2 (idx 50000-150000)
│   ├── 进程1: idx 94216-96350
│   ├── 进程2: idx 96351-98485
│   └── ... (8个活跃进程)
└── Worker3-5 (其他批次)
```

### 1.2 推荐批次脚本

#### 脚本 A: extract_from_model_list.py (顺序处理)

```python
#!/usr/bin/env python3
"""从模型列表顺序抓取 - 适用于大规模索引范围"""
import os
import sys
import argparse

# 环境设置
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["http_proxy"] = "http://agent.baidu.com:8891"
os.environ["https_proxy"] = "http://agent.baidu.com:8891"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

sys.path.insert(0, '/root/GraphNet')
from graph_net.torch.extractor import extract as torch_extract
from transformers import AutoConfig, AutoModel
import torch

# 不支持的模型格式
UNSUPPORTED_SUFFIXES = ['-gguf', '-awq', '-8bits', '-4bits', '-gptq']

def is_supported_model(model_id):
    model_lower = model_id.lower()
    return not any(suffix in model_lower for suffix in UNSUPPORTED_SUFFIXES)

def extract_model(model_id, output_dir):
    """抓取单个模型"""
    try:
        task = categorize_model(model_id)
        model_name = model_id.replace("/", "_")
        model_path = os.path.join(output_dir, task, model_name)
        os.makedirs(model_path, exist_ok=True)

        # 跳过已存在
        if os.path.exists(os.path.join(model_path, "model.py")):
            return True

        # 加载和提取
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModel.from_config(config, trust_remote_code=True)
        model.eval()

        vocab_size = getattr(config, 'vocab_size', 32000)
        max_pos = getattr(config, 'max_position_embeddings', 512)
        seq_len = min(32, max_pos)

        # 包装和运行
        wrapped = torch_extract(model_name)(model)
        input_ids = torch.randint(0, min(vocab_size, 1000), (1, seq_len))

        with torch.no_grad():
            _ = wrapped(input_ids)

        # 内存释放 (关键!)
        del wrapped
        del model
        import gc
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return True

    except Exception as e:
        print(f"Error: {model_id} - {e}")
        return False

def categorize_model(model_id):
    """推断任务类型"""
    model_lower = model_id.lower()
    if any(kw in model_lower for kw in ['gpt', 'llama', 'mistral', 'qwen']):
        return 'text-generation'
    elif any(kw in model_lower for kw in ['sentiment', 'classification']):
        return 'text-classification'
    else:
        return 'feature-extraction'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-idx', type=int, required=True)
    parser.add_argument('--end-idx', type=int, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    args = parser.parse_args()

    # 加载模型列表
    models = load_models_from_list(args.start_idx, args.end_idx)

    for i, model_id in enumerate(models, 1):
        print(f"[{i}/{len(models)}] {model_id}")
        extract_model(model_id, args.output_dir)

if __name__ == "__main__":
    main()
```

**使用方式:**
```bash
python3 extract_from_model_list.py \
    --start-idx 94216 \
    --end-idx 96350 \
    --output-dir /root/graphnet_workspace/huggingface/worker2
```

---

#### 脚本 B: batch_repair_worker_ducc.py (子进程隔离)

```python
#!/usr/bin/env python3
"""Ducc批量修复 - 支持多worker并行，子进程隔离"""
import os
import sys
import json
import subprocess
import argparse
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["http_proxy"] = "http://agent.baidu.com:8891"
os.environ["https_proxy"] = "http://agent.baidu.com:8891"

WORKSPACE = "/root/graphnet_workspace/huggingface"

def generate_extract_script(model_dir, model_id, task_type):
    """生成任务特定的 extract.py"""
    model_class = get_model_class(task_type)

    content = f'''#!/usr/bin/env python3
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["http_proxy"] = "http://agent.baidu.com:8891"

import torch
import json
import hashlib
import sys
sys.path.insert(0, '/root/GraphNet')
from graph_net.torch.extractor import extract as torch_extract
from transformers import AutoConfig, {model_class}, AutoModel

MODEL_ID = "{model_id}"
OUTPUT_DIR = "."

def main():
    try:
        # 清理旧文件
        for f in ["graph_net.json", "graph_hash.txt", "model.py"]:
            if os.path.exists(f):
                os.remove(f)

        config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
        try:
            model = {model_class}.from_config(config, trust_remote_code=True)
        except:
            model = AutoModel.from_config(config, trust_remote_code=True)
        model.eval()

        vocab_size = getattr(config, 'vocab_size', 32000)
        input_ids = torch.randint(0, min(vocab_size, 1000), (1, 32))

        with torch.no_grad():
            _ = model(input_ids)

        result = torch_extract(
            model=model,
            model_name=MODEL_ID,
            output_path=OUTPUT_DIR,
            input_shape=(1, 32),
            source="huggingface",
            task="{task_type}"
        )

        # 生成 hash
        if os.path.exists("graph_net.json"):
            with open("graph_net.json", 'r') as f:
                graph_data = json.load(f)
            graph_hash = hashlib.md5(
                json.dumps(graph_data, sort_keys=True).encode()
            ).hexdigest()[:16]
            with open("graph_hash.txt", 'w') as f:
                f.write(graph_hash)
            print(f"Success! Hash: {{graph_hash}}")
            return True
        return False

    except Exception as e:
        print(f"Error: {{e}}")
        return False

if __name__ == "__main__":
    main()
'''
    extract_path = os.path.join(model_dir, "extract.py")
    with open(extract_path, 'w') as f:
        f.write(content)
    os.chmod(extract_path, 0o755)
    return extract_path

def run_extract(model_dir, timeout=180):
    """在子进程中运行提取 - 内存完全隔离!"""
    extract_path = os.path.join(model_dir, "extract.py")

    try:
        result = subprocess.run(
            [sys.executable, extract_path],
            cwd=model_dir,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        # 子进程退出 = 内存完全释放
        success = result.returncode == 0
        return success, "Success" if success else result.stderr[:200]
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)[:200]

def process_batch(batch_id, task_types, start_idx, end_idx):
    """处理一批模型"""
    models = load_models(task_types, start_idx, end_idx)
    log_file = f"/tmp/repair_batch_{batch_id}.log"

    success_count = 0
    with open(log_file, 'w') as f:
        for i, model in enumerate(models, 1):
            # 生成脚本并运行
            generate_extract_script(model['path'], model['id'], model['task'])
            success, msg = run_extract(model['path'])

            status = "SUCCESS" if success else "FAILED"
            f.write(f"[{i}/{len(models)}] {model['id']}: {status}\n")
            f.flush()

            if success:
                success_count += 1

    return batch_id, success_count, len(models) - success_count

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-id', type=int, required=True)
    parser.add_argument('--task-types', type=str, required=True)
    parser.add_argument('--start-idx', type=int, required=True)
    parser.add_argument('--end-idx', type=int, required=True)
    args = parser.parse_args()

    task_types = args.task_types.split(',')
    batch_id, success, failed = process_batch(
        args.batch_id, task_types, args.start_idx, args.end_idx
    )
    print(f"Batch {batch_id}: success={success}, failed={failed}")

if __name__ == "__main__":
    main()
```

**使用方式:**
```bash
python3 batch_repair_worker_ducc.py \
    --batch-id 1 \
    --task-types text-generation,translation \
    --start-idx 0 \
    --end-idx 100
```

---

### 1.3 多进程启动脚本

```bash
#!/bin/bash
# start_workers.sh - 启动多个 Worker2 进程

OUTPUT_DIR="/root/graphnet_workspace/huggingface/worker2"

# Worker2 的批次范围
BATCHES=(
    "94216:96350"
    "96351:98485"
    "102756:104890"
    "104891:107025"
    "113431:115565"
    "124106:126240"
    "128376:130510"
    "136916:139050"
)

for batch in "${BATCHES[@]}"; do
    START=$(echo $batch | cut -d: -f1)
    END=$(echo $batch | cut -d: -f2)

    echo "启动 Worker: $START - $END"
    nohup python3 /root/extract_from_model_list.py \
        --start-idx $START \
        --end-idx $END \
        --output-dir $OUTPUT_DIR \
        > /tmp/worker_${START}_${END}.log 2>&1 &

done

echo "所有 Worker 已启动"
ps aux | grep extract_from_model_list | grep -v grep
```

---

## 二、内存管理实战

### 2.1 内存监控守护进程

```bash
#!/bin/bash
# monitor_memory.sh - 内存监控子代理

THRESHOLD_KB=67108864  # 64GB
LOG_FILE="monitor.log"

echo "$(date): 内存监控启动，阈值 ${THRESHOLD_KB}KB" >> $LOG_FILE

while true; do
    for pid in $(pgrep -f "extract_from_model_list" | head -20); do
        # 读取进程内存 (KB)
        mem_kb=$(cat /proc/$pid/status 2>/dev/null | grep VmRSS | awk '{print $2}')

        if [ -n "$mem_kb" ] && [ "$mem_kb" -gt $THRESHOLD_KB ]; then
            echo "$(date): PID $pid using ${mem_kb}KB (>64GB), killing..." >> $LOG_FILE
            kill -9 $pid 2>/dev/null
        fi
    done
    sleep 60
done
```

**启动:**
```bash
chmod +x monitor_memory.sh
nohup ./monitor_memory.sh >> monitor.log 2>&1 &
```

### 2.2 代码层内存清理

```python
import gc
import torch

def cleanup_memory():
    """标准内存清理"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def aggressive_cleanup():
    """激进内存清理 - 处理顽固泄漏"""
    # 强制删除张量
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                del obj
        except:
            pass

    # 多次GC
    gc.collect()
    gc.collect()

    # 同步和清理CUDA
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
```

### 2.3 ulimit 系统限制

```bash
# 设置虚拟内存限制 (160GB)
ulimit -v 41943040

# 查看当前限制
ulimit -a

# 启动工作进程
python3 extract_from_model_list.py --start-idx 94216 --end-idx 96350
```

---

## 三、新 Agent 继承工作流

### 3.1 工作交接清单

新 Agent 接手 Worker2 时，应按以下步骤进行:

#### Step 1: 环境确认

```bash
# 1. 检查工作环境
cd /root/graphnet_workspace/huggingface/worker2
pwd

# 2. 确认代理设置
echo $http_proxy
echo $https_proxy
echo $HF_ENDPOINT

# 3. 检查 GraphNet 安装
ls -la /root/GraphNet/
python3 -c "import sys; sys.path.insert(0, '/root/GraphNet'); from graph_net.torch.extractor import extract; print('OK')"
```

#### Step 2: 状态评估

```bash
# 1. 已抓取模型数
find . -name "extract.py" | wc -l

# 2. 成功提取数
find . -name "graph_net.json" | wc -l

# 3. 运行中进程
ps aux | grep "extract_from_model_list" | grep -v grep

# 4. 内存使用
free -h
```

#### Step 3: 启动监控

```bash
# 1. 检查监控脚本是否运行
ps aux | grep monitor_memory | grep -v grep

# 2. 如未运行，启动监控
if [ ! -f monitor_memory.sh ]; then
    cat > monitor_memory.sh << 'EOF'
#!/bin/bash
while true; do
    for pid in $(pgrep -f "extract_from_model_list" | head -10); do
        mem_kb=$(cat /proc/$pid/status 2>/dev/null | grep VmRSS | awk '{print $2}')
        if [ -n "$mem_kb" ] && [ "$mem_kb" -gt 67108864 ]; then
            echo "$(date): PID $pid using ${mem_kb}KB, killing..." >> monitor.log
            kill -9 $pid 2>/dev/null
        fi
    done
    sleep 60
done
EOF
    chmod +x monitor_memory.sh
fi

# 3. 启动
nohup ./monitor_memory.sh >> monitor.log 2>&1 &
```

#### Step 4: 查看日志

```bash
# 查看监控日志
tail -f monitor.log

# 查看工作进程日志
tail -f /tmp/worker_*.log
```

#### Step 5: 继续工作

```bash
# 识别缺失的批次并补充
# 查看哪些索引范围没有活跃进程
ps aux | grep extract_from_model_list | grep -v grep | awk '{print $13, $14}'

# 如有需要，启动新进程补充
python3 extract_from_model_list.py --start-idx NEW_START --end-idx NEW_END --output-dir /root/graphnet_workspace/huggingface/worker2
```

---

### 3.2 继承文档模板

```markdown
## Agent 交接记录

**时间:** 2026-XX-XX
**交接人:** Worker2 (Previous Agent)
**接收入:** Worker2 (New Agent)

### 当前状态
- 已处理模型: X,XXX
- 成功提取: XX,XXX
- 活跃进程: X个
- 内存使用: XXXGB / 1.5TB

### 正在进行的批次
| 批次 | 索引范围 | 进程ID | 状态 |
|------|----------|--------|------|
| 1 | 94216-96350 | PID | 运行中 |
| ... | ... | ... | ... |

### 已知问题
- 内存泄漏: 已配置监控脚本处理
- 失败模型: 记录在 /tmp/failed_models.log

### 下一步工作
1. 监控现有进程
2. 补充缺失批次
3. 修复失败模型
```

---

### 3.3 快速检查命令

```bash
# 一键状态检查
echo "=== Worker2 状态 ===" && \
echo "模型数: $(find . -name 'extract.py' | wc -l)" && \
echo "成功提取: $(find . -name 'graph_net.json' | wc -l)" && \
echo "活跃进程: $(ps aux | grep extract_from_model_list | grep -v grep | wc -l)" && \
free -h | grep Mem
```

---

## 四、监控与运维

### 4.1 实时监控面板

```bash
#!/bin/bash
# dashboard.sh - 实时监控面板

while true; do
    clear
    echo "=== Worker2 实时监控 ==="
    echo "时间: $(date)"
    echo ""

    # 进程状态
    echo "--- 活跃进程 ---"
    ps aux | grep extract_from_model_list | grep -v grep | awk '{print $2, $4, $6, $11, $12, $13}' | head -10

    echo ""
    echo "--- 内存状态 ---"
    free -h | grep -E "Mem|Swap"

    echo ""
    echo "--- 监控日志 ---"
    tail -3 monitor.log 2>/dev/null

    sleep 5
done
```

### 4.2 常见问题处理

| 问题 | 症状 | 解决方案 |
|------|------|----------|
| 内存泄漏 | RSS > 64GB | 监控脚本自动杀死 |
| 进程卡死 | CPU 0%, 无输出 | kill -9 后重启 |
| 模型不存在 | 404错误 | 跳过，继续下一个 |
| 权限不足 | Gated model | 记录到需授权列表 |
| 超时 | 超过180秒 | 增加超时阈值或跳过 |

### 4.3 关键文件位置

| 文件 | 路径 | 说明 |
|------|------|------|
| 批次脚本 | `/root/extract_from_model_list.py` | 主抓取脚本 |
| 修复脚本 | `/root/batch_repair_worker_ducc.py` | 子进程隔离版 |
| 监控脚本 | `./monitor_memory.sh` | 内存监控 |
| 监控日志 | `./monitor.log` | 被杀死的进程记录 |
| 工作日志 | `/tmp/worker_*.log` | 各批次工作日志 |
| 模型目录 | `/root/graphnet_workspace/huggingface/worker2/` | 输出目录 |

---

## 附录：一键启动脚本

```bash
#!/bin/bash
# setup_worker2.sh - 新 Agent 一键设置

echo "=== Worker2 初始化 ==="

# 1. 进入工作目录
cd /root/graphnet_workspace/huggingface/worker2

# 2. 设置环境
export HF_ENDPOINT=https://hf-mirror.com
export http_proxy=http://agent.baidu.com:8891
export https_proxy=http://agent.baidu.com:8891
export CUDA_VISIBLE_DEVICES=""

# 3. 检查状态
echo "模型数: $(find . -name 'extract.py' | wc -l)"
echo "成功提取: $(find . -name 'graph_net.json' | wc -l)"
echo "当前进程: $(ps aux | grep extract_from_model_list | grep -v grep | wc -l)"

# 4. 启动监控
if ! pgrep -f "monitor_memory.sh" > /dev/null; then
    if [ ! -f monitor_memory.sh ]; then
        cat > monitor_memory.sh << 'EOF'
#!/bin/bash
while true; do
    for pid in $(pgrep -f "extract_from_model_list" | head -10); do
        mem_kb=$(cat /proc/$pid/status 2>/dev/null | grep VmRSS | awk '{print $2}')
        if [ -n "$mem_kb" ] && [ "$mem_kb" -gt 67108864 ]; then
            echo "$(date): PID $pid using ${mem_kb}KB, killing..." >> monitor.log
            kill -9 $pid 2>/dev/null
        fi
    done
    sleep 60
done
EOF
        chmod +x monitor_memory.sh
    fi
    nohup ./monitor_memory.sh >> monitor.log 2>&1 &
    echo "监控脚本已启动 (PID: $!)"
else
    echo "监控脚本已在运行"
fi

echo "=== 初始化完成 ==="
```

---

*Worker2 实战经验总结 - 祝新 Agent 工作顺利！*

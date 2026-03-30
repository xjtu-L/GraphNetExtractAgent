# Worker2 残图修复标准流程

> 标准化的残图修复四步流程
> 创建时间: 2026-03-29
> 作者: Worker2

---

## 流程概览

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  1. 下载config  │ -> │ 2. 生成extract  │ -> │ 3. 运行extract  │ -> │ 4. 完整目录验证 │
│    .json        │    │    .py          │    │    脚本         │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

---

## 步骤1: 下载 config.json

### 目标
仅下载模型的 `config.json` 文件，**不下载 pytorch_model.bin 等权重文件**（祖训）

### 命令
```python
from huggingface_hub import hf_hub_download
import os

# 设置代理
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 仅下载config.json
config_path = hf_hub_download(
    repo_id=model_id,
    filename="config.json",
    local_dir=model_dir,
    local_dir_use_symlinks=False
)
```

### 验证
```bash
ls -la $model_dir/config.json
```

### 常见问题
| 问题 | 原因 | 解决 |
|------|------|------|
| 404错误 | 模型不存在/已删除 | 标记为失败，跳过 |
| 超时 | 网络问题 | 重试3次后跳过 |

---

## 步骤2: 生成 extract.py

### 目标
生成可独立运行的复现脚本，满足**祖训级别要求**

### extract.py 必备内容模板
```python
#!/usr/bin/env python3
"""
Extract script for {MODEL_ID}
Task: {TASK_TYPE}
Generated: {timestamp}
"""

import os
import sys
import torch
import argparse
from datetime import datetime

# ==================== 必须设置 ====================
os.environ["http_proxy"] = "http://agent.baidu.com:8891"
os.environ["https_proxy"] = "http://agent.baidu.com:8891"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

sys.path.insert(0, "/root/GraphNet")
from graph_net.torch.extractor import extract

from transformers import {MODEL_CLASS}, AutoConfig
# ==================== 模型配置 ====================
MODEL_ID = "{model_id}"
TASK_TYPE = "{task_type}"
MAX_LENGTH = 128

def main():
    parser = argparse.ArgumentParser(description=f"Extract {MODEL_ID}")
    parser.add_argument("--no-pretrained", action="store_true",
                       help="Use random initialization (祖训: 不下载权重)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output directory")
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH,
                       help="Max sequence length")
    args = parser.parse_args()

    # 设置workspace
    if args.output:
        os.environ["GRAPH_NET_EXTRACT_WORKSPACE"] = args.output
    else:
        os.environ["GRAPH_NET_EXTRACT_WORKSPACE"] = f"/work/graphnet_workspace/huggingface/{TASK_TYPE}"

    print(f"[Extract] {MODEL_ID}")
    print(f"  Task: {TASK_TYPE}")
    print(f"  Method: from_config() (no pretrained weights)")

    # ==================== 加载模型 ====================
    # 祖训: 使用from_config()，不下载权重
    config = AutoConfig.from_pretrained(
        MODEL_ID,
        trust_remote_code=True
    )
    model = {MODEL_CLASS}.from_config(
        config,
        trust_remote_code=True
    )
    model = model.cpu().eval()

    # ==================== 构造输入 ====================
    vocab_size = getattr(config, 'vocab_size', 32000)
    dummy_input_ids = torch.randint(0, min(vocab_size, 32000), (1, args.max_length))
    attention_mask = torch.ones((1, args.max_length))
    inputs = {"input_ids": dummy_input_ids, "attention_mask": attention_mask}

    # ==================== 抽取计算图 ====================
    safe_name = MODEL_ID.replace("/", "_")
    wrapped = extract(name=safe_name, dynamic=False)(model).eval()

    with torch.no_grad():
        wrapped(**inputs)

    # ==================== 更新元数据 ====================
    import json
    from pathlib import Path

    model_dir = Path(os.environ["GRAPH_NET_EXTRACT_WORKSPACE"]) / safe_name
    json_path = model_dir / "graph_net.json"

    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)
        data.update({
            "model_id": MODEL_ID,
            "source": "huggingface",
            "task": TASK_TYPE,
            "extraction_method": "from_config",
            "extracted_at": datetime.now().isoformat(),
        })
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)

    print(f"✓ Extraction completed for {MODEL_ID}")
    print(f"  Output: {model_dir}")

if __name__ == "__main__":
    main()
```

### 验证
```bash
# 1. 检查文件存在
ls -la $model_dir/extract.py

# 2. 检查语法正确
python3 -m py_compile $model_dir/extract.py

# 3. 设置可执行权限
chmod +x $model_dir/extract.py
```

---

## 步骤3: 运行 extract.py 生成核心文件

### 目标
运行extract.py脚本，生成以下核心文件：
- `graph_hash.txt` - 计算图哈希标识
- `model.py` - 模型结构定义
- `weight_meta.py` - 权重元数据

### 命令
```bash
cd $model_dir
python3 extract.py --no-pretrained --max-length 128
```

### 预期输出
```
[Extract] organization/model-name
  Task: text-generation
  Method: from_config() (no pretrained weights)
Graph and tensors for 'organization_model-name' extracted successfully to: /work/graphnet_workspace/huggingface/text-generation/organization_model-name
✓ Extraction completed for organization/model-name
  Output: /work/graphnet_workspace/huggingface/text-generation/organization_model-name
```

### 生成文件检查
```bash
# 检查生成的文件
ls -la $model_dir/
# 预期输出:
# - graph_hash.txt
# - model.py (或 subgraph_0/model.py)
# - weight_meta.py (或 subgraph_0/weight_meta.py)
# - graph_net.json
# - input_meta.py
# - input_tensor_constraints.py
```

### 常见问题
| 问题 | 原因 | 解决 |
|------|------|------|
| "未生成必需文件" | GraphNet输出到subgraph_0/ | 检查subgraph_0/目录 |
| 抽取超时 | 模型太大 | 增加超时时间或跳过 |
| CUDA错误 | GPU内存不足 | 使用CPU模式 |

---

## 步骤4: 完整目录验证

### 目标
验证模型目录包含所有必需文件，满足**完整目录标准**

### 验证脚本
```bash
#!/bin/bash
# verify_completeness.sh

verify_model() {
    model_dir=$1
    issues=()

    # 1. 检查 graph_hash.txt
    if [ ! -f "$model_dir/graph_hash.txt" ]; then
        issues+=("missing: graph_hash.txt")
    fi

    # 2. 检查 model.py（根目录或subgraph_0）
    if [ ! -f "$model_dir/model.py" ] && [ ! -f "$model_dir/subgraph_0/model.py" ]; then
        issues+=("missing: model.py")
    fi

    # 3. 检查 weight_meta.py（根目录或subgraph_0）
    if [ ! -f "$model_dir/weight_meta.py" ] && [ ! -f "$model_dir/subgraph_0/weight_meta.py" ]; then
        issues+=("missing: weight_meta.py")
    fi

    # 4. 检查 extract.py（根目录或subgraph_0）
    if [ ! -f "$model_dir/extract.py" ] && [ ! -f "$model_dir/subgraph_0/extract.py" ]; then
        issues+=("missing: extract.py")
    fi

    # 5. 检查 config.json
    if [ ! -f "$model_dir/config.json" ]; then
        issues+=("missing: config.json")
    fi

    if [ ${#issues[@]} -eq 0 ]; then
        echo "✅ 完整目录"
        return 0
    else
        echo "❌ 不完整: ${issues[*]}"
        return 1
    fi
}

# 使用示例
verify_model "/work/graphnet_workspace/huggingface/text-generation/model_name"
```

### 完整目录标准
```
model_dir/
├── graph_hash.txt           ✅ 必须（残图判定核心）
├── model.py                 ✅ 必须（可在subgraph_0/）
├── weight_meta.py           ✅ 必须（可在subgraph_0/）
├── extract.py               ✅ 必须（祖训：复现脚本）
├── config.json              ✅ 必须（模型配置）
├── graph_net.json           ✅ 自动生成
├── input_meta.py            ✅ 自动生成
└── input_tensor_constraints.py ✅ 自动生成
```

### 标记修复状态
```python
def mark_repair_status(model_dir, status, worker="Worker2"):
    """标记修复状态"""
    from datetime import datetime

    status_file = os.path.join(model_dir, ".repair_status")

    with open(status_file, "w") as f:
        f.write(f"worker: {worker}\n")
        f.write(f"status: {status}\n")  # completed/failed/in_progress
        f.write(f"timestamp: {datetime.now().isoformat()}\n")

    print(f"✓ Status marked as {status}")
```

---

## 批量修复流程

### 批量脚本模板
```bash
#!/bin/bash
# batch_repair.sh

MODEL_LIST="/tmp/batches_worker2/batch_XX.txt"
BATCH_ID="w2-batch-XX"

success=0
failed=0
skipped=0

while read model_id; do
    [ -z "$model_id" ] && continue

    echo "[$BATCH_ID] Processing: $model_id"

    # 步骤1: 下载config.json
    python3 -c "
from huggingface_hub import hf_hub_download
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
try:
    hf_hub_download(repo_id='$model_id', filename='config.json',
                    local_dir='model_dir', local_dir_use_symlinks=False)
    print('✓ config.json downloaded')
except Exception as e:
    print(f'✗ Failed: {e}')
"

    if [ $? -ne 0 ]; then
        failed=$((failed + 1))
        continue
    fi

    # 步骤2: 生成extract.py
    python3 generate_extract_script.py --model-id "$model_id"

    # 步骤3: 运行extract.py
    cd model_dir && python3 extract.py --no-pretrained

    # 步骤4: 验证
    if verify_model "model_dir"; then
        mark_repair_status "model_dir" "completed"
        success=$((success + 1))
    else
        mark_repair_status "model_dir" "failed"
        failed=$((failed + 1))
    fi

done < "$MODEL_LIST"

echo "========================================"
echo "Batch $BATCH_ID completed"
echo "Success: $success, Failed: $failed, Skipped: $skipped"
```

---

## 质量指标

### 关键指标
| 指标 | 目标 | 说明 |
|------|------|------|
| **graph_hash.txt覆盖率** | 100% | 残图修复完成标志 |
| **完整目录率** | 100% | 4个必备文件齐全 |
| **extract.py覆盖率** | 100% | 祖训级别要求 |

### Worker2当前进度
| 指标 | 数值 | 状态 |
|------|------|------|
| graph_hash.txt覆盖率 | 96% | ✅ 优秀 |
| 完整目录率 | 23% | 🔄 进行中 |
| extract.py覆盖率 | 100% | ✅ 已修复 |

---

## 故障排查

### 问题1: extract.py运行成功但无输出文件
**原因**: GraphNet输出到subgraph_0/子目录
**解决**: 检查subgraph_0/目录，而非仅根目录

### 问题2: 404错误下载config失败
**原因**: 模型在HF上已删除或不存在
**解决**: 标记为失败，记录到黑名单

### 问题3: 抽取过程超时
**原因**: 模型太大或网络慢
**解决**: 设置超时限制，失败后重试3次

### 问题4: .repair_status与实际状态不符
**原因**: 早期脚本验证逻辑错误
**解决**: 重新运行验证脚本，更新状态

---

## 祖训重申

1. **只下载config.json，不下载pytorch_model.bin**（from_config方式）
2. **每个模型必须有extract.py**（复现脚本）
3. **检查subgraph_0/目录**（GraphNet可能输出到子目录）
4. **及时标记.repair_status**（便于进度追踪）
5. **完整目录=4个文件**（hash + model.py + weight_meta.py + extract.py）

---

**文档版本**: v1.0
**最后更新**: 2026-03-29
**作者**: Worker2

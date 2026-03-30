# 并行执行任务最佳实践

> 本文档记录 GraphNetExtractAgent 并行执行残图修复和新模型抓取任务的经验、优化点和问题解决方案。
>
> **创建时间**: 2026-03-30
> **适用场景**: 大规模模型计算图抽取和修复

---

## 1. 执行策略概述

### 1.1 任务分类

| 任务类型 | 说明 | 优先级 |
|---------|------|--------|
| 残图修复 | 修复已有但未完整抽取的模型 | P0 |
| 新模型抓取 | 从 HuggingFace 抓取新模型 | P1 |
| 验证和去重 | 验证抽取结果，去除重复 | P2 |

### 1.2 并行架构

```
主控进程 (batch_repair_v4.py)
    ├── Subagent 0: 处理模型 0-20
    ├── Subagent 1: 处理模型 20-40
    ├── Subagent 2: 处理模型 40-60
    └── ...

新模型抓取 (extract_new_models.py)
    ├── Subagent TG-0: text-generation 模型 0-3
    ├── Subagent TG-1: text-generation 模型 3-6
    ├── Subagent TC-0: text-classification 模型 0-3
    └── ...
```

---

## 2. 关键优化点

### 2.1 v4 修复脚本改进

#### 问题：No GraphModule in output

**根本原因**：
1. 模型目录缺少 `extract.py` 脚本
2. 抽取成功后 GraphModule 在子目录中，但检查逻辑只检查根目录

**v4 解决方案**：

```python
# 1. 自动生成 extract.py
def generate_extract_py(model_dir, model_name, task):
    """为缺失脚本的模型生成标准 extract.py"""
    model_id = model_name.replace("_", "/", 1)
    extract_content = EXTRACT_PY_TEMPLATE.format(
        model_id=model_id, task=task
    )
    with open(os.path.join(model_dir, "extract.py"), 'w') as f:
        f.write(extract_content)

# 2. 完善检查逻辑（同时检查根目录和子目录）
def check_success(model_dir):
    # 检查根目录
    model_py = os.path.join(model_dir, "model.py")
    if os.path.exists(model_py):
        with open(model_py) as f:
            content = f.read()
        if "class GraphModule" in content and len(content) > 500:
            return True

    # 检查子目录
    for item in os.listdir(model_dir):
        if item.startswith("subgraph_"):
            sg_model = os.path.join(model_dir, item, "model.py")
            if os.path.exists(sg_model):
                with open(sg_model) as f:
                    content = f.read()
                if "class GraphModule" in content and len(content) > 500:
                    return True
    return False
```

#### 效果对比

| 指标 | v3 脚本 | v4 脚本 | 提升 |
|------|---------|---------|------|
| 成功率 | 0% | 24.8% | +24.8% |
| 自动生成 extract.py | ❌ | ✅ | - |
| 子目录检查 | ❌ | ✅ | - |

### 2.2 批量并行策略

#### 分批次处理

```bash
# 每批 20 个模型，避免单个进程处理过多模型
total_models=793
batch_size=20
num_batches=$((total_models / batch_size + 1))

for i in $(seq 0 $((num_batches - 1))); do
    start=$((i * batch_size))
    end=$((start + batch_size))
    python3 batch_repair_residuals_v4.py \
        --batch-id $i \
        --start-idx $start \
        --end-idx $end \
        > /tmp/repair_residual_batch_${i}.log 2>&1 &
done
```

#### 进程数量控制

| 场景 | 推荐进程数 | 说明 |
|------|-----------|------|
| 残图修复 | 20-25 | CPU 密集型，受限于 extract.py 子进程 |
| 新模型抓取 | 6-10 | 网络 + CPU 混合，避免 HF 限流 |
| 总并发 | 30-40 | 根据系统资源调整 |

### 2.3 Worker 目录组织

#### 多 Worker 并行目录结构

```
graphnet_workspace/
└── huggingface/
    ├── worker0/                    # Worker0 工作目录
    │   ├── text-generation/
    │   ├── text-classification/
    │   └── fill-mask/
    ├── worker1/                    # Worker1 工作目录
    │   ├── text-generation/
    │   ├── text-classification/
    │   └── fill-mask/
    └── shared/                     # 共享资源
```

#### 好处

1. **避免冲突**：多个 Worker 同时写入不会互相干扰
2. **责任清晰**：便于追踪问题责任人
3. **易于统计**：按 Worker 统计贡献
4. **便于合并**：后期统一合并结果

---

## 3. 遇到的问题及解决方案

### 3.1 问题：extract.py 调用顺序错误

**现象**：
```
No GraphModule in output
```

**错误代码**：
```python
# 错误：先执行 forward 再包装
_ = model(input_ids)  # 先执行
model = torch_extract(MODEL_ID)(model)  # 后包装
```

**正确代码**：
```python
# 正确：先包装再执行 forward
os.environ["GRAPH_NET_EXTRACT_WORKSPACE"] = OUTPUT_DIR
model = torch_extract(MODEL_ID)(model)  # 先包装
with torch.no_grad():
    _ = model(input_ids)  # 再执行触发提取
```

### 3.2 问题：模型配置错误

**现象**：
```
If you want to use `XmodLMHeadModel` as a standalone, add `is_decoder=True.`
`rope_parameters`'s factor field must be a float >= 1, got 40
```

**解决方案**：

```python
# 针对特殊模型的定制处理
config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

# Xmod/Ernie 模型需要 is_decoder
if "xmod" in model_id.lower() or "ernie" in model_id.lower():
    config.is_decoder = True

# RoPE 参数修复
if hasattr(config, 'rope_parameters'):
    config.rope_parameters = None

model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
```

### 3.3 问题：网络超时

**现象**：
```
OSError: We couldn't connect to 'https://hf-mirror.com' to load the files
```

**解决方案**：

```bash
# 设置代理
export http_proxy=http://agent.baidu.com:8891
export https_proxy=http://agent.baidu.com:8891
export HF_ENDPOINT=https://hf-mirror.com

# 使用本地 config.json（如果已下载）
config = AutoConfig.from_pretrained(".", trust_remote_code=True)
```

### 3.4 问题：残图未被识别

**现象**：模型抽取成功，但生成的是残图（< 50 行的简化版 model.py）

**原因**：extract.py 脚本执行异常，只捕获到部分计算图

**解决方案**：

```python
# 在 extract.py 中添加清理逻辑
def main():
    # 清理旧文件（包括子目录）
    for f in ["graph_net.json", "graph_hash.txt", "model.py", "weight_meta.py"]:
        if os.path.exists(f):
            os.remove(f)

    # 清理旧子图目录
    for item in os.listdir("."):
        if item.startswith("subgraph_") and os.path.isdir(item):
            shutil.rmtree(item)

    # 继续正常抽取流程...
```

### 3.5 问题：子进程残留

**现象**：大量 extract.py 子进程占用系统资源

**解决方案**：

```bash
# 定期清理僵尸进程
pkill -f "extract.py"  # 停止所有 extract.py

# 或只清理超过 10 分钟的进程
ps aux | grep extract.py | awk '{if($10 > 10:00) print $2}' | xargs kill -9
```

---

## 4. 监控和调试

### 4.1 实时进度监控

```bash
# 查看各批次进度
for i in {0..39}; do
    log="/tmp/repair_residual_batch_${i}.log"
    if [ -f "$log" ]; then
        total=$(grep -c "^\[" "$log" 2>/dev/null || echo "0")
        success=$(grep -c "✓ SUCCESS" "$log" 2>/dev/null || echo "0")
        echo "Batch $i: 已处理=$total, 成功=$success"
    fi
done

# 统计总成功数
grep -r "✓ SUCCESS" /tmp/repair_residual_batch_*.log | wc -l
```

### 4.2 失败原因分析

```bash
# 统计失败类型
grep -r "✗ FAILED" /tmp/repair_residual_batch_*.log | \
    sed 's/.*FAILED: //' | \
    sort | uniq -c | sort -rn | head -10
```

### 4.3 日志文件管理

| 文件 | 用途 |
|------|------|
| `/tmp/repair_residual_batch_{N}.log` | 批次 N 的修复日志 |
| `/tmp/failed_models_batch_{N}.txt` | 批次 N 的失败模型列表 |
| `/tmp/extract_worker2_{task}_{N}.log` | Worker2 新模型抓取日志 |

---

## 5. 性能优化建议

### 5.1 CPU 优化

```bash
# 限制单进程 CPU 使用
cpulimit -p $PID -l 50  # 限制为 50% CPU

# 使用 nice 调整优先级
nice -n 10 python3 extract_script.py
```

### 5.2 内存优化

```python
# 在 extract.py 中使用 CPU 而非 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# 减小输入尺寸
seq_len = min(32, max_position_embeddings)  # 而非 512
```

### 5.3 网络优化

```python
# 使用本地缓存
from huggingface_hub import snapshot_download

# 先下载到本地
local_path = snapshot_download(repo_id=model_id, local_dir="./cache")

# 从本地加载
config = AutoConfig.from_pretrained("./cache", trust_remote_code=True)
```

---

## 6. 检查清单

### 6.1 启动前检查

- [ ] Worker 目录已创建
- [ ] 代理环境变量已设置
- [ ] 旧日志已清理
- [ ] 系统资源充足（CPU/内存/磁盘）

### 6.2 执行中检查

- [ ] 进程正常运行（无僵尸进程）
- [ ] 日志文件正常写入
- [ ] 系统负载合理（CPU < 80%）

### 6.3 完成后检查

- [ ] 统计成功/失败数量
- [ ] 验证生成的 model.py 完整性
- [ ] 清理临时文件
- [ ] 备份结果

---

## 附录 A: 动态任务调度系统

### A.1 概述

动态任务调度系统根据系统 CPU 负载自动调整并行任务数量，保持系统稳定运行的同时最大化处理效率。

### A.2 工作原理

```
CPU 负载监测 (每30秒)
    ↓
负载 > 75% → 减少3个任务
负载 < 30% → 增加2个任务
30% ≤ 负载 ≤ 75% → 保持不变
    ↓
调整当前任务数 (min ≤ 任务数 ≤ max)
    ↓
启动新任务 或 停止旧任务
```

### A.3 使用方法

```bash
# 安装依赖
pip install psutil

# 启动动态任务管理器
python3 /root/dynamic_task_manager.py \
    --script /root/batch_repair_residuals_v4.py \
    --total 793 \
    --min-workers 5 \
    --max-workers 40 \
    --cpu-high 75.0 \
    --cpu-low 30.0 \
    --interval 30
```

### A.4 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--script` | 任务脚本路径 | 必填 |
| `--total` | 总任务数 | 必填 |
| `--min-workers` | 最小并发任务数 | 5 |
| `--max-workers` | 最大并发任务数 | 40 |
| `--cpu-high` | CPU高负载阈值(%) | 75.0 |
| `--cpu-low` | CPU低负载阈值(%) | 30.0 |
| `--interval` | 检查间隔(秒) | 30 |

### A.5 调度策略

```python
class DynamicTaskManager:
    def adjust_worker_count(self, cpu_usage):
        """根据CPU使用率调整工作进程数"""
        if cpu_usage > self.cpu_high_threshold:
            # CPU负载高，减少任务数
            self.current_workers = max(
                self.min_workers,
                self.current_workers - 3
            )
        elif cpu_usage < self.cpu_low_threshold:
            # CPU负载低，增加任务数
            self.current_workers = min(
                self.max_workers,
                self.current_workers + 2
            )
```

### A.6 输出示例

```
============================================================
动态任务管理器启动
任务脚本: /root/batch_repair_residuals_v4.py
总任务数: 793
CPU阈值: 高=75.0% 低=30.0%
任务数范围: 5-40
============================================================

[13:00:00] 启动批次 0: 模型 0-20 (PID: 12345)
[13:00:01] 启动批次 1: 模型 20-40 (PID: 12346)
...
[13:01:30] CPU: 82.5% | 任务数 减少: 25 → 22
[13:01:31] 停止批次 2 (PID: 12347)
[13:02:00] CPU: 45.2% | 任务数 保持: 22
[13:02:30] CPU: 25.1% | 任务数 增加: 22 → 24
[13:02:31] 启动批次 40: 模型 800-820 (PID: 12450)

[13:05:00] 状态报告:
  CPU: 55.3% | 内存: 68.5%
  当前任务数: 24/30
  已完成: 156 | 失败: 48
  总进度: 25.7%
------------------------------------------------------------
```

### A.7 与固定并发数对比

| 对比项 | 固定并发数 | 动态调度 |
|--------|-----------|---------|
| CPU利用率 | 波动大 | 稳定在目标范围 |
| 任务完成速度 | 固定 | 自适应调整 |
| 系统稳定性 | 可能过载 | 保持稳定 |
| 资源浪费 | 可能空闲 | 充分利用 |
| 适用场景 | 资源充足 | 资源共享环境 |

---

## 7. 相关文档

- [Worker 目录组织规范](WORKER_DIRECTORY_GUIDE.md)
- [多 Worker 协作流程](multi_worker_collaboration.md)
- [失败案例处理指南](../repair/FAILED_CASES_GUIDE.md)
- [HuggingFace 抽取指南](HUGGINGFACE_EXTRACTION_GUIDE.md)

---

**最后更新**: 2026-03-30
**维护者**: Ducc
**经验来源**: 793 个残图修复 + 新模型抓取并行执行实践

# 抽取问题发现机制与处理经验

## 概述

本文档记录 2026-04-02 在 HuggingFace 模型抽取过程中遇到的各种问题、发现机制和处理经验。

## 问题分类与处理

### 1. 嵌套目录问题

#### 现象
```
worker2/model-name/
└── model-name/              # 同名子目录（嵌套层）
    ├── subgraph_0/
    └── ...
```

#### 发现机制
- 监控脚本检查目录结构时发现重复层级
- `find` 命令检测同名子目录

#### 根本原因
- 抽取在 `torch_extract()` 阶段中断
- `fix_nested_directory()` 只在抽取成功后运行
- 外层目录和内层目录同时存在

#### 处理方案
1. **脚本修复**：在抽取前添加嵌套目录检查和修复
2. **批量修复脚本**：`/tmp/fix_nested_dirs.py`
3. **修复后自动生成 extract.py**（重要！）

#### 修复代码
```python
def fix_nested_directory(model_path, model_name):
    nested_path = os.path.join(model_path, model_name)
    if os.path.isdir(nested_path):
        # 1. 移动内层文件到外层
        for item in os.listdir(nested_path):
            src = os.path.join(nested_path, item)
            dst = os.path.join(model_path, item)
            if not os.path.exists(dst):
                shutil.move(src, dst)
        # 2. 删除空内层目录
        os.rmdir(nested_path)
        # 3. 【重要】生成 extract.py
        if not os.path.exists(os.path.join(model_path, 'extract.py')):
            create_extract_py(model_path, model_name)
        return True
    return False
```

---

### 2. extract.py 缺失问题

#### 现象
- 成功抽取的模型目录中没有 `extract.py`
- 无法识别模型任务类型
- 迁移脚本无法正确处理

#### 发现机制
- 批量检查：`find worker2 -type d -not -path "*fill-mask*" | while read dir; do [ -f "$dir/extract.py" ] || echo "缺失: $dir"; done`

#### 根本原因
- 早期抽取脚本未生成 extract.py
- 嵌套目录修复后忘记生成
- 手动复制模型时遗漏

#### 处理方案
1. **批量生成脚本**：`/tmp/generate_extract_py.py`
2. **自动推断任务类型**：
   - T5/BART/Pegasus → text2text-generation
   - GPT/Llama/Qwen → text-generation
   - BERT/RoBERTa → fill-mask
3. **统一模板**：
```python
#!/usr/bin/env python3
"""Extract script for model_id
Task: task_type
"""
import torch
from transformers import AutoModel, AutoConfig
import sys, os
sys.path.insert(0, '/root/GraphNet')

def extract_model(model_path, device='cpu'):
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_config(config, trust_remote_code=True)
    model.eval()
    try:
        from graph_net.torch.extractor import extract
        return extract(model)
    except Exception as e:
        print(f"Extraction failed: {e}")
        return None

if __name__ == "__main__":
    # ...
```

---

### 3. 空目录问题

#### 现象
- 模型目录存在但无任何内容（只有 . 和 ..）
- Worker2 中发现 258+ 个空目录
- 示例：`abacusai_Giraffe-13b-32k-v3`

#### 发现机制
```bash
# 查找空目录
find /root/graphnet_workspace/huggingface/worker2 -maxdepth 1 -type d -empty

# 统计空目录数量
find /root/graphnet_workspace/huggingface/worker2 -maxdepth 1 -type d -empty | wc -l
```

#### 根本原因
1. **抽取早期失败**：
   - 目录创建：`os.makedirs(model_path, exist_ok=True)`（第490行）
   - 抽取失败：`torch_extract` 抛出异常（第509行）
   - 异常捕获后**只打印错误，没有清理目录**

2. **未记录淘汰**：
   - 异常处理中没有调用 `record_eliminated_model()`
   - 导致无法追踪失败原因

#### 修复方案
在 `extract_model()` 的异常处理中添加：
1. **记录淘汰原因**
2. **清理空目录**

```python
except Exception as e:
    error_msg = str(e)
    # 1. 记录淘汰
    if "404" in error_msg:
        record_eliminated_model(model_id, "模型不存在(404)")
    elif "gated" in error_msg.lower():
        record_eliminated_model(model_id, "需要授权访问")
    else:
        record_eliminated_model(model_id, f"抽取失败: {error_msg[:50]}")

    # 2. 清理空目录
    try:
        if 'model_path' in locals() and os.path.exists(model_path):
            has_content = False
            for item in os.listdir(model_path):
                item_path = os.path.join(model_path, item)
                if os.path.isdir(item_path) and os.listdir(item_path):
                    has_content = True
                    break
                elif os.path.isfile(item_path) and os.path.getsize(item_path) > 0:
                    has_content = True
                    break

            if not has_content:
                shutil.rmtree(model_path)
                print(f"  已清理空目录: {model_path}")
    except Exception as cleanup_error:
        print(f"  警告: 清理目录失败: {cleanup_error}")

    return False
```

---

### 4. 抽取失败未记录问题

#### 现象
- 模型抽取失败
- 但 `/root/GraphNetExtractAgent/data/eliminated_models.txt` 中没有记录
- 无法追踪失败模式和原因

#### 发现机制
- 对比空目录数量和淘汰记录数量
- 发现不匹配：空目录多，淘汰记录少

#### 根本原因
- 异常处理分支中没有统一调用 `record_eliminated_model()`
- 只有特定错误（如不支持架构）才记录

#### 修复方案
**所有失败路径都必须记录**：
1. 不支持架构 → 记录
2. 404错误 → 记录
3. 需要授权 → 记录
4. 其他异常 → 记录
5. 抽取失败 → 记录

---

### 5. 进程数不稳定问题

#### 现象
- 抽取进程数随时间下降
- 从 40 个逐渐降到 10 个或更低
- 新批次启动不及时

#### 发现机制
- `ps aux | grep extract_from_model_list | wc -l`
- 监控脚本 `/root/monitor_extraction.py`

#### 处理方案
**创建监控 Subagent**：
```python
# /root/monitor_extraction.py
TARGET_PARALLEL = 40      # 目标并行数
MIN_PARALLEL = 30         # 最低阈值
CHECK_INTERVAL = 60       # 检查间隔（秒）

def main():
    while True:
        current_count, _ = get_running_extract_processes()

        # 进程数过低，立即补充
        if current_count < MIN_PARALLEL:
            needed = TARGET_PARALLEL - current_count
            for _ in range(needed):
                start_new_batch(next_start_idx, next_end_idx)

        # 进程数不足， gradual refill
        elif current_count < TARGET_PARALLEL:
            needed = min(2, TARGET_PARALLEL - current_count)
            for _ in range(needed):
                start_new_batch(next_start_idx, next_end_idx)

        time.sleep(CHECK_INTERVAL)
```

---

### 6. fill-mask 空目录重复抽取

#### 现象
- fill-mask 目录下大量空目录（3371个）
- 这些是之前抽取失败的残留

#### 处理方案
1. **记录空目录模型列表**
2. **启动专门 Subagent** 重新抽取：
   - 5个并行 worker
   - 每批次50个模型
   - 批次间休息10秒

```bash
# 启动命令
for i in {0..4}; do
    nohup python3 /tmp/reextract_fill_mask.py $i 50 > /tmp/reextract_fill_mask_logs/worker${i}_main.log 2>&1 &
done
```

---

## 最佳实践总结

### 1. 预防措施

| 措施 | 实现 |
|------|------|
| 抽取前检查嵌套目录 | `fix_nested_directory()` 在 `os.makedirs()` 前调用 |
| 异常后清理目录 | 所有异常路径都清理空目录 |
| 所有失败记录 | 统一调用 `record_eliminated_model()` |
| 自动生成 extract.py | 修复嵌套目录后自动生成 |
| 进程数监控 | 监控 Subagent 每60秒检查 |

### 2. 检查命令

```bash
# 检查空目录
find /root/graphnet_workspace/huggingface/worker2 -maxdepth 1 -type d -empty | wc -l

# 检查嵌套目录
for dir in /root/graphnet_workspace/huggingface/worker2/*/; do
    model=$(basename "$dir")
    [ -d "$dir/$model" ] && echo "$model"
done

# 检查缺少 extract.py 的模型
for dir in /root/graphnet_workspace/huggingface/worker2/*/; do
    [ -f "$dir/extract.py" ] || echo "缺失: $(basename "$dir")"
done

# 检查进程数
ps aux | grep "extract_from_model_list.py" | grep -v grep | wc -l
```

### 3. 修复脚本清单

| 脚本 | 用途 |
|------|------|
| `/tmp/fix_nested_dirs.py` | 修复嵌套目录 + 生成 extract.py |
| `/tmp/generate_extract_py.py` | 批量生成缺失的 extract.py |
| `/tmp/reextract_fill_mask.py` | 重新抽取 fill-mask 空目录 |
| `/root/monitor_extraction.py` | 监控并补充抽取进程 |

---

## 今日成果（2026-04-02）

| 指标 | 数值 |
|------|------|
| 修复嵌套目录 | 9个 |
| 生成 extract.py | 33+个 |
| 清理 fill-mask 空目录 | 3371个 |
| 扩大抽取进程 | 10→40个 |
| 今日新抽模型 | 376个 |
| 今日计算图数 | 2560个 |
| 平均计算图/模型 | 6.8个 |

---

## 相关文档

- `/root/GraphNetExtractAgent/docs/guides/extract/NESTED_DIRECTORY_ISSUE.md` - 嵌套目录专项记录
- `/root/extract_from_model_list.py` - 主抽取脚本（已修复）

# Worker2 修复标准分享

> 分享给 Worker1 的修复质量标准与最佳实践
> 创建时间: 2026-03-29
> 分享者: Worker2

---

## 一、什么是完整目录率？

### 定义
**完整目录率** = 满足"完整目录标准"的模型数 / 总模型数 × 100%

### 完整目录标准（祖训级别）
一个模型目录必须同时包含以下4个核心文件，才被视为**完整**：

| 文件 | 作用 | 检查路径 |
|------|------|----------|
| `graph_hash.txt` | 计算图唯一哈希标识 | 根目录或 subgraph_0/ |
| `model.py` | 模型结构定义代码 | 根目录或 subgraph_0/ |
| `weight_meta.py` | 权重张量元数据 | 根目录或 subgraph_0/ |
| `extract.py` | 复现脚本（祖训新增） | 根目录或 subgraph_0/ |

### 检查命令
```bash
# 检查单个模型完整性
check_model_complete() {
    model_dir=$1
    has_hash=false
    has_model=false
    has_weight=false
    has_extract=false

    [ -f "$model_dir/graph_hash.txt" ] && has_hash=true
    [ -f "$model_dir/model.py" ] || [ -f "$model_dir/subgraph_0/model.py" ] && has_model=true
    [ -f "$model_dir/weight_meta.py" ] || [ -f "$model_dir/subgraph_0/weight_meta.py" ] && has_weight=true
    [ -f "$model_dir/extract.py" ] || [ -f "$model_dir/subgraph_0/extract.py" ] && has_extract=true

    if [ "$has_hash" = true ] && [ "$has_model" = true ] && [ "$has_weight" = true ] && [ "$has_extract" = true ]; then
        echo "✅ 完整"
    else
        echo "❌ 不完整: hash=$has_hash, model=$has_model, weight=$has_weight, extract=$has_extract"
    fi
}
```

---

## 二、如何确保模型目录完整？

### 阶段1：抽取过程检查

#### 1.1 抽取前验证
```python
# 检查目录是否已完整，避免重复工作
if os.path.exists(os.path.join(model_dir, "graph_hash.txt")):
    if os.path.exists(os.path.join(model_dir, "model.py")) or \
       os.path.exists(os.path.join(model_dir, "subgraph_0", "model.py")):
        print("✓ 已完整，跳过")
        return True, "已存在"
```

#### 1.2 抽取后验证（关键！）
```python
def verify_extraction(model_dir):
    """验证抽取是否真正成功"""
    required_files = [
        ("graph_hash.txt", ["graph_hash.txt"]),
        ("model.py", ["model.py", "subgraph_0/model.py"]),
        ("weight_meta.py", ["weight_meta.py", "subgraph_0/weight_meta.py"]),
    ]

    missing = []
    for name, paths in required_files:
        found = any(os.path.exists(os.path.join(model_dir, p)) for p in paths)
        if not found:
            missing.append(name)

    return len(missing) == 0, missing
```

### 阶段2：extract.py 生成（祖训要求）

#### 2.1 生成时机
**必须在抽取成功后立即生成**，不能延后！

#### 2.2 extract.py 必备内容
```python
#!/usr/bin/env python3
"""
Extract script for {MODEL_ID}
Task: {TASK_TYPE}
"""

import os
import sys
import torch
import argparse

# 必须设置代理
os.environ["http_proxy"] = "http://agent.baidu.com:8891"
os.environ["https_proxy"] = "http://agent.baidu.com:8891"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

sys.path.insert(0, "/root/GraphNet")
from graph_net.torch.extractor import extract
from transformers import {MODEL_CLASS}, AutoConfig

MODEL_ID = "xxx"
TASK_TYPE = "xxx"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-pretrained", action="store_true",
                       help="Use random initialization")
    args = parser.parse_args()

    # 从config加载（祖训：不下载权重）
    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = {MODEL_CLASS}.from_config(config, trust_remote_code=True)

    # 执行抽取...
```

#### 2.3 生成后验证
```bash
# 检查extract.py是否可执行
chmod +x $model_dir/extract.py
python3 -m py_compile $model_dir/extract.py  # 语法检查
```

### 阶段3：状态标记

#### 3.1 创建.repair_status文件
```python
def mark_status(model_dir, status, worker="Worker1", error=""):
    """标记修复状态"""
    status_file = os.path.join(model_dir, ".repair_status")

    with open(status_file, "w") as f:
        f.write(f"worker: {worker}\n")
        f.write(f"status: {status}\n")  # in_progress/completed/failed
        f.write(f"timestamp: {datetime.now().isoformat()}\n")
        if error:
            f.write(f"error: {error}\n")
```

#### 3.2 状态检查清单
- [ ] 抽取成功
- [ ] graph_hash.txt 已生成
- [ ] model.py 已生成（根目录或subgraph_0）
- [ ] weight_meta.py 已生成（根目录或subgraph_0）
- [ ] extract.py 已生成并可执行
- [ ] .repair_status 标记为 completed

---

## 三、常见陷阱与解决方案

### 陷阱1：GraphNet输出到subgraph_0但检查只在根目录
**现象**: 显示"未生成model.py"，但实际在subgraph_0/

**解决**:
```python
# 错误检查
if not os.path.exists(os.path.join(model_dir, "model.py")):  # ❌ 漏检subgraph_0

# 正确检查
if not os.path.exists(os.path.join(model_dir, "model.py")) and \
   not os.path.exists(os.path.join(model_dir, "subgraph_0", "model.py")):  # ✅
```

### 陷阱2：抽取成功但状态文件标记为failed
**原因**: 早期脚本bug，验证逻辑错误

**解决**: 重新运行验证脚本，更新状态文件

### 陷阱3：extract.py生成失败或被遗漏
**后果**: 不满足祖训要求，视为不完整

**解决**: 使用批量生成脚本补充

---

## 四、批量检查脚本

```bash
#!/bin/bash
# check_completeness.sh - 批量检查目录完整性

WORKER_LIST="/tmp/empty_dirs_worker1.txt"  # Worker1修改为自己的列表
TOTAL=0
COMPLETE=0
INCOMPLETE=0

echo "=== 目录完整性检查 ==="
echo "时间: $(date)"
echo ""

while read model_id; do
    [ -z "$model_id" ] && continue
    TOTAL=$((TOTAL + 1))

    safe_name=$(echo "$model_id" | tr '/' '_')

    for task_dir in /work/graphnet_workspace/huggingface/*/; do
        model_dir="$task_dir/$safe_name"
        if [ -d "$model_dir" ]; then
            # 检查4个必需文件
            has_hash=false
            has_model=false
            has_weight=false
            has_extract=false

            [ -f "$model_dir/graph_hash.txt" ] && has_hash=true
            [ -f "$model_dir/model.py" ] || [ -f "$model_dir/subgraph_0/model.py" ] && has_model=true
            [ -f "$model_dir/weight_meta.py" ] || [ -f "$model_dir/subgraph_0/weight_meta.py" ] && has_weight=true
            [ -f "$model_dir/extract.py" ] || [ -f "$model_dir/subgraph_0/extract.py" ] && has_extract=true

            if [ "$has_hash" = true ] && [ "$has_model" = true ] && [ "$has_weight" = true ] && [ "$has_extract" = true ]; then
                COMPLETE=$((COMPLETE + 1))
            else
                INCOMPLETE=$((INCOMPLETE + 1))
                echo "❌ $model_id: hash=$has_hash, model=$has_model, weight=$has_weight, extract=$has_extract"
            fi
            break
        fi
    done
done < "$WORKER_LIST"

echo ""
echo "=== 统计 ==="
echo "总计: $TOTAL"
echo "完整: $COMPLETE ($((COMPLETE * 100 / TOTAL))%)"
echo "不完整: $INCOMPLETE"
```

---

## 五、Worker2 当前成果参考

| 指标 | Worker2数值 | 说明 |
|------|-------------|------|
| graph_hash.txt覆盖率 | 96% | 残图修复基本完成 |
| 完整目录率 | 23% → 目标100% | 进行中 |
| extract.py覆盖率 | 100%（修复后） | 已补充17个缺失 |

---

## 六、关键原则总结

1. **graph_hash.txt是判定残图的核心标准**
2. **完整目录需要4个文件**：hash + model.py + weight_meta.py + extract.py
3. **extract.py是祖训级别要求**，每个模型必须有
4. **检查subgraph_0目录**，不要只检查根目录
5. **及时标记.repair_status**，便于进度追踪

---

**分享者**: Worker2
**时间**: 2026-03-29
**状态**: 持续更新中

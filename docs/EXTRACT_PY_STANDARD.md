# Extract.py 标准格式规范

## 概述

本文档定义了 GraphNet HuggingFace 模型抽取任务中 `extract.py` 的标准格式。所有抓取成功的模型必须在主目录下包含符合此标准的 `extract.py` 文件。

## 标准格式要求

### 文件位置
- **主目录**: `extract.py` 必须位于模型主目录下
- **Subgraph 目录**: 不需要单独的 `extract.py`，由主目录统一管理

### 标准模板

```python
#!/usr/bin/env python3
"""Extract script for {model_id}
Task: {task_type}
"""
import torch
from transformers import {model_class}, AutoConfig
import sys, os
sys.path.insert(0, '/root/GraphNet')

def extract_model(model_path, device='cpu'):
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = {model_class}.from_config(config, trust_remote_code=True)
    model.eval()
    try:
        from graph_net.torch.extractor import extract
    except ImportError:
        print("Error: graph_net not found")
        return None
    vocab_size = getattr(config, 'vocab_size', 32000)
    max_pos = getattr(config, 'max_position_embeddings', 512)
    seq_len = min(32, max_pos)
    input_ids = torch.randint(0, min(vocab_size, 1000), (1, seq_len))
    try:
        return extract(model, input_ids)
    except Exception as e:
        print(f"Extraction failed: {e}")
        return None

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=".")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = extract_model(args.model_path, args.device)
    print("Success" if result else "Failed")
```

## 变量替换说明

| 变量 | 说明 | 可选值 |
|------|------|--------|
| `{model_id}` | 原始模型ID (如 "bert-base-uncased") | - |
| `{task_type}` | 任务类型 | feature-extraction, text-generation, text-classification, fill-mask, token-classification, question-answering |
| `{model_class}` | Transformers模型类 | AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoModelForMaskedLM, AutoModelForTokenClassification, AutoModelForQuestionAnswering |

## 模型类选择指南

根据 `model.py` 内容自动选择对应的模型类：

```python
if "ForSequenceClassification" in content:
    model_class = "AutoModelForSequenceClassification"
    task_type = "text-classification"
elif "ForTokenClassification" in content:
    model_class = "AutoModelForTokenClassification"
    task_type = "token-classification"
elif "ForQuestionAnswering" in content:
    model_class = "AutoModelForQuestionAnswering"
    task_type = "question-answering"
elif "ForCausalLM" in content or "LMHead" in content:
    model_class = "AutoModelForCausalLM"
    task_type = "text-generation"
elif "ForMaskedLM" in content:
    model_class = "AutoModelForMaskedLM"
    task_type = "fill-mask"
else:
    model_class = "AutoModel"
    task_type = "feature-extraction"
```

## 验证标准

一个合格的 `extract.py` 必须满足：

1. ✅ 包含 `from graph_net.torch.extractor import extract`
2. ✅ 包含 `def extract_model` 函数定义
3. ✅ 包含 `if __name__ == "__main__"` 入口
4. ✅ 使用正确的 `{model_class}` (非固定 AutoModel)
5. ✅ 文件可执行权限 (`chmod +x extract.py`)

## 生成函数参考

```python
def generate_extract_py(model_path, model_id):
    """生成标准格式的 extract.py"""
    extract_py_path = os.path.join(model_path, "extract.py")

    if os.path.exists(extract_py_path):
        return  # 已存在则跳过

    # 读取 model.py 分析模型类型
    model_class, task_type = analyze_model_type(model_path)

    extract_content = f'''#!/usr/bin/env python3
"""Extract script for {model_id}
Task: {task_type}
"""
import torch
from transformers import {model_class}, AutoConfig
import sys, os
sys.path.insert(0, '/root/GraphNet')

def extract_model(model_path, device='cpu'):
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = {model_class}.from_config(config, trust_remote_code=True)
    model.eval()
    try:
        from graph_net.torch.extractor import extract
    except ImportError:
        print("Error: graph_net not found")
        return None
    vocab_size = getattr(config, 'vocab_size', 32000)
    max_pos = getattr(config, 'max_position_embeddings', 512)
    seq_len = min(32, max_pos)
    input_ids = torch.randint(0, min(vocab_size, 1000), (1, seq_len))
    try:
        return extract(model, input_ids)
    except Exception as e:
        print(f"Extraction failed: {{e}}")
        return None

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=".")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = extract_model(args.model_path, args.device)
    print("Success" if result else "Failed")
'''
    with open(extract_py_path, 'w', encoding='utf-8') as f:
        f.write(extract_content)
    os.chmod(extract_py_path, 0o755)
```

## 目录结构与统计规则

### 核心原则
> **一个模型只需要一个 extract.py（在主目录），多个 subgraph 由主 extract.py 统一抽取，subgraph 目录不需要单独 extract.py。**

### 正确目录结构
```
model_directory/                    # 模型主目录
├── extract.py                      # 唯一的提取脚本（主目录）
├── graph_net.json                  # 主图结构
├── subgraph_0/                     # 子图目录
│   └── graph_net.json              # 子图结构（由主 extract.py 生成）
└── subgraph_1/
    └── graph_net.json
```

### 统计规则
| 统计项 | 计算方法 | 说明 |
|--------|----------|------|
| **总模型数** | 一级子目录数量 | 每个模型一个目录 |
| **成功抽取数** | 一级子目录中有 `extract.py` 的数量 | 主目录有 extract.py 即算成功 |
| **空目录数** | 一级子目录中没有 `extract.py` 的数量 | 需要重新抽取的模型 |

### 统计命令
```bash
# 统计当前 Worker 的模型情况
cd /ssd1/liangtai-work/graphnet_workspace/huggingface/worker1
total=$(ls -d */ 2>/dev/null | wc -l)
extracted=0
for dir in */; do
    if [ -f "${dir}extract.py" ]; then
        extracted=$((extracted + 1))
    fi
done
empty=$((total - extracted))
echo "总目录: $total, 已抽取: $extracted, 空目录: $empty"

# 检查 extract.py 格式是否正确
grep -l "from graph_net.torch.extractor import extract" \
  /ssd1/liangtai-work/graphnet_workspace/huggingface/worker1/*/extract.py | wc -l
```

### 注意事项
- **子图目录不应包含 extract.py** - subgraph_*/ 目录只应有 graph_net.json
- **避免重复统计** - 使用 `maxdepth 1` 或 `ls -d */` 统计一级目录
- **所有级别 extract.py 总数 > 一级目录中有 extract.py 的数量** - 因为子图目录也可能有（错误的）extract.py

## 生成 graph_hash.txt

### 概述
每个包含 `model.py` 的计算图目录都应该有一个对应的 `graph_hash.txt` 文件，用于唯一标识该计算图。

### 生成方法

#### 1. 使用 generate_graph_hash.py 工具

```bash
cd /root/GraphNet

# 为单个 Worker 生成（仅生成缺失的）
python tools/generate_graph_hash.py \
  --samples-dir /ssd1/liangtai-work/graphnet_workspace/huggingface/worker1

# 为所有 Worker 生成
python tools/generate_graph_hash.py \
  --samples-dir /ssd1/liangtai-work/graphnet_workspace/huggingface/worker1 \
  --samples-dir /ssd1/liangtai-work/graphnet_workspace/huggingface/worker2

# 强制重新生成（覆盖现有文件）
python tools/generate_graph_hash.py \
  --samples-dir /ssd1/liangtai-work/graphnet_workspace/huggingface \
  --overwrite
```

#### 2. 为整个 workspace 生成

```bash
cd /root/GraphNet
python tools/generate_graph_hash.py \
  --samples-dir /ssd1/liangtai-work/graphnet_workspace \
  --overwrite
```

#### 3. 汇总所有哈希值

```bash
cd /ssd1/liangtai-work/graphnet_workspace
OUTPUT_FILE="all_graph_hashes.txt"

echo "# GraphNet 所有计算图哈希汇总" > "$OUTPUT_FILE"
echo "# 生成时间: $(date)" >> "$OUTPUT_FILE"
echo "# 格式: 目录路径 | 哈希值" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

find . -name "graph_hash.txt" -type f | while read f; do
    dir=$(dirname "$f")
    hash=$(cat "$f")
    echo "${dir#./} | $hash" >> "$OUTPUT_FILE"
done

echo "汇总完成: $OUTPUT_FILE"
echo "总条目数: $(wc -l < "$OUTPUT_FILE")"
```

### 验证生成结果

```bash
cd /ssd1/liangtai-work/graphnet_workspace/huggingface

# 统计 model.py 和 graph_hash.txt 数量
MODEL_PY_COUNT=$(find . -name "model.py" -type f | wc -l)
HASH_COUNT=$(find . -name "graph_hash.txt" -type f | wc -l)

echo "model.py 文件总数: $MODEL_PY_COUNT"
echo "graph_hash.txt 文件总数: $HASH_COUNT"

if [ "$MODEL_PY_COUNT" -eq "$HASH_COUNT" ]; then
    echo "✅ 完美匹配！每个 model.py 都有对应的 graph_hash.txt"
else
    echo "⚠️ 数量不匹配，差值: $((HASH_COUNT - MODEL_PY_COUNT))"
fi
```

### 清理多余的 graph_hash.txt

```bash
cd /ssd1/liangtai-work/graphnet_workspace/huggingface

find . -name "graph_hash.txt" -type f | while read f; do
    dir=$(dirname "$f")
    if [ ! -f "${dir}/model.py" ]; then
        rm -f "$f"
        echo "删除多余文件: $f"
    fi
done
```

## 质量要求

> **核心目标**：抽取的计算图必须能够正确执行 `run_model`（加载模型并执行 forward）。

### SymInt Example Value 推断要求

抽取时必须从 tensor shapes 推断 SymInt 参数的真实值，不能全部设为 placeholder 值 `4`。

| SymInt 参数 | 推断来源 |
|------------|---------|
| `s0` (seq_len) | 从输入 tensor（非权重）的第二维推断 |
| `L_self_head_dim` | 从 `position_embeddings` tensor 的最后一维推断 |

### API 兼容性要求

避免使用低级 API 如 `torch.ops.aten._assert_scalar`，或确保参数类型符合 API 要求。

### 验证方法

抽取完成后建议运行：
```bash
python3.10 -m graph_net.torch.run_model --model-path <subgraph_path>
```

---

## 历史记录

- **2024-03-31**: 定义标准格式，修复286个模型的extract.py
- **2024-04-01**: 添加目录结构与统计规则
- **2024-04-01**: 添加 graph_hash.txt 生成方法
- **2026-04-13**: 添加质量要求（SymInt 推断、API 兼容性）
- **标准版本**: v1.3

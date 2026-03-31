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

## 检查命令

```bash
# 统计有 extract.py 的模型数
find /root/graphnet_workspace/huggingface/worker1 -maxdepth 2 -name "extract.py" | wc -l

# 检查 extract.py 格式是否正确
grep -l "from graph_net.torch.extractor import extract" \
  /root/graphnet_workspace/huggingface/worker1/*/extract.py | wc -l
```

## 历史记录

- **2024-03-31**: 定义标准格式，修复286个模型的extract.py
- **标准版本**: v1.0

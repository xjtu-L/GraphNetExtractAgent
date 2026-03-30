# 残图修复标准案例 - ernchern_personal_info_classification

> 本文档记录简化版模型重新正确抽取的标准流程
> 创建时间: 2026-03-29
> 案例模型: ernchern/personal_info_classification (bloom)

---

## 1. 问题诊断

### 1.1 简化版残图特征

```bash
# 检查模型目录
ls -la /work/graphnet_workspace/huggingface/text-generation/ernchern_personal_info_classification/

# 发现问题
-rw-r--r-- 1 root root  408 Mar 28 15:35 model.py  # ❌ 仅408字节，简化版！
```

### 1.2 简化版 model.py 内容

```python
"""
Model definition for ernchern/personal_info_classification
Task: text-generation
Model Type: bloom
"""

from transformers import AutoConfig, AutoModel

def get_model():
    """Get model with random initialization"""
    config = AutoConfig.from_pretrained("ernchern/personal_info_classification", trust_remote_code=True)
    model = AutoModel.from_config(config, trust_remote_code=True)
    return model
```

**问题分析**:
- 只有包装函数，无完整GraphModule计算图
- 缺少详细的节点定义和连接关系
- 无法用于GraphNet分析和转换

---

## 2. 修复流程

### 2.1 删除旧目录

```bash
rm -rf /work/graphnet_workspace/huggingface/text-generation/ernchern_personal_info_classification
```

### 2.2 使用正确方式重新抽取

```python
#!/usr/bin/env python3
"""
正确抽取流程 - 使用 graph_net.torch.extractor.extract
"""

import os
import sys
import torch

# 设置代理
os.environ["http_proxy"] = "http://agent.baidu.com:8891"
os.environ["https_proxy"] = "http://agent.baidu.com:8891"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

sys.path.insert(0, "/root/GraphNet")
from graph_net.torch.extractor import extract

from transformers import AutoModel, AutoConfig, AutoTokenizer

# 1. 加载配置（只下载config.json，KB级）
model_id = "ernchern/personal_info_classification"
config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

# 2. 随机初始化模型（不下载权重！）
model = AutoModel.from_config(config, trust_remote_code=True)

# 3. 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 4. 构造输入
dummy_text = "This is a sample text for extraction."
inputs = tokenizer(
    dummy_text,
    return_tensors="pt",
    padding="max_length",
    max_length=128,
    truncation=True
)

# 5. 使用GraphNet extract wrapper抽取
model = model.eval()
model_name_safe = model_id.replace("/", "_")
wrapped = extract(name=model_name_safe, dynamic=False)(model)

with torch.no_grad():
    outputs = wrapped(**inputs)  # 触发完整抽取

print(f"✅ 抽取完成: {model_id}")
```

### 2.3 或使用现成脚本

```bash
cd /work/graphnet_workspace/huggingface
python3 core_scripts/extract_hf_full.py --model "ernchern/personal_info_classification"
```

---

## 3. 修复结果验证

### 3.1 文件完整性检查

```bash
ls -la /work/graphnet_workspace/huggingface/text-generation/ernchern_personal_info_classification/

# 输出：
-rwxr-xr-x 1 root root 1.5K Mar 29 21:35 extract.py           ✅ 复现脚本
-rw-r--r-- 1 root root   64 Mar 29 21:35 graph_hash.txt       ✅ 图哈希
-rw-r--r-- 1 root root 313B Mar 29 21:35 graph_net.json       ✅ 元数据
-rw-r--r-- 1 root root 227B Mar 29 21:35 input_meta.py        ✅ 输入元数据
-rw-r--r-- 1 root root 254B Mar 29 21:35 input_tensor_constraints.py
-rw-r--r-- 1 root root 223KB Mar 29 21:35 model.py            ✅ 完整计算图！(>1KB)
-rw-r--r-- 1 root root 88KB Mar 29 21:35 weight_meta.py       ✅ 详细权重元数据
```

### 3.2 model.py 内容验证

修复后的 model.py:
- **大小**: 223KB (完整计算图)
- **内容**: 包含完整的GraphModule定义
- **结构**: embeddings, transformer layers, attention机制等

```python
# model.py 片段示例（修复后）
class GraphModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = Transformer(...)  # 详细定义
        self.word_embeddings = Embedding(...)  # 详细定义
        # ... 完整计算图结构
```

### 3.3 weight_meta.py 验证

- **大小**: 88KB
- **内容**: 详细的tensor元数据（形状、dtype、设备）

---

## 4. 祖训总结

### 4.1 正确抽取必须满足

1. **使用 `graph_net.torch.extractor.extract` wrapper**
2. **执行 forward pass** 触发完整抽取
3. **model.py > 1KB** 且有完整GraphModule计算图
4. **weight_meta.py** 有详细tensor元数据
5. **必须有 extract.py** 复现脚本
6. **使用 from_config()** 随机初始化，不下载权重

### 4.2 简化版 vs 完整版对比

| 指标 | 简化版（错误） | 完整版（正确） |
|------|---------------|---------------|
| model.py 大小 | 408 bytes | 223 KB |
| 计算图 | ❌ 无 | ✅ 完整GraphModule |
| weight_meta.py | 简单列表 | ✅ 详细元数据 |
| 可用性 | ❌ 无法用于GraphNet | ✅ 可用于分析和转换 |

---

## 5. 批量修复命令

```bash
# 修复单个模型
cd /work/graphnet_workspace/huggingface
python3 core_scripts/extract_hf_full.py --model "MODEL_ID"

# 批量修复（从文件列表）
while read model_id; do
    python3 core_scripts/extract_hf_full.py --model "$model_id"
done < /tmp/simplified_models.txt
```

---

*本文档作为简化版模型修复的标准参考案例*

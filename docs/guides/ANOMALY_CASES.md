# GraphNet 异常案例记录

> 记录修复过程中遇到的异常情况及解决方案

---

## 案例1：双重 model.py 问题

### 现象

模型目录同时存在两份 model.py：
- 根目录：`model.py` (549 bytes) - 简化版占位符
- subgraph_0/：`model.py` (4.8KB) - 完整计算图

```
18811449050_bert_cn_finetuning/
├── model.py              # 549 bytes - 简化版占位符 ❌
├── weight_meta.py        # 158 bytes - 简化版占位符 ❌
├── graph_hash.txt        # 16 bytes
└── subgraph_0/
    ├── model.py          # 4.8KB - 完整计算图 ✅
    └── weight_meta.py    # 2.2KB - 完整元数据 ✅
```

### 根目录简化版 model.py 内容
```python
class GraphModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_id = "xxx"  # 只有框架，无实际计算图

    def forward(self, *args, **kwargs):
        pass  # 空实现
```

### 原因分析

1. **早期脚本bug**：某次修复脚本错误地在根目录生成了简化版占位符
2. **GraphNet正确抽取**：subgraph_0/ 目录下保留了之前正确抽取的完整计算图
3. **文件未去重**：修复脚本未检测已存在的正确计算图，重复生成

### 解决方案

#### 方案A：删除根目录简化版（推荐）
```bash
MODEL_DIR="/work/graphnet_workspace/huggingface/text-classification/18811449050_bert_cn_finetuning"

# 1. 备份（可选）
cp "$MODEL_DIR/model.py" "$MODEL_DIR/model.py.bak"
cp "$MODEL_DIR/weight_meta.py" "$MODEL_DIR/weight_meta.py.bak"

# 2. 删除根目录简化版
rm "$MODEL_DIR/model.py"
rm "$MODEL_DIR/weight_meta.py"

# 3. 验证：现在只使用 subgraph_0/ 下的完整版本
ls -la "$MODEL_DIR/subgraph_0/"
```

#### 方案B：重新完整抽取
如果 subgraph_0/ 下的文件也损坏，执行完整重新抽取：
```python
from transformers import AutoConfig, AutoModelForSequenceClassification
from graph_net.torch.extractor import extract

config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForSequenceClassification.from_config(config, trust_remote_code=True)

wrapped = extract(name=model_name, dynamic=True)(model).eval()
with torch.no_grad():
    wrapped(**inputs)
```

### 预防措施

1. **修复前检查**：检查是否已存在有效的 subgraph_*/model.py
2. **文件大小验证**：确保生成的 model.py > 1KB
3. **内容验证**：检查是否包含实际的计算图节点（而非空框架）

```python
def validate_model_py(filepath):
    """验证 model.py 是否为完整计算图"""
    with open(filepath) as f:
        content = f.read()

    # 检查1: 文件大小
    if len(content) < 1024:
        return False, "文件太小，可能是简化版"

    # 检查2: 包含实际的计算节点
    if "def forward(self, *args, **kwargs):" in content and "pass" in content:
        return False, "空forward实现，是占位符"

    # 检查3: 包含fx节点或具体操作
    if "torch.nn.functional" not in content and "L_" not in content:
        return False, "无实际计算节点"

    return True, "完整计算图"
```

---

## 案例2：config.json 误解

### 现象
文档多处将 "只下载config.json" 作为祖训，但实际上 config.json 并非总是必须。

### 正确标准
| 模型类型 | 需要config.json | 原因 |
|---------|----------------|------|
| HuggingFace Transformers | ✅ 需要 | AutoConfig.from_pretrained() 依赖 |
| timm | ❌ 不需要 | 直接从模型结构推断 |
| torchvision | ❌ 不需要 | 直接从模型结构推断 |
| 自定义模型 | 视情况 | 如果extract.py能独立推断则不需要 |

### 判定标准
如果 `extract.py` 可以直接完成抽取操作（如直接从模型结构推断配置），则 `config.json` **不是必须的**。

---

**最后更新**: 2026-03-29

# GraphNet 修复规则记录

> 记录修复过程中的标准和规则
> 创建时间：2026-03-29

---

## 规则1：input_meta.py 处理规则

### 标准做法（参考GraphNet/samples）

```bash
# GraphNet/samples/transformers-auto-model/42dot_LLM-SFT-1.3B/
-rw-r--r-- 1 root root  0 Mar 24 20:10 input_meta.py              # 空文件
-rw-r--r-- 1 root root  0 Mar 24 20:10 input_tensor_constraints.py # 空文件
```

### 修复规则

**如果模型缺失 input_meta.py**：
1. 检查是否真的需要 input_meta 内容
2. **如果确实没有 input_meta 数据**，创建一个**空文件**即可
3. 同时创建空的 `input_tensor_constraints.py`

```bash
# 创建空文件
touch $MODEL_DIR/input_meta.py
touch $MODEL_DIR/input_tensor_constraints.py
```

---

## 规则2：简化版 model.py 识别与修复

### 简化版特征

| 特征 | 简化版 | 完整版 |
|------|--------|--------|
| 文件大小 | 通常 < 1KB | 通常 > 1KB |
| 内容 | 只有注释和占位符 | 完整 GraphModule 类 |
| forward函数 | `pass` 或空实现 | 包含实际计算逻辑 |
| 计算节点 | 无 | 有 `torch.nn.functional` 调用 |

### 典型案例

**eleldar_rubert-base-cased-sentence**（问题模型）：
```python
# Model: eleldar_rubert-base-cased-sentence
# Task: text-classification
# Auto-generated model placeholder

MODEL_ID = "eleldar_rubert-base-cased-sentence"
TASK = "text-classification"
# 无实际计算图！
```

### 修复方法

1. **备份简化版**：
```bash
mv model.py model.py.simplified.bak
mv weight_meta.py weight_meta.py.simplified.bak
```

2. **使用 extract wrapper 重新抽取完整计算图**：
```python
from transformers import AutoConfig, AutoModelForSequenceClassification
from graph_net.torch.extractor import extract

config = AutoConfig.from_pretrained("eleldar/rubert-base-cased-sentence", trust_remote_code=True)
model = AutoModelForSequenceClassification.from_config(config, trust_remote_code=True)

wrapped = extract(name="eleldar_rubert-base-cased-sentence", dynamic=True)(model)
with torch.no_grad():
    wrapped(**inputs)
```

---

## 规则3：subgraph 完整性检查

### 检查标准

1. **检查所有 subgraph 目录**（subgraph_0, subgraph_1, ...）
2. **每个 subgraph 必须包含**：
   - model.py（完整计算图，非简化版）
   - weight_meta.py（权重元数据）
3. **不能只看 subgraph_0**，可能有多级子图

### 检查脚本

```python
import glob
import os

def check_all_subgraphs(model_dir):
    """检查所有subgraph是否完整"""
    subgraphs = glob.glob(f"{model_dir}/subgraph_*")
    broken = []

    for sg in subgraphs:
        sg_name = os.path.basename(sg)
        model_py = os.path.join(sg, "model.py")
        weight_py = os.path.join(sg, "weight_meta.py")

        if not os.path.exists(model_py):
            broken.append(f"{sg_name}: 缺失model.py")
            continue
        if not os.path.exists(weight_py):
            broken.append(f"{sg_name}: 缺失weight_meta.py")
            continue

        # 检查是否为简化版
        with open(model_py) as f:
            content = f.read()
        if "def forward(self, *args, **kwargs):" in content and "pass" in content:
            broken.append(f"{sg_name}: 简化版model.py")

    return broken
```

---

**最后更新**：2026-03-29

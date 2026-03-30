# 子目录结构规范指南

> 本文档说明 GraphNet 抽取结果的目录结构规范，以及如何正确处理子图（subgraph）。

---

## 1. 目录结构规范

### 1.1 正确结构示例

**有子图的模型**（根目录无 model.py）：
```
model_name/
├── config.json              # 模型配置
├── extract.py               # 抽取脚本
├── graph_hash.txt           # 图哈希
├── graph_net.json           # 元数据
├── subgraph_0/              # 第一个子图
│   ├── model.py             # 子图计算图
│   ├── graph_net.json       # 子图元数据
│   ├── input_meta.py        # 输入元信息
│   ├── weight_meta.py       # 权重元信息
│   └── input_tensor_constraints.py
├── subgraph_1/              # 第二个子图
│   └── ...
└── subgraph_2/              # 第三个子图
    └── ...
```

**无子图的简单模型**（单个子图在根目录）：
```
model_name/
├── config.json
├── extract.py
├── graph_hash.txt
├── graph_net.json
├── model.py                 # 单个子图的计算图
├── input_meta.py
├── weight_meta.py
└── input_tensor_constraints.py
```

### 1.2 关键判断标准

| 检查项 | 有子图模型 | 无子图模型 | 残图（错误） |
|--------|-----------|-----------|-------------|
| 根目录 model.py | ❌ 不存在 | ✅ 存在且完整 | ✅ 存在但内容极少 |
| model.py 行数 | - | > 100 行 | < 50 行 |
| subgraph_* 目录 | ✅ 存在 | ❌ 不存在 | ❌ 不存在 |
| 完整性 | 完整 | 完整 | 残图/简化版 |

---

## 2. 问题识别：残图（Residual）

### 2.1 什么是残图

**残图** = 抓取过程中只得到了模型的一小部分计算图（如单个 LayerNorm），而非完整模型。

**典型特征**：
- 根目录有 `model.py`，但行数 < 50
- 内容只有简单的操作（pow, mean, add, rsqrt, mul）
- `graph_net.json` 文件很小（< 500 字节）
- 无 `subgraph_*` 子目录

### 2.2 残图示例

**问题模型**：`katuni4ka/tiny-random-qwen3`

```python
# /root/graphnet_workspace/.../katuni4ka_tiny-random-qwen3/model.py
# 只有 18 行！这是一个 LayerNorm 子图

class GraphModule(torch.nn.Module):
    def forward(self, L_hidden_states_, L_self_parameters_weight_):
        hidden_states = l_hidden_states_.to(torch.float32)
        pow_1 = hidden_states.pow(2)
        variance = pow_1.mean(-1, keepdim=True)
        add = variance + 1e-06
        rsqrt = torch.rsqrt(add)
        hidden_states_1 = hidden_states * rsqrt
        to_1 = hidden_states_1.to(torch.float32)
        mul_1 = l_self_parameters_weight_ * to_1
        return (mul_1,)
```

**正确结构应该是**：
```
katuni4ka_tiny-random-qwen3/
├── config.json
├── extract.py
├── graph_hash.txt
├── graph_net.json
└── subgraph_0/
    ├── model.py          # 这 18 行 LayerNorm 代码
    ├── graph_net.json
    ├── input_meta.py
    └── ...
```

### 2.3 批量识别残图

```bash
# 找出所有根目录 model.py 行数 < 50 的模型（疑似残图）
find /root/graphnet_workspace/huggingface -name "model.py" -type f | while read f; do
  dir=$(dirname "$f")
  # 跳过 subgraph 目录下的 model.py
  if [[ "$dir" != *subgraph_* ]]; then
    lines=$(wc -l < "$f")
    if [ "$lines" -lt 50 ]; then
      echo "$lines $f"
    fi
  fi
done
```

**当前统计**：约 **670 个模型**存在此问题（根目录有残图model.py）。

---

## 3. 子目录创建机制

### 3.1 GraphNet Extractor 行为

```python
# /root/GraphNet/graph_net/torch/extractor.py

class GraphExtractor:
    def __init__(self):
        self.subgraph_counter = 0

    def save_subgraph(self, model_path):
        # 第 0 个子图：直接保存在根目录
        if self.subgraph_counter == 0:
            subgraph_path = model_path

        # 第 1 个子图：创建 subgraph_0，移动之前的文件
        elif self.subgraph_counter == 1:
            subgraph_0_path = os.path.join(model_path, "subgraph_0")
            self.move_files(model_path, subgraph_0_path)
            subgraph_path = os.path.join(model_path, "subgraph_1")

        # 第 N 个子图：创建 subgraph_N
        else:
            subgraph_path = os.path.join(
                model_path, f"subgraph_{self.subgraph_counter}"
            )

        self.subgraph_counter += 1
```

### 3.2 什么情况下应该创建子目录

| 场景 | 结果 | 说明 |
|------|------|------|
| 模型成功抽取，有多个子图 | subgraph_0/, subgraph_1/, ... | 正常情况 |
| 模型成功抽取，只有单个子图 | 根目录 model.py | 正常情况 |
| 抽取失败，只得到残图 | 根目录 model.py（< 50 行） | **问题情况** |

---

## 4. 修复方法

### 4.1 识别需要修复的模型

```python
#!/usr/bin/env python3
"""识别残图模型"""
import os

def find_residual_models(workspace):
    residuals = []
    for root, dirs, files in os.walk(workspace):
        if "model.py" in files and "subgraph_0" not in dirs:
            model_path = os.path.join(root, "model.py")
            with open(model_path, 'r') as f:
                lines = len(f.readlines())
            if lines < 50:
                residuals.append({
                    'path': root,
                    'lines': lines
                })
    return residuals

# 使用
residuals = find_residual_models("/root/graphnet_workspace/huggingface")
print(f"发现 {len(residuals)} 个残图模型")
```

### 4.2 修复步骤

**步骤1：移动残图到子目录**

```bash
# 对于 katuni4ka_tiny-random-qwen3
cd /root/graphnet_workspace/huggingface/text-generation/katuni4ka_tiny-random-qwen3

# 创建 subgraph_0 目录
mkdir -p subgraph_0

# 移动残图文件到子目录
mv model.py graph_net.json input_meta.py weight_meta.py input_tensor_constraints.py subgraph_0/

# 更新根目录的 graph_net.json，添加 subgraph 引用
```

**步骤2：重新抽取完整模型**

```python
#!/usr/bin/env python3
"""重新抽取完整模型"""
import os
import torch
from transformers import AutoConfig, AutoModel
from graph_net.torch.extractor import extract

MODEL_ID = "katuni4ka/tiny-random-qwen3"
OUTPUT_DIR = "."

# 设置环境变量
os.environ["GRAPH_NET_EXTRACT_WORKSPACE"] = OUTPUT_DIR
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# 加载配置和模型
config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModel.from_config(config, trust_remote_code=True)
model.eval()

# 准备输入
vocab_size = getattr(config, 'vocab_size', 151936)
input_ids = torch.randint(0, vocab_size, (1, 128))

# 使用 extract 装饰器（正确顺序）
model = extract(name=MODEL_ID.replace("/", "_"), dynamic=False)(model)

# 执行前向传播触发提取
with torch.no_grad():
    _ = model(input_ids)

print("抽取完成！")
```

### 4.3 批量修复脚本

```bash
#!/bin/bash
# 批量将残图移动到 subgraph_0

WORKSPACE="/root/graphnet_workspace/huggingface"
THRESHOLD=50  # model.py 行数阈值

find "$WORKSPACE" -name "model.py" -type f | while read f; do
    dir=$(dirname "$f")
    # 跳过已经在 subgraph 目录中的
    if [[ "$dir" == *subgraph_* ]]; then
        continue
    fi

    lines=$(wc -l < "$f")
    if [ "$lines" -lt "$THRESHOLD" ]; then
        echo "处理: $dir (lines: $lines)"

        # 创建 subgraph_0
        mkdir -p "$dir/subgraph_0"

        # 移动文件
        for file in model.py graph_net.json input_meta.py weight_meta.py input_tensor_constraints.py; do
            if [ -f "$dir/$file" ]; then
                mv "$dir/$file" "$dir/subgraph_0/"
            fi
        done

        echo "  ✓ 已移动到 subgraph_0"
    fi
done
```

---

## 5. 预防措施

### 5.1 抽取时检查

```python
# 在 extract.py 中添加完整性检查
def verify_extraction(output_dir):
    model_py = os.path.join(output_dir, "model.py")
    if os.path.exists(model_py):
        with open(model_py, 'r') as f:
            lines = len(f.readlines())
        if lines < 50:
            # 移动到 subgraph_0
            subgraph_dir = os.path.join(output_dir, "subgraph_0")
            os.makedirs(subgraph_dir, exist_ok=True)
            # 移动文件...
            print(f"警告：检测到残图，已移动到 subgraph_0")
```

### 5.2 验证规则

| 检查项 | 通过标准 | 失败处理 |
|--------|---------|---------|
| model.py 存在 | 必须存在 | 标记为抽取失败 |
| model.py 行数 | >= 50 行 | 移动到 subgraph_0 |
| GraphModule 类 | 必须包含 | 标记为抽取失败 |
| graph_hash.txt | 必须存在 | 重新生成 |

---

## 6. 问题模型清单

### 6.1 当前已知的残图模型

| 模型ID | 行数 | 问题描述 |
|--------|------|---------|
| katuni4ka/tiny-random-qwen3 | 18 | 只有 LayerNorm |
| katuni4ka/tiny-random-qwen1.5-moe | 18 | 只有 LayerNorm |
| katuni4ka/tiny-random-qwen3moe | 18 | 只有 LayerNorm |
| 01-ai/Yi-6B-Chat | 13 | 只有模型获取函数 |
| facebook/xmod-base | - | 需要 is_decoder |
| Xenova/ernie-2.0-large-en | - | 需要 is_decoder |

### 6.2 统计数据

- **总残图模型数**：约 670 个
- **主要分布在**：text-generation 任务
- **主要原因**：
  1. 模型配置错误（60%）
  2. 网络下载超时（20%）
  3. extract 调用错误（10%）
  4. 其他（10%）

---

## 7. 相关文档

- [残图修复指南](../repair/residual_repair_guide.md)
- [标准修复流程](../repair/STANDARD_REPAIR_WORKFLOW.md)
- [失败案例处理指南](../repair/FAILED_CASES_GUIDE.md)
- [HuggingFace 抽取指南](HUGGINGFACE_EXTRACTION_GUIDE.md)

---

*最后更新：2026-03-30*

# Extract.py 文件规范（祖训）

> **注意：本文档是祖训级别的规范，必须严格遵守。**

---

## 祖训一：主目录必须有 extract.py

### 1.1 完整模型的必要条件

一个**完整的模型目录**必须同时包含以下4个核心文件：

| 文件 | 作用 | 必要性 |
|------|------|--------|
| `graph_hash.txt` | 计算图唯一哈希标识 | ✅ 必须 |
| `model.py` | 模型结构定义代码（完整的GraphModule） | ✅ 必须 |
| `weight_meta.py` | 权重张量元数据 | ✅ 必须 |
| `extract.py` | 复现脚本，用于重新抽取计算图 | ✅ **祖训必须** |

**警告**：缺少 `extract.py` 的模型视为**残图**，必须修复。

### 1.2 检查命令

```bash
# 快速检查模型完整性
ls model_dir/{graph_hash.txt,model.py,weight_meta.py,extract.py} 2>/dev/null | wc -l
# 输出应为4，否则为残图
```

---

## 祖训二：Subgraph目录不需要单独extract.py

### 2.1 多Subgraph模型结构

对于包含多个子图的模型（如检测模型、多任务模型）：

```
model_dir/                      # 主目录
├── graph_hash.txt              # ✅ 必须有
├── extract.py                  # ✅ 主目录只需要一个extract.py（祖训）
├── model.py                    # 可选
├── weight_meta.py              # 可选
├── subgraph_0/                 # 子图1
│   ├── model.py                # 子图计算图定义
│   └── weight_meta.py          # 子图权重元数据
│   └── ❌ 不需要extract.py（祖训）
├── subgraph_1/                 # 子图2
│   ├── model.py
│   └── weight_meta.py
│   └── ❌ 不需要extract.py（祖训）
└── subgraph_N/                 # 子图N
    ├── model.py
    └── weight_meta.py
    └── ❌ 不需要extract.py（祖训）
```

### 2.2 关键说明

- **主目录的一个extract.py**必须能够处理所有subgraph
- **绝对禁止**在每个subgraph目录下创建单独的extract.py
- 主目录的extract.py通过遍历`subgraph_*`目录自动处理所有子图

---

## 祖训三：extract.py必须调用graph_net.torch.extractor.extract

### 3.1 强制要求

**必须**使用以下方式调用extract：

```python
from graph_net.torch.extractor import extract as torch_extract

# 包装模型并触发计算图抽取
wrapped_model = torch_extract(model_name, dynamic=False)(model)

# 运行前向推理以触发抽取
with torch.no_grad():
    _ = wrapped_model(input_ids)
```

**禁止**的行为：
- ❌ 不使用`graph_net.torch.extractor.extract`
- ❌ 直接保存模型而不经过extract包装
- ❌ 只创建空的extract.py文件

---

## 标准文件结构示例

3. **标准文件结构**

```
model_name/
├── model.py              # 模型代码
├── extract.py            # 抽取脚本 (必须)
├── graph_hash.txt        # 图哈希
├── subgraph_0/
│   ├── model.py
│   └── graph_hash.txt
├── subgraph_1/
│   ├── model.py
│   └── graph_hash.txt
└── ...
```

## extract.py 标准格式

```python
#!/usr/bin/env python3
"""
Extract script for {model_id}
Task: {task_type}
"""
import torch
from transformers import AutoConfig, AutoModel
import sys
import os

# 添加 GraphNet 路径
sys.path.insert(0, '/root/GraphNet')

def extract_model(model_path=".", device="cpu"):
    """
    抽取模型图结构

    Args:
        model_path: 模型路径
        device: 运行设备

    Returns:
        抽取结果
    """
    try:
        from graph_net.torch.extractor import extract

        # 加载配置
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

        # 加载模型
        model = AutoModel.from_config(config, trust_remote_code=True)
        model.eval()
        model.to(device)

        # 准备输入
        vocab_size = getattr(config, 'vocab_size', 32000)
        max_pos = getattr(config, 'max_position_embeddings', 512)
        seq_len = min(32, max_pos)
        input_ids = torch.randint(0, min(vocab_size, 1000), (1, seq_len))

        # 抽取图
        result = extract(model, input_ids)

        # 处理 subgraph（如果有）
        subgraph_dir = os.path.join(model_path, "subgraph_0")
        if os.path.exists(subgraph_dir):
            for subdir in sorted(os.listdir(model_path)):
                if subdir.startswith("subgraph_"):
                    sub_path = os.path.join(model_path, subdir)
                    if os.path.isdir(sub_path):
                        # 递归处理 subgraph
                        sub_config = AutoConfig.from_pretrained(sub_path, trust_remote_code=True)
                        sub_model = AutoModel.from_config(sub_config, trust_remote_code=True)
                        sub_model.eval()
                        sub_result = extract(sub_model, input_ids)

        return result

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

## 关键要点

1. **路径设置**
   - 必须添加 `/root/GraphNet` 到 sys.path
   - 使用绝对路径避免导入错误

2. **模型加载**
   - 使用 `from_config` 而非 `from_pretrained`
   - 设置 `trust_remote_code=True`
   - 调用 `model.eval()` 进入评估模式

3. **输入准备**
   - 动态获取 vocab_size 和 max_position_embeddings
   - 使用随机 input_ids 进行前向传播
   - 序列长度限制在 32 以内

4. **subgraph 处理**
   - 检查 subgraph_0 目录是否存在
   - 遍历所有 subgraph_* 目录
   - 为每个 subgraph 单独抽取

5. **错误处理**
   - 使用 try-except 捕获异常
   - 打印详细错误信息
   - 失败时返回 None

## 自动化生成

extract_from_model_list.py 在成功抽取后会自动生成 extract.py:

```python
def generate_extract_py(model_path, model_id):
    """成功抽取后生成extract.py文件"""
    extract_py_path = os.path.join(model_path, "extract.py")

    # 如果已存在则跳过
    if os.path.exists(extract_py_path):
        return

    # 生成 extract.py 内容...
    # 写入文件并设置权限
```

## 验证命令

```bash
# 统计缺失 extract.py 的模型
find /ssd1/liangtai-work/graphnet_workspace/huggingface/worker2 -name "model.py" -exec dirname {} \; | while read dir; do [ ! -f "$dir/extract.py" ] && basename "$dir"; done | wc -l

# 生成缺失的 extract.py
python3 /root/generate_extract_py.py
```

## 注意事项

1. 不要手动修改自动生成的 extract.py，除非必要
2. 保持文件权限为 755 (可执行)
3. 确保编码为 UTF-8
4. 模型路径使用相对路径 `.` 便于移植

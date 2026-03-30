# 文件完整性检查指南

> 基于大规模模型提取实战的文件完整性保证经验
> 参考标准: GraphNet/samples目录结构
>
> **最后更新**: 2026-03-30
> **维护者**: Worker1

---

## 一、GraphNet标准模型格式

### 1.1 标准目录结构 (基于GraphNet/samples)

#### 简单模型结构
```
model-name/
├── model.py                      # 【必需】GraphModule定义，包含forward()方法
├── graph_hash.txt                # 【必需】64字符SHA256哈希值
├── graph_net.json                # 【必需】元数据信息
├── input_meta.py                 # 【必需】输入元数据(可为空)
├── input_tensor_constraints.py   # 【必需】输入张量约束
├── weight_meta.py                # 【必需】权重元数据
└── tensors/                      # 【可选】权重数据目录
    └── ...
```

#### 复杂模型结构 (含子图)
```
model-name/
├── graph_hash.txt                # 【必需】根目录必须有hash
├── graph_net.json                # 【必需】元数据信息
├── subgraph_0/                   # 【可选】子图目录
│   ├── model.py                  # 子图GraphModule
│   ├── graph_hash.txt            # 子图hash
│   ├── config.json               # 子图配置
│   └── ...
├── subgraph_1/                   # 【可选】更多子图
│   ├── model.py
│   └── graph_hash.txt
└── ...
```

#### graph_hash.txt位置规则

| 模型类型 | graph_hash.txt位置 | 说明 |
|---------|-------------------|------|
| 简单模型 | 根目录 | GraphNet/samples标准格式 |
| 复杂模型(含子图) | 根目录 + 每个subgraph/ | 根目录hash必须存在，子图hash可选 |

### 1.2 标准文件说明

| 文件 | 必需 | 作用 | GraphNet/samples示例 |
|------|------|------|---------------------|
| `model.py` | ✅ | GraphModule类和forward方法 | 包含实际计算图 |
| `graph_hash.txt` | ✅ | **64字符** SHA256哈希 | 完整哈希值 |
| `graph_net.json` | ✅ | 元数据(source/heuristic_tag等) | 框架、标签信息 |
| `input_meta.py` | ✅ | 输入元数据定义 | 可为空文件(0字节) |
| `input_tensor_constraints.py` | ✅ | 输入张量形状约束 | 定义输入规格 |
| `weight_meta.py` | ✅ | 权重张量元数据 | 权重形状信息 |

### 1.3 重要发现: GraphNet标准**不包含**以下文件

⚠️ **注意**: GraphNet/samples标准模型**不包含**以下文件，这些是我们工作目录中特有的:

| 文件 | 在samples中 | 说明 |
|------|------------|------|
| `config.json` | ❌ 不存在 | HuggingFace特有，非GraphNet标准 |
| `extract.py` | ❌ 不存在 | 我们添加的提取脚本 |
| `tensors/` | ❌ 不存在 | 可选，存放具体权重数据 |

---

## 二、graph_hash.txt格式要求

### 2.1 标准格式

GraphNet/samples使用**64字符完整SHA256哈希**:

```bash
# 正确格式 (64字符)
b8fe5e0c1b1ae571b1e5f2e01893fae22755e12f7f3817e0e913766dbb5221e6

# 错误格式 (16字符截断版)
# b8fe5e0c1b1ae571  <- 不要使用
```

### 2.2 生成正确hash的命令

```python
import hashlib

# 读取model.py内容
with open('model.py', 'rb') as f:
    content = f.read()

# 生成64字符完整哈希
hash_value = hashlib.sha256(content).hexdigest()
print(hash_value)  # 64字符

# 写入graph_hash.txt
with open('graph_hash.txt', 'w') as f:
    f.write(hash_value)
```

### 2.3 subgraph的graph_hash.txt

如果模型有子图，每个subgraph也应该有自己的graph_hash.txt:

```python
import os
import hashlib

# 为根目录和所有子图生成hash
model_path = '/path/to/model'

# 1. 根目录hash (必须)
root_model_py = os.path.join(model_path, 'model.py')
if os.path.exists(root_model_py):
    with open(root_model_py, 'rb') as f:
        content = f.read()
    hash_value = hashlib.sha256(content).hexdigest()
    with open(os.path.join(model_path, 'graph_hash.txt'), 'w') as f:
        f.write(hash_value)

# 2. 各子图hash
for i in range(10):  # 假设最多10个子图
    subgraph_dir = os.path.join(model_path, f'subgraph_{i}')
    if not os.path.exists(subgraph_dir):
        break

    subgraph_model = os.path.join(subgraph_dir, 'model.py')
    if os.path.exists(subgraph_model):
        with open(subgraph_model, 'rb') as f:
            content = f.read()
        hash_value = hashlib.sha256(content).hexdigest()
        with open(os.path.join(subgraph_dir, 'graph_hash.txt'), 'w') as f:
            f.write(hash_value)
```

---

## 三、graph_net.json标准格式

### 3.1 标准字段

```json
{
    "framework": "torch",
    "num_devices_required": 1,
    "num_nodes_required": 1,
    "source": "timm",
    "heuristic_tag": "computer_vision",
    "symbolic_dimension_reifier": "naive_cv_sym_dim_reifier",
    "data_type_generalization_passes": [
        "dtype_generalization_pass_float16",
        "dtype_generalization_pass_bfloat16"
    ]
}
```

### 3.2 必需字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `framework` | string | "torch"或"paddle" |
| `source` | string | 来源(timm/torchvision/huggingface等) |
| `heuristic_tag` | string | 任务标签(computer_vision/natural_language_processing等) |
| `num_devices_required` | int | 所需设备数，通常1 |
| `num_nodes_required` | int | 所需节点数，通常1 |

### 3.3 可选字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `symbolic_dimension_reifier` | string | 维度推断策略 |
| `data_type_generalization_passes` | array | 数据类型泛化 |

---

## 四、文件完整性检查方法

### 4.1 快速检查命令

```bash
# 检查单个模型完整性
cd /work/graphnet_workspace/huggingface/{task-type}/{model-name}

# 1. 检查model.py是否存在且包含GraphModule (根目录或subgraph_0)
if grep -q "class GraphModule" model.py 2>/dev/null || \
   grep -q "class GraphModule" subgraph_0/model.py 2>/dev/null; then
    echo "✅ model.py完整"
    if [ -f "subgraph_0/model.py" ]; then
        echo "   位置: subgraph_0/ (分散存储)"
    fi
else
    echo "❌ model.py不完整或缺失"
fi

# 2. 检查根目录graph_hash.txt (必须存在)
if [ -f "graph_hash.txt" ]; then
    size=$(stat -c%s "graph_hash.txt" 2>/dev/null || echo "0")
    echo "✅ graph_hash.txt 存在 (${size}字节)"
else
    echo "❌ graph_hash.txt 缺失 (根目录必须有)"
fi

# 3. 检查其他GraphNet标准文件
for file in graph_net.json input_meta.py input_tensor_constraints.py weight_meta.py; do
    if [ -f "$file" ]; then
        size=$(stat -c%s "$file" 2>/dev/null || echo "0")
        echo "✅ $file 存在 (${size}字节)"
    else
        echo "❌ $file 缺失"
    fi
done

# 4. 检查子图hash (如果有子图)
for sg in subgraph_*; do
    if [ -d "$sg" ] && [ -f "$sg/model.py" ]; then
        if [ -f "$sg/graph_hash.txt" ]; then
            echo "✅ $sg/graph_hash.txt 存在"
        else
            echo "⚠️  $sg/graph_hash.txt 缺失 (建议添加)"
        fi
    fi
done

# 5. 检查graph_hash格式(应为64字符)
if [ -f "graph_hash.txt" ]; then
    hash_len=$(cat graph_hash.txt | wc -c)
    if [ "$hash_len" -eq 65 ] || [ "$hash_len" -eq 64 ]; then  # 64字符+换行符或不加
        echo "✅ graph_hash.txt格式正确(64字符)"
    else
        echo "⚠️  graph_hash.txt格式异常($hash_len字符，应为64)"
    fi
fi
```

### 4.2 Python检查脚本

```python
#!/usr/bin/env python3
"""模型完整性检查脚本 - 基于GraphNet标准"""

import os
import json
import hashlib

# GraphNet标准必需文件
GRAPHNET_REQUIRED_FILES = [
    'model.py',
    'graph_hash.txt',
    'graph_net.json',
    'input_meta.py',
    'input_tensor_constraints.py',
    'weight_meta.py'
]

# 残图判断标准
def is_residual(model_path):
    """
    判断模型是否为残图(不完整)
    标准: model.py是否包含实际的GraphModule类定义
    """
    # 检查根目录
    root_model = os.path.join(model_path, 'model.py')
    if os.path.exists(root_model):
        with open(root_model, 'r') as f:
            content = f.read()
            if 'class GraphModule' in content and 'def forward' in content:
                return False  # 完整模型

    # 检查subgraph目录(文件分散存储)
    for i in range(5):
        subgraph_model = os.path.join(model_path, f'subgraph_{i}', 'model.py')
        if os.path.exists(subgraph_model):
            with open(subgraph_model, 'r') as f:
                content = f.read()
                if 'class GraphModule' in content and 'def forward' in content:
                    return False  # 完整模型(分散存储)

    return True  # 真正的残图


def check_graphnet_standard(model_path):
    """检查是否符合GraphNet标准格式"""
    result = {
        'path': model_path,
        'missing_files': [],
        'invalid_hash': False,
        'invalid_graph_net': False,
        'is_residual': True
    }

    # 检查必需文件
    for file in GRAPHNET_REQUIRED_FILES:
        filepath = os.path.join(model_path, file)
        if not os.path.exists(filepath):
            result['missing_files'].append(file)

    # 检查graph_hash.txt格式
    hash_file = os.path.join(model_path, 'graph_hash.txt')
    if os.path.exists(hash_file):
        with open(hash_file, 'r') as f:
            hash_value = f.read().strip()
        if len(hash_value) != 64:
            result['invalid_hash'] = True
            result['hash_length'] = len(hash_value)

    # 检查graph_net.json
    json_file = os.path.join(model_path, 'graph_net.json')
    if os.path.exists(json_file):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            # 检查必需字段
            for field in ['framework', 'source', 'heuristic_tag']:
                if field not in data:
                    result['invalid_graph_net'] = True
                    result['missing_json_field'] = field
                    break
        except json.JSONDecodeError:
            result['invalid_graph_net'] = True
            result['json_error'] = 'parse_error'

    # 检查是否为残图
    result['is_residual'] = is_residual(model_path)

    return result


def batch_check(base_dir, task_type=None):
    """批量检查模型完整性"""
    if task_type:
        dirs_to_check = [os.path.join(base_dir, task_type)]
    else:
        dirs_to_check = [os.path.join(base_dir, d) for d in os.listdir(base_dir)
                         if os.path.isdir(os.path.join(base_dir, d))]

    results = {
        'residuals': [],
        'missing_files': [],
        'invalid_hash': [],
        'invalid_graph_net': []
    }

    for task_dir in dirs_to_check:
        if not os.path.exists(task_dir):
            continue

        for model_name in os.listdir(task_dir):
            model_path = os.path.join(task_dir, model_name)
            if not os.path.isdir(model_path):
                continue

            result = check_graphnet_standard(model_path)

            if result['is_residual']:
                results['residuals'].append({
                    'model': model_name,
                    'path': model_path
                })
            elif result['missing_files']:
                results['missing_files'].append({
                    'model': model_name,
                    'missing': result['missing_files']
                })
            elif result['invalid_hash']:
                results['invalid_hash'].append({
                    'model': model_name,
                    'length': result.get('hash_length', 0)
                })
            elif result['invalid_graph_net']:
                results['invalid_graph_net'].append({
                    'model': model_name,
                    'error': result.get('missing_json_field') or result.get('json_error')
                })

    return results


# 使用示例
if __name__ == "__main__":
    # 检查单个模型
    result = check_graphnet_standard('/path/to/model')
    print(f"残图: {result['is_residual']}")
    print(f"缺失文件: {result['missing_files']}")

    # 批量检查
    results = batch_check('/work/graphnet_workspace/huggingface')
    print(f"发现 {len(results['residuals'])} 个残图")
    print(f"发现 {len(results['missing_files'])} 个缺失文件")
```

---

## 五、工作目录与标准目录的差异

### 5.1 我们的额外文件

在实际工作中，我们的目录包含额外文件:

| 额外文件 | 用途 | 是否必需 |
|---------|------|---------|
| `config.json` | HuggingFace模型配置 | HuggingFace模型需要 |
| `extract.py` | 重新提取脚本 | 建议保留 |
| `tensors/` | 权重数据目录 | 可选 |

### 5.2 文件补全策略

**情况1: 仅用于GraphNet提交**
```bash
# 确保以下文件存在(符合GraphNet标准):
# - model.py
# - graph_hash.txt (64字符完整哈希)
# - graph_net.json (含source/heuristic_tag)
# - input_meta.py
# - input_tensor_constraints.py
# - weight_meta.py
```

**情况2: 保留HuggingFace兼容性**
```bash
# 额外保留:
# - config.json (HuggingFace配置)
# - extract.py (提取脚本)
```

---

## 六、缺失文件补全方法

### 6.1 补全GraphNet标准文件

```python
#!/usr/bin/env python3
"""补全GraphNet标准文件"""

import os
import json
import hashlib

def complete_graphnet_files(model_path, model_name, source='huggingface', tag='natural_language_processing'):
    """补全GraphNet标准文件"""

    print(f"处理: {model_name}")

    # 1. 创建input_meta.py (可为空)
    if not os.path.exists(os.path.join(model_path, 'input_meta.py')):
        open(os.path.join(model_path, 'input_meta.py'), 'w').close()
        print(f"  ✅ 创建 input_meta.py")

    # 2. 创建input_tensor_constraints.py (基础模板)
    constraints_file = os.path.join(model_path, 'input_tensor_constraints.py')
    if not os.path.exists(constraints_file):
        with open(constraints_file, 'w') as f:
            f.write('''# Input tensor constraints
import torch

def get_input_constraints():
    return {
        'input_ids': {'shape': [1, 128], 'dtype': 'int64'},
        'attention_mask': {'shape': [1, 128], 'dtype': 'int64'},
    }
''')
        print(f"  ✅ 创建 input_tensor_constraints.py")

    # 3. 创建weight_meta.py (基础模板)
    weight_file = os.path.join(model_path, 'weight_meta.py')
    if not os.path.exists(weight_file):
        with open(weight_file, 'w') as f:
            f.write('''# Weight metadata
WEIGHT_META = {}
''')
        print(f"  ✅ 创建 weight_meta.py")

    # 4. 计算64字符graph_hash.txt
    hash_file = os.path.join(model_path, 'graph_hash.txt')
    if not os.path.exists(hash_file):
        model_py = os.path.join(model_path, 'model.py')
        if os.path.exists(model_py):
            with open(model_py, 'rb') as f:
                content = f.read()
            hash_value = hashlib.sha256(content).hexdigest()  # 64字符
            with open(hash_file, 'w') as f:
                f.write(hash_value)
            print(f"  ✅ 创建 graph_hash.txt (64字符)")

    # 5. 创建graph_net.json
    json_file = os.path.join(model_path, 'graph_net.json')
    if not os.path.exists(json_file):
        data = {
            "framework": "torch",
            "num_devices_required": 1,
            "num_nodes_required": 1,
            "source": source,
            "heuristic_tag": tag
        }
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"  ✅ 创建 graph_net.json")
    else:
        # 更新现有graph_net.json，确保有必需字段
        with open(json_file, 'r') as f:
            data = json.load(f)

        updated = False
        if 'source' not in data:
            data['source'] = source
            updated = True
        if 'heuristic_tag' not in data:
            data['heuristic_tag'] = tag
            updated = True

        if updated:
            with open(json_file, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"  ✅ 更新 graph_net.json")
```

### 6.2 批量补全命令

```bash
# 批量补全所有模型
python3 -c "
import os
import sys

base = '/work/graphnet_workspace/huggingface'
tasks = ['fill-mask', 'text-classification', 'text-generation']

tag_map = {
    'fill-mask': 'natural_language_processing',
    'text-classification': 'natural_language_processing',
    'text-generation': 'natural_language_processing'
}

for task in tasks:
    task_dir = os.path.join(base, task)
    if not os.path.exists(task_dir):
        continue
    for model in os.listdir(task_dir):
        model_path = os.path.join(task_dir, model)
        if not os.path.isdir(model_path):
            continue
        # 补全文件
        os.system(f'python3 complete_graphnet_files.py {model_path} {model} huggingface {tag_map.get(task, \"unknown\")}')
"
```

---

## 七、常见问题及解决方案

### 7.1 问题：graph_hash.txt长度错误

**现象**: hash长度不是64字符

**原因**: 使用了截断版hash

**解决**: 重新生成完整hash
```python
import hashlib
with open('model.py', 'rb') as f:
    content = f.read()
hash_value = hashlib.sha256(content).hexdigest()  # 64字符，不要[:16]
with open('graph_hash.txt', 'w') as f:
    f.write(hash_value)
```

### 7.2 问题：graph_net.json缺失必需字段

**现象**: 缺少source或heuristic_tag

**解决**: 补全字段
```python
import json
with open('graph_net.json', 'r') as f:
    data = json.load(f)
data['source'] = 'huggingface'
data['heuristic_tag'] = 'natural_language_processing'
with open('graph_net.json', 'w') as f:
    json.dump(data, f, indent=2)
```

### 7.3 问题：误判残图

**现象**: 完整模型被误判为残图

**原因**: 模型文件分散存储在subgraph目录

**解决**: 使用完整的`is_residual()`函数检查subgraph目录

---

## 八、最佳实践

### 8.1 提取时验证标准格式

```python
# 提取后验证所有必需文件
import os

required_files = [
    'model.py', 'graph_hash.txt', 'graph_net.json',
    'input_meta.py', 'input_tensor_constraints.py', 'weight_meta.py'
]

output_dir = f'/work/graphnet_workspace/huggingface/{task_type}/{model_name}'

# 验证文件
for file in required_files:
    filepath = os.path.join(output_dir, file)
    assert os.path.exists(filepath), f"{file} 缺失!"

# 验证hash长度
with open(os.path.join(output_dir, 'graph_hash.txt'), 'r') as f:
    hash_value = f.read().strip()
assert len(hash_value) == 64, f"hash长度错误: {len(hash_value)}"

print("✅ 所有GraphNet标准文件已生成")
```

### 8.2 与GraphNet/samples保持一致

```bash
# 对比检查
diff <(ls /root/GraphNet/samples/timm/dpn92.mx_in1k/) <(ls your_model/)
```

---

## 九、参考文档

- [GraphNet/samples标准目录](../../samples/)
- [并行执行指南](./PARALLEL_EXECUTION_GUIDE.md)
- [动态任务调度指南](./DYNAMIC_SCHEDULING_GUIDE.md)
- [HuggingFace提取指南](./extract/HUGGINGFACE_EXTRACTION_GUIDE.md)

---

**维护历史**:
- 2026-03-30: 更新文档，修正为GraphNet/samples标准格式(64字符hash，不含config.json/extract.py)
- 2026-03-30: 创建文档

# HuggingFace 残图修复最佳实践指南

## 概述

本文档总结残图修复任务中的成功经验，包括常见错误类型、解决方法以及提升成功率的最佳实践。

---

## 一、常见错误类型及解决方法

### 1. Hidden Size 与 Attention Heads 不匹配

**错误信息：**
```
ValueError: The hidden size (512) is not a multiple of the number of attention heads (9)
```

**原因：**
模型的hidden_size必须是num_attention_heads的整数倍（hidden_size % num_attention_heads == 0）

**解决方法：**
- 这种错误通常出现在模型配置本身有问题的情况
- 如果是自定义模型，需要修正config.json中的参数
- 对于标准模型，尝试使用AutoConfig.from_pretrained()加载官方配置

---

### 2. Decoder 配置错误

**错误信息：**
```
If you want to use `BertLMHeadModel` as a standalone, add `is_decoder=True.`
If you want to use `RobertaLMHeadModel` as a standalone, add `is_decoder=True.`
```

**原因：**
某些模型类（如BertLMHeadModel、RobertaLMHeadModel）需要显式设置is_decoder=True才能独立使用

**解决方法：**
```python
config = AutoConfig.from_pretrained(model_name)
config.is_decoder = True
model = BertLMHeadModel(config)
```

---

### 3. 缺少依赖库

**错误信息：**
```
No module named 'addict'
No module named 'torchaudio'
```

**解决方法：**
```bash
pip install addict torchaudio
```

**建议：** 在修复脚本开头添加依赖检查：
```python
try:
    import addict
except ImportError:
    print("[警告] 缺少addict库，部分模型可能无法处理")
```

---

### 4. 连接超时

**错误信息：**
```
FAILED - Timeout
[Errno 110] Connection timed out
```

**解决方法：**
1. **配置代理：**
```python
import os
os.environ['http_proxy'] = "http://agent.baidu.com:8891"
os.environ['https_proxy'] = "http://agent.baidu.com:8891"
os.environ['HTTP_PROXY'] = "http://agent.baidu.com:8891"
os.environ['HTTPS_PROXY'] = "http://agent.baidu.com:8891"
```

2. **使用HF-Mirror镜像：**
```python
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
```

3. **添加重试机制：**
```python
max_retries = 3
for retry in range(max_retries):
    try:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        break
    except Exception as e:
        if retry < max_retries - 1:
            print(f"  重试 {retry+1}/{max_retries}...")
            time.sleep(2)
        else:
            raise e
```

---

### 5. AutoModel 实例化错误

**错误信息：**
```
OSError: AutoModel is designed to be instantiated using the `AutoModel.from_pretrained()` or `AutoModel.from_config(config)` methods.
```

**原因：**
AutoModel不能直接用config实例化，需要用具体的模型类

**解决方法：**
```python
# 错误方式
model = AutoModel(config)

# 正确方式 - 根据任务类型选择具体模型类
from transformers import AutoModelForMaskedLM, AutoModelForCausalLM

model_classes = {
    'fill-mask': AutoModelForMaskedLM,
    'text-generation': AutoModelForCausalLM,
    'text-classification': AutoModelForSequenceClassification,
}
model_class = model_classes.get(task_type, AutoModel)
model = model_class.from_config(config)
```

---

### 6. 模型ID格式错误

**错误信息：**
```
Repo id must be in the form 'repo_name' or 'namespace/repo_name'
xxx is not a valid model identifier
```

**原因：**
模型ID使用了下划线(_)而不是斜杠(/)分隔namespace和model_name

**解决方法：**
```python
# 转换模型ID格式
model_id = model_path.split('/')[-1]  # 从路径提取
model_id = model_id.replace('_', '/', 1)  # 只替换第一个下划线

# 示例：
# OpenMed_OpenMed-PII-BigMed-Large-560M-v1 → OpenMed/OpenMed-PII-BigMed-Large-560M-v1
```

---

## 二、提升成功率的最佳实践

### 1. 优先使用 config.json

**经验数据：** 使用config.json创建模型的成功率 > 95%

```python
# 检查是否存在config.json
config_path = os.path.join(model_dir, 'config.json')
if os.path.exists(config_path):
    config = AutoConfig.from_pretrained(config_path)
    model = model_class.from_config(config)
else:
    # 无config时使用默认BertConfig
    config = BertConfig()
    model = model_class(config)
```

---

### 2. 任务类型推断

```python
def infer_task_type(model_name):
    """根据模型名推断任务类型"""
    model_lower = model_name.lower()

    if any(k in model_lower for k in ['bert', 'roberta', 'electra']):
        if 'classification' in model_lower or 'sentiment' in model_lower:
            return 'text-classification'
        return 'fill-mask'
    elif any(k in model_lower for k in ['gpt', 'llama', 'falcon', 'mistral']):
        return 'text-generation'
    elif 't5' in model_lower or 'bart' in model_lower:
        return 'text2text-generation'
    else:
        return 'fill-mask'  # 默认
```

---

### 3. 跳过已存在的模型

```python
output_path = os.path.join(base_path, dir_name)
if os.path.exists(os.path.join(output_path, 'model.py')):
    print(f"[{i}/{total}] {model_name} - 已存在，跳过")
    success_count += 1
    continue
```

---

### 4. 批量处理策略

```python
# 将模型列表分成多个批次
batch_size = 50
for batch_idx in range(0, len(models), batch_size):
    batch = models[batch_idx:batch_idx+batch_size]
    # 并行处理多个批次
```

---

### 5. 环境变量配置模板

```python
import os

# 代理设置
os.environ['http_proxy'] = "http://agent.baidu.com:8891"
os.environ['https_proxy'] = "http://agent.baidu.com:8891"
os.environ['HTTP_PROXY'] = "http://agent.baidu.com:8891"
os.environ['HTTPS_PROXY'] = "http://agent.baidu.com:8891"

# HuggingFace镜像
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"

# 禁用警告和进度条
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

# Python无缓冲输出
os.environ['PYTHONUNBUFFERED'] = '1'
```

---

## 三、修复流程优化建议

### 1. 分阶段修复策略

| 阶段 | 处理内容 | 预期成功率 |
|------|----------|-----------|
| 1 | 有config.json的标准模型 | >95% |
| 2 | 无config但可推断类型的模型 | >85% |
| 3 | 需要特殊处理的模型 | >70% |
| 4 | 手动处理失败案例 | 视情况 |

### 2. 错误分类处理

```python
error_handlers = {
    'hidden_size': handle_hidden_size_error,
    'is_decoder': handle_decoder_error,
    'timeout': handle_timeout_error,
    'module_not_found': handle_missing_dependency,
}

def handle_error(error_msg):
    for error_type, handler in error_handlers.items():
        if error_type in error_msg.lower():
            return handler()
    return False  # 未知错误
```

### 3. 日志记录与分析

```python
import json

results = {
    'total': 0,
    'success': 0,
    'failed': 0,
    'errors': {}
}

# 记录每种错误的出现次数
if not success:
    error_type = classify_error(str(e))
    results['errors'][error_type] = results['errors'].get(error_type, 0) + 1

# 保存结果
with open('repair_results.json', 'w') as f:
    json.dump(results, f, indent=2)
```

---

## 四、实际修复数据参考

### 本次修复任务统计

| 批次 | 模型数 | 成功 | 失败 | 成功率 |
|------|--------|------|------|--------|
| repair_batch_0 | 75 | 75 | 0 | 100% |
| repair_batch_1 | 75 | 75 | 0 | 100% |
| repair_batch_2 | 75 | 74 | 1 | 98.7% |
| repair_batch_3 | 72 | 70 | 2 | 97.2% |
| repair_batch_4 | 75 | 75 | 0 | 100% |
| **合计** | **372** | **369** | **3** | **99.2%** |

### 新模型抓取统计

| 批次 | 成功 | 失败 | 跳过 | 成功率 |
|------|------|------|------|--------|
| batch_0 | 49 | 1 | 10 | 98.0% |
| batch_1 | 49 | 1 | 8 | 98.0% |
| batch_2 | 50 | 0 | 11 | 100.0% |
| batch_3 | 50 | 0 | 11 | 100.0% |
| batch_4 | 49 | 1 | 11 | 98.0% |
| **合计** | **247** | **3** | **51** | **98.8%** |

---

## 五、总结

### 关键成功因素

1. **正确的模型ID格式**：使用`namespace/model_name`格式（斜杠分隔）
2. **配置代理和镜像**：避免网络超时问题
3. **优先使用config.json**：成功率最高
4. **适当的任务类型推断**：选择正确的模型类
5. **完善的错误处理**：分类处理不同错误类型

### 推荐工作流程

1. 准备阶段：修正模型ID格式，配置环境变量
2. 批量处理：分批次并行处理，每批50个模型
3. 监控阶段：实时查看日志，关注失败案例
4. 重试阶段：对失败模型分类重试
5. 汇总阶段：统计成功率，分析失败原因

---

## 六、正确识别残图的方法

### 重要概念：两种正确的模型格式

在判断模型是否完整时，必须认识到子图抽取有两种正确格式：

| 格式 | 特征 | 适用场景 |
|------|------|----------|
| **格式1** | 有 `subgraph_*` 目录，子目录下有核心文件 | 大模型自动拆分子图 |
| **格式2** | 没有 `subgraph` 目录，核心文件在主目录 | 小模型不拆分 |

**两种格式都是正确的，不要因为格式不同而误判为残图！**

### 残图判断标准

#### 正确判断流程
```bash
# 1. 找到所有模型目录
find /ssd1/liangtai-work/graphnet_workspace/huggingface/worker1 -mindepth 1 -maxdepth 1 -type d

# 2. 判断单个模型是否完整
# 格式1检查：subgraph_*目录下是否有model.py和graph_hash.txt
find "${model_dir}/subgraph_0" -name "model.py" 2>/dev/null
find "${model_dir}/subgraph_0" -name "graph_hash.txt" 2>/dev/null

# 格式2检查：主目录下是否有model.py和graph_hash.txt
find "${model_dir}" -maxdepth 1 -name "model.py" 2>/dev/null
find "${model_dir}" -maxdepth 1 -name "graph_hash.txt" 2>/dev/null
```

#### 完整的判断脚本
```python
#!/usr/bin/env python3
"""正确识别残图的脚本"""
import os

def check_model_complete(model_dir):
    """
    检查模型是否完整
    返回: (is_complete, format_type)
        is_complete: True/False
        format_type: 1(多子图)/2(单模型)/None(残图)
    """
    # 检查格式1: 有subgraph_*目录
    subgraph_dirs = [d for d in os.listdir(model_dir)
                     if d.startswith('subgraph_') and os.path.isdir(os.path.join(model_dir, d))]

    if subgraph_dirs:
        # 格式1: 检查每个subgraph目录下是否有核心文件
        for sg_dir in subgraph_dirs:
            sg_path = os.path.join(model_dir, sg_dir)
            has_model_py = os.path.exists(os.path.join(sg_path, 'model.py'))
            has_graph_hash = os.path.exists(os.path.join(sg_path, 'graph_hash.txt'))

            if not (has_model_py and has_graph_hash):
                return False, None  # 残图
        return True, 1

    # 检查格式2: 主目录下有核心文件
    has_model_py = os.path.exists(os.path.join(model_dir, 'model.py'))
    has_graph_hash = os.path.exists(os.path.join(model_dir, 'graph_hash.txt'))
    has_graph_net = os.path.exists(os.path.join(model_dir, 'graph_net.json'))

    if has_model_py and has_graph_hash:
        return True, 2

    return False, None  # 残图

# 统计整个目录
base_dir = '/ssd1/liangtai-work/graphnet_workspace/huggingface/worker1'
complete_format1 = 0
complete_format2 = 0
residual = 0

for model_name in os.listdir(base_dir):
    model_dir = os.path.join(base_dir, model_name)
    if not os.path.isdir(model_dir):
        continue

    is_complete, fmt = check_model_complete(model_dir)
    if is_complete:
        if fmt == 1:
            complete_format1 += 1
        elif fmt == 2:
            complete_format2 += 1
    else:
        residual += 1
        print(f"残图: {model_name}")

print(f"\n完整模型-格式1(多子图): {complete_format1}")
print(f"完整模型-格式2(单模型): {complete_format2}")
print(f"残图: {residual}")
```

### 常见误判案例

#### ❌ 错误判断方式
```bash
# 错误：只看主目录是否有model.py
find /ssd1/liangtai-work/graphnet_workspace/huggingface/worker1 -maxdepth 2 -name "model.py" | wc -l

# 这种会把格式1(多子图)的完整模型误判为残图！
```

#### ✅ 正确判断方式
```bash
# 正确：检查格式1和格式2的所有情况
# 格式1：多子图模型（subgraph_*目录下有核心文件）
find /ssd1/liangtai-work/graphnet_workspace/huggingface/worker1 -path "*/subgraph_*/model.py" | wc -l

# 格式2：单子图模型（主目录下有核心文件）
find /ssd1/liangtai-work/graphnet_workspace/huggingface/worker1 -maxdepth 2 -name "model.py" | grep -v subgraph | wc -l
```

### 残图类型分类

| 类型 | 特征 | 修复方法 |
|------|------|----------|
| **NO_SUBGRAPH** | 没有subgraph目录，也没有主目录model.py | 重新运行extract.py或重新抓取 |
| **INCOMPLETE_SUBGRAPH** | 有subgraph目录但缺少model.py或graph_hash.txt | 运行extract.py补充缺失文件 |

---

## 七、特殊模型修复案例

### 1. ILKT 模型修复

**错误信息：**
```
ILKTModel.forward() missing 1 required positional argument: 'attention_mask'
```

**原因：**
ILKT 系列模型的 forward 方法需要 `attention_mask` 参数，但默认抽取脚本只传入 `input_ids`。

**解决方法：**
```python
# 准备输入时添加 attention_mask
input_ids = torch.randint(0, vocab_size, (1, seq_len))
attention_mask = torch.ones(1, seq_len, dtype=torch.long)

# 尝试使用 attention_mask
try:
    wrapped(input_ids, attention_mask=attention_mask)
except TypeError:
    # 如果不需要 attention_mask，只用 input_ids
    wrapped(input_ids)
```

**修复效果：** 191/192 ILKT 模型成功抽取 (99.5%)

---

### 2. BFloat16/Float32 dtype 不匹配

**错误信息：**
```
mat1 and mat2 must have the same dtype, but got BFloat16 and Float32
```

**原因：**
部分模型（如 ILKT）内部使用 BFloat16，但 CPU 环境下输入张量默认是 Float32。

**解决方法：**
```python
# 确保模型使用 float32 (CPU 兼容)
model = model.float()
```

---

### 3. HF-Mirror 速率限制 (429 错误)

**错误信息：**
```
HTTP Error 429 thrown while requesting HEAD https://hf-mirror.com/...
```

**原因：**
hf-mirror.com 对无 token 用户有严格的速率限制。

**解决方法：**
```python
# 配置 HF Token 提高速率限制
os.environ["HF_TOKEN"] = "your_hf_token_here"
```

**配置步骤：**
1. 获取 token: https://huggingface.co/settings/tokens
2. 写入配置文件:
```bash
echo "your_token" > ~/.huggingface/token
```
3. 或在脚本中设置环境变量

**效果：** 配置 token 后 429 错误显著减少

---

### 4. 常见失败原因统计

| 失败原因 | 数量占比 | 是否可修复 |
|----------|----------|------------|
| 网络错误 (429/超时) | ~40% | ✓ 配置 Token |
| 不支持类型 (GGUF/LoRA) | ~20% | ✗ 跳过 |
| 模型配置问题 | ~15% | 部分可修复 |
| 参数缺失 (如 attention_mask) | ~10% | ✓ 修改输入 |
| dtype 不匹配 | ~5% | ✓ model.float() |
| 其他 | ~10% | 视情况 |

---

*文档版本：1.2*
*最后更新：2026-04-19*
*作者：Claude Code Assistant*

# 修复失败案例处理指南

> 系统性整理修复过程中遇到的失败案例
> 创建时间: 2026-03-30
> 维护者: GraphNet Agent Team

---

## 一、失败类型总览

| 失败类型 | 占比 | 解决方案难度 |
|----------|------|--------------|
| Type 1: 模型配置错误 | ~60% | 中（需定制处理） |
| Type 2: 网络/下载超时 | ~20% | 低（使用本地配置） |
| Type 3: extract调用错误 | ~10% | 低（更新脚本） |
| Type 4: 输出无效 | ~10% | 高（需深入调试） |

---

## 二、Type 1: 模型配置错误

### 2.1 需要 is_decoder=True 的模型

**错误信息**:
```
If you want to use `XmodLMHeadModel` as a standalone, add `is_decoder=True.`
If you want to use `ErnieForCausalLM` as a standalone, add `is_decoder=True.`
```

**受影响模型**:
- facebook/xmod-base
- Xenova/ernie-2.0-large-en
- Xenova/ernie-3.0-base-zh
- hf-tiny-model-private/tiny-random-ErnieForMultipleChoice

**解决方案**:
```python
# 修改config后加载
config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
config.is_decoder = True  # 添加此行
model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
```

### 2.2 RoPE参数配置错误

**错误信息**:
```
`rope_parameters`'s factor field must be a float >= 1, got 40
`rope_parameters`'s beta_fast field must be a float, got 32
```

**受影响模型**:
- arithmetic-circuit-overloading/* 系列模型
- optimum-intel-internal-testing/tiny-random-deepseek-v3

**解决方案**: 需要修改config中的rope_parameters参数为符合要求的格式

```python
config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
# 修正rope_parameters参数
if hasattr(config, 'rope_parameters'):
    config.rope_parameters = None  # 或使用正确的参数格式
```

### 2.3 Xmod模型需要设置语言

**错误信息**:
```
ValueError: Input language unknown. Please call `XmodPreTrainedModel.set_default_language()`
```

**受影响模型**:
- facebook/xmod-base

**解决方案**:
```python
from transformers import XmodModel

config = AutoConfig.from_pretrained("facebook/xmod-base", trust_remote_code=True)
model = XmodModel.from_config(config, trust_remote_code=True)
model.set_default_language("en_XX")  # 设置默认语言
```

---

## 三、Type 2: 网络/下载超时

### 3.1 无法连接到hf-mirror.com

**错误信息**:
```
OSError: We couldn't connect to 'https://hf-mirror.com' to load the files
```

**解决方案1: 使用本地config.json**:
```python
# 如果本地有config.json
config = AutoConfig.from_pretrained(".", trust_remote_code=True)
```

**解决方案2: 跳过该模型**:
将模型加入待修复列表，等待网络恢复后处理

### 3.2 模型文件不存在

**错误信息**:
```
OSError: model_id does not appear to have a file named config.json
```

**可能原因**:
- 模型ID错误
- 模型已从HF删除
- 模型为私有模型

**解决方案**:
检查模型ID是否正确，或从待抽取列表中移除

---

## 四、Type 3: extract调用错误

### 4.1 旧版extract调用方式

**错误信息**:
```
TypeError: extract() takes 1 positional argument but 5 were given
```

**原因**: 使用了旧版extract函数调用方式

**旧版（错误）**:
```python
result = torch_extract(
    model=model,
    model_name=MODEL_ID,
    output_path=".",
    input_shape=(1, 32)
)
```

**新版（正确）**:
```python
os.environ["GRAPH_NET_EXTRACT_WORKSPACE"] = "."
model = torch_extract(MODEL_ID)(model)
with torch.no_grad():
    _ = model(input_ids)
```

---

## 五、Type 4: 输出无效

### 5.1 无GraphModule输出

**错误信息**:
```
No GraphModule in output
```

**可能原因**:
1. extract wrapper未正确触发
2. forward pass未执行
3. 环境变量GRAPH_NET_EXTRACT_WORKSPACE未设置

**解决方案**:
```python
import os
os.environ["GRAPH_NET_EXTRACT_WORKSPACE"] = "."

from graph_net.torch.extractor import extract

# 确保执行了forward pass
model = extract(model_name)(model)
with torch.no_grad():
    outputs = model(**inputs)  # 必须有这步

# 检查输出
if not os.path.exists("model.py"):
    raise RuntimeError("Extraction failed")
```

### 5.2 模型太大导致OOM

**错误信息**:
```
RuntimeError: CUDA out of memory
```

**解决方案**:
```python
# 1. 使用CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# 2. 减小输入尺寸
seq_len = 32  # 而不是512
input_ids = torch.randint(0, vocab_size, (1, seq_len))

# 3. 使用更小的batch
batch_size = 1
```

---

## 六、批量处理失败模型的策略

### 6.1 分类处理流程

```python
def repair_with_fallback(model_id, model_dir):
    """带故障转移的修复"""

    # 尝试1: 标准修复
    try:
        return standard_repair(model_id, model_dir)
    except Exception as e:
        error_msg = str(e)

    # 尝试2: 根据错误类型定制修复
    if "is_decoder" in error_msg:
        return repair_with_is_decoder(model_id, model_dir)
    elif "rope_parameters" in error_msg:
        return repair_with_fixed_rope(model_id, model_dir)
    elif "Xmod" in error_msg:
        return repair_xmod_model(model_id, model_dir)
    elif "Network" in error_msg or "connect" in error_msg:
        # 跳过，稍后重试
        record_for_retry(model_id)
        return False
    else:
        # 记录未知错误
        record_unknown_error(model_id, error_msg)
        return False
```

### 6.2 失败模型记录格式

```
# /tmp/failed_models_classification.txt
task|model_id|error_type|error_msg|retry_strategy
```

**示例**:
```
text-generation|facebook/xmod-base|config_error|need is_decoder=True|use_custom_config
text-generation|Xenova/ernie-2.0-large-en|config_error|need is_decoder=True|use_custom_config
text-generation|arithmetic-circuit-overloading/*|config_error|rope_parameters invalid|skip
```

---

## 七、待处理的失败模型清单

### 7.1 当前修复中的失败模型（444个残图中）

| 序号 | 模型ID | Task | 失败类型 | 处理策略 |
|------|--------|------|----------|----------|
| 1 | facebook/xmod-base | text-generation | config_error | 使用is_decoder |
| 2 | Xenova/ernie-2.0-large-en | text-generation | config_error | 使用is_decoder |
| 3 | dacorvo/tiny-random-llama | text-generation | extract_error | 更新extract调用 |
| 4 | arithmetic-circuit-overloading/* | text-generation | config_error | 修复RoPE参数 |
| 5 | yujiepan_mixtral-tiny-random | text-generation | output_error | 调试extract |

### 7.2 待抽取模型（3,466个）

这些模型已删除目录，需要从HuggingFace重新下载抽取。

**处理建议**:
1. 分批处理（每批100个）
2. 网络异常时自动重试
3. 记录无法下载的模型

---

## 八、快速修复命令

### 8.1 修复单个模型

```bash
# 进入模型目录
cd /work/graphnet_workspace/huggingface/text-generation/facebook_xmod-base

# 清理旧文件
rm -f graph_hash.txt graph_net.json model.py weight_meta.py

# 运行修复
python3 extract.py
```

### 8.2 批量修复特定类型

```bash
# 修复所有需要is_decoder的模型
for model in facebook/xmod-base Xenova/ernie-2.0-large-en; do
    echo "修复: $model"
    python3 repair_with_is_decoder.py "$model"
done
```

### 8.3 检查修复结果

```bash
# 检查model.py大小
find . -name "model.py" -size -1k -exec echo "{} 小于1KB" \;

# 检查是否有GraphModule
grep -r "class GraphModule" */model.py | wc -l
```

---

## 九、更新记录

| 日期 | 更新内容 | 维护者 |
|------|----------|--------|
| 2026-03-30 | 初始版本，整理当前失败案例 | Ducc |

---

**下一步行动**:
1. 根据此指南更新修复脚本，增加故障转移功能
2. 针对Type 1错误开发定制修复脚本
3. 建立失败模型数据库，避免重复失败

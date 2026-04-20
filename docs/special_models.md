
## 超大 MoE 模型抽取案例 (2026-04-09)

### 模型信息
- **模型ID**: `adaarsh28/Nihongo-L1-TTS-2026`
- **架构**: DeepseekV3ForCausalLM
- **配置**: 61层, 384专家, 7168隐藏维度, ~600B参数

### 处理步骤

1. **下载远程代码文件**
```python
from huggingface_hub import hf_hub_download
files = ["configuration_deepseek.py", "modeling_deepseek.py"]
for f in files:
    hf_hub_download(repo_id=model_id, filename=f, local_dir=local_dir)
```

2. **移除量化配置**
```python
if hasattr(config, 'quantization_config'):
    delattr(config, 'quantization_config')
```

3. **尝试抽取**
- float32: 需要 ~2400GB，OOM
- 结论: 内存不足

### 后续尝试方案

```python
# 方案1: float16 (节省50%)
model = AutoModelForCausalLM.from_config(
    config, torch_dtype=torch.float16, trust_remote_code=True
)

# 方案2: float8 (节省75%)
model = AutoModelForCausalLM.from_config(
    config, torch_dtype=torch.float8_e4m3fn, trust_remote_code=True
)

# 方案3: meta device (不分配内存)
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
```

### 状态
- 当前标记: SKIP (内存不足)
- 待后续尝试: float16/float8 方案

---

## ILKT 模型抽取案例 (2026-04-19)

### 模型信息
- **模型ID**: `ILKT/2024-06-xx_xx-xx-xx` 系列 (共192个)
- **架构**: ILKTModel (基于 BERT)
- **特点**: 需要 attention_mask 参数，内部使用 BFloat16

### 遇到的问题

**问题1: 缺少 attention_mask 参数**
```
ILKTModel.forward() missing 1 required positional argument: 'attention_mask'
```

**问题2: dtype 不匹配**
```
mat1 and mat2 must have the same dtype, but got BFloat16 and Float32
```

### 解决方案

```python
# 1. 添加 attention_mask
input_ids = torch.randint(0, vocab_size, (1, seq_len))
attention_mask = torch.ones(1, seq_len, dtype=torch.long)

# 2. 确保 float32 兼容
model = model.float()

# 3. 尝试使用 attention_mask
try:
    wrapped(input_ids, attention_mask=attention_mask)
except TypeError:
    wrapped(input_ids)  # 回退到只用 input_ids
```

### 抽取结果

| 项目 | 数值 |
|------|------|
| 模型总数 | 192 |
| 成功抽取 | 191 |
| 成功率 | 99.5% |

### 唯一失败案例

- `ILKT/2024-06-20_12-31-55` - 需要 `eval_utils` 依赖包

---

## Worker1 改进 (2026-04-19)

### 借鉴 Worker2 的优化

Worker1 脚本已借鉴 Worker2 的处理思路，新增以下功能：

### 1. 模型类型自动检测

```python
def detect_model_type(config, model_id):
    """根据config和model_id检测模型类型"""
    model_type = getattr(config, 'model_type', '').lower()
    architectures = getattr(config, 'architectures', [])
    arch_name = architectures[0].lower() if architectures else ''
    model_id_lower = model_id.lower()

    # 支持: audio, vision, detr, sam, multimodal, text
    ...
```

### 2. 根据模型类型准备输入

```python
def prepare_inputs(model_type, config):
    """根据模型类型准备输入"""
    if model_type == 'audio':
        # Whisper, Wav2Vec 等
        return {'input_features': ...}
    elif model_type == 'vision':
        # ViT, CLIP 等
        return {'pixel_values': ...}
    elif model_type == 'detr':
        # DETR 系列
        return {'pixel_values': ..., 'pixel_mask': ...}
    ...
```

### 3. 支持的模型类型

| 类型 | 示例模型 | 输入 |
|------|----------|------|
| audio | Whisper, Wav2Vec | input_features/input_values |
| vision | ViT, CLIP, Swin | pixel_values |
| detr | DETR, RT-DETR | pixel_values, pixel_mask |
| sam | SAM, SAM2 | pixel_values |
| multimodal | LLaVA, BLIP | input_ids, pixel_values |
| text | BERT, GPT | input_ids, attention_mask |

### 4. 错误重试机制

```python
# 针对特定错误自动重试
if 'decoder_input_ids' in error_msg:
    inputs['decoder_input_ids'] = inputs['input_ids'].clone()
    wrapped(**inputs)
```


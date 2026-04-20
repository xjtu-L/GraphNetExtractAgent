# 已验证支持的模型类型

本文档记录已成功抽取的模型类型及其抽取方法。

## 音频模型

### Wav2Vec2
- **model_type**: `wav2vec2`
- **架构**: `Wav2Vec2ForSequenceClassification`, `Wav2Vec2ForCTC`, `Wav2Vec2Model`
- **测试模型**: `ar5entum/wav2vec2-base-language-classification-en-hi-ta`
- **抽取日期**: 2026-04-19

**抽取方法：**
```python
from transformers import AutoConfig, AutoModel
import torch

# 1. 加载配置
config = AutoConfig.for_model('wav2vec2', **config_dict)

# 2. 创建模型
model = AutoModel.from_config(config, trust_remote_code=True)

# 3. 构造输入 - 音频波形 (batch, sequence_length)
inputs = {"input_values": torch.randn(1, 16000)}  # 1秒音频

# 4. 抽取
wrapped = extract(name=model_name, dynamic=False)(model).eval()
wrapped(**inputs)
```

**关键点：**
- 输入是原始音频波形，不是频谱
- 序列长度 16000 对应 1 秒音频（采样率 16kHz）
- 使用 `AutoModel` 而非 `AutoModelForCTC` 等特定类

---

### Whisper
- **model_type**: `whisper`
- **架构**: `WhisperForConditionalGeneration`
- **测试模型**: `aadel4/kid-whisper-small-en-myst`
- **抽取日期**: 2026-04-19

**抽取方法：**
```python
from transformers import AutoConfig, AutoModel
import torch

# 1. 加载配置
config = AutoConfig.for_model('whisper', **config_dict)

# 2. 创建模型
model = AutoModel.from_config(config, trust_remote_code=True)

# 3. 构造输入 - log-mel spectrogram + decoder tokens
num_mel_bins = getattr(config, 'num_mel_bins', 80)  # 通常 80
vocab_size = getattr(config, 'vocab_size', 51864)

inputs = {
    "input_features": torch.randn(1, num_mel_bins, 3000),  # (batch, mel_bins, time)
    "decoder_input_ids": torch.randint(0, vocab_size, (1, 20))  # decoder 输入
}

# 4. 抽取
wrapped = extract(name=model_name, dynamic=False)(model).eval()
wrapped(**inputs)
```

**关键点：**
- Whisper 是 encoder-decoder 架构，需要两个输入
- `input_features` 是 log-mel 频谱，形状 (batch, 80, 3000)
- `decoder_input_ids` 是解码器输入 token，长度可变
- 3000 是时间维度，对应 30 秒音频

---

## 视频模型

### ViViT (Video Vision Transformer)
- **model_type**: `vivit`
- **架构**: `VivitForVideoClassification`, `VivitModel`
- **测试模型**: `Temo27Anas/vivit-b-16x2-kinetics400-ft-66989`
- **抽取日期**: 2026-04-19

**抽取方法：**
```python
from transformers import AutoConfig, AutoModel
import torch

# 1. 加载配置
config = AutoConfig.for_model('vivit', **config_dict)

# 2. 创建模型
model = AutoModel.from_config(config, trust_remote_code=True)

# 3. 构造输入 - 视频帧序列
num_frames = getattr(config, 'num_frames', 32)  # 默认 32 帧
image_size = getattr(config, 'image_size', 224)  # 默认 224x224

inputs = {
    "pixel_values": torch.randn(1, num_frames, 3, image_size, image_size)
    # 形状: (batch, num_frames, channels, height, width)
}

# 4. 抽取
wrapped = extract(name=model_name, dynamic=False)(model).eval()
wrapped(**inputs)
```

**关键点：**
- 输入是 5D 张量：(batch, frames, channels, height, width)
- `num_frames` 从配置读取，通常是 32
- `image_size` 通常是 224

---

## 视觉模型

### ViT (Vision Transformer)
- **model_type**: `vit`
- **架构**: `ViTForImageClassification`, `ViTModel`

**抽取方法：**
```python
# 输入格式
inputs = {"pixel_values": torch.randn(1, 3, 224, 224)}
```

### CLIP
- **model_type**: `clip`
- **架构**: `CLIPModel`

**抽取方法：**
```python
# 输入格式 - 需要图像和文本
inputs = {
    "pixel_values": torch.randn(1, 3, 224, 224),
    "input_ids": torch.randint(0, 49408, (1, 77))
}
```

---

## 语言模型

### BERT 系列
- **model_type**: `bert`, `roberta`, `distilbert`, `albert`, `electra`, `xlm-roberta`

**抽取方法：**
```python
# 尝试加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
inputs = tokenizer("sample text", return_tensors="pt", padding="max_length", max_length=128)

# 或手动构造
inputs = {
    "input_ids": torch.randint(0, vocab_size, (1, 32)),
    "attention_mask": torch.ones(1, 32)
}
```

### GPT/LLaMA 系列 (CausalLM)
- **model_type**: `gpt2`, `gpt_neox`, `llama`, `llama3`, `qwen2`, `qwen3`, `mistral`, `opt`

**抽取方法：**
```python
from transformers import AutoModelForCausalLM

# 使用 CausalLM 类
model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

# 输入格式同 BERT
inputs = {"input_ids": torch.randint(0, vocab_size, (1, 32))}
```

### T5/BART 系列 (Seq2Seq)
- **model_type**: `t5`, `bart`, `mbart`, `pegasus`, `marian`

**抽取方法：**
```python
# encoder-decoder 架构，需要两个输入
inputs = {
    "input_ids": torch.randint(0, vocab_size, (1, 32)),
    "decoder_input_ids": torch.randint(0, vocab_size, (1, 16))
}
```

---

## 添加新模型类型的流程

1. **检查 config.json**
   - 查看 `model_type` 字段
   - 查看 `architectures` 字段

2. **确定输入格式**
   - 查阅 HuggingFace 文档
   - 检查 config 中的特殊字段（如 `num_mel_bins`, `num_frames`）

3. **测试抽取**
   ```python
   config = AutoConfig.for_model(model_type, **config_dict)
   model = AutoModel.from_config(config, trust_remote_code=True)
   # 构造输入...
   wrapped = extract(...)(model)
   wrapped(**inputs)
   ```

4. **记录到本文档**
   - model_type
   - 架构
   - 测试模型
   - 抽取方法代码

---

## 不支持的模型类型

| 类型 | 原因 |
|------|------|
| 超大模型 (>5B) | 内存限制，OOM |
| GLM4 MoE | 183B 参数，超大 |
| Mixtral, DeepseekV3 | MoE 架构，超大 |
| 无 model_type | 非 transformers 模型 |
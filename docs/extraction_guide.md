# GraphNet 模型抽取指南

## 概述

本文档记录了从 HuggingFace 模型抽取计算图的最佳实践和经验总结。

## 支持的模型类型

### 1. 文本模型 (Text Models)
**示例**: BERT, GPT, T5, RoBERTa, LLaMA

**输入格式**:
```python
{
    'input_ids': torch.randint(0, vocab_size, (1, 128)),
    'attention_mask': torch.ones(1, 128, dtype=torch.long)
}
```

**成功率**: ~70-80%

### 2. 语音模型 - ASR (Automatic Speech Recognition)
**示例**: Whisper, Wav2Vec2, HuBERT, WavLM

**输入格式**:
- Whisper: `input_features` (mel频谱图)
- Wav2Vec2: `input_values` (原始音频波形)

```python
# Whisper
{'input_features': torch.randn(1, num_mel_bins, 3000)}

# Wav2Vec2
{
    'input_values': torch.randn(1, 16000),
    'attention_mask': torch.ones(1, 16000)
}
```

**成功率**: ~60-70%

### 3. 语音模型 - TTS (Text-to-Speech)
**示例**: SpeechT5, VITS, Bark

**输入格式**:
```python
{
    'input_ids': torch.randint(0, vocab_size, (1, 50)),
    'attention_mask': torch.ones(1, 50),
    'decoder_input_ids': torch.full((1, 1), decoder_start_token_id)
}
```

**注意**: SpeechT5 需要 decoder_input_ids，部分变体可能失败

**成功率**: ~30-40%

### 4. 视觉模型 (Vision Models)
**示例**: ViT, CLIP, BEiT, DeiT, Swin Transformer

**输入格式**:
```python
{'pixel_values': torch.randn(1, 3, image_size, image_size)}
```

**成功率**: ~60-70%

### 5. 目标检测模型 (Object Detection)
**示例**: DETR, RT-DETR

**输入格式**:
```python
{
    'pixel_values': torch.randn(1, 3, image_size, image_size),
    'pixel_mask': torch.ones(1, image_size, image_size)
}
```

**注意**:
- 标准 DETR 使用 800x800
- RT-DETR 使用 640x640

**成功率**: ~50-60%

### 6. 分割模型 (Segmentation)
**示例**: SAM (Segment Anything Model)

**输入格式**:
```python
{'pixel_values': torch.randn(1, 3, 1024, 1024)}
```

**注意**: SAM2 可能需要额外的 inference_session 参数

**成功率**: ~40-50%

### 7. 多模态模型 (Multimodal)
**示例**: LLaVA, BLIP

**输入格式**:
```python
{
    'input_ids': torch.randint(0, vocab_size, (1, 128)),
    'attention_mask': torch.ones(1, 128),
    'pixel_values': torch.randn(1, 3, 224, 224)
}
```

**成功率**: ~50-60%

## 抽取失败常见原因

### 1. 模型格式问题
- **GGUF/量化模型**: llama.cpp 格式，非标准 HF 格式
- **LoRA 适配器**: 不是完整模型，只是权重差分
- **ONNX 格式**: 需要专用运行时

**解决方案**: 跳过这些模型类型

### 2. 输入格式不匹配
- 语音模型使用文本输入
- 视觉模型缺少 pixel_values
- TTS 模型缺少 decoder_input_ids

**解决方案**: 使用通用抽取脚本自动检测模型类型

### 3. 内存不足 (OOM)
- 大模型 (>7B) 在 40GB GPU 上可能 OOM
- 多进程并行时显存竞争

**解决方案**:
- 减少每 GPU 进程数
- 使用 CPU 模式抽取大模型
- 启用 `torch.cuda.empty_cache()`

### 4. 自定义代码
- 部分模型需要 `trust_remote_code=True`
- 自定义架构不在 transformers 库中

**解决方案**: 设置 `trust_remote_code=True` 并确保网络连接

## 抽取输出要求

### 必须生成的文件

抽取成功后，每个模型目录必须包含以下文件：

| 文件 | 说明 | 必须 |
|------|------|------|
| `model.py` | 计算图主文件 | ✅ |
| `graph_hash.txt` | model.py 的 SHA256 哈希值 | ✅ |
| `graph_net.json` | 模型配置信息 | ✅ |
| `input_meta.py` | 输入元数据 | ✅ |
| `input_tensor_constraints.py` | 输入约束 | ✅ |
| `weight_meta.py` | 权重元数据 | ✅ |
| `subgraph_*/` | 子图目录（大模型才有） | 可选 |

### graph_hash.txt 生成

**重要**: 必须在抽取成功后生成 graph_hash.txt：

```python
import hashlib

def generate_graph_hash(model_py_path):
    """生成 model.py 的 SHA256 哈希值"""
    with open(model_py_path, 'r') as f:
        hash_val = hashlib.sha256(f.read().encode()).hexdigest()
    hash_path = model_py_path.replace("model.py", "graph_hash.txt")
    with open(hash_path, 'w') as f:
        f.write(hash_val)
```

**注意**: 如果有子图（subgraph_* 目录），每个子图的 model.py 也需要生成对应的 graph_hash.txt。

### 完整性检查

```bash
# 检查所有模型是否有 graph_hash.txt
find /path/to/extracted -name "model.py" | while read f; do
    dir=$(dirname "$f")
    if [ ! -f "$dir/graph_hash.txt" ]; then
        echo "Missing: $dir/graph_hash.txt"
    fi
done
```

## 优化经验总结

### 版本迭代记录

#### V1 - 基础版本
- 仅支持文本模型
- 使用简单 input_ids 输入
- 成功率: ~16%

#### V2 - 通用版本
- 添加语音/视觉模型支持
- 自动检测模型类型
- 成功率: ~20%

#### V3 - 增强版本
- 优化 TTS/DETR/SAM 等特殊模型
- 添加 retry 机制
- 成功率: ~25%

#### Final - 终极版本
- 整合所有优化
- 支持所有模型类型
- 预期成功率: ~30%

### 关键优化点

1. **自动类型检测**
   - 根据 model_id 和 config 自动判断模型类型
   - 避免手动分类

2. **动态输入生成**
   - 根据模型类型生成正确的输入格式
   - 处理不同的图像尺寸、序列长度

3. **错误重试**
   - 检测缺失参数并自动补充
   - 如: decoder_input_ids, pixel_mask

4. **CPU vs GPU**
   - CPU 模式更稳定，适合大模型
   - GPU 模式更快，适合小模型

## 使用建议

### 对于大规模抽取

1. **预处理过滤**
   ```bash
   # 过滤掉明显无法抽取的模型
   grep -v -iE "gguf|lora|onnx|awq|gptq" model_list.txt > filtered_list.txt
   ```

2. **分批处理**
   - 每批 1000-2000 个模型
   - 避免单批任务过多导致内存问题

3. **监控和自动补全**
   - 定期检查进程数
   - 进程崩溃时自动重启

### 推荐脚本

**最终通用脚本**:
```bash
python worker2_multi_gpu_extract_final.py \
    --list model_list.txt \
    --gpus 1,2,3,4,5 \
    --procs-per-gpu 3 \
    --task feature-extraction
```

## 参考文档

- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [GraphNet 项目](https://github.com/PaddlePaddle/GraphNet)
- [PyTorch FX](https://pytorch.org/docs/stable/fx.html)

## 更新日志

- **2024-04-19**: 创建文档，整合 V1/V2/V3 优化经验
- 支持模型类型: 文本、ASR、TTS、视觉、检测、分割、多模态

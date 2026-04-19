# 列表抽取脚本 V5

## 功能特点

- **支持自定义模型**: `trust_remote_code=True` 支持 HuggingFace 上的自定义模型架构
- **智能输入生成**: 优先尝试加载 tokenizer 生成正确格式的输入
- **多模型类支持**: 自动识别并加载正确的模型类
  - AutoModelForCausalLM
  - AutoModelForSequenceClassification
  - AutoModelForTokenClassification
  - AutoModelForMaskedLM
  - AutoModelForQuestionAnswering
  - AutoModel (默认)
- **音频模型支持**: 支持 Whisper 和 Wav2Vec2 模型
- **视觉模型支持**: 自动识别 ViT/CLIP 等视觉模型
- **参数量限制**: 可配置，默认 5B

## 用法

```bash
# 基本用法
python extract_from_list_v5.py --list models.txt --device cuda:0

# 指定范围
python extract_from_list_v5.py --list models.txt --start 0 --end 1000 --device cuda:0

# 指定输出目录和进程ID
python extract_from_list_v5.py \n    --list models.txt \n    --start 0 --end 1000 \n    --output /path/to/output \n    --device cuda:0 \n    --agent-id my_agent
```

## 参数说明

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `--list` | 是 | - | 模型列表文件，每行一个模型ID |
| `--start` | 否 | 0 | 起始索引 |
| `--end` | 否 | -1 (全部) | 结束索引 |
| `--output` | 否 | `/ssd1/liangtai-work/graphnet_workspace/huggingface/worker2` | 输出目录 |
| `--device` | 否 | `cuda:0` | GPU 设备 |
| `--agent-id` | 否 | `list_extract` | 进程标识符 |

## 模型列表格式

每行一个 HuggingFace 模型 ID：

```
bert-base-uncased
gpt2
facebook/wav2vec2-base-960h
openai/whisper-tiny
```

## 跳过规则

### 跳过的模型类型
- bark, musicgen, sam, encodec
- sd, stable-diffusion
- speecht5, parler_tts, gguf

### 跳过的模型架构
- DeepseekV3ForCausalLM
- MixtralForCausalLM
- Step1MoEForCausalLM

### 跳过的模型名称关键词
- -gguf, _gguf, -GGUF, _GGUF
- -mlx, _mlx
- -awq, _awq
- -gptq, _gptq
- -onnx, _onnx

## 输入生成策略

1. **Whisper 模型**: `input_features` (batch, 80, 3000) + `decoder_input_ids`
2. **Wav2Vec2 模型**: `input_values` (batch, 16000)
3. **视觉模型**: `pixel_values` (batch, 3, 224, 224)
4. **其他模型**: 尝试加载 tokenizer 生成输入，失败则手动构造

## 与 V3 版本的区别

| 特性 | V3 | V5 |
|------|----|----|
| trust_remote_code | True | True |
| Tokenizer 加载 | 总是加载 | 尝试加载，失败则跳过 |
| 模型类选择 | 按任务类型 | 按架构自动识别 |
| 音频模型 | 不支持 | 支持 |
| 参数限制 | 无 | 5B |

## 示例：并行抽取

```bash
# 在多个 GPU 上并行抽取
models_per_gpu=$((total_models / 3))

nohup python extract_from_list_v5.py --list models.txt \n    --start 0 --end $models_per_gpu --device cuda:2 --agent-id g2_0 &

nohup python extract_from_list_v5.py --list models.txt \n    --start $models_per_gpu --end $((models_per_gpu * 2)) --device cuda:4 --agent-id g4_0 &

nohup python extract_from_list_v5.py --list models.txt \n    --start $((models_per_gpu * 2)) --end $total_models --device cuda:6 --agent-id g6_0 &
```

## 注意事项

1. 确保设置了正确的代理环境变量
2. 大模型（>5B 参数）会被跳过以避免 OOM
3. 每个模型抽取后会自动清理 GPU 内存
4. 已存在的模型会跳过，不会重复抽取
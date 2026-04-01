# Worker1 模型抽取指南 (全类型支持)

## 任务概述

Worker1 负责从模型列表中提取**所有类型**的HuggingFace模型的计算图，不限于text-classification。

### 支持的模型类型

| 类型 | 关键词示例 | 模型类 |
|------|-----------|--------|
| **text-generation** | gpt, llama, mistral, qwen, yi, falcon, bloom | AutoModelForCausalLM |
| **text-classification** | sentiment, classification, emotion, nli | AutoModelForSequenceClassification |
| **token-classification** | ner, token, pos-tag | AutoModelForTokenClassification |
| **question-answering** | qa, squad, question-answering | AutoModelForQuestionAnswering |
| **fill-mask** | bert, roberta, distilbert | AutoModelForMaskedLM |
| **text2text-generation** | t5, bart, pegasus, marian, translation | AutoModelForSeq2SeqLM |
| **automatic-speech-recognition** | whisper, wav2vec, speech | AutoModelForSpeechSeq2Seq |
| **audio-classification** | audio, wav, sound | AutoModelForAudioClassification |
| **image-classification** | vit, resnet, image, vision | AutoModelForImageClassification |
| **feature-extraction** | 其他类型 | AutoModel |

## 工作环境

### 目录结构
- 模型列表: `/root/GraphNetExtractAgent/data/model_lists/our_models_300k.txt`
- 抽取脚本: `/root/extract_from_model_list.py`
- 输出目录: `/root/graphnet_workspace/huggingface/worker1/`
- 批次模板: `/root/graphnet_workspace/huggingface/worker1/batch_extraction_template.py`
- 监控报告: `/root/graphnet_workspace/huggingface/worker1/subagent_report.json`

### 输出子目录
按任务类型自动分类到子目录：
```
worker1/
├── text-classification/
├── text-generation/
├── fill-mask/
├── feature-extraction/
├── token-classification/
├── question-answering/
├── automatic-speech-recognition/
├── text2text-generation/
└── image-classification/
```

### 责任范围
- Worker1: 行 0-74860 (共74,861个模型)
- Worker2: 行 74861-149720 (后半部分)

## 启动抽取进程

### 并行抽取策略 (推荐)

使用批次模板启动多进程并行抽取：

```bash
cd /root/graphnet_workspace/huggingface/worker1

# 启动36个批次，每批2135个模型，最大20并行
python3 batch_extraction_template.py \
    --start-idx 0 \
    --end-idx 74860 \
    --batch-size 2135 \
    --max-workers 20 \
    --output-dir "/root/graphnet_workspace/huggingface/worker1"
```

### 独立批次启动 (备用)

如需单独启动特定批次：

```bash
# 批次示例: 0-2135
nohup python3 /root/extract_from_model_list.py \
    --start-idx 0 \
    --end-idx 2135 \
    --output-dir /root/graphnet_workspace/huggingface/worker1 \
    > /tmp/worker1_batch_001.log 2>&1 &
```

## 进度监控

### Subagent监控 (推荐)

已启动3个subagent实时监控：

```bash
# 查看监控Subagent
ps aux | grep worker1_monitor_subagent

# 查看实时日志
tail -f /tmp/worker1_subagent_monitor.log

# 查看最新报告
cat /root/graphnet_workspace/huggingface/worker1/subagent_report.json
```

### 手动查看进度

```bash
# 统计各类型完整模型数
for task in text-classification text-generation fill-mask feature-extraction \
            token-classification question-answering automatic-speech-recognition; do
    count=$(find /root/graphnet_workspace/huggingface/worker1/$task \
            -name "graph_net.json" 2>/dev/null | wc -l)
    echo "$task: $count"
done

# 总完成数
find /root/graphnet_workspace/huggingface/worker1 \
    -name "graph_net.json" | wc -l
```

## 任务类型推断规则

脚本根据模型名称自动推断任务类型：

| 任务类型 | 识别关键词 |
|---------|-----------|
| text-generation | gpt, llama, mistral, qwen, yi, falcon, bloom, gemma, phi, pythia, opt, vicuna, wizard, chat, instruct, causallm |
| text2text-generation | t5, bart, pegasus, marian, mt5, longt5, translation, summarize, summarization |
| text-classification | sentiment, classification, emotion, nli, mnli, finetuned, classifier, sst, imdb, yelp |
| token-classification | ner, token, pos-tag, pos_tag, named-entity, bio, conll |
| question-answering | qa, question, answering, squad, glue |
| fill-mask | bert, roberta, distilbert, electra, albert, fill-mask, masked, mlm |
| automatic-speech-recognition | whisper, wav2vec, speech, asr, audio, wav, voice |
| image-classification | vit, resnet, convnext, image, vision, clip, detr, sam |
| vision2seq | llava, blip, flamingo, multimodal, vqa, vision-language |
| feature-extraction | 其他（默认） |

## 常见问题

### 1. 内存监控
- 设置内存监控脚本，防止OOM
- 当内存使用超过64GB时自动杀死进程
- 监控Subagent每5秒报告一次内存状态

### 2. 进程恢复
如果进程被杀死，批次模板会自动检测并重启。如需手动恢复：
```bash
# 检查批次状态
cat /root/graphnet_workspace/huggingface/worker1/subagent_report.json

# 重启缺失的批次
python3 batch_extraction_template.py --start-idx X --end-idx Y ...
```

### 3. 模型跳过
脚本会自动跳过已存在的模型（通过检查graph_net.json或subgraph目录）。

### 4. 输入类型自适应
- 文本模型: `torch.randint(0, vocab_size, (1, seq_len))`
- 语音模型: `torch.randn(1, num_mel_bins, 3000)`
- 视觉模型: `torch.randn(1, 3, 224, 224)`

## 关键经验

1. **目录名转换**: 模型ID中的 `/` 替换为 `_`
   - 例如: `0-hero/Matter-0.1-7B` -> `0-hero_Matter-0.1-7B`

2. **任务类型推断**: 脚本根据模型名称自动推断任务类型并分类到对应目录

3. **代理设置**: 必须配置代理以访问HuggingFace
   - `export http_proxy=http://agent.baidu.com:8891`
   - `export HF_ENDPOINT=https://hf-mirror.com`

4. **所有类型统一处理**: 脚本已更新支持所有HuggingFace模型类型，无需区分处理

---
*文档创建时间: 2026-04-01*
*最后更新: 2026-04-01 (扩展为全类型支持)*

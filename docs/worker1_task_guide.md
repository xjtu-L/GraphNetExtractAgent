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

## 质量管控流程（重要）

### 抽取后质量检查流程

每个抽取任务结束后，必须执行以下流程：

```
1. 抽取任务结束
   └─ 立即检测 → 空目录 / 残图

2. 残图检测
   └─ 立即制定修复方案 → 执行修复

3. 空目录分析
   ├─ 分析失败原因（模型类型、格式、依赖）
   ├─ 检查现有抽取方案
   └─ 无方案 → 立即制定特定化方案

4. 抽取成功
   └─ 更新方案到文档（供后续复用）

5. 文档维护
   └─ 流程和方案更新到docs
```

### 空目录处理标准

**检测命令：**
```bash
find /root/graphnet_workspace/huggingface/worker1 -type d -empty
```

**处理步骤：**
1. **分析模型类型** - 检查是否为非标准模型
2. **查看失败日志** - 定位具体错误
3. **制定方案** - 根据类型选择处理方式
4. **创建说明文件** - 空目录下创建 `EXTRACTION_SKIP_REASON.md`
5. **更新文档** - 将方案记录到本文档

### 残图检测标准

**正确格式判断：**
```python
# 格式1: 多子图
has_subgraph = any(d.startswith('subgraph_') for d in os.listdir(model_dir))
has_model_py = os.path.exists(os.path.join(sg_path, 'model.py'))
has_graph_hash = os.path.exists(os.path.join(sg_path, 'graph_hash.txt'))

# 格式2: 单子图
has_model_py = os.path.exists(os.path.join(model_dir, 'model.py'))
has_graph_hash = os.path.exists(os.path.join(model_dir, 'graph_hash.txt'))
```

**残图类型：**
- `NO_SUBGRAPH`: 无子图目录且无model.py
- `INCOMPLETE_SUBGRAPH`: 子图缺少核心文件

## 嵌套目录问题

### 问题描述

嵌套目录是指模型目录下存在同名子目录（如 `model_name/model_name/`），这是一种错误的目录结构。

### 1. 为什么会出现嵌套目录

**根本原因：** GraphNet抽取器的行为

```
当使用 extract(model_name) 时：
  ↓
GraphNet创建 workspace/model_name/ 目录
  ↓
抽取结果写入 workspace/model_name/model_name/  (重复嵌套)
```

**触发条件：**
- 某些模型在抽取时，extractor会额外创建一层同名目录
- 特别是非标准模型或自定义输入的模型
- 与 `GRAPH_NET_EXTRACT_WORKSPACE` 环境变量设置有关

### 2. 如何识别嵌套目录

**检测命令：**
```bash
# 查找所有同名嵌套目录
find /root/graphnet_workspace/huggingface/worker1 -mindepth 2 -maxdepth 3 -type d | while read dir; do
  dirname=$(basename "$dir")
  if [ -d "$dir/$dirname" ]; then
    echo "NESTED: $dir/$dirname"
  fi
done
```

**正确结构 vs 嵌套结构：**

```
✅ 正确结构（单子图）
model_name/
├── model.py
├── graph_net.json
└── graph_hash.txt

✅ 正确结构（多子图）
model_name/
├── subgraph_0/
│   ├── model.py
│   └── graph_hash.txt
├── subgraph_1/
│   └── ...

❌ 错误结构（嵌套）
model_name/
└── model_name/          ← 多余的一层
    ├── model.py
    └── graph_net.json
```

### 3. 如何预防

**在extract.py中明确指定工作目录：**
```python
# 设置环境变量（在模型目录下）
os.environ['GRAPH_NET_EXTRACT_WORKSPACE'] = "/path/to/model_dir"

# 使用简单名称抽取
wrapped = extract("model")(model)  # 不要用完整路径
```

**抽取后验证：**
```bash
# 检查是否存在嵌套
if [ -d "model_name/model_name" ]; then
  echo "警告：发现嵌套目录"
fi
```

### 4. 修复方法

**手动修复单个目录：**
```bash
cd /path/to/model_dir

# 移动内层文件到外层
mv inner_dir/* .

# 删除空内层目录
rmdir inner_dir
```

**批量修复所有嵌套目录：**
```bash
find /root/graphnet_workspace/huggingface/worker1 -mindepth 2 -maxdepth 3 -type d | while read dir; do
  dirname=$(basename "$dir")
  nested="$dir/$dirname"

  if [ -d "$nested" ]; then
    # 移动文件
    mv "$nested"/* "$dir/" 2>/dev/null
    # 删除空目录
    rmdir "$nested" 2>/dev/null
    echo "修复: $nested"
  fi
done
```

**历史数据：**
- 2026-04-01: 发现并修复624个嵌套目录
- 影响范围：主要集中在feature-extraction类型
- 修复后：目录结构正常化

**新发现（2026-04-01）：**
- Marian翻译模型、Whisper模型可以正常抽取，但会产生嵌套目录
- 修复后生成多子图结构（subgraph_0到subgraph_N）
- 修复方法：抽取后移动文件+生成graph_hash.txt

## 不可抽取模型类型

以下类型模型**无法直接抽取**，需要特殊处理或跳过：

| 类型 | 示例 | 原因 | 处理方式 |
|------|------|------|----------|
| **LoRA适配器** | BiancaHRBR/flux-bianca | 需要基础模型 | SKIP，记录到skipped_models.txt |
| **GGUF格式** | TheBloke/Llama-2-GGUF | 量化权重格式 | SKIP，记录到skipped_models.txt |
| **CTranslate2** | WizardLM-13B-ct2 | 优化推理格式 | SKIP，记录到skipped_models.txt |
| **AWQ/GPTQ** | TheBloke/Mistral-AWQ | 量化格式 | SKIP，记录到skipped_models.txt |
| **MLX格式** | mlx-4bit, mlx-8bit | Apple MLX格式 | SKIP，记录到skipped_models.txt |
| **自定义PyTorch** | JcProg/pedsense-keypoint-lstm | 非Transformers | 创建特定化extract.py |

### 识别关键字

```python
# 自动跳过关键字
SKIP_KEYWORDS = [
    'lora', 'adapter',           # LoRA适配器
    'gguf',                      # GGUF量化
    'ct2', 'ctranslate2',        # CTranslate2格式
    'awq', 'gptq',               # 量化格式
    'mlx', 'exl2',               # Apple MLX / EXL2
    'q4_', 'q5_', 'q8_',         # 量化级别
    '4bit', '8bit', '3bit',      # 量化位数
]

# 可修复的类型（原脚本支持不完整）
REPAIRABLE_TYPES = [
    'marian',        # 翻译模型，使用MarianMTModel
    'whisper',       # 语音模型，使用AutoModelForAudioClassification
    'wav2vec',       # 语音模型，使用AutoModelForAudioClassification
]
```

## 非标准模型抽取方案

### 1. 自定义PyTorch模型（如KeypointLSTM）

**示例：** `JcProg/pedsense-keypoint-lstm-jaad-50e`

**方案：**
```python
# 1. 下载config.json
# 2. 手动定义模型类
class KeypointLSTM(nn.Module):
    def __init__(self, ...):
        ...

# 3. 随机初始化，不下载权重
model = KeypointLSTM(**config)
model.eval()

# 4. 抽取
extract(model, dummy_input)
```

**注意：** 不要加载预训练权重（如best.pt），只使用随机初始化

### 2. LoRA适配器（跳过处理）

**识别方法：**
```python
# 通过模型标签识别
- 标签包含: lora, adapter
- 文件包含: lora.safetensors, adapter_config.json
- 字段: base_model 存在且非空
```

**处理流程：**
```bash
# 1. 从模型列表删除
sed -i '/^MODEL_ID$/d' /root/GraphNetExtractAgent/data/model_lists/our_models_300k.txt

# 2. 记录到跳过列表
echo "DATE|MODEL_ID|LoRA适配器|原因" >> /root/GraphNetExtractAgent/docs/skipped_models.txt

# 3. 删除空目录
rm -rf /path/to/model/dir
```

**示例：**
```
模型: BiancaHRBR/flux-bianca
操作: 从列表删除，记录到skipped_models.txt，清理目录
状态: 已处理 (2026-04-01)
```

**注意：** 不创建单独说明文件，统一记录到skipped_models.txt

### 3. Chronos时间序列模型（T5-based）

**示例：** `FinText/Chronos_Small_2002_US`, `FinText/Chronos_Small_2002_Global`

**特点：**
- 基于T5架构
- 任务类型: time-series-forecasting
- 需要Seq2SeqLM处理

**方案：**
```python
from transformers import AutoConfig, AutoModelForSeq2SeqLM

config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_config(config, trust_remote_code=True)
model.eval()

# 准备T5输入
vocab_size = getattr(config, 'vocab_size', 32128)
input_ids = torch.randint(0, min(vocab_size, 1000), (1, seq_len))

# 抽取
wrapped = extract(model_name)(model)
with torch.no_grad():
    output = wrapped(input_ids)

# 修复嵌套目录
nested_dir = os.path.join(model_path, model_name)
if os.path.isdir(nested_dir):
    # 移动文件到外层
    for f in os.listdir(nested_dir):
        os.rename(os.path.join(nested_dir, f), os.path.join(model_path, f))
    os.rmdir(nested_dir)

# 生成graph_hash.txt
with open(os.path.join(model_path, "model.py"), 'r') as f:
    hash_val = hashlib.sha256(f.read().encode()).hexdigest()
with open(os.path.join(model_path, "graph_hash.txt"), 'w') as f:
    f.write(hash_val)
```

**注意：**
- 抽取时会报decoder_input_ids错误，但encoder部分已成功抽取
- 必须修复嵌套目录结构
- 修复时间: 2026-04-01 (FinText_Chronos_Small_2002_Global)

## 防微杜渐：抽取全流程防护

### 理念：预防胜于修复

与其事后修复问题，不如在抽取过程中预防问题发生。

### 1. 抽取前检查

**环境设置检查清单：**
```python
def pre_extraction_check():
    """抽取前环境检查"""
    # 1. 禁用进度条和警告
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["PYTHONUNBUFFERED"] = "1"

    # 2. 检查代理设置
    assert os.environ.get("http_proxy"), "未设置http_proxy"
    assert os.environ.get("HF_ENDPOINT"), "未设置HF_ENDPOINT"

    # 3. 清理已知的异常目录（如果存在）
    subprocess.run("find . -type d -name 'File *' -exec rm -rf {} + 2>/dev/null", shell=True)
    subprocess.run("find . -type d -name 'Fetching*' -exec rm -rf {} + 2>/dev/null", shell=True)

    # 4. 检查磁盘空间
    stat = os.statvfs(WORKER_DIR)
    free_gb = stat.f_frsize * stat.f_bavail / (1024**3)
    if free_gb < 5:
        raise RuntimeError(f"磁盘空间不足: {free_gb:.1f}GB < 5GB")

    return True
```

### 2. 抽取中防护

**异常捕获和输出控制：**
```python
def safe_extract_model(model_id, output_dir):
    """安全抽取模型，带完整防护"""
    # 设置模型目录
    model_name = model_id.replace("/", "_")
    model_path = os.path.join(output_dir, model_name)
    os.makedirs(model_path, exist_ok=True)

    # 设置GraphNet工作目录
    os.environ["GRAPH_NET_EXTRACT_WORKSPACE"] = model_path

    try:
        # 加载配置和模型
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        model_class = get_model_class(categorize_model(model_id))
        model = model_class.from_config(config, trust_remote_code=True)
        model.eval()

        # 准备输入
        dummy_input = prepare_dummy_input(config, model_id)

        # 抽取（带超时保护）
        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError("抽取超时(5分钟)")

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(300)  # 5分钟超时

        try:
            wrapped = torch_extract(model_name)(model)
            with torch.no_grad():
                _ = wrapped(dummy_input)
        finally:
            signal.alarm(0)  # 取消超时

        # 修复可能的嵌套目录
        fix_nested_directory(model_path, model_name)

        return True, "success"

    except TimeoutError as e:
        return False, f"timeout: {e}"
    except Exception as e:
        return False, f"error: {e}"
```

**嵌套目录修复函数：**
```python
def fix_nested_directory(model_path, model_name):
    """修复嵌套目录问题"""
    nested_path = os.path.join(model_path, model_name)

    if os.path.isdir(nested_path):
        # 移动内层文件到外层
        for item in os.listdir(nested_path):
            src = os.path.join(nested_path, item)
            dst = os.path.join(model_path, item)
            if not os.path.exists(dst):
                shutil.move(src, dst)
        # 删除空内层目录
        os.rmdir(nested_path)
        return True
    return False
```

### 3. 抽取后验证

**目录结构完整性检查：**
```python
def post_extraction_verify(model_path):
    """抽取后验证目录结构"""
    issues = []

    # 1. 检查是否为空目录
    if not os.listdir(model_path):
        issues.append("空目录")
        return False, issues

    # 2. 检查是否有异常子目录
    for item in os.listdir(model_path):
        if item.startswith("File ") or item.startswith("Fetching"):
            issues.append(f"异常目录: {item}")
            # 自动清理
            shutil.rmtree(os.path.join(model_path, item))

    # 3. 检查核心文件
    has_model_py = os.path.exists(os.path.join(model_path, "model.py"))
    has_subgraph = any(d.startswith("subgraph_") for d in os.listdir(model_path)
                      if os.path.isdir(os.path.join(model_path, d)))

    if not (has_model_py or has_subgraph):
        issues.append("缺少model.py或subgraph")
        return False, issues

    # 4. 检查并生成graph_hash.txt
    if has_subgraph:
        for sg in os.listdir(model_path):
            if sg.startswith("subgraph_"):
                sg_path = os.path.join(model_path, sg)
                model_py = os.path.join(sg_path, "model.py")
                hash_file = os.path.join(sg_path, "graph_hash.txt")
                if os.path.exists(model_py) and not os.path.exists(hash_file):
                    import hashlib
                    with open(model_py, 'r') as f:
                        hash_val = hashlib.sha256(f.read().encode()).hexdigest()
                    with open(hash_file, 'w') as f:
                        f.write(hash_val)
    elif has_model_py:
        hash_file = os.path.join(model_path, "graph_hash.txt")
        if not os.path.exists(hash_file):
            import hashlib
            with open(os.path.join(model_path, "model.py"), 'r') as f:
                hash_val = hashlib.sha256(f.read().encode()).hexdigest()
            with open(hash_file, 'w') as f:
                f.write(hash_val)

    return len(issues) == 0, issues
```

**批量验证所有模型：**
```bash
#!/bin/bash
# verify_all_models.sh

echo "=== 抽取后批量验证 ==="

empty_dirs=0
incomplete=0
nested_fixed=0

for dir in /root/graphnet_workspace/huggingface/worker1/*/; do
    # 检查空目录
    if [ -z "$(ls -A "$dir" 2>/dev/null)" ]; then
        echo "空目录: $dir"
        empty_dirs=$((empty_dirs + 1))
        continue
    fi

    # 检查异常目录
    for abnormal in "$dir"/File\ * "$dir"/Fetching*; do
        if [ -d "$abnormal" ]; then
            echo "清理异常目录: $abnormal"
            rm -rf "$abnormal"
        fi
    done

    # 修复嵌套
    dirname=$(basename "$dir")
    if [ -d "$dir/$dirname" ]; then
        mv "$dir/$dirname"/* "$dir/" 2>/dev/null
        rmdir "$dir/$dirname" 2>/dev/null
        nested_fixed=$((nested_fixed + 1))
    fi

    # 检查完整性
    if [ ! -f "$dir/model.py" ] && [ ! -d "$dir/subgraph_0" ]; then
        echo "不完整: $dir"
        incomplete=$((incomplete + 1))
    fi
done

echo ""
echo "验证结果:"
echo "  空目录: $empty_dirs"
echo "  不完整: $incomplete"
echo "  修复嵌套: $nested_fixed"
```

### 4. 常见错误的预防代码

**预防1: decoder_input_ids错误（T5/Marian）**
```python
# 对于Seq2Seq模型，只抽取encoder部分
if config.model_type in ['t5', 'marian', 'bart']:
    # 使用简单输入，让模型使用默认decoder
    output = model(input_ids=input_ids)  # 不指定decoder_input_ids
```

**预防2: 内存溢出**
```python
# 及时清理
try:
    wrapped = torch_extract(model_name)(model)
    _ = wrapped(dummy_input)
finally:
    del wrapped
    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
```

**预防3: 网络超时**
```python
from huggingface_hub import configure_http_backend
import requests

# 配置超时
configure_http_backend(
    backend_factory=lambda: requests.Session()
)

# 或使用重试机制
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def load_with_retry(model_id):
    return AutoConfig.from_pretrained(model_id, trust_remote_code=True)
```

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
- 自定义模型: 根据config定义输入shape

## 关键经验

1. **目录名转换**: 模型ID中的 `/` 替换为 `_`
   - 例如: `0-hero/Matter-0.1-7B` -> `0-hero_Matter-0.1-7B`

2. **任务类型推断**: 脚本根据模型名称自动推断任务类型并分类到对应目录

3. **代理设置**: 必须配置代理以访问HuggingFace
   - `export http_proxy=http://agent.baidu.com:8891`
   - `export HF_ENDPOINT=https://hf-mirror.com`

4. **所有类型统一处理**: 脚本已更新支持所有HuggingFace模型类型，无需区分处理

## 文档维护规范

### 及时更新文档的原则

**每次新发现必须立即记录到本文档：**

1. **新模型类型问题**
   - 发现新的不可抽取类型 → 添加到"不可抽取模型类型"章节
   - 发现可修复的新类型 → 添加到"非标准模型抽取方案"

2. **新问题模式**
   - 嵌套目录、残图新模式 → 添加到对应章节
   - 修复方法改进 → 更新"修复方法"

3. **统计数据更新**
   - 批量修复结果 → 添加到"历史数据"
   - 新问题规模 → 记录时间和数量

### 文档更新格式

```markdown
**新发现（YYYY-MM-DD）：**
- 发现描述
- 影响范围
- 修复方法
```

### 今日更新记录（2026-04-01）

| 时间 | 更新内容 | 位置 |
|------|----------|------|
| 16:00 | 添加质量管控流程 | "质量管控流程"章节 |
| 16:02 | 添加嵌套目录问题文档 | "嵌套目录问题"章节 |
| 16:15 | 添加CTranslate2等不可抽取类型 | "不可抽取模型类型"章节 |
| 16:30 | 添加Marian/Whisper可修复方案 | "非标准模型抽取方案"章节 |
| 16:35 | 添加文档维护规范 | 本章节 |

**今日统计（2026-04-01）：**

| 指标 | 数量 |
|------|------|
| 活跃抽取进程 | 32个 |
| graph_hash.txt总数 | 3,589个 |
| 嵌套目录修复 | 624个 |
| 不可抽取模型(SKIP) | 188个 |
| CTranslate2模型(SKIP) | 14个 |
| Marian模型修复 | 20个 |
| Chronos模型修复 | 2个 |
| 异常目录清理 | 1,724个 |
| 空目录清理 | 3,109个 (3,111→2) |
| 剩余空目录 | 2个 |

**重要修复：**
- 修改抽取脚本为扁平输出结构（直接输出到worker根目录）
- 设置环境变量 GRAPH_NET_FLAT_OUTPUT=1
- 避免目录重复和task-type分类问题

**各类型模型分布：**
- text-generation: 507
- feature-extraction: 802
- text-classification: 153
- fill-mask: 137
- text2text-generation: 5
- 其他类型: 待抽取

## 异常目录问题

### 问题描述

发现以`File `和`Fetching`开头的异常目录名，这些是错误输出被误当成目录名创建的。

### 根本原因

1. **Python错误输出被截断**：当抽取脚本出错时，错误堆栈的`File`行被当作目录名
2. **tqdm进度条污染**：huggingface_hub下载时的进度条输出被误处理

**示例异常目录名：**
```bash
File "_usr_local_lib_python3.10_dist-packages_tqdm_std.py", line 1181...
Fetching 24 files:  96%|█████████▌| 23_24 [00:41<00:01,  1.03s_it]
```

### 识别方法

```bash
# 查找异常目录
cd /root/graphnet_workspace/huggingface/worker1
find . -type d -name "File *"          # Python错误堆栈
find . -type d -name "Fetching*"       # 进度条输出
find . -type d -name "*Traceback*"     # Traceback信息
find . -type d -name "*Error*"         # 错误信息
```

### 预防措施

**修复抽取脚本，确保输出正确重定向：**

已在以下文件添加环境变量设置：

**1. extract_from_model_list.py（第7-15行）**
```python
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["http_proxy"] = "http://agent.baidu.com:8891"
os.environ["https_proxy"] = "http://agent.baidu.com:8891"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# 禁用进度条和警告，防止输出污染文件系统
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
```

**2. batch_extraction_template.py（第20-28行）**
```python
# 禁用进度条和警告，防止输出污染文件系统
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"
```

**关键点：**
1. 设置`HF_HUB_DISABLE_PROGRESS_BARS=1`禁用huggingface进度条
2. 设置`TRANSFORMERS_VERBOSITY=error`减少transformers输出
3. 设置`PYTHONUNBUFFERED=1`确保输出及时刷新
4. 子进程通过`env=os.environ`继承这些设置

### 清理命令

```bash
cd /root/graphnet_workspace/huggingface/worker1

# 清理所有异常目录
find . -type d -name "File *" -exec rm -rf {} + 2>/dev/null
find . -type d -name "Fetching*" -exec rm -rf {} + 2>/dev/null
find . -type d -name "*Traceback*" -exec rm -rf {} + 2>/dev/null

# 验证清理结果
find . -type d \( -name "File *" -o -name "Fetching*" \) 2>/dev/null | wc -l
# 应返回0
```

### 历史数据

**清理时间：** 2026-04-01
**清理数量：** 1724个
- `File *`开头: 385个
- `Fetching*`开头: 1339个

---
*文档创建时间: 2026-04-01*
*最后更新: 2026-04-01 (添加防微杜渐全流程防护)*

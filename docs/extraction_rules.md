
# 新模型抓取运行规则

## 1. Subagent数量规则
- **最低要求**: 每人至少保持 **8个subagent** 同时运行
- **推荐配置**: 8-12个subagent（根据系统资源调整）
- **最大限制**: 不超过16个（避免资源耗尽）

## 2. 批次分配规则
- **Worker1**: 负责模型列表前半部分（索引0-75,000）
- **Worker2**: 负责模型列表后半部分（索引75,001-149,720）
- **批次大小**: 每批次约1,000个模型（每个subagent）
- **并行要求**: 每个Worker保持至少8个subagent同时运行，各自处理不同区间

## 3. 自动管理机制
- 当某个批次完成时，自动从待处理队列中启动新批次
- 保持并行进程数不低于8个
- 监控日志目录: `/tmp/extract_logs/`

## 4. 监控命令
```bash
# 查看运行中的subagent数量
ps aux | grep extract_from_model_list | grep -v grep | wc -l

# 查看各批次进度
for log in /tmp/extract_batches/worker2_batch_*.log; do
    success=$(grep -c "成功:" "$log" 2>/dev/null || echo "0")
    echo "$(basename $log): 成功 $success 个"
done

# 查看实时日志
tail -f /tmp/extract_batches/worker2_batch_0_*.log
```

## 5. 模型列表来源（重要规则）

### 唯一授权来源
- **文件路径**: `/root/GraphNetExtractAgent/data/model_lists/our_models_300k.txt`
- **总模型数**: 149,720个（约15万个）
- **格式**: 每行一个model_id，可选包含framework、task、downloads等字段（以tab分隔）

### 任务分配说明
- **咱们的任务**: 这149,720个模型（约15万）
- **外包任务**: 另外15万个模型由其他团队负责
- **规则**: 必须从此文件读取模型列表，**不要从其他来源获取**（如硬编码字典、在线API等）

### 模型格式过滤要求
抽取脚本必须过滤以下不支持的格式：
- **GGUF格式**: `-gguf`, `.gguf` (用于llama.cpp)
- **量化格式**: `-awq`, `-gptq`, `-8bits`, `-4bits`
- **原始二进制**: `.bin`, `.safetensors`

这些格式无法被transformers库直接加载，会导致"Unrecognized model"错误。

### 实施要求
所有新模型抽取脚本必须：
1. 从上述文件加载模型列表
2. **过滤掉GGUF/AWQ等不支持的格式**（约2,151个模型）
3. 根据Worker分工处理指定区间
4. 记录已处理的模型以避免重复

### 使用示例
```python
MODEL_LIST_FILE = "/root/GraphNetExtractAgent/data/model_lists/our_models_300k.txt"

def load_models(start_idx, end_idx):
    models = []
    with open(MODEL_LIST_FILE, 'r') as f:
        lines = f.readlines()
        for line in lines[start_idx:end_idx]:
            model_id = line.strip().split('\t')[0]
            if model_id:
                models.append(model_id)
    return models
```

---

## 6. extract.py 标准格式规范（重要）

### 6.1 标准要求

每个抽取成功的模型必须生成 `extract.py` 文件，格式必须统一，确保可重复执行和结果一致性。

### 6.2 标准模板

```python
#!/usr/bin/env python3
"""Extract script for {model_id}
Task: {task_type}
"""
import torch
from transformers import {model_class}, AutoConfig
import sys, os
sys.path.insert(0, '/root/GraphNet')

def extract_model(model_path, device='cpu'):
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = {model_class}.from_config(config, trust_remote_code=True)
    model.eval()
    try:
        from graph_net.torch.extractor import extract
    except ImportError:
        print("Error: graph_net not found")
        return None
    vocab_size = getattr(config, 'vocab_size', 32000)
    max_pos = getattr(config, 'max_position_embeddings', 512)
    seq_len = min(32, max_pos)
    input_ids = torch.randint(0, min(vocab_size, 1000), (1, seq_len))
    try:
        return extract(model, input_ids)
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

### 6.3 关键要素说明

| 要素 | 说明 | 必须 |
|------|------|------|
| **Task信息** | 在docstring中标注模型任务类型 | ✅ |
| **模型类** | 根据模型类型选择AutoModel/AutoModelForCausalLM等 | ✅ |
| **from_config()** | 必须使用随机初始化，禁止from_pretrained()下载权重 | ✅ |
| **input_ids构造** | 构造随机输入张量，vocab_size和seq_len从config获取 | ✅ |
| **extract(model, input_ids)** | 正确调用方式，传入模型和输入张量 | ✅ |
| **argparse** | 支持命令行参数--model-path和--device | ✅ |
| **错误处理** | 包含try-except捕获异常 | ✅ |

### 6.4 模型类选择指南

根据模型类型选择合适的AutoModel类：

| 模型特征 | 模型类 | Task类型 |
|----------|--------|----------|
| 基础模型 | `AutoModel` | feature-extraction |
| 因果语言模型/GPT类 | `AutoModelForCausalLM` | text-generation |
| 掩码语言模型/BERT类 | `AutoModelForMaskedLM` | fill-mask |
| 序列分类 | `AutoModelForSequenceClassification` | text-classification |
| Token分类/NER | `AutoModelForTokenClassification` | token-classification |
| 问答模型 | `AutoModelForQuestionAnswering` | question-answering |
| Seq2Seq/T5类 | `AutoModelForSeq2SeqLM` | text2text-generation |

### 6.5 禁止事项

❌ **以下写法是错误的，必须避免：**

```python
# 错误1: 使用from_pretrained()下载权重
model = AutoModel.from_pretrained(model_path)  # ❌ 禁止

# 错误2: 调用方式错误
extract(model, model_path)  # ❌ 禁止传入路径

# 错误3: 缺少input_ids构造
return extract(model)  # ❌ 必须构造input_ids
```

✅ **正确写法：**

```python
# 正确: 使用from_config()随机初始化
model = AutoModel.from_config(config)  # ✅

# 正确: 传入构造好的input_ids
input_ids = torch.randint(0, vocab_size, (1, seq_len))
return extract(model, input_ids)  # ✅
```

### 6.6 验证命令

```bash
# 检查extract.py格式是否正确
grep -l "extract(model, input_ids)" */extract.py | wc -l

# 检查是否有错误格式
grep -l "extract(model, model_path)" */extract.py

# 检查是否使用了from_config()
grep -l "from_config" */extract.py | wc -l
```



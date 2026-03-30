# Worker 协调分工规则

## 概述

为避免多个Worker重复抽取相同模型，我们采用基于模型列表文件的分工策略。

## 模型列表来源

**唯一授权的模型列表文件**:
```
/root/GraphNetExtractAgent/data/model_lists/our_models_300k.txt
```

- **总模型数**: 149,720个（约15万）
- **说明**: 这是咱们自己的15万个模型任务列表，另外15万个外包给别人做
- **格式**: 每行一个model_id，可选包含framework、task、downloads等字段（以tab分隔）

## Worker 分工规则

| Worker | 负责区间 | 起始索引 | 结束索引 | 模型数量 |
|--------|----------|----------|----------|----------|
| Worker1 | 前半部分 | 0 | 75,000 | ~75,000 |
| Worker2 | 后半部分 | 75,001 | 149,720 | ~74,720 |

## 实施方法

### 1. 使用专用抽取脚本

Worker2应使用基于模型列表的抽取脚本:
```bash
python3 /root/extract_from_model_list.py \
    --start-idx 75001 \
    --end-idx 76001 \
    --output-dir /root/graphnet_workspace/huggingface/worker2
```

### 2. 批量启动多个Subagent

每个Worker应保持至少8个subagent并行运行，各自处理不同区间:

```bash
# Worker2示例：启动8个subagent
for i in $(seq 0 7); do
    batch_start=$((75001 + i * 1000))
    batch_end=$((batch_start + 1000))

    nohup python3 /root/extract_from_model_list.py \
        --start-idx $batch_start \
        --end-idx $batch_end \
        --output-dir /root/graphnet_workspace/huggingface/worker2 \
        > /tmp/extract_batches/worker2_batch_${i}.log 2>&1 &
done
```

### 3. 监控运行状态

```bash
# 查看运行中的subagent数量
ps aux | grep extract_from_model_list | grep -v grep | wc -l

# 查看详细信息
ps aux | grep extract_from_model_list | grep -v grep
```

## 重要规则

1. **必须使用指定的模型列表文件**: `/root/GraphNetExtractAgent/data/model_lists/our_models_300k.txt`
2. **不要从其他来源获取模型列表**（如硬编码字典、在线API等）
3. **严格按照分工区间处理**，避免与Worker1重复
4. **保持至少8个subagent并行运行**
5. **每个subagent处理不同的批次区间**

## 输出目录结构

```
/root/graphnet_workspace/huggingface/worker2/
├── text-generation/
│   ├── model_name_1/
│   │   ├── graph_hash.txt
│   │   ├── graph_net.json
│   │   ├── model.py
│   │   ├── weight_meta.py
│   │   └── input_meta.py
│   └── ...
├── text-classification/
├── fill-mask/
└── feature-extraction/
```

## 模型格式过滤规则

### 不支持的格式
以下格式的模型**无法被transformers库直接加载**，抽取脚本必须过滤掉这些模型：

| 格式后缀 | 说明 |
|----------|------|
| `-gguf` | GGUF格式，用于llama.cpp等推理引擎 |
| `-awq` | AWQ量化格式 |
| `-gptq` | GPTQ量化格式 |
| `-8bits` / `-4bits` | 特定位数量化 |
| `.gguf` / `.bin` | 原始二进制格式 |

### 过滤实现
在抽取脚本中必须实现过滤逻辑：

```python
# 不支持的模型格式后缀
UNSUPPORTED_SUFFIXES = ['-gguf', '-awq', '-8bits', '-4bits', '-gptq', '-safetensors', '.gguf', '.bin']

def is_supported_model(model_id):
    """检查模型是否为transformers支持的格式"""
    model_lower = model_id.lower()
    for suffix in UNSUPPORTED_SUFFIXES:
        if suffix in model_lower:
            return False
    return True

def load_models(start_idx, end_idx):
    """加载模型列表，自动过滤不支持的格式"""
    models = []
    skipped = 0
    with open(MODEL_LIST_FILE, 'r') as f:
        lines = f.readlines()
        for line in lines[start_idx:end_idx]:
            model_id = line.strip().split('\t')[0]
            if model_id and is_supported_model(model_id):
                models.append(model_id)
            else:
                skipped += 1
    print(f"过滤: 跳过 {skipped} 个不支持的格式(GGUF/AWQ等)")
    return models
```

### 影响说明
- **模型列表中约有2,151个GGUF/AWQ格式模型**（占1.4%）
- 某些区间可能全部为GGUF格式（如75001-76001），需要跳过这些区间
- Worker2负责区间(75001-149720)中，约有72,568个正常transformers模型

## 常见问题

### 模型格式不支持
某些模型可能是GGUF格式或其他特殊格式，会导致"Unrecognized model"错误。这是正常的，脚本会自动跳过这些模型。

### 某些区间全是GGUF模型
如果某个区间的1000个模型全是GGUF格式，脚本会显示"跳过1000个不支持的格式"，这是正常的，继续处理下一区间即可。

### 模型需要授权
部分模型需要HuggingFace授权访问，会报错"gated"或"access"。这些模型需要手动申请权限后单独处理。

### 404错误
模型不存在或已被删除，会自动跳过。

## 更新记录

- 2026-03-30: 创建Worker分工规则文档

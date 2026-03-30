
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


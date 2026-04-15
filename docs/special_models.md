
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


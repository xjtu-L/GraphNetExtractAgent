# HuggingFace 模型计算图抽取方案

> 本文档说明如何使用 GraphNet Agent 从 HuggingFace Hub 抽取模型计算图。

---

## 1. 方案概述

GraphNet 提供了 **GraphNetAgent** 自动化管线，实现从 HuggingFace Model ID 到 GraphNet Sample 的全自动转换。

### 与 torchvision/timm 的区别

| 对比项 | torchvision/timm | HuggingFace |
|--------|------------------|-------------|
| 抽取方式 | 手写脚本 + `extract()` 装饰器 | **Agent 自动化管线** |
| 模型获取 | `get_model(name)` 或 `timm.create_model()` | 从 HF Hub 自动下载 |
| 输入构造 | 手动根据 forward 签名构造 | Agent 从 config.json 自动推断 |
| 脚本生成 | 手写 | Agent 自动生成 `run_model.py` |
| 适用场景 | 本地模型库、标准输入格式 | 海量 HF 模型、自动批量抽取 |
| **模型类型** | CV模型为主 | **支持所有模型类型** (NLP、Vision、Speech等) |

---

## 2. Agent 管线流程

```
┌─────────────────────────────────────────────────────────────────────┐
│ GraphNetAgent.extract_sample("bert-base-uncased")                   │
├─────────────────────────────────────────────────────────────────────┤
│ 1. Fetch      → 从 HuggingFace Hub 下载模型到本地缓存                  │
│ 2. Analyze    → 解析 config.json，提取元数据（model_type, vocab_size）  │
│ 3. CodeGen    → 根据元数据生成 run_model.py 脚本                       │
│ 4. Extract    → 执行脚本，调用 extract 装饰器提取计算图                  │
│ 5. Hash       → 生成 graph_hash.txt                                   │
│ 6. Dedup      → 检查是否与已有样本重复                                  │
│ 7. Verify     → 验证样本完整性                                         │
│ 8. Archive    → 保存 run_model.py 到样本目录                           │
└─────────────────────────────────────────────────────────────────────┘
```

### 关键模块

| 模块 | 文件 | 功能 |
|------|------|------|
| `HFFetcher` | `model_fetcher/huggingface_fetcher.py` | 从 HF Hub 下载模型 |
| `ConfigMetadataAnalyzer` | `metadata_analyzer/config_metadata_analyzer.py` | 解析 config.json 提取元数据 |
| `TemplateCodeGenerator` | `code_generator/template_generator.py` | 生成 run_model.py |
| `SubprocessGraphExtractor` | `graph_extractor/subprocess_graph_extractor.py` | 子进程执行提取 |
| `BasicSampleVerifier` | `sample_verifier/basic_sample_verifier.py` | 验证样本 |

---

## 3. 使用方法

### 3.1 单模型抽取

```python
from graph_net.agent import GraphNetAgent

# 初始化 Agent
agent = GraphNetAgent(
    workspace="/ssd1/liangtai-work/graphnet_workspace",
    hf_token=None  # 可选，用于访问私有模型
)

# 抽取单个模型
success = agent.extract_sample("bert-base-uncased")

if success:
    print("✅ 抽取成功")
else:
    print("❌ 抽取失败")
```

### 3.2 批量抽取

**方法一：使用内置测试脚本**

```bash
# 从 HF Hub 获取热门模型（按下载量排序）
python -m graph_net.agent.tests.test_batch_success_rate \
    --count 100 \
    --workspace /ssd1/liangtai-work/graphnet_workspace \
    --output batch_results.json

# 按任务类型筛选
python -m graph_net.agent.tests.test_batch_success_rate \
    --task text-classification \
    --count 50 \
    --workspace /ssd1/liangtai-work/graphnet_workspace

# 使用默认测试模型列表
python -m graph_net.agent.tests.test_batch_success_rate \
    --use-default-models \
    --workspace /ssd1/liangtai-work/graphnet_workspace
```

**方法二：自定义模型列表文件**

```bash
# 创建模型列表文件
cat > models.txt << 'EOF'
bert-base-uncased
roberta-base
distilbert-base-uncased
gpt2
t5-small
EOF

# 批量抽取
python -m graph_net.agent.tests.test_batch_success_rate \
    --model-list-file models.txt \
    --workspace /ssd1/liangtai-work/graphnet_workspace
```

**方法三：Python 脚本批量抽取**

```python
from graph_net.agent import GraphNetAgent
from huggingface_hub import list_models

agent = GraphNetAgent(workspace="/ssd1/liangtai-work/graphnet_workspace")

# 获取热门 NLP 模型
models = list_models(sort="downloads", direction=-1, limit=100)

for model in models:
    model_id = model.id
    print(f"抽取: {model_id}")
    success = agent.extract_sample(model_id)
    print(f"  结果: {'成功' if success else '失败'}")
```

---

## 4. Agent 输入推断逻辑

### 4.1 NLP 模型（BERT, GPT, T5 等）

Agent 从 `config.json` 读取：
- `max_position_embeddings` → 序列长度（默认 512）
- `vocab_size` / `embedding_size` → 词表大小
- `model_type` / `architectures` → 模型类型

生成输入：
```python
inputs = {
    "input_ids": torch.randint(0, vocab_size, (1, seq_len), dtype=torch.int64),
    "attention_mask": torch.ones((1, seq_len), dtype=torch.int64),
}
```

### 4.2 Vision 模型（ViT 等）

从 `config.json` 读取：
- `image_size` → 图像尺寸（默认 224）
- `num_channels` → 通道数（默认 3）

生成输入：
```python
inputs = {
    "pixel_values": torch.randn((1, 3, 224, 224), dtype=torch.float32),
}
```

### 4.3 未知模型类型

默认使用 NLP 输入格式：
```python
inputs = {
    "input_ids": torch.randint(0, 128, (1, 128), dtype=torch.int64),
}
```

---

## 5. 生成的脚本示例

Agent 为 `bert-base-uncased` 生成的 `run_model.py`：

```python
import torch
try:
    from transformers import AutoModel
except ImportError:
    raise ImportError("transformers is required. Install with: pip install transformers")

import graph_net

def main():
    # Load model
    model = AutoModel.from_pretrained("/path/to/cache/bert-base-uncased")

    # Prepare inputs
    inputs = {}
    inputs["input_ids"] = torch.randint(0, 28996, (1, 512), dtype=torch.int64)
    inputs["attention_mask"] = torch.ones((1, 512), dtype=torch.int64)

    # Extract graph
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    # Move inputs to same device as model
    inputs = {k: v.to(device) for k, v in inputs.items()}

    wrapped = graph_net.torch.extract(name="bert-base-uncased", dynamic=True)(model).eval()

    with torch.no_grad():
        wrapped(**inputs)

if __name__ == "__main__":
    main()
```

---

## 6. 常见问题与解决方案

### 6.1 下载模型失败

**问题**：网络问题导致模型下载超时

**解决**：设置代理
```bash
export http_proxy=http://agent.baidu.com:8891
export https_proxy=http://agent.baidu.com:8891
```

### 6.2 内存不足

**问题**：大模型（如 LLaMA）抽取时 OOM

**解决**：
1. 使用 CPU 抽取（速度慢但稳定）
2. 减小输入序列长度
3. 先测试小模型，确认流程正常

### 6.3 输入格式推断错误

**问题**：Agent 推断的输入格式与模型实际要求不匹配

**解决**：
1. 查看 `config.json` 的 `architectures` 字段
2. 手动修改生成的 `run_model.py`
3. 向 Agent 反馈，改进 `ConfigMetadataAnalyzer`

### 6.4 重复样本检测

**问题**：同一模型的不同微调版本可能产生相同计算图

**解决**：Agent 自动通过 `graph_hash.txt` 检测重复，跳过验证

### 6.5 动态输入 vs 静态输入

**问题**：默认使用 `dynamic=True`，某些模型需要 `dynamic=False`

**解决**：修改生成脚本中的 `dynamic` 参数

---

## 7. 任务规划

### 7.1 差距分析

已有 `transformers-auto-model` 样本：**1360 个**

需分析：
1. HF Hub 热门模型中还有哪些未覆盖
2. 按架构类型（BERT 系、GPT 系、T5 系等）统计缺口

### 7.2 执行策略

```
Phase 1: 验证 Agent 可用
  ├─ 用默认模型列表测试（8 个常见模型）
  └─ 确认成功率 > 50%

Phase 2: 批量抽取热门模型
  ├─ 按 downloads 排序抽取 Top 500
  └─ 记录失败模型和错误原因

Phase 3: 按架构分批
  ├─ text-classification 模型
  ├─ text-generation 模型
  ├─ fill-mask 模型
  └─ 其他任务类型

Phase 4: 处理失败模型
  ├─ 分析失败原因
  ├─ 改进 Agent 或提 Issue
  └─ 手动修复可修复的模型
```

### 7.3 并行化建议

Agent 的 `SubprocessGraphExtractor` 在子进程中执行，天然支持并行：

```python
from concurrent.futures import ThreadPoolExecutor
from graph_net.agent import GraphNetAgent

agent = GraphNetAgent(workspace="/ssd1/liangtai-work/graphnet_workspace")

def extract_one(model_id):
    return model_id, agent.extract_sample(model_id)

model_ids = ["bert-base-uncased", "roberta-base", "gpt2"]

with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(extract_one, model_ids))

for model_id, success in results:
    print(f"{model_id}: {'成功' if success else '失败'}")
```

---

## 8. 产出物清单

每次成功抽取后，Agent 在 workspace 下生成：

```
$GRAPH_NET_EXTRACT_WORKSPACE/
└── model_id/
    ├── model.py                    # 计算图结构
    ├── input_meta.py               # 输入元信息
    ├── weight_meta.py              # 权重元信息
    ├── input_tensor_constraints.py # 输入约束
    ├── graph_hash.txt              # 图哈希（去重用）
    ├── graph_net.json              # 元数据
    └── run_model.py                # 生成的抽取脚本
```

---

## 9. 参考文档

- GraphNet Agent README: `/root/GraphNet/graph_net/agent/README.md`
- Agent 源码: `/root/GraphNet/graph_net/agent/`
- 批量测试脚本: `/root/GraphNet/graph_net/agent/tests/test_batch_success_rate.py`

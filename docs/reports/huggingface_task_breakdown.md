# HuggingFace 模型抽取任务分解（改进版）

> 参考 timm 抽取经验，设计完整的 extract + validate 流程
> 创建时间：2026-03-26
> 创建者：worker2

---

## 一、模型分析结果

### 1.1 数据概览

| 指标 | 数值 |
|------|------|
| HuggingFace 热门模型 | 1500 个 |
| 已有样本（samples/transformers-auto-model） | 1360 个 |
| 重复模型 | 80 个 |
| **待抽取模型** | **1420 个** |

### 1.2 按任务类型分布

| 任务类型 | 数量 | 优先级 | 说明 |
|----------|------|--------|------|
| text-generation | 341 | **高** | LLM 类，如 LLaMA、Mistral、Gemma |
| sentence-similarity | 95 | 高 | Embedding 模型，如 BGE、E5 |
| automatic-speech-recognition | 90 | 中 | 语音识别 |
| feature-extraction | 80 | 中 | 特征提取 |
| fill-mask | 65 | 中 | MLM 模型 |
| token-classification | 64 | 中 | NER 等 |
| image-classification | 58 | 中 | 视觉模型 |
| text-classification | 53 | 中 | 文本分类 |
| 其他 | 574 | 低 | 翻译、分割、检测等 |

### 1.3 关键发现

- **重复模型**：80 个已在 samples 中，需跳过
- **热门 LLM 缺失**：LLaMA、Mistral、Gemma、Phi 等均未抽取
- **Embedding 缺失**：BGE、E5 等热门 embedding 模型未抽取

---

## 二、抽取流程设计（参考 timm 经验）

### 2.1 流程图

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Phase 0: 准备工作                                                        │
│   ├─ 从 HF Hub 获取模型列表（按下载量排序）                                  │
│   ├─ 按 task 类型分类                                                    │
│   └─ 检查与 samples 的重复，生成 missing_models.txt                        │
├─────────────────────────────────────────────────────────────────────────┤
│ Phase 1: Agent 可用性验证                                                 │
│   ├─ 用默认 8 模型测试 GraphNetAgent                                       │
│   └─ 确认成功率 > 50%                                                    │
├─────────────────────────────────────────────────────────────────────────┤
│ Phase 2: 批量抽取（按 task 分批）                                          │
│   ├─ 每个 task 类型启动独立 subagent                                       │
│   ├─ 抽取成功后生成 extract.py 复现脚本                                    │
│   ├─ 存档成功/失败结果到 JSON                                              │
│   └─ 失败案例存档到 bad_cases/                                            │
├─────────────────────────────────────────────────────────────────────────┤
│ Phase 3: 批量验证                                                         │
│   ├─ 启动多个 subagent 并行验证                                           │
│   ├─ 使用 graph_net.torch.validate                                       │
│   └─ 记录验证结果                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│ Phase 4: 文档更新                                                         │
│   ├─ 更新 VALIDATED_MODELS.md                                            │
│   ├─ 生成抽取统计报告                                                     │
│   └─ 准备 PR 提交                                                        │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 与 timm 经验对照

| timm 经验 | HuggingFace 应用 |
|-----------|------------------|
| 抽取前检查 samples/ 去重 | ✅ 已检查，80 个重复已过滤 |
| 名称包含关系检查（如 .in1k 后缀） | HF 模型名格式为 `org/model`，转换为 `org_model` |
| 按 family 分批 | 按 **task 类型** 分批（text-generation、sentence-similarity 等） |
| 生成 extract.py 复现脚本 | ✅ 抽取成功后生成，包含 MODEL_NAME、INPUT_SHAPE、SOURCE="huggingface" |
| 结果存档 JSON | ✅ 保存成功/失败列表，失败存档到 bad_cases/ |
| 多 subagent 并行 | ✅ 每个 task 类型独立 subagent |

---

## 三、任务分解

### Phase 0: 准备工作 ✅ 已完成

| 任务 | 状态 | 产出 |
|------|------|------|
| 获取 HF 模型列表 | ✅ | 1500 个热门模型 |
| 按 task 分类 | ✅ | model_analysis.json |
| 检查重复 | ✅ | 80 个重复已过滤 |
| 生成 missing_models.txt | ✅ | 1420 个待抽取 |

### Phase 1: Agent 可用性验证

| 任务ID | 任务 | 命令 | 状态 |
|--------|------|------|------|
| HF-P1-1 | 默认模型测试 | `python -m graph_net.agent.tests.test_batch_success_rate --use-default-models` | 待执行 |
| HF-P1-2 | 验证成功率 | 检查结果 JSON | 待执行 |

### Phase 2: 批量抽取（按 task 分批）

| 任务ID | Task 类型 | 模型数 | 优先级 | 状态 |
|--------|-----------|--------|--------|------|
| HF-P2-01 | text-generation | 341 | **高** | 待执行 |
| HF-P2-02 | sentence-similarity | 95 | 高 | 待执行 |
| HF-P2-03 | automatic-speech-recognition | 90 | 中 | 待执行 |
| HF-P2-04 | feature-extraction | 80 | 中 | 待执行 |
| HF-P2-05 | fill-mask | 65 | 中 | 待执行 |
| HF-P2-06 | token-classification | 64 | 中 | 待执行 |
| HF-P2-07 | image-classification | 58 | 中 | 待执行 |
| HF-P2-08 | text-classification | 53 | 中 | 待执行 |
| HF-P2-09 | 其他 | 574 | 低 | 待执行 |

**并行策略**：每个 task 类型启动独立 subagent

### Phase 3: 批量验证

| 任务ID | 任务 | 说明 | 状态 |
|--------|------|------|------|
| HF-P3-1 | 并行验证 | 启动多个 subagent，每个验证一部分样本 | 待执行 |
| HF-P3-2 | 验证报告 | 统计通过/失败率 | 待执行 |

### Phase 4: 文档更新

| 任务ID | 任务 | 产出 | 状态 |
|--------|------|------|------|
| HF-P4-1 | 更新 VALIDATED_MODELS.md | 记录验证通过模型 | 待执行 |
| HF-P4-2 | 生成统计报告 | 抽取/验证统计 | 待执行 |
| HF-P4-3 | 准备 PR | 分批提交 | 待执行 |

---

## 四、extract.py 脚本模板

抽取成功后，为每个模型生成如下格式的 extract.py：

```python
#!/usr/bin/env python3
"""
HuggingFace 模型计算图抓取复现脚本

模型: bert-base-uncased
任务: fill-mask
来源: HuggingFace Hub
"""

import os
import sys
import json
import argparse
import torch

# ============ 模型参数 ============
MODEL_NAME = "bert-base-uncased"
MODEL_ID = "google-bert/bert-base-uncased"  # HF Hub ID
SOURCE = "huggingface"
HEURISTIC_TAG = "nlp"  # nlp / computer_vision / audio / multimodal / other
TASK_TYPE = "fill-mask"

# 输入参数
MAX_LENGTH = 512
BATCH_SIZE = 1
VOCAB_SIZE = 30522  # 从 config.json 获取

# 是否使用动态形状
DYNAMIC = True

def get_model():
    """加载模型"""
    from transformers import AutoModel
    return AutoModel.from_pretrained(MODEL_ID)

def get_input_data():
    """构造输入数据"""
    return {
        "input_ids": torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, MAX_LENGTH), dtype=torch.int64),
        "attention_mask": torch.ones((BATCH_SIZE, MAX_LENGTH), dtype=torch.int64),
    }

def extract_graph(output_dir=None, dynamic=False):
    """执行计算图抓取"""
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))

    print(f"模型: {MODEL_NAME}")
    print(f"来源: {SOURCE}")
    print(f"任务: {TASK_TYPE}")
    print(f"输出目录: {output_dir}")

    # 加载模型
    model = get_model()
    model.eval()

    # 构造输入
    inputs = get_input_data()

    # 提取计算图
    from graph_net.torch.extractor import extract
    wrapped = extract(name=MODEL_NAME, dynamic=dynamic)(model).eval()

    with torch.no_grad():
        wrapped(**inputs)

    # 更新 graph_net.json
    json_path = os.path.join(output_dir, "graph_net.json")
    if os.path.exists(json_path):
        with open(json_path) as f:
            data = json.load(f)
    else:
        data = {}

    data["model_name"] = MODEL_NAME
    data["source"] = SOURCE
    data["heuristic_tag"] = HEURISTIC_TAG
    data["task_type"] = TASK_TYPE

    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"抓取完成!")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"复现 {MODEL_NAME} 计算图抓取")
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--dynamic", action="store_true", default=DYNAMIC)
    args = parser.parse_args()
    extract_graph(args.output, args.dynamic)
```

---

## 五、执行命令速查

### 5.1 单模型测试
```bash
# 设置代理
export http_proxy=http://agent.baidu.com:8891
export https_proxy=http://agent.baidu.com:8891

# 单模型抽取
python -c "
from graph_net.agent import GraphNetAgent
agent = GraphNetAgent(workspace='/root/graphnet_workspace/huggingface')
print(agent.extract_sample('bert-base-uncased'))
"
```

### 5.2 批量抽取（按 task）
```bash
# text-generation 类模型
python /root/graphnet_workspace/huggingface/extract_huggingface_batch.py \
    --task text-generation \
    --count 50 \
    --workspace /root/graphnet_workspace/huggingface

# sentence-similarity 类模型
python /root/graphnet_workspace/huggingface/extract_huggingface_batch.py \
    --task sentence-similarity \
    --count 50 \
    --workspace /root/graphnet_workspace/huggingface
```

### 5.3 批量验证
```bash
# 验证所有已抽取模型
for dir in /root/graphnet_workspace/huggingface/*/; do
    python -m graph_net.torch.validate --model-path "$dir" --no-check-redundancy
done
```

---

## 六、产出文件

```
/root/graphnet_workspace/huggingface/
├── TASK_BREAKDOWN.md           # 本文档
├── model_analysis.json         # 模型分析结果
├── missing_models.txt          # 待抽取模型列表
├── extract_huggingface_batch.py # 批量抽取脚本
├── extract_results/            # 抽取结果存档
│   ├── extract_result_YYYYMMDD_HHMMSS.json
│   └── ...
├── bad_cases/                  # 失败案例存档
│   └── bad_cases_YYYYMMDD_HHMMSS.json
└── {model_name}/               # 抽取产出
    ├── model.py
    ├── weight_meta.py
    ├── input_meta.py
    ├── input_tensor_constraints.py
    ├── graph_net.json
    ├── graph_hash.txt
    └── extract.py              # 复现脚本
```

---

## 七、进度跟踪

| 指标 | 当前值 |
|------|--------|
| Phase 0 准备 | ✅ 完成 |
| 待抽取模型数 | 1420 |
| 已抽取模型数 | 0 |
| 验证通过模型数 | 0 |
| 失败模型数 | 0 |

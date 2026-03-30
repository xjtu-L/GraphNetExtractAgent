# HuggingFace 模型抽取策略设计 V2

> 设计目标：尽可能多地发现模型，边发现边抓取，多 agent 并行
> 创建时间：2026-03-26
> 创建者：worker2

---

## 一、模型发现策略

### 1.1 模型数量估算

| Task 类型 | 最少模型数 | 说明 |
|-----------|------------|------|
| text-generation | 2000+ | LLM 类 |
| text-classification | 2000+ | 文本分类 |
| token-classification | 2000+ | NER |
| fill-mask | 2000+ | MLM |
| question-answering | 2000+ | QA |
| summarization | 2000+ | 摘要 |
| translation | 2000+ | 翻译 |
| sentence-similarity | 2000+ | Embedding |
| feature-extraction | 2000+ | 特征提取 |
| image-classification | 2000+ | 视觉 |
| ... | ... | ... |
| **预估总计** | **30,000+** | 18 个主要 task 类型 |

### 1.2 边发现边抓取流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          边发现边抓取架构                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                   │
│   │  Agent 1    │    │  Agent 2    │    │  Agent N    │                   │
│   │ text-gen    │    │ text-cls    │    │ image-cls   │                   │
│   └──────┬──────┘    └──────┬──────┘    └──────┬──────┘                   │
│          │                  │                  │                           │
│          ▼                  ▼                  ▼                           │
│   ┌─────────────────────────────────────────────────────┐                 │
│   │              共享去重服务                             │                 │
│   │  - 已抽取模型集合（内存 + 文件持久化）                  │                 │
│   │  - samples/transformers-auto-model/ (1360)          │                 │
│   │  - workspace/ 新抽取模型                              │                 │
│   └─────────────────────────────────────────────────────┘                 │
│          │                  │                  │                           │
│          ▼                  ▼                  ▼                           │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                   │
│   │ 抽取 + 生成  │    │ 抽取 + 生成  │    │ 抽取 + 生成  │                   │
│   │ extract.py  │    │ extract.py  │    │ extract.py  │                   │
│   └─────────────┘    └─────────────┘    └─────────────┘                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.3 发现流程设计

```python
# 每个 Agent 的发现-抽取循环
def discover_and_extract_loop(task_type, batch_size=100):
    """边发现边抓取"""
    offset = 0
    while True:
        # 1. 发现一批模型
        models = discover_models(task_type, offset, batch_size)
        if not models:
            break  # 没有更多模型

        # 2. 去重过滤
        new_models = filter_duplicates(models)
        if not new_models:
            offset += batch_size
            continue  # 这批全是重复，继续下一批

        # 3. 抽取
        for model_id in new_models:
            success = extract_model(model_id)
            record_result(model_id, success)

        # 4. 更新偏移，继续下一批
        offset += batch_size
```

---

## 二、去重策略

### 2.1 去重检查点

| 检查点 | 数据源 | 检查时机 |
|--------|--------|----------|
| samples/ 目录 | `/root/GraphNet/samples/transformers-auto-model/` | 抽取前 |
| workspace/ 目录 | `/root/graphnet_workspace/huggingface/` | 抽取前 |
| 已抽取内存集合 | Python set() | 运行时 |

### 2.2 去重逻辑

```python
class DeduplicationService:
    """去重服务"""

    def __init__(self):
        self.extracted = set()
        self._load_existing()

    def _load_existing(self):
        """加载已存在模型"""
        # samples 目录
        samples_dir = "/root/GraphNet/samples/transformers-auto-model"
        for name in os.listdir(samples_dir):
            self.extracted.add(name)  # 格式: org_model

        # workspace 目录
        ws_dir = "/root/graphnet_workspace/huggingface"
        for name in os.listdir(ws_dir):
            self.extracted.add(name)

    def is_duplicate(self, model_id: str) -> bool:
        """检查是否重复"""
        sample_name = model_id.replace("/", "_")
        return sample_name in self.extracted

    def mark_extracted(self, model_id: str):
        """标记已抽取"""
        sample_name = model_id.replace("/", "_")
        self.extracted.add(sample_name)
```

---

## 三、多 Agent 并行协作

### 3.1 Agent 分工

| Agent ID | 负责类别 | 说明 |
|----------|----------|------|
| Agent-NLP-1 | text-generation | LLM |
| Agent-NLP-2 | text-classification, token-classification | 分类 |
| Agent-NLP-3 | fill-mask, question-answering | MLM, QA |
| Agent-NLP-4 | translation, summarization | Seq2Seq |
| Agent-EMB | sentence-similarity, feature-extraction | Embedding |
| Agent-VISION | image-classification, image-segmentation, object-detection | 视觉 |
| Agent-AUDIO | automatic-speech-recognition, audio-classification, text-to-speech | 音频 |
| Agent-MISC | 其他类型 | 杂项 |

### 3.2 并行架构

```
orchestrator (pane 0)
    │
    ├── worker (pane 1) ─── timm/torchvision 抽取
    │
    └── worker2 (我) ─── HuggingFace 协调
            │
            ├── subagent-1: text-generation
            ├── subagent-2: text-classification
            ├── subagent-3: fill-mask
            ├── subagent-4: sentence-similarity
            ├── subagent-5: image-classification
            └── subagent-6: automatic-speech-recognition
```

### 3.3 协调机制

```python
# 共享状态文件
SHARED_STATE = "/root/graphnet_workspace/huggingface/shared_state.json"

class Coordinator:
    """协调器"""

    def __init__(self):
        self.state_file = SHARED_STATE
        self._load_state()

    def _load_state(self):
        """加载状态"""
        if os.path.exists(self.state_file):
            with open(self.state_file) as f:
                self.state = json.load(f)
        else:
            self.state = {
                "total_discovered": 0,
                "total_extracted": 0,
                "total_duplicates": 0,
                "by_task": {},
                "by_agent": {}
            }

    def save_state(self):
        """保存状态"""
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=2)

    def report_progress(self, agent_id, task_type, extracted, duplicates, failed):
        """报告进度"""
        self.state["by_agent"][agent_id] = {
            "task": task_type,
            "extracted": extracted,
            "duplicates": duplicates,
            "failed": failed,
            "last_update": datetime.now().isoformat()
        }
        self.save_state()
```

---

## 四、抽取脚本设计

### 4.1 单 Agent 抽取脚本

```bash
#!/usr/bin/env python3
"""
HuggingFace 单类别抽取脚本

用法:
    python extract_by_task.py --task text-generation --limit 1000 --agent-id agent-nlp-1
"""

import os
import sys
import json
import argparse
from datetime import datetime

os.environ.setdefault("http_proxy", "http://agent.baidu.com:8891")
os.environ.setdefault("https_proxy", "http://agent.baidu.com:8891")
os.environ.setdefault("GRAPH_NET_EXTRACT_WORKSPACE", "/root/graphnet_workspace/huggingface")

from huggingface_hub import HfApi
from graph_net.agent import GraphNetAgent

class TaskExtractor:
    """单类别抽取器"""

    def __init__(self, task_type, agent_id, workspace):
        self.task_type = task_type
        self.agent_id = agent_id
        self.workspace = workspace
        self.api = HfApi()
        self.agent = GraphNetAgent(workspace=workspace)

        # 统计
        self.stats = {
            "discovered": 0,
            "extracted": 0,
            "duplicates": 0,
            "failed": 0,
            "failed_models": []
        }

        # 去重集合
        self.extracted_set = self._load_extracted_set()

    def _load_extracted_set(self):
        """加载已抽取模型集合"""
        extracted = set()

        # samples
        samples_dir = "/root/GraphNet/samples/transformers-auto-model"
        if os.path.isdir(samples_dir):
            extracted.update(os.listdir(samples_dir))

        # workspace
        if os.path.isdir(self.workspace):
            for name in os.listdir(self.workspace):
                if os.path.isdir(os.path.join(self.workspace, name)):
                    extracted.add(name)

        return extracted

    def is_duplicate(self, model_id):
        """检查重复"""
        sample_name = model_id.replace("/", "_")
        return sample_name in self.extracted_set

    def discover_models(self, offset=0, limit=100):
        """发现一批模型"""
        models = list(self.api.list_models(
            pipeline_tag=self.task_type,
            sort="downloads",
            limit=limit,
        ))
        return [m.id for m in models]

    def extract_one(self, model_id):
        """抽取单个模型"""
        try:
            success = self.agent.extract_sample(model_id)
            if success:
                self.extracted_set.add(model_id.replace("/", "_"))
            return success
        except Exception as e:
            return False

    def run(self, max_models=1000):
        """运行抽取"""
        print(f"[{self.agent_id}] 开始抽取 {self.task_type}")
        print(f"  目标数量: {max_models}")
        print(f"  已存在: {len(self.extracted_set)}")

        discovered = set()
        offset = 0
        batch_size = 100

        while len(self.stats["extracted"]) < max_models:
            # 发现一批
            models = self.discover_models(offset=offset, limit=batch_size)
            if not models:
                print(f"[{self.agent_id}] 没有更多模型")
                break

            # 过滤重复
            for model_id in models:
                if model_id in discovered:
                    continue
                discovered.add(model_id)
                self.stats["discovered"] += 1

                if self.is_duplicate(model_id):
                    self.stats["duplicates"] += 1
                    continue

                # 抽取
                print(f"[{self.agent_id}] {model_id}...", end=" ")
                success = self.extract_one(model_id)

                if success:
                    self.stats["extracted"] += 1
                    print("✓")
                else:
                    self.stats["failed"] += 1
                    self.stats["failed_models"].append(model_id)
                    print("✗")

                # 检查是否达到目标
                if self.stats["extracted"] >= max_models:
                    break

            offset += batch_size

        # 报告结果
        print(f"\n[{self.agent_id}] 完成")
        print(f"  发现: {self.stats['discovered']}")
        print(f"  抽取: {self.stats['extracted']}")
        print(f"  重复: {self.stats['duplicates']}")
        print(f"  失败: {self.stats['failed']}")

        return self.stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--agent-id", default="agent-default")
    parser.add_argument("--workspace", default="/root/graphnet_workspace/huggingface")
    args = parser.parse_args()

    extractor = TaskExtractor(
        task_type=args.task,
        agent_id=args.agent_id,
        workspace=args.workspace
    )

    stats = extractor.run(max_models=args.limit)

    # 保存结果
    result_file = f"{args.workspace}/extract_results/{args.agent_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    with open(result_file, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n结果保存: {result_file}")


if __name__ == "__main__":
    main()
```

---

## 五、执行计划

### 5.1 第一阶段：小规模验证

| 任务 | 模型数 | 目的 |
|------|--------|------|
| 验证 Agent 可用性 | 8 | 确认流程正常 |
| 小批量测试 | 100 | 验证成功率 |

### 5.2 第二阶段：并行抽取

启动 6 个 subagent 并行抽取：

```bash
# subagent-1: text-generation (LLM)
python extract_by_task.py --task text-generation --limit 500 --agent-id agent-tg &

# subagent-2: text-classification
python extract_by_task.py --task text-classification --limit 500 --agent-id agent-tc &

# subagent-3: sentence-similarity (Embedding)
python extract_by_task.py --task sentence-similarity --limit 500 --agent-id agent-ss &

# subagent-4: image-classification
python extract_by_task.py --task image-classification --limit 500 --agent-id agent-ic &

# subagent-5: automatic-speech-recognition
python extract_by_task.py --task automatic-speech-recognition --limit 500 --agent-id agent-asr &

# subagent-6: fill-mask
python extract_by_task.py --task fill-mask --limit 500 --agent-id agent-fm &
```

### 5.3 第三阶段：批量验证

抽取完成后，启动验证：

```bash
# 并行验证
for dir in /root/graphnet_workspace/huggingface/*/; do
    python -m graph_net.torch.validate --model-path "$dir" --no-check-redundancy &
done
```

---

## 六、产出物

```
/root/graphnet_workspace/huggingface/
├── STRATEGY_V2.md               # 本文档
├── shared_state.json            # 共享状态
├── extract_by_task.py           # 单类别抽取脚本
├── extract_results/             # 抽取结果
│   └── {agent_id}_{timestamp}.json
├── bad_cases/                   # 失败案例
└── {org}_{model}/               # 抽取产出
    ├── model.py
    ├── graph_net.json
    ├── extract.py               # 复现脚本
    └── ...
```

---

## 七、与 orchestrator 和 worker 的协作

| 角色 | 职责 |
|------|------|
| orchestrator | 总协调，分配任务 |
| worker | timm/torchvision 抽取 |
| worker2 | HuggingFace 协调 + 执行 subagent |
| subagents | 并行执行不同类别的抽取 |

**同步机制**：
- 定期向 orchestrator 汇报进度
- 与 worker 共享 workspace/ 目录
- 通过 shared_state.json 共享状态

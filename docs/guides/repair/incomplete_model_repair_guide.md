# GraphNet 残图修复指南

## 文档信息
- 创建时间: 2026-03-29
- 适用场景: HuggingFace模型残图修复
- 作者: GraphNetExtractAgent

---

## 1. 残图定义

### 1.1 什么是残图

**残图** = 有 `graph_hash.txt` 但缺少 `model.py` 或 `weight_meta.py` 的模型目录

### 1.2 完整模型的标准

**完整模型** = 有 `graph_hash.txt` + (根目录或任意 `subgraph_*` 子目录下有 `model.py` 和 `weight_meta.py`)

```
模型目录结构示例（完整）:
model_dir/
├── graph_hash.txt          # 必须
├── model.py                # 根目录有 或 subgraph_* 有
├── weight_meta.py          # 根目录有 或 subgraph_* 有
└── subgraph_0/             # 可选
    ├── model.py            # 子目录有
    └── weight_meta.py      # 子目录有
```

### 1.3 常见残图类型

| 类型 | 描述 | 修复方法 |
|------|------|----------|
| 无model.py | 只有hash，缺少模型定义 | 重新抽取 |
| 无weight_meta.py | 有model.py但缺少权重元数据 | 重新抽取 |
| 两者都缺 | 最彻底的残图 | 重新抽取 |

---

## 2. 统计口径

### 2.1 统一统计标准

```python
def check_model_complete(model_path):
    """
    完整模型 = 有graph_hash.txt + (根目录或任意subgraph有model.py和weight_meta.py)
    """
    has_hash = os.path.exists(os.path.join(model_path, 'graph_hash.txt'))
    if not has_hash:
        return False  # 没有hash，可能是未抓取的目录

    # 检查根目录
    has_model = os.path.exists(os.path.join(model_path, 'model.py'))
    has_weight = os.path.exists(os.path.join(model_path, 'weight_meta.py'))

    if has_model and has_weight:
        return True  # 根目录完整

    # 检查任意subgraph子目录
    for item in os.listdir(model_path):
        if item.startswith('subgraph_'):
            subdir = os.path.join(model_path, item)
            if os.path.isdir(subdir):
                has_model_sub = os.path.exists(os.path.join(subdir, 'model.py'))
                has_weight_sub = os.path.exists(os.path.join(subdir, 'weight_meta.py'))
                if has_model_sub and has_weight_sub:
                    return True  # subgraph完整

    return False  # 残图
```

### 2.2 统计范围

必须包含的任务类型:
- `text-classification`
- `text-generation`
- `text2text-generation`
- `fill-mask`
- `translation`
- `feature-extraction`
- `automatic-speech-recognition`
- `image-classification`
- `question-answering`
- `token-classification`

额外目录:
- `timm/`
- `torchvision/`
- huggingface根目录下的其他模型目录

---

## 3. 修复方法（祖训级别）

### 3.1 核心原则

**⚠️ 必须使用 `from_config()` 方式，不要从HF下载权重！**

原因:
1. 只下载 `config.json` (KB级)，不下载权重文件 (GB级)
2. 使用随机初始化权重抽取计算图
3. 支持任意大小模型 (7B/13B/70B/100B+)
4. 抽取速度更快
5. 避免网络问题和存储限制

### 3.2 正确代码示例

```python
from transformers import AutoModel, AutoConfig, AutoTokenizer
import torch

# 1. 下载配置 (KB级)
config = AutoConfig.from_pretrained(
    model_id,
    trust_remote_code=True
)

# 2. 随机初始化模型 (不下载权重!)
model = AutoModel.from_config(config).to(device).eval()

# 3. 加载tokenizer (可能还是需要下载)
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True
)

# 4. 处理padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 5. 使用GraphNet抽取
from graph_net.torch.extractor import extract
wrapped = extract(name=model_name_safe, dynamic=False)(model)

with torch.no_grad():
    wrapped(**inputs)
```

### 3.3 错误代码示例（禁止使用）

```python
# ❌ 错误: 会下载权重文件 (GB级)
model = AutoModel.from_pretrained(
    model_id,
    torch_dtype=torch.float32,
    trust_remote_code=True
)
```

---

## 4. 修复脚本模板

```python
#!/usr/bin/env python3
"""
残图修复脚本 - 使用from_config()随机初始化
"""

import os
import sys
import json
import torch
from datetime import datetime

# 设置代理
os.environ["http_proxy"] = "http://agent.baidu.com:8891"
os.environ["https_proxy"] = "http://agent.baidu.com:8891"
os.environ["HF_HOME"] = "/work/.cache/huggingface"

sys.path.insert(0, "/root/GraphNet")
from graph_net.torch.extractor import extract

from transformers import (
    AutoModel, AutoConfig, AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoModelForMaskedLM
)

# 任务类型映射
TASK_MODEL_MAP = {
    "text-classification": AutoModelForSequenceClassification,
    "text-generation": AutoModelForCausalLM,
    "fill-mask": AutoModelForMaskedLM,
    # ... 其他任务类型
}

def repair_model(model_id, task_type, device="cpu", max_length=128):
    """修复单个残图模型"""
    try:
        print(f"[修复] {model_id} ({task_type})")

        # 1. 加载配置 (祖训: 只下载config.json)
        config = AutoConfig.from_pretrained(
            model_id,
            trust_remote_code=True
        )

        # 2. 随机初始化模型 (祖训: 不下载权重)
        model_class = TASK_MODEL_MAP.get(task_type, AutoModel)
        model = model_class.from_config(config).to(device).eval()

        # 3. 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True
        )

        # 4. 处理padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # 5. 构造输入
        dummy_text = "This is a sample text."
        inputs = tokenizer(
            dummy_text,
            return_tensors="pt",
            padding="max_length",
            max_length=max_length,
            truncation=True
        )

        # 6. GraphNet抽取
        model_name_safe = model_id.replace("/", "_").replace("-", "_")
        wrapped = extract(name=model_name_safe, dynamic=False)(model)

        with torch.no_grad():
            wrapped(**inputs)

        print(f"✓ 完成: {model_id}")
        return True, "OK"

    except Exception as e:
        print(f"✗ 失败: {model_id} - {str(e)[:100]}")
        return False, str(e)

def main():
    # 示例: 修复fill-mask任务
    model_id = "bert-base-uncased"
    task_type = "fill-mask"
    repair_model(model_id, task_type)

if __name__ == "__main__":
    main()
```

---

## 5. 常见错误处理

### 5.1 Tokenizer没有pad_token

```python
# 错误: tokenizer.pad_token is None
# 解决:
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

### 5.2 模型架构不支持

某些特殊模型（如Mamba、Mamba2）可能不支持标准AutoModel，需要:
- 使用特定模型类
- 或标记为无法修复

### 5.3 配置文件缺失

如果 `config.json` 无法下载，模型无法修复，记录为失败。

---

## 6. 工作流程

### 6.1 残图修复流程

```
1. 统计残图数量 → 生成批次文件
2. 启动修复subagent → 并行处理
3. 监控修复进度 → 记录成功/失败
4. 验证修复结果 → 确认model.py和weight_meta.py生成
5. 导出修复报告 → 更新统计
```

### 6.2 与Worker协作

- 按任务类型分配批次
- Worker1/Worker2各自处理不同批次
- 定期同步进度
- 避免重复修复

---

## 7. 经验总结

### 7.1 关键数字

- 残图修复成功率: ~81%
- 修复速度: 1-2分钟/模型
- 抓取新模型风险: 高（可能产生无效计算图）

### 7.2 最佳实践

1. **先修复残图，再抓新模型**
2. **必须使用 `from_config()` 方式**
3. **统计口径要统一（包含subgraph）**
4. **定期停止抓取进程，检查无效计算图**

### 7.3 注意事项

- 不要信任 `extract_hf_full.py` 产生的计算图（有bug）
- 每个成功抓取的模型必须有 `extract.py` 复现脚本
- 定期检查残图数量，避免积压

---

## 8. 相关文件

- 修复脚本: `/root/graphnet_workspace/huggingface/repair_from_json.py`
- 残图列表: `/work/graphnet_workspace/incomplete_for_worker2.txt`
- 批次文件: `/work/graphnet_workspace/batch_for_worker2_v2/`
- 统计报告: `/work/graphnet_workspace/strict_stats_report.json`

---

## 附录：快速命令

```bash
# 统计残图数量
cd /work/graphnet_workspace/huggingface && python3 count_incomplete_models.py

# 启动修复subagent
nohup python3 repair_from_json.py --task fill-mask --agent-id w1-fm-01 >logs/fm_01.log 2>&1 &

# 检查修复进度
grep -r "✓ 完成" repair_logs/*.log | wc -l

# 停止所有抓取进程
pkill -9 -f extract_hf_full.py

# 导出残图列表给Worker2
cat incomplete_for_worker2.txt | wc -l
```

---

*本文档记录了GraphNetExtractAgent修复残图的核心经验，供后续任务参考。*

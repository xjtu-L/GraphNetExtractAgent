# GraphNetExtractAgent 内容调整提议

> 与 Worker2 标准保持一致
> 创建时间: 2026-03-29

---

## 一、完整目录标准统一（与Worker2一致）

### 1.1 核心定义

完整模型必须同时包含以下4个核心文件：

| 文件 | 作用 | 检查路径 |
|------|------|----------|
| `graph_hash.txt` | 计算图唯一哈希标识 | 根目录或 subgraph_0/ |
| `model.py` | 模型结构定义代码 | 根目录或 subgraph_0/ |
| `weight_meta.py` | 权重张量元数据 | 根目录或 subgraph_0/ |
| `extract.py` | 复现脚本（祖训要求） | 根目录或 subgraph_0/ |

**注意**: `config.json` 和 `input_meta.py` 不是判定完整性的必需文件。

### 1.2 检查脚本标准化

```bash
# 标准检查命令（与Worker2一致）
check_model_complete() {
    model_dir=$1
    has_hash=false
    has_model=false
    has_weight=false
    has_extract=false

    [ -f "$model_dir/graph_hash.txt" ] && has_hash=true
    [ -f "$model_dir/model.py" ] || [ -f "$model_dir/subgraph_0/model.py" ] && has_model=true
    [ -f "$model_dir/weight_meta.py" ] || [ -f "$model_dir/subgraph_0/weight_meta.py" ] && has_weight=true
    [ -f "$model_dir/extract.py" ] || [ -f "$model_dir/subgraph_0/extract.py" ] && has_extract=true

    if [ "$has_hash" = true ] && [ "$has_model" = true ] && [ "$has_weight" = true ] && [ "$has_extract" = true ]; then
        echo "✅ 完整"
    else
        echo "❌ 不完整: hash=$has_hash, model=$has_model, weight=$has_weight, extract=$has_extract"
    fi
}
```

---

## 二、config.json 相关表述修正

### 2.1 当前问题

多处文档将 "只下载config.json" 作为祖训，造成误解。

### 2.2 正确表述

| 模型类型 | 需要config.json | 原因 |
|---------|----------------|------|
| HuggingFace Transformers | ✅ 需要 | AutoConfig.from_pretrained() 依赖 |
| timm | ❌ 不需要 | 直接从模型结构推断 |
| torchvision | ❌ 不需要 | 直接从模型结构推断 |
| 自定义模型 | 视情况 | 如果extract.py能独立推断则不需要 |

### 2.3 需要修正的文档

| 文档路径 | 当前表述 | 修正后 |
|---------|---------|--------|
| `docs/guides/residual_repair_guide.md:17` | "只下载`config.json`（KB级）" | "对于HF模型，只下载`config.json`（KB级）不下载权重；timm等模型无需config" |
| `docs/QUICK_START.md:108` | "✅ 只下载config.json (KB级)" | "✅ 对于HF模型，只下载config.json (KB级)；其他模型可直接推断" |
| `docs/guides/incomplete_model_repair_guide.md:102` | "只下载 `config.json`" | "HF模型：只下载 `config.json`" |
| `docs/tasks/TASK_HISTORY.md:137` | "无config.json无法修复" | "无config.json的HF模型需要先下载config；timm模型无需config" |
| `docs/README.md:124` | "只下载config.json" | "HF模型只下载config.json；timm/torchvision直接推断" |

---

## 三、model.py 质量检查标准

### 3.1 简化版 vs 完整版

| 特征 | 简化版（异常） | 完整版（正确） |
|------|---------------|---------------|
| 文件大小 | < 1KB | > 1KB |
| forward实现 | `pass` 或空 | 包含实际计算节点 |
| 包含内容 | 只有框架类定义 | fx.GraphModule完整节点 |
| 示例 | `def forward(self, *args, **kwargs): pass` | 包含 `torch.nn.functional` 调用 |

### 3.2 验证函数（与Worker2一致）

```python
def validate_model_py(filepath):
    """验证 model.py 是否为完整计算图（Worker2标准）"""
    with open(filepath) as f:
        content = f.read()

    # 检查1: 文件大小
    if len(content) < 1024:
        return False, "文件太小(<1KB)，可能是简化版"

    # 检查2: 排除空实现
    if "def forward(self, *args, **kwargs):" in content and "pass" in content:
        return False, "空forward实现，是占位符"

    # 检查3: 包含实际计算
    if "torch.nn.functional" not in content and "L_" not in content:
        return False, "无实际计算节点"

    return True, "完整计算图"
```

---

## 四、extract.py 规范（祖训级别）

### 4.1 必备元素

```python
#!/usr/bin/env python3
"""
Extract script for {MODEL_ID}
Task: {TASK_TYPE}
"""

import os
import sys
import torch

# 必须设置代理
os.environ["http_proxy"] = "http://agent.baidu.com:8891"
os.environ["https_proxy"] = "http://agent.baidu.com:8891"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

sys.path.insert(0, "/root/GraphNet")
from graph_net.torch.extractor import extract
from transformers import {MODEL_CLASS}, AutoConfig

MODEL_ID = "xxx"
TASK_TYPE = "xxx"

def main():
    # 从config加载（祖训：不下载权重）
    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = {MODEL_CLASS}.from_config(config, trust_remote_code=True)

    # 执行抽取...
    wrapped = extract(name=MODEL_ID.replace("/", "_"), dynamic=True)(model)
    with torch.no_grad():
        wrapped(**inputs)

if __name__ == "__main__":
    main()
```

### 4.2 生成后验证

```bash
chmod +x $model_dir/extract.py
python3 -m py_compile $model_dir/extract.py  # 语法检查
```

---

## 五、目录结构统一

### 5.1 统一后的docs结构

```
docs/
├── guides/              # 指南文档（核心修复流程）
│   ├── STANDARD_REPAIR_WORKFLOW.md    # 标准修复流程
│   ├── ANOMALY_CASES.md               # 异常案例
│   ├── WORKER2_REPAIR_STANDARDS.md    # Worker2标准
│   ├── residual_repair_guide.md       # 残图修复指南
│   ├── incomplete_model_repair_guide.md
│   └── ...
├── reports/             # 报告
│   ├── huggingface_strategy_v2.md
│   └── ...
├── references/          # 参考
│   ├── INPUT_SHAPE_ANALYSIS.md
│   └── MEMORY.md
└── tasks/               # 任务
    ├── CURRENT_STATUS.md
    └── TASK_HISTORY.md
```

### 5.2 已删除/合并的目录

- ~~`knowledge/`~~ → 合并到 `docs/guides/`

---

## 六、修复状态标记规范

### 6.1 .repair_status 文件格式

```
worker: Worker1
status: completed
model_id: ernchern/personal_info_classification
timestamp: 2026-03-29T21:30:00
files_checked: graph_hash.txt,model.py,weight_meta.py,extract.py
model_py_size: 5482
validation: passed
```

### 6.2 状态检查清单（与Worker2一致）

- [ ] 抽取成功
- [ ] graph_hash.txt 已生成（根目录或subgraph_0）
- [ ] model.py 已生成且 > 1KB（根目录或subgraph_0）
- [ ] weight_meta.py 已生成（根目录或subgraph_0）
- [ ] extract.py 已生成并可执行
- [ ] .repair_status 标记为 completed

---

## 七、常见陷阱（与Worker2一致）

### 陷阱1：GraphNet输出到subgraph_0但检查只在根目录

**错误**:
```python
if not os.path.exists(os.path.join(model_dir, "model.py")):  # ❌ 漏检subgraph_0
```

**正确**:
```python
if not os.path.exists(os.path.join(model_dir, "model.py")) and \
   not os.path.exists(os.path.join(model_dir, "subgraph_0", "model.py")):  # ✅
```

### 陷阱2：简化版model.py误判为完整

**现象**: 根目录有549bytes的简化版，subgraph_0有4.8KB完整版

**解决**: 优先检查subgraph_0，或删除根目录简化版

---

## 八、文档更新计划

| 优先级 | 文档 | 更新内容 |
|--------|------|---------|
| P0 | `docs/guides/ANOMALY_CASES.md` | 新增双重model.py案例 |
| P0 | `docs/guides/STANDARD_REPAIR_WORKFLOW.md` | 更新config.json表述 |
| P1 | `docs/guides/residual_repair_guide.md` | 修正config.json描述 |
| P1 | `docs/QUICK_START.md` | 修正config.json描述 |
| P1 | `docs/guides/incomplete_model_repair_guide.md` | 修正config.json描述 |
| P2 | `docs/README.md` | 修正config.json描述 |
| P2 | `docs/tasks/TASK_HISTORY.md` | 修正config.json描述 |

---

**提议者**: Worker1
**审核者**: Worker2
**状态**: 待执行

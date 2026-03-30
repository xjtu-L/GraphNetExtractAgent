# 文档审核与失败案例整理报告

> 生成时间: 2026-03-30
> 审核人: Worker1

---

## 一、发现的过时/需要更新的方法

### 1. CONTRIBUTE_TUTORIAL.md / CONTRIBUTE_TUTORIAL_cn.md

**问题**: 安装依赖命令有误

**原文**:
```bash
pip install torch, torchvision
```

**应改为**:
```bash
pip install torch torchvision
```

**原因**: pip install 参数之间不应有逗号

---

### 2. STANDARD_REPAIR_WORKFLOW.md

**问题**: 修复脚本使用 `AutoModelForSequenceClassification.from_config()` 可能失败

**具体位置**: 第89-93行

**问题描述**:
- `AutoConfig.from_dict()` 方法不存在
- `PretrainedConfig` 类不被 `AutoModelForMaskedLM` 等接受
- 导致残图修复0成功率

**建议修复**:
```python
# 使用具体的Config类
from transformers import BertConfig, BertForMaskedLM, RobertaConfig, RobertaForMaskedLM

if config_dict.get('model_type') == 'bert':
    config = BertConfig(**config_dict)
    model = BertForMaskedLM(config)
elif config_dict.get('model_type') == 'roberta':
    config = RobertaConfig(**config_dict)
    model = RobertaForMaskedLM(config)
```

---

### 3. 多文档config.json描述不一致

| 文档 | 描述 | 建议 |
|------|------|------|
| ANOMALY_CASES.md | config.json并非总是必须 | ✅ 正确，需明确区分场景 |
| STANDARD_REPAIR_WORKFLOW.md | 需要config.json | ⚠️ 过于绝对，需补充说明 |
| residual_repair_guide.md | 未明确说明 | 建议添加config.json说明 |

**统一标准建议**:
- HuggingFace Transformers模型: **需要** config.json
- timm/torchvision模型: **不需要** config.json
- 已有完整model.py时: config.json**可选**（用于参考而非必须）

---

## 二、需要新增/改进的文档内容

### 1. 网络故障处理指南

**问题**: 当前抓取经常遇到HuggingFace连接超时

**已创建**: `/root/GraphNetExtractAgent/docs/guides/extract/huggingface_download_methods.md`

**建议在所有抓取脚本中默认启用**:
```python
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
```

---

### 2. 修复脚本改进

**问题**: 当前602个残图修复0成功

**主要原因**:
1. `AutoConfig.from_dict()` 方法不存在
2. `PretrainedConfig` 类不被AutoModel类接受
3. 未处理网络超时问题

**建议修复方案**:
```python
# 标准修复流程 V4
import json
from transformers import (
    BertConfig, BertForMaskedLM, BertForSequenceClassification,
    RobertaConfig, RobertaForMaskedLM, RobertaForSequenceClassification,
    DistilBertConfig, DistilBertForMaskedLM, DistilBertForSequenceClassification
)

def get_model_from_config(config_dict, task_type):
    model_type = config_dict.get('model_type', 'bert')

    if task_type == 'fill-mask':
        if model_type == 'bert':
            return BertForMaskedLM(BertConfig(**config_dict))
        elif model_type == 'roberta':
            return RobertaForMaskedLM(RobertaConfig(**config_dict))
        elif model_type == 'distilbert':
            return DistilBertForMaskedLM(DistilBertConfig(**config_dict))
    elif task_type == 'text-classification':
        if model_type == 'bert':
            return BertForSequenceClassification(BertConfig(**config_dict))
        elif model_type == 'roberta':
            return RobertaForSequenceClassification(RobertaConfig(**config_dict))
        elif model_type == 'distilbert':
            return DistilBertForSequenceClassification(DistilBertConfig(**config_dict))

    # 默认使用Bert
    return BertForMaskedLM(BertConfig(**config_dict))
```

---

## 三、系统整理失败案例

### 案例1: 残图修复失败（当前主要问题）

**规模**: 602个残图修复0成功

**错误类型**:
```
AttributeError: type object 'AutoConfig' has no attribute 'from_dict'
ValueError: Unrecognized configuration class <class 'transformers.configuration_utils.PretrainedConfig'>
OSError: We couldn't connect to 'https://huggingface.co' to load the files
```

**根本原因**:
1. 修复脚本使用了不存在的`AutoConfig.from_dict()`方法
2. `PretrainedConfig`类不被AutoModel类接受
3. 未配置hf-mirror镜像导致网络超时

**解决方案**: 使用具体Config类（BertConfig, RobertaConfig等）

---

### 案例2: 新模型抓取网络超时

**规模**: 149,720个模型待抓取

**错误类型**:
```
OSError: [Errno 110] Connection timed out
```

**解决方案**:
- 已配置`HF_ENDPOINT=https://hf-mirror.com`
- 5个subagent正在使用镜像抓取

---

### 案例3: 双重model.py问题

**文档记录**: `ANOMALY_CASES.md` 案例1

**现象**: 根目录有简化版model.py，subgraph_0有完整版

**解决方案**: 删除根目录简化版，保留subgraph_0完整版

---

### 案例4: config.json误解

**文档记录**: `ANOMALY_CASES.md` 案例2

**现象**: 文档对config.json是否必须描述不一致

**统一标准**:
| 场景 | 是否需要config.json |
|------|---------------------|
| HF Transformers模型 | ✅ 需要 |
| timm/torchvision | ❌ 不需要 |
| 已有完整model.py | ⚠️ 可选 |

---

### 案例5: 验证失败模型

**文档记录**: `validated_failed.md`

**失败模型**:
- hrnet_w18
- levit_conv_256d
- regnetx_032, regnetx_040
- regnety_1280, regnety_160, regnety_320

**可能原因**:
1. 计算图结构复杂
2. 自定义算子
3. 模型权重问题

---

## 四、修复建议汇总

### 立即修复（高优先级）

1. **更新CONTRIBUTE_TUTORIAL.md**: 修复pip install命令
2. **更新STANDARD_REPAIR_WORKFLOW.md**: 使用具体Config类
3. **统一config.json说明**: 在各文档中明确说明

### 中期改进（中优先级）

1. **完善网络故障处理指南**: 整合到所有相关文档
2. **创建失败案例知识库**: 系统化整理解决方案
3. **优化修复脚本**: 提高修复成功率

### 长期规划（低优先级）

1. **文档自动化检查**: 定期检查命令语法
2. **失败案例自动归档**: 自动记录和分类失败案例
3. **修复流程自动化**: 减少人工干预

---

## 五、当前任务状态

### 文档审核
- [x] CONTRIBUTE_TUTORIAL.md
- [x] CONTRIBUTE_TUTORIAL_cn.md
- [x] STANDARD_REPAIR_WORKFLOW.md
- [x] ANOMALY_CASES.md
- [x] residual_repair_guide.md
- [x] huggingface_download_methods.md
- [x] validated_failed.md

### 失败案例整理
- [x] 602个残图修复失败案例
- [x] 网络超时失败案例
- [x] 双重model.py案例
- [x] config.json误解案例
- [x] 验证失败模型案例

### 正在进行的任务
- [ ] 新模型抓取（149,720个模型，5个subagent进行中）

---

**报告完成时间**: 2026-03-30
**下次更新建议**: 抓取任务完成后更新失败案例

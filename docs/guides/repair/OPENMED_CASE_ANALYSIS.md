# OpenMed 案例分析

> 案例分析：OpenMed_* 系列模型目录不全问题
> 创建时间：2026-03-29

---

## 问题描述

**案例模型**: `OpenMed_OpenMed-PII-Spanish-BiomedBERT-Base-110M-v1`

**问题**:
1. 缺失 `graph_hash.txt`
2. 缺失 `config.json`
3. `model.py` 只有 509 bytes（简化版，非完整计算图）
4. 无子图目录 (`subgraph_*`)

**文件状态**:
| 文件 | 状态 | 大小 |
|------|------|------|
| graph_hash.txt | ❌ 缺失 | - |
| config.json | ❌ 缺失 | - |
| model.py | ⚠️ 简化版 | 509 bytes |
| weight_meta.py | ✅ 存在 | 224 bytes |
| extract.py | ✅ 存在 | 2.5K |

---

## 类似案例统计

**OpenMed_* 前缀模型总数**: 20个

完整列表：
1. OpenMed_OpenMed-NER-ChemicalDetect-PubMed-v2-109M
2. OpenMed_OpenMed-NER-OncologyDetect-BioPatient-108M
3. OpenMed_OpenMed-NER-PathologyDetect-BigMed-278M
4. OpenMed_OpenMed-NER-PathologyDetect-BioMed-109M
5. OpenMed_OpenMed-NER-PharmaDetect-BigMed-278M
6. OpenMed_OpenMed-NER-OncologyDetect-ElectraMed-33M
7. OpenMed_OpenMed-NER-GenomicDetect-TinyMed-82M
8. OpenMed_OpenMed-NER-GenomicDetect-BioMed-335M
9. OpenMed_OpenMed-NER-AnatomyDetect-ElectraMed-335M
10. OpenMed_OpenMed-NER-ProteinDetect-PubMed-335M
11. OpenMed_OpenMed-NER-SpeciesDetect-BioPatient-108M
12. OpenMed_OpenMed-NER-OncologyDetect-SnowMed-568M
13. OpenMed_OpenMed-NER-AnatomyDetect-PubMed-v2-109M
14. OpenMed_OpenMed-NER-SpeciesDetect-BioClinical-108M
15. OpenMed_OpenMed-PII-BigMed-Large-560M-v1
16. OpenMed_OpenMed-NER-DNADetect-SuperMedical-355M
17. OpenMed_OpenMed-NER-DiseaseDetect-PubMed-109M
18. OpenMed_OpenMed-NER-PharmaDetect-ElectraMed-109M
19. OpenMed_OpenMed-NER-DNADetect-PubMed-335M
20. OpenMed_OpenMed-PII-Spanish-BiomedBERT-Base-110M-v1

---

## 修复方案

由于缺失 `config.json`，需要先下载配置：

```python
from transformers import AutoConfig, AutoModelForTokenClassification
from graph_net.torch.extractor import extract
import torch
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

model_id = "OpenMed/OpenMed-PII-Spanish-BiomedBERT-Base-110M-v1"

# 1. 下载 config.json
config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

# 2. 随机初始化模型
model = AutoModelForTokenClassification.from_config(config, trust_remote_code=True)

# 3. 执行抽取
wrapped = extract(name=model_id.replace("/", "_"), dynamic=True)(model)
with torch.no_grad():
    wrapped(**inputs)
```

---

**状态**: 待修复
**优先级**: P1（批量修复）

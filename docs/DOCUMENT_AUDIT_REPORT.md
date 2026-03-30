# GraphNetExtractAgent 文档审查报告

> 审查时间: 2026-03-30
> 审查人: Ducc
> 文档范围: /root/GraphNetExtractAgent/docs/

---

## 一、文档概览

### 1.1 文档清单

| 类别 | 数量 | 说明 |
|------|------|------|
| guides/ | 18 | 使用指南和操作手册 |
| references/ | 3 | 参考资料 |
| reports/ | 10 | 项目报告和统计 |
| tasks/ | 2 | 任务状态记录 |
| 根目录 | 2 | README, QUICK_START |
| **总计** | **35** | - |

### 1.2 关键文档

- `residual_repair_guide.md` - 残图修复指南（核心）
- `STANDARD_REPAIR_WORKFLOW.md` - 标准修复流程（核心）
- `REPAIR_STANDARD_CASE.md` - 标准修复案例
- `ANOMALY_CASES.md` - 异常案例记录
- `REPAIR_RULES.md` - 修复规则

---

## 二、发现的过时/需要更新的内容

### 2.1 【重要】config.json 要求过时

**问题**: 多份文档将 "必须有config.json" 或 "只下载config.json" 作为祖训

**实际验证**:
- HuggingFace模型: 需要config.json（用于AutoConfig.from_pretrained）
- timm/torchvision模型: **不需要** config.json
- 修复残图时: **有就用，没有也能直接抽取**

**涉及的文档**:
- [x] `residual_repair_guide.md` - 第148行: "HF模型只下载config.json"
- [x] `ANOMALY_CASES.md` - 已正确说明config.json非必须
- [x] `STANDARD_REPAIR_WORKFLOW.md` - 第430行已注明config.json非必须

**建议更新**:
```markdown
config.json 要求:
- HuggingFace Transformers: ✅ 需要（用于AutoConfig）
- timm/torchvision: ❌ 不需要（直接从结构推断）
- 修复残图时: 有就用，没有也能直接抽取
```

### 2.2 【重要】extract函数调用方式过时

**问题**: 多份文档和脚本使用旧版extract函数调用方式

**旧版（过时）**:
```python
result = torch_extract(
    model=model,
    model_name=MODEL_ID,
    output_path=".",
    input_shape=(1, 32),
    source="huggingface",
    task="text-generation"
)
```

**新版（正确）**:
```python
os.environ["GRAPH_NET_EXTRACT_WORKSPACE"] = "."
model = torch_extract(MODEL_ID)(model)
with torch.no_grad():
    _ = model(input_ids)  # 触发提取
```

**涉及的文档/脚本**:
- [x] `batch_repair_residuals_v2.py` - 已更新
- [x] `REPAIR_STANDARD_CASE.md` - 第30-38行需要更新

### 2.3 【中等】统计数字过时

**问题**: `residual_repair_guide.md` 中的残图统计数字已过时

**当前实际统计**（2026-03-30）:
| 任务类型 | 残图数量 |
|----------|----------|
| text-generation | 428 |
| text-classification | 待Worker1统计 |
| fill-mask | 待Worker1统计 |
| **我负责总计** | **444** |

**文档中的数字**（2026-03-30）:
| 任务类型 | 残图数量 |
|----------|----------|
| text-generation | 3,697 |
| text-classification | 1,549 |
| fill-mask | 536 |

**建议**: 添加"统计日期"标注，区分"历史数据"和"当前数据"

### 2.4 【轻微】部分文档路径不一致

**问题**: `STANDARD_REPAIR_WORKFLOW.md` 与 `residual_repair_guide.md` 中部分术语不统一

| 文档A | 文档B | 建议统一 |
|-------|-------|----------|
| "残图" | "简化版模型" | 统一使用"残图" |
| "完整计算图" | "GraphModule" | 统一使用"完整计算图（GraphModule）" |

---

## 三、系统性失败案例整理

### 3.1 失败类型分类

#### Type 1: 模型配置错误（最常见）

**特征**: 需要特殊配置参数

**案例**:
```
facebook/xmod-base
  - 错误: "If you want to use XmodLMHeadModel as a standalone, add is_decoder=True."
  - 原因: Xmod模型需要设置默认语言

Xenova/ernie-2.0-large-en
  - 错误: "If you want to use ErnieForCausalLM as a standalone, add is_decoder=True."
  - 原因: Ernie模型需要is_decoder配置

arithmetic-circuit-overloading/*
  - 错误: "rope_parameters's factor field must be a float >= 1, got 40"
  - 原因: RoPE参数配置不兼容
```

**解决方案**: 需要针对特定模型定制extract.py

#### Type 2: 网络/下载超时

**特征**: 无法下载config.json或模型文件

**案例**:
```
01-ai/Yi-6B
  - 错误: "We couldn't connect to 'https://hf-mirror.com'"
  - 原因: 网络超时或模型不存在
```

**解决方案**: 使用本地config.json或跳过

#### Type 3: extract函数调用错误

**特征**: TypeError: extract()

**案例**:
```
dacorvo/tiny-random-llama
  - 错误: TypeError: extract()
  - 原因: 使用旧版extract调用方式
```

**解决方案**: 更新为新版extract调用方式

#### Type 4: 输出无效（无GraphModule）

**特征**: 抽取成功但生成的model.py仍是简化版

**案例**:
```
yujiepan_mixtral-tiny-random
  - 错误: "No GraphModule in output"
  - 原因: extract wrapper未正确触发
```

**解决方案**: 检查extract wrapper调用和环境变量

### 3.2 当前修复失败统计（实时）

| Agent | 已处理 | 成功 | 失败 | 主要失败原因 |
|-------|--------|------|------|--------------|
| Agent 0 | 5 | 0 | 5 | 模型配置错误 |
| Agent 1 | 5 | 0 | 5 | 模型配置错误 |
| Agent 2 | 2 | 0 | 2 | 模型配置错误 |
| Agent 3 | 5 | 0 | 5 | 模型配置错误 |
| Agent 4 | 7 | 0 | 7 | 模型配置错误 |
| **总计** | **24** | **0** | **24** | - |

### 3.3 失败模型列表（待处理）

**当前444个残图中的关键失败模型**:

| 模型ID | Task | 失败原因 | 优先级 |
|--------|------|----------|--------|
| facebook/xmod-base | text-generation | 需要is_decoder | P1 |
| Xenova/ernie-2.0-large-en | text-generation | 需要is_decoder | P1 |
| dacorvo/tiny-random-llama | text-generation | extract调用错误 | P0 |
| arithmetic-circuit-overloading/* | text-generation | RoPE参数错误 | P2 |
| yujiepan_mixtral-tiny-random | text-generation | 无GraphModule输出 | P1 |

**待抽取列表**: 3,466个模型（已删除，无config.json）

---

## 四、文档更新建议

### 4.1 优先级P0（立即更新）

1. **统一extract函数调用方式**
   - 更新所有脚本使用新版extract调用
   - 文档中添加"extract函数使用说明"章节

2. **修正config.json要求**
   - 明确区分HuggingFace/timm/torchvision的不同要求
   - 强调"有就用，没有也能抽"

### 4.2 优先级P1（本周内更新）

1. **更新统计数据**
   - 添加统计日期标注
   - 区分"历史数据"和"当前数据"

2. **添加失败案例处理指南**
   - 创建`FAILED_CASES_GUIDE.md`
   - 分类说明各类失败原因和解决方案

### 4.3 优先级P2（本月内更新）

1. **统一术语**
   - 统一"残图"、"简化版"、"完整计算图"等术语

2. **完善案例库**
   - 补充更多典型修复案例
   - 添加失败案例分析

---

## 五、行动项

| 序号 | 任务 | 负责人 | 优先级 | 状态 |
|------|------|--------|--------|------|
| 1 | 更新extract函数调用方式 | Ducc | P0 | 进行中 |
| 2 | 修正config.json文档说明 | - | P0 | 待分配 |
| 3 | 创建失败案例处理指南 | - | P1 | 待分配 |
| 4 | 更新统计数据 | - | P1 | 待分配 |
| 5 | 统一文档术语 | - | P2 | 待分配 |

---

**报告生成时间**: 2026-03-30
**下次审查时间**: 建议一周后复查更新进度

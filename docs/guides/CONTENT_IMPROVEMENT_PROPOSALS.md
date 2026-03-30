# 文档内容调整提议汇总

> 本文档汇总所有发现的内容问题和改进建议
> 创建时间: 2026-03-29

---

## 一、文档错误修正

### 1.1 incomplete_model_repair_guide.md

**问题1**: 残图定义不完整，缺少"简化版模型"定义

**当前定义**:
```
残图 = 有 graph_hash.txt 但缺少 model.py 或 weight_meta.py 的模型目录
```

**修正建议**: 添加简化版模型定义
```
残图类型:
1. 缺失型: 有 graph_hash.txt 但缺少 model.py 或 weight_meta.py
2. 简化版型: model.py < 1KB，只有包装函数，无完整GraphModule计算图
```

**问题2**: 完整模型标准未包含 extract.py

**当前标准**:
```
完整模型 = 有 graph_hash.txt + (根目录或任意 subgraph_* 子目录下有 model.py 和 weight_meta.py)
```

**修正建议**:
```
完整模型 = 有 graph_hash.txt + model.py + weight_meta.py + extract.py
（model.py 必须在根目录或 subgraph_* 子目录，且大小 >= 1KB）
```

---

## 二、内容合并建议

### 2.1 重复文档识别

| 文档A | 文档B | 重复程度 | 建议 |
|-------|-------|---------|------|
| STANDARD_REPAIR_WORKFLOW.md | REPAIR_STANDARD_CASE.md | 高（都是修复流程） | 合并为一个文档，分"通用模板"和"具体案例"两节 |
| incomplete_model_repair_guide.md | residual_repair_guide.md | 中（都讲修复） | 明确分工：incomplete讲原理，residual讲具体from_config方法 |

### 2.2 合并方案

**方案**: 将 STANDARD_REPAIR_WORKFLOW.md 和 REPAIR_STANDARD_CASE.md 合并为 `COMPLETE_REPAIR_GUIDE.md`

**新结构**:
```
COMPLETE_REPAIR_GUIDE.md
├── 第一部分：标准修复流程（通用模板）
│   ├── 修复前准备
│   ├── 标准修复脚本
│   └── 验证步骤
└── 第二部分：标准案例（ernchern_personal_info_classification）
    ├── 问题诊断
    ├── 修复执行
    └── 结果验证
```

---

## 三、简化版模型定义补充

### 3.1 需要在 incomplete_model_repair_guide.md 添加的内容

**添加位置**: 第1.3节后，新增"1.4 简化版残图"

**添加内容**:
```markdown
### 1.4 简化版残图

#### 什么是简化版残图

**简化版残图** = model.py 存在但大小 < 1KB 的模型目录

这类模型看似有 model.py，但实际上只是包装函数，**没有完整的GraphModule计算图**。

#### 简化版特征

```python
# 简化版 model.py 示例（错误！）
def get_model():
    config = AutoConfig.from_pretrained("model_id")
    model = AutoModel.from_config(config)
    return model
```

**判断标准**:
- model.py 大小 < 1KB
- 内容只有包装函数
- 缺少详细的节点定义和连接关系

#### 正确 vs 简化版对比

| 特征 | 简化版（错误） | 完整版（正确） |
|------|---------------|---------------|
| model.py 大小 | < 1KB | > 1KB（通常几十KB到几百KB） |
| 计算图 | ❌ 无 | ✅ 完整GraphModule |
| 节点定义 | ❌ 无 | ✅ 详细的节点和连接 |
| 可用性 | ❌ 无法用于GraphNet分析 | ✅ 可用于分析和转换 |

#### 修复方法

简化版模型必须**重新正确抽取**：

1. 删除旧目录
2. 使用 `graph_net.torch.extractor.extract` wrapper
3. 执行 forward pass 触发完整抽取
4. 验证 model.py >= 1KB

```bash
rm -rf model_dir/
python3 extract_hf_full.py --model "model_id"
```
```

---

## 四、config.json描述修正

### 4.1 HUGGINGFACE_EXTRACTION_GUIDE.md 修正

**当前问题**: 过度强调 config.json 的必要性

**需要修改的位置**:

**位置1**: 第2节 Agent 管线流程
```
当前: 2. Analyze    → 解析 config.json，提取元数据
建议: 2. Analyze    → 解析模型配置（从config.json或模型结构推断）
```

**位置2**: 第4.1节 NLP模型输入推断
```
当前: Agent 从 `config.json` 读取：
建议: Agent 从模型配置读取（优先从config.json，如不存在则从模型结构推断）：
```

**位置3**: 添加说明框
```markdown
> **注意**: config.json 不是必须的
>
- HuggingFace模型：通常从 config.json 读取配置
- timm/torchvision模型：直接从模型结构推断配置
- 只有需要从HF重新加载模型时才必须下载 config.json
```

### 4.2 incomplete_model_repair_guide.md 修正

**位置**: 第3.1节核心原则

**当前表述**:
```
原因:
1. 只下载 `config.json` (KB级)，不下载权重文件 (GB级)
```

**修正表述**:
```
原因:
1. 使用 `from_config()` 随机初始化，只下载配置元数据（通常从config.json），不下载权重文件 (GB级)
```

---

## 五、入口文档确定

### 5.1 选定入口文档

**入口文档**: `docs/README.md`

**理由**:
1. 已经是文档索引
2. 新Agent首先阅读此文档
3. 包含快速导航和目录结构

### 5.2 需要在入口文档添加的内容

**添加位置**: 在"快速导航"后新增"任务文档链接"部分

**添加内容**:
```markdown
---

## 📋 任务文档链接

当前活跃任务文档：

| 任务 | 状态文档 | 进度 | 备注 |
|------|---------|------|------|
| 简化版模型修复 | [tasks/simplified_model_repair.md](tasks/simplified_model_repair.md) | 51.9% (41/79) | Worker2负责 |
| text-generation残图修复 | [tasks/tg_residual_repair.md](tasks/tg_residual_repair.md) | 刚开始 | 已启动5个agent |
| 新模型抓取 | [tasks/new_model_extraction.md](tasks/new_model_extraction.md) | 待开始 | 等待残图修复完成 |

历史任务：
- [tasks/TASK_HISTORY.md](tasks/TASK_HISTORY.md)

---
```

---

## 六、优先级排序

### P0 (必须立即修复)
1. incomplete_model_repair_guide.md - 添加简化版模型定义
2. HUGGINGFACE_EXTRACTION_GUIDE.md - 修正config.json描述

### P1 (重要)
3. 合并 STANDARD_REPAIR_WORKFLOW.md 和 REPAIR_STANDARD_CASE.md
4. docs/README.md - 添加任务文档链接

### P2 (可选)
5. 完善所有文档的交叉引用
6. 统一术语和格式

---

## 七、执行清单

- [ ] 修正 incomplete_model_repair_guide.md 残图定义
- [ ] 修正 incomplete_model_repair_guide.md 添加简化版模型定义
- [ ] 修正 HUGGINGFACE_EXTRACTION_GUIDE.md config.json描述
- [ ] 合并重复文档
- [ ] 更新 docs/README.md 添加任务链接
- [ ] 验证所有修改后的文档

---

*本文档作为内容改进的参考，按计划逐步实施*

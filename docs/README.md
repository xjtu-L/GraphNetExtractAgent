# GraphNetExtractAgent 文档中心

> HuggingFace 模型计算图抽取项目文档

---

## 核心入口

**新用户请从这里开始：**
- **[CONTRIBUTE_TUTORIAL.md](CONTRIBUTE_TUTORIAL.md)** - 贡献教程（核心文档）
  - 如何抽取 HuggingFace 模型计算图
  - 标准抽取流程
  - 注意事项

---

## 文档目录结构

```
docs/
├── CONTRIBUTE_TUTORIAL.md          # 核心贡献教程（从此开始）
├── CONTRIBUTE_TUTORIAL_cn.md       # 中文贡献教程
├── guides/
│   ├── WORKER_CHECKPOINT_GUIDE.md   # 【新增】Worker读档检查指南
│   ├── extract/                    # 提取相关文档
│   │   ├── EXTRACTION_EXPLAINED.md
│   │   ├── extraction_issues_and_solutions.md  # 【新增】抽取问题与解决方案
│   │   ├── HUGGINGFACE_EXTRACTION_GUIDE.md
│   │   ├── multi_worker_collaboration.md
│   │   ├── NESTED_DIRECTORY_ISSUE.md           # 【新增】嵌套目录专项记录
│   │   ├── PARALLEL_EXECUTION_BEST_PRACTICES.md  # 【新增】并行执行最佳实践
│   │   ├── quick_reference.md
│   │   ├── WORKER_DIRECTORY_GUIDE.md  # 【新增】Worker目录组织规范
│   │   └── worker_collaboration_workflow.md
│   │
│   ├── repair/                     # 修复相关文档（辅助参考）
│   │   ├── DOUBLE_MODEL_PY_CASES.md
│   │   ├── FAILED_CASES_GUIDE.md      # 【新增】失败案例处理指南
│   │   ├── incomplete_model_repair_guide.md
│   │   ├── OPENMED_CASE_ANALYSIS.md
│   │   ├── REPAIR_RULES.md
│   │   ├── REPAIR_STANDARD_CASE.md
│   │   ├── residual_repair_guide.md
│   │   └── STANDARD_REPAIR_WORKFLOW.md
│   │
│   └── validate/                   # 验证相关文档（待补充）
│
├── AGENT_CONTEXT.md                # Agent上下文
├── DOCUMENT_AUDIT_REPORT.md        # 【新增】文档审查报告
└── FAILED_CASES_GUIDE.md           # 【新增】失败案例处理指南
```

---

## 快速导航

### 我要开始抽取模型
→ 阅读 [CONTRIBUTE_TUTORIAL.md](CONTRIBUTE_TUTORIAL.md)

### 我要了解具体抽取流程
→ 查看 [guides/extract/](guides/extract/)

### 我要了解多Worker协作
→ 查看 [guides/extract/multi_worker_collaboration.md](guides/extract/multi_worker_collaboration.md)

### 我要了解并行执行最佳实践
→ 查看 [guides/extract/PARALLEL_EXECUTION_BEST_PRACTICES.md](guides/extract/PARALLEL_EXECUTION_BEST_PRACTICES.md) - 大规模并行任务执行经验总结

### 我要了解目录组织规范
→ 查看 [guides/extract/WORKER_DIRECTORY_GUIDE.md](guides/extract/WORKER_DIRECTORY_GUIDE.md) - Worker目录结构和组织方式

### 我要了解抽取常见问题与解决方案
→ 查看 [guides/extract/extraction_issues_and_solutions.md](guides/extract/extraction_issues_and_solutions.md) - 6大问题发现机制与处理经验（嵌套目录、空目录、进程监控等）

### 我要检查任务状态或恢复中断进程
→ 查看 [guides/WORKER_CHECKPOINT_GUIDE.md](guides/WORKER_CHECKPOINT_GUIDE.md) - Worker读档检查指南（进程状态、批次进度、恢复方法）

### 我要了解残图修复（辅助参考）
→ 查看 [guides/repair/](guides/repair/)

### 我要处理修复失败的案例
→ 查看 [FAILED_CASES_GUIDE.md](FAILED_CASES_GUIDE.md) - 系统性失败案例处理指南

### 我要了解文档更新情况
→ 查看 [DOCUMENT_AUDIT_REPORT.md](DOCUMENT_AUDIT_REPORT.md) - 文档审查报告

---

## 重要说明

**修复残图不是主要工作。** 我们的核心任务是使用正确的流程抽取新模型的计算图。

- **主要工作**：按标准流程抽取 HuggingFace 模型计算图
- **辅助工作**：修复已有残图（仅在必要时进行）

---

*最后更新：2026-04-02*

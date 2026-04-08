# GraphNetExtractAgent 快速入门指南

> 🚀 新Agent第一次接手项目？从这里开始！
>
> **阅读时间**: 5分钟
> **最后更新**: 2026-03-29

---

## 一分钟了解项目

**GraphNetExtractAgent** 是一个自动从各种深度学习框架（PyTorch/HuggingFace/timm等）提取计算图并生成GraphNet样本的项目。

### 核心任务流程
```
选择模型 → 编写提取脚本 → 执行extract → 验证 → 提交PR
```

---

## 第一步：了解当前状态

### 查看当前进度
```bash
# 查看最新任务状态
cat /root/GraphNetExtractAgent/docs/tasks/CURRENT_STATUS.md

# 查看工作记忆
cat /root/GraphNetExtractAgent/docs/references/MEMORY.md
```

### 关键信息速览
| 项目 | 状态 |
|------|------|
| 完整模型总数 | ~2,417个 |
| 待修复残图 | fill-mask: 78/79完成 |
| 当前优先任务 | 新模型抓取 |

---

## 第二步：选择你的任务

### 🎯 任务类型选择

| 如果你... | 选择这个任务 |
|-----------|-------------|
| 刚开始，想熟悉流程 | **修复残图** (简单，有现成脚本) |
| 想贡献新模型 | **HuggingFace模型提取** |
| 擅长处理边界情况 | **特殊模型处理** |

### 当前推荐任务
**✅ 立即开始：新模型抓取**

Worker2已完成残图修复，可以全力投入新模型抓取。

---

## 第三步：5分钟上手

### 场景1：修复残图（最简单）

```bash
# 1. 找到待修复的残图
find /ssd1/liangtai-work/graphnet_workspace/huggingface -type d -name "subgraph_*" \
  ! -exec test -f {}/../graph_hash.txt \; -print 2>/dev/null | head -5

# 2. 使用from_config修复脚本
cd /work/graphnet_workspace/huggingface
python3 fix_from_config.py \
  --list /tmp/residual_models.txt \
  --agent-id w2-fix-01 \
  --limit 10
```

**关键原则（祖训）**: 使用`from_config()`随机初始化，**不下载HF权重**！

### 场景2：抓取新模型

```bash
# 1. 选择batch文件
ls /tmp/new_extract_batches/*/

# 2. 启动subagent
cd /work/graphnet_workspace/huggingface
python3 extract_hf_full_v2.py \
  --batch-file /tmp/new_extract_batches/fm/batch_aa \
  --task fill-mask \
  --agent-id w2-new-01 \
  --max-length 128
```

---

## 第四步：关键概念

### 完整模型标准
```
完整模型 = graph_hash.txt + (根目录或subgraph下有model.py和weight_meta.py)
```

### 残图类型
1. **无graph_hash.txt** - 需要重新抓取
2. **空壳目录** - 有graph_hash但无model.py，需用from_config修复

### 祖训（必须遵守）
1. ✅ 使用`from_config()`随机初始化
2. ❌ 不下载HF权重文件
3. ✅ 只下载config.json (KB级)

---

## 第五步：遇到问题

### 常见问题速查

| 问题 | 解决方案 |
|------|---------|
| 模型加载失败 | 检查是否用了`from_config()`而不是`from_pretrained()` |
| 无config.json | 先从HF下载config.json |
| 显存不足 | 使用CPU: `CUDA_VISIBLE_DEVICES=""` |

### 求助资源
1. [residual_repair_guide.md](guides/residual_repair_guide.md) - 残图修复详解
2. [HUGGINGFACE_EXTRACTION_GUIDE.md](guides/HUGGINGFACE_EXTRACTION_GUIDE.md) - HF提取完整指南
3. [quick_reference.md](guides/quick_reference.md) - 命令速查

---

## 第六步：报告进度

### 汇报模板
```bash
echo "=== 进度汇报 ==="
echo "1) 运行中subagent: $(pgrep -c extract_hf_full) 个"
echo "2) 已修复/抓取: X 个模型"
echo "3) 遇到问题: [描述]"
echo "4) 下一步计划: [计划]"
```

---

## 快速导航

- 📚 [完整文档索引](README.md)
- 🎯 [当前任务状态](tasks/CURRENT_STATUS.md)
- 🔧 [残图修复指南](guides/residual_repair_guide.md)
- 📖 [HF提取指南](guides/HUGGINGFACE_EXTRACTION_GUIDE.md)
- 💡 [操作原理](guides/EXTRACTION_EXPLAINED.md)

---

## 检查清单

开始工作前，确认：

- [ ] 已阅读本文档
- [ ] 已查看当前任务状态
- [ ] 了解祖训（from_config不下载权重）
- [ ] 知道如何汇报进度

---

**准备好了吗？开始你的第一个任务吧！** 🚀

如有疑问，先查看[AGENT_CONTEXT.md](guides/AGENT_CONTEXT.md)了解项目全貌。

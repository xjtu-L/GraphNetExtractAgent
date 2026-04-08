# 多Worker协作流程指南

> 🤝 Worker1与Worker2任务协作的最佳实践
>
> **适用对象**: 所有参与GraphNetExtractAgent的Worker
> **重要性**: 必须遵守，避免重复工作

---

## 核心原则

**铁律**: 执行任何任务前必须检查任务列表，执行后必须立即更新状态！

**目录组织**: 每个Worker有自己的工作目录，模型按Task分类存放！

---

## 目录组织规范（重要）

### 新的目录结构

为避免多个Worker之间的冲突和数据混乱，采用以下目录组织方式：

```
graphnet_workspace/
└── huggingface/
    ├── worker0/                    # Worker0 的工作目录
    │   ├── text-generation/
    │   │   ├── model1_name/
    │   │   └── model2_name/
    │   ├── text-classification/
    │   ├── fill-mask/
    │   └── feature-extraction/
    │
    ├── worker1/                    # Worker1 的工作目录
    │   ├── text-generation/
    │   ├── text-classification/
    │   ├── fill-mask/
    │   └── feature-extraction/
    │
    ├── worker2/                    # Worker2 的工作目录
    │   ├── text-generation/
    │   ├── text-classification/
    │   ├── fill-mask/
    │   └── feature-extraction/
    │
    └── shared/                     # 共享目录（可选）
        └── common_resources/
```

### 目录规则

| 层级 | 命名规范 | 说明 |
|------|---------|------|
| Worker目录 | `worker{N}/` | N为Worker编号：0, 1, 2... |
| Task目录 | 使用HF任务类型名 | text-generation, text-classification, fill-mask... |
| 模型目录 | `org_name_model_name/` | /替换为_，如 facebook_xmod-base |

### 任务类型映射规则

根据模型名称自动判断应放入哪个Task目录：

| 模型特征关键词 | 任务类型 | 示例模型 |
|--------------|---------|---------|
| GPT, LLaMA, Qwen, Mistral, Phi, OPT, CausalLM, GPTNeoX | `text-generation` | gpt2, meta-llama/Llama-2-7b |
| 带 classification, sentiment, SequenceClassification | `text-classification` | bert-base-uncased-sentiment |
| BERT, RoBERTa, ELECTRA, DistilBERT（非分类版） | `fill-mask` | bert-base-uncased, roberta-base |
| T5, BART, text2text | `text2text-generation` | t5-small, facebook/bart-base |
| 其他编码器模型 | `feature-extraction` | sentence-transformers/all-MiniLM |

**自动分类脚本示例**：
```python
def infer_task_type(model_name):
    model_lower = model_name.lower()
    if any(k in model_lower for k in ['gpt', 'llama', 'qwen', 'mistral', 'phi', 'neox', 'opt', 'causallm']):
        return 'text-generation'
    elif any(k in model_lower for k in ['classification', 'sentiment', 'sequenceclassification']):
        return 'text-classification'
    elif any(k in model_lower for k in ['t5', 'bart', 'text2text']):
        return 'text2text-generation'
    elif any(k in model_lower for k in ['bert', 'roberta', 'electra', 'distilbert']):
        return 'fill-mask'
    return 'feature-extraction'
```

### 使用示例

**Worker0 抓取模型时**：
```bash
# 输出路径设置为 Worker0 的目录
export GRAPH_NET_EXTRACT_WORKSPACE="/ssd1/liangtai-work/graphnet_workspace/huggingface/worker0/text-generation"

# 抓取模型
python extract_script.py facebook/xmod-base

# 结果位置
# /ssd1/liangtai-work/graphnet_workspace/huggingface/worker0/text-generation/facebook_xmod-base/
```

**Worker1 抓取模型时**：
```bash
# 输出路径设置为 Worker1 的目录
export GRAPH_NET_EXTRACT_WORKSPACE="/ssd1/liangtai-work/graphnet_workspace/huggingface/worker1/text-generation"

# 抓取模型
python extract_script.py Xenova/ernie-2.0-large-en

# 结果位置
# /ssd1/liangtai-work/graphnet_workspace/huggingface/worker1/text-generation/Xenova_ernie-2.0-large-en/
```

### 为什么要这样组织

1. **避免冲突**: 多个Worker同时写入同一目录会导致数据混乱
2. **责任清晰**: 每个Worker对自己的目录负责，便于追踪问题
3. **便于统计**: 按Worker和Task分类，方便统计各Worker的贡献
4. **易于合并**: 后期可以将各Worker的结果合并到统一的数据库

### 旧目录结构的问题

```
# ❌ 旧结构（不推荐使用）
huggingface/
├── text-generation/
│   ├── model1/          # Worker0写入
│   ├── model2/          # Worker1写入（可能冲突）
│   └── model3/          # 不知道是谁抓的
├── text-classification/
└── ...
```

**问题**：
- 多个Worker同时写入同一目录，数据容易混乱
- 无法追踪哪个模型是谁抓取的
- 一个Worker的问题可能影响其他Worker

### 迁移指南

如果你之前使用旧结构，按以下步骤迁移：

```bash
# 1. 停止所有抓取任务
pkill -f extract

# 2. 创建新的Worker目录结构
mkdir -p /ssd1/liangtai-work/graphnet_workspace/huggingface/worker{N}/{text-generation,text-classification,fill-mask,feature-extraction}

# 3. 将已抓取的模型移动到自己的Worker目录（可选）
mv /ssd1/liangtai-work/graphnet_workspace/huggingface/text-generation/model1 \
   /ssd1/liangtai-work/graphnet_workspace/huggingface/worker0/text-generation/

# 4. 更新脚本中的输出路径
sed -i 's|huggingface/text-generation|huggingface/worker0/text-generation|g' extract_script.py
```

---

## 任务分配流程

### 1. 检查任务是否已被分配

```bash
# 方法1: 查看任务跟踪系统
cat /root/GraphNetExtractAgent/task_tracking/README.md

# 方法2: 查看待执行任务
ls -la /root/GraphNetExtractAgent/task_tracking/pending/

# 方法3: 查看进行中任务
ls -la /root/GraphNetExtractAgent/task_tracking/in_progress/

# 方法4: 查看最新状态文档
cat /root/GraphNetExtractAgent/docs/tasks/CURRENT_STATUS.md
```

### 2. 领取任务的步骤

```
Step 1: 检查 pending/ 目录，选择未分配的任务
    ↓
Step 2: 将任务文件移动到 in_progress/
    ↓
Step 3: 更新任务文件，添加执行者信息
    ↓
Step 4: 更新 CURRENT_STATUS.md
    ↓
Step 5: 开始执行任务
```

**示例**:
```bash
# 1. 领取任务
cd /root/GraphNetExtractAgent/task_tracking
mv pending/empty_dirs_reextract.txt in_progress/

# 2. 编辑任务文件，添加执行者
# 在文件中添加: 执行者: Worker2

# 3. 更新状态文档
# 编辑 CURRENT_STATUS.md，更新Worker状态
```

### 3. 避免多个Worker同时执行同一任务

#### 机制1: 文件锁（推荐）
```bash
# 创建锁文件
lock_file="/tmp/task_lock_${TASK_ID}"
if [ -f "$lock_file" ]; then
    echo "任务已被其他Worker领取"
    exit 1
fi
touch "$lock_file"
echo "Worker2 $(date)" > "$lock_file"
```

#### 机制2: 任务文件状态
```bash
# 检查任务状态
status=$(grep "状态:" task_file.txt | head -1)
if [[ "$status" == *"in_progress"* ]]; then
    echo "任务进行中，请勿重复执行"
    exit 1
fi
```

#### 机制3: Worker间沟通
```
Worker1: 领取任务前在群里/文档中声明
Worker2: 看到声明后不再领取同一任务
```

### 4. 任务状态实时同步

#### 实时同步策略

| 时机 | 操作 | 目标文件 |
|------|------|----------|
| 领取任务时 | 移动到in_progress/ | task_tracking/in_progress/ |
| 执行过程中 | 每30分钟更新进度 | CURRENT_STATUS.md |
| 完成时 | 移动到completed/ | task_tracking/completed/ |
| 失败时 | 移动到failed/ | task_tracking/failed/ |

#### 同步命令
```bash
# 快速更新任务状态
echo "=== 任务状态更新 ==="
echo "时间: $(date '+%Y-%m-%d %H:%M')"
echo "Worker: Worker2"
echo "任务: fill-mask残图修复"
echo "进度: 50/79 (63%)"
echo "预计完成: 2小时后"
```

#### 状态汇报模板
```
=== Worker2 进度汇报 ===
时间: 2026-03-29 16:00
运行中Subagent: 5个
当前任务: 空目录修复
进度: 100/542 (18%)
遇到问题: 无
下一步: 继续执行
```

---

## Worker间沟通协议

### 沟通渠道

1. **任务跟踪系统** (主要)
   - 位置: `task_tracking/`
   - 用途: 任务分配和状态同步
   - 更新频率: 实时

2. **当前状态文档** (次要)
   - 位置: `docs/tasks/CURRENT_STATUS.md`
   - 用途: 总体进度概览
   - 更新频率: 每30分钟

3. **直接沟通** (辅助)
   - 用途: 协调、问题讨论
   - 注意: 沟通结果必须同步到任务跟踪系统

### 沟通场景

#### 场景1: Worker1发现新任务
```
Worker1:
- 发现text-generation有100个残图需要修复
- 在pending/创建任务文件
- 更新CURRENT_STATUS.md
- （可选）通知Worker2有新任务
```

#### 场景2: Worker2想领取任务
```
Worker2:
- 检查pending/目录
- 发现Worker1创建的任务
- 确认无人执行后，移动到in_progress/
- 开始执行
- 更新CURRENT_STATUS.md
```

#### 场景3: 任务依赖
```
Worker1: 需要先完成A任务才能做B任务
Worker2: 想执行B任务

解决方案:
1. Worker1在A任务文件中标记"B任务依赖此任务"
2. Worker2检查依赖关系
3. Worker2等待A完成或协助Worker1完成A
```

---

## 任务交接流程

### 从Worker1交接给Worker2

```bash
# Step 1: Worker1准备交接材料
# - 任务描述文件
# - 相关脚本/数据位置
# - 已完成的部分
# - 遇到的问题

# Step 2: Worker1更新任务状态
cd /root/GraphNetExtractAgent/task_tracking
# 在任务文件中添加:
# 状态: waiting_handover
# 交接给: Worker2

# Step 3: Worker2确认接收
# 更新任务文件:
# 状态: in_progress
# 执行者: Worker2
# 开始时间: 2026-03-29 16:00

# Step 4: Worker2继续执行
```

---

## 常见问题处理

### 问题1: 两个Worker同时开始同一任务

**解决方案**:
```bash
# 1. 立即停止执行
# 2. 比较进度，保留进度快的
# 3. 另一个Worker认领其他任务
# 4. 更新任务跟踪系统，记录问题原因
```

### 问题2: 任务状态不同步

**解决方案**:
```bash
# 1. 以文件系统状态为准
# 2. 双方同步最新状态
# 3. 更新所有相关文档
```

### 问题3: 任务依赖冲突

**解决方案**:
```
1. 列出所有依赖关系
2. 协商执行顺序
3. 更新任务文件标记依赖
4. 按顺序执行
```

---

## 检查清单

### Worker1检查清单

- [ ] 创建任务时是否更新了pending/
- [ ] 是否标记了任务依赖关系
- [ ] 交接时是否提供了完整信息
- [ ] 是否及时更新了CURRENT_STATUS.md

### Worker2检查清单

- [ ] 领取任务前是否检查了in_progress/
- [ ] 是否创建了锁文件
- [ ] 执行中是否定期更新进度
- [ ] 完成后是否移动到completed/

---

## 附录

### 快速命令参考

```bash
# 查看所有待完成任务
cat /root/GraphNetExtractAgent/task_tracking/pending/*

# 查看进行中任务
cat /root/GraphNetExtractAgent/task_tracking/in_progress/*

# 查看最新状态
cat /root/GraphNetExtractAgent/docs/tasks/CURRENT_STATUS.md

# 创建新任务
touch /root/GraphNetExtractAgent/task_tracking/pending/my_task.txt

# 领取任务
mv /root/GraphNetExtractAgent/task_tracking/pending/my_task.txt \
   /root/GraphNetExtractAgent/task_tracking/in_progress/

# 完成任务
mv /root/GraphNetExtractAgent/task_tracking/in_progress/my_task.txt \
   /root/GraphNetExtractAgent/task_tracking/completed/
```

---

**最后更新**: 2026-03-29
**维护者**: Worker2
**待讨论**: 与Worker1确认此流程是否可行

# Worker协作任务执行流程

> **铁律**: 执行任何任务前必须先检查任务列表，执行后必须立即更新任务列表！
>
> **最后更新**: 2026-03-29

---

## 1. 任务状态系统

### 1.1 四种任务状态

```
pending（待处理） → in_progress（进行中） → completed（已完成）
                                      ↓
                                 failed（失败）
```

### 1.2 状态文件位置

```
task_tracking/
├── pending/          # 待处理任务
│   ├── repair.txt    # 待修复残图
│   ├── empty_dirs.txt # 待处理空目录
│   └── extraction.txt # 待抓取模型
├── in_progress/      # 进行中任务
│   └── worker_id_model_id.task  # 进行中标记文件
├── completed/        # 已完成任务
│   ├── repair.txt
│   └── extraction.txt
└── failed/           # 失败任务
    ├── repair.txt
    └── retry_queue.txt
```

---

## 2. 任务执行流程（铁律）

### 2.1 执行前检查（必须！）

```bash
# 步骤1: 检查任务是否已在进行中
ls /ssd1/liangtai-work/graphnet_workspace/task_tracking/in_progress/ | grep model_id
# 如果有输出 → 任务已被分配，跳过

# 步骤2: 检查任务是否已完成
grep "model_id" /ssd1/liangtai-work/graphnet_workspace/task_tracking/completed/*.txt
# 如果有输出 → 任务已完成，跳过

# 步骤3: 检查任务是否已失败
grep "model_id" /ssd1/liangtai-work/graphnet_workspace/task_tracking/failed/*.txt
# 如果有输出 → 任务曾失败，查看原因后决定是否重试

# 步骤4: 检查pending列表确认任务存在
grep "model_id" /ssd1/liangtai-work/graphnet_workspace/task_tracking/pending/*.txt
```

### 2.2 执行时标记（必须！）

```bash
# 创建进行中标记文件
worker_id="w1"  # Worker1用w1，Worker2用w2
task_type="repair"  # repair/extraction/empty_dirs
model_id="bert-base-uncased"
start_time=$(date +%Y-%m-%d_%H:%M:%S)
pid=$$

# 写入标记文件
echo "${worker_id}|${model_id}|${task_type}|${start_time}|${pid}|$(hostname)" \
  > /ssd1/liangtai-work/graphnet_workspace/task_tracking/in_progress/${worker_id}_${model_id}.${task_type}
```

### 2.3 执行后更新（必须！）

#### 成功时

```bash
# 步骤1: 删除进行中标记
rm /ssd1/liangtai-work/graphnet_workspace/task_tracking/in_progress/${worker_id}_${model_id}.${task_type}

# 步骤2: 添加到完成列表
echo "${task_type}|${model_id}|${worker_id}|$(date +%Y-%m-%d)" \
  >> /ssd1/liangtai-work/graphnet_workspace/task_tracking/completed/${task_type}.txt

# 步骤3: 从pending删除
sed -i "/${model_id}/d" /ssd1/liangtai-work/graphnet_workspace/task_tracking/pending/${task_type}.txt
```

#### 失败时

```bash
# 步骤1: 删除进行中标记
rm /ssd1/liangtai-work/graphnet_workspace/task_tracking/in_progress/${worker_id}_${model_id}.${task_type}

# 步骤2: 添加到失败列表（包含失败原因）
echo "${task_type}|${model_id}|${worker_id}|reason|$(date +%Y-%m-%d)" \
  >> /ssd1/liangtai-work/graphnet_workspace/task_tracking/failed/${task_type}.txt

# 步骤3: 从pending删除
sed -i "/${model_id}/d" /ssd1/liangtai-work/graphnet_workspace/task_tracking/pending/${task_type}.txt
```

---

## 3. 多Worker协作规则

### 3.1 任务分配策略

| Worker | 负责任务类型 | 优先级 |
|--------|-------------|--------|
| Worker1 | text-generation (689) + text-classification (299) | 高 |
| Worker2 | fill-mask (716) + 其他 (1375) | 高 |

### 3.2 避免冲突机制

#### 方法1: 文件锁（推荐）

```bash
# 获取任务前加锁
lock_file="/ssd1/liangtai-work/graphnet_workspace/task_tracking/.lock"

# 尝试获取锁（最多等待30秒）
for i in {1..30}; do
    if mkdir "$lock_file" 2>/dev/null; then
        # 获取锁成功
        break
    fi
    sleep 1
done

# 执行任务获取和标记...

# 释放锁
rmdir "$lock_file"
```

#### 方法2: 时间戳检查

```bash
# 检查标记文件时间戳
if [ -f "in_progress/${model_id}.task" ]; then
    file_time=$(stat -c %Y "in_progress/${model_id}.task")
    current_time=$(date +%s)
    diff=$((current_time - file_time))

    # 如果超过2小时，认为任务已失效
    if [ $diff -gt 7200 ]; then
        echo "任务标记超时，可以接管"
        rm "in_progress/${model_id}.task"
    fi
fi
```

### 3.3 实时同步方法

```bash
# 方法1: 定期轮询（每5分钟）
while true; do
    # 检查in_progress目录更新
    ls -lt in_progress/ | head -10

    # 同步任务
    # ...

    sleep 300
done

# 方法2: 信号通知（高级）
# Worker1完成任务后触发通知
# Worker2接收通知后更新本地状态
```

---

## 4. 实用脚本

### 4.1 检查任务状态

```bash
#!/bin/bash
# check_task.sh - 检查单个任务状态

model_id=$1
task_tracking="/ssd1/liangtai-work/graphnet_workspace/task_tracking"

echo "检查任务: $model_id"
echo ""

# 检查进行中
if ls ${task_tracking}/in_progress/*${model_id}* 1>/dev/null 2>&1; then
    echo "❌ 任务进行中:"
    cat ${task_tracking}/in_progress/*${model_id}*
    exit 1
fi

# 检查已完成
if grep -q "$model_id" ${task_tracking}/completed/*.txt 2>/dev/null; then
    echo "✅ 任务已完成:"
    grep "$model_id" ${task_tracking}/completed/*.txt
    exit 1
fi

# 检查失败
if grep -q "$model_id" ${task_tracking}/failed/*.txt 2>/dev/null; then
    echo "⚠️ 任务曾失败:"
    grep "$model_id" ${task_tracking}/failed/*.txt
    echo "如需重试，请手动移回pending"
    exit 1
fi

echo "✅ 任务可以开始"
exit 0
```

### 4.2 批量获取任务

```bash
#!/bin/bash
# get_batch_tasks.sh - 批量获取待处理任务

worker_id=$1
task_type=$2  # repair/empty_dirs/extraction
batch_size=${3:-10}
task_tracking="/ssd1/liangtai-work/graphnet_workspace/task_tracking"

# 获取lock
lock_file="${task_tracking}/.lock"
while ! mkdir "$lock_file" 2>/dev/null; do
    sleep 1
done

# 读取任务
tasks=$(head -n $batch_size "${task_tracking}/pending/${task_type}.txt")

# 标记为进行中
echo "$tasks" | while IFS= read -r line; do
    model_id=$(echo "$line" | cut -d'|' -f2)
    echo "${worker_id}|${model_id}|${task_type}|$(date +%Y-%m-%d_%H:%M:%S)|$$" \
      > "${task_tracking}/in_progress/${worker_id}_${model_id}.${task_type}"
done

# 释放lock
rmdir "$lock_file"

# 输出获取的任务列表
echo "$tasks"
```

### 4.3 更新任务状态

```bash
#!/bin/bash
# update_task_status.sh - 更新任务状态

model_id=$1
worker_id=$2
task_type=$3
status=$4  # completed/failed
reason=$5  # 失败原因（可选）
task_tracking="/ssd1/liangtai-work/graphnet_workspace/task_tracking"

# 删除进行中标记
rm -f "${task_tracking}/in_progress/${worker_id}_${model_id}.${task_type}"

if [ "$status" = "completed" ]; then
    # 添加到完成列表
    echo "${task_type}|${model_id}|${worker_id}|$(date +%Y-%m-%d)" \
      >> "${task_tracking}/completed/${task_type}.txt"
    echo "✅ 任务标记为完成"
else
    # 添加到失败列表
    echo "${task_type}|${model_id}|${worker_id}|${reason}|$(date +%Y-%m-%d)" \
      >> "${task_tracking}/failed/${task_type}.txt"
    echo "⚠️ 任务标记为失败"
fi

# 从pending删除
sed -i "/${model_id}/d" "${task_tracking}/pending/${task_type}.txt"
```

---

## 5. 日常检查清单

### 每日开始工作前

- [ ] 检查in_progress目录，清理超时任务（>2小时）
- [ ] 确认自己的任务分配
- [ ] 检查pending任务数量

### 执行每个任务前

- [ ] 检查任务是否已在进行中
- [ ] 检查任务是否已完成
- [ ] 检查任务是否已失败
- [ ] 创建进行中标记文件

### 执行每个任务后

- [ ] 删除进行中标记
- [ ] 更新completed或failed列表
- [ ] 从pending删除

### 每日结束工作前

- [ ] 确认所有进行中任务已更新状态
- [ ] 生成今日工作报告
- [ ] 同步任务状态给另一Worker

---

## 6. 常见问题

### Q1: 两个Worker同时获取了同一任务？

A: 使用文件锁机制避免。如果发生冲突，比较时间戳，后开始的Worker放弃。

### Q2: 任务进行中但Worker崩溃了？

A: 标记文件会保留。另一Worker在2小时后可检查并接管（视为超时）。

### Q3: 如何批量获取一批任务？

A: 使用get_batch_tasks.sh脚本，一次获取10个任务，批量标记为进行中。

### Q4: 失败后想重试怎么办？

A: 手动将任务从failed移到pending：
```bash
grep "model_id" failed/repair.txt >> pending/repair.txt
sed -i "/model_id/d" failed/repair.txt
```

---

## 7. 与Worker2沟通确认

### 待确认事项

1. **任务分配** - Worker2负责哪些任务类型？
2. **锁机制** - 是否使用文件锁？超时时间设为多久？
3. **同步频率** - 多长时间同步一次状态？
4. **失败处理** - 失败任务何时重试？

### 建议流程

1. Worker1和Worker2各自维护自己的in_progress标记
2. 每小时互相通报进度
3. 每天汇总completed和failed数量
4. 发现问题立即沟通解决

---

## 8. 相关文档

- [WORKFLOW.md](../../task_tracking/WORKFLOW.md) - 任务流程详细说明
- [README.md](../../task_tracking/README.md) - 任务跟踪系统说明
- [residual_repair_guide.md](residual_repair_guide.md) - 残图修复指南

---

**记住：执行前检查！执行时标记！执行后更新！**

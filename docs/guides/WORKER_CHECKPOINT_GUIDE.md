# Worker 读档检查指南

> 用于检查抽取任务状态、恢复中断进程的快速参考手册
>
> **创建时间**: 2026-04-02

---

## 1. 检查进程状态

### 快速检查

```bash
# 检查Worker1进程数
ps aux | grep extract_from_model_list | grep worker1 | grep -v grep | wc -l

# 检查所有抽取进程详情
ps aux | grep extract_from_model_list | grep -v grep | while read line; do
  pid=$(echo "$line" | awk '{print $2}')
  idx=$(echo "$line" | grep -oE '\-\-start-idx [0-9]+|\-\-end-idx [0-9]+' | tr '\n' ' ')
  output=$(echo "$line" | grep -oE '\-\-output-dir [^ ]+' | cut -d' ' -f2)
  echo "PID $pid: $idx -> $output"
done
```

### 预期输出示例

```
PID 161386: --start-idx 0 --end-idx 20000  -> /root/graphnet_workspace/huggingface/worker1
PID 161390: --start-idx 20000 --end-idx 40000  -> /root/graphnet_workspace/huggingface/worker1
PID 161403: --start-idx 40000 --end-idx 60000  -> /root/graphnet_workspace/huggingface/worker1
PID 161420: --start-idx 60000 --end-idx 74860  -> /root/graphnet_workspace/huggingface/worker1

Worker1进程数: 4
```

### 正常状态

| Worker | 预期进程数 | 批次范围 |
|--------|-----------|---------|
| Worker1 | 4 | 0-74860 (分4批) |
| Worker2 | 40 (通过监控) | 74861-149719 |

### 关键组件

| 组件 | 正常状态 | 检查命令 | 关键文件 |
|------|---------|---------|---------|
| 监控进程 | 运行中 | `pgrep -f "monitor_extraction.py"` | `/tmp/monitor_extraction.pid` |
| 抽取进程 | 30-40个 | `pgrep -f "extract_from_model_list.py" \| wc -l` | - |
| 监控日志 | 持续更新 | `tail -5 /tmp/monitor_extraction.log` | `/tmp/monitor_extraction.log` |

### 监控进程检查 (Worker2)

```bash
# 检查监控进程
pgrep -f "monitor_extraction.py" && echo "运行中" || echo "未运行"

# 检查监控日志
tail -10 /tmp/monitor_extraction.log

# 检查PID文件
cat /tmp/monitor_extraction.pid
```

---

## 2. 任务列表和黑名单位置

### 主任务列表

| 文件 | 路径 | 说明 |
|------|------|------|
| 主列表 | `/root/GraphNetExtractAgent/data/model_lists/our_models_300k.txt` | 149,719 个模型 |
| 新模型 | `/root/GraphNetExtractAgent/data/model_lists/new_models.txt` | 补充模型 |
| 其他模型 | `/root/GraphNetExtractAgent/data/model_lists/others_models_300k.txt` | 备用列表 |

### 黑名单/淘汰记录

| 文件 | 路径 | 说明 |
|------|------|------|
| 淘汰模型 | `/root/GraphNetExtractAgent/data/eliminated_models.txt` | 记录失败/排除的模型及原因 |

### 查看淘汰记录

```bash
# 查看淘汰总数
wc -l /root/GraphNetExtractAgent/data/eliminated_models.txt

# 查看最近淘汰
 tail -20 /root/GraphNetExtractAgent/data/eliminated_models.txt

# 统计淘汰原因
cut -f2 /root/GraphNetExtractAgent/data/eliminated_models.txt | sort | uniq -c | sort -rn
```

---

## 3. 确认当前批次进度

### 查看日志进度

```bash
# Worker1 各批次日志
for log in /tmp/w1_*.log; do
  if [ -f "$log" ]; then
    echo ">>> $log"
    tail -5 "$log" 2>/dev/null | grep -E "^\[|模型索引|正在抓取" | tail -3
    echo ""
  fi
done
```

### 单个批次查看

```bash
# 查看特定批次最新进度
tail -20 /tmp/w1_0_20k.log    # 0-20000批次
tail -20 /tmp/w1_20k_40k.log  # 20000-40000批次
tail -20 /tmp/w1_40k_60k.log  # 40000-60000批次
tail -20 /tmp/w1_60k_74k.log  # 60000-74860批次
```

### 统计当前完成数量

```bash
# Worker1 模型目录数
find /root/graphnet_workspace/huggingface/worker1 -mindepth 1 -maxdepth 1 -type d ! -empty | wc -l

# Worker1 计算图数量
find /root/graphnet_workspace/huggingface/worker1 -name "graph_hash.txt" | wc -l

# 一行输出
echo "Worker1: 目录=$(find /root/graphnet_workspace/huggingface/worker1 -mindepth 1 -maxdepth 1 -type d ! -empty | wc -l) 计算图=$(find /root/graphnet_workspace/huggingface/worker1 -name 'graph_hash.txt' | wc -l)"
```

---

## 3. 任务列表位置

### 主模型列表

| 文件 | 路径 | 数量 |
|------|------|------|
| 主列表 | `/root/GraphNetExtractAgent/data/model_lists/our_models_300k.txt` | 149,719 |
| 新模型 | `/root/GraphNetExtractAgent/data/model_lists/new_models.txt` | - |
| 其他模型 | `/root/GraphNetExtractAgent/data/model_lists/others_models_300k.txt` | - |

### Worker 责任范围

| Worker | 索引范围 | 模型数 |
|--------|---------|-------|
| Worker1 | 行 0-74860 | 74,861 |
| Worker2 | 行 74861-149719 | 74,858 |

### 待修复列表

```bash
# Worker1 空目录修复列表
/tmp/empty_dirs_worker1.txt  # 271个

# Worker2 空目录修复列表
/tmp/empty_dirs_worker2.txt  # 269个
```

### 查看模型列表

```bash
# 查看列表总行数
wc -l /root/GraphNetExtractAgent/data/model_lists/our_models_300k.txt

# 查看特定行
sed -n '1p' /root/GraphNetExtractAgent/data/model_lists/our_models_300k.txt      # 第1个模型
sed -n '74860p' /root/GraphNetExtractAgent/data/model_lists/our_models_300k.txt  # Worker1最后一个
sed -n '74861p' /root/GraphNetExtractAgent/data/model_lists/our_models_300k.txt  # Worker2第一个
```

### 输出目录结构

| 目录 | 用途 |
|------|------|
| `/root/graphnet_workspace/huggingface/worker1` | Worker1 输出目录 |
| `/root/graphnet_workspace/huggingface/worker2` | Worker2 输出目录 |
| `/root/graphnet_workspace/hash_analysis/` | Hash 分析结果 |
| `/root/graphnet_workspace/hash_collection/` | 收集的完整 hash 集合 |

---

## 4. 恢复中断的进程

## 4. 恢复中断的进程

### 场景一：监控进程停止

**症状**：
- `pgrep -f "monitor_extraction.py"` 无输出
- 抽取进程数逐渐减少至 0
- 日志停止更新

**恢复步骤**：

```bash
# 1. 检查监控进程状态
pgrep -f "monitor_extraction.py"

# 2. 如果停止，清理残留进程（可选）
pkill -f "extract_from_model_list.py"

# 3. 重新启动监控
nohup python3 /root/monitor_extraction.py > /tmp/monitor_restart.log 2>&1 &

# 4. 验证启动成功
sleep 2 && pgrep -f "monitor_extraction.py"
```

### 场景二：抽取进程数过低但监控运行中

**症状**：
- 监控进程存在
- 抽取进程数 < 30
- 日志显示"进程数过低"且"所有模型已分配"

**解决方法**：
- 这是正常现象，监控正在等待当前批次完成
- 如需加速，可手动启动额外进程：

```bash
# 手动启动补充批次
python3 /root/extract_from_model_list.py --start-idx <idx> --end-idx <idx+300> --output-dir /root/graphnet_workspace/huggingface/worker2
```

### 场景三：fill-mask 重新抽取中断

```bash
# 1. 检查 fill-mask 抽取进程
ps aux | grep "extract_from_model_list.py" | grep fill-mask

# 2. 清理空目录（可选）
find /root/graphnet_workspace/huggingface/fill-mask -type d -empty -delete

# 3. 重新启动
mkdir -p /tmp/reextract_fill_mask_logs
for i in {0..4}; do
    nohup python3 /tmp/reextract_fill_mask.py $i 50 > /tmp/reextract_fill_mask_logs/worker${i}_main.log 2>&1 &
done
```

### 快速恢复脚本

```bash
#!/bin/bash
# Worker1 恢复脚本

export http_proxy=http://agent.baidu.com:8891
export https_proxy=http://agent.baidu.com:8891
export HF_ENDPOINT=https://hf-mirror.com

# 启动4个批次
nohup python3 /root/extract_from_model_list.py --start-idx 0 --end-idx 20000 --output-dir /root/graphnet_workspace/huggingface/worker1 > /tmp/w1_0_20k.log 2>&1 &
echo "启动批次 0-20000, PID: $!"

nohup python3 /root/extract_from_model_list.py --start-idx 20000 --end-idx 40000 --output-dir /root/graphnet_workspace/huggingface/worker1 > /tmp/w1_20k_40k.log 2>&1 &
echo "启动批次 20000-40000, PID: $!"

nohup python3 /root/extract_from_model_list.py --start-idx 40000 --end-idx 60000 --output-dir /root/graphnet_workspace/huggingface/worker1 > /tmp/w1_40k_60k.log 2>&1 &
echo "启动批次 40000-60000, PID: $!"

nohup python3 /root/extract_from_model_list.py --start-idx 60000 --end-idx 74860 --output-dir /root/graphnet_workspace/huggingface/worker1 > /tmp/w1_60k_74k.log 2>&1 &
echo "启动批次 60000-74860, PID: $!"

sleep 2
echo "Worker1进程数: $(ps aux | grep extract_from_model_list | grep worker1 | grep -v grep | wc -l)"
```

### 恢复单个批次

```bash
# 只恢复特定批次
nohup python3 /root/extract_from_model_list.py \
  --start-idx 0 \
  --end-idx 20000 \
  --output-dir /root/graphnet_workspace/huggingface/worker1 \
  > /tmp/w1_0_20k.log 2>&1 &
```

### Worker2 恢复脚本

```bash
#!/bin/bash
# Worker2 恢复脚本

export http_proxy=http://agent.baidu.com:8891
export https_proxy=http://agent.baidu.com:8891
export HF_ENDPOINT=https://hf-mirror.com

nohup python3 /root/extract_from_model_list.py --start-idx 74861 --end-idx 100000 --output-dir /root/graphnet_workspace/huggingface/worker2 > /tmp/w2_74k_100k.log 2>&1 &
nohup python3 /root/extract_from_model_list.py --start-idx 100000 --end-idx 125000 --output-dir /root/graphnet_workspace/huggingface/worker2 > /tmp/w2_100k_125k.log 2>&1 &
nohup python3 /root/extract_from_model_list.py --start-idx 125000 --end-idx 149719 --output-dir /root/graphnet_workspace/huggingface/worker2 > /tmp/w2_125k_149k.log 2>&1 &
```

### 验证恢复成功

```bash
# 检查进程是否启动
ps aux | grep extract_from_model_list | grep worker1 | grep -v grep

# 检查日志是否有输出
tail -5 /tmp/w1_0_20k.log
```

---

## 5. 日志查看方法

### 日志文件位置

| 批次 | 日志文件 |
|------|---------|
| Worker1 0-20k | `/tmp/w1_0_20k.log` |
| Worker1 20k-40k | `/tmp/w1_20k_40k.log` |
| Worker1 40k-60k | `/tmp/w1_40k_60k.log` |
| Worker1 60k-74k | `/tmp/w1_60k_74k.log` |

### 实时监控

```bash
# 实时查看单个批次
tail -f /tmp/w1_0_20k.log

# 实时查看所有批次（多窗口建议使用tmux）
tail -f /tmp/w1_*.log
```

### 查看错误

```bash
# 查看最近错误
grep -E "错误|失败|Error|Failed" /tmp/w1_*.log | tail -20

# 统计错误数量
grep -c "错误" /tmp/w1_*.log
```

### 查看成功数量

```bash
# 统计成功抽取数量
grep -c "成功" /tmp/w1_*.log

# 查看最近成功
grep "成功" /tmp/w1_*.log | tail -10
```

### 日志关键字

| 关键字 | 含义 |
|--------|------|
| `正在抓取` | 当前处理的模型 |
| `[N/M]` | 进度 (第N个/共M个) |
| `✓ 成功` | 抽取成功 |
| `✗ 错误` | 抽取失败 |
| `跳过（已存在）` | 已抽取过，跳过 |
| `过滤: 跳过` | 不支持的格式(GGUF等) |

---

## 5. 监控配置参数

`/root/monitor_extraction.py` 关键配置：

```python
TARGET_PARALLEL = 40      # 目标并行进程数
MIN_PARALLEL = 30         # 最低进程数阈值
CHECK_INTERVAL = 60       # 检查间隔（秒）
BATCH_SIZE = 300          # 每批次模型数
TOTAL_MODELS = 149719     # 总模型数
MODEL_LIST_FILE = "/root/GraphNetExtractAgent/data/model_lists/our_models_300k.txt"
OUTPUT_DIR = "/root/graphnet_workspace/huggingface/worker2"
```

---

## 6. 故障排查

### 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| 进程数持续下降 | 任务完成/异常退出 | 检查日志，重启监控 |
| "所有模型已分配"但无进程 | 索引已到达 TOTAL_MODELS | 更新 TOTAL_MODELS 或检查模型列表 |
| 监控日志停止更新 | 监控进程卡死 | 重启监控进程 |
| 大量空目录 | 抽取失败未清理 | 运行清理脚本 |

---

## 7. 新Worker如何读档

当新的Worker Agent接管任务时，按以下流程快速恢复工作：

### Step 1: 读取Checkpoint文件

```bash
# 查看可用的checkpoint文件
ls -la /root/GraphNetExtractAgent/docs/checkpoints/

# 读取对应的checkpoint文件
cat /root/GraphNetExtractAgent/docs/checkpoints/worker1_checkpoint.md  # Worker1
cat /root/GraphNetExtractAgent/docs/checkpoints/worker2_checkpoint.md  # Worker2
```

### Step 2: 检查进程状态

```bash
# 快速检查当前进程数
ps aux | grep extract_from_model_list | grep worker1 | grep -v grep | wc -l

# 查看详细进程信息
ps aux | grep extract_from_model_list | grep -v grep | while read line; do
  pid=$(echo "$line" | awk '{print $2}')
  idx=$(echo "$line" | grep -oE '\-\-start-idx [0-9]+|\-\-end-idx [0-9]+' | tr '\n' ' ')
  echo "PID $pid: $idx"
done
```

### Step 3: 如需恢复，执行Checkpoint中的恢复命令

从checkpoint文件中复制恢复命令并执行：

```bash
# 示例：Worker1恢复命令（从checkpoint复制）
export http_proxy=http://agent.baidu.com:8891
export https_proxy=http://agent.baidu.com:8891
export HF_ENDPOINT=https://hf-mirror.com

nohup python3 /root/extract_from_model_list.py --start-idx 0 --end-idx 20000 --output-dir /root/graphnet_workspace/huggingface/worker1 > /tmp/w1_0_20k.log 2>&1 &
nohup python3 /root/extract_from_model_list.py --start-idx 20000 --end-idx 40000 --output-dir /root/graphnet_workspace/huggingface/worker1 > /tmp/w1_20k_40k.log 2>&1 &
nohup python3 /root/extract_from_model_list.py --start-idx 40000 --end-idx 60000 --output-dir /root/graphnet_workspace/huggingface/worker1 > /tmp/w1_40k_60k.log 2>&1 &
nohup python3 /root/extract_from_model_list.py --start-idx 60000 --end-idx 74860 --output-dir /root/graphnet_workspace/huggingface/worker1 > /tmp/w1_60k_74k.log 2>&1 &
```

### Step 4: 更新Checkpoint为新Worker信息

恢复后，更新checkpoint文件记录新状态：

```bash
# 使用文本编辑器更新checkpoint
nano /root/GraphNetExtractAgent/docs/checkpoints/worker1_checkpoint.md

# 或使用脚本自动更新（推荐）
cat > /tmp/update_checkpoint.sh << 'EOF'
#!/bin/bash
CHECKPOINT_FILE="/root/GraphNetExtractAgent/docs/checkpoints/worker1_checkpoint.md"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M')
DIRS=$(find /root/graphnet_workspace/huggingface/worker1 -mindepth 1 -maxdepth 1 -type d ! -empty | wc -l)
GRAPHS=$(find /root/graphnet_workspace/huggingface/worker1 -name 'graph_hash.txt' | wc -l)
PROCS=$(ps aux | grep extract_from_model_list | grep worker1 | grep -v grep | wc -l)

echo "更新时间: $TIMESTAMP"
echo "模型目录: $DIRS"
echo "计算图数: $GRAPHS"
echo "进程数: $PROCS"

# 更新文件中的关键数值
sed -i "s/快照时间.*/快照时间: $TIMESTAMP/" $CHECKPOINT_FILE
sed -i "s/模型目录 | \*\*.*\*\*/模型目录 | **$DIRS**/" $CHECKPOINT_FILE
sed -i "s/计算图 (graph_hash.txt) | \*\*.*\*\*/计算图 (graph_hash.txt) | **$GRAPHS**/" $CHECKPOINT_FILE
EOF

chmod +x /tmp/update_checkpoint.sh
/tmp/update_checkpoint.sh
```

### 读档流程图

```
┌─────────────────────────────────────────────────────────┐
│                   新Worker读档流程                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. 读取Checkpoint文件                                   │
│     └─→ cat docs/checkpoints/worker1_checkpoint.md      │
│                                                         │
│  2. 检查进程状态                                         │
│     └─→ ps aux | grep extract_from_model_list           │
│                                                         │
│  3. 判断是否需要恢复                                     │
│     ├─→ 进程数=0: 需要恢复，执行Step 3                   │
│     └─→ 进程数>0: 正常运行，跳到Step 4                   │
│                                                         │
│  4. 更新Checkpoint                                      │
│     └─→ 记录新时间、进度、进程信息                        │
│                                                         │
│  5. 继续监控任务进度                                     │
│     └─→ tail -f /tmp/w1_*.log                          │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 快速读档命令（一键执行）

```bash
#!/bin/bash
# 一键读档脚本 - Worker1

echo "=== Worker1 读档检查 ==="
echo ""

# 1. 显示checkpoint关键信息
echo ">>> Checkpoint摘要:"
head -30 /root/GraphNetExtractAgent/docs/checkpoints/worker1_checkpoint.md | grep -E "快照时间|模型目录|计算图|进程数"

echo ""

# 2. 检查当前进程
PROCS=$(ps aux | grep extract_from_model_list | grep worker1 | grep -v grep | wc -l)
echo ">>> 当前进程数: $PROCS"

if [ "$PROCS" -eq 0 ]; then
    echo ">>> 需要恢复进程，执行恢复命令..."
    # 这里可以添加自动恢复逻辑
else
    echo ">>> 进程正常运行中"
fi

echo ""

# 3. 显示当前进度
DIRS=$(find /root/graphnet_workspace/huggingface/worker1 -mindepth 1 -maxdepth 1 -type d ! -empty | wc -l)
GRAPHS=$(find /root/graphnet_workspace/huggingface/worker1 -name 'graph_hash.txt' | wc -l)
echo ">>> 当前进度: 目录=$DIRS 计算图=$GRAPHS"
```

---

## 快速检查命令汇总

```bash
# 一键检查 Worker1 状态
echo "=== $(date +%H:%M) Worker1 状态 ===" && \
echo "进程数: $(ps aux | grep extract_from_model_list | grep worker1 | grep -v grep | wc -l)" && \
echo "目录数: $(find /root/graphnet_workspace/huggingface/worker1 -mindepth 1 -maxdepth 1 -type d ! -empty | wc -l)" && \
echo "计算图: $(find /root/graphnet_workspace/huggingface/worker1 -name 'graph_hash.txt' | wc -l)"
```

---

*最后更新: 2026-04-02*

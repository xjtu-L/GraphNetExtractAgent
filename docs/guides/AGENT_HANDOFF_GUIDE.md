# Agent 工作交接指南 (Agent Handoff Guide)

> **最后更新**: 2026-03-31
> **适用场景**: HuggingFace 模型抽取任务的 Agent 工作交接与故障恢复

---

## 1. 内存阈值设为 100GB 的经验

### 1.1 背景
在 HuggingFace 模型抽取过程中，我们发现不同的模型大小会导致内存使用量差异巨大：

| 模型类型 | 典型内存使用 | 说明 |
|---------|-------------|------|
| 小型模型 (<1B) | 2-10 GB | 正常范围 |
| 中型模型 (1B-7B) | 10-30 GB | 需要监控 |
| 大型模型 (7B-70B) | 50-200+ GB | 高风险 |

### 1.2 阈值演进
- **初始**: 64GB (83886080 KB)
- **优化后**: 100GB (104857600 KB)

### 1.3 选择 100GB 的理由
1. **平衡稳定性与效率**: 64GB 过于保守，会误杀处理大型模型的正常进程
2. **系统资源充足**: 服务器总内存可支持多个 100GB 进程并行
3. **实际观察数据**: 大多数异常进程超过 200GB，100GB 是合理警戒线

### 1.4 监控脚本配置
```bash
#!/bin/bash
# monitor_memory.sh - Worker1 内存监控
while true; do
  for pid in $(pgrep -f "extract_from_model_list" | head -10); do
    mem_kb=$(cat /proc/$pid/status 2>/dev/null | grep VmRSS | awk "{print \$2}")
    if [ -n "$mem_kb" ] && [ "$mem_kb" -gt 104857600 ]; then
      echo "$(date): PID $pid using ${mem_kb}KB (>100GB), killing..." >> monitor.log
      kill -9 $pid 2>/dev/null
    fi
  done
  sleep 60
done
```

### 1.5 最佳实践
- **定期检查**: `cat monitor.log | tail -20`
- **阈值调整**: 根据任务特性可调整至 80GB-120GB
- **配套措施**: 使用 `ulimit -v 16777216` 限制虚拟内存为 64GB

---

## 2. 新 Agent 继承工作和接管任务的流程方法

### 2.1 接管检查清单

#### 步骤 1: 环境确认
```bash
cd /ssd1/liangtai-work/graphnet_workspace/huggingface/worker1
echo "当前目录: $(pwd)"
echo "磁盘空间: $(df -h . | tail -1)"
```

#### 步骤 2: 进程状态检查
```bash
# 检查活跃抽取进程
ps aux | grep "extract_from_model_list" | grep "worker1" | grep -v grep

# 检查监控进程
ps aux | grep -E "monitor_memory|monitor_subagent" | grep -v grep
```

#### 步骤 3: 进度统计
```bash
# 已完成模型数
find . -name "graph_net.json" | wc -l

# 待处理模型数
find . -name "extract.py" | wc -l

# 模型目录总数
ls -1 | wc -l
```

### 2.2 工作继承流程

```
新 Agent 启动
    ↓
1. 读取 /root/.claude/projects/-root/memory/MEMORY.md
    ↓
2. 检查活跃进程 → 确认索引范围
    ↓
3. 统计完成进度 → graph_net.json 数量
    ↓
4. 验证监控脚本 → 确保内存保护
    ↓
5. 识别缺失批次 → 补充或重启
    ↓
6. 定期汇报 → 已抓取/运行中/内存状态
```

### 2.3 关键文件位置

| 文件/目录 | 路径 | 用途 |
|----------|------|------|
| 工作目录 | `/ssd1/liangtai-work/graphnet_workspace/huggingface/worker1` | 模型抽取主目录 |
| 监控脚本 | `./monitor_memory.sh` | 内存监控 |
| 日志文件 | `./monitor.log` | 监控日志 |
| 内存日志 | `/tmp/worker1_mem.log` | 内存使用记录 |
| 抽取脚本 | `/root/extract_from_model_list.py` | 主抽取程序 |

### 2.4 常用接管命令
```bash
# 快速状态检查
echo "Worker1 Report:" && \
  find . -name "graph_net.json" | wc -l && echo "completed" && \
  find . -name "extract.py" | wc -l && echo "pending" && \
  ps aux | grep -E "(extract|subagent)" | grep -v grep | wc -l && echo "processes"

# 内存状态检查
ps aux | grep "extract_from_model_list" | grep "worker1" | \
  awk '{printf "PID %s: CPU=%s%% MEM=%s%% RSS=%.1fGB\n", $2, $3, $4, $6/1024/1024}'
```

### 2.5 交接报告模板
```
=== Worker1 状态汇报 ===
1) 已抓取模型数: [X] 个 graph_net.json
2) 运行中subagent数: [Y] 个
3) 当前批次: [列出索引范围]
4) 内存状态: [总内存]GB, 最高 [单进程内存]GB
```

---

## 3. 今天的故障处理经验 (2026-03-31)

### 3.1 故障现象
**时间**: 2026-03-31 22:56-23:00
**问题**: Worker1 多个抽取进程内存占用过高，最高达 350GB

### 3.2 诊断过程

#### 发现
```bash
ps aux --sort=-%mem | head -20
```
发现：
- PID 30695: 350GB (worker2)
- PID 141104: 216GB (worker1)
- PID 74718: 176GB (worker1)

#### 根因分析
1. **无内存限制**: 抽取进程未设置 ulimit
2. **大型模型加载**: 处理 7B+ 参数模型时内存激增
3. **监控阈值过低**: 原 64GB 阈值过于保守

### 3.3 处理步骤

#### 步骤 1: 立即停止高内存进程
```bash
kill -9 74718 141104
```

#### 步骤 2: 使用内存限制重启
```bash
cd /ssd1/liangtai-work/graphnet_workspace/huggingface/worker1
ulimit -v 16777216  # 64GB 虚拟内存限制
nohup python3 -u /root/extract_from_model_list.py \
  --start-idx 15000 --end-idx 18750 \
  --output-dir /ssd1/liangtai-work/graphnet_workspace/huggingface/worker1 \
  > /tmp/batch_15000_18750.log 2>&1 &
```

#### 步骤 3: 更新监控脚本
```bash
# 更新阈值为 100GB (104857600 KB)
pkill -f monitor_memory.sh
cat > monitor_memory.sh << 'EOF'
#!/bin/bash
while true; do
  for pid in $(pgrep -f "extract_from_model_list" | head -10); do
    mem_kb=$(cat /proc/$pid/status 2>/dev/null | grep VmRSS | awk "{print \$2}")
    if [ -n "$mem_kb" ] && [ "$mem_kb" -gt 104857600 ]; then
      echo "$(date): PID $pid using ${mem_kb}KB (>100GB), killing..." >> monitor.log
      kill -9 $pid 2>/dev/null
    fi
  done
  sleep 60
done
EOF
chmod +x monitor_memory.sh
nohup ./monitor_memory.sh >> monitor.log 2>&1 &
```

### 3.4 故障恢复验证
- ✅ 已杀死 3 个高内存进程
- ✅ 已重启 2 个批次 (11250-15000, 15000-18750)
- ✅ 监控脚本更新为 100GB 阈值
- ✅ 补充重启缺失批次 (15000-18750)

### 3.5 经验总结

#### 预防性措施
1. **始终设置 ulimit**: 新进程启动时配置虚拟内存限制
2. **合理监控阈值**: 100GB 是平衡点和安全线
3. **定期检查**: 每小时检查一次内存状态

#### 应急处理
1. **快速识别**: `ps aux --sort=-%mem | head -20`
2. **安全重启**: kill 后立即使用 ulimit 限制重启
3. **日志记录**: 所有操作记录到 monitor.log

#### 监控脚本最佳实践
- 运行多个监控层:
  - `monitor_memory.sh` - 杀死超过阈值的进程
  - `monitor_subagent.py` (PID 36907-36916) - 空闲备用
  - `/tmp/mem_monitor.sh` - 日志记录

---

## 附录 A: 快速参考命令

```bash
# 查看所有 worker1 进程
ps aux | grep "worker1" | grep -v grep

# 查看内存最高的 5 个进程
ps aux --sort=-%mem | head -6

# 统计完成情况
find /ssd1/liangtai-work/graphnet_workspace/huggingface/worker1 -name "graph_net.json" | wc -l

# 查看监控日志
tail -f /ssd1/liangtai-work/graphnet_workspace/huggingface/worker1/monitor.log

# 重启监控脚本
pkill -f monitor_memory.sh
cd /ssd1/liangtai-work/graphnet_workspace/huggingface/worker1
nohup ./monitor_memory.sh >> monitor.log 2>&1 &
```

## 附录 B: 索引范围分配

| Worker | 索引范围 | 状态 |
|--------|----------|------|
| Worker1 | 3750-7500 | 运行中 |
| Worker1 | 7500-11250 | 运行中 |
| Worker1 | 11250-15000 | 运行中 |
| Worker1 | 15000-18750 | 运行中 (已重启) |
| Worker1 | 33750-37500 | 运行中 |

---

*本文档由 Ducc (达克) 于 2026-03-31 创建，用于指导后续 Agent 接管 HuggingFace 模型抽取任务。*

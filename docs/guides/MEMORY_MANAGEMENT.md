# HuggingFace 模型抓取内存管理最佳实践

> 基于 Worker2 实战经验总结
> 最后更新: 2026-03-31

## 问题背景

在处理大规模 HuggingFace 模型抓取任务时（300K+ 模型），内存泄漏是常见挑战。单个进程在处理数千个模型后，内存占用可能从几百 MB 增长到数百 GB。

## 内存问题根因

1. **PyTorch 计算图累积** - 前向传播后梯度计算图未完全释放
2. **Python 对象循环引用** - gc 无法自动回收
3. **CUDA 缓存未清空** - 即使使用 CPU，PyTorch 仍会分配 CUDA 相关缓存
4. **Transformers 模型缓存** - AutoConfig 和 AutoModel 的内部缓存

## 解决方案

### 方案一: 内存监控守护进程 (推荐)

最简单有效的方案，无需修改业务代码。

```bash
#!/bin/bash
# monitor_memory.sh
while true; do
  for pid in $(pgrep -f "extract_from_model_list" | head -10); do
    mem_kb=$(cat /proc/$pid/status 2>/dev/null | grep VmRSS | awk "{print \$2}")
    if [ -n "$mem_kb" ] && [ "$mem_kb" -gt 83886016 ]; then  # 64GB阈值
      echo "$(date): PID $pid using ${mem_kb}KB (>64GB), killing..." >> monitor.log
      kill -9 $pid 2>/dev/null
    fi
  done
  sleep 60
done
```

**使用方法:**
```bash
chmod +x monitor_memory.sh
nohup ./monitor_memory.sh >> monitor.log 2>&1 &
```

**Worker2 实战数据:**
- 阈值: 64GB (67,108,864 KB)
- 检查间隔: 60秒
- 已处理超标进程: 3个 (最大334GB)

---

### 方案二: 子进程隔离

每个模型在独立子进程中处理，进程退出后内存完全释放。

```python
import subprocess
import sys

def extract_model_with_isolation(model_dir, timeout=180):
    """在子进程中运行提取，完全隔离内存"""
    extract_path = os.path.join(model_dir, "extract.py")

    result = subprocess.run(
        [sys.executable, extract_path],
        cwd=model_dir,
        capture_output=True,
        text=True,
        timeout=timeout
    )

    # 子进程退出 = 内存完全释放
    return result.returncode == 0
```

**优点:**
- 100% 内存回收
- 超时保护
- 崩溃隔离

**缺点:**
- 进程启动开销 (~100-500ms)
- 需要序列化通信

---

### 方案三: 激进的内存清理

在单个进程内最大化内存回收。

```python
import gc
import torch

def aggressive_cleanup():
    """强制激进的内存清理"""
    # 删除所有张量引用
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                del obj
        except:
            pass

    # 多次GC pass处理循环引用
    gc.collect()
    gc.collect()

    # 清理CUDA缓存
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
```

**使用时机:**
```python
# 每个模型处理后调用
for model_id in model_list:
    extract_model(model_id)
    aggressive_cleanup()  # 防止内存累积
```

---

### 方案四: 定期进程重启

处理N个模型后主动退出，由监督器重启。

```python
def process_batch(models, batch_size=50):
    """处理一批模型后重启"""
    processed = 0

    for model_id in models:
        extract_model(model_id)
        processed += 1

        if processed >= batch_size:
            print(f"已处理 {batch_size} 个模型，重启进程...")
            sys.exit(0)  # 监督器将使用下一个批次重启
```

**监督器示例:**
```bash
#!/bin/bash
# supervisor.sh
START_IDX=0
BATCH_SIZE=50

while [ $START_IDX -lt 300000 ]; do
    END_IDX=$((START_IDX + BATCH_SIZE))
    python3 extract.py --start-idx $START_IDX --end-idx $END_IDX
    START_IDX=$END_IDX
    echo "进程退出，准备下一批次..."
done
```

---

### 方案五: ulimit 硬性限制

设置系统级内存上限，超标即杀死。

```bash
# 设置虚拟内存限制 (160GB)
ulimit -v 41943040

# 启动工作进程
python3 extract_from_model_list.py \
    --start-idx 94216 \
    --end-idx 96350 \
    --output-dir /workspace/worker2
```

**注意:** ulimit 限制的是虚拟内存(VSZ)，而非物理内存(RSS)。

---

## 内存管理策略对比

| 策略 | 内存回收效果 | 实施难度 | 性能影响 | 适用场景 |
|------|-------------|----------|----------|----------|
| 监控守护进程 | ★★★★☆ | 低 | 无 | 所有场景 |
| 子进程隔离 | ★★★★★ | 中 | 中 | 高可靠性要求 |
| 激进GC清理 | ★★★☆☆ | 低 | 低 | 快速修复 |
| 定期重启 | ★★★★★ | 中 | 低 | 长时间任务 |
| ulimit限制 | ★★★☆☆ | 极低 | 无 | 紧急保护 |

## 推荐组合方案

**生产环境推荐:**
1. **监控守护进程** (必开) - 防止内存无限增长
2. **ulimit限制** - 系统级保护
3. **定期重启** - 主动内存回收

**开发环境推荐:**
1. **子进程隔离** - 便于调试，内存问题不影响主进程
2. **激进GC清理** - 快速验证修复效果

## 监控与告警

### 实时监控命令

```bash
# 查看内存占用Top 10
ps aux --sort=-%mem | head -10

# 持续监控特定进程
watch -n 5 'ps aux | grep extract_from_model_list | grep -v grep'

# 查看系统内存状态
free -h

# 查看进程详细内存 (KB)
cat /proc/PID/status | grep -E "VmRSS|VmSize"
```

### 内存使用阈值建议

| 系统内存 | 单进程限制 | 监控阈值 | 动作 |
|----------|-----------|----------|------|
| 512GB | 100GB | 80GB | 警告 |
| 1TB | 200GB | 150GB | 杀死重启 |
| 1.5TB | 300GB | 200GB | 杀死重启 |

## Worker2 实战经验

### 运行状态 (2026-03-31)
- **已处理模型:** 3,876个
- **成功提取:** 10,795个 graph_net.json
- **活跃批次:** 8个并行进程
- **内存使用:** 416GB / 1.5TB (27%)

### 内存泄漏案例

**问题进程:**
- PID 30695: 内存增长至 334GB
- PID 28262: 内存增长至 93GB

**解决:** 监控脚本自动检测并杀死，释放内存。

### 关键配置

```bash
# 环境变量
export HF_ENDPOINT=https://hf-mirror.com
export http_proxy=http://agent.baidu.com:8891
export https_proxy=http://agent.baidu.com:8891
export CUDA_VISIBLE_DEVICES=""  # 强制CPU模式

# 内存限制
ulimit -v 41943040  # 160GB
```

## 常见问题

### Q: 为什么 gc.collect() 不能完全回收内存？
A: PyTorch 有内部的内存池和缓存机制，需要配合 `torch.cuda.empty_cache()` 使用。

### Q: 子进程方案太慢怎么办？
A: 可以增大每个子进程处理的模型数（如100-200个），平衡内存和性能。

### Q: 监控脚本误杀怎么办？
A: 调整阈值，或加入白名单机制（排除特定PID）。

## 参考资源

- PyTorch Memory Management: https://pytorch.org/docs/stable/notes/cuda.html#memory-management
- Python GC Module: https://docs.python.org/3/library/gc.html
- Linux ulimit: https://ss64.com/bash/ulimit.html

---

*Worker2 实战经验总结 - 祝您抓取顺利！*

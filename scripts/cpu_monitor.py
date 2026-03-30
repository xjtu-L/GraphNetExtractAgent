#!/usr/bin/env python3
"""
CPU负载监控脚本 - 简单的系统状态查看

使用方法:
    python3 cpu_monitor.py              # 单次检查
    python3 cpu_monitor.py --watch      # 持续监控
    python3 cpu_monitor.py --interval 5 # 每5秒检查一次
"""

import psutil
import time
import argparse
import os
from datetime import datetime


def get_system_status():
    """获取系统状态"""
    # CPU
    cpu_percent = psutil.cpu_percent(interval=1)
    cpu_count = psutil.cpu_count()
    load_avg = os.getloadavg() if hasattr(os, 'getloadavg') else (0, 0, 0)

    # 内存
    memory = psutil.virtual_memory()

    # 磁盘
    disk = psutil.disk_usage('/')

    # 任务数
    import subprocess
    try:
        result = subprocess.run(
            ["pgrep", "-f", "repair|extract"],
            capture_output=True,
            text=True
        )
        task_count = len([p for p in result.stdout.strip().split('\n') if p])
    except:
        task_count = 0

    return {
        "cpu_percent": cpu_percent,
        "cpu_count": cpu_count,
        "load_avg": load_avg,
        "memory_percent": memory.percent,
        "memory_used_gb": memory.used / (1024**3),
        "memory_total_gb": memory.total / (1024**3),
        "disk_percent": disk.percent,
        "disk_free_gb": disk.free / (1024**3),
        "task_count": task_count
    }


def print_status(status):
    """打印系统状态"""
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
    print("=" * 60)
    print(f"CPU 使用率: {status['cpu_percent']:.1f}% / {status['cpu_count']} 核")
    print(f"负载均衡: {status['load_avg'][0]:.2f}, {status['load_avg'][1]:.2f}, {status['load_avg'][2]:.2f}")
    print(f"内存使用: {status['memory_percent']:.1f}% ({status['memory_used_gb']:.1f} / {status['memory_total_gb']:.1f} GB)")
    print(f"磁盘使用: {status['disk_percent']:.1f}% (剩余 {status['disk_free_gb']:.1f} GB)")
    print(f"运行任务: {status['task_count']} 个")

    # 建议
    print("-" * 60)
    if status['cpu_percent'] > 85:
        print("⚠️  CPU负载过高! 建议减少并行任务数")
    elif status['cpu_percent'] < 40:
        print("✅ CPU负载较低, 可以增加并行任务数")
    else:
        print("✅ CPU负载正常")

    if status['memory_percent'] > 90:
        print("⚠️  内存使用过高! 建议立即减少任务")
    elif status['memory_percent'] > 80:
        print("⚠️  内存使用较高, 需要监控")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="CPU负载监控")
    parser.add_argument("--watch", "-w", action="store_true", help="持续监控")
    parser.add_argument("--interval", "-i", type=int, default=10, help="检查间隔(秒)")
    args = parser.parse_args()

    if args.watch:
        print(f"开始持续监控 (间隔: {args.interval}秒), 按Ctrl+C停止...")
        try:
            while True:
                status = get_system_status()
                print_status(status)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n监控已停止")
    else:
        status = get_system_status()
        print_status(status)


if __name__ == "__main__":
    main()

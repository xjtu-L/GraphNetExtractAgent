#!/usr/bin/env python3
"""
Worker2监控脚本 - 维持至少50个subagent运行
任务类型: fill-mask/FE/t2t/ASR/translation
"""

import os
import sys
import json
import time
import subprocess
from datetime import datetime

def count_running_processes():
    """统计当前运行的extract_hf进程数"""
    result = subprocess.run(
        "ps aux | grep extract_hf | grep -v grep | wc -l",
        shell=True, capture_output=True, text=True
    )
    try:
        return int(result.stdout.strip())
    except:
        return 0

def get_process_details():
    """获取进程详情"""
    result = subprocess.run(
        "ps aux | grep extract_hf | grep -v grep",
        shell=True, capture_output=True, text=True
    )
    return result.stdout

def count_models_by_task():
    """统计各任务类型的已抓取模型数"""
    base = "/work/graphnet_workspace/huggingface"
    tasks = {
        "fill-mask": "fill-mask",
        "feature-extraction": "feature-extraction",
        "translation": "translation",
        "text-generation": "text-generation",
        "automatic-speech-recognition": "automatic-speech-recognition",
    }
    stats = {}
    for task, dirname in tasks.items():
        path = os.path.join(base, dirname)
        if os.path.exists(path):
            count = len([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])
            stats[task] = count
        else:
            stats[task] = 0
    return stats

def log_status():
    """记录状态日志"""
    count = count_running_processes()
    stats = count_models_by_task()

    print(f"\n{'='*60}")
    print(f"[Worker2 状态报告] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    print(f"运行中进程: {count} 个")
    print(f"目标: >= 50 个")
    status = "✓ 正常" if count >= 50 else "✗ 需要补充"
    print(f"状态: {status}")
    print(f"\n已抓取模型统计:")
    for task, num in sorted(stats.items()):
        print(f"  {task}: {num}")
    print(f"{'='*60}\n")

    return count, stats

def main():
    print("="*60)
    print("Worker2 监控系统启动")
    print("任务: 维持 >= 50 个subagent")
    print("类型: fill-mask/FE/t2t/ASR/translation")
    print("="*60)

    # 初始状态
    count, stats = log_status()

    # 如果不足50个，报警
    if count < 50:
        print(f"\n⚠️ 警告: 当前只有 {count} 个进程，需要补充!")
        print("请执行批量启动脚本补充进程\n")
        return 1
    else:
        print(f"\n✓ 进程数量正常: {count} 个")

    # 持续监控模式
    if "--watch" in sys.argv:
        interval = 60  # 默认60秒检查一次
        print(f"\n进入监控模式 (每 {interval} 秒检查一次)...")
        print("按 Ctrl+C 退出\n")

        try:
            while True:
                count, stats = log_status()
                if count < 50:
                    print(f"⚠️ 进程数不足 ({count} < 50)，需要立即补充!")
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n\n监控已停止")

    return 0

if __name__ == "__main__":
    sys.exit(main())

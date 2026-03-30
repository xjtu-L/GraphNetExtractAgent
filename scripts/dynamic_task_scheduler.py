#!/usr/bin/env python3
"""
动态任务调度器 - 根据CPU负载自动调整并行任务数量

功能：
1. 监控CPU使用率
2. 根据负载动态启动/停止任务
3. 保持系统稳定运行

使用方法：
    python3 dynamic_task_scheduler.py --max-tasks 10 --min-tasks 3 --target-cpu 70
"""

import os
import sys
import json
import time
import psutil
import signal
import subprocess
from datetime import datetime
from collections import deque

# 默认配置
DEFAULT_CONFIG = {
    "max_tasks": 10,           # 最大并行任务数
    "min_tasks": 3,            # 最小并行任务数
    "target_cpu": 70,          # 目标CPU使用率(%)
    "high_cpu_threshold": 85,  # 高负载阈值
    "low_cpu_threshold": 40,   # 低负载阈值
    "check_interval": 30,      # 检查间隔(秒)
    "task_batch_size": 20,     # 每批任务处理模型数
}

# 任务类型配置
TASK_TYPES = {
    "repair": {
        "script": "/tmp/standard_repair_v3.py",
        "batch_prefix": "/tmp/residual_batch_",
        "log_prefix": "/tmp/repair_batch_",
        "active_tasks": [],  # 当前运行的任务进程
    },
    "extract": {
        "script": "/tmp/new_model_extract_batch.py",
        "batch_prefix": "/tmp/new_extract_batch_",
        "log_prefix": "/tmp/new_extract_subagent_",
        "active_tasks": [],
    }
}


class DynamicTaskScheduler:
    def __init__(self, config=None):
        self.config = config or DEFAULT_CONFIG
        self.running = False
        self.cpu_history = deque(maxlen=5)  # 保存最近5次CPU读数
        self.current_tasks = {task_type: 0 for task_type in TASK_TYPES}

        # 设置信号处理
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """处理退出信号"""
        print(f"\n[{datetime.now()}] 收到信号 {signum}, 正在停止...")
        self.running = False

    def get_cpu_usage(self):
        """获取CPU使用率（多核平均）"""
        cpu_percent = psutil.cpu_percent(interval=1)
        self.cpu_history.append(cpu_percent)

        # 计算平滑后的CPU使用率（移动平均）
        if len(self.cpu_history) > 0:
            smoothed_cpu = sum(self.cpu_history) / len(self.cpu_history)
        else:
            smoothed_cpu = cpu_percent

        return {
            "current": cpu_percent,
            "smoothed": smoothed_cpu,
            "history": list(self.cpu_history),
            "load_avg": os.getloadavg() if hasattr(os, 'getloadavg') else (0, 0, 0)
        }

    def get_system_status(self):
        """获取系统整体状态"""
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        return {
            "cpu": self.get_cpu_usage(),
            "memory": {
                "total_gb": memory.total / (1024**3),
                "used_gb": memory.used / (1024**3),
                "percent": memory.percent
            },
            "disk": {
                "total_gb": disk.total / (1024**3),
                "free_gb": disk.free / (1024**3),
                "percent": disk.percent
            }
        }

    def count_running_tasks(self, task_type):
        """统计正在运行的任务数"""
        script_name = os.path.basename(TASK_TYPES[task_type]["script"])
        try:
            result = subprocess.run(
                ["pgrep", "-f", script_name],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                pids = [p for p in result.stdout.strip().split('\n') if p]
                return len(pids)
            return 0
        except:
            return 0

    def start_task(self, task_type, batch_id):
        """启动一个新任务"""
        task_config = TASK_TYPES[task_type]
        script = task_config["script"]
        batch_file = f"{task_config['batch_prefix']}{batch_id}.json"
        log_file = f"{task_config['log_prefix']}{batch_id}.log"

        # 检查批次文件是否存在
        if not os.path.exists(batch_file):
            print(f"  批次文件不存在: {batch_file}")
            return False

        # 检查任务是否已完成（日志中存在"完成"）
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                content = f.read()
                if "完成" in content or "成功" in content:
                    print(f"  任务已完成: {task_type} batch {batch_id}")
                    return False

        # 启动任务
        try:
            env = os.environ.copy()
            env["http_proxy"] = "http://agent.baidu.com:8891"
            env["https_proxy"] = "http://agent.baidu.com:8891"
            env["HF_ENDPOINT"] = "https://hf-mirror.com"

            process = subprocess.Popen(
                ["python3", script, batch_file, str(batch_id)],
                stdout=open(log_file, 'a'),
                stderr=subprocess.STDOUT,
                env=env,
                preexec_fn=os.setsid  # 创建新进程组
            )

            TASK_TYPES[task_type]["active_tasks"].append({
                "batch_id": batch_id,
                "pid": process.pid,
                "start_time": datetime.now()
            })

            print(f"  启动任务: {task_type} batch {batch_id} (PID: {process.pid})")
            return True

        except Exception as e:
            print(f"  启动失败: {e}")
            return False

    def stop_task(self, task_type, batch_id=None):
        """停止一个任务"""
        task_config = TASK_TYPES[task_type]

        if batch_id is not None:
            # 停止指定批次
            for task in task_config["active_tasks"]:
                if task["batch_id"] == batch_id:
                    try:
                        os.killpg(os.getpgid(task["pid"]), signal.SIGTERM)
                        print(f"  停止任务: {task_type} batch {batch_id} (PID: {task['pid']})")
                    except:
                        pass
                    task_config["active_tasks"].remove(task)
                    return True
        else:
            # 停止最近启动的任务
            if task_config["active_tasks"]:
                task = task_config["active_tasks"].pop()
                try:
                    os.killpg(os.getpgid(task["pid"]), signal.SIGTERM)
                    print(f"  停止任务: {task_type} batch {task['batch_id']} (PID: {task['pid']})")
                    return True
                except:
                    pass

        return False

    def adjust_tasks(self, status):
        """根据系统状态调整任务数"""
        cpu = status["cpu"]["smoothed"]
        memory = status["memory"]["percent"]

        print(f"\n[{datetime.now()}] 系统状态:")
        print(f"  CPU: {cpu:.1f}% (目标: {self.config['target_cpu']}%)")
        print(f"  内存: {memory:.1f}%")
        print(f"  当前任务: repair={self.current_tasks['repair']}, extract={self.current_tasks['extract']}")

        # 计算总任务数
        total_tasks = sum(self.current_tasks.values())

        # 决策逻辑
        if cpu > self.config["high_cpu_threshold"] or memory > 90:
            # 高负载：减少任务
            if total_tasks > self.config["min_tasks"]:
                # 优先减少extract任务（CPU密集型）
                if self.current_tasks["extract"] > 0:
                    print(f"  [高负载] 减少extract任务")
                    if self.stop_task("extract"):
                        self.current_tasks["extract"] -= 1
                elif self.current_tasks["repair"] > 1:
                    print(f"  [高负载] 减少repair任务")
                    if self.stop_task("repair"):
                        self.current_tasks["repair"] -= 1
            else:
                print(f"  [高负载] 已达到最小任务数，保持当前配置")

        elif cpu < self.config["low_cpu_threshold"]:
            # 低负载：增加任务
            if total_tasks < self.config["max_tasks"]:
                # 优先增加repair任务（离线，更稳定）
                if self.current_tasks["repair"] < 5:
                    batch_id = len(TASK_TYPES["repair"]["active_tasks"])
                    print(f"  [低负载] 增加repair任务")
                    if self.start_task("repair", batch_id):
                        self.current_tasks["repair"] += 1
                elif self.current_tasks["extract"] < 6:
                    batch_id = len(TASK_TYPES["extract"]["active_tasks"])
                    print(f"  [低负载] 增加extract任务")
                    if self.start_task("extract", batch_id):
                        self.current_tasks["extract"] += 1
            else:
                print(f"  [低负载] 已达到最大任务数")

        else:
            print(f"  [正常] 保持当前配置")

    def run(self):
        """主运行循环"""
        print("=" * 70)
        print("动态任务调度器启动")
        print(f"配置: max={self.config['max_tasks']}, min={self.config['min_tasks']}, target_cpu={self.config['target_cpu']}%")
        print("=" * 70)

        self.running = True
        iteration = 0

        # 初始启动最小任务数
        print(f"\n[{datetime.now()}] 初始启动任务...")
        for _ in range(self.config["min_tasks"] // 2 + 1):
            self.start_task("repair", self.current_tasks["repair"])
            self.current_tasks["repair"] += 1

        for _ in range(self.config["min_tasks"] // 2):
            self.start_task("extract", self.current_tasks["extract"])
            self.current_tasks["extract"] += 1

        while self.running:
            iteration += 1
            print(f"\n{'=' * 70}")
            print(f"第 {iteration} 轮检查")
            print('=' * 70)

            # 获取系统状态
            status = self.get_system_status()

            # 更新当前任务计数
            for task_type in TASK_TYPES:
                self.current_tasks[task_type] = self.count_running_tasks(task_type)

            # 调整任务
            self.adjust_tasks(status)

            # 打印当前运行的任务
            print(f"\n[{datetime.now()}] 当前运行任务:")
            for task_type in TASK_TYPES:
                count = self.count_running_tasks(task_type)
                print(f"  {task_type}: {count} 个")

            # 等待下一次检查
            print(f"\n  等待 {self.config['check_interval']} 秒...")
            time.sleep(self.config["check_interval"])

        # 清理
        print(f"\n[{datetime.now()}] 正在清理任务...")
        for task_type in TASK_TYPES:
            for task in TASK_TYPES[task_type]["active_tasks"][:]:
                try:
                    os.killpg(os.getpgid(task["pid"]), signal.SIGTERM)
                except:
                    pass

        print("调度器已停止")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="动态任务调度器")
    parser.add_argument("--max-tasks", type=int, default=10, help="最大并行任务数")
    parser.add_argument("--min-tasks", type=int, default=3, help="最小并行任务数")
    parser.add_argument("--target-cpu", type=int, default=70, help="目标CPU使用率")
    parser.add_argument("--check-interval", type=int, default=30, help="检查间隔(秒)")
    parser.add_argument("--daemon", action="store_true", help="后台运行")

    args = parser.parse_args()

    config = {
        "max_tasks": args.max_tasks,
        "min_tasks": args.min_tasks,
        "target_cpu": args.target_cpu,
        "check_interval": args.check_interval,
        "high_cpu_threshold": 85,
        "low_cpu_threshold": 40,
    }

    scheduler = DynamicTaskScheduler(config)

    if args.daemon:
        # 后台运行
        pid = os.fork()
        if pid > 0:
            print(f"调度器已后台运行 (PID: {pid})")
            sys.exit(0)

    try:
        scheduler.run()
    except KeyboardInterrupt:
        print("\n用户中断，正在停止...")
        scheduler.running = False


if __name__ == "__main__":
    main()

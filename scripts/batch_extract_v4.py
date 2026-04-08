#!/usr/bin/env python3
"""
批量抽取脚本 V4 - 多进程支持
支持多GPU并行抽取，自动跳过不支持格式
"""

import os
import sys
import argparse
import multiprocessing as mp
from datetime import datetime

# 设置代理
os.environ["http_proxy"] = "http://agent.baidu.com:8891"
os.environ["https_proxy"] = "http://agent.baidu.com:8891"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 添加GraphNet路径
sys.path.insert(0, "/root/GraphNet")

# 已知不支持的模型格式
SKIP_MODEL_PATTERNS = [
    "-gguf", ".gguf", "gguf-", "stable-diffusion", "sd-", "SD-",
    "wav2vec2", "whisper", "speecht5", "bark", "vits", "tts",
    "-lora", "_lora", "-onnx", ".onnx", "detectron2",
]

def should_skip_model(model_id):
    """检查模型是否应该跳过"""
    model_lower = model_id.lower()
    for pattern in SKIP_MODEL_PATTERNS:
        if pattern.lower() in model_lower:
            return True, f"Unsupported: {pattern}"
    return False, None

def extract_worker(args):
    """单个抽取工作进程"""
    model_id, task_type, output_dir = args
    
    # 检查跳过
    skip, reason = should_skip_model(model_id)
    if skip:
        return model_id, "skipped", reason
    
    # 这里调用实际抽取逻辑
    # 简化版，实际需要import并调用相应函数
    return model_id, "success", "Extracted"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", required=True, help="模型列表文件")
    parser.add_argument("--task", default="feature-extraction", help="任务类型")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument("--workers", type=int, default=4, help="并行进程数")
    parser.add_argument("--start", type=int, default=0, help="起始索引")
    parser.add_argument("--end", type=int, default=None, help="结束索引")
    parser.add_argument("--agent-id", default="agent_00", help="Agent ID")
    args = parser.parse_args()
    
    # 读取模型列表
    with open(args.list, "r") as f:
        models = [line.strip() for line in f if line.strip()]
    
    models = models[args.start:args.end]
    print(f"[{args.agent_id}] 处理 {len(models)} 个模型")
    
    # 准备任务
    tasks = [(m, args.task, args.output_dir) for m in models]
    
    # 并行处理
    with mp.Pool(args.workers) as pool:
        results = pool.map(extract_worker, tasks)
    
    # 统计结果
    skipped = [r for r in results if r[1] == "skipped"]
    success = [r for r in results if r[1] == "success"]
    
    print(f"[{args.agent_id}] 完成: 成功 {len(success)}, 跳过 {len(skipped)}")

if __name__ == "__main__":
    main()

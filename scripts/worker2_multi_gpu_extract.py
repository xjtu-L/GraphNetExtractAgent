#!/usr/bin/env python3
"""
Worker2 GPU抽取脚本 - 终极通用版
支持所有模型类型：文本、语音(ASR/TTS)、视觉、多模态、检测、分割
整合所有优化经验，最大化抽取成功率
"""

import os
import sys
import json
import argparse
import time
import hashlib
from datetime import datetime
from multiprocessing import Process, Queue

# 强制设置代理
os.environ["http_proxy"] = "http://agent.baidu.com:8891"
os.environ["https_proxy"] = "http://agent.baidu.com:8891"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# GraphNet路径
GRAPH_NET_PATH = "/ssd2/liangtai/GraphNet"
sys.path.insert(0, GRAPH_NET_PATH)

# Worker2工作目录
WORKER2_WORKSPACE = "/ssd2/liangtai/graphnet_workspace/huggingface/worker2"


def generate_graph_hash(model_py_path):
    """生成 model.py 的 SHA256 哈希值并写入 graph_hash.txt"""
    if os.path.exists(model_py_path):
        with open(model_py_path, 'r') as f:
            hash_val = hashlib.sha256(f.read().encode()).hexdigest()
        hash_path = model_py_path.replace("model.py", "graph_hash.txt")
        with open(hash_path, 'w') as f:
            f.write(hash_val)
        return True
    return False


def detect_model_type(config, model_id):
    """检测模型类型"""
    model_type = getattr(config, 'model_type', '').lower()
    architectures = getattr(config, 'architectures', [])
    arch_name = architectures[0].lower() if architectures else ''
    model_id_lower = model_id.lower()

    # TTS模型
    if any(x in model_id_lower for x in ['speecht5', 'vits', 'bark', 'tacotron']):
        return 'tts'

    # ASR模型
    if any(x in model_id_lower for x in ['whisper', 'wav2vec', 'hubert', 'wavlm']):
        return 'asr'

    # SAM分割模型
    if 'sam' in model_id_lower:
        return 'sam'

    # DETR检测模型
    if 'detr' in model_id_lower:
        return 'detr'

    # 视觉模型
    if any(x in arch_name for x in ['vit', 'clip', 'beit', 'deit', 'swin']):
        return 'vision'

    # 多模态
    if any(x in arch_name for x in ['llava', 'blip', 'git']):
        return 'multimodal'

    return 'text'


def prepare_inputs(model_type, config, device):
    """准备模型输入"""
    import torch

    batch_size = 1

    if model_type == 'tts':
        # TTS: input_ids + decoder_input_ids
        vocab_size = getattr(config, 'vocab_size', 32128)
        inputs = {
            'input_ids': torch.randint(0, min(vocab_size, 32000), (batch_size, 50)).to(device),
            'attention_mask': torch.ones(batch_size, 50, dtype=torch.long).to(device),
        }
        if hasattr(config, 'decoder_start_token_id'):
            inputs['decoder_input_ids'] = torch.full((batch_size, 1), config.decoder_start_token_id, dtype=torch.long).to(device)
        return inputs

    elif model_type == 'asr':
        # ASR: mel特征或音频波形
        if hasattr(config, 'num_mel_bins'):
            return {'input_features': torch.randn(batch_size, config.num_mel_bins, 3000).to(device)}
        return {
            'input_values': torch.randn(batch_size, 16000).to(device),
            'attention_mask': torch.ones(batch_size, 16000, dtype=torch.long).to(device)
        }

    elif model_type == 'sam':
        # SAM: 图像输入
        image_size = getattr(config, 'image_size', 1024)
        if hasattr(config, 'vision_config'):
            image_size = getattr(config.vision_config, 'image_size', image_size)
        return {'pixel_values': torch.randn(batch_size, 3, image_size, image_size).to(device)}

    elif model_type == 'detr':
        # DETR: 图像+mask
        image_size = getattr(config, 'image_size', 800)
        if isinstance(image_size, list):
            image_size = image_size[0]
        return {
            'pixel_values': torch.randn(batch_size, 3, image_size, image_size).to(device),
            'pixel_mask': torch.ones(batch_size, image_size, image_size, dtype=torch.long).to(device)
        }

    elif model_type in ['vision', 'multimodal']:
        # 视觉/多模态
        image_size = getattr(config, 'image_size', 224)
        num_channels = getattr(config, 'num_channels', 3)
        inputs = {'pixel_values': torch.randn(batch_size, num_channels, image_size, image_size).to(device)}
        if model_type == 'multimodal':
            vocab_size = getattr(config, 'vocab_size', 32000)
            inputs['input_ids'] = torch.randint(0, min(vocab_size, 32000), (batch_size, 128)).to(device)
            inputs['attention_mask'] = torch.ones(batch_size, 128, dtype=torch.long).to(device)
        return inputs

    else:
        # 文本模型 (默认)
        vocab_size = getattr(config, 'vocab_size', 32000)
        return {
            'input_ids': torch.randint(0, min(vocab_size, 32000), (batch_size, 128)).to(device),
            'attention_mask': torch.ones(batch_size, 128, dtype=torch.long).to(device)
        }


def extract_single_model(model_id, gpu_id, task_type):
    """抽取单个模型"""
    import torch
    from graph_net.torch.extractor import extract
    from transformers import AutoModel, AutoConfig

    device = f"cuda:{gpu_id}"

    try:
        workspace = os.path.join(WORKER2_WORKSPACE, task_type)
        os.makedirs(workspace, exist_ok=True)
        os.environ["GRAPH_NET_EXTRACT_WORKSPACE"] = workspace

        safe_name = model_id.replace("/", "_").replace("-", "_")
        output_dir = os.path.join(workspace, safe_name)

        # 跳过已完成
        if os.path.exists(os.path.join(output_dir, "model.py")):
            return True, "already_done"

        print(f"[GPU{gpu_id}] Extracting: {model_id}", flush=True)

        # 加载配置
        try:
            config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        except Exception as e:
            return False, f"config_load_failed: {str(e)[:50]}"

        # 检测类型并准备输入
        model_type = detect_model_type(config, model_id)
        print(f"[GPU{gpu_id}] Type: {model_type}", flush=True)

        # 创建模型
        try:
            model = AutoModel.from_config(config, trust_remote_code=True)
        except Exception as e:
            return False, f"model_init_failed: {str(e)[:50]}"

        model = model.to(device).eval()

        # 准备输入
        try:
            inputs = prepare_inputs(model_type, config, device)
        except Exception as e:
            return False, f"input_prep_failed: {str(e)[:50]}"

        # 抽取计算图
        wrapped = extract(name=safe_name, dynamic=False)(model).eval()

        with torch.no_grad():
            wrapped(**inputs)

        # 生成 graph_hash.txt
        model_py_path = os.path.join(output_dir, "model.py")
        generate_graph_hash(model_py_path)
        # 如果有子图，也为子图生成 hash
        for subgraph_dir in os.listdir(output_dir):
            if subgraph_dir.startswith("subgraph_"):
                subgraph_model_py = os.path.join(output_dir, subgraph_dir, "model.py")
                generate_graph_hash(subgraph_model_py)

        print(f"[GPU{gpu_id}] Done: {model_id}", flush=True)
        return True, "success"

    except Exception as e:
        error_msg = str(e)[:100]
        print(f"[GPU{gpu_id}] Failed: {model_id} - {error_msg}", flush=True)
        return False, error_msg


def worker_process(task_queue, result_queue, gpu_id, task_type):
    """工作进程"""
    while True:
        try:
            model_id = task_queue.get(timeout=5)
            if model_id is None:
                break

            success, msg = extract_single_model(model_id, gpu_id, task_type)
            result_queue.put((model_id, success, msg))
            time.sleep(2)

        except Exception as e:
            if "Empty" not in str(e):
                print(f"[GPU{gpu_id}] Error: {e}", flush=True)
            break


def main():
    parser = argparse.ArgumentParser(description="Worker2 GPU通用抽取")
    parser.add_argument("--list", required=True, help="模型列表文件")
    parser.add_argument("--gpus", default="1,2,3,4,5", help="GPU列表")
    parser.add_argument("--task", default="feature-extraction", help="任务类型")
    parser.add_argument("--procs-per-gpu", type=int, default=3, help="每GPU进程数")
    args = parser.parse_args()

    gpu_list = [int(g) for g in args.gpus.split(",")]
    num_gpus = len(gpu_list)
    procs_per_gpu = args.procs_per_gpu
    total_procs = num_gpus * procs_per_gpu

    with open(args.list) as f:
        models = [line.strip() for line in f if line.strip()]

    print(f"\n{'='*60}")
    print(f"Worker2 GPU通用抽取")
    print(f"GPU: {gpu_list}")
    print(f"总进程数: {total_procs}")
    print(f"模型总数: {len(models)}")
    print(f"{'='*60}\n")

    task_queue = Queue()
    result_queue = Queue()

    for model_id in models:
        task_queue.put(model_id)

    for _ in range(total_procs):
        task_queue.put(None)

    processes = []
    worker_id = 0

    for gpu_id in gpu_list:
        for _ in range(procs_per_gpu):
            p = Process(target=worker_process, args=(task_queue, result_queue, gpu_id, args.task))
            p.start()
            processes.append(p)
            print(f"启动 Worker {worker_id} (GPU {gpu_id}, PID: {p.pid})")
            worker_id += 1
            time.sleep(0.5)

    success_count = 0
    failed_count = 0
    total = len(models)
    start_time = time.time()

    while success_count + failed_count < total:
        try:
            model_id, success, msg = result_queue.get(timeout=10)
            if success:
                success_count += 1
            else:
                failed_count += 1

            elapsed = time.time() - start_time
            done = success_count + failed_count
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total - done) / rate if rate > 0 else 0

            print(f"\rProgress: {done}/{total} | Success: {success_count} | Failed: {failed_count} | ETA: {eta/60:.1f}min", end="", flush=True)

        except:
            pass

    print(f"\n\n{'='*60}")
    print(f"Completed!")
    print(f"  Success: {success_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Elapsed: {(time.time() - start_time)/60:.1f} minutes")
    print(f"{'='*60}")

    for p in processes:
        p.join(timeout=5)


if __name__ == "__main__":
    main()

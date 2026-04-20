#!/usr/bin/env python3
"""
CPU批量抽取脚本 - Worker1 (多进程并行版)
"""

import os
import sys
import json
import argparse
import torch
import traceback
import time
import hashlib
from datetime import datetime
from multiprocessing import Pool, Manager

# 强制设置代理
os.environ["http_proxy"] = "http://agent.baidu.com:8891"
os.environ["https_proxy"] = "http://agent.baidu.com:8891"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

sys.path.insert(0, "/ssd2/liangtai/GraphNet")
from graph_net.torch.extractor import extract

from transformers import (
    AutoModel, AutoModelForSequenceClassification,
    AutoModelForTokenClassification, AutoModelForQuestionAnswering,
    AutoModelForMaskedLM, AutoModelForCausalLM, AutoModelForSeq2SeqLM,
    AutoModelForSpeechSeq2Seq, AutoModelForImageClassification,
    AutoTokenizer, AutoConfig
)

TASK_MODEL_MAP = {
    "text-classification": AutoModelForSequenceClassification,
    "token-classification": AutoModelForTokenClassification,
    "fill-mask": AutoModelForMaskedLM,
    "feature-extraction": AutoModel,
    "question-answering": AutoModelForQuestionAnswering,
    "text-generation": AutoModelForCausalLM,
    "text2text-generation": AutoModelForSeq2SeqLM,
    "translation": AutoModelForSeq2SeqLM,
    "summarization": AutoModelForSeq2SeqLM,
    "automatic-speech-recognition": AutoModelForSpeechSeq2Seq,
    "image-classification": AutoModelForImageClassification,
}

TASK_KEYWORDS = {
    "text-generation": ["gpt", "llama", "mistral", "qwen", "yi", "falcon", "bloom", "gemma", "phi", "pythia", "opt", "vicuna", "wizard", "chat", "instruct", "causallm"],
    "text2text-generation": ["t5", "bart", "pegasus", "marian", "mt5", "longt5", "translation", "summarize", "summarization"],
    "text-classification": ["sentiment", "classification", "emotion", "nli", "mnli", "finetuned", "classifier", "sst", "imdb", "yelp"],
    "token-classification": ["ner", "token", "pos-tag", "pos_tag", "named-entity", "bio", "conll"],
    "question-answering": ["qa", "question", "answering", "squad", "glue"],
    "fill-mask": ["bert", "roberta", "distilbert", "electra", "albert", "fill-mask", "masked", "mlm"],
    "automatic-speech-recognition": ["whisper", "wav2vec", "speech", "asr", "audio", "wav", "voice"],
    "image-classification": ["vit", "resnet", "convnext", "image", "vision", "clip", "detr", "sam"],
}

SKIP_KEYWORDS = [
    'lora', 'adapter', 'gguf', 'ct2', 'ctranslate2',
    'awq', 'gptq', 'mlx', 'exl2', 'q4_', 'q5_', 'q8_',
    '4bit', '8bit', '3bit'
]


def infer_task_type(model_id):
    model_lower = model_id.lower()
    for kw in SKIP_KEYWORDS:
        if kw in model_lower:
            return None
    for task, keywords in TASK_KEYWORDS.items():
        for kw in keywords:
            if kw in model_lower:
                return task
    return "feature-extraction"


def detect_model_type(config, model_id):
    """根据config和model_id检测模型类型 - 借鉴Worker2"""
    model_type = getattr(config, 'model_type', '').lower()
    architectures = getattr(config, 'architectures', [])
    arch_name = architectures[0].lower() if architectures else ''
    model_id_lower = model_id.lower()

    # TTS/语音模型
    if any(x in model_id_lower for x in ['speecht5', 'vits', 'bark', 'whisper', 'wav2vec', 'hubert']):
        return 'audio'

    # 视觉模型
    if any(x in model_type or x in arch_name for x in ['vit', 'clip', 'beit', 'deit', 'swin', 'convnext', 'resnet']):
        return 'vision'

    # DETR系列
    if 'detr' in model_id_lower:
        return 'detr'

    # SAM系列
    if 'sam' in model_id_lower:
        return 'sam'

    # 多模态
    if any(x in model_type or x in arch_name for x in ['llava', 'blip', 'git', 'pix2struct']):
        return 'multimodal'

    return 'text'


def prepare_inputs(model_type, config):
    """根据模型类型准备输入 - 借鉴Worker2"""
    vocab_size = getattr(config, 'vocab_size', 32000)
    max_pos = getattr(config, 'max_position_embeddings', 512)

    if model_type == 'audio':
        # 音频模型
        if hasattr(config, 'num_mel_bins'):
            # Whisper
            return {
                'input_features': torch.randn(1, config.num_mel_bins, 3000)
            }
        else:
            # Wav2Vec
            return {
                'input_values': torch.randn(1, 16000),
                'attention_mask': torch.ones(1, 16000, dtype=torch.long)
            }

    elif model_type == 'vision':
        # 视觉模型
        image_size = getattr(config, 'image_size', 224)
        if isinstance(image_size, list):
            image_size = image_size[0]
        num_channels = getattr(config, 'num_channels', 3)
        return {
            'pixel_values': torch.randn(1, num_channels, image_size, image_size)
        }

    elif model_type == 'detr':
        # DETR
        image_size = getattr(config, 'image_size', 800)
        if isinstance(image_size, list):
            image_size = image_size[0]
        return {
            'pixel_values': torch.randn(1, 3, image_size, image_size),
            'pixel_mask': torch.ones(1, image_size, image_size, dtype=torch.long)
        }

    elif model_type == 'sam':
        # SAM
        image_size = getattr(config, 'image_size', 1024)
        return {
            'pixel_values': torch.randn(1, 3, image_size, image_size)
        }

    elif model_type == 'multimodal':
        # 多模态
        image_size = getattr(config, 'image_size', 224)
        seq_len = min(32, max_pos)
        return {
            'input_ids': torch.randint(0, min(vocab_size, 1000), (1, seq_len)),
            'attention_mask': torch.ones(1, seq_len, dtype=torch.long),
            'pixel_values': torch.randn(1, 3, image_size, image_size)
        }

    else:
        # 默认文本模型
        seq_len = min(32, max_pos)
        return {
            'input_ids': torch.randint(0, min(vocab_size, 1000), (1, seq_len)),
            'attention_mask': torch.ones(1, seq_len, dtype=torch.long)
        }


def get_model_class(task_type):
    return TASK_MODEL_MAP.get(task_type, AutoModel)


def safe_model_name(model_id):
    return model_id.replace("/", "_")


def generate_graph_hash(model_py_path):
    if os.path.exists(model_py_path):
        with open(model_py_path, 'r') as f:
            hash_val = hashlib.sha256(f.read().encode()).hexdigest()
        hash_path = model_py_path.replace("model.py", "graph_hash.txt")
        with open(hash_path, 'w') as f:
            f.write(hash_val)
        return True
    return False


def extract_single_model(args):
    """抽取单个模型 - 用于多进程"""
    model_id, output_base = args

    task_type = infer_task_type(model_id)
    if task_type is None:
        return model_id, False, "SKIP: unsupported type"

    model_name_safe = safe_model_name(model_id)
    output_dir = os.path.join(output_base, model_name_safe)

    graph_json = os.path.join(output_dir, "graph_net.json")
    if os.path.exists(graph_json):
        return model_id, True, "Already exists"

    os.makedirs(output_dir, exist_ok=True)
    os.environ["GRAPH_NET_EXTRACT_WORKSPACE"] = output_base

    try:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

        # 检测模型类型并准备输入
        model_type = detect_model_type(config, model_id)

        model_class = get_model_class(task_type)
        model = model_class.from_config(config, trust_remote_code=True)
        model.eval()

        # 确保模型使用 float32 (CPU 兼容)
        model = model.float()

        # 根据模型类型准备输入
        inputs = prepare_inputs(model_type, config)

        wrapped = extract(name=model_name_safe, dynamic=False)(model)
        with torch.no_grad():
            # 使用 **inputs 解包，更灵活
            try:
                wrapped(**inputs)
            except TypeError as e:
                # 尝试只用 input_ids
                if 'input_ids' in inputs:
                    try:
                        wrapped(inputs['input_ids'])
                    except:
                        raise e
                else:
                    raise e
            except Exception as e:
                # 针对特定错误重试 - 借鉴Worker2
                error_msg = str(e).lower()
                if 'decoder_input_ids' in error_msg and 'input_ids' in inputs:
                    inputs['decoder_input_ids'] = inputs['input_ids'].clone()
                    try:
                        wrapped(**inputs)
                    except:
                        raise e
                else:
                    raise e

        model_py = os.path.join(output_dir, "model.py")
        if os.path.exists(model_py):
            generate_graph_hash(model_py)
        else:
            for i in range(20):
                subgraph_model_py = os.path.join(output_dir, f"subgraph_{i}", "model.py")
                if os.path.exists(subgraph_model_py):
                    generate_graph_hash(subgraph_model_py)

        if os.path.exists(graph_json):
            with open(graph_json, 'r') as f:
                data = json.load(f)
            data["model_id"] = model_id
            data["task"] = task_type
            data["strategy"] = "cpu_random_init"
            with open(graph_json, 'w') as f:
                json.dump(data, f, indent=2)

        return model_id, True, "Success"

    except Exception as e:
        try:
            if not os.listdir(output_dir):
                os.rmdir(output_dir)
        except:
            pass
        return model_id, False, str(e)[:200]


def get_completed_models(output_base):
    completed = set()
    if not os.path.exists(output_base):
        return completed

    for model_dir in os.listdir(output_base):
        model_path = os.path.join(output_base, model_dir)
        if not os.path.isdir(model_path):
            continue

        graph_json = os.path.join(model_path, "graph_net.json")
        subgraph_json = os.path.join(model_path, "subgraph_0", "graph_net.json")

        if os.path.exists(graph_json) or os.path.exists(subgraph_json):
            completed.add(model_dir)
            if "_" in model_dir:
                parts = model_dir.split("_")
                if len(parts) >= 2:
                    original_id = parts[0] + "/" + "_".join(parts[1:])
                    completed.add(original_id)

    return completed


def main():
    parser = argparse.ArgumentParser(description="CPU批量抽取脚本 - Worker1 并行版")
    parser.add_argument("--list", default="/ssd2/liangtai/task_split/worker1_cpu_1500.txt")
    parser.add_argument("--output", default="/ssd2/liangtai/graphnet_workspace/huggingface/worker1")
    parser.add_argument("--workers", type=int, default=10, help="并行进程数")

    args = parser.parse_args()

    with open(args.list, 'r') as f:
        models = [line.strip().split('\t')[0] for line in f if line.strip()]

    print(f"{'='*60}")
    print(f"Worker1 CPU 抽取任务 (并行版)")
    print(f"{'='*60}")
    print(f"待抽取模型数: {len(models)}")
    print(f"输出目录: {args.output}")
    print(f"并行进程: {args.workers}")

    completed = get_completed_models(args.output)
    print(f"已完成模型数: {len(completed)}")

    pending = []
    for m in models:
        safe_name = safe_model_name(m)
        if m not in completed and safe_name not in completed:
            pending.append(m)

    print(f"过滤后待抽取: {len(pending)}")
    print(f"{'='*60}")

    os.makedirs(args.output, exist_ok=True)

    # 准备任务参数
    tasks = [(m, args.output) for m in pending]

    success, failed, skipped = 0, 0, 0
    start_time = time.time()

    # 使用进程池并行执行
    with Pool(processes=args.workers) as pool:
        for i, result in enumerate(pool.imap_unordered(extract_single_model, tasks)):
            model_id, ok, msg = result
            status = "✓" if ok else "✗"
            print(f"[{i+1}/{len(pending)}] {status} {model_id}: {msg}")

            if ok:
                success += 1
            else:
                if "SKIP" in msg:
                    skipped += 1
                else:
                    failed += 1

            # 每50个输出进度
            if (i + 1) % 50 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                remaining = (len(pending) - i - 1) / rate / 60
                print(f"\n{'='*60}")
                print(f"进度: {i+1}/{len(pending)}, 成功: {success}, 失败: {failed}")
                print(f"预计剩余: {remaining:.1f} 分钟")
                print(f"{'='*60}\n")

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"抽取完成!")
    print(f"成功: {success}, 失败: {failed}, 跳过: {skipped}")
    print(f"总耗时: {elapsed/60:.1f} 分钟")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
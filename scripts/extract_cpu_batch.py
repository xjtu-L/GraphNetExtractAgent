#!/usr/bin/env python3
"""
CPU批量抽取脚本
- 从模型列表抽取 0-75000 范围的模型
- 自动跳过已完成的模型
- 支持多进程并行
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
from multiprocessing import Pool, cpu_count

# 强制设置代理
os.environ["http_proxy"] = "http://agent.baidu.com:8891"
os.environ["https_proxy"] = "http://agent.baidu.com:8891"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

sys.path.insert(0, "/ssd1/liangtai-work/GraphNet")
from graph_net.torch.extractor import extract

from transformers import (
    AutoModel, AutoModelForSequenceClassification,
    AutoModelForTokenClassification, AutoModelForQuestionAnswering,
    AutoModelForMaskedLM, AutoModelForCausalLM, AutoModelForSeq2SeqLM,
    AutoModelForSpeechSeq2Seq, AutoModelForImageClassification,
    AutoTokenizer, AutoConfig
)

# 任务类型到模型类的映射
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

# 模型名称关键词到任务类型的映射
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

# 跳过关键字
SKIP_KEYWORDS = [
    'lora', 'adapter', 'gguf', 'ct2', 'ctranslate2',
    'awq', 'gptq', 'mlx', 'exl2', 'q4_', 'q5_', 'q8_',
    '4bit', '8bit', '3bit'
]


def infer_task_type(model_id):
    """根据模型名称推断任务类型"""
    model_lower = model_id.lower()

    # 检查跳过关键字
    for kw in SKIP_KEYWORDS:
        if kw in model_lower:
            return None

    for task, keywords in TASK_KEYWORDS.items():
        for kw in keywords:
            if kw in model_lower:
                return task

    return "feature-extraction"


def get_model_class(task_type):
    """获取模型类"""
    return TASK_MODEL_MAP.get(task_type, AutoModel)


def safe_model_name(model_id):
    """安全的模型目录名"""
    return model_id.replace("/", "_")


def get_completed_models(huggingface_base):
    """获取已完成的模型列表 - 扫描整个huggingface目录"""
    completed = set()

    # 扫描huggingface目录下所有子目录
    for task_dir in os.listdir(huggingface_base):
        task_path = os.path.join(huggingface_base, task_dir)
        if not os.path.isdir(task_path):
            continue
        if task_dir in ['logs', 'archive', 'batch_logs', 'batch_results', 'batch_runs',
                        'complete_batch', 'core_scripts', 'extraction_results', 'mega_search',
                        'other', '__pycache__', 'hpu_handover', 'batch_reextract']:
            continue

        # 遍历该类型目录下的所有模型
        for model_dir in os.listdir(task_path):
            model_path = os.path.join(task_path, model_dir)
            if not os.path.isdir(model_path):
                continue

            # 检查是否有graph_net.json
            graph_json = os.path.join(model_path, "graph_net.json")
            subgraph_json = os.path.join(model_path, "subgraph_0", "graph_net.json")

            if os.path.exists(graph_json) or os.path.exists(subgraph_json):
                # 保存两种格式：原始模型ID和安全名
                completed.add(model_dir)
                # 尝试恢复原始模型ID
                if "_" in model_dir:
                    parts = model_dir.split("_")
                    if len(parts) >= 2:
                        original_id = parts[0] + "/" + "_".join(parts[1:])
                        completed.add(original_id)
                        # 也保存安全名格式
                        safe_name = model_dir.replace("_", "/", 1) if "/" not in model_dir else model_dir
                        completed.add(safe_name)

    return completed


def generate_graph_hash(model_py_path):
    """生成graph_hash.txt"""
    if os.path.exists(model_py_path):
        with open(model_py_path, 'r') as f:
            hash_val = hashlib.sha256(f.read().encode()).hexdigest()
        hash_path = model_py_path.replace("model.py", "graph_hash.txt")
        with open(hash_path, 'w') as f:
            f.write(hash_val)
        return True
    return False


def extract_single_model(model_id, output_base):
    """抽取单个模型"""
    task_type = infer_task_type(model_id)
    if task_type is None:
        return False, f"SKIP: unsupported type"

    model_name_safe = safe_model_name(model_id)
    # 扁平输出到worker1目录
    output_dir = os.path.join(output_base, model_name_safe)

    # 检查是否已完成
    graph_json = os.path.join(output_dir, "graph_net.json")
    if os.path.exists(graph_json):
        return True, "Already exists"

    # 创建目录
    os.makedirs(output_dir, exist_ok=True)

    # 设置工作目录 - 直接输出到worker1
    os.environ["GRAPH_NET_EXTRACT_WORKSPACE"] = output_base

    try:
        # 加载config
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        model_class = get_model_class(task_type)

        # 随机初始化
        model = model_class.from_config(config, trust_remote_code=True)
        model.eval()

        # 准备输入
        vocab_size = getattr(config, 'vocab_size', 32000)
        max_pos = getattr(config, 'max_position_embeddings', 512)
        seq_len = min(32, max_pos)

        input_ids = torch.randint(0, min(vocab_size, 1000), (1, seq_len))

        # 抽取
        wrapped = extract(name=model_name_safe, dynamic=False)(model)
        with torch.no_grad():
            wrapped(input_ids)

        # 生成graph_hash.txt
        model_py = os.path.join(output_dir, "model.py")
        if os.path.exists(model_py):
            generate_graph_hash(model_py)
        else:
            # 检查子图
            for i in range(20):
                subgraph_model_py = os.path.join(output_dir, f"subgraph_{i}", "model.py")
                if os.path.exists(subgraph_model_py):
                    generate_graph_hash(subgraph_model_py)

        # 更新graph_net.json
        if os.path.exists(graph_json):
            with open(graph_json, 'r') as f:
                data = json.load(f)
            data["model_id"] = model_id
            data["task"] = task_type
            data["strategy"] = "cpu_random_init"
            with open(graph_json, 'w') as f:
                json.dump(data, f, indent=2)

        return True, "Success"

    except Exception as e:
        # 清理空目录
        try:
            if not os.listdir(output_dir):
                os.rmdir(output_dir)
        except:
            pass
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(description="CPU批量抽取脚本")
    parser.add_argument("--list", default="/tmp/pending_models.txt", help="模型列表文件")
    parser.add_argument("--output", default="/ssd1/liangtai-work/graphnet_workspace/huggingface/worker1", help="输出目录")
    parser.add_argument("--workers", type=int, default=4, help="并行进程数")
    parser.add_argument("--batch", type=int, default=None, help="批次ID")
    parser.add_argument("--total-batches", type=int, default=1, help="总批次数")

    args = parser.parse_args()

    # 读取模型列表
    with open(args.list, 'r') as f:
        models = [line.strip().split('\t')[0] for line in f if line.strip()]

    # 批次处理
    if args.batch is not None:
        batch_size = len(models) // args.total_batches
        start_idx = args.batch * batch_size
        models = models[start_idx:start_idx + batch_size]

    print(f"待抽取模型数: {len(models)}")
    print(f"输出目录: {args.output}")
    print(f"并行进程: {args.workers}")

    # 获取已完成模型 - 扫描整个huggingface目录
    huggingface_base = os.path.dirname(args.output)  # /ssd1/liangtai-work/graphnet_workspace/huggingface
    completed = get_completed_models(huggingface_base)
    print(f"已完成模型数: {len(completed)}")

    # 过滤已完成
    pending = []
    for m in models:
        safe_name = safe_model_name(m)
        if m not in completed and safe_name not in completed:
            pending.append(m)

    print(f"过滤后待抽取: {len(pending)}")

    # 执行抽取
    success, failed = 0, 0
    for i, model_id in enumerate(pending):
        print(f"\n[{i+1}/{len(pending)}] {model_id}")
        ok, msg = extract_single_model(model_id, args.output)
        if ok:
            success += 1
            print(f"  ✓ {msg}")
        else:
            failed += 1
            print(f"  ✗ {msg}")
        time.sleep(2)  # 避免速率限制

    print(f"\n{'='*60}")
    print(f"完成: 成功 {success}, 失败 {failed}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Slow-mode HuggingFace extraction script - designed to avoid rate limiting
- Minimal concurrent requests
- Built-in delays between models
- Retry logic for failed downloads

Usage:
    python extract_hf_slow_v3.py --list models.txt --task text-classification
"""

import os
import sys
import json
import argparse
import torch
import traceback
import time
import random
from datetime import datetime

# 强制设置代理（必须显式设置，不要依赖setdefault）
os.environ["http_proxy"] = "http://agent.baidu.com:8891"
os.environ["https_proxy"] = "http://agent.baidu.com:8891"

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

sys.path.insert(0, "/root/GraphNet")
from graph_net.torch.extractor import extract

try:
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoModelForTokenClassification
    from transformers import AutoModelForQuestionAnswering, AutoModelForMaskedLM
    from transformers import AutoTokenizer, AutoConfig
except ImportError as e:
    print(f"Error: transformers library not installed - {e}")
    sys.exit(1)

TASK_MODEL_MAP = {
    "text-classification": AutoModelForSequenceClassification,
    "token-classification": AutoModelForTokenClassification,
    "fill-mask": AutoModelForMaskedLM,
    "feature-extraction": AutoModel,
    "question-answering": AutoModelForQuestionAnswering,
    "text-generation": AutoModel,
    "translation": AutoModel,
    "summarization": AutoModel,
}

def safe_model_name(model_id):
    return model_id.replace("/", "_").replace("-", "_")

def get_workspace(task_type):
    base = "/root/graphnet_workspace/huggingface"
    return os.path.join(base, task_type)

def get_model_class(task_type):
    return TASK_MODEL_MAP.get(task_type, AutoModel)

def extract_model(model_id, task_type, device="cpu", dynamic=False, max_length=128, delay=5):
    """Extract single model with built-in delays and retries"""
    model_name_safe = safe_model_name(model_id)
    workspace = get_workspace(task_type)
    output_dir = os.path.join(workspace, model_name_safe)
    os.makedirs(output_dir, exist_ok=True)
    os.environ["GRAPH_NET_EXTRACT_WORKSPACE"] = workspace

    print(f"\n{'='*60}")
    print(f"[V3-Slow] {task_type}: {model_id}")
    print(f"{'='*60}")

    # Step 1: Load tokenizer with retry
    tokenizer = None
    for attempt in range(3):
        try:
            print(f"[1/4] Loading Tokenizer (attempt {attempt+1})...", end=" ", flush=True)
            time.sleep(random.uniform(1, 3))  # Random delay to avoid burst
            tokenizer = AutoTokenizer.from_pretrained(
                model_id, trust_remote_code=True, local_files_only=False
            )
            print("OK")
            break
        except Exception as e:
            print(f"Failed: {e}")
            if attempt < 2:
                time.sleep(5 * (attempt + 1))  # Exponential backoff
            else:
                return False, f"Tokenizer failed after 3 attempts: {e}"

    # Step 2: Config-only random initialization with retry
    try:
        print("[2/4] Loading Config & Random Init...", end=" ", flush=True)
        time.sleep(random.uniform(0.5, 1.5))
        model_class = get_model_class(task_type)
        config = AutoConfig.from_pretrained(
            model_id, trust_remote_code=True, local_files_only=False
        )
        model = model_class.from_config(config)
        model = model.to(device).eval()
        param_count = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"OK ({param_count:.1f}M params)")
    except Exception as e:
        return False, f"Model init failed: {e}"

    # Step 3: Construct input
    try:
        print("[3/4] Preparing Input...", end=" ", flush=True)
        dummy_text = "This is a sample text for model extraction."
        inputs = tokenizer(
            dummy_text, return_tensors="pt", padding="max_length",
            max_length=max_length, truncation=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        print(f"OK {inputs['input_ids'].shape}")
    except Exception as e:
        return False, f"Input prep failed: {e}"

    # Step 4: Extract graph
    try:
        print("[4/4] Extracting Graph...", end=" ", flush=True)
        wrapped = extract(name=model_name_safe, dynamic=dynamic)(model).eval()
        with torch.no_grad():
            wrapped(**inputs)

        # Verify output
        if not os.path.exists(os.path.join(output_dir, "graph_net.json")):
            found = False
            for i in range(20):
                if os.path.exists(os.path.join(output_dir, f"subgraph_{i}", "graph_net.json")):
                    found = True
                    break
            if not found:
                raise RuntimeError("graph_net.json not generated")
        print("OK Success!")
    except Exception as e:
        print(f"Failed: {e}")
        traceback.print_exc()
        return False, f"Extract failed: {e}"

    # Update metadata
    try:
        json_path = os.path.join(output_dir, "graph_net.json")
        data = {}
        if os.path.exists(json_path):
            with open(json_path) as f:
                data = json.load(f)
        data.update({
            "model_id": model_id,
            "source": "huggingface",
            "task": task_type,
            "max_length": max_length,
            "strategy": "v3_slow_mode",
            "extracted_at": datetime.now().isoformat(),
        })
        with open(json_path, "w") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"[!] Warning: Metadata update failed: {e}")

    # Delay before next model
    time.sleep(delay)
    return True, "Success"

def main():
    parser = argparse.ArgumentParser(description="[V3-Slow] HuggingFace extraction with rate limiting")
    parser.add_argument("--list", required=True, help="Model list file")
    parser.add_argument("--task", required=True, help="Task type")
    parser.add_argument("--device", default="cpu", help="Device")
    parser.add_argument("--dynamic", action="store_true", help="Dynamic shapes")
    parser.add_argument("--max-length", type=int, default=128, help="Max sequence length")
    parser.add_argument("--delay", type=int, default=5, help="Delay between models (seconds)")
    parser.add_argument("--start", type=int, default=0, help="Start index")
    parser.add_argument("--count", type=int, default=None, help="Number of models to process")
    args = parser.parse_args()

    with open(args.list) as f:
        models = [line.strip() for line in f if line.strip()]

    if args.count:
        models = models[args.start:args.start + args.count]
    else:
        models = models[args.start:]

    print(f"\n{'#'*60}")
    print(f"# [V3-Slow] {args.task}")
    print(f"# Models: {len(models)} (from index {args.start})")
    print(f"# Delay: {args.delay}s between models")
    print(f"# Workspace: {get_workspace(args.task)}")
    print(f"{'#'*60}\n")

    success = []
    failed = []

    for i, model_id in enumerate(models):
        print(f"\n[{i+1}/{len(models)}] {model_id}")
        ok, msg = extract_model(
            model_id, args.task, args.device, args.dynamic, args.max_length, args.delay
        )
        if ok:
            success.append(model_id)
        else:
            failed.append((model_id, msg))
            print(f"[!] Failed: {msg}")

    # Stats
    print(f"\n{'='*60}")
    print(f"[V3-Slow] {args.task} Complete")
    print(f"Total: {len(models)}")
    print(f"Success: {len(success)}")
    print(f"Failed: {len(failed)}")
    print(f"{'='*60}")

    # Save results
    result_dir = "/root/GraphNetExtractAgent/results"
    os.makedirs(result_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(result_dir, f"slow_v3_{args.task}_{timestamp}.json")

    with open(result_file, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "task": args.task,
            "total": len(models),
            "success": len(success),
            "failed": len(failed),
            "success_models": success,
            "failed_models": [{"model": m, "error": e} for m, e in failed]
        }, f, indent=2)

    print(f"\nResults saved: {result_file}")
    return 0 if not failed else 1

if __name__ == "__main__":
    sys.exit(main())

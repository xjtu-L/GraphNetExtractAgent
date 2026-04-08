#!/usr/bin/env python3
"""
Offline-mode HuggingFace extraction - pre-downloads all files to avoid cache issues
- Downloads config and tokenizer files manually using curl
- Avoids HF Hub cache permission bugs
- No concurrent file access issues

Usage:
    python extract_hf_offline_v3.py --list models.txt --task text-classification
"""

import os
import sys
import json
import argparse
import torch
import traceback
import time
import subprocess
import tempfile

# 强制设置代理（必须显式设置，不要依赖setdefault）
os.environ["http_proxy"] = "http://agent.baidu.com:8891"
os.environ["https_proxy"] = "http://agent.baidu.com:8891"
import shutil
from datetime import datetime

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

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
    """获取任务类型对应的workspace路径 - 强制归类"""
    base = "/ssd1/liangtai-work/graphnet_workspace/huggingface"
    workspace = os.path.join(base, task_type)
    # 确保目录存在
    os.makedirs(workspace, exist_ok=True)
    return workspace


def validate_workspace():
    """验证workspace路径，防止保存到根目录"""
    ws = os.environ.get("GRAPH_NET_EXTRACT_WORKSPACE", "")
    if not ws or ws == "/ssd1/liangtai-work/graphnet_workspace":
        print("[ERROR] GRAPH_NET_EXTRACT_WORKSPACE not set or set to root workspace!")
        print("[ERROR] Models must be saved to task-specific folders!")
        sys.exit(1)
    if "huggingface" not in ws:
        print(f"[WARNING] Workspace {ws} may not be in huggingface folder!")
    return ws

def get_model_class(task_type):
    return TASK_MODEL_MAP.get(task_type, AutoModel)

def download_model_files(model_id, local_dir):
    """Manually download model files using curl to avoid HF Hub cache issues"""
    hf_endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
    base_url = f"{hf_endpoint}/{model_id}/resolve/main"

    files_to_try = [
        "config.json",
        "tokenizer_config.json",
        "vocab.txt",
        "tokenizer.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "spiece.model",
        "merges.txt",
    ]

    os.makedirs(local_dir, exist_ok=True)
    proxy = os.environ.get("https_proxy", os.environ.get("http_proxy", ""))

    downloaded = []
    for filename in files_to_try:
        url = f"{base_url}/{filename}"
        output_path = os.path.join(local_dir, filename)
        try:
            if proxy:
                cmd = ["curl", "-sL", "-f", "--max-time", "30", "-x", proxy, url, "-o", output_path]
            else:
                cmd = ["curl", "-sL", "-f", "--max-time", "30", url, "-o", output_path]
            result = subprocess.run(cmd, capture_output=True, timeout=35)
            if result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                downloaded.append(filename)
        except Exception:
            pass

    return downloaded

def extract_model_offline(model_id, task_type, device="cpu", dynamic=False, max_length=128):
    """Extract model using offline mode (manual download)"""
    model_name_safe = safe_model_name(model_id)
    workspace = get_workspace(task_type)
    output_dir = os.path.join(workspace, model_name_safe)
    os.makedirs(output_dir, exist_ok=True)
    os.environ["GRAPH_NET_EXTRACT_WORKSPACE"] = workspace
    validate_workspace()  # 强制验证归类

    print(f"\n{'='*60}")
    print(f"[V3-Offline] {task_type}: {model_id}")
    print(f"{'='*60}")

    # Step 1: Manual download of files
    local_dir = tempfile.mkdtemp()
    try:
        print("[1/5] Downloading files manually...", end=" ", flush=True)
        downloaded = download_model_files(model_id, local_dir)
        if "config.json" not in downloaded:
            return False, "config.json not found"
        print(f"OK ({len(downloaded)} files)")
    except Exception as e:
        shutil.rmtree(local_dir, ignore_errors=True)
        return False, f"Download failed: {e}"

    # Step 2: Load tokenizer from local files
    try:
        print("[2/5] Loading Tokenizer from local...", end=" ", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(local_dir, trust_remote_code=True, local_files_only=True)
        print("OK")
    except Exception as e:
        shutil.rmtree(local_dir, ignore_errors=True)
        return False, f"Tokenizer load failed: {e}"

    # Step 3: Random initialization from local config
    try:
        print("[3/5] Loading Config & Random Init...", end=" ", flush=True)
        model_class = get_model_class(task_type)
        config = AutoConfig.from_pretrained(local_dir, trust_remote_code=True, local_files_only=True)
        model = model_class.from_config(config)
        model = model.to(device).eval()
        param_count = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"OK ({param_count:.1f}M params)")
    except Exception as e:
        shutil.rmtree(local_dir, ignore_errors=True)
        return False, f"Model init failed: {e}"

    # Step 4: Prepare input
    try:
        print("[4/5] Preparing Input...", end=" ", flush=True)
        dummy_text = "This is a sample text for model extraction."
        inputs = tokenizer(dummy_text, return_tensors="pt", padding="max_length",
                          max_length=max_length, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        print(f"OK {inputs['input_ids'].shape}")
    except Exception as e:
        shutil.rmtree(local_dir, ignore_errors=True)
        return False, f"Input prep failed: {e}"

    # Step 5: Extract graph
    try:
        print("[5/5] Extracting Graph...", end=" ", flush=True)
        wrapped = extract(name=model_name_safe, dynamic=dynamic)(model).eval()
        with torch.no_grad():
            wrapped(**inputs)

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
        shutil.rmtree(local_dir, ignore_errors=True)
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
            "strategy": "v3_offline_mode",
            "extracted_at": datetime.now().isoformat(),
        })
        with open(json_path, "w") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"[!] Warning: Metadata update failed: {e}")

    # Cleanup
    shutil.rmtree(local_dir, ignore_errors=True)

    # Delay to be nice to HF servers
    time.sleep(3)
    return True, "Success"

def main():
    parser = argparse.ArgumentParser(description="[V3-Offline] HuggingFace extraction without cache")
    parser.add_argument("--list", required=True, help="Model list file")
    parser.add_argument("--task", required=True, help="Task type")
    parser.add_argument("--device", default="cpu", help="Device")
    parser.add_argument("--dynamic", action="store_true", help="Dynamic shapes")
    parser.add_argument("--max-length", type=int, default=128, help="Max sequence length")
    parser.add_argument("--start", type=int, default=0, help="Start index")
    parser.add_argument("--count", type=int, default=None, help="Number of models")
    args = parser.parse_args()

    # 设置并验证工作目录 - 强制归类到任务文件夹
    workspace = get_workspace(args.task)
    os.environ["GRAPH_NET_EXTRACT_WORKSPACE"] = workspace
    validate_workspace()

    with open(args.list) as f:
        models = [line.strip() for line in f if line.strip()]

    if args.count:
        models = models[args.start:args.start + args.count]
    else:
        models = models[args.start:]

    print(f"\n{'#'*60}")
    print(f"# [V3-Offline] {args.task}")
    print(f"# Models: {len(models)} (from index {args.start})")
    print(f"# Workspace: {get_workspace(args.task)}")
    print(f"{'#'*60}\n")

    success = []
    failed = []

    for i, model_id in enumerate(models):
        print(f"\n[{i+1}/{len(models)}] {model_id}")
        ok, msg = extract_model_offline(model_id, args.task, args.device, args.dynamic, args.max_length)
        if ok:
            success.append(model_id)
        else:
            failed.append((model_id, msg))
            print(f"[!] Failed: {msg}")

    print(f"\n{'='*60}")
    print(f"[V3-Offline] {args.task} Complete")
    print(f"Total: {len(models)}")
    print(f"Success: {len(success)}")
    print(f"Failed: {len(failed)}")
    print(f"{'='*60}")

    result_dir = "/root/GraphNetExtractAgent/results"
    os.makedirs(result_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(result_dir, f"offline_v3_{args.task}_{timestamp}.json")

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

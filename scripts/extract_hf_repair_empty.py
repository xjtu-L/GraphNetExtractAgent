#!/usr/bin/env python3
"""
空目录修复脚本 - 使用from_config()方式重新抓取
1. 下载config.json (不下载权重)
2. 使用from_config()随机初始化
3. 抽取完整计算图
"""

import os
import sys
import json
import argparse
import torch
from datetime import datetime
from pathlib import Path

# 设置代理
os.environ["http_proxy"] = "http://agent.baidu.com:8891"
os.environ["https_proxy"] = "http://agent.baidu.com:8891"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

sys.path.insert(0, "/root/GraphNet")
from graph_net.torch.extractor import extract

try:
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoModelForTokenClassification
    from transformers import AutoModelForQuestionAnswering, AutoModelForMaskedLM, AutoModelForCausalLM
    from transformers import AutoTokenizer, AutoConfig
    from huggingface_hub import hf_hub_download
except ImportError as e:
    print(f"错误: 库未安装 - {e}")
    sys.exit(1)

TASK_MODEL_MAP = {
    "text-classification": AutoModelForSequenceClassification,
    "token-classification": AutoModelForTokenClassification,
    "fill-mask": AutoModelForMaskedLM,
    "feature-extraction": AutoModel,
    "question-answering": AutoModelForQuestionAnswering,
    "text-generation": AutoModelForCausalLM,
    "translation": AutoModel,
    "summarization": AutoModel,
    "text2text-generation": AutoModel,
}

def get_workspace(task_type):
    base = "/work/graphnet_workspace/huggingface"
    workspace = os.path.join(base, task_type)
    os.makedirs(workspace, exist_ok=True)
    return workspace

def safe_model_name(model_id):
    return model_id.replace("/", "_")

def download_config_only(model_id, model_dir):
    """仅下载config.json"""
    try:
        config_path = hf_hub_download(
            repo_id=model_id,
            filename="config.json",
            local_dir=model_dir,
            local_dir_use_symlinks=False,
            resume_download=False
        )
        return config_path
    except Exception as e:
        print(f"    下载config失败: {e}")
        return None

def mark_status(model_dir, status, worker="Worker2", error=""):
    """标记修复状态"""
    os.makedirs(model_dir, exist_ok=True)
    status_file = os.path.join(model_dir, ".repair_status")
    if status == "in_progress":
        with open(status_file, "w") as f:
            f.write(f"worker: {worker}\n")
            f.write(f"status: in_progress\n")
            f.write(f"start_time: {datetime.now().isoformat()}\n")
    elif status == "completed":
        start_time = datetime.now().isoformat()
        if os.path.exists(status_file):
            try:
                with open(status_file, "r") as f:
                    lines = f.readlines()
                start_time = [l for l in lines if l.startswith("start_time:")][0].split(":", 1)[1].strip()
            except:
                pass
        with open(status_file, "w") as f:
            f.write(f"worker: {worker}\n")
            f.write(f"status: completed\n")
            f.write(f"start_time: {start_time}\n")
            f.write(f"end_time: {datetime.now().isoformat()}\n")
            f.write(f"method: from_config\n")
    elif status == "failed":
        start_time = datetime.now().isoformat()
        if os.path.exists(status_file):
            try:
                with open(status_file, "r") as f:
                    lines = f.readlines()
                start_time = [l for l in lines if l.startswith("start_time:")][0].split(":", 1)[1].strip()
            except:
                pass
        with open(status_file, "w") as f:
            f.write(f"worker: {worker}\n")
            f.write(f"status: failed\n")
            f.write(f"start_time: {start_time}\n")
            f.write(f"end_time: {datetime.now().isoformat()}\n")
            f.write(f"error: {error}\n")

def find_model_task(model_id):
    """查找模型所属的任务类型"""
    safe_name = safe_model_name(model_id)
    base = "/work/graphnet_workspace/huggingface"

    # 尝试在各个任务目录中查找
    for task in TASK_MODEL_MAP.keys():
        task_dir = os.path.join(base, task, safe_name)
        if os.path.exists(task_dir):
            return task, task_dir

    # 扫描所有子目录查找
    try:
        for task in os.listdir(base):
            task_path = os.path.join(base, task)
            if os.path.isdir(task_path):
                model_path = os.path.join(task_path, safe_name)
                if os.path.exists(model_path):
                    # 返回找到的任务类型，如果不在TASK_MODEL_MAP中则使用AutoModel
                    return task, model_path
    except:
        pass

    # 默认使用text-generation
    return "text-generation", os.path.join(base, "text-generation", safe_name)

def repair_model(model_id, device="cpu", max_length=128):
    """修复单个模型"""
    task_type, model_dir = find_model_task(model_id)
    safe_name = safe_model_name(model_id)

    print(f"\n[修复] {model_id} ({task_type})")
    print(f"  目录: {model_dir}")

    # 确保目录存在
    os.makedirs(model_dir, exist_ok=True)

    # 检查是否已被修复（检查根目录或subgraph_0目录）
    has_model_py = os.path.exists(os.path.join(model_dir, "model.py")) or \
                   os.path.exists(os.path.join(model_dir, "subgraph_0", "model.py"))
    has_weight_meta = os.path.exists(os.path.join(model_dir, "weight_meta.py")) or \
                      os.path.exists(os.path.join(model_dir, "subgraph_0", "weight_meta.py"))
    if has_model_py and has_weight_meta:
        print(f"  ✓ 已修复，跳过")
        return True, "已存在"

    # 标记为进行中
    mark_status(model_dir, "in_progress")

    try:
        # 步骤1: 下载config.json
        config_path = os.path.join(model_dir, "config.json")
        if not os.path.exists(config_path):
            print(f"  下载config.json...", end=" ", flush=True)
            downloaded = download_config_only(model_id, model_dir)
            if not downloaded:
                mark_status(model_dir, "failed", error="下载config失败")
                return False, "下载config失败"
            print("OK")

        # 步骤2: 从config加载
        print(f"  加载config...", end=" ", flush=True)
        model_class = TASK_MODEL_MAP.get(task_type, AutoModel)
        config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)
        print("OK", end=" ")

        # 步骤3: 使用from_config初始化
        print(f"初始化...", end=" ", flush=True)
        model = model_class.from_config(config, trust_remote_code=True)
        model = model.to(device).eval()
        print("OK", end=" ")

        # 步骤4: 设置workspace并抽取
        workspace = get_workspace(task_type)
        os.environ["GRAPH_NET_EXTRACT_WORKSPACE"] = workspace

        # 构造输入
        vocab_size = getattr(config, 'vocab_size', 32000)
        dummy_input_ids = torch.randint(0, min(vocab_size, 32000), (1, max_length))
        attention_mask = torch.ones((1, max_length))
        inputs = {"input_ids": dummy_input_ids, "attention_mask": attention_mask}

        # 使用GraphNet抽取
        print(f"抽取...", end=" ", flush=True)
        wrapped = extract(name=safe_name, dynamic=False)(model).eval()

        with torch.no_grad():
            wrapped(**inputs)
        print("OK")

        # 验证结果（检查根目录或subgraph_0目录）
        has_model_py = os.path.exists(os.path.join(model_dir, "model.py")) or \
                       os.path.exists(os.path.join(model_dir, "subgraph_0", "model.py"))
        has_weight_meta = os.path.exists(os.path.join(model_dir, "weight_meta.py")) or \
                          os.path.exists(os.path.join(model_dir, "subgraph_0", "weight_meta.py"))
        if has_model_py and has_weight_meta:
            # 更新元数据
            json_path = os.path.join(model_dir, "graph_net.json")
            if os.path.exists(json_path):
                with open(json_path) as f:
                    data = json.load(f)
                data.update({
                    "model_id": model_id,
                    "source": "huggingface",
                    "task": task_type,
                    "max_length": max_length,
                    "extraction_method": "repair_from_config",
                    "extracted_at": datetime.now().isoformat(),
                })
                with open(json_path, "w") as f:
                    json.dump(data, f, indent=2)

            mark_status(model_dir, "completed")
            print(f"  ✓ 修复成功")
            return True, "OK"
        else:
            mark_status(model_dir, "failed", error="未生成model.py或weight_meta.py")
            print(f"  ✗ 失败: 未生成必需文件")
            return False, "未生成必需文件"

    except Exception as e:
        error_msg = str(e)[:200]
        mark_status(model_dir, "failed", error=error_msg)
        print(f"  ✗ 失败: {error_msg}")
        return False, error_msg

def main():
    parser = argparse.ArgumentParser(description="空目录修复脚本")
    parser.add_argument("--list", required=True, help="模型列表文件")
    parser.add_argument("--device", default="cpu", help="设备")
    parser.add_argument("--max-length", type=int, default=128, help="最大长度")
    parser.add_argument("--agent-id", default="repair", help="Agent ID")
    parser.add_argument("--worker", default="Worker2", help="Worker名称")

    args = parser.parse_args()

    # 读取模型列表
    with open(args.list) as f:
        models = [line.strip() for line in f if line.strip()]

    print(f"[{args.agent_id}] 开始修复: {len(models)} 个模型")
    print(f"  Worker: {args.worker}")
    print(f"  设备: {args.device}")
    print("="*60)

    success = 0
    failed = 0
    skipped = 0

    for i, model_id in enumerate(models, 1):
        print(f"\n[{i}/{len(models)}]", end=" ")
        ok, msg = repair_model(model_id, args.device, args.max_length)
        if ok:
            if msg == "已存在":
                skipped += 1
            else:
                success += 1
        else:
            failed += 1

    print(f"\n{'='*60}")
    print(f"完成: 成功 {success}, 失败 {failed}, 跳过 {skipped}")

    # 写入进度文件
    progress_file = f"/tmp/progress_{args.agent_id}.txt"
    with open(progress_file, "w") as f:
        f.write(f"agent: {args.agent_id}\n")
        f.write(f"total: {len(models)}\n")
        f.write(f"success: {success}\n")
        f.write(f"failed: {failed}\n")
        f.write(f"skipped: {skipped}\n")
        f.write(f"completed_at: {datetime.now().isoformat()}\n")

if __name__ == "__main__":
    main()

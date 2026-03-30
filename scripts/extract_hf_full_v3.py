#!/usr/bin/env python3
"""
完整计算图抽取脚本 V3 - 修复版
- 使用from_config()随机初始化，不下载完整权重
- 使用正确的GraphNet extract() wrapper
- 生成完整的model.py, weight_meta.py等文件
- 修复网络连接问题
"""

import os
import sys
import json
import argparse
import torch
from datetime import datetime
from pathlib import Path

# 强制设置代理（必须显式设置，不要依赖setdefault）
os.environ["http_proxy"] = "http://agent.baidu.com:8891"
os.environ["https_proxy"] = "http://agent.baidu.com:8891"

# 强制使用hf-mirror.com
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "0"

sys.path.insert(0, "/root/GraphNet")
from graph_net.torch.extractor import extract

try:
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoModelForTokenClassification
    from transformers import AutoModelForQuestionAnswering, AutoModelForMaskedLM, AutoModelForCausalLM
    from transformers import AutoTokenizer, AutoConfig
    from huggingface_hub import hf_hub_download
except ImportError as e:
    print(f"错误: 依赖库未安装 - {e}")
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
}

def get_workspace(task_type):
    base = "/work/graphnet_workspace/huggingface"
    workspace = os.path.join(base, task_type)
    os.makedirs(workspace, exist_ok=True)
    return workspace

def safe_model_name(model_id):
    return model_id.replace("/", "_").replace("-", "_")

def download_config_only(model_id, local_dir):
    """只下载config.json，不下载权重"""
    try:
        config_path = hf_hub_download(
            repo_id=model_id,
            filename="config.json",
            local_dir=local_dir,
            local_dir_use_symlinks=False,
        )
        return config_path
    except Exception as e:
        print(f"  下载config失败: {str(e)[:80]}")
        return None

def extract_model_full(model_id, task_type, device="cpu", dynamic=False, max_length=128):
    """使用完整GraphNet抽取流程 - from_config版本"""
    try:
        print(f"  [抽取] {model_id} ({task_type})", flush=True)

        # 设置workspace
        workspace = get_workspace(task_type)
        os.environ["GRAPH_NET_EXTRACT_WORKSPACE"] = workspace

        # 创建临时目录下载config
        safe_name = safe_model_name(model_id)
        temp_dir = os.path.join("/tmp/hf_configs", safe_name)
        os.makedirs(temp_dir, exist_ok=True)

        # 步骤1: 下载config.json
        print(f"    下载config...", end=" ", flush=True)
        config_path = download_config_only(model_id, temp_dir)
        if not config_path:
            print(f"✗ 失败")
            return False, "config下载失败"
        print(f"OK", end=" ", flush=True)

        # 步骤2: 从config加载（随机初始化）
        print(f"加载模型...", end=" ", flush=True)
        model_class = TASK_MODEL_MAP.get(task_type, AutoModel)
        config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)

        # 使用from_config随机初始化（不下载权重）
        model = model_class.from_config(config, trust_remote_code=True)
        model = model.to(device).eval()
        print(f"OK", end=" ", flush=True)

        # 步骤3: 创建dummy输入（不需要tokenizer）
        # 从config获取vocab_size和max_length
        vocab_size = getattr(config, 'vocab_size', 32000)
        hidden_size = getattr(config, 'hidden_size', 768)

        # 创建dummy input_ids
        dummy_input_ids = torch.randint(0, vocab_size, (1, max_length)).to(device)
        inputs = {"input_ids": dummy_input_ids}

        # 尝试添加attention_mask
        inputs["attention_mask"] = torch.ones_like(dummy_input_ids)

        print(f"准备抽取...", end=" ", flush=True)

        # 步骤4: 使用GraphNet extract抽取
        wrapped = extract(name=safe_name, dynamic=dynamic)(model).eval()

        with torch.no_grad():
            wrapped(**inputs)

        print(f"✓ 完成", flush=True)

        # 步骤5: 更新元数据
        output_dir = os.path.join(workspace, safe_name)
        json_path = os.path.join(output_dir, "graph_net.json")
        if os.path.exists(json_path):
            with open(json_path) as f:
                data = json.load(f)
            data.update({
                "model_id": model_id,
                "source": "huggingface",
                "task": task_type,
                "max_length": max_length,
                "extraction_method": "full_v3_from_config",
                "random_init": True,
                "extracted_at": datetime.now().isoformat(),
            })
            with open(json_path, "w") as f:
                json.dump(data, f, indent=2)

        # 清理临时目录
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

        return True, "OK"

    except Exception as e:
        import traceback
        error_msg = str(e)[:100]
        print(f"✗ 失败: {error_msg}")
        # 打印完整错误用于调试
        # traceback.print_exc()
        return False, error_msg

def main():
    parser = argparse.ArgumentParser(description="完整计算图抽取 V3 - from_config版本")
    parser.add_argument("--list", required=True, help="模型列表文件")
    parser.add_argument("--task", required=True, help="任务类型")
    parser.add_argument("--device", default="cpu", help="设备")
    parser.add_argument("--dynamic", action="store_true", help="动态形状")
    parser.add_argument("--max-length", type=int, default=128, help="最大长度")
    parser.add_argument("--agent-id", default="full-v3", help="Agent ID")

    args = parser.parse_args()

    # 读取模型列表
    with open(args.list) as f:
        models = [line.strip() for line in f if line.strip()]

    print(f"[{args.agent_id}] 开始抽取: {len(models)} 个模型")
    print(f"  任务类型: {args.task}")
    print(f"  设备: {args.device}")
    print(f"  方法: from_config()随机初始化")
    print(f"  HF_ENDPOINT: {os.environ.get('HF_ENDPOINT', '默认')}")
    print("="*60)

    success = 0
    failed = 0

    for i, model_id in enumerate(models, 1):
        print(f"\n[{i}/{len(models)}] {model_id}")
        ok, msg = extract_model_full(model_id, args.task, args.device, args.dynamic, args.max_length)
        if ok:
            success += 1
        else:
            failed += 1

    print(f"\n{'='*60}")
    print(f"完成: 成功 {success}, 失败 {failed}")

    # 保存结果
    result = {
        "agent_id": args.agent_id,
        "task": args.task,
        "total": len(models),
        "success": success,
        "failed": failed,
        "timestamp": datetime.now().isoformat(),
    }
    result_file = f"/work/graphnet_workspace/batch_reextract/result_{args.agent_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"结果已保存: {result_file}")

if __name__ == "__main__":
    main()

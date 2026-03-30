#!/usr/bin/env python3
"""
完整计算图抽取脚本 V4 - 离线模式
- 使用本地已下载的config.json
- 使用from_config()随机初始化，不下载权重
- 使用正确的GraphNet extract() wrapper
- 生成完整的model.py, weight_meta.py等文件
- 完全离线，无需网络
"""

import os
import sys
import json
import argparse
import torch
import shutil
from datetime import datetime
from pathlib import Path

# 强制设置代理（必须显式设置，不要依赖setdefault）
os.environ["http_proxy"] = "http://agent.baidu.com:8891"
os.environ["https_proxy"] = "http://agent.baidu.com:8891"

# 离线模式
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

sys.path.insert(0, "/root/GraphNet")
from graph_net.torch.extractor import extract

try:
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoModelForTokenClassification
    from transformers import AutoModelForQuestionAnswering, AutoModelForMaskedLM, AutoModelForCausalLM
    from transformers import AutoConfig
except ImportError as e:
    print(f"错误: transformers库未安装 - {e}")
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
    # Only replace / with _, keep - as is to match directory naming
    return model_id.replace("/", "_")

def extract_model_offline(model_id, task_type, device="cpu", dynamic=False, max_length=128):
    """使用本地config进行GraphNet抽取"""
    try:
        print(f"  [抽取] {model_id}", flush=True)

        # 设置workspace
        workspace = get_workspace(task_type)
        os.environ["GRAPH_NET_EXTRACT_WORKSPACE"] = workspace

        safe_name = safe_model_name(model_id)
        model_dir = os.path.join(workspace, safe_name)

        # 检查本地config是否存在
        config_path = os.path.join(model_dir, "config.json")
        if not os.path.exists(config_path):
            print(f"    ✗ 跳过: 无config.json")
            return False, "无config.json"

        # 检查是否已完整抽取
        if os.path.exists(os.path.join(model_dir, "model.py")) and \
           os.path.exists(os.path.join(model_dir, "weight_meta.py")):
            print(f"    ✓ 已完整抽取，跳过")
            return True, "已存在"

        # 从本地config加载
        print(f"    加载config...", end=" ", flush=True)
        model_class = TASK_MODEL_MAP.get(task_type, AutoModel)
        config = AutoConfig.from_pretrained(config_path, trust_remote_code=True, local_files_only=True)
        print(f"OK", end=" ", flush=True)

        # 使用from_config随机初始化
        print(f"初始化模型...", end=" ", flush=True)
        model = model_class.from_config(config, trust_remote_code=True)
        model = model.to(device).eval()
        print(f"OK", end=" ", flush=True)

        # 从config获取模型信息
        vocab_size = getattr(config, 'vocab_size', None) or \
                     getattr(config, 'vocab_size_or_config_json_file', None) or \
                     getattr(config, 'n_positions', 32000) or 32000

        # 创建dummy输入
        print(f"准备输入...", end=" ", flush=True)
        dummy_input_ids = torch.randint(0, min(vocab_size, 1000), (1, max_length)).to(device)
        inputs = {"input_ids": dummy_input_ids}
        inputs["attention_mask"] = torch.ones_like(dummy_input_ids)
        print(f"OK", end=" ", flush=True)

        # 使用GraphNet extract抽取
        print(f"抽取中...", end=" ", flush=True)
        wrapped = extract(name=safe_name, dynamic=dynamic)(model).eval()

        with torch.no_grad():
            wrapped(**inputs)

        print(f"✓ 完成", flush=True)

        # 验证输出文件（可能在subgraph_*子目录中）
        output_dir = os.path.join(workspace, safe_name)

        # 查找所有子目录中的model.py
        model_py_files = list(Path(output_dir).rglob("model.py"))
        weight_meta_files = list(Path(output_dir).rglob("weight_meta.py"))
        input_meta_files = list(Path(output_dir).rglob("input_meta.py"))

        if not model_py_files:
            return False, "缺少model.py"

        if not weight_meta_files:
            return False, "缺少weight_meta.py"

        if not input_meta_files:
            return False, "缺少input_meta.py"

        # 更新元数据
        json_path = os.path.join(output_dir, "graph_net.json")
        if os.path.exists(json_path):
            with open(json_path) as f:
                data = json.load(f)
            data.update({
                "model_id": model_id,
                "source": "huggingface",
                "task": task_type,
                "max_length": max_length,
                "extraction_method": "offline_v4_from_config",
                "random_init": True,
                "extracted_at": datetime.now().isoformat(),
            })
            with open(json_path, "w") as f:
                json.dump(data, f, indent=2)

        return True, "OK"

    except Exception as e:
        import traceback
        error_msg = str(e)[:100]
        print(f"✗ 失败: {error_msg}")
        # traceback.print_exc()
        return False, error_msg

def main():
    parser = argparse.ArgumentParser(description="离线计算图抽取 V4")
    parser.add_argument("--list", required=True, help="模型列表文件")
    parser.add_argument("--task", required=True, help="任务类型")
    parser.add_argument("--device", default="cpu", help="设备")
    parser.add_argument("--dynamic", action="store_true", help="动态形状")
    parser.add_argument("--max-length", type=int, default=128, help="最大长度")
    parser.add_argument("--agent-id", default="offline-v4", help="Agent ID")

    args = parser.parse_args()

    # 读取模型列表
    with open(args.list) as f:
        models = [line.strip() for line in f if line.strip()]

    print(f"[{args.agent_id}] 开始离线抽取: {len(models)} 个模型")
    print(f"  任务类型: {args.task}")
    print(f"  设备: {args.device}")
    print(f"  方法: 离线from_config()随机初始化")
    print("="*60)

    success = 0
    failed = 0
    skipped = 0

    for i, model_id in enumerate(models, 1):
        print(f"\n[{i}/{len(models)}] {model_id}")
        ok, msg = extract_model_offline(model_id, args.task, args.device, args.dynamic, args.max_length)
        if ok:
            if msg == "已存在":
                skipped += 1
            else:
                success += 1
        else:
            failed += 1

    print(f"\n{'='*60}")
    print(f"完成: 成功 {success}, 跳过 {skipped}, 失败 {failed}")

    # 保存结果
    result = {
        "agent_id": args.agent_id,
        "task": args.task,
        "total": len(models),
        "success": success,
        "skipped": skipped,
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

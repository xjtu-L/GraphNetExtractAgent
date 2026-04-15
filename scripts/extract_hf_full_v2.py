#!/usr/bin/env python3
"""
完整计算图抽取脚本 V2
- 使用正确的GraphNet extract() wrapper
- 生成完整的model.py, weight_meta.py等文件
- 支持多种任务类型
"""

import os
import sys
import json
import argparse
import torch
from datetime import datetime

# 强制设置代理（必须显式设置，不要依赖setdefault）
os.environ["http_proxy"] = "http://agent.baidu.com:8891"
os.environ["https_proxy"] = "http://agent.baidu.com:8891"
os.environ["HTTP_PROXY"] = "http://agent.baidu.com:8891"
os.environ["HTTPS_PROXY"] = "http://agent.baidu.com:8891"

# 设置HF缓存目录
os.environ["HF_HOME"] = "/work/.cache/huggingface"

sys.path.insert(0, "/root/GraphNet")
from graph_net.torch.extractor import extract

try:
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoModelForTokenClassification
    from transformers import AutoModelForQuestionAnswering, AutoModelForMaskedLM, AutoModelForCausalLM
    from transformers import AutoTokenizer, AutoConfig
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
    base = "/ssd1/liangtai-work/graphnet_workspace/huggingface"
    workspace = os.path.join(base, task_type)
    os.makedirs(workspace, exist_ok=True)
    return workspace

def safe_model_name(model_id):
    return model_id.replace("/", "_").replace("-", "_")

def extract_model_full(model_id, task_type, device="cpu", dynamic=False, max_length=128):
    """使用完整GraphNet抽取流程"""
    try:
        print(f"  [抽取] {model_id} ({task_type})", flush=True)
        
        # 设置workspace
        workspace = get_workspace(task_type)
        os.environ["GRAPH_NET_EXTRACT_WORKSPACE"] = workspace
        
        # 加载模型类
        model_class = TASK_MODEL_MAP.get(task_type, AutoModel)
        
        # 加载模型（使用预训练权重以获取完整结构）
        print(f"    加载模型...", end=" ", flush=True)
        model = model_class.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            trust_remote_code=True
        ).to(device).eval()
        print(f"OK", end=" ", flush=True)
        
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        
        # 构造输入
        dummy_text = "This is a sample text for model extraction."
        inputs = tokenizer(
            dummy_text,
            return_tensors="pt",
            padding="max_length",
            max_length=max_length,
            truncation=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        print(f"Tokenized", end=" ", flush=True)
        
        # 使用GraphNet extract抽取
        model_name_safe = safe_model_name(model_id)
        wrapped = extract(name=model_name_safe, dynamic=dynamic)(model).eval()
        
        with torch.no_grad():
            wrapped(**inputs)
        
        print(f"✓ 完成", flush=True)
        
        # 更新元数据
        output_dir = os.path.join(workspace, model_name_safe)
        json_path = os.path.join(output_dir, "graph_net.json")
        if os.path.exists(json_path):
            with open(json_path) as f:
                data = json.load(f)
            data.update({
                "model_id": model_id,
                "source": "huggingface",
                "task": task_type,
                "max_length": max_length,
                "extraction_method": "full_v2",
                "extracted_at": datetime.now().isoformat(),
            })
            with open(json_path, "w") as f:
                json.dump(data, f, indent=2)
        
        return True, "OK"
        
    except Exception as e:
        error_msg = str(e)[:100]
        print(f"✗ 失败: {error_msg}")
        return False, error_msg

def main():
    parser = argparse.ArgumentParser(description="完整计算图抽取 V2")
    parser.add_argument("--list", required=True, help="模型列表文件")
    parser.add_argument("--task", required=True, help="任务类型")
    parser.add_argument("--device", default="cpu", help="设备")
    parser.add_argument("--dynamic", action="store_true", help="动态形状")
    parser.add_argument("--max-length", type=int, default=128, help="最大长度")
    parser.add_argument("--agent-id", default="full-v2", help="Agent ID")
    
    args = parser.parse_args()
    
    # 读取模型列表
    with open(args.list) as f:
        models = [line.strip() for line in f if line.strip()]
    
    print(f"[{args.agent_id}] 开始抽取: {len(models)} 个模型")
    print(f"  任务类型: {args.task}")
    print(f"  设备: {args.device}")
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

if __name__ == "__main__":
    main()

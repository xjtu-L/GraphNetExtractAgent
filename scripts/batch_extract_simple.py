#!/usr/bin/env python3
"""
分批抽取 timm 模型 - 每次处理一个模型，避免超时
"""

import os
import sys
import json
import torch
import timm
from datetime import datetime

# 设置环境变量
os.environ["http_proxy"] = "http://agent.baidu.com:8891"
os.environ["https_proxy"] = "http://agent.baidu.com:8891"
os.environ["GRAPH_NET_EXTRACT_WORKSPACE"] = "/root/graphnet_workspace"

from graph_net.torch.extractor import extract

BATCH_FILE = "/root/GraphNetExtractAgent/batch_missing_00"
RESULT_DIR = "/root/GraphNetExtractAgent"


def load_models():
    with open(BATCH_FILE) as f:
        return [line.strip() for line in f if line.strip()]


def get_extracted_models():
    """获取已成功抽取的模型"""
    extracted = set()
    workspace = os.environ["GRAPH_NET_EXTRACT_WORKSPACE"]
    for name in os.listdir(workspace):
        model_dir = os.path.join(workspace, name)
        json_path = os.path.join(model_dir, "graph_net.json")
        if os.path.isdir(model_dir) and os.path.exists(json_path):
            extracted.add(name)
    return extracted


def get_input_size(model):
    """获取模型输入尺寸"""
    cfg = model.default_cfg
    input_size = cfg.get("input_size", (3, 224, 224))
    return input_size


def extract_model(name, device="cpu"):
    """抽取单个模型"""
    try:
        # 创建模型
        model = timm.create_model(name, pretrained=False).to(device).eval()
        input_size = get_input_size(model)

        # 构造输入
        input_data = torch.rand(1, *input_size, device=device)

        # 抽取
        wrapped = extract(name=name, dynamic=False)(model).eval()
        with torch.no_grad():
            wrapped(input_data)

        # 补充 graph_net.json 字段
        output_dir = os.path.join(os.environ["GRAPH_NET_EXTRACT_WORKSPACE"], name)
        json_path = os.path.join(output_dir, "graph_net.json")
        if os.path.exists(json_path):
            with open(json_path) as f:
                data = json.load(f)
            data["source"] = "timm"
            data["heuristic_tag"] = "computer_vision"
            with open(json_path, "w") as f:
                json.dump(data, f, indent=4)

        return True, f"OK (input_size={input_size})"

    except Exception as e:
        return False, str(e)


def main():
    all_models = load_models()
    extracted = get_extracted_models()

    # 找出待处理的模型
    pending = [m for m in all_models if m not in extracted]

    print(f"总模型数: {len(all_models)}")
    print(f"已抽取: {len(extracted)}")
    print(f"待处理: {len(pending)}")
    print("-" * 50)

    if not pending:
        print("所有模型已处理完成!")
        return

    success_list = []
    failed_list = []

    for i, name in enumerate(pending):
        print(f"[{i+1}/{len(pending)}] {name}...", end=" ", flush=True)
        ok, msg = extract_model(name, device="cpu")
        if ok:
            success_list.append(name)
            print(f"OK - {msg}")
        else:
            failed_list.append((name, msg[:200]))
            print(f"FAIL - {msg[:80]}")

        # 每20个模型保存一次进度
        if (i + 1) % 20 == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_file = os.path.join(RESULT_DIR, f"batch_missing_00_progress_{timestamp}.json")
            with open(result_file, "w") as f:
                json.dump({
                    "processed": i + 1,
                    "total_pending": len(pending),
                    "success": success_list,
                    "failed": failed_list
                }, f, indent=2)
            print(f"进度已保存: {len(success_list)} 成功, {len(failed_list)} 失败")

    # 保存最终结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(RESULT_DIR, f"batch_missing_00_final_{timestamp}.json")

    final_result = {
        "total_models": len(all_models),
        "already_extracted_before": len(extracted),
        "newly_extracted": len(success_list),
        "failed_count": len(failed_list),
        "success_list": success_list,
        "failed_list": failed_list
    }

    with open(result_file, "w") as f:
        json.dump(final_result, f, indent=2)

    print(f"\n最终结果已保存到: {result_file}")
    print(f"成功: {len(success_list)}, 失败: {len(failed_list)}")


if __name__ == "__main__":
    main()

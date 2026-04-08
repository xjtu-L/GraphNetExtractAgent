#!/usr/bin/env python3
"""
Torchvision 模型后处理脚本

用途：
    修复已抽取的 torchvision 模型的存储位置和元数据字段

    1. 将 workspace 根目录下的 torchvision 模型移动到 torchvision 子目录
    2. 为 graph_net.json 补充 source 和 heuristic_tag 字段

环境变量：
    GRAPH_NET_EXTRACT_WORKSPACE  抽取输出目录
"""

import os
import json
import shutil


def fix_torchvision_models():
    workspace = os.environ.get("GRAPH_NET_EXTRACT_WORKSPACE", "/ssd1/liangtai-work/graphnet_workspace")

    # 创建 torchvision 子目录
    torchvision_dir = os.path.join(workspace, "torchvision")
    os.makedirs(torchvision_dir, exist_ok=True)

    # 获取 workspace 根目录下的所有模型
    all_models = [d for d in os.listdir(workspace) if os.path.isdir(os.path.join(workspace, d))]

    # 排除已分类的子目录
    exclude_dirs = ["timm", "torchvision", "bad_cases"]

    fixed_count = 0
    for model_name in all_models:
        if model_name in exclude_dirs:
            continue

        model_path = os.path.join(workspace, model_name)
        json_path = os.path.join(model_path, "graph_net.json")

        # 检查是否是 torchvision 模型（通过 graph_net.json 中的 model_name 判断）
        if os.path.exists(json_path):
            with open(json_path) as f:
                try:
                    data = json.load(f)
                except:
                    continue

            # 如果没有 source 字段， 可能是 torchvision 模型
            if "source" not in data:
                # 移动到 torchvision 子目录
                new_path = os.path.join(torchvision_dir, model_name)
                shutil.move(model_path, new_path)

                # 补充字段
                data["source"] = "torchvision"
                data["heuristic_tag"] = "computer_vision"

                # 保存修改后的 JSON
                new_json_path = os.path.join(new_path, "graph_net.json")
                with open(new_json_path, "w") as f:
                    json.dump(data, f, indent=4)

                fixed_count += 1
                print(f"[FIX] {model_name}: moved and fields added")

    print(f"\n总计修复 {fixed_count} 个 torchvision 模型")


if __name__ == "__main__":
    fix_torchvision_models()

#!/usr/bin/env python3
"""
批量抽取 batch_missing_00 模型，每次处理10个，记录进度
"""

import os
import sys
import json
import subprocess
from datetime import datetime

# 设置环境变量
os.environ["http_proxy"] = "http://agent.baidu.com:8891"
os.environ["https_proxy"] = "http://agent.baidu.com:8891"
os.environ["GRAPH_NET_EXTRACT_WORKSPACE"] = "/root/graphnet_workspace"

BATCH_FILE = "/root/GraphNetExtractAgent/batch_missing_00"
RESULT_DIR = "/root/GraphNetExtractAgent"
BATCH_SIZE = 10


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

    # 处理待抽取模型
    success_total = []
    failed_total = []

    for i in range(0, len(pending), BATCH_SIZE):
        batch = pending[i:i+BATCH_SIZE]
        print(f"\n处理批次 {i//BATCH_SIZE + 1}: {len(batch)} 个模型")
        print(f"模型: {', '.join(batch)}")

        # 构建命令
        cmd = [
            sys.executable,
            "/root/GraphNetExtractAgent/scripts/extract_timm_batch.py",
            "--list", BATCH_FILE,
            "--device", "cpu",
            "--start", "0",
            "--count", str(len(all_models))  # 处理全部，脚本会跳过已存在的
        ]

        # 运行抽取
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr[:500])

        # 更新已抽取列表
        new_extracted = get_extracted_models()
        new_success = [m for m in batch if m in new_extracted]
        new_failed = [m for m in batch if m not in new_extracted]

        success_total.extend(new_success)
        failed_total.extend(new_failed)

        print(f"\n批次结果: 成功 {len(new_success)}, 失败 {len(new_failed)}")

    # 保存最终结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(RESULT_DIR, f"batch_missing_00_result_{timestamp}.json")

    final_result = {
        "total": len(all_models),
        "already_extracted": len(extracted),
        "newly_extracted": len(success_total),
        "failed": failed_total,
        "success_list": success_total,
        "failed_list": failed_total
    }

    with open(result_file, "w") as f:
        json.dump(final_result, f, indent=2)

    print(f"\n最终结果已保存到: {result_file}")


if __name__ == "__main__":
    main()

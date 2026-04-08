#!/usr/bin/env python3
"""
验证已抽取的计算图样本

用法:
    # 验证单个模型
    python validate_sample.py --model resnet18

    # 验证多个模型
    python validate_sample.py --model swin_s swin_t

    # 指定 workspace 路径
    python validate_sample.py --model resnet18 --workspace /ssd1/liangtai-work/graphnet_workspace

    # 跳过冗余检查
    python validate_sample.py --model resnet18 --no-check-redundancy

环境变量:
    GRAPH_NET_EXTRACT_WORKSPACE  抽取输出目录 (必须设置)
"""

import argparse
import os
import sys
import subprocess


def validate_model(model_name, workspace, no_check_redundancy=False):
    """验证单个模型的抽取结果"""
    model_path = os.path.join(workspace, model_name)

    if not os.path.isdir(model_path):
        print(f"[SKIP] {model_name}: directory not found at {model_path}")
        return False

    # 检查必要文件
    required_files = ["model.py", "graph_net.json"]
    for f in required_files:
        if not os.path.isfile(os.path.join(model_path, f)):
            print(f"[FAIL] {model_name}: missing {f}")
            return False

    print(f"\n{'='*60}")
    print(f"Validating: {model_name}")
    print(f"  Path: {model_path}")

    cmd = [
        sys.executable, "-m", "graph_net.torch.validate",
        "--model-path", model_path,
    ]
    if no_check_redundancy:
        cmd.append("--no-check-redundancy")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1000)
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        if result.returncode == 0:
            print(f"[OK] {model_name}: validation passed")
            return True
        else:
            print(f"[FAIL] {model_name}: validation failed (exit code {result.returncode})")
            return False
    except subprocess.TimeoutExpired:
        print(f"[FAIL] {model_name}: validation timed out (1000s)")
        return False
    except Exception as e:
        print(f"[FAIL] {model_name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="验证抽取的计算图样本")
    parser.add_argument("--model", nargs="+", required=True, help="模型名称列表")
    parser.add_argument("--workspace", default=None, help="覆盖 GRAPH_NET_EXTRACT_WORKSPACE")
    parser.add_argument("--no-check-redundancy", action="store_true", help="跳过冗余检查")
    args = parser.parse_args()

    if args.workspace:
        os.environ["GRAPH_NET_EXTRACT_WORKSPACE"] = args.workspace
    workspace = os.environ.get("GRAPH_NET_EXTRACT_WORKSPACE")
    if not workspace:
        print("ERROR: GRAPH_NET_EXTRACT_WORKSPACE 环境变量未设置")
        sys.exit(1)

    success, fail = [], []
    for name in args.model:
        ok = validate_model(name, workspace, args.no_check_redundancy)
        (success if ok else fail).append(name)

    print(f"\n{'='*60}")
    print(f"Validation Results: {len(success)} OK, {len(fail)} FAIL")
    if success:
        print(f"  OK: {', '.join(success)}")
    if fail:
        print(f"  FAIL: {', '.join(fail)}")

    return 0 if not fail else 1


if __name__ == "__main__":
    sys.exit(main())

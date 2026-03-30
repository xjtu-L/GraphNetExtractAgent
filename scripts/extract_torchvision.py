#!/usr/bin/env python3
"""
torchvision 模型计算图抽取脚本

用法:
    # 抽取单个图像分类模型 (默认 224x224)
    python extract_torchvision.py --models swin_s swin_t

    # 抽取语义分割模型 (指定输入尺寸)
    python extract_torchvision.py --models lraspp_mobilenet_v3_large --height 520 --width 520 --type segmentation

    # 抽取视频模型 (5D 输入)
    python extract_torchvision.py --models r2plus1d_18 s3d --type video --frames 16

    # 抽取目标检测模型
    python extract_torchvision.py --models fasterrcnn_resnet50_fpn --type detection

    # 抽取光流模型
    python extract_torchvision.py --models raft_large raft_small --type opticalflow

    # 不使用预训练权重
    python extract_torchvision.py --models swin_s --no-pretrained

环境变量:
    GRAPH_NET_EXTRACT_WORKSPACE  抽取输出目录 (必须设置)
    http_proxy / https_proxy     代理 (下载权重时需要)
"""

import argparse
import os
import sys
import torch
from torchvision.models import get_model, get_model_weights
import graph_net.torch


def build_input(model_type, height, width, channels, frames, device):
    """根据模型类型构造输入数据"""
    if model_type == "classification" or model_type == "segmentation":
        # (B, C, H, W)
        return torch.rand(1, channels, height, width, device=device)

    elif model_type == "video":
        # (B, C, T, H, W)
        return torch.rand(1, channels, frames, height, width, device=device)

    elif model_type == "detection":
        # 检测模型接受图像列表
        return [torch.rand(channels, height, width, device=device)]

    elif model_type == "opticalflow":
        # 光流模型接受两张图像
        img1 = torch.rand(1, channels, height, width, device=device)
        img2 = torch.rand(1, channels, height, width, device=device)
        return (img1, img2)

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def extract_model(name, model_type, height, width, channels, frames, use_pretrained, device_str):
    """对单个模型执行 extract"""
    device = torch.device(device_str)
    print(f"\n{'='*60}")
    print(f"Extracting: {name}")
    print(f"  Type: {model_type}, Input: ", end="")

    # 1. 加载模型
    weights = None
    if use_pretrained:
        try:
            weights = get_model_weights(name).DEFAULT
            print(f"(pretrained weights)")
        except Exception:
            print(f"(no pretrained weights available, using random init)")
    else:
        print(f"(random init)")

    try:
        model = get_model(name, weights=weights).to(device).eval()
    except Exception as e:
        print(f"[FAIL] {name}: cannot instantiate model - {e}")
        return False

    # 2. 构造输入
    input_data = build_input(model_type, height, width, channels, frames, device)
    if model_type == "classification" or model_type == "segmentation":
        print(f"  Shape: {input_data.shape}")
    elif model_type == "video":
        print(f"  Shape: {input_data.shape}")
    elif model_type == "detection":
        print(f"  Shape: list of [{input_data[0].shape}]")
    elif model_type == "opticalflow":
        print(f"  Shape: pair of [{input_data[0].shape}]")

    # 3. extract
    wrapped = graph_net.torch.extract(name=name, dynamic=True)(model).eval()
    try:
        with torch.no_grad():
            if model_type == "opticalflow":
                wrapped(*input_data)
            elif model_type == "detection":
                wrapped(input_data)
            else:
                wrapped(input_data)
        print(f"[OK] {name}: extract succeeded")
        return True
    except Exception as e:
        print(f"[FAIL] {name}: extract error - {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="torchvision 模型计算图抽取")
    parser.add_argument("--models", nargs="+", required=True, help="模型名称列表")
    parser.add_argument("--type", default="classification",
                        choices=["classification", "segmentation", "video", "detection", "opticalflow"],
                        help="模型类型 (默认: classification)")
    parser.add_argument("--height", type=int, default=224, help="输入高度 (默认: 224)")
    parser.add_argument("--width", type=int, default=224, help="输入宽度 (默认: 224)")
    parser.add_argument("--channels", type=int, default=3, help="输入通道数 (默认: 3)")
    parser.add_argument("--frames", type=int, default=16, help="视频帧数 (默认: 16)")
    parser.add_argument("--no-pretrained", action="store_true", help="不使用预训练权重")
    parser.add_argument("--device", default="cpu", help="设备 (默认: cpu)")
    parser.add_argument("--workspace", default=None, help="覆盖 GRAPH_NET_EXTRACT_WORKSPACE")
    args = parser.parse_args()

    # 设置 workspace
    if args.workspace:
        os.environ["GRAPH_NET_EXTRACT_WORKSPACE"] = args.workspace
    workspace = os.environ.get("GRAPH_NET_EXTRACT_WORKSPACE")
    if not workspace:
        print("ERROR: GRAPH_NET_EXTRACT_WORKSPACE 环境变量未设置")
        sys.exit(1)
    print(f"Workspace: {workspace}")

    use_pretrained = not args.no_pretrained
    success, fail = [], []

    for name in args.models:
        ok = extract_model(
            name=name,
            model_type=args.type,
            height=args.height,
            width=args.width,
            channels=args.channels,
            frames=args.frames,
            use_pretrained=use_pretrained,
            device_str=args.device,
        )
        (success if ok else fail).append(name)

    # 汇总
    print(f"\n{'='*60}")
    print(f"Results: {len(success)} OK, {len(fail)} FAIL")
    if success:
        print(f"  OK: {', '.join(success)}")
    if fail:
        print(f"  FAIL: {', '.join(fail)}")

    return 0 if not fail else 1


if __name__ == "__main__":
    sys.exit(main())

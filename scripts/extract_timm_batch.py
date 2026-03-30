#!/usr/bin/env python3
"""
timm 模型批量抽取脚本

用法:
    python extract_timm_batch.py --batch standard   # 标准图像模型
    python extract_timm_batch.py --batch vit        # ViT 家族
    python extract_timm_batch.py --models model1 model2  # 指定模型
    python extract_timm_batch.py --list missing.txt --start 0 --count 100  # 从列表抽取

特性:
    - 抽取前检查 samples/timm/ 目录，排除已存在的模型
    - 名称互相包含时（如 resnetrs101 和 resnetrs101.tf_in1k），只保留规范名称
"""

import os
import sys
import json
import argparse
import torch
import timm
from datetime import datetime

# 设置环境变量
os.environ.setdefault("GRAPH_NET_EXTRACT_WORKSPACE", "/root/graphnet_workspace/timm")

from graph_net.torch.extractor import extract

# 已存在模型的目录
SAMPLES_DIR = "/root/GraphNet/samples/timm"

# 标准图像家族（纯 4D 输入，无额外参数）
STANDARD_FAMILIES = [
    "efficientnet", "convnext", "resnetv2", "regnety", "regnetx",
    "mobilenetv4", "repvgg", "hrnet", "nf", "seresnet", "senet",
    "res2net", "resnest", "sknet", "dla", "mobilenetv2", "mobilenetv3"
]

# ViT 家族（可能有 cls_token 额外输入）
VIT_FAMILIES = ["vit"]

# Swin 家族（注意分辨率）
SWIN_FAMILIES = ["swinv2"]

# 其他复杂家族
COMPLEX_FAMILIES = ["xcit", "maxvit", "levit", "vitamin", "eva02", "fastvit", "focalnet", "efficientvit"]


def get_family(model_name):
    """提取模型家族名"""
    return model_name.split("_")[0]


def get_existing_models():
    """获取已存在的模型列表（包括 samples 和 workspace）"""
    existing = set()

    # 检查 samples 目录
    if os.path.isdir(SAMPLES_DIR):
        for model in os.listdir(SAMPLES_DIR):
            model_path = os.path.join(SAMPLES_DIR, model)
            if os.path.isdir(model_path):
                existing.add(model)

    # 检查 workspace 目录
    workspace = os.environ.get("GRAPH_NET_EXTRACT_WORKSPACE", "")
    if os.path.isdir(workspace):
        for model in os.listdir(workspace):
            model_path = os.path.join(workspace, model)
            if os.path.isdir(model_path):
                existing.add(model)

    return existing


def is_model_exists(model_name, existing_models):
    """检查模型是否已存在（包括名称包含关系检查）"""
    # 直接匹配
    if model_name in existing_models:
        return True

    # 检查名称包含关系（如 resnetrs101 vs resnetrs101.tf_in1k）
    for existing in existing_models:
        # 规范名称已存在（当前模型是带后缀的版本）
        if model_name.startswith(existing + ".") or model_name.startswith(existing + "_"):
            return True
        # 带后缀版本已存在（当前模型是规范名称）
        if existing.startswith(model_name + ".") or existing.startswith(model_name + "_"):
            return True

    return False


def filter_missing_models(target_models, existing_models):
    """过滤掉已存在的模型"""
    missing = []
    skipped = []

    for model in target_models:
        if is_model_exists(model, existing_models):
            skipped.append(model)
        else:
            missing.append(model)

    return missing, skipped


def load_missing_list(path):
    """加载待抽取模型列表"""
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def get_input_size(model):
    """获取模型输入尺寸"""
    cfg = model.default_cfg
    input_size = cfg.get("input_size", (3, 224, 224))
    return input_size


def generate_extract_script(output_dir, model_name, source, heuristic_tag, input_shape=(1, 3, 224, 224)):
    """生成 extract.py 脚本，用于复现计算图抓取"""
    script_content = f'''#!/usr/bin/env python3
"""
计算图抓取复现脚本

此脚本用于复现模型的计算图抓取过程。
包含所有必要的参数，独立执行抓取生成 graph_net.json。

用法:
    python extract.py                    # 执行抓取 (使用预训练权重)
    python extract.py --no-pretrained     # 不使用预训练权重 (网络受限时)
    python extract.py --output /path/to  # 指定输出目录
    python extract.py --dynamic          # 使用动态形状
    python extract.py --help             # 显示帮助

环境变量:
    HF_ENDPOINT: HuggingFace镜像地址 (默认: https://hf-mirror.com)
    HF_HUB_ENABLE_HF_TRANSFER=1 可加速下载

环境要求:
    - graph_net 库已安装
    - timm 或 torchvision 库已安装
"""

import os
import sys
import json
import argparse
import torch

# ============ HuggingFace 镜像设置 ============
# 使用镜像站加速下载
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# ============ 模型参数 ============
MODEL_NAME = "{model_name}"
SOURCE = "{source}"
HEURISTIC_TAG = "{heuristic_tag}"
# 输入形状: (batch, channels, height, width)
INPUT_SHAPE = {input_shape}
# 是否使用动态形状
DYNAMIC = False

def get_model_creator(pretrained=True):
    """获取模型创建函数"""
    if SOURCE == "timm":
        import timm
        return lambda: timm.create_model(MODEL_NAME, pretrained=pretrained)
    elif SOURCE == "torchvision":
        import torchvision.models as models
        model_func = getattr(models, MODEL_NAME, None)
        if model_func is None:
            raise ValueError(f"Model {{MODEL_NAME}} not found in torchvision")
        if pretrained:
            weights_name = f"{{MODEL_NAME[0].upper()}}{{MODEL_NAME[1:]}}_Weights"
            weights = getattr(models, weights_name, None)
            if weights is not None:
                return lambda: model_func(weights=weights.DEFAULT)
        return lambda: model_func(pretrained=pretrained)
    else:
        raise ValueError(f"Unknown source: {{SOURCE}}")

def extract_graph(output_dir=None, dynamic=False, input_shape=None, pretrained=True):
    """执行计算图抓取"""
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))

    if input_shape is None:
        input_shape = INPUT_SHAPE

    print(f"提取模型: {{MODEL_NAME}}")
    print(f"来源: {{SOURCE}}")
    print(f"输入形状: {{input_shape}}")
    print(f"动态形状: {{dynamic}}")
    print(f"预训练权重: {{pretrained}}")
    print(f"输出目录: {{output_dir}}")

    # 创建模型
    model = get_model_creator(pretrained=pretrained)()
    model.eval()

    # 构造输入数据
    input_data = torch.rand(*input_shape)

    # 使用 graph_net 提取
    from graph_net.torch.extractor import extract

    MODEL_NAME_safe = MODEL_NAME if pretrained else f"{{MODEL_NAME}}_no_pretrained"
    wrapped = extract(name=MODEL_NAME_safe, dynamic=dynamic)(model).eval()
    with torch.no_grad():
        wrapped(input_data)

    # 更新 graph_net.json
    json_path = os.path.join(output_dir, "graph_net.json")
    if os.path.exists(json_path):
        with open(json_path) as f:
            data = json.load(f)
    else:
        data = {{}}

    data["model_name"] = MODEL_NAME
    data["source"] = SOURCE
    data["heuristic_tag"] = HEURISTIC_TAG
    data["input_shape"] = list(input_shape)
    data["pretrained"] = pretrained

    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"抓取完成! 输出目录: {{output_dir}}")
    return True

def main():
    parser = argparse.ArgumentParser(
        description=f"复现 {{MODEL_NAME}} 的计算图抓取"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="输出目录 (默认: 脚本所在目录)"
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        default=DYNAMIC,
        help="使用动态形状"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=INPUT_SHAPE[0],
        help=f"批次大小 (默认: {{INPUT_SHAPE[0]}})"
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="不使用预训练权重 (网络受限时使用)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只显示信息，不执行抓取"
    )

    args = parser.parse_args()

    if args.dry_run:
        print(f"模型名称: {{MODEL_NAME}}")
        print(f"来源: {{SOURCE}}")
        print(f"输入形状: {{INPUT_SHAPE}}")
        print(f"动态形状: {{args.dynamic}}")
        print(f"预训练权重: {{not args.no_pretrained}}")
        return

    # 更新批次大小
    batch_shape = (args.batch_size,) + INPUT_SHAPE[1:]
    pretrained = not args.no_pretrained

    extract_graph(args.output, args.dynamic, batch_shape, pretrained)

if __name__ == "__main__":
    main()
'''
    script_path = os.path.join(output_dir, "extract.py")
    with open(script_path, "w") as f:
        f.write(script_content)


def extract_model(name, device="cpu", dynamic=False):
    """抽取单个模型"""
    try:
        # 创建模型
        model = timm.create_model(name, pretrained=False).to(device).eval()
        input_size = get_input_size(model)

        # 构造输入
        input_data = torch.rand(1, *input_size, device=device)

        # 抽取
        wrapped = extract(name=name, dynamic=dynamic)(model).eval()
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

        # 生成 extract.py 脚本
        input_shape = (1,) + input_size  # 添加 batch 维度
        generate_extract_script(output_dir, name, "timm", "computer_vision", input_shape)

        return True, f"OK (input_size={input_size})"

    except Exception as e:
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(description="timm 模型批量抽取")
    parser.add_argument("--batch", choices=["standard", "vit", "swin", "complex", "all"],
                        help="按批次类型抽取")
    parser.add_argument("--models", nargs="+", help="指定模型列表")
    parser.add_argument("--list", help="从文件读取模型列表")
    parser.add_argument("--start", type=int, default=0, help="起始索引")
    parser.add_argument("--count", type=int, default=100, help="抽取数量")
    parser.add_argument("--device", default="cpu", help="设备")
    parser.add_argument("--dynamic", action="store_true", help="使用动态形状")

    args = parser.parse_args()

    # 获取待抽取模型列表
    missing_path = "/root/GraphNetExtractAgent/timm_missing_models.txt"
    all_missing = load_missing_list(missing_path)

    if args.models:
        target_models = args.models
    elif args.batch:
        families = {
            "standard": STANDARD_FAMILIES,
            "vit": VIT_FAMILIES,
            "swin": SWIN_FAMILIES,
            "complex": COMPLEX_FAMILIES,
            "all": None
        }
        target_families = families[args.batch]
        if target_families:
            target_models = [m for m in all_missing if get_family(m) in target_families]
        else:
            target_models = all_missing
    elif args.list:
        target_models = load_missing_list(args.list)
    else:
        target_models = all_missing

    # 应用 start 和 count
    target_models = target_models[args.start:args.start + args.count]

    # 检查已存在的模型，过滤重复
    existing_models = get_existing_models()
    missing_models, skipped_models = filter_missing_models(target_models, existing_models)

    print(f"目标模型数: {len(target_models)}")
    print(f"已存在跳过: {len(skipped_models)}")
    print(f"待抽取模型: {len(missing_models)}")
    print(f"起始索引: {args.start}")
    print(f"设备: {args.device}")
    print(f"动态形状: {args.dynamic}")
    print("-" * 50)

    if skipped_models and len(skipped_models) <= 10:
        print(f"跳过的已存在模型: {', '.join(skipped_models[:10])}")
        print("-" * 50)

    # 执行抽取
    success = []
    failed = []
    skipped = skipped_models

    for i, name in enumerate(missing_models):
        print(f"[{i+1}/{len(target_models)}] {name}...", end=" ", flush=True)
        ok, msg = extract_model(name, device=args.device, dynamic=args.dynamic)
        if ok:
            success.append(name)
            print(f"✓ {msg}")
        else:
            failed.append((name, msg))
            print(f"✗ {msg[:80]}")

    # 打印统计
    print("\n" + "=" * 50)
    print(f"完成: {len(success)}/{len(missing_models)}")
    print(f"成功: {len(success)}")
    print(f"失败: {len(failed)}")
    print(f"跳过: {len(skipped)} (已存在)")

    if failed:
        print("\n失败列表:")
        for name, msg in failed:
            print(f"  {name}: {msg[:100]}")

    # 保存结果
    result_dir = "/root/GraphNetExtractAgent"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(result_dir, f"timm_extract_result_{timestamp}.json")

    with open(result_file, "w") as f:
        json.dump({
            "total": len(target_models),
            "skipped": len(skipped),
            "extracted": len(missing_models),
            "success": success,
            "failed": [(n, m[:200]) for n, m in failed]
        }, f, indent=2)

    # 存档失败案例到 bad_cases 目录
    if failed:
        bad_cases_dir = os.path.join(result_dir, "bad_cases")
        os.makedirs(bad_cases_dir, exist_ok=True)
        bad_cases_file = os.path.join(bad_cases_dir, f"bad_cases_{timestamp}.json")
        with open(bad_cases_file, "w") as f:
            json.dump({
                "timestamp": timestamp,
                "total_failed": len(failed),
                "batch_file": args.list if args.list else None,
                "failed_models": [
                    {"model_name": n, "error": m} for n, m in failed
                ]
            }, f, indent=2)
        print(f"\n失败案例已存档: {bad_cases_file}")

    print(f"\n结果已保存到: {result_file}")


if __name__ == "__main__":
    main()

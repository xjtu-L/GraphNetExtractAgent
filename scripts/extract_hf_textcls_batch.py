#!/usr/bin/env python3
"""
HuggingFace text-classification 模型批量抽取脚本

用法:
    python extract_hf_textcls_batch.py --list models.txt --start 0 --count 100
    python extract_hf_textcls_batch.py --models "bert-base-uncased" "roberta-base"
    python extract_hf_textcls_batch.py --batch 0 --total-batches 10  # 分批次并行

环境变量:
    GRAPH_NET_EXTRACT_WORKSPACE  抽取输出目录 (默认: /work/graphnet_workspace/huggingface/text-classification)
    http_proxy / https_proxy     代理 (下载权重时需要)
    HF_ENDPOINT                  HuggingFace镜像 (默认: https://hf-mirror.com)
"""

import os
import sys
import json
import argparse
import torch
from datetime import datetime
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 强制设置代理（必须显式设置，不要依赖setdefault）
os.environ["http_proxy"] = "http://agent.baidu.com:8891"
os.environ["https_proxy"] = "http://agent.baidu.com:8891"

# 设置环境变量
os.environ["GRAPH_NET_EXTRACT_WORKSPACE"] = "/work/graphnet_workspace/huggingface/text-classification"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# 添加 GraphNet 路径
sys.path.insert(0, "/root/GraphNet")
from graph_net.torch.extractor import extract

# 默认输入配置 (text-classification 模型)
DEFAULT_MAX_LENGTH = 128
DEFAULT_INPUT_SHAPE = (1, 128)  # (batch_size, seq_length)


def get_existing_models():
    """获取已存在的模型列表"""
    existing = set()
    workspace = os.environ.get("GRAPH_NET_EXTRACT_WORKSPACE", "")
    if os.path.isdir(workspace):
        for model in os.listdir(workspace):
            model_path = os.path.join(workspace, model)
            if os.path.isdir(model_path):
                # 标准化名称（替换 / 为 _）
                existing.add(model.replace("_", "/"))
    return existing


def safe_model_name(model_id):
    """将模型ID转换为安全的目录名"""
    return model_id.replace("/", "_").replace("-", "_")


def generate_extract_script(output_dir, model_id, max_length=128):
    """生成 extract.py 脚本"""
    script_content = f'''#!/usr/bin/env python3
"""
计算图抓取复现脚本

模型: {model_id}
任务: text-classification
"""

import os
import sys
import json
import argparse
import torch

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

MODEL_ID = "{model_id}"
MAX_LENGTH = {max_length}


def extract_graph(output_dir=None, dynamic=False, max_length=None, pretrained=True):
    """执行计算图抓取"""
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))

    if max_length is None:
        max_length = MAX_LENGTH

    print(f"提取模型: {{MODEL_ID}}")
    print(f"最大长度: {{max_length}}")
    print(f"动态形状: {{dynamic}}")
    print(f"预训练权重: {{pretrained}}")

    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        # 加载模型和tokenizer
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

        # 构造输入 (使用虚拟文本)
        dummy_text = "This is a sample text for model extraction."
        inputs = tokenizer(
            dummy_text,
            return_tensors="pt",
            padding="max_length",
            max_length=max_length,
            truncation=True
        )

        # 使用 graph_net 提取
        from graph_net.torch.extractor import extract

        model_name_safe = MODEL_ID.replace("/", "_")
        if not pretrained:
            model_name_safe += "_no_pretrained"

        wrapped = extract(name=model_name_safe, dynamic=dynamic)(model).eval()

        with torch.no_grad():
            wrapped(**inputs)

        # 更新 graph_net.json
        json_path = os.path.join(output_dir, "graph_net.json")
        if os.path.exists(json_path):
            with open(json_path) as f:
                data = json.load(f)
        else:
            data = {{}}

        data["model_id"] = MODEL_ID
        data["source"] = "huggingface"
        data["task"] = "text-classification"
        data["max_length"] = max_length
        data["pretrained"] = pretrained

        with open(json_path, "w") as f:
            json.dump(data, f, indent=4)

        print(f"抓取完成! 输出目录: {{output_dir}}")
        return True

    except Exception as e:
        print(f"抓取失败: {{e}}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description=f"复现 {{MODEL_ID}} 的计算图抓取")
    parser.add_argument("--output", "-o", default=None, help="输出目录")
    parser.add_argument("--dynamic", action="store_true", help="使用动态形状")
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH, help="最大序列长度")
    parser.add_argument("--no-pretrained", action="store_true", help="不使用预训练权重")

    args = parser.parse_args()

    pretrained = not args.no_pretrained
    extract_graph(args.output, args.dynamic, args.max_length, pretrained)

if __name__ == "__main__":
    main()
'''
    script_path = os.path.join(output_dir, "extract.py")
    with open(script_path, "w") as f:
        f.write(script_content)
    os.chmod(script_path, 0o755)


def extract_model(model_id, device="cpu", dynamic=False, max_length=128):
    """抽取单个模型"""
    try:
        print(f"  Loading {model_id}...", end=" ", flush=True)

        # 加载模型
        model = AutoModelForSequenceClassification.from_pretrained(
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

        # 将输入移到设备
        inputs = {k: v.to(device) for k, v in inputs.items()}

        print(f"Tokenized", end=" ", flush=True)

        # 抽取
        model_name_safe = safe_model_name(model_id)
        wrapped = extract(name=model_name_safe, dynamic=dynamic)(model).eval()

        with torch.no_grad():
            wrapped(**inputs)

        print(f"Extracted", end=" ", flush=True)

        # 更新 graph_net.json
        output_dir = os.path.join(os.environ["GRAPH_NET_EXTRACT_WORKSPACE"], model_name_safe)
        json_path = os.path.join(output_dir, "graph_net.json")
        if os.path.exists(json_path):
            with open(json_path) as f:
                data = json.load(f)
            data["model_id"] = model_id
            data["source"] = "huggingface"
            data["task"] = "text-classification"
            data["max_length"] = max_length
            with open(json_path, "w") as f:
                json.dump(data, f, indent=4)

        # 生成 extract.py 脚本
        generate_extract_script(output_dir, model_id, max_length)
        print(f"Script OK")

        return True, f"OK (max_length={max_length})"

    except Exception as e:
        error_msg = str(e)
        # 截断长错误信息
        if len(error_msg) > 100:
            error_msg = error_msg[:100] + "..."
        return False, error_msg


def main():
    parser = argparse.ArgumentParser(description="HuggingFace text-classification 模型批量抽取")
    parser.add_argument("--models", nargs="+", help="指定模型列表")
    parser.add_argument("--list", help="从文件读取模型列表")
    parser.add_argument("--start", type=int, default=0, help="起始索引")
    parser.add_argument("--count", type=int, default=100, help="抽取数量")
    parser.add_argument("--batch", type=int, default=None, help="批次编号（用于并行）")
    parser.add_argument("--total-batches", type=int, default=10, help="总批次数")
    parser.add_argument("--device", default="cpu", help="设备")
    parser.add_argument("--dynamic", action="store_true", help="使用动态形状")
    parser.add_argument("--max-length", type=int, default=128, help="最大序列长度")
    parser.add_argument("--skip-existing", action="store_true", default=True, help="跳过已存在的模型")

    args = parser.parse_args()

    # 获取待抽取模型列表
    if args.models:
        target_models = args.models
    elif args.list:
        with open(args.list) as f:
            target_models = [line.strip() for line in f if line.strip()]
    else:
        print("错误: 需要指定 --models 或 --list")
        sys.exit(1)

    # 如果指定了批次模式，计算当前批次的模型范围
    if args.batch is not None:
        batch_size = len(target_models) // args.total_batches
        remainder = len(target_models) % args.total_batches

        # 计算当前批次的起始和结束
        start_idx = args.batch * batch_size + min(args.batch, remainder)
        if args.batch < remainder:
            batch_size += 1
        end_idx = start_idx + batch_size

        target_models = target_models[start_idx:end_idx]
        print(f"批次 {args.batch}/{args.total_batches}: 处理索引 {start_idx}-{end_idx-1}, 共 {len(target_models)} 个模型")
    else:
        # 使用 start 和 count
        target_models = target_models[args.start:args.start + args.count]
        print(f"处理索引 {args.start}-{args.start + len(target_models) - 1}, 共 {len(target_models)} 个模型")

    # 检查已存在的模型
    if args.skip_existing:
        existing_models = get_existing_models()
        original_count = len(target_models)
        target_models = [m for m in target_models if m not in existing_models]
        skipped_count = original_count - len(target_models)
        print(f"已存在跳过: {skipped_count} 个")

    print(f"待抽取模型: {len(target_models)} 个")
    print(f"设备: {args.device}")
    print(f"动态形状: {args.dynamic}")
    print(f"最大长度: {args.max_length}")
    print("-" * 60)

    # 执行抽取
    success = []
    failed = []

    for i, model_id in enumerate(target_models):
        print(f"[{i+1}/{len(target_models)}] {model_id}...", end=" ", flush=True)
        ok, msg = extract_model(model_id, device=args.device, dynamic=args.dynamic, max_length=args.max_length)
        if ok:
            success.append(model_id)
            print(f"✓ {msg}")
        else:
            failed.append((model_id, msg))
            print(f"✗ {msg}")

    # 打印统计
    print("\n" + "=" * 60)
    print(f"总计: {len(target_models)}")
    print(f"成功: {len(success)}")
    print(f"失败: {len(failed)}")

    if failed:
        print("\n失败列表:")
        for model_id, msg in failed:
            print(f"  {model_id}: {msg}")

    # 保存结果
    result_dir = "/root/GraphNetExtractAgent/results"
    os.makedirs(result_dir, exist_ok=True)

    batch_suffix = f"_batch{args.batch}" if args.batch is not None else ""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(result_dir, f"hf_textcls_extract{batch_suffix}_{timestamp}.json")

    with open(result_file, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "batch": args.batch,
            "total_batches": args.total_batches,
            "total": len(target_models),
            "success_count": len(success),
            "failed_count": len(failed),
            "success": success,
            "failed": [(m, e[:200]) for m, e in failed]
        }, f, indent=2)

    print(f"\n结果已保存到: {result_file}")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())

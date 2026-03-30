#!/usr/bin/env python3
"""
HuggingFace text-classification 模型强力重抓脚本 (V3策略)
- 只下载 config.json，使用随机初始化权重
- 必须抓到计算图，不留死角

用法:
    python extract_hf_textcls_force_v3.py --list failed_models.txt --start 0 --count 10
    python extract_hf_textcls_force_v3.py --models "bert-base-uncased"
"""

import os
import sys
import json
import argparse
import torch
import traceback
import subprocess
import tempfile
import shutil
import time
from datetime import datetime

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

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
except ImportError:
    print("错误: transformers 库未安装")
    sys.exit(1)


def try_specific_tokenizers(model_id, local_temp_dir=None):
    """尝试使用特定的tokenizer类加载"""
    try:
        from transformers import (
            BertTokenizer, RobertaTokenizer, GPT2Tokenizer,
            XLNetTokenizer, AlbertTokenizer, ElectraTokenizer,
            DistilBertTokenizer, XLMRobertaTokenizer, BartTokenizer,
            DebertaTokenizer, DebertaV2Tokenizer, XLMTokenizer,
            T5Tokenizer, MarianTokenizer, HerbertTokenizer,
            HerbertTokenizerFast, AutoTokenizer
        )
    except ImportError as e:
        print(f"(导入失败: {e})", end=" ", flush=True)
        return None

    tokenizer_classes = [
        (BertTokenizer, "BertTokenizer"),
        (RobertaTokenizer, "RobertaTokenizer"),
        (DebertaTokenizer, "DebertaTokenizer"),
        (DebertaV2Tokenizer, "DebertaV2Tokenizer"),
        (ElectraTokenizer, "ElectraTokenizer"),
        (DistilBertTokenizer, "DistilBertTokenizer"),
        (XLMRobertaTokenizer, "XLMRobertaTokenizer"),
        (BartTokenizer, "BartTokenizer"),
        (HerbertTokenizer, "HerbertTokenizer"),
        (GPT2Tokenizer, "GPT2Tokenizer"),
        (T5Tokenizer, "T5Tokenizer"),
        (XLNetTokenizer, "XLNetTokenizer"),
        (AlbertTokenizer, "AlbertTokenizer"),
        (XLMTokenizer, "XLMTokenizer"),
    ]

    cache_dir = local_temp_dir if local_temp_dir else None

    for tokenizer_cls, cls_name in tokenizer_classes:
        try:
            print(f"(尝试{cls_name})", end=" ", flush=True)
            if cache_dir:
                tokenizer = tokenizer_cls.from_pretrained(cache_dir)
            else:
                tokenizer = tokenizer_cls.from_pretrained(model_id)
            print(f"✓", end=" ", flush=True)
            return tokenizer
        except Exception:
            continue

    return None


def download_tokenizer_files(model_id, local_dir):
    """手动下载tokenizer文件到本地目录"""
    hf_endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
    base_url = f"{hf_endpoint}/{model_id}/resolve/main"

    # 需要下载的文件列表
    files_to_download = [
        "config.json",
        "tokenizer_config.json",
        "vocab.txt",
        "tokenizer.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "spiece.model",  # sentencepiece
    ]

    os.makedirs(local_dir, exist_ok=True)

    # 获取代理设置 - 硬编码确保可用
    proxy = os.environ.get("https_proxy", os.environ.get("http_proxy", "http://agent.baidu.com:8891"))

    downloaded = []
    for filename in files_to_download:
        url = f"{base_url}/{filename}"
        output_path = os.path.join(local_dir, filename)
        try:
            if proxy:
                cmd = ["curl", "-sL", "-f", "-x", proxy, url, "-o", output_path]
            else:
                cmd = ["curl", "-sL", "-f", url, "-o", output_path]
            result = subprocess.run(cmd, capture_output=True, timeout=60)
            if result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                downloaded.append(filename)
        except Exception:
            pass

    return downloaded


def safe_model_name(model_id):
    """将模型ID转换为安全的目录名"""
    return model_id.replace("/", "_").replace("-", "_")


def generate_extract_script(output_dir, model_id, max_length=128):
    """生成 extract.py 脚本"""
    script_content = f'''#!/usr/bin/env python3
"""
计算图抓取复现脚本 (V3-强制随机初始化)

模型: {model_id}
任务: text-classification
策略: 仅下载config，随机初始化权重
"""

import os
import sys
import json
import argparse
import torch

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

MODEL_ID = "{model_id}"
MAX_LENGTH = {max_length}


def extract_graph(output_dir=None, dynamic=False, max_length=None, use_random_init=True):
    """执行计算图抓取"""
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))

    if max_length is None:
        max_length = MAX_LENGTH

    print(f"[V3] 提取模型: {{MODEL_ID}}")
    print(f"[V3] 最大长度: {{max_length}}")
    print(f"[V3] 动态形状: {{dynamic}}")
    print(f"[V3] 随机初始化: {{use_random_init}}")

    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig

        # 加载tokenizer (必须)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

        # V3策略: 只下载config，随机初始化
        if use_random_init:
            print("[V3] 使用随机初始化权重...")
            config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
            model = AutoModelForSequenceClassification.from_config(config)
        else:
            print("[V3] 尝试加载预训练权重...")
            model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float32,
                trust_remote_code=True
            )

        model.eval()

        # 构造输入
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
        data["strategy"] = "v3_random_init"

        with open(json_path, "w") as f:
            json.dump(data, f, indent=4)

        print(f"[V3] 抓取完成! 输出目录: {{output_dir}}")
        return True

    except Exception as e:
        print(f"[V3] 抓取失败: {{e}}")
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description=f"[V3] 复现 {{MODEL_ID}} 的计算图抓取")
    parser.add_argument("--output", "-o", default=None, help="输出目录")
    parser.add_argument("--dynamic", action="store_true", help="使用动态形状")
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH, help="最大序列长度")
    parser.add_argument("--use-pretrained", action="store_true", help="使用预训练权重(默认随机初始化)")

    args = parser.parse_args()

    extract_graph(args.output, args.dynamic, args.max_length, use_random_init=not args.use_pretrained)


if __name__ == "__main__":
    main()
'''
    script_path = os.path.join(output_dir, "extract.py")
    with open(script_path, "w") as f:
        f.write(script_content)
    os.chmod(script_path, 0o755)


def force_extract_model(model_id, device="cpu", dynamic=False, max_length=128):
    """
    强力抓取模型 - V3策略
    必须抓到计算图，不留死角
    """
    # 修复模型ID: 将下划线替换回斜杠 (用于命名空间/模型名格式)
    # 注意: 有些模型名本身包含下划线，需要特殊处理
    # 格式: Namespace_ModelName -> Namespace/ModelName
    original_model_id = model_id
    if '_' in model_id and '/' not in model_id:
        # 尝试恢复斜杠格式 (第一个下划线替换为斜杠)
        parts = model_id.split('_', 1)
        if len(parts) == 2:
            model_id = f"{parts[0]}/{parts[1]}"
            print(f"[INFO] 修复模型ID: {original_model_id} -> {model_id}")

    model_name_safe = safe_model_name(model_id)
    output_dir = os.path.join(os.environ["GRAPH_NET_EXTRACT_WORKSPACE"], model_name_safe)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"[FORCE V3] 正在强力抓取: {model_id}")
    print(f"{'='*60}")

    # 步骤1: 加载tokenizer (尝试fast，失败则使用slow，再失败则手动下载，最后尝试特定tokenizer类)
    tokenizer = None
    local_temp_dir = None
    try:
        print("[1/4] 加载 Tokenizer...", end=" ", flush=True)
        try:
            # 首先尝试使用fast tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True,
                local_files_only=False,
                use_fast=True
            )
        except Exception as e:
            error_msg = str(e).lower()
            # 尝试使用slow tokenizer
            print(f"(use_fast=False)", end=" ", flush=True)
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    local_files_only=False,
                    use_fast=False
                )
            except Exception as e2:
                # 手动下载tokenizer文件
                print(f"(手动下载)", end=" ", flush=True)
                local_temp_dir = tempfile.mkdtemp()
                downloaded = download_tokenizer_files(model_id, local_temp_dir)
                print(f"(下载了{len(downloaded)}个文件)", end=" ", flush=True)
                if "tokenizer.json" in downloaded or "vocab.txt" in downloaded or "spiece.model" in downloaded:
                    try:
                        # 先尝试从下载目录加载
                        tokenizer = AutoTokenizer.from_pretrained(
                            local_temp_dir,
                            trust_remote_code=True,
                            use_fast=False
                        )
                    except Exception as e3:
                        # 尝试特定tokenizer类
                        tokenizer = try_specific_tokenizers(model_id, local_temp_dir)
                        if tokenizer is None:
                            raise RuntimeError(f"下载后加载失败，所有tokenizer类都失败: {e3}")
                else:
                    # 尝试特定tokenizer类从远程加载
                    tokenizer = try_specific_tokenizers(model_id)
                    if tokenizer is None:
                        raise RuntimeError(f"无法下载tokenizer文件，所有tokenizer类都失败: {e2}")
        print("✓")
    except Exception as e:
        print(f"✗ Tokenizer加载失败: {e}")
        if local_temp_dir and os.path.exists(local_temp_dir):
            shutil.rmtree(local_temp_dir)
        return False, f"Tokenizer failed: {e}"

    # 步骤2: V3策略 - 只下载config，随机初始化
    try:
        print("[2/4] 下载 Config 并随机初始化模型...", end=" ", flush=True)
        # 如果本地有config，优先使用
        if local_temp_dir and os.path.exists(os.path.join(local_temp_dir, "config.json")):
            config = AutoConfig.from_pretrained(local_temp_dir, trust_remote_code=True)
        else:
            config = AutoConfig.from_pretrained(
                model_id,
                trust_remote_code=True,
                local_files_only=False
            )
        model = AutoModelForSequenceClassification.from_config(config)
        model = model.to(device).eval()
        print(f"✓ (参数数: {sum(p.numel() for p in model.parameters())/1e6:.1f}M)")
    except Exception as e:
        print(f"✗ 模型初始化失败: {e}")
        if local_temp_dir and os.path.exists(local_temp_dir):
            shutil.rmtree(local_temp_dir)
        return False, f"Model init failed: {e}"

    # 步骤3: 构造输入
    try:
        print("[3/4] 构造输入数据...", end=" ", flush=True)
        dummy_text = "This is a sample text for model extraction."
        inputs = tokenizer(
            dummy_text,
            return_tensors="pt",
            padding="max_length",
            max_length=max_length,
            truncation=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        print(f"✓ (输入shape: {inputs['input_ids'].shape})")
    except Exception as e:
        print(f"✗ 输入构造失败: {e}")
        return False, f"Input failed: {e}"

    # 步骤4: 抽取计算图 - 这是关键！
    try:
        print("[4/4] 抽取计算图...", end=" ", flush=True)
        wrapped = extract(name=model_name_safe, dynamic=dynamic)(model).eval()

        with torch.no_grad():
            wrapped(**inputs)

        # 验证是否生成了graph_net.json
        graph_net_path = os.path.join(output_dir, "graph_net.json")
        if not os.path.exists(graph_net_path):
            # 检查子图
            found = False
            for i in range(10):  # 检查subgraph_0到subgraph_9
                subgraph_path = os.path.join(output_dir, f"subgraph_{i}", "graph_net.json")
                if os.path.exists(subgraph_path):
                    found = True
                    break
            if not found:
                raise RuntimeError("graph_net.json 未生成")

        print("✓ 成功!")
    except Exception as e:
        print(f"✗ 抽取失败: {e}")
        traceback.print_exc()
        if local_temp_dir and os.path.exists(local_temp_dir):
            shutil.rmtree(local_temp_dir)
        return False, f"Extract failed: {e}"

    # 更新 graph_net.json
    try:
        json_path = os.path.join(output_dir, "graph_net.json")
        if os.path.exists(json_path):
            with open(json_path) as f:
                data = json.load(f)
        else:
            data = {}

        data["model_id"] = model_id
        data["source"] = "huggingface"
        data["task"] = "text-classification"
        data["max_length"] = max_length
        data["strategy"] = "v3_random_init"
        data["force_extracted"] = True

        with open(json_path, "w") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"[!] 警告: 更新 graph_net.json 失败: {e}")

    # 生成 extract.py
    try:
        generate_extract_script(output_dir, model_id, max_length)
        print("[✓] extract.py 已生成")
    except Exception as e:
        print(f"[!] 警告: 生成 extract.py 失败: {e}")

    # 清理临时目录
    if local_temp_dir and os.path.exists(local_temp_dir):
        shutil.rmtree(local_temp_dir)

    return True, "Success with V3 random init"


def main():
    parser = argparse.ArgumentParser(description="[FORCE V3] HuggingFace 模型强力重抓")
    parser.add_argument("--models", nargs="+", help="指定模型列表")
    parser.add_argument("--list", help="从文件读取模型列表")
    parser.add_argument("--start", type=int, default=0, help="起始索引")
    parser.add_argument("--count", type=int, default=100, help="抽取数量")
    parser.add_argument("--device", default="cpu", help="设备")
    parser.add_argument("--dynamic", action="store_true", help="使用动态形状")
    parser.add_argument("--max-length", type=int, default=128, help="最大序列长度")
    parser.add_argument("--batch-id", type=int, default=None, help="批次ID(用于并行)")
    parser.add_argument("--total-batches", type=int, default=1, help="总批次数")

    args = parser.parse_args()

    # 获取模型列表
    if args.models:
        target_models = args.models
    elif args.list:
        with open(args.list) as f:
            target_models = [line.strip() for line in f if line.strip()]
    else:
        print("错误: 需要指定 --models 或 --list")
        sys.exit(1)

    # 批次处理
    if args.batch_id is not None:
        batch_size = len(target_models) // args.total_batches
        remainder = len(target_models) % args.total_batches
        start_idx = args.batch_id * batch_size + min(args.batch_id, remainder)
        if args.batch_id < remainder:
            batch_size += 1
        end_idx = start_idx + batch_size
        target_models = target_models[start_idx:end_idx]
        print(f"批次 {args.batch_id}/{args.total_batches}: 处理 {len(target_models)} 个模型")
    else:
        target_models = target_models[args.start:args.start + args.count]

    print(f"\n{'#'*60}")
    print(f"# [FORCE V3] 强力重抓开始")
    print(f"# 目标模型数: {len(target_models)}")
    print(f"# 策略: 仅下载config + 随机初始化")
    print(f"# 设备: {args.device}")
    print(f"{'#'*60}\n")

    success = []
    failed = []

    for i, model_id in enumerate(target_models):
        print(f"\n[{i+1}/{len(target_models)}] {model_id}")
        ok, msg = force_extract_model(
            model_id,
            device=args.device,
            dynamic=args.dynamic,
            max_length=args.max_length
        )
        if ok:
            success.append(model_id)
        else:
            failed.append((model_id, msg))
            print(f"[!] 失败: {msg}")

        # 添加延迟避免429错误
        if i < len(target_models) - 1:
            time.sleep(5)

    # 统计
    print(f"\n{'='*60}")
    print(f"[FORCE V3] 重抓完成")
    print(f"总计: {len(target_models)}")
    print(f"成功: {len(success)}")
    print(f"失败: {len(failed)}")
    print(f"{'='*60}")

    if failed:
        print("\n失败列表:")
        for model_id, msg in failed:
            print(f"  - {model_id}: {msg}")

    # 保存结果
    result_dir = "/root/GraphNetExtractAgent/results"
    os.makedirs(result_dir, exist_ok=True)

    batch_suffix = f"_batch{args.batch_id}" if args.batch_id is not None else ""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(result_dir, f"force_v3_reextract{batch_suffix}_{timestamp}.json")

    with open(result_file, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "strategy": "v3_random_init",
            "total": len(target_models),
            "success_count": len(success),
            "failed_count": len(failed),
            "success": success,
            "failed": [{"model": m, "error": e} for m, e in failed]
        }, f, indent=2)

    print(f"\n结果已保存: {result_file}")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())

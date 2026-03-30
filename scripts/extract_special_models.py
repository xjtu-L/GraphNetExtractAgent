#!/usr/bin/env python3
"""
特殊模型抓取脚本 - 处理难以抓取的模型
"""

import os
import sys
import json
import torch
import traceback
import tempfile
import shutil
import subprocess
import time

os.environ.setdefault("GRAPH_NET_EXTRACT_WORKSPACE", "/root/graphnet_workspace/huggingface/text-classification")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

sys.path.insert(0, "/root/GraphNet")
from graph_net.torch.extractor import extract

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
except ImportError:
    print("错误: transformers 库未安装")
    sys.exit(1)


def download_file(url, output_path, proxy="http://agent.baidu.com:8891"):
    """下载单个文件"""
    try:
        if proxy:
            cmd = ["curl", "-sL", "-f", "-x", proxy, url, "-o", output_path]
        else:
            cmd = ["curl", "-sL", "-f", url, "-o", output_path]
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        return result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0
    except Exception:
        return False


def download_model_files(model_id, local_dir):
    """下载模型所有必要文件"""
    hf_endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
    base_url = f"{hf_endpoint}/{model_id}/resolve/main"

    files_to_download = [
        "config.json",
        "tokenizer_config.json",
        "vocab.txt",
        "tokenizer.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "spiece.model",
        "sentencepiece.bpe.model",
        "merges.txt",
        "pytorch_model.bin",
        "model.safetensors",
    ]

    os.makedirs(local_dir, exist_ok=True)

    downloaded = []
    for filename in files_to_download:
        url = f"{base_url}/{filename}"
        output_path = os.path.join(local_dir, filename)
        if download_file(url, output_path):
            downloaded.append(filename)

    return downloaded


def safe_model_name(model_id):
    """将模型ID转换为安全的目录名"""
    return model_id.replace("/", "_").replace("-", "_")


def extract_bleurt_model(model_id, device="cpu", max_length=128):
    """特殊处理BLEURT模型"""
    print(f"\n[特殊处理] BLEURT模型: {model_id}")

    # 尝试安装bleurt支持
    try:
        import pip
        pip.main(['install', '-q', 'git+https://github.com/google-research/bleurt.git'])
    except Exception:
        pass

    local_dir = tempfile.mkdtemp()
    try:
        # 下载所有文件
        print("[1/4] 下载模型文件...", end=" ", flush=True)
        downloaded = download_model_files(model_id, local_dir)
        print(f"✓ (下载了{len(downloaded)}个文件)")

        # 读取并修改config
        config_path = os.path.join(local_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                config_data = json.load(f)
            # 将bleurt类型改为bert（架构兼容）
            if config_data.get("model_type") == "bleurt":
                print("[INFO] 将model_type从'bleurt'改为'bert'")
                config_data["model_type"] = "bert"
                config_data["architectures"] = ["BertForSequenceClassification"]
                with open(config_path, "w") as f:
                    json.dump(config_data, f, indent=2)

        # 使用BertTokenizer
        print("[2/4] 加载Tokenizer...", end=" ", flush=True)
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained(local_dir)
        print("✓")

        # 使用BertForSequenceClassification
        print("[3/4] 初始化模型...", end=" ", flush=True)
        from transformers import BertForSequenceClassification
        config = AutoConfig.from_pretrained(local_dir)
        model = BertForSequenceClassification.from_config(config)
        model = model.to(device).eval()
        print(f"✓ (参数数: {sum(p.numel() for p in model.parameters())/1e6:.1f}M)")

        # 构造输入并提取
        print("[4/4] 抽取计算图...", end=" ", flush=True)
        dummy_text = "This is a sample text for model extraction."
        inputs = tokenizer(
            dummy_text,
            return_tensors="pt",
            padding="max_length",
            max_length=max_length,
            truncation=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        model_name_safe = safe_model_name(model_id)
        output_dir = os.path.join(os.environ["GRAPH_NET_EXTRACT_WORKSPACE"], model_name_safe)
        os.makedirs(output_dir, exist_ok=True)

        wrapped = extract(name=model_name_safe, dynamic=False)(model).eval()
        with torch.no_grad():
            wrapped(**inputs)

        # 验证结果
        graph_net_path = os.path.join(output_dir, "graph_net.json")
        if not os.path.exists(graph_net_path):
            raise RuntimeError("graph_net.json 未生成")

        # 更新metadata
        with open(graph_net_path) as f:
            data = json.load(f)
        data["model_id"] = model_id
        data["source"] = "huggingface"
        data["task"] = "text-classification"
        data["max_length"] = max_length
        data["strategy"] = "special_bleurt_as_bert"
        with open(graph_net_path, "w") as f:
            json.dump(data, f, indent=4)

        print("✓ 成功!")
        return True, "Success with BLEURT special handling"

    except Exception as e:
        print(f"✗ 失败: {e}")
        traceback.print_exc()
        return False, str(e)
    finally:
        if os.path.exists(local_dir):
            shutil.rmtree(local_dir)


def extract_with_manual_config(model_id, device="cpu", max_length=128):
    """使用手动下载的配置文件提取"""
    print(f"\n[特殊处理] 手动配置: {model_id}")

    local_dir = tempfile.mkdtemp()
    try:
        # 下载文件
        print("[1/4] 下载模型文件...", end=" ", flush=True)
        downloaded = download_model_files(model_id, local_dir)
        print(f"✓ (下载了{len(downloaded)}个文件)")

        # 检查config
        config_path = os.path.join(local_dir, "config.json")
        if not os.path.exists(config_path):
            raise RuntimeError("config.json 不存在")

        with open(config_path) as f:
            config_data = json.load(f)

        # 如果没有model_type，尝试推断
        if "model_type" not in config_data:
            print("[INFO] 缺少model_type，尝试推断...")
            # 根据文件内容推断
            if "vocab.txt" in downloaded:
                config_data["model_type"] = "bert"
                config_data["architectures"] = ["BertForSequenceClassification"]
            elif "spiece.model" in downloaded:
                config_data["model_type"] = "xlm-roberta"
                config_data["architectures"] = ["XLMRobertaForSequenceClassification"]
            else:
                config_data["model_type"] = "bert"
                config_data["architectures"] = ["BertForSequenceClassification"]

            with open(config_path, "w") as f:
                json.dump(config_data, f, indent=2)
            print(f"[INFO] 设置model_type为: {config_data['model_type']}")

        # 加载tokenizer
        print("[2/4] 加载Tokenizer...", end=" ", flush=True)
        try:
            tokenizer = AutoTokenizer.from_pretrained(local_dir, use_fast=False)
        except Exception:
            from transformers import BertTokenizer
            tokenizer = BertTokenizer.from_pretrained(local_dir)
        print("✓")

        # 加载模型
        print("[3/4] 初始化模型...", end=" ", flush=True)
        config = AutoConfig.from_pretrained(local_dir)
        model = AutoModelForSequenceClassification.from_config(config)
        model = model.to(device).eval()
        print(f"✓ (参数数: {sum(p.numel() for p in model.parameters())/1e6:.1f}M)")

        # 构造输入并提取
        print("[4/4] 抽取计算图...", end=" ", flush=True)
        dummy_text = "This is a sample text for model extraction."
        inputs = tokenizer(
            dummy_text,
            return_tensors="pt",
            padding="max_length",
            max_length=max_length,
            truncation=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        model_name_safe = safe_model_name(model_id)
        output_dir = os.path.join(os.environ["GRAPH_NET_EXTRACT_WORKSPACE"], model_name_safe)
        os.makedirs(output_dir, exist_ok=True)

        wrapped = extract(name=model_name_safe, dynamic=False)(model).eval()
        with torch.no_grad():
            wrapped(**inputs)

        # 验证结果
        graph_net_path = os.path.join(output_dir, "graph_net.json")
        if not os.path.exists(graph_net_path):
            raise RuntimeError("graph_net.json 未生成")

        # 更新metadata
        with open(graph_net_path) as f:
            data = json.load(f)
        data["model_id"] = model_id
        data["source"] = "huggingface"
        data["task"] = "text-classification"
        data["max_length"] = max_length
        data["strategy"] = "special_manual_config"
        with open(graph_net_path, "w") as f:
            json.dump(data, f, indent=4)

        print("✓ 成功!")
        return True, "Success with manual config"

    except Exception as e:
        print(f"✗ 失败: {e}")
        traceback.print_exc()
        return False, str(e)
    finally:
        if os.path.exists(local_dir):
            shutil.rmtree(local_dir)


def main():
    # 特殊模型列表及处理策略
    special_models = [
        ("lucadiliello/BLEURT-20-D12", extract_bleurt_model),
        ("TakoData/en_tako_query_analyzer", extract_with_manual_config),
        ("synapti/nci-technique-classifier-v5.2", extract_with_manual_config),
        ("trl-internal-testing/tiny-Qwen3ForSequenceClassification", extract_with_manual_config),
        ("Voicelab/herbert-base-cased-sentiment", extract_with_manual_config),
    ]

    print("="*60)
    print("特殊模型抓取")
    print("="*60)

    success = []
    failed = []

    for i, (model_id, handler) in enumerate(special_models):
        print(f"\n[{i+1}/{len(special_models)}] {model_id}")
        ok, msg = handler(model_id)
        if ok:
            success.append(model_id)
        else:
            failed.append((model_id, msg))
        time.sleep(3)  # 避免请求过快

    print("\n" + "="*60)
    print("结果统计")
    print("="*60)
    print(f"总计: {len(special_models)}")
    print(f"成功: {len(success)}")
    print(f"失败: {len(failed)}")

    if failed:
        print("\n失败列表:")
        for model_id, msg in failed:
            print(f"  - {model_id}: {msg}")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())

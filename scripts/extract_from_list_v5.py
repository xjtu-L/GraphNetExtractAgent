#!/usr/bin/env python3
"""从模型列表下载并抽取"""

import os
import sys
import json
import torch
import hashlib
import argparse
from huggingface_hub import snapshot_download

os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda-11.8/lib64:/usr/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
os.environ["GRAPH_NET_EXTRACT_WORKSPACE"] = "/ssd1/liangtai-work/graphnet_workspace/huggingface/worker2"
os.environ["HF_HUB_OFFLINE"] = "0"  # 允许下载
os.environ["http_proxy"] = "http://agent.baidu.com:8891"
os.environ["https_proxy"] = "http://agent.baidu.com:8891"
os.environ["HF_ENDPOINT"] = "https://huggingface.co"

sys.path.insert(0, "/ssd1/liangtai-work/GraphNet")
from graph_net.torch.extractor import extract
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

# 跳过的类型 - 减少跳过，提高抽取率
SKIP_TYPES = {"bark", "musicgen", "sam", "encodec", "sd", "stable-diffusion", "speecht5", "parler_tts", "gguf"}
SKIP_ARCHS = {"DeepseekV3ForCausalLM", "MixtralForCausalLM", "Step1MoEForCausalLM"}
TYPE_MAP = {"qwen3_5_text": "qwen3", "qwen2_5_text": "qwen2", "llama3": "llama", "hf_olmo": "olmo"}

# 跳过的模型名称关键词
SKIP_NAME_PATTERNS = ["-gguf", "_gguf", "-GGUF", "_GGUF", "-mlx", "_mlx", "-awq", "_awq", "-gptq", "_gptq", "-onnx", "_onnx"]

def should_skip_by_name(model_id):
    """检查模型名称是否应该跳过"""
    model_lower = model_id.lower()
    for pattern in SKIP_NAME_PATTERNS:
        if pattern.lower() in model_lower:
            return True
    return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", required=True, help="模型列表文件")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--output", default="/ssd1/liangtai-work/graphnet_workspace/huggingface/worker2")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--agent-id", default="list_extract")
    args = parser.parse_args()

    # 读取列表
    with open(args.list) as f:
        models = [line.strip() for line in f if line.strip()]
    
    if args.end == -1:
        args.end = len(models)
    
    models = models[args.start:args.end]
    print(f"[{args.agent_id}] 处理 {len(models)} 个模型 ({args.start}-{args.start + len(models)})")

    ok, skip, fail = 0, 0, 0

    for i, model_id in enumerate(models, 1):
        # 检查模型名称是否应该跳过
        if should_skip_by_name(model_id):
            print(f"[{i}/{len(models)}] {model_id}... SKIP: 格式不支持")
            skip += 1
            continue

        # 转换为目录名
        name = model_id.replace("/", "_", 1)
        output_dir = os.path.join(args.output, name)

        # 检查是否已抽取
        if os.path.exists(f"{output_dir}/model.py"):
            print(f"[{i}/{len(models)}] {model_id}... 已存在")
            continue

        try:
            # 下载模型（只下载 config.json）
            config_path = f"{output_dir}/config.json"
            if not os.path.exists(config_path):
                os.makedirs(output_dir, exist_ok=True)
                try:
                    snapshot_download(
                        repo_id=model_id,
                        local_dir=output_dir,
                        allow_patterns=["config.json"],
                        max_workers=1,
                        resume_download=False
                    )
                except Exception as e:
                    print(f"[{i}/{len(models)}] {model_id}... SKIP: 下载失败 ({str(e)[:40]})")
                    skip += 1
                    continue

            # 读取配置
            with open(config_path) as f:
                config_dict = json.load(f)

            model_type = config_dict.get('model_type', '')
            architectures = config_dict.get('architectures', [])

            # 检查是否跳过
            if model_type.lower() in SKIP_TYPES:
                print(f"[{i}/{len(models)}] {model_id}... SKIP: 不支持 {model_type}")
                skip += 1
                continue

            if architectures:
                for skip_arch in SKIP_ARCHS:
                    if skip_arch in architectures[0]:
                        print(f"[{i}/{len(models)}] {model_id}... SKIP: 大模型")
                        skip += 1
                        break
                else:
                    pass
                if "SKIP: 大模型" in str(locals()):
                    continue

            # 检查配置一致性
            num_layers = config_dict.get('num_hidden_layers', 0)
            layer_types = config_dict.get('layer_types')
            if layer_types and len(layer_types) != num_layers:
                print(f"[{i}/{len(models)}] {model_id}... SKIP: 配置损坏")
                skip += 1
                continue

            # 映射 model_type
            if model_type in TYPE_MAP:
                config_dict['model_type'] = TYPE_MAP[model_type]
            config_dict.pop('auto_map', None)

            # 创建配置
            try:
                mt = config_dict.get('model_type', 'bert')
                config_dict_copy = {k: v for k, v in config_dict.items() if k != 'model_type'}
                config = AutoConfig.for_model(mt, **config_dict_copy)
            except Exception as e:
                print(f"[{i}/{len(models)}] {model_id}... SKIP: 配置不支持")
                skip += 1
                continue

            # 估算参数量
            h = getattr(config, 'hidden_size', 0) or 0
            layers = getattr(config, 'num_hidden_layers', 0) or 0
            vocab = getattr(config, 'vocab_size', 0) or 0
            inter = getattr(config, 'intermediate_size', 0) or (h * 4 if h else 0)
            n_experts = getattr(config, 'num_local_experts', 1) or 1
            
            if h and layers:
                params = ((h * h * 3 + h * h + h * inter * 2) * n_experts * layers) + vocab * h
                if params > 5_000_000_000:
                    print(f"[{i}/{len(models)}] {model_id}... SKIP: {params/1e9:.1f}B 参数 (>5B)")
                    skip += 1
                    continue

            # 创建模型 - 使用 trust_remote_code=True 支持自定义模型
            arch_name = str(getattr(config, 'architectures', [''])[0])

            # 根据架构选择正确的模型类
            if 'CausalLM' in arch_name:
                from transformers import AutoModelForCausalLM
                model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
            elif 'SequenceClassification' in arch_name:
                from transformers import AutoModelForSequenceClassification
                model = AutoModelForSequenceClassification.from_config(config, trust_remote_code=True)
            elif 'TokenClassification' in arch_name:
                from transformers import AutoModelForTokenClassification
                model = AutoModelForTokenClassification.from_config(config, trust_remote_code=True)
            elif 'MaskedLM' in arch_name:
                from transformers import AutoModelForMaskedLM
                model = AutoModelForMaskedLM.from_config(config, trust_remote_code=True)
            elif 'QuestionAnswering' in arch_name:
                from transformers import AutoModelForQuestionAnswering
                model = AutoModelForQuestionAnswering.from_config(config, trust_remote_code=True)
            else:
                model = AutoModel.from_config(config, trust_remote_code=True)

            model = model.to(args.device).eval()

            # 尝试加载 tokenizer 生成输入
            tokenizer = None
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, local_files_only=False)
            except:
                pass

            # 根据模型架构选择输入
            arch_lower = arch_name.lower()
            model_type_lower = model_type.lower()

            if 'whisper' in model_type_lower or 'whisper' in arch_lower:
                # Whisper: input_features (batch, num_mel_bins, time) + decoder_input_ids
                num_mel_bins = getattr(config, 'num_mel_bins', 80) or 80
                decoder_vocab = getattr(config, 'vocab_size', 51864) or 51864
                inputs = {
                    "input_features": torch.randn(1, num_mel_bins, 3000).to(args.device),
                    "decoder_input_ids": torch.randint(0, min(decoder_vocab, 1000), (1, 20)).to(args.device)
                }
            elif 'wav2vec' in model_type_lower or 'wav2vec' in arch_lower:
                # Wav2Vec2: input_values (batch, sequence_length)
                inputs = {"input_values": torch.randn(1, 16000).to(args.device)}
            elif tokenizer is not None:
                # 使用 tokenizer 生成输入
                dummy_text = "This is a sample text for model extraction."
                inputs = tokenizer(dummy_text, return_tensors="pt", padding="max_length", max_length=128, truncation=True)
                inputs = {k: v.to(args.device) for k, v in inputs.items()}
            elif any(x in arch_lower for x in ['vision', 'vit', 'clip', 'image']):
                inputs = {"pixel_values": torch.randn(1, 3, 224, 224).to(args.device)}
            else:
                # 手动构造输入
                vocab_size = getattr(config, 'vocab_size', 32000) or 32000
                input_ids = torch.randint(0, min(vocab_size, 1000), (1, 32)).to(args.device)
                inputs = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}

            # 抽取
            torch._dynamo.config.suppress_errors = True
            # 准备 config 信息用于 SymInt 推断
            config_for_extraction = {
                'num_hidden_layers': getattr(config, 'num_hidden_layers', 12),
                'num_attention_heads': getattr(config, 'num_attention_heads', 12),
                'hidden_size': getattr(config, 'hidden_size', 768),
                'intermediate_size': getattr(config, 'intermediate_size', 3072),
                'vocab_size': getattr(config, 'vocab_size', 32000),
                'max_position_embeddings': getattr(config, 'max_position_embeddings', 512),
            }
            wrapped = extract(name=name, dynamic=False, model_config=config_for_extraction)(model).eval()
            
            with torch.no_grad():
                try:
                    wrapped(**inputs)
                except TypeError:
                    wrapped(list(inputs.values())[0])

            # 生成 graph_hash.txt
            if os.path.exists(f"{output_dir}/model.py"):
                with open(f"{output_dir}/model.py", 'rb') as f:
                    hash_val = hashlib.sha256(f.read()).hexdigest()
                with open(f"{output_dir}/graph_hash.txt", 'w') as f:
                    f.write(hash_val)

            print(f"[{i}/{len(models)}] {model_id}... OK")
            ok += 1

            # 清理
            del model, wrapped
            torch.cuda.empty_cache()
            torch._dynamo.reset()

        except Exception as e:
            err = str(e)[:50]
            print(f"[{i}/{len(models)}] {model_id}... FAIL: {err}")
            fail += 1
            try:
                torch.cuda.empty_cache()
                torch._dynamo.reset()
            except:
                pass

    print(f"\n[{args.agent_id}] 完成: OK={ok}, SKIP={skip}, FAIL={fail}")

if __name__ == "__main__":
    main()

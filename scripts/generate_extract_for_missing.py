#!/usr/bin/env python3
"""
为缺少extract.py的模型生成extract.py脚本
祖训: 每个成功抓取的模型必须有extract.py
"""

import os
import sys
import json
from pathlib import Path

sys.path.insert(0, "/root/GraphNet")

def generate_extract_script(model_dir, model_id, task_type):
    """生成extract.py脚本"""

    # 查找模型目录
    safe_name = model_id.replace("/", "_")

    # 确定模型类
    task_to_class = {
        "text-classification": "AutoModelForSequenceClassification",
        "token-classification": "AutoModelForTokenClassification",
        "fill-mask": "AutoModelForMaskedLM",
        "feature-extraction": "AutoModel",
        "question-answering": "AutoModelForQuestionAnswering",
        "text-generation": "AutoModelForCausalLM",
        "translation": "AutoModel",
        "summarization": "AutoModel",
    }
    model_class = task_to_class.get(task_type, "AutoModel")

    script_content = f'''#!/usr/bin/env python3
"""
Extract script for {model_id}
Task: {task_type}
Generated: {__import__('datetime').datetime.now().isoformat()}
"""

import os
import sys
import torch
import argparse

# 设置代理
os.environ["http_proxy"] = "http://agent.baidu.com:8891"
os.environ["https_proxy"] = "http://agent.baidu.com:8891"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

sys.path.insert(0, "/root/GraphNet")
from graph_net.torch.extractor import extract

from transformers import {model_class}, AutoConfig

MODEL_ID = "{model_id}"
TASK_TYPE = "{task_type}"


def main():
    parser = argparse.ArgumentParser(description="Extract {model_id}")
    parser.add_argument("--no-pretrained", action="store_true", help="Use random initialization")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--max-length", type=int, default=128, help="Max sequence length")
    args = parser.parse_args()

    # 设置workspace
    if args.output:
        os.environ["GRAPH_NET_EXTRACT_WORKSPACE"] = args.output
    else:
        os.environ["GRAPH_NET_EXTRACT_WORKSPACE"] = "/work/graphnet_workspace/huggingface/{task_type}"

    # 加载模型
    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = {model_class}.from_config(config, trust_remote_code=True)
    model = model.cpu().eval()

    # 构造输入
    vocab_size = getattr(config, 'vocab_size', 32000)
    dummy_input_ids = torch.randint(0, min(vocab_size, 32000), (1, args.max_length))
    attention_mask = torch.ones((1, args.max_length))
    inputs = {{"input_ids": dummy_input_ids, "attention_mask": attention_mask}}

    # 抽取
    safe_name = MODEL_ID.replace("/", "_")
    wrapped = extract(name=safe_name, dynamic=False)(model).eval()

    with torch.no_grad():
        wrapped(**inputs)

    print(f"Extraction completed for {{MODEL_ID}}")


if __name__ == "__main__":
    main()
'''
    return script_content


def main():
    # 读取无extract.py的模型列表
    missing_list = "/tmp/worker2_no_extract.txt"

    if not os.path.exists(missing_list):
        print("No missing extract.py list found")
        return

    success = 0
    failed = 0

    with open(missing_list) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split("|")
            model_id = parts[0]
            batch = parts[1] if len(parts) > 1 else "unknown"

            safe_name = model_id.replace("/", "_")

            # 查找模型目录
            model_dir = None
            task_type = None

            for task in ["text-classification", "token-classification", "fill-mask",
                        "feature-extraction", "question-answering", "text-generation",
                        "translation", "summarization"]:
                task_dir = f"/work/graphnet_workspace/huggingface/{task}"
                potential_dir = os.path.join(task_dir, safe_name)
                if os.path.exists(potential_dir):
                    model_dir = potential_dir
                    task_type = task
                    break

            if not model_dir:
                print(f"❌ Directory not found: {model_id}")
                failed += 1
                continue

            # 生成extract.py
            try:
                script = generate_extract_script(model_dir, model_id, task_type)
                extract_path = os.path.join(model_dir, "extract.py")

                with open(extract_path, "w") as f:
                    f.write(script)

                os.chmod(extract_path, 0o755)
                print(f"✅ Generated extract.py for {model_id}")
                success += 1

            except Exception as e:
                print(f"❌ Failed to generate extract.py for {model_id}: {e}")
                failed += 1

    print(f"\n{'='*60}")
    print(f"Completed: Success {success}, Failed {failed}")


if __name__ == "__main__":
    main()

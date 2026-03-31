#!/usr/bin/env python3
"""
批量生成extract.py文件
为Worker2目录下缺少extract.py的模型生成标准extract.py文件
"""

import os
import sys
import json
from pathlib import Path

# 模板文件内容
EXTRACT_TEMPLATE = '''#!/usr/bin/env python3
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["http_proxy"] = "http://agent.baidu.com:8891"
os.environ["https_proxy"] = "http://agent.baidu.com:8891"

import torch
import sys
sys.path.insert(0, '/root/GraphNet')
from graph_net.torch.extractor import extract as torch_extract
from transformers import AutoConfig, AutoModelForCausalLM, AutoModel, AutoModelForSequenceClassification, AutoModelForMaskedLM, AutoModelForTokenClassification, AutoModelForQuestionAnswering, AutoModelForSeq2SeqLM

MODEL_ID = "{model_id}"
OUTPUT_DIR = "."

def main():
    try:
        config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)

        # 根据任务类型选择模型类
        task = "{task_type}"
        if task == "text-generation":
            try:
                model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
            except:
                model = AutoModel.from_config(config, trust_remote_code=True)
        elif task == "text-classification":
            model = AutoModelForSequenceClassification.from_config(config, trust_remote_code=True)
        elif task == "fill-mask":
            model = AutoModelForMaskedLM.from_config(config, trust_remote_code=True)
        elif task == "token-classification":
            model = AutoModelForTokenClassification.from_config(config, trust_remote_code=True)
        elif task == "question-answering":
            model = AutoModelForQuestionAnswering.from_config(config, trust_remote_code=True)
        elif task == "translation" or task == "summarization":
            model = AutoModelForSeq2SeqLM.from_config(config, trust_remote_code=True)
        else:
            model = AutoModel.from_config(config, trust_remote_code=True)

        model.eval()

        vocab_size = getattr(config, 'vocab_size', 32000)
        max_pos = getattr(config, 'max_position_embeddings', 512)
        seq_len = min(32, max_pos)

        os.environ["GRAPH_NET_EXTRACT_WORKSPACE"] = OUTPUT_DIR
        model = torch_extract(MODEL_ID.replace("/", "_"))(model)

        input_ids = torch.randint(0, min(vocab_size, 1000), (1, seq_len))
        with torch.no_grad():
            _ = model(input_ids)

        print(f"✓ Extraction completed for {{MODEL_ID}}")
        return True
    except Exception as e:
        print(f"✗ Error: {{e}}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()
'''


def extract_model_info(model_dir: str):
    """从目录路径提取模型ID和任务类型"""
    # 目录格式: /root/graphnet_workspace/huggingface/worker2/{task}/{model_id}/subgraph_*
    parts = model_dir.split('/')

    # 找到worker2后的索引
    try:
        worker2_idx = parts.index('worker2')
        if len(parts) > worker2_idx + 2:
            task_type = parts[worker2_idx + 1]
            model_id = parts[worker2_idx + 2]
            return model_id, task_type
    except ValueError:
        pass

    # 备用方案：从倒数第二个目录名推断
    if len(parts) >= 2:
        # 检查是否是subgraph目录
        if 'subgraph' in parts[-1]:
            model_id = parts[-2]
            # 尝试找到task类型
            for i, part in enumerate(parts):
                if part in ['fill-mask', 'text-classification', 'text-generation',
                           'token-classification', 'question-answering', 'translation',
                           'summarization', 'image-classification', 'automatic-speech-recognition']:
                    return model_id, part
            return model_id, "text-classification"  # 默认任务类型

    return None, None


def generate_extract_py(model_dir: str) -> bool:
    """为指定模型目录生成extract.py"""
    model_id, task_type = extract_model_info(model_dir)

    if not model_id:
        print(f"无法从路径解析模型ID: {model_dir}")
        return False

    # 处理subgraph目录的情况
    if 'subgraph' in model_dir:
        # 对于subgraph目录，使用父目录作为模型ID
        parent_dir = os.path.dirname(model_dir)
        model_id = os.path.basename(parent_dir)

    # 生成extract.py内容
    content = EXTRACT_TEMPLATE.format(
        model_id=model_id,
        task_type=task_type or "text-classification"
    )

    # 写入文件
    extract_path = os.path.join(model_dir, "extract.py")
    try:
        with open(extract_path, 'w') as f:
            f.write(content)
        os.chmod(extract_path, 0o755)  # 设置可执行权限
        print(f"✓ 生成成功: {extract_path}")
        return True
    except Exception as e:
        print(f"✗ 生成失败: {extract_path} - {e}")
        return False


def main():
    """主函数"""
    # 从文件读取需要处理的目录列表
    input_file = "/tmp/missing_extract_dirs.txt"
    output_log = "/tmp/extract_gen_progress.txt"

    if len(sys.argv) > 1:
        input_file = sys.argv[1]

    if not os.path.exists(input_file):
        print(f"错误: 找不到输入文件 {input_file}")
        sys.exit(1)

    # 读取目录列表
    with open(input_file, 'r') as f:
        model_dirs = [line.strip() for line in f if line.strip()]

    print(f"需要处理 {len(model_dirs)} 个模型目录")

    success_count = 0
    fail_count = 0

    for i, model_dir in enumerate(model_dirs, 1):
        print(f"[{i}/{len(model_dirs)}] 处理: {model_dir}")

        if generate_extract_py(model_dir):
            success_count += 1
        else:
            fail_count += 1

    # 记录进度
    with open(output_log, 'a') as f:
        f.write(f"批次完成: 成功 {success_count}, 失败 {fail_count}, 总计 {len(model_dirs)}\n")

    print(f"\n批次完成统计:")
    print(f"  成功: {success_count}")
    print(f"  失败: {fail_count}")
    print(f"  总计: {len(model_dirs)}")


if __name__ == "__main__":
    main()

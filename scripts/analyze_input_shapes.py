"""
分析 GraphNet samples 中所有模型的纯数据输入 shape。

从 weight_meta.py 中提取非权重参数的输入张量（即模型真正的数据输入），
按 shape 维度、来源分类统计，输出汇总报告。

用法:
    python analyze_input_shapes.py [--samples-dir /path/to/samples] [--output report.md]
"""

import argparse
import ast
import os
import re
import sys
from collections import defaultdict
from pathlib import Path


def parse_weight_meta(filepath: str) -> list[dict]:
    """
    解析 weight_meta.py，提取所有 tensor meta 类的信息。

    返回:
        [{"name": str, "shape": list, "dtype": str, "device": str,
          "mean": float|None, "std": float|None, "data": any}, ...]
    """
    results = []
    try:
        with open(filepath, "r") as f:
            content = f.read()
    except Exception:
        return results

    # 按 class 块拆分
    class_pattern = re.compile(
        r"class\s+Program_weight_tensor_meta_(\S+?)\s*:",
        re.MULTILINE,
    )
    matches = list(class_pattern.finditer(content))

    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        block = content[start:end]

        meta = {"class_suffix": match.group(1)}

        # 提取 name
        m = re.search(r'name\s*=\s*"([^"]*)"', block)
        meta["name"] = m.group(1) if m else match.group(1)

        # 提取 shape
        m = re.search(r"shape\s*=\s*(\[.*?\])", block, re.DOTALL)
        if m:
            try:
                meta["shape"] = ast.literal_eval(m.group(1))
            except Exception:
                meta["shape"] = []
        else:
            meta["shape"] = []

        # 提取 dtype
        m = re.search(r'dtype\s*=\s*"([^"]*)"', block)
        meta["dtype"] = m.group(1) if m else "unknown"

        # 提取 device
        m = re.search(r'device\s*=\s*"([^"]*)"', block)
        meta["device"] = m.group(1) if m else "unknown"

        # 提取 data（判断是否有具体数据）
        m = re.search(r"data\s*=\s*(\S+)", block)
        if m:
            raw = m.group(1).strip().rstrip(",")
            meta["has_data"] = raw != "None"
        else:
            meta["has_data"] = False

        results.append(meta)

    return results


def is_pure_input(meta: dict) -> bool:
    """
    判断一个 tensor 是否为模型的纯数据输入（非权重参数）。

    排除规则：
    - 名称含 "parameters_weight_" 或 "parameters_bias_" → 模型权重
    - 名称含 "buffers_running_mean_" 或 "buffers_running_var_" → BN 统计量
    - 名称含 "momentum" 或 "eps" → BN 超参数
    - 名称含 "num_batches_tracked" → BN 计数器
    - shape 为 [] 且 dtype 为 int → 标量常量（如 s53）
    """
    name = meta["name"]

    # 排除模型权重和参数
    weight_keywords = [
        "parameters_weight_",
        "parameters_bias_",
        "buffers_running_mean_",
        "buffers_running_var_",
        "momentum",
        "_eps",
        "num_batches_tracked",
        "embed_tokens",
        "embed_positions",
        "layernorm",
        "layer_norm",
        "ln_",
        "wte",
        "wpe",
        "lm_head",
        "cls",
        "pooler",
        "classifier",
        "head",
    ]

    name_lower = name.lower()
    # 如果名称包含 modules_ 且包含权重相关关键字，排除
    if "modules_" in name_lower:
        for kw in weight_keywords:
            if kw.lower() in name_lower:
                return False

    # 排除标量常量（shape=[], int类型，通常是符号常量如 s53）
    if meta["shape"] == [] and "int" in meta["dtype"]:
        return False

    # 排除以 s + 数字 命名的符号常量
    if re.match(r"^s\d+$", name):
        return False

    # 包含 L_x_ 或 L_kwargs_ 等明确输入标记的，一定是输入
    input_markers = ["L_x_", "L_kwargs_", "L_input", "L_args_"]
    for marker in input_markers:
        if name.startswith(marker):
            return True

    # 如果名称含 "modules_" 前缀且不在上面的 input_markers 中
    # 大概率是模型内部参数，排除
    if name.startswith("L_self_modules_"):
        return False

    # 其余情况：如果 shape 非空且维度 >= 2，可能是输入
    if len(meta["shape"]) >= 2:
        return True

    return False


def analyze_samples(samples_dir: str) -> dict:
    """
    遍历所有样本目录，提取并分类输入 shape。

    返回:
        {
            "by_source": {source: [{model, inputs: [{name, shape, dtype}]}]},
            "shape_stats": {ndim: count},
            "all_inputs": [{source, model, name, shape, dtype}],
        }
    """
    results = {
        "by_source": defaultdict(list),
        "shape_stats": defaultdict(int),
        "all_inputs": [],
    }

    samples_path = Path(samples_dir)
    if not samples_path.exists():
        print(f"ERROR: samples directory not found: {samples_dir}", file=sys.stderr)
        return results

    for source_dir in sorted(samples_path.iterdir()):
        if not source_dir.is_dir():
            continue
        source = source_dir.name

        for model_dir in sorted(source_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            if model_dir.name == "_decomposed":
                continue

            model_name = model_dir.name
            wm_path = model_dir / "weight_meta.py"
            if not wm_path.exists():
                continue

            metas = parse_weight_meta(str(wm_path))
            inputs = []
            for m in metas:
                if is_pure_input(m):
                    inputs.append({
                        "name": m["name"],
                        "shape": m["shape"],
                        "dtype": m["dtype"],
                    })
                    ndim = len(m["shape"])
                    results["shape_stats"][ndim] += 1
                    results["all_inputs"].append({
                        "source": source,
                        "model": model_name,
                        "name": m["name"],
                        "shape": m["shape"],
                        "dtype": m["dtype"],
                    })

            results["by_source"][source].append({
                "model": model_name,
                "inputs": inputs,
            })

    return results


def generate_report(results: dict) -> str:
    """生成 Markdown 分析报告。"""
    lines = []
    lines.append("# GraphNet 样本输入 Shape 分析报告\n")

    # === 总览 ===
    total_models = sum(len(models) for models in results["by_source"].values())
    total_inputs = len(results["all_inputs"])
    models_with_inputs = sum(
        1 for models in results["by_source"].values()
        for m in models if m["inputs"]
    )
    models_without_inputs = total_models - models_with_inputs

    lines.append("## 1. 总览\n")
    lines.append(f"| 指标 | 数值 |")
    lines.append(f"|------|------|")
    lines.append(f"| 总样本数 | {total_models} |")
    lines.append(f"| 检测到纯数据输入的样本 | {models_with_inputs} |")
    lines.append(f"| 未检测到输入的样本 | {models_without_inputs} |")
    lines.append(f"| 总输入张量数 | {total_inputs} |")
    lines.append("")

    # === 按维度统计 ===
    lines.append("## 2. 输入张量维度分布\n")
    lines.append("| 维度 (ndim) | 数量 | 典型用途 |")
    lines.append("|-------------|------|----------|")
    ndim_desc = {
        0: "标量",
        1: "1D 序列 / 掩码",
        2: "2D (batch, seq_len) — NLP 文本输入",
        3: "3D (batch, channels, len) — 1D 卷积/音频",
        4: "4D (batch, C, H, W) — 图像",
        5: "5D (batch, C, T, H, W) — 视频",
    }
    for ndim in sorted(results["shape_stats"].keys()):
        desc = ndim_desc.get(ndim, "其他")
        lines.append(f"| {ndim} | {results['shape_stats'][ndim]} | {desc} |")
    lines.append("")

    # === 按来源分类 ===
    lines.append("## 3. 各来源输入 Shape 汇总\n")

    for source in sorted(results["by_source"].keys()):
        models = results["by_source"][source]
        models_with = [m for m in models if m["inputs"]]

        lines.append(f"### {source} ({len(models_with)}/{len(models)} 个模型有输入)\n")

        if not models_with:
            lines.append("> 未检测到纯数据输入\n")
            continue

        # 统计该来源下的 shape 模式
        shape_patterns = defaultdict(list)
        for m in models_with:
            for inp in m["inputs"]:
                ndim = len(inp["shape"])
                pattern = f"{ndim}D: {inp['shape']}"
                shape_patterns[pattern].append(m["model"])

        lines.append("| Shape 模式 | 出现次数 | 示例模型 |")
        lines.append("|-----------|----------|----------|")
        for pattern, model_list in sorted(
            shape_patterns.items(), key=lambda x: -len(x[1])
        ):
            examples = ", ".join(model_list[:3])
            if len(model_list) > 3:
                examples += f" ...等{len(model_list)}个"
            lines.append(f"| `{pattern}` | {len(model_list)} | {examples} |")
        lines.append("")

    # === 4D 输入（图像）详细分析 ===
    lines.append("## 4. 4D 图像输入详细分析\n")
    image_inputs = [
        inp for inp in results["all_inputs"] if len(inp["shape"]) == 4
    ]
    if image_inputs:
        # 按 (C, H, W) 分组
        chw_groups = defaultdict(list)
        for inp in image_inputs:
            _, c, h, w = inp["shape"]
            chw_groups[(c, h, w)].append(inp)

        lines.append("| (C, H, W) | 数量 | 示例来源 |")
        lines.append("|-----------|------|----------|")
        for chw, items in sorted(chw_groups.items(), key=lambda x: -len(x[1])):
            sources = set(it["source"] for it in items)
            lines.append(f"| {chw} | {len(items)} | {', '.join(sorted(sources))} |")
        lines.append("")

    # === 2D 输入（NLP）详细分析 ===
    lines.append("## 5. 2D NLP 输入详细分析\n")
    nlp_inputs = [
        inp for inp in results["all_inputs"] if len(inp["shape"]) == 2
    ]
    if nlp_inputs:
        seq_len_groups = defaultdict(list)
        for inp in nlp_inputs:
            _, seq_len = inp["shape"]
            seq_len_groups[seq_len].append(inp)

        lines.append("| Sequence Length | 数量 | 示例来源 |")
        lines.append("|----------------|------|----------|")
        for seq_len, items in sorted(seq_len_groups.items(), key=lambda x: -len(x[1])):
            sources = set(it["source"] for it in items)
            lines.append(f"| {seq_len} | {len(items)} | {', '.join(sorted(sources))} |")
        lines.append("")

    # === 特殊输入（5D 视频、多输入等）===
    lines.append("## 6. 特殊输入模式\n")

    # 5D 视频
    video_inputs = [
        inp for inp in results["all_inputs"] if len(inp["shape"]) == 5
    ]
    if video_inputs:
        lines.append("### 5D 视频输入\n")
        lines.append("| 模型 | 来源 | Shape | dtype |")
        lines.append("|------|------|-------|-------|")
        for inp in video_inputs:
            lines.append(
                f"| {inp['model']} | {inp['source']} | `{inp['shape']}` | {inp['dtype']} |"
            )
        lines.append("")

    # 多输入模型
    lines.append("### 多输入模型（>1 个纯数据输入）\n")
    lines.append("| 模型 | 来源 | 输入数 | 各输入 Shape |")
    lines.append("|------|------|--------|-------------|")
    for source, models in sorted(results["by_source"].items()):
        for m in models:
            if len(m["inputs"]) > 1:
                shapes_str = ", ".join(f'`{i["shape"]}`' for i in m["inputs"])
                lines.append(
                    f"| {m['model']} | {source} | {len(m['inputs'])} | {shapes_str} |"
                )
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="分析 GraphNet 样本的输入 Shape")
    parser.add_argument(
        "--samples-dir",
        default="/root/GraphNet/samples",
        help="样本目录路径 (default: /root/GraphNet/samples)",
    )
    parser.add_argument(
        "--output",
        default="/root/GraphNetExtractAgent/INPUT_SHAPE_ANALYSIS.md",
        help="输出报告路径 (default: INPUT_SHAPE_ANALYSIS.md)",
    )
    parser.add_argument(
        "--also-check",
        default="/ssd1/liangtai-work/graphnet_workspace",
        help="额外检查的 workspace 目录",
    )
    args = parser.parse_args()

    print(f"扫描样本目录: {args.samples_dir}")
    results = analyze_samples(args.samples_dir)

    # 也检查 workspace 中的样本
    if args.also_check and os.path.isdir(args.also_check):
        print(f"扫描 workspace 目录: {args.also_check}")
        ws_results = analyze_samples(args.also_check)
        # workspace 中的样本归入 "workspace" 来源
        # 由于 workspace 结构不同（没有 source 子目录），单独处理
        ws_path = Path(args.also_check)
        for model_dir in sorted(ws_path.iterdir()):
            if not model_dir.is_dir():
                continue
            wm_path = model_dir / "weight_meta.py"
            if not wm_path.exists():
                continue
            metas = parse_weight_meta(str(wm_path))
            inputs = []
            for m in metas:
                if is_pure_input(m):
                    inputs.append({
                        "name": m["name"],
                        "shape": m["shape"],
                        "dtype": m["dtype"],
                    })
                    ndim = len(m["shape"])
                    results["shape_stats"][ndim] += 1
                    results["all_inputs"].append({
                        "source": "workspace",
                        "model": model_dir.name,
                        "name": m["name"],
                        "shape": m["shape"],
                        "dtype": m["dtype"],
                    })
            results["by_source"]["workspace"].append({
                "model": model_dir.name,
                "inputs": inputs,
            })

    report = generate_report(results)

    with open(args.output, "w") as f:
        f.write(report)

    print(f"\n报告已生成: {args.output}")
    print(f"总样本: {sum(len(m) for m in results['by_source'].values())}")
    print(f"总输入张量: {len(results['all_inputs'])}")
    print(f"维度分布: {dict(sorted(results['shape_stats'].items()))}")


if __name__ == "__main__":
    main()

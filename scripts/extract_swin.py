#!/usr/bin/env python3
"""
T2: torchvision Vision Transformer 补全 - swin_s, swin_t 抽取脚本

用法:
    export GRAPH_NET_EXTRACT_WORKSPACE=/ssd1/liangtai-work/graphnet_workspace
    export http_proxy=http://agent.baidu.com:8891
    export https_proxy=http://agent.baidu.com:8891
    python extract_swin.py

注意:
    - Swin Transformer 包含 padding 操作，使用 dynamic=False 避免 unbacked symbol 问题
    - 已有的 swin_b/swin_v2_b 样本同样使用 dynamic=False
    - 输入: torch.rand(1, 3, 224, 224)

验证:
    python -m graph_net.torch.validate --model-path $GRAPH_NET_EXTRACT_WORKSPACE/swin_s --no-check-redundancy
    python -m graph_net.torch.validate --model-path $GRAPH_NET_EXTRACT_WORKSPACE/swin_t --no-check-redundancy
"""

import torch
from torchvision.models import get_model, get_model_weights
from graph_net.torch.extractor import extract

device = torch.device("cpu")

for name in ["swin_s", "swin_t"]:
    print(f"\n=== Extracting {name} (dynamic=False) ===")
    weights = get_model_weights(name).DEFAULT
    model = get_model(name, weights=weights).to(device).eval()
    # dynamic=False: Swin 的 padding 操作在 dynamic=True 时会产生 unbacked symbol
    wrapped = extract(name=name, dynamic=False)(model).eval()
    with torch.no_grad():
        output = wrapped(torch.rand(1, 3, 224, 224, device=device))
    print(f"[OK] {name} extracted, output shape: {output.shape}")

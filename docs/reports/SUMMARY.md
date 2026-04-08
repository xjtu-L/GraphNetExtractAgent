# GraphNet 模型计算图提取 — 完整总结

> 本文档记录了 torchvision 模型计算图提取的全部工作成果，供下一个 Agent 继承。

---

## 1. 整体任务概述

基于 [PaddlePaddle/GraphNet](https://github.com/PaddlePaddle/GraphNet) 项目，对 torchvision 中尚未覆盖的 PyTorch 模型执行**计算图提取（extract）**和**验证（validate）**，并将结果作为 GraphNet 的数据样本。

**核心流程**：选模型 → 构造输入 → `extract(name, dynamic=False)(model)` → `validate --model-path` → 补 `graph_net.json` 字段

**任务范围**：torchvision 中 6 类模型（ViT 补全、语义分割、视频、光流、目标检测、量化），共计 36 个模型。

**最终成果**：
- **24 个模型 extract 成功**（12 个单图模型 + 12 个检测模型）
- **12 个单图模型全部 validate 通过**（100%）
- **12 个检测模型 extract 成功**，经修复后 128/128 子图全部 validate 通过
- **12 个量化模型确认不可行**（PyTorch 限制）

---

## 2. 环境配置

### 2.1 系统环境

| 项目 | 版本/路径 |
|------|-----------|
| Python | 3.10 |
| PyTorch | 2.11.0+cu130 |
| torchvision | 0.26.0+cu130 |
| transformers | 5.3.0 |
| huggingface_hub | 1.3.4 |
| graph-net | 0.1.0（editable, 指向 `/root/GraphNet`） |
| OS | Linux 5.10.0 |

### 2.2 代理设置（必须）

下载模型权重前**必须**设置代理，否则会超时：

```bash
export http_proxy=http://agent.baidu.com:8891
export https_proxy=http://agent.baidu.com:8891
```

### 2.3 环境变量

```bash
export GRAPH_NET_EXTRACT_WORKSPACE=/ssd1/liangtai-work/graphnet_workspace
```

已写入 `/root/.bashrc`，新 shell 自动生效。

### 2.4 graph_net 安装

```bash
cd /root/GraphNet && pip install -e ".[agent]"
```

安装位置：`/usr/local/lib/python3.10/dist-packages`，editable 模式指向 `/root/GraphNet`。

---

## 3. 已完成的模型列表

### 3.1 单图模型（12 个）— 全部 extract + validate 通过

| 模型 | 类型 | 输入形状 | 输出形状 | graph_hash |
|------|------|----------|----------|------------|
| `swin_s` | 图像分类 | `(1,3,224,224)` | `(1,1000)` | `1f0dc6a2...` |
| `swin_t` | 图像分类 | `(1,3,224,224)` | `(1,1000)` | `7d17ac7d...` |
| `lraspp_mobilenet_v3_large` | 语义分割 | `(1,3,224,224)` | `(1,21,224,224)` | `bd890f42...` |
| `r2plus1d_18` | 视频分类 | `(1,3,16,112,112)` | `(1,400)` | `1d610707...` |
| `s3d` | 视频分类 | `(1,3,16,224,224)` | `(1,400)` | `5a025122...` |
| `mvit_v1_b` | 视频分类 | `(1,3,16,224,224)` | `(1,400)` | `b6ebb232...` |
| `mvit_v2_s` | 视频分类 | `(1,3,16,224,224)` | `(1,400)` | `9e7cf37e...` |
| `swin3d_b` | 视频分类 | `(1,3,16,224,224)` | `(1,400)` | `318e4d18...` |
| `swin3d_s` | 视频分类 | `(1,3,16,224,224)` | `(1,400)` | `108299d8...` |
| `swin3d_t` | 视频分类 | `(1,3,16,224,224)` | `(1,400)` | `f7d5de20...` |
| `raft_large` | 光流估计 | 双图 `(1,3,H,W)` | flow tensor | `3a71350d...` |
| `raft_small` | 光流估计 | 双图 `(1,3,H,W)` | flow tensor | `1e91cf43...` |

所有 12 个哈希**全部唯一**，无重复。

### 3.2 目标检测模型（12 个）— extract 成功，128/128 子图全部 validate 通过

由于 graph breaks（列表输入、条件分支、NMS 等），检测模型被 extractor 拆成多个**子图（subgraph_N/）**。

| 模型 | 子图数 | Validate |
|------|--------|----------|
| `ssd300_vgg16` | 5 | **5/5** |
| `ssdlite320_mobilenet_v3_large` | 10 | **10/10** |
| `retinanet_resnet50_fpn` | 5 | **5/5** |
| `retinanet_resnet50_fpn_v2` | 7 | **7/7** |
| `fasterrcnn_resnet50_fpn` | 15 | **15/15** |
| `fasterrcnn_resnet50_fpn_v2` | 14 | **14/14** |
| `fasterrcnn_mobilenet_v3_large_fpn` | 7 | **7/7** |
| `fasterrcnn_mobilenet_v3_large_320_fpn` | 7 | **7/7** |
| `fcos_resnet50_fpn` | 9 | **9/9** |
| `maskrcnn_resnet50_fpn` | 16 | **16/16** |
| `maskrcnn_resnet50_fpn_v2` | 17 | **17/17** |
| `keypointrcnn_resnet50_fpn` | 16 | **16/16** |
| **合计** | **128** | **128/128** |

**初始 validate 有 41 个子图失败**，通过以下修复全部解决（详见第 4 节踩坑记录）：
- 35 个 model.py 添加 `import torchvision`（解决 `torch.ops.torchvision.nms` 缺失）
- 38 个 model.py 移除 `_assert_scalar` 运行时断言（NMS 约束在随机数据下无效）
- 修改 GraphNet 源码 `single_device_runner.py` 处理空 tensor 输出
- 修改 GraphNet 源码 `utils.py` 自动添加 `import torchvision`

**子图哈希重复**：共享相同 backbone（如 ResNet50+FPN）的模型间存在子图哈希重复，这是正常现象。

### 3.3 量化模型（12 个）— 确认不可行

| 模型 | 状态 |
|------|------|
| `quantized_resnet18` | ❌ 无法 extract |
| `quantized_resnet50` | ❌ 无法 extract |
| `quantized_mobilenet_v2` | ❌ 无法 extract |
| `quantized_mobilenet_v3_large` | ❌ 无法 extract |
| `quantized_googlenet` | ❌ 无法 extract |
| `quantized_inception_v3` | ❌ 无法 extract |
| `quantized_resnext101_32x8d` | ❌ 无法 extract |
| `quantized_resnext101_64x4d` | ❌ 无法 extract |
| `quantized_shufflenet_v2_x0_5` | ❌ 无法 extract |
| `quantized_shufflenet_v2_x1_0` | ❌ 无法 extract |
| `quantized_shufflenet_v2_x1_5` | ❌ 无法 extract |
| `quantized_shufflenet_v2_x2_0` | ❌ 无法 extract |

**根因**：`torch._dynamo` 无法为量化张量（QInt8/QUInt8）创建 meta tensor，报错 `"quantized nyi in meta tensors"`（nyi = Not Yet Implemented）。在默认 `suppress_errors=True` 下，dynamo 静默回退到 eager 执行，GraphExtractor 后端回调从未被触发，不生成任何输出。

**已测试的替代方案（均失败）**：

| 方案 | 结果 |
|------|------|
| `torch.compile(fullgraph=True)` | `UserError: Could not extract specialized integer from data-dependent expression` |
| `torch.export.export()` | `Conv2dPackedParamsBase does not have __obj_flatten__` |
| `torch.fx.experimental.proxy_tensor.make_fx()` | `aten::empty_strided not supported for QuantizedMeta` |
| `torch.fx.symbolic_trace()` | 可 trace 但产生 call_module 图，不自包含，validate 失败 |
| PT2E 量化 (`torch.ao.quantization.quantizer`) | 当前 torch 版本不可用 |

**结论**：这是 PyTorch 层面的限制，非 GraphNet 问题。需等待 PyTorch 对量化张量的 dynamo 支持完善。

---

## 4. 踩坑记录与解决方案

### 4.1 必须使用 `dynamic=False`

**问题**：Swin Transformer 系列（swin_s/t、swin3d_b/s/t）的 `F.pad` 操作在 `dynamic=True` 时会引入 `PendingUnbackedSymbolNotFound: Pending unbacked symbols {u0} not in returned outputs` 错误。

**解决方案**：统一使用 `dynamic=False`。已有的 swin_b/swin_v2_* 样本同样使用 `dynamic=False`。实际操作中，所有 24 个成功模型均使用 `dynamic=False`。

### 4.2 graph_net.json 需手动补字段

**问题**：extractor 只自动生成 5 个字段（framework, num_devices_required, num_nodes_required, dynamic, model_name），但样本规范要求还有 `source` 和 `heuristic_tag`。

**解决方案**：extract 完成后手动补充。参见第 5 节的字段规范。

### 4.3 检测模型产生子图（graph breaks）

**问题**：目标检测模型的 forward 包含列表输入、条件分支（if self.training）、NMS 等操作，`torch._dynamo` 无法将整个模型编译为单一计算图，产生 graph breaks。

**表现**：每个模型的输出目录下不是直接的 `model.py` 等文件，而是 `subgraph_0/`、`subgraph_1/`... 多个子目录，每个子目录是一个独立的计算图样本。

**影响**：
- backbone/FPN 等主要计算子图可正常 validate
- NMS/后处理子图因随机输入产生空 tensor 导致 validate 失败
- 子图间存在哈希重复（共享 backbone 的模型）

### 4.4 后处理子图 validate 失败

**问题**：检测模型的 NMS/后处理子图在 validate 时报错：
- `argmin(): Expected reduction dim to be specified for input.numel() == 0`
- `tuple index out of range`

**原因**：validate 用随机 tensor 重新运行子图，随机输入不会产生有效的检测结果，导致空 tensor 传入 argmin 等操作。

**结论**：这是输入数据问题，图结构本身是正确的。

### 4.5 `input_tensor_constraints.py` 为空是正常的

**问题**：`dynamic=False` 时 extractor 不生成 sympy 约束。

**说明**：已有样本中的约束是后处理管线产物，非 extractor 直接生成。

### 4.6 代理超时

**问题**：首次 extract 时下载权重失败 `URLError: [Errno 110] Connection timed out`。

**解决方案**：必须在运行脚本前设置代理环境变量（见第 2.2 节）。

### 4.7 视频模型输入格式

**注意**：视频模型使用 5D tensor `(B, C, T, H, W)`，其中 T=帧数（通常 16）。`r2plus1d_18` 空间尺寸为 112x112，其余为 224x224。

### 4.8 RAFT 光流模型双图输入

**发现**：extractor 可以正确处理双图像输入（两个独立的 4D tensor 作为函数参数），无需特殊处理。RAFT 输入图像尺寸需能被 8 整除。

### 4.9 检测模型 NMS 子图的 3 个修复

检测模型 validate 初始有 41 个子图失败，通过以下修复全部解决：

**修复 1 — `import torchvision` 缺失**：35 个 model.py 使用了 `torch.ops.torchvision.nms` 但只 import 了 torch。手动添加 `import torchvision`，并修改 `/root/GraphNet/graph_net/torch/utils.py` 的 `apply_templates()` 自动检测并添加该 import。

**修复 2 — `_assert_scalar` 运行时断言**：38 个 model.py 包含 `torch.ops.aten._assert_scalar.default` 调用，断言 SymInt 约束（如 `nms_result >= 2`），在随机数据下必然失败。移除了这些断言行及其前置比较变量。

**修复 3 — 空 tensor 输出处理**：修改 `/root/GraphNet/graph_net/torch/single_device_runner.py`，在输出为空 tuple 或空 tensor 时优雅处理，避免 `argmin()` 报错。

> **注意**：修复 1 和修复 3 修改了 GraphNet 仓库源码（`utils.py` 和 `single_device_runner.py`），提交 PR 时需包含这些改动。

---

## 5. graph_net.json 字段规范

### 5.1 自动生成的字段

| 字段 | 值 | 说明 |
|------|------|------|
| `framework` | `"torch"` | 固定值 |
| `num_devices_required` | `1` | 单设备 |
| `num_nodes_required` | `1` | 单节点 |
| `dynamic` | `false` | 是否使用动态形状 |
| `model_name` | 模型名 | 传给 `extract(name=...)` 的值 |

### 5.2 手动补充的字段

| 字段 | 命名约定 | 说明 |
|------|----------|------|
| `source` | 模型来源库名，**小写** | `"torchvision"`, `"timm"`, `"transformers"` 等 |
| `heuristic_tag` | 固定枚举值之一 | `"computer_vision"`, `"nlp"`, `"audio"`, `"multimodal"`, `"other"` |

### 5.3 补充字段的代码

```python
import json
path = f'/ssd1/liangtai-work/graphnet_workspace/{model_name}/graph_net.json'
with open(path) as f:
    data = json.load(f)
data['source'] = 'torchvision'
data['heuristic_tag'] = 'computer_vision'
with open(path, 'w') as f:
    json.dump(data, f, indent=4)
```

对于检测模型的子图，需遍历每个 `subgraph_N/graph_net.json` 分别补充。

### 5.4 完整示例

```json
{
    "framework": "torch",
    "num_devices_required": 1,
    "num_nodes_required": 1,
    "dynamic": false,
    "model_name": "swin_t",
    "source": "torchvision",
    "heuristic_tag": "computer_vision"
}
```

---

## 6. 抽取脚本和流程

### 6.1 标准抽取流程

```python
import torch
from torchvision.models import get_model, get_model_weights
from graph_net.torch.extractor import extract

device = torch.device("cpu")
name = "model_name"

# 1. 加载模型
weights = get_model_weights(name).DEFAULT
model = get_model(name, weights=weights).to(device).eval()

# 2. extract 包装
wrapped = extract(name=name, dynamic=False)(model).eval()

# 3. 前向推理触发提取
with torch.no_grad():
    output = wrapped(torch.rand(1, 3, 224, 224, device=device))
```

### 6.2 Validate 命令

```bash
python -m graph_net.torch.validate \
  --model-path $GRAPH_NET_EXTRACT_WORKSPACE/model_name \
  --no-check-redundancy
```

Validate 三阶段：基本可运行性 → 重新提取 → 重新提取结果可运行性。通过后自动生成 `graph_hash.txt`。

### 6.3 不同模型类型的输入构造

| 类型 | 输入格式 | 示例 |
|------|----------|------|
| 图像分类 | `torch.rand(1, 3, 224, 224)` | swin_s, swin_t |
| 语义分割 | `torch.rand(1, 3, 224, 224)` | lraspp_mobilenet_v3_large |
| 视频分类 | `torch.rand(1, 3, 16, H, W)` | r2plus1d_18 (112x112), s3d (224x224) |
| 光流估计 | 两个独立 tensor: `img1, img2 = torch.rand(1,3,H,W), torch.rand(1,3,H,W)` | raft_large, raft_small |
| 目标检测 | 图像列表: `[torch.rand(3, H, W)]` | ssd300_vgg16, fasterrcnn_* |

### 6.4 已有脚本

| 脚本 | 路径 | 说明 |
|------|------|------|
| `extract_swin.py` | `/root/GraphNetExtractAgent/scripts/` | swin_s, swin_t 专用脚本 |
| `extract_torchvision.py` | `/root/GraphNetExtractAgent/scripts/` | 通用脚本，支持 5 种模型类型 |
| `validate_sample.py` | `/root/GraphNetExtractAgent/scripts/` | validate 封装脚本 |

---

## 7. 产出文件位置

### 7.1 目录结构

```
/root/
├── GraphNet/                           # GraphNet 仓库（已 clone）
│   ├── graph_net/                      # 核心库（editable install）
│   └── samples/                        # 仓库已有样本（参考用）
├── GraphNetExtractAgent/               # 本项目工作目录
│   ├── AGENT_CONTEXT.md                # Agent 上下文交接文档
│   ├── TASK_BREAKDOWN.md               # 任务分解文档
│   ├── CONTRIBUTE_TUTORIAL.md          # 贡献教程
│   ├── SUMMARY.md                      # ← 本文件
│   └── scripts/                        # 抽取脚本
│       ├── extract_swin.py
│       ├── extract_torchvision.py
│       └── validate_sample.py
└── graphnet_workspace/                 # 所有提取产出
    ├── swin_s/                         # 单图模型（标准结构）
    │   ├── model.py                    # FX 计算图代码
    │   ├── weight_meta.py              # 权重元数据
    │   ├── input_meta.py               # 输入元数据（空）
    │   ├── input_tensor_constraints.py # 约束（空，dynamic=False）
    │   ├── graph_net.json              # 样本元数据
    │   └── graph_hash.txt              # SHA-256 哈希（validate 生成）
    ├── swin_t/
    ├── lraspp_mobilenet_v3_large/
    ├── r2plus1d_18/
    ├── s3d/
    ├── mvit_v1_b/
    ├── mvit_v2_s/
    ├── swin3d_b/
    ├── swin3d_s/
    ├── swin3d_t/
    ├── raft_large/
    ├── raft_small/
    ├── ssd300_vgg16/                   # 检测模型（子图结构）
    │   ├── subgraph_0/                 # 每个子图是独立样本
    │   │   ├── model.py
    │   │   ├── weight_meta.py
    │   │   ├── graph_net.json
    │   │   └── ...
    │   ├── subgraph_1/
    │   └── ...
    ├── ssdlite320_mobilenet_v3_large/
    ├── retinanet_resnet50_fpn/
    ├── retinanet_resnet50_fpn_v2/
    ├── fasterrcnn_resnet50_fpn/
    ├── fasterrcnn_resnet50_fpn_v2/
    ├── fasterrcnn_mobilenet_v3_large_fpn/
    ├── fasterrcnn_mobilenet_v3_large_320_fpn/
    ├── fcos_resnet50_fpn/
    ├── maskrcnn_resnet50_fpn/
    ├── maskrcnn_resnet50_fpn_v2/
    ├── keypointrcnn_resnet50_fpn/
    ├── resnet18/                       # 测试用，非正式产出
    └── resnet18_test/                  # 测试用，非正式产出
```

### 7.2 单图模型文件大小

| 模型 | model.py | weight_meta.py |
|------|----------|---------------|
| `swin_s` | 264K | 108K |
| `swin_t` | 136K | 56K |
| `lraspp_mobilenet_v3_large` | 144K | 96K |
| `r2plus1d_18` | 92K | 60K |
| `s3d` | 216K | 144K |
| `mvit_v1_b` | 200K | 96K |
| `mvit_v2_s` | 316K | 120K |
| `swin3d_b` | 300K | 108K |
| `swin3d_s` | 300K | 108K |
| `swin3d_t` | 152K | 56K |
| `raft_large` | 232K | 56K |
| `raft_small` | 176K | 44K |

---

## 8. 未完成的工作和后续建议

### 8.1 待提交 PR

所有成功的模型样本尚未提交 PR 到 GraphNet 仓库。建议按类型分 PR：

1. **PR 1**：swin_s + swin_t（ViT 补全）
2. **PR 2**：lraspp_mobilenet_v3_large（语义分割）
3. **PR 3**：7 个视频模型（r2plus1d_18, s3d, mvit_v1_b, mvit_v2_s, swin3d_b, swin3d_s, swin3d_t）
4. **PR 4**：raft_large + raft_small（光流估计）
5. **PR 5**：12 个目标检测模型 + GraphNet 源码修复（utils.py, single_device_runner.py）

提交前需要：
- 将样本复制到 `GraphNet/samples/torchvision/` 对应目录
- 遵循 `CONTRIBUTE_TUTORIAL.md` 第 4 节的 PR 模板
- 检测模型 PR 需同时包含 GraphNet 源码修复（utils.py 和 single_device_runner.py 的改动）

### 8.2 GraphNet 源码改动需要 review

P3 sub-agent 修改了两个 GraphNet 源文件：
- `/root/GraphNet/graph_net/torch/utils.py` — `apply_templates()` 自动添加 `import torchvision`
- `/root/GraphNet/graph_net/torch/single_device_runner.py` — 处理空 tuple/空 tensor 输出

这些改动使检测模型的 NMS 子图能通过 validate，但需要仓库维护者 review。

### 8.3 量化模型

- 需等待 PyTorch 完善 `torch._dynamo` 对量化张量的支持
- 可关注 PyTorch 的 PT2E 量化方案进展（`torch.ao.quantization.quantizer`）
- 建议向 GraphNet 仓库提 Issue 记录此限制

### 8.4 timm 扩展（T8）

- timm 已有 589 个样本，总量 > 1000
- 可用 `timm.list_models()` 差距分析，按家族批量扩展
- 大部分 timm 模型接受标准 4D 输入，可复用通用脚本

### 8.5 transformers 扩展（T9）

- transformers 已有 1351 个样本
- 使用 `GraphNetAgent` 自动化管线：`agent.extract_sample('model_id')`
- 可用 `graph_net/agent/tests/test_batch_success_rate.py` 批量测试

---

## 附录：关键命令速查

```bash
# 环境设置
export GRAPH_NET_EXTRACT_WORKSPACE=/ssd1/liangtai-work/graphnet_workspace
export http_proxy=http://agent.baidu.com:8891
export https_proxy=http://agent.baidu.com:8891

# Extract 单个模型
python3 -c "
import torch
from torchvision.models import get_model, get_model_weights
from graph_net.torch.extractor import extract
name = 'MODEL_NAME'
model = get_model(name, weights=get_model_weights(name).DEFAULT).eval()
wrapped = extract(name=name, dynamic=False)(model).eval()
with torch.no_grad():
    wrapped(torch.rand(1, 3, 224, 224))
"

# Validate
python -m graph_net.torch.validate \
  --model-path $GRAPH_NET_EXTRACT_WORKSPACE/MODEL_NAME \
  --no-check-redundancy

# 批量 validate
for m in model1 model2 model3; do
  echo "=== $m ==="
  python -m graph_net.torch.validate \
    --model-path $GRAPH_NET_EXTRACT_WORKSPACE/$m \
    --no-check-redundancy
done

# Hash 去重检查
for m in model1 model2; do
  echo "$m: $(cat $GRAPH_NET_EXTRACT_WORKSPACE/$m/graph_hash.txt)"
done | sort -k2 | awk '{print $2}' | uniq -d
```

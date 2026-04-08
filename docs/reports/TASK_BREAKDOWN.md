# GraphNet 抽取任务划分

> 按"先复现 → 再扩展"的思路，把每个任务拆成可执行的子任务。

---

## 一、通用工作流程

每个抽取任务都遵循以下标准流程：

```
┌─────────────────────────────────────────────────────────────────┐
│ Phase 1: 复现与验证                                              │
│  1.1 阅读已有成功样本，理解该类型模型的抽取方式                         │
│  1.2 跑通已有样本的 validate，确认环境正常                            │
│  1.3 手动复现一个已有模型的 extract 流程                              │
├─────────────────────────────────────────────────────────────────┤
│ Phase 2: 设计脚本                                                │
│  2.1 分析目标模型的 forward() 签名，确定输入格式                       │
│  2.2 编写提取脚本（加载模型 → 构造输入 → extract 包装 → 前向推理）      │
│  2.3 先对单个模型调试通过                                            │
├─────────────────────────────────────────────────────────────────┤
│ Phase 3: 批量抽取                                                │
│  3.1 将脚本扩展为支持该类别下所有目标模型                               │
│  3.2 逐个运行 extract，记录成功/失败                                  │
│  3.3 对失败模型分析错误日志，调整输入或报 Issue                         │
├─────────────────────────────────────────────────────────────────┤
│ Phase 4: 验证与提交                                               │
│  4.1 对每个成功样本运行 validate                                     │
│  4.2 将通过验证的样本移入 samples/ 目录                               │
│  4.3 按 PR 模板提交（每个模型或每类改进一个 PR）                        │
└─────────────────────────────────────────────────────────────────┘
```

### 关键命令

```bash
# 设置环境变量（每次新会话）
export GRAPH_NET_EXTRACT_WORKSPACE=/ssd1/liangtai-work/graphnet_workspace

# 运行 extract
cd /root/GraphNet && python your_script.py

# 运行 validate
python -m graph_net.torch.validate --model-path $GRAPH_NET_EXTRACT_WORKSPACE/model_name

# 也可以用 Agent 管线（针对 HuggingFace 模型）
python -c "
from graph_net.agent import GraphNetAgent
agent = GraphNetAgent()
agent.extract_sample('model_id')
"
```

### 项目中已有的两种抽取方式

| 方式 | 适用场景 | 说明 |
|------|---------|------|
| **直接 extract** | torchvision、timm 等可直接 `import` 的模型 | 手写脚本，用 `extract(name)(model)` 包装 |
| **Agent 管线** | HuggingFace Hub 上的模型 | `GraphNetAgent` 自动下载、分析 config、生成脚本、运行提取、验证 |

---

## 二、任务总览

| 任务 | 目标 | 模型数 | 难度 | 抽取方式 |
|------|------|--------|------|----------|
| T1 | 环境验证 + 已有样本复现 | - | 入门 | 直接 extract |
| T2 | torchvision - ViT 补全 | 2 | 低 | 直接 extract |
| T3 | torchvision - 语义分割补全 | 1 | 低 | 直接 extract |
| T4 | torchvision - 视频模型 | 7 | 中 | 直接 extract |
| T5 | torchvision - 光流估计 | 2 | 中 | 直接 extract |
| T6 | torchvision - 目标检测 | 12 | 中-高 | 直接 extract |
| T7 | torchvision - 量化模型 | 12 | 高 | 直接 extract |
| T8 | timm 扩展 | 待定 | 中 | 直接 extract |
| T9 | transformers 扩展 | 待定 | 中-高 | Agent 管线 |

---

## T1: 环境验证 + 已有样本复现（入门任务）

> 目的：确保环境完好，理解完整流程。不产出新样本。

### 子任务

| # | 子任务 | 具体操作 | 预期输出 |
|---|--------|---------|---------|
| 1.1 | 验证已有样本可运行 | `python -m graph_net.torch.run_model --model-path /ssd1/liangtai-work/graphnet_workspace/resnet18_test` | 模型成功执行前向推理 |
| 1.2 | 验证 validate 可用 | `python -m graph_net.torch.validate --model-path /ssd1/liangtai-work/graphnet_workspace/resnet18_test` | validate 通过 |
| 1.3 | 复现 resnet18 的 extract | 手写脚本提取 resnet18，与已有输出对比 | 生成结构一致的输出文件 |
| 1.4 | 阅读已有 torchvision 测试脚本 | 读 `graph_net/test/vision_model_test.py`（如存在） | 理解批量抽取模式 |
| 1.5 | 阅读 Agent 管线代码 | 读 `graph_net/agent/graph_net_agent.py` | 理解自动化流程 |

### 复现脚本示例

```python
import torch
from torchvision.models import get_model, get_model_weights
from graph_net.torch.extractor import extract

name = "resnet18"
weights = get_model_weights(name).DEFAULT
model = get_model(name, weights=weights).eval()
wrapped = extract(name=f"{name}_reproduce")(model).eval()
with torch.no_grad():
    wrapped(torch.rand(1, 3, 224, 224))
print(f"[OK] {name}")
```

---

## T2: torchvision - Vision Transformer 补全（2 个）

> 标准 4D 图像输入，已有 vit_b_16 等成功样本可参考。

### 子任务

| # | 子任务 | 说明 |
|---|--------|------|
| 2.1 | 阅读已有 ViT 样本 | 查看 `samples/torchvision/vit_b_16/` 的 `weight_meta.py` 和 `model.py`，了解 ViT 的抽取特点（class token 作为额外输入） |
| 2.2 | 编写 swin_s 抽取脚本 | 加载 `swin_s`，输入 `torch.rand(1, 3, 224, 224)`，运行 extract |
| 2.3 | 运行 extract + validate | 验证 swin_s 输出 |
| 2.4 | 扩展到 swin_t | 同一脚本，换模型名 |
| 2.5 | 运行 validate | 验证 swin_t |

### 目标模型

| 模型 | 输入 | 参考样本 |
|------|------|---------|
| `swin_s` | `torch.rand(1, 3, 224, 224)` | `samples/torchvision/vit_b_16/` |
| `swin_t` | `torch.rand(1, 3, 224, 224)` | 同上 |

---

## T3: torchvision - 语义分割补全（1 个）

> 输入是标准图像，输出是像素级分割图。extract 只关心输入侧，难度低。

### 子任务

| # | 子任务 | 说明 |
|---|--------|------|
| 3.1 | 阅读已有分割样本 | 查看 `samples/torchvision/deeplabv3_resnet50/` 的结构 |
| 3.2 | 确定输入分辨率 | 检查 `lraspp_mobilenet_v3_large` 的默认输入尺寸 |
| 3.3 | 编写脚本 + extract | 运行提取 |
| 3.4 | validate | 验证输出 |

### 目标模型

| 模型 | 输入 | 备注 |
|------|------|------|
| `lraspp_mobilenet_v3_large` | `torch.rand(1, 3, 520, 520)` | 轻量语义分割 |

---

## T4: torchvision - 视频模型（7 个）

> **核心挑战**：输入从 4D 变为 5D `(B, C, T, H, W)`，需要额外指定帧数 `T`。

### 子任务

| # | 子任务 | 说明 |
|---|--------|------|
| 4.1 | 阅读已有视频样本 | 查看 `samples/torchvision/mc3_18/` 和 `r3d_18/`，分析 5D 输入的 weight_meta |
| 4.2 | 研究各模型的 forward 签名 | `import inspect; inspect.signature(model.forward)` 确定每个模型的输入要求 |
| 4.3 | 编写 r2plus1d_18 脚本 | 教程已提到该模型需要 `(1, 3, T, H, W)` 输入 |
| 4.4 | 调试 r2plus1d_18 | 如遇错误参考教程 2.3 节的排错方法 |
| 4.5 | 扩展到 s3d | 同样 5D 输入，可能 H/W 不同 |
| 4.6 | 扩展到 mvit_v1_b, mvit_v2_s | MViT 系列，注意可能有不同的帧数要求 |
| 4.7 | 扩展到 swin3d_b, swin3d_s, swin3d_t | Video Swin 系列 |
| 4.8 | 批量 validate | 对所有成功样本运行 validate |

### 目标模型

| 模型 | 预期输入 | 参考 |
|------|----------|------|
| `r2plus1d_18` | `torch.rand(1, 3, 16, 112, 112)` | 教程 2.3 节 |
| `s3d` | `torch.rand(1, 3, 16, 224, 224)` | mc3_18 样本 |
| `mvit_v1_b` | `torch.rand(1, 3, 16, 224, 224)` | 需验证 |
| `mvit_v2_s` | `torch.rand(1, 3, 16, 224, 224)` | 需验证 |
| `swin3d_b` | `torch.rand(1, 3, 16, 224, 224)` | 需验证 |
| `swin3d_s` | `torch.rand(1, 3, 16, 224, 224)` | 需验证 |
| `swin3d_t` | `torch.rand(1, 3, 16, 224, 224)` | 需验证 |

> 注意事项：
> - 已有 `mc3_18` 和 `r3d_18` 用的输入是 `[1, 3, 16, 224, 224]`
> - 但 r2plus1d_18 可能需要 `112x112` 而不是 `224x224`（参考教程）
> - MViT 和 Swin3D 的帧数可能不是 16，需要检查

---

## T5: torchvision - 光流估计（2 个）

> **核心挑战**：RAFT 模型接受一**对**图像作为输入（双图像），extract 能否正确 trace 元组输入需要验证。

### 子任务

| # | 子任务 | 说明 |
|---|--------|------|
| 5.1 | 研究 RAFT forward 签名 | `inspect.signature(raft_large.forward)` 确认输入格式 |
| 5.2 | 尝试 extract raft_large | 用 `(img1, img2)` 元组输入 |
| 5.3 | 如果失败 → 分析错误 | extract 可能不支持元组输入，需要特殊处理 |
| 5.4 | 如果成功 → validate | 验证输出 |
| 5.5 | 扩展到 raft_small | 同样的输入模式 |

### 目标模型

| 模型 | 预期输入 |
|------|----------|
| `raft_large` | `(torch.rand(1, 3, 520, 960), torch.rand(1, 3, 520, 960))` |
| `raft_small` | 同上 |

---

## T6: torchvision - 目标检测（12 个）

> **核心挑战**：
> 1. 输入是**图像列表** `[tensor1, tensor2, ...]` 而非批次张量
> 2. 不同检测模型的输入尺寸不同（SSD: 300x300, FasterRCNN: 800x800 等）
> 3. extract 装饰器对列表输入的兼容性未知

### 子任务

| # | 子任务 | 说明 |
|---|--------|------|
| 6.1 | 研究检测模型输入格式 | 阅读 torchvision 文档，确认各检测模型的输入要求 |
| 6.2 | 先试 ssd300_vgg16 | SSD 结构最简单，输入 `[torch.rand(3, 300, 300)]` |
| 6.3 | 分析 extract 对列表输入的处理 | 如果报错，阅读 `extractor.py` 源码分析原因 |
| 6.4 | 如需修改 extractor → 提 Issue | 这属于"extractor 改进"，需单独 PR |
| 6.5 | ssd 系列 (2 个) | `ssd300_vgg16`, `ssdlite320_mobilenet_v3_large` |
| 6.6 | retinanet 系列 (2 个) | `retinanet_resnet50_fpn`, `retinanet_resnet50_fpn_v2` |
| 6.7 | fasterrcnn 系列 (4 个) | 4 个变体 |
| 6.8 | fcos (1 个) | `fcos_resnet50_fpn` |
| 6.9 | maskrcnn 系列 (2 个) | `maskrcnn_resnet50_fpn`, `v2` |
| 6.10 | keypointrcnn (1 个) | `keypointrcnn_resnet50_fpn` |
| 6.11 | 批量 validate | 对所有成功样本验证 |

### 目标模型

| 子类 | 模型 | 预期输入 |
|------|------|----------|
| SSD | `ssd300_vgg16` | `[torch.rand(3, 300, 300)]` |
| SSD | `ssdlite320_mobilenet_v3_large` | `[torch.rand(3, 320, 320)]` |
| RetinaNet | `retinanet_resnet50_fpn` | `[torch.rand(3, 800, 800)]` |
| RetinaNet | `retinanet_resnet50_fpn_v2` | `[torch.rand(3, 800, 800)]` |
| FasterRCNN | `fasterrcnn_resnet50_fpn` | `[torch.rand(3, 800, 800)]` |
| FasterRCNN | `fasterrcnn_resnet50_fpn_v2` | `[torch.rand(3, 800, 800)]` |
| FasterRCNN | `fasterrcnn_mobilenet_v3_large_fpn` | `[torch.rand(3, 800, 800)]` |
| FasterRCNN | `fasterrcnn_mobilenet_v3_large_320_fpn` | `[torch.rand(3, 320, 320)]` |
| FCOS | `fcos_resnet50_fpn` | `[torch.rand(3, 800, 800)]` |
| MaskRCNN | `maskrcnn_resnet50_fpn` | `[torch.rand(3, 800, 800)]` |
| MaskRCNN | `maskrcnn_resnet50_fpn_v2` | `[torch.rand(3, 800, 800)]` |
| Keypoint | `keypointrcnn_resnet50_fpn` | `[torch.rand(3, 800, 800)]` |

---

## T7: torchvision - 量化模型（12 个）

> **核心挑战**：量化模型使用 `torch.quantization` 的特殊算子（`QuantizedConv2d`、`QuantizedLinear` 等），`torch.compile` 可能无法正确 trace 这些算子。
>
> **这是实验性任务**，成功率不确定。

### 子任务

| # | 子任务 | 说明 |
|---|--------|------|
| 7.1 | 可行性验证 | 用 `quantized_resnet18` 做最小实验：能否 `torch.compile` 量化模型？ |
| 7.2 | 如果可以 → 编写批量脚本 | 量化模型输入和非量化版本相同，都是 `torch.rand(1, 3, 224, 224)` |
| 7.3 | 如果不行 → 记录限制并提 Issue | 说明哪些量化算子不支持 trace |
| 7.4 | 尝试替代方案 | 如 `torch.jit.trace` 或其他方式 |

### 目标模型（12 个）

`quantized_resnet18`, `quantized_resnet50`, `quantized_mobilenet_v2`, `quantized_mobilenet_v3_large`, `quantized_googlenet`, `quantized_inception_v3`, `quantized_resnext101_32x8d`, `quantized_resnext101_64x4d`, `quantized_shufflenet_v2_x0_5`, `quantized_shufflenet_v2_x1_0`, `quantized_shufflenet_v2_x1_5`, `quantized_shufflenet_v2_x2_0`

---

## T8: timm 扩展（待定）

> timm 已有 589 个样本，但 timm 模型总量 > 1000。可按模型家族批量扩展。

### 子任务

| # | 子任务 | 说明 |
|---|--------|------|
| 8.1 | 差距分析 | 用 `timm.list_models()` 列出所有模型，与 `samples/timm/` 对比，找出缺失列表 |
| 8.2 | 按家族分组 | 将缺失模型按家族分组（ConvNeXt、EfficientNet、DeiT 等） |
| 8.3 | 编写通用 timm 抽取脚本 | timm 模型大多接受标准 4D 输入，可写一个通用脚本 |
| 8.4 | 批量运行 | 逐家族运行，记录成功/失败 |
| 8.5 | 处理特殊模型 | 对失败模型分析原因，调整输入 |
| 8.6 | 批量 validate + 提交 | 分批 PR |

### 通用 timm 脚本思路

```python
import timm
from graph_net.torch.extractor import extract

for name in target_models:
    model = timm.create_model(name, pretrained=False).eval()
    cfg = model.default_cfg
    input_size = cfg.get("input_size", (3, 224, 224))
    input_data = torch.rand(1, *input_size)
    wrapped = extract(name=name)(model).eval()
    with torch.no_grad():
        wrapped(input_data)
```

---

## T9: transformers 扩展（待定）

> transformers 已有 1351 个样本。使用 **Agent 管线**自动化抽取。

### 子任务

| # | 子任务 | 说明 |
|---|--------|------|
| 9.1 | 理解 Agent 管线 | 阅读 `graph_net/agent/graph_net_agent.py`，理解全流程 |
| 9.2 | 复现已有模型 | 用 Agent 管线跑一个已知成功的 HuggingFace 模型 |
| 9.3 | 差距分析 | 对比已有样本和热门 HuggingFace 模型列表 |
| 9.4 | 按架构分批 | BERT 系、GPT 系、T5 系、Vision 系、Speech 系 |
| 9.5 | 利用批量测试框架 | 使用 `graph_net/agent/tests/test_batch_success_rate.py` 批量跑 |
| 9.6 | 处理失败模型 | 分析 `ConfigMetadataAnalyzer` 对新模型架构的支持情况 |
| 9.7 | 如需改进 Agent → 提 PR | 属于 Feature Enhancement 类 PR |

### Agent 管线命令

```python
from graph_net.agent import GraphNetAgent

agent = GraphNetAgent()
# 单模型
result = agent.extract_sample("bert-base-uncased")
# 批量（使用测试框架）
# python -m graph_net.agent.tests.test_batch_success_rate
```

---

## 三、建议执行顺序

```
T1 环境验证               ← 必须先做，确保一切正常
 ↓
T2 ViT 补全 (2个)         ← 最简单，建立信心
T3 分割补全 (1个)          ← 顺手完成
 ↓
T4 视频模型 (7个)          ← 中等挑战，5D 输入
 ↓
T5 光流 (2个)             ← 小任务，验证元组输入
 ↓
T6 目标检测 (12个)         ← 较大工作量，列表输入
 ↓
T7 量化模型 (12个)         ← 实验性，可能需要 extractor 改进
 ↓
T8 timm 扩展              ← 长期任务，可并行
T9 transformers 扩展       ← 长期任务，可并行
```

### 并行建议

- T2 + T3 可以同时做（都是简单图像模型）
- T8 + T9 完全独立，可并行
- T4/T5/T6/T7 建议顺序做（后一个的经验建立在前一个之上）

---

## 四、每个任务的交付物

每个任务完成后应产出：

1. **提取脚本**：保存在 `/root/GraphNetExtractAgent/scripts/` 目录下
2. **成功样本**：保存在 `$GRAPH_NET_EXTRACT_WORKSPACE/` 下，validate 通过
3. **失败记录**：哪些模型失败了、错误日志、原因分析
4. **PR 提交**：按 SRP 原则，每个模型或每类改进一个 PR

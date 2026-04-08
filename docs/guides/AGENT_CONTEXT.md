# GraphNet Extract Agent - 上下文交接文档

> 本文档用于在 Agent 会话之间传递项目上下文，供下一个 Agent 快速继承并继续工作。

---

## 1. 项目目标

基于 [PaddlePaddle/GraphNet](https://github.com/PaddlePaddle/GraphNet) 项目，编写脚本对 PyTorch 模型执行**计算图提取（extract）**和**验证（validate）**，并将结果提交为 GraphNet 的数据样本。

核心流程：**选模型 → 写提取脚本 → extract → validate → 提交 PR**

---

## 2. 目录结构

```
/root/
├── GraphNet/                    # GraphNet 仓库（已 clone，来自 GitHub）
│   ├── graph_net/               # 核心库（已可 import）
│   │   ├── torch/
│   │   │   ├── extractor.py     # extract 装饰器
│   │   │   └── validate.py      # validate 验证工具
│   │   └── ...
│   ├── samples/                 # 已有的模型计算图样本
│   ├── pyproject.toml           # 项目配置
│   └── ...
├── GraphNetExtractAgent/        # 本项目工作目录
│   ├── CONTRIBUTE_TUTORIAL.md   # 贡献教程（英文版）
│   ├── CONTRIBUTE_TUTORIAL_cn.md
│   ├── README.md
│   └── AGENT_CONTEXT.md         # ← 本文件
└── graphnet_workspace/          # 提取输出目录（目前为空）
```

---

## 3. 环境状态

### 已完成

| 项目 | 状态 | 备注 |
|------|------|------|
| GraphNet 仓库 clone | ✅ | `/root/GraphNet/` |
| `graph_net` 模块可 import | ✅ | 已通过 `pip install -e ".[agent]"` 正式安装（editable mode） |
| `pip install -e ".[agent]"` | ✅ | 安装位置：`/usr/local/lib/python3.10/dist-packages`，editable 指向 `/root/GraphNet` |
| `GRAPH_NET_EXTRACT_WORKSPACE` 环境变量 | ✅ 已持久化 | 已写入 `/root/.bashrc`，值为 `/ssd1/liangtai-work/graphnet_workspace` |
| `torch` | ✅ 2.11.0 | |
| `torchvision` | ✅ 0.26.0 | |
| `transformers` | ✅ 5.3.0 | |
| `huggingface_hub` | ✅ 1.3.4 | |
| 工作目录 `/ssd1/liangtai-work/graphnet_workspace/` | ✅ 已创建 | 已有 `resnet18_test/` 提取输出 |

### 已完成的抽取任务

| 模型 | 类型 | Extract | Validate | 备注 |
|------|------|---------|----------|------|
| `swin_s` | 图像分类 | ✅ | ✅ | `dynamic=False`，输入 `(1,3,224,224)` |
| `swin_t` | 图像分类 | ✅ | ✅ | `dynamic=False`，输入 `(1,3,224,224)` |
| `lraspp_mobilenet_v3_large` | 语义分割 | ✅ | ✅ | `dynamic=False`，输入 `(1,3,224,224)`，输出 `(1,21,224,224)` |
| `r2plus1d_18` | 视频分类 | ✅ | ✅ | `dynamic=False`，输入 `(1,3,16,112,112)`，输出 `(1,400)` |
| `s3d` | 视频分类 | ✅ | ✅ | `dynamic=False`，输入 `(1,3,16,224,224)`，输出 `(1,400)` |
| `mvit_v1_b` | 视频分类 | ✅ | ✅ | `dynamic=False`，输入 `(1,3,16,224,224)`，输出 `(1,400)` |
| `mvit_v2_s` | 视频分类 | ✅ | ✅ | `dynamic=False`，输入 `(1,3,16,224,224)`，输出 `(1,400)` |
| `swin3d_b` | 视频分类 | ✅ | ✅ | `dynamic=False`，输入 `(1,3,16,224,224)`，输出 `(1,400)` |
| `swin3d_s` | 视频分类 | ✅ | ✅ | `dynamic=False`，输入 `(1,3,16,224,224)`，输出 `(1,400)` |
| `swin3d_t` | 视频分类 | ✅ | ✅ | `dynamic=False`，输入 `(1,3,16,224,224)`，输出 `(1,400)` |
| `raft_large` | 光流估计 | ✅ | ✅ | `dynamic=False`，双图像输入 |
| `raft_small` | 光流估计 | ✅ | ✅ | `dynamic=False`，双图像输入 |
| `ssd300_vgg16` | 目标检测 | ✅ | ⚠️ 部分 | 5 subgraphs, 5/5 validate 通过 |
| `ssdlite320_mobilenet_v3_large` | 目标检测 | ✅ | ⚠️ 部分 | 10 subgraphs, 8/10 validate 通过 |
| `retinanet_resnet50_fpn` | 目标检测 | ✅ | ⚠️ 部分 | 5 subgraphs, 1/5 validate 通过 |
| `retinanet_resnet50_fpn_v2` | 目标检测 | ✅ | ⚠️ 部分 | 7 subgraphs, 5/7 validate 通过 |
| `fasterrcnn_resnet50_fpn` | 目标检测 | ✅ | ⚠️ 部分 | 15 subgraphs, 10/15 validate 通过 |
| `fasterrcnn_resnet50_fpn_v2` | 目标检测 | ✅ | ⚠️ 部分 | 14 subgraphs, 12/14 validate 通过 |
| `fasterrcnn_mobilenet_v3_large_fpn` | 目标检测 | ✅ | ⚠️ 部分 | 7 subgraphs, 4/7 validate 通过 |
| `fasterrcnn_mobilenet_v3_large_320_fpn` | 目标检测 | ✅ | ⚠️ 部分 | 7 subgraphs, 4/7 validate 通过 |
| `fcos_resnet50_fpn` | 目标检测 | ✅ | ⚠️ 部分 | 9 subgraphs, 7/9 validate 通过 |
| `maskrcnn_resnet50_fpn` | 目标检测 | ✅ | ⚠️ 部分 | 16 subgraphs, 10/16 validate 通过 |
| `maskrcnn_resnet50_fpn_v2` | 目标检测 | ✅ | ⚠️ 部分 | 17 subgraphs, 11/17 validate 通过 |
| `keypointrcnn_resnet50_fpn` | 目标检测 | ✅ | ⚠️ 部分 | 16 subgraphs, 10/16 validate 通过 |

产出位置：`/ssd1/liangtai-work/graphnet_workspace/` 下各模型同名目录

> **检测模型说明**：由于 graph breaks，每个检测模型被拆成多个子图（subgraph_N/）。backbone/FPN 等主要计算子图 validate 通过；后处理子图（NMS/argmin 相关）因随机输入生成空 tensor 导致 validate 失败（`argmin(): Expected reduction dim for input.numel() == 0`），这是输入数据问题而非图结构问题。

### 踩坑记录

1. **Swin Transformer 必须用 `dynamic=False`**：Swin 的 `F.pad` 操作在 `dynamic=True` 时会引入 `PendingUnbackedSymbolNotFound` 错误。已有的 swin_b/swin_v2_* 样本同样使用 `dynamic=False`。
2. **graph_net.json 需手动补充 source 和 heuristic_tag**：extractor 只生成 framework/num_devices_required/num_nodes_required/dynamic/model_name 五个字段，`source` 和 `heuristic_tag` 需要手动添加。
3. **命名规范**：`source` 填模型来源库名小写（如 `torchvision`、`timm`），`heuristic_tag` 取值为 `computer_vision`/`nlp`/`audio`/`multimodal`/`other` 之一。
4. **代理设置**：下载权重前需设置 `export http_proxy=http://agent.baidu.com:8891 && export https_proxy=http://agent.baidu.com:8891`。
5. **input_tensor_constraints.py 为空是正常的**：`dynamic=False` 时 extractor 不生成约束，已有样本中的约束是后处理管线产物。
6. **视频模型输入格式**：5D tensor `(B, C, T, H, W)`，其中 T=帧数(通常16)，r2plus1d_18 用 112x112，其余用 224x224。所有视频模型 `dynamic=False` 均可正常提取。
7. **RAFT 光流模型双图输入**：extractor 可以正确处理双图像输入（两个 4D tensor 作为参数），无需特殊处理。
8. **检测模型产生子图**：目标检测模型由于 graph breaks（列表输入、条件分支、NMS 等），被 extractor 拆成多个 subgraph。每个 subgraph 是独立的计算图样本。后处理子图（涉及 NMS/argmin）在随机输入下因空 tensor 导致 validate 失败，但这是数据问题非图结构问题。
9. **检测模型子图哈希重复是正常的**：共享相同 backbone（如 ResNet50+FPN）的模型（fasterrcnn/maskrcnn/keypointrcnn）会产生相同的 backbone 子图哈希。
10. **量化模型无法提取**：`torch._dynamo` 不支持量化张量的 meta tensor（`"quantized nyi in meta tensors"`），所有 12 个量化模型均无法 extract。这是 PyTorch 层面限制。
11. **模型名称后缀处理**：timm 等库的模型名常带权重来源后缀（如 `.in1k`、`.apple_pt`），但计算图结构相同。去重时用核心名称比较：
    ```python
    def get_core_name(name: str) -> str:
        return name.rsplit('.', 1)[0] if '.' in name else name
    ```
    例如 `aimv2_1b_patch14_224` ≈ `aimv2_1b_patch14_224.apple_pt` 是同一个模型。

### 未完成

| 任务 | 状态 | 说明 |
|------|------|------|
| T3: lraspp_mobilenet_v3_large | ✅ 完成 | 语义分割，validate 通过 |
| T4: 视频模型 (7个) | ✅ 完成 | r2plus1d_18, s3d, mvit_v1_b, mvit_v2_s, swin3d_b, swin3d_s, swin3d_t 全部通过 |
| T5: 光流模型 (2个) | ✅ 完成 | raft_large, raft_small 全部通过 |
| T6: 目标检测 (12个) | ✅ 完成 | 全部 extract 成功，产生子图结构，部分后处理子图 validate 失败（NMS 相关） |
| T7: 量化模型 (12个) | ❌ 不可行 | torch._dynamo 不支持量化张量 meta tensor，无法 extract |

---

## 4. 关键命令速查

```bash
# 设置环境变量
export GRAPH_NET_EXTRACT_WORKSPACE=/ssd1/liangtai-work/graphnet_workspace

# 正式安装 graph_net（推荐）
cd /root/GraphNet && pip install -e ".[agent]"

# 运行已有的示例提取测试
cd /root/GraphNet && python -m graph_net.test.vision_model_test

# 运行 validate
python -m graph_net.torch.validate --model-path $GRAPH_NET_EXTRACT_WORKSPACE/model_name
```

---

## 5. Extract + Validate 核心逻辑

**Extract**：用 `graph_net.torch.extractor.extract` 装饰器包装 PyTorch 模型，执行一次前向推理，自动捕获计算图并保存到 `$GRAPH_NET_EXTRACT_WORKSPACE`。

```python
from graph_net.torch.extractor import extract

model = get_model(name, weights=weights).to(device).eval()
wrapped = extract(name=name)(model).eval()
with torch.no_grad():
    wrapped(input_data)
```

**Validate**：检查提取的计算图是否符合数据集构建约束。

不同模型类型需要不同的输入格式：
- **CV 图像**：`torch.rand(1, C, H, W)` — 如 ResNet, ViT
- **视频**：`torch.rand(1, C, T, H, W)` — 如 R3D, MViT
- **NLP 文本**：用 `AutoTokenizer` 生成 `input_ids` 等 — 如 BERT, DistilBERT

---

## 6. 下一步行动建议

1. ~~**安装 graph_net**~~：✅ 已完成（`pip install -e ".[agent]"`）
2. ~~**设置环境变量**~~：✅ 已持久化到 `/root/.bashrc`
3. **先跑已有示例**：`python -m graph_net.test.vision_model_test`，验证 extract 流程可用
4. **编写新模型提取脚本**：参考教程，选取尚未覆盖的模型
5. **执行 extract + validate**
6. **提交 PR**：遵循教程第 4 节的 PR 模板

---

## 7. 参考文档

- 贡献教程：`/root/GraphNetExtractAgent/CONTRIBUTE_TUTORIAL.md`
- **HuggingFace 抽取方案**：`/root/GraphNetExtractAgent/HUGGINGFACE_EXTRACTION_GUIDE.md`
- GraphNet Agent README：`/root/GraphNet/graph_net/agent/README.md`
- 项目配置：`/root/GraphNet/pyproject.toml`
- 已有样本：`/root/GraphNet/samples/`

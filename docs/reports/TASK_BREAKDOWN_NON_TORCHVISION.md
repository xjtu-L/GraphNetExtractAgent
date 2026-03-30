# GraphNet 抽取子任务：非 torchvision 来源

> 本文档是 TASK_BREAKDOWN.md 的子任务扩展，覆盖 torchvision 以外的所有模型来源。
> torchvision 的任务拆分见 [TASK_BREAKDOWN.md](TASK_BREAKDOWN.md)。
> 通用工作流程（Phase 1-4）同样适用，不再重复。

---

## 缺口总览

| 来源 | 已抽取 | 预估总量 | 未抽取 | 抽取方式 | 难度 |
|------|--------|---------|--------|----------|------|
| **timm** | 589 | 1284 | **~1196** | 直接 extract | 中 |
| **transformers** | 1360 | 数万 | **大量** | Agent 管线 | 中-高 |
| **ultralytics** | 89 | ~100+ | **少量** | 直接 extract | 中 |
| **mmseg** | 124 | ~150+ | **待查** | 直接 extract | 中 |
| **mmpose** | 82 | ~100+ | **待查** | 直接 extract | 中 |
| **torchaudio** | 10 | ~20+ | **~10+** | 直接 extract | 中 |
| **torchgeometric** | 14 | ~40+ | **~26+** | 直接 extract | 中-高 |
| **nemo** | 16 | ~50+ | **~34+** | 直接 extract | 高 |
| **cosyvoice** | 2 | ~2 | 0 | - | 已完成 |

---

## T-timm: timm 模型扩展（~1196 个未抽取）

> **ROI 最高的任务**：模型多、输入格式统一（绝大多数是标准 4D 图像），可批量脚本化。

### 缺口按家族分组（Top 20）

| 家族 | 未抽取 | 示例模型 | 输入格式 |
|------|--------|---------|----------|
| vit | 194 | vit_base_patch14_dinov2, vit_large_patch16_224 | `(1,3,224,224)` |
| efficientnet | 35 | efficientnet_b0, efficientnet_b3_gn | `(1,3,224,224)` ~ `(1,3,300,300)` |
| swinv2 | 28 | swinv2_base_window12_192, swinv2_large_window12to24 | `(1,3,192,192)` ~ `(1,3,384,384)` |
| xcit | 28 | xcit_large_24_p16_224, xcit_small_12_p8_384 | `(1,3,224,224)` / `(1,3,384,384)` |
| maxvit | 27 | maxvit_base_tf_224, maxvit_xlarge_tf_512 | `(1,3,224,224)` ~ `(1,3,512,512)` |
| tf_efficientnet | 21 | tf_efficientnet_b0 ~ b8 | 同 efficientnet |
| levit | 20 | levit_128, levit_384 | `(1,3,224,224)` |
| resnetv2 | 20 | resnetv2_101, resnetv2_50x1_bit | `(1,3,224,224)` |
| convnext | 18 | convnext_atto, convnext_xlarge | `(1,3,224,224)` |
| regnety | 18 | regnety_002 ~ regnety_320 | `(1,3,224,224)` |
| vitamin | 13 | vitamin_base_224, vitamin_xlarge_384 | `(1,3,224,224)` / `(1,3,384,384)` |
| efficientvit | 13 | efficientvit_b0 ~ efficientvit_l3 | `(1,3,224,224)` ~ `(1,3,512,512)` |
| fastvit | 12 | fastvit_ma36, fastvit_sa12 | `(1,3,256,256)` |
| mobilenetv4 | 12 | mobilenetv4_conv_aa_large | `(1,3,224,224)` ~ `(1,3,384,384)` |
| focalnet | 12 | focalnet_base_lrf, focalnet_huge_fl3 | `(1,3,224,224)` |
| eva02 | 12 | eva02_base_patch14_224 | `(1,3,224,224)` |
| repvgg | 11 | repvgg_a0 ~ repvgg_d2se | `(1,3,224,224)` |
| hrnet | 11 | hrnet_w18 ~ hrnet_w64 | `(1,3,224,224)` |
| crossvit | 11 | crossvit_15_240, crossvit_base_240 | `(1,3,240,240)` |
| volo | 11 | volo_d1_224 ~ volo_d5_512 | `(1,3,224,224)` ~ `(1,3,512,512)` |
| 其余 245 个家族 | ~416 | ... | 大多标准图像输入 |

### 子任务

| # | 子任务 | 说明 |
|---|--------|------|
| timm-1 | 差距分析脚本 | 用 `timm.list_models()` 对比 `samples/timm/`，输出缺失列表并按家族分组 |
| timm-2 | 编写通用 timm 抽取脚本 | 利用 `timm.create_model()` + `model.default_cfg["input_size"]` 自动获取输入尺寸 |
| timm-3 | 批量抽取：标准图像家族 | 先跑 efficientnet、convnext、resnetv2 等纯 4D 输入的家族（~200 个） |
| timm-4 | 批量抽取：ViT 家族 | vit 系列 194 个，部分有额外 cls_token 输入（参考已有 timm ViT 样本） |
| timm-5 | 批量抽取：Swin/MaxViT 家族 | swinv2 28 + maxvit 27，注意不同分辨率 |
| timm-6 | 批量抽取：其余家族 | xcit、levit、fastvit、focalnet 等 |
| timm-7 | 处理失败模型 | 分析错误日志，调整输入或标记为不支持 |
| timm-8 | 批量 validate + 提交 | 分批 PR，每个家族或每批模型一个 PR |

### 通用脚本思路

```python
import torch, timm, os, sys
from graph_net.torch.extractor import extract

os.environ.setdefault("GRAPH_NET_EXTRACT_WORKSPACE", "/root/graphnet_workspace")

def extract_timm_model(name, device="cpu"):
    model = timm.create_model(name, pretrained=False).eval()
    cfg = model.default_cfg
    input_size = cfg.get("input_size", (3, 224, 224))
    input_data = torch.rand(1, *input_size, device=device)
    wrapped = extract(name=name)(model).eval()
    with torch.no_grad():
        wrapped(input_data)
    print(f"[OK] {name}")

# 批量跑
for name in target_list:
    try:
        extract_timm_model(name)
    except Exception as e:
        print(f"[FAIL] {name}: {e}")
```

---

## T-transformers: HuggingFace transformers 扩展

> 已抽取 1360 个，但热门大模型（LLaMA、Mistral、Gemma、Phi 等）几乎全部缺失。
> 使用项目已有的 **Agent 管线**自动化抽取。

### 已覆盖 vs 未覆盖的热门架构

| 架构 | 状态 | 说明 |
|------|------|------|
| BERT / DistilBERT / ALBERT | ✅ 已覆盖 | 多个变体 |
| GPT-2 / OPT | ✅ 已覆盖 | 小模型 |
| T5 / mT5 | ✅ 部分覆盖 | |
| ViT / BEiT / DeiT | ✅ 已覆盖 | 视觉模型 |
| DETR / Deformable-DETR | ✅ 已覆盖 | 检测 |
| Whisper | ✅ 1 个 | 语音 |
| CLIP | ✅ 2 个 | 多模态 |
| Qwen | ✅ 3 个 | |
| **LLaMA / Llama-2/3** | ❌ 缺失 | 热门 LLM |
| **Mistral / Mixtral** | ❌ 缺失 | 热门 LLM |
| **Gemma / Gemma-2** | ❌ 缺失 | Google LLM |
| **Phi / Phi-2/3** | ❌ 缺失 | Microsoft LLM |
| **BLOOM** | ❌ 缺失 | BigScience LLM |
| **Falcon** | ❌ 缺失 | TII LLM |
| **GPT-NeoX / Pythia** | ❌ 缺失 | EleutherAI |
| **BGE / E5** | ❌ 缺失 | Embedding 模型 |
| **Stable Diffusion (UNet)** | ❌ 缺失 | 图像生成 |

### 子任务

| # | 子任务 | 说明 |
|---|--------|------|
| tf-1 | 复现 Agent 管线 | 用已知模型（如 `bert-base-uncased`）跑通 `GraphNetAgent.extract_sample()` |
| tf-2 | 小模型批量测试 | 使用 `test_batch_success_rate.py` 跑 100 个小模型，验证管线稳定性 |
| tf-3 | NLP 类扩展 | 抽取 LLaMA/Mistral/Gemma 等（需要 GPU，模型较大） |
| tf-4 | Vision 类扩展 | ViT 变体、Swin 等（部分已在 timm 覆盖，注意去重） |
| tf-5 | Speech 类扩展 | Whisper、wav2vec2 更多变体 |
| tf-6 | 多模态类扩展 | CLIP 变体、BLIP、LLaVA 等 |
| tf-7 | Embedding 类扩展 | BGE、E5 等文本嵌入模型 |
| tf-8 | 处理失败模型 | 分析 `ConfigMetadataAnalyzer` 不支持的架构，按需提 PR 改进 Agent |

### 注意事项

- 大模型（>1B 参数）需要足够 GPU 显存，或用 CPU + 小 batch
- Agent 管线会自动下载模型权重，注意磁盘空间
- 命名规则：transformers 样本名用 `org_model` 格式（如 `meta-llama_Llama-2-7b`），斜杠 `/` 替换为 `_`

---

## T-ultralytics: YOLO 模型补全

> 已抽取 89 个，覆盖了 YOLO v3/v5/v6/v8/v9/v10/v11/v12，但部分尺寸和任务变体可能遗漏。

### 已覆盖的版本

YOLOv3(3), v5(8), v6(3), v8(3), v9(3), v10(5), v11(5), v12(5) × 任务变体(base/cls/seg/obb/pose)

### 子任务

| # | 子任务 | 说明 |
|---|--------|------|
| ultra-1 | 差距分析 | 列出 ultralytics 支持的所有模型 & 变体，对比已抽取列表 |
| ultra-2 | 补全缺失变体 | 如 yolov5s/yolov8s 等可能遗漏的尺寸 |
| ultra-3 | 编写抽取脚本 | ultralytics 模型的加载方式与 torchvision 不同，需用 `YOLO()` API |
| ultra-4 | validate + 提交 | |

### 脚本思路

```python
from ultralytics import YOLO
from graph_net.torch.extractor import extract
import torch

model = YOLO("yolov8n.pt").model.eval()
wrapped = extract(name="yolov8n")(model).eval()
with torch.no_grad():
    wrapped(torch.rand(1, 3, 640, 640))
```

---

## T-torchaudio: 音频模型扩展

> 已抽取 10 个：hubert(2), wav2vec2(3), wavlm(2), squim(2), convtasnet(1)

### 可扩展方向

| 模型 | 是否已抽取 | 输入格式 |
|------|----------|----------|
| hubert_base | ✅ | `(1, 80000)` 原始波形 |
| hubert_large | ✅ | 同上 |
| wav2vec2_base | ✅ | `(1, 16000)` |
| wav2vec2_large | ✅ | `(1, 80000)` |
| wavlm_base | ✅ | `(1, 64000)` |
| wavlm_large | ✅ | `(1, 64000)` |
| tacotron2 | ❌ | 文本→语音，输入为 token 序列 |
| wavernn | ❌ | 声码器 |
| emformer | ❌ | 流式 ASR |
| conformer | ❌ | ASR encoder |
| deepspeech | ❌ | ASR |

### 子任务

| # | 子任务 | 说明 |
|---|--------|------|
| audio-1 | 列出 torchaudio.pipelines 所有可用模型 | 对比已抽取列表 |
| audio-2 | 补全 ASR 类模型 | emformer, conformer 等，输入为 1D/2D 波形 |
| audio-3 | 尝试 TTS 类模型 | tacotron2 等，输入为 token 序列，格式特殊 |
| audio-4 | validate + 提交 | |

---

## T-torchgeometric: 图神经网络扩展

> 已抽取 14 个。PyG 模型的输入格式特殊：节点特征矩阵 + 边索引矩阵。

### 已覆盖模型

EdgeCNN, GAE, GAT, GCN, GIN, GraphSAGE, LINKX, LightGCN, MetaPath2Vec, Node2Vec, PMLP, PNA, RECT_L, SignedGCN

### 可扩展模型

| 模型 | 说明 | 输入格式 |
|------|------|----------|
| GATv2 | 注意力改进版 | `x: (N, F)`, `edge_index: (2, E)` |
| APPNP | 近似 PageRank 传播 | 同上 |
| SGC | 简化图卷积 | 同上 |
| ChebConv | 切比雪夫卷积 | 同上 |
| TransformerConv | 图 Transformer | 同上 |
| HeteroGNN | 异构图 | 多类型节点+边 |
| PointNet/PointNet++ | 3D 点云 | `pos: (N, 3)` |

### 子任务

| # | 子任务 | 说明 |
|---|--------|------|
| geo-1 | 列出 PyG 内置模型 | `torch_geometric.nn.models` 中的所有模型类 |
| geo-2 | 分析已有样本的输入模式 | 已有样本用 `(N, F)` + `(2, E)` 格式 |
| geo-3 | 补全常见 GNN | GATv2, APPNP, SGC 等标准消息传递模型 |
| geo-4 | 尝试特殊 GNN | 异构图、点云模型等 |
| geo-5 | validate + 提交 | |

---

## T-nemo: NVIDIA NeMo 扩展

> 已抽取 16 个，全部是 ASR (STT) 模型。NeMo 还有 TTS、NLP、多模态等。

### 已覆盖

parakeet-ctc(2), parakeet-tdt(2), conformer_ctc(3), fastconformer(3), squeezeformer(6)

### 可扩展方向

| 类别 | 示例模型 | 说明 |
|------|---------|------|
| TTS | FastPitch, HiFi-GAN, Tacotron2 | 文本转语音 |
| NLP | Megatron-GPT, NeMo-BERT | 大语言模型 |
| Speaker | TitaNet, ECAPA-TDNN | 说话人识别 |
| Multimodal | NeVA | 多模态 |

### 子任务

| # | 子任务 | 说明 |
|---|--------|------|
| nemo-1 | 列出 NeMo NGC 上可用的预训练模型 | 按类别分组 |
| nemo-2 | 补全 ASR 模型 | 其他语言的 STT 模型 |
| nemo-3 | 尝试 TTS 模型 | 输入格式与 ASR 不同 |
| nemo-4 | 尝试 Speaker 模型 | 说话人嵌入模型 |
| nemo-5 | validate + 提交 | |

> 注意：NeMo 模型依赖 `nemo_toolkit`，可能需要额外安装。

---

## T-mmseg / T-mmpose: OpenMMLab 补全

> mmseg 124 个、mmpose 82 个，覆盖率已较高。
> OpenMMLab 模型需要 mmcv/mmseg/mmpose 等库支持，安装配置较复杂。

### 子任务

| # | 子任务 | 说明 |
|---|--------|------|
| mm-1 | 安装 mmcv + mmseg + mmpose | 确认版本兼容 |
| mm-2 | 差距分析 | 列出 mmseg/mmpose config 中所有模型，对比已有样本 |
| mm-3 | 补全 mmseg 缺失模型 | 预估 20~30 个 |
| mm-4 | 补全 mmpose 缺失模型 | 预估 10~20 个 |
| mm-5 | validate + 提交 | |

---

## 建议执行顺序

```
T-timm (标准图像家族)     ← ROI 最高，大量模型可批量跑
  ↓
T-timm (ViT/Swin 家族)   ← 数量最多但可能有额外输入
  ↓
T-ultralytics             ← 小任务，快速补全
  ↓
T-torchaudio              ← 小任务，扩展音频覆盖
  ↓
T-torchgeometric          ← 输入格式特殊，需要单独调试
  ↓
T-transformers (小模型)    ← 用 Agent 管线批量跑
  ↓
T-transformers (大模型)    ← 需要 GPU 显存
  ↓
T-mmseg / T-mmpose        ← 依赖较多，最后处理
  ↓
T-nemo                    ← 依赖最重，优先级最低
```

### 并行建议

- **T-timm** 和 **T-ultralytics** 可并行（不同模型库，互不影响）
- **T-transformers** 和 **T-torchaudio** 可并行（Agent 管线 vs 直接 extract）
- **T-mmseg** 和 **T-mmpose** 可并行（同属 OpenMMLab，共享 mmcv 依赖）

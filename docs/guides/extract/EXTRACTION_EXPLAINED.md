# GraphNet 计算图抽取说明

## 1. 什么是"抽取"？

**抽取（Extract）= 从深度学习模型中提取计算图（Computational Graph）。**

具体过程：
1. 用 `graph_net.torch.extractor.extract` 装饰器包装一个 PyTorch 模型
2. 构造合适的输入数据，执行一次前向推理（`model.forward()`）
3. 装饰器在底层 hook 所有算子调用，自动记录：
   - 每个算子的类型（conv2d、matmul、relu、softmax…）
   - 算子之间的数据流向（谁的输出传给谁）
   - 张量的 shape、dtype 等元信息
   - 权重参数的结构
4. 生成一组结构化文件，保存到 `$GRAPH_NET_EXTRACT_WORKSPACE` 目录

> 简单理解：**把"活"的模型代码 → 变成一份静态的图结构数据**，就像给模型拍了一张"X光片"。

### 每个样本包含的文件

```
model_name/
├── model.py                      # 计算图的结构定义
├── input_meta.py                 # 输入张量的元信息（shape、dtype）
├── weight_meta.py                # 权重参数的元信息
├── input_tensor_constraints.py   # 输入约束条件
└── _decomposed/                  # 分解后的子图（可选）
    └── model_name_startX_endY_Z/
```

## 2. 为什么要做抽取？

GraphNet 项目的目标是构建一个**大规模计算图数据集**，用于：
- 研究深度学习模型的图结构特征
- 训练图神经网络来理解和优化模型结构
- 为编译器优化、模型压缩等下游任务提供数据基础

## 3. 抽取对象有哪些？

理论上，**任何能执行 `forward()` 的 PyTorch/Paddle 模型**都可以抽取。目前项目已覆盖 11 个来源：

| 来源 | 已抽取 | 说明 |
|------|--------|------|
| transformers-auto-model | 1360 | HuggingFace 模型（BERT、GPT、ViT、Whisper 等） |
| timm | 589 | PyTorch Image Models（EfficientNet、ConvNeXt、DeiT 等） |
| mmseg | 124 | OpenMMLab 语义分割 |
| ultralytics | 89 | YOLO 系列检测模型 |
| torchvision | 85 | PyTorch 官方视觉模型（ResNet、VGG、ViT 等） |
| mmpose | 82 | OpenMMLab 姿态估计 |
| nemo | 16 | NVIDIA NeMo 语音/NLP 模型 |
| torchgeometric | 14 | 图神经网络 |
| torchaudio | 10 | 音频模型 |
| cosyvoice | 2 | 语音合成 |
| others | 1 | 其他 |
| **合计** | **2372** | |

### 不同模型类型的输入格式

| 模型类型 | 输入构造方式 | 示例模型 |
|----------|-------------|----------|
| 图像分类/CV | `torch.rand(1, C, H, W)` | ResNet、EfficientNet、ViT |
| 视频模型 | `torch.rand(1, C, T, H, W)` | R3D、MViT、S3D |
| 目标检测 | `[torch.rand(3, H, W)]`（图像列表） | FasterRCNN、SSD、RetinaNet |
| 光流估计 | `(torch.rand(1,3,H,W), torch.rand(1,3,H,W))` | RAFT |
| NLP 文本 | `AutoTokenizer` 生成 `input_ids` 等 | BERT、GPT-2 |
| 语音 | `torch.rand(1, seq_len)` 原始波形 | wav2vec2、HuBERT |

## 4. 当前覆盖情况

已抽取 **2372** 个模型，但各生态系统的模型总量远大于此。

### torchvision 缺口分析（36 个未覆盖）

| 类别 | 总数 | 已完成 | 缺失 | 缺失模型 |
|------|------|--------|------|----------|
| 图像分类 | 68 | 68 | 0 | **全部完成** |
| 目标检测 | 12 | 0 | **12** | fasterrcnn_*, fcos_*, maskrcnn_*, retinanet_*, ssd* |
| 量化模型 | 12 | 0 | **12** | quantized_* |
| 视频模型 | 9 | 2 | **7** | mvit_v1_b, mvit_v2_s, r2plus1d_18, s3d, swin3d_* |
| 光流估计 | 2 | 0 | **2** | raft_large, raft_small |
| 语义分割 | 6 | 5 | **1** | lraspp_mobilenet_v3_large |
| Vision Transformer | 12 | 10 | **2** | swin_s, swin_t |

> **图像分类已 100% 覆盖**，但目标检测、量化模型、视频模型整个类别都是空白——这些是最有价值的贡献方向。

## 5. Validate 验证

抽取完成后，运行 validate 检查计算图是否合格：

```bash
python -m graph_net.torch.validate --model-path $GRAPH_NET_EXTRACT_WORKSPACE/model_name
```

验证通过后才可以提交 PR。

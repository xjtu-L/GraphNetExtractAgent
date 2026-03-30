# GraphNet 样本输入 Shape 分析报告

## 1. 总览

| 指标 | 数值 |
|------|------|
| 总样本数 | 2364 |
| 检测到纯数据输入的样本 | 2362 |
| 未检测到输入的样本 | 2 |
| 总输入张量数 | 6069 |

## 2. 输入张量维度分布

| 维度 (ndim) | 数量 | 典型用途 |
|-------------|------|----------|
| 1 | 17 | 1D 序列 / 掩码 |
| 2 | 4228 | 2D (batch, seq_len) — NLP 文本输入 |
| 3 | 570 | 3D (batch, channels, len) — 1D 卷积/音频 |
| 4 | 1233 | 4D (batch, C, H, W) — 图像 |
| 5 | 21 | 5D (batch, C, T, H, W) — 视频 |

## 3. 各来源输入 Shape 汇总

### cosyvoice (2/2 个模型有输入)

| Shape 模式 | 出现次数 | 示例模型 |
|-----------|----------|----------|
| `3D: [1, 1, 763]` | 1 | CosyVoice-300M |
| `3D: [1, 763, 512]` | 1 | CosyVoice-300M |
| `3D: [1, 1525, 512]` | 1 | CosyVoice-300M |
| `3D: [1, 1, 796]` | 1 | CosyVoice2-0.5B |
| `3D: [1, 796, 512]` | 1 | CosyVoice2-0.5B |
| `3D: [1, 1591, 512]` | 1 | CosyVoice2-0.5B |

### mmpose (82/82 个模型有输入)

| Shape 模式 | 出现次数 | 示例模型 |
|-----------|----------|----------|
| `4D: [1, 3, 256, 192]` | 60 | 2xmspn_50, 2xrsn_50, 3xmspn_50 ...等60个 |
| `4D: [1, 3, 256, 256]` | 21 | RTMO-l, RTMO-m, RTMO-s ...等21个 |
| `3D: [1, 34, 1]` | 1 | SimpleBaseline3D |

### mmseg (124/124 个模型有输入)

| Shape 模式 | 出现次数 | 示例模型 |
|-----------|----------|----------|
| `4D: [1, 3, 512, 512]` | 114 | BiSeNetV1_R18, BiSeNetV1_R50, BiSeNetV2 ...等114个 |
| `4D: [1, 3, 512, 1024]` | 7 | ANN_R101, ANN_R50, APCNet_R101 ...等7个 |
| `4D: [1, 3, 640, 640]` | 2 | BEiT-B, BEiT-L |
| `4D: [1, 256, 128, 128]` | 1 | Mask2Former_R50 |
| `4D: [1, 256, 16, 16]` | 1 | Mask2Former_R50 |
| `4D: [1, 256, 32, 32]` | 1 | Mask2Former_R50 |
| `4D: [1, 256, 64, 64]` | 1 | Mask2Former_R50 |

### nemo (16/16 个模型有输入)

| Shape 模式 | 出现次数 | 示例模型 |
|-----------|----------|----------|
| `2D: [8, 128]` | 264 | parakeet-ctc-0.6b, parakeet-ctc-0.6b, parakeet-ctc-0.6b ...等264个 |
| `2D: [8, 64]` | 140 | stt_en_conformer_ctc_large, stt_en_conformer_ctc_large, stt_en_conformer_ctc_large ...等140个 |
| `2D: [4, 64]` | 100 | stt_en_conformer_ctc_medium, stt_en_conformer_ctc_medium, stt_en_conformer_ctc_medium ...等100个 |
| `2D: [8, 80]` | 44 | stt_en_squeezeformer_ctc_large_ls, stt_en_squeezeformer_ctc_large_ls, stt_en_squeezeformer_ctc_large_ls ...等44个 |
| `2D: [4, 96]` | 40 | stt_en_squeezeformer_ctc_medium_ls, stt_en_squeezeformer_ctc_medium_ls, stt_en_squeezeformer_ctc_medium_ls ...等40个 |
| `2D: [4, 49]` | 36 | stt_en_squeezeformer_ctc_small_ls, stt_en_squeezeformer_ctc_small_ls, stt_en_squeezeformer_ctc_small_ls ...等36个 |
| `2D: [4, 44]` | 32 | stt_en_conformer_ctc_small, stt_en_conformer_ctc_small, stt_en_conformer_ctc_small ...等32个 |
| `2D: [4, 36]` | 32 | stt_en_squeezeformer_ctc_xsmall_ls, stt_en_squeezeformer_ctc_xsmall_ls, stt_en_squeezeformer_ctc_xsmall_ls ...等32个 |
| `1D: [1]` | 16 | parakeet-ctc-0.6b, parakeet-ctc-1.1b, parakeet-tdt-0.6b-v3 ...等16个 |
| `3D: [1, 80, 521]` | 15 | parakeet-ctc-0.6b, parakeet-ctc-1.1b, parakeet-tdt-1.1b ...等15个 |
| `3D: [1, 9999, 512]` | 5 | stt_en_conformer_ctc_large, stt_en_fastconformer_hybrid_large_pc, stt_en_fastconformer_hybrid_large_streaming_multi ...等5个 |
| `3D: [1, 9999, 1024]` | 4 | parakeet-ctc-0.6b, parakeet-ctc-1.1b, parakeet-tdt-0.6b-v3 ...等4个 |
| `3D: [1, 9999, 256]` | 4 | stt_en_conformer_ctc_medium, stt_en_fastconformer_hybrid_medium_streaming_80ms_pc, stt_en_squeezeformer_ctc_small_medium_ls ...等4个 |
| `3D: [1, 9999, 640]` | 2 | stt_en_squeezeformer_ctc_large_ls, stt_en_squeezeformer_ctc_large_ls |
| `3D: [1, 9999, 384]` | 2 | stt_en_squeezeformer_ctc_medium_ls, stt_en_squeezeformer_ctc_medium_ls |
| `3D: [1, 9999, 196]` | 2 | stt_en_squeezeformer_ctc_small_ls, stt_en_squeezeformer_ctc_small_ls |
| `3D: [1, 9999, 144]` | 2 | stt_en_squeezeformer_ctc_xsmall_ls, stt_en_squeezeformer_ctc_xsmall_ls |
| `3D: [1, 128, 521]` | 1 | parakeet-tdt-0.6b-v3 |
| `3D: [1, 9999, 176]` | 1 | stt_en_conformer_ctc_small |

### others (1/1 个模型有输入)

| Shape 模式 | 出现次数 | 示例模型 |
|-----------|----------|----------|
| `3D: [1, 65, 3]` | 1 | pigu_hybrid |
| `3D: [1, 65, 2]` | 1 | pigu_hybrid |
| `4D: [1, 128, 256, 2]` | 1 | pigu_hybrid |
| `4D: [1, 3, 128, 256]` | 1 | pigu_hybrid |

### timm (589/589 个模型有输入)

| Shape 模式 | 出现次数 | 示例模型 |
|-----------|----------|----------|
| `4D: [1, 3, 224, 224]` | 549 | aimv2_1b_patch14_224.apple_pt, aimv2_3b_patch14_224.apple_pt, aimv2_huge_patch14_224.apple_pt ...等549个 |
| `3D: [1, 1, 768]` | 29 | beit3_base_patch16_224.in22k_ft_in1k, beitv2_base_patch16_224, cait_m36_384 ...等29个 |
| `3D: [1, 1, 384]` | 21 | cait_s24_224.fb_dist_in1k, cait_s24_384, cait_s36_384 ...等21个 |
| `3D: [1, 1, 1024]` | 15 | beit3_large_patch16_224.in22k_ft_in1k, beitv2_large_patch16_224, deit3_large_patch16_224.fb_in1k ...等15个 |
| `3D: [1, 1, 192]` | 14 | cait_xxs24_224.fb_dist_in1k, cait_xxs24_384, cait_xxs36_224.fb_dist_in1k ...等14个 |
| `4D: [1, 3, 384, 384]` | 13 | cait_m36_384, cait_s24_384, cait_s36_384 ...等13个 |
| `4D: [1, 3, 256, 256]` | 12 | bat_resnext26ts.ch_in1k, botnet26t_256.c1_in1k, cs3darknet_focus_x ...等12个 |
| `3D: [1, 197, 768]` | 8 | beit3_base_patch16_224.in22k_ft_in1k, crossvit_base_240.in1k, deit_base_patch16_224.fb_in1k ...等8个 |
| `3D: [1, 1, 320]` | 8 | coat_lite_medium.in1k, coat_lite_medium_384, coat_lite_mini.in1k ...等8个 |
| `3D: [1, 1, 128]` | 7 | coat_lite_medium.in1k, coat_lite_medium_384, coat_lite_mini.in1k ...等7个 |
| `3D: [1, 1, 512]` | 7 | coat_lite_medium.in1k, coat_lite_medium_384, coat_lite_mini.in1k ...等7个 |
| `3D: [1, 1, 256]` | 6 | coat_lite_medium.in1k, coat_lite_medium_384, coat_lite_tiny.in1k ...等6个 |
| `3D: [1, 1, 152]` | 6 | coat_mini, coat_small, coat_tiny ...等6个 |
| `3D: [1, 196, 768]` | 6 | convit_base, deit3_base_patch16_224.fb_in1k, vit_base_patch16_rope_ape_224.naver_in1k ...等6个 |
| `3D: [1, 197, 384]` | 5 | crossvit_15_240.in1k, crossvit_15_dagger_240.in1k, crossvit_small_240.in1k ...等5个 |
| `4D: [1, 3, 448, 448]` | 4 | aimv2_1b_patch14_448.apple_pt, aimv2_3b_patch14_448.apple_pt, cait_m48_448 ...等4个 |
| `3D: [1, 196, 384]` | 4 | cait_s24_224.fb_dist_in1k, deit3_small_patch16_224.fb_in1k, vit_small_patch16_rope_ape_224.naver_in1k ...等4个 |
| `3D: [1, 1, 1280]` | 4 | deit3_huge_patch14_224.fb_in1k, vit_huge_patch14_224.mae, vit_huge_patch14_clip_224.dfn5b ...等4个 |
| `3D: [1, 257, 1024]` | 4 | eva02_large_patch14_224.mim_in22k, eva02_large_patch14_clip_224.merged2b, vit_large_patch14_clip_224.datacompxl ...等4个 |
| `3D: [1, 50, 768]` | 4 | vit_base_patch32_224.augreg_in1k, vit_base_patch32_224.orig_in21k, vit_base_patch32_clip_224.datacompxl ...等4个 |
| `3D: [1, 197, 1024]` | 3 | beit3_large_patch16_224.in22k_ft_in1k, vit_large_patch16_224.augreg_in21k, vit_large_patch16_224.mae |
| `3D: [1, 576, 384]` | 3 | cait_s24_384, cait_s36_384, deit3_small_patch16_384 |
| `3D: [1, 196, 192]` | 3 | cait_xxs24_224.fb_dist_in1k, cait_xxs36_224.fb_dist_in1k, convit_tiny |
| `3D: [1, 1, 64]` | 3 | coat_lite_mini.in1k, coat_lite_small.in1k, coat_lite_tiny.in1k |
| `3D: [1, 1, 216]` | 3 | coat_mini, coat_mini, coat_mini |
| `3D: [1, 401, 192]` | 3 | crossvit_15_240.in1k, crossvit_15_dagger_240.in1k, crossvit_small_240.in1k |
| `3D: [1, 1, 224]` | 3 | crossvit_18_240.in1k, crossvit_18_dagger_240.in1k, crossvit_18_dagger_408.in1k |
| `3D: [1, 1, 448]` | 3 | crossvit_18_240.in1k, crossvit_18_dagger_240.in1k, crossvit_18_dagger_408.in1k |
| `3D: [1, 197, 256]` | 3 | crossvit_9_240.in1k, crossvit_9_dagger_240.in1k, vit_xsmall_patch16_clip_224.tinyclip_yfcc15m |
| `3D: [1, 196, 1024]` | 3 | deit3_large_patch16_224.fb_in1k, vit_large_patch16_rope_ape_224.naver_in1k, vit_large_patch16_rope_mixed_ape_224.naver_in1k |
| `3D: [1, 257, 1408]` | 3 | eva_giant_patch14_224.clip_ft_in1k, eva_giant_patch14_clip_224.laion400m, vit_giant_patch14_clip_224.laion2b |
| `3D: [1, 1, 1408]` | 3 | eva_giant_patch14_224.clip_ft_in1k, eva_giant_patch14_clip_224.laion400m, vit_giant_patch14_clip_224.laion2b |
| `3D: [1, 257, 1280]` | 3 | vit_huge_patch14_224.mae, vit_huge_patch14_clip_224.dfn5b, vit_huge_patch14_clip_quickgelu_224.dfn5b |
| `4D: [1, 3, 336, 336]` | 2 | aimv2_1b_patch14_336.apple_pt, aimv2_3b_patch14_336.apple_pt |
| `3D: [1, 576, 768]` | 2 | cait_m36_384, deit3_base_patch16_384 |
| `3D: [1, 576, 192]` | 2 | cait_xxs24_384, cait_xxs36_384 |
| `3D: [1, 401, 224]` | 2 | crossvit_18_240.in1k, crossvit_18_dagger_240.in1k |
| `3D: [1, 197, 448]` | 2 | crossvit_18_240.in1k, crossvit_18_dagger_240.in1k |
| `3D: [1, 401, 128]` | 2 | crossvit_9_240.in1k, crossvit_9_dagger_240.in1k |
| `3D: [1, 197, 192]` | 2 | crossvit_tiny_240.in1k, deit_tiny_patch16_224.fb_in1k |
| `4D: [1, 3, 768, 768]` | 2 | davit_base_fl, davit_huge_fl |
| `3D: [1, 256, 1280]` | 2 | deit3_huge_patch14_224.fb_in1k, vit_huge_patch14_gap_224.in1k_ijepa |
| `4D: [1, 3, 320, 320]` | 2 | dm_nfnet_f3, eca_nfnet_l2 |
| `3D: [1, 785, 768]` | 2 | vit_base_patch8_224.augreg2_in21k_ft_in1k, vit_base_patch8_224.dino |
| `3D: [1, 257, 1664]` | 2 | vit_gigantic_patch14_clip_224.laion2b, vit_gigantic_patch14_clip_quickgelu_224.metaclip_2pt5b |
| `3D: [1, 1, 1664]` | 2 | vit_gigantic_patch14_clip_224.laion2b, vit_gigantic_patch14_clip_quickgelu_224.metaclip_2pt5b |
| `3D: [1, 256, 2048]` | 1 | aimv2_1b_patch14_224.apple_pt |
| `3D: [1, 576, 2048]` | 1 | aimv2_1b_patch14_336.apple_pt |
| `3D: [1, 1024, 2048]` | 1 | aimv2_1b_patch14_448.apple_pt |
| `3D: [1, 256, 3072]` | 1 | aimv2_3b_patch14_224.apple_pt |
| `3D: [1, 576, 3072]` | 1 | aimv2_3b_patch14_336.apple_pt |
| `3D: [1, 1024, 3072]` | 1 | aimv2_3b_patch14_448.apple_pt |
| `3D: [1, 256, 1536]` | 1 | aimv2_huge_patch14_224.apple_pt |
| `3D: [1, 256, 1024]` | 1 | aimv2_large_patch14_224.apple_pt |
| `3D: [1, 784, 768]` | 1 | cait_m48_448 |
| `3D: [1, 576, 288]` | 1 | cait_xs24_384 |
| `3D: [1, 1, 288]` | 1 | cait_xs24_384 |
| `3D: [1, 196, 432]` | 1 | convit_small |
| `3D: [1, 1, 432]` | 1 | convit_small |
| `3D: [1, 1157, 192]` | 1 | crossvit_15_dagger_408.in1k |
| `3D: [1, 577, 384]` | 1 | crossvit_15_dagger_408.in1k |
| `3D: [1, 1157, 224]` | 1 | crossvit_18_dagger_408.in1k |
| `3D: [1, 577, 448]` | 1 | crossvit_18_dagger_408.in1k |
| `3D: [1, 401, 384]` | 1 | crossvit_base_240.in1k |
| `3D: [1, 1, 96]` | 1 | crossvit_tiny_240.in1k |
| `3D: [1, 401, 96]` | 1 | crossvit_tiny_240.in1k |
| `3D: [1, 576, 1024]` | 1 | deit3_large_patch16_384 |
| `3D: [1, 196, 512]` | 1 | deit3_medium_patch16_224.fb_in1k |
| `3D: [1, 198, 768]` | 1 | deit_base_distilled_patch16_224.fb_in1k |
| `3D: [1, 578, 768]` | 1 | deit_base_distilled_patch16_384 |
| `3D: [1, 198, 384]` | 1 | deit_small_distilled_patch16_224.fb_in1k |
| `3D: [1, 198, 192]` | 1 | deit_tiny_distilled_patch16_224.fb_in1k |
| `4D: [1, 3, 192, 192]` | 1 | dm_nfnet_f0 |
| `4D: [1, 3, 416, 416]` | 1 | dm_nfnet_f5 |
| `4D: [1, 3, 352, 352]` | 1 | eca_nfnet_l3 |
| `3D: [1, 257, 768]` | 1 | eva02_base_patch14_224.mim_in22k |
| `3D: [1, 257, 1792]` | 1 | eva02_enormous_patch14_clip_224.laion2b |
| `3D: [1, 1, 1792]` | 1 | eva02_enormous_patch14_clip_224.laion2b |
| `3D: [1, 257, 384]` | 1 | eva02_small_patch14_224.mim_in22k |
| `3D: [1, 257, 192]` | 1 | eva02_tiny_patch14_224.mim_in22k |
| `4D: [1, 3, 299, 299]` | 1 | inception_v4 |
| `4D: [1, 64, 64, 768]` | 1 | samvit_base_patch16.sa1b |
| `4D: [1, 64, 64, 1280]` | 1 | samvit_huge_patch16.sa1b |
| `4D: [1, 64, 64, 1024]` | 1 | samvit_large_patch16.sa1b |
| `4D: [1, 16, 16, 384]` | 1 | sequencer2d_l.in1k |
| `3D: [1, 50, 640]` | 1 | vit_betwixt_patch32_clip_224.tinyclip_laion400m |
| `3D: [1, 1, 640]` | 1 | vit_betwixt_patch32_clip_224.tinyclip_laion400m |
| `3D: [1, 196, 1408]` | 1 | vit_giant_patch16_gap_224.in22k_ijepa |
| `3D: [1, 50, 1024]` | 1 | vit_large_patch32_224.orig_in21k |
| `3D: [1, 197, 512]` | 1 | vit_medium_patch16_clip_224.tinyclip_yfcc15m |
| `3D: [1, 50, 512]` | 1 | vit_medium_patch32_clip_224.tinyclip_laion400m |
| `3D: [1, 50, 384]` | 1 | vit_small_patch32_224.augreg_in21k |
| `3D: [1, 785, 384]` | 1 | vit_small_patch8_224.dino |
| `3D: [1, 256, 1152]` | 1 | vit_so400m_patch14_siglip_gap_224.pali2_3b_pt |

### torchaudio (10/10 个模型有输入)

| Shape 模式 | 出现次数 | 示例模型 |
|-----------|----------|----------|
| `2D: [1, 80000]` | 6 | hubert_base, hubert_large, squim_subjective ...等6个 |
| `2D: [1, 64000]` | 2 | wavlm_base, wavlm_large |
| `3D: [1, 1, 32000]` | 1 | convtasnet_base_libri2mix |
| `3D: [1, 2499, 256]` | 1 | squim_objective |
| `2D: [1, 16000]` | 1 | wav2vec2_base |

### torchgeometric (12/14 个模型有输入)

| Shape 模式 | 出现次数 | 示例模型 |
|-----------|----------|----------|
| `2D: [2, 100]` | 10 | EdgeCNN, GAE, GAT ...等10个 |
| `2D: [1000, 128]` | 8 | EdgeCNN, GAT, GCN ...等8个 |
| `2D: [1000, 32]` | 2 | GAE, SignedGCN |
| `2D: [2, 500]` | 1 | LightGCN |
| `2D: [2, 1000]` | 1 | PMLP |
| `2D: [128, 128]` | 1 | RECT_L |
| `2D: [2, 128]` | 1 | RECT_L |

### torchvision (85/85 个模型有输入)

| Shape 模式 | 出现次数 | 示例模型 |
|-----------|----------|----------|
| `4D: [1, 3, 224, 224]` | 83 | alexnet, convnext_base, convnext_large ...等83个 |
| `5D: [1, 3, 16, 224, 224]` | 2 | mc3_18, r3d_18 |
| `3D: [1, 1, 768]` | 2 | vit_b_16, vit_b_32 |
| `3D: [1, 1, 1024]` | 2 | vit_l_16, vit_l_32 |
| `3D: [1, 1, 1280]` | 1 | vit_h_14 |

### transformers-auto-model (1351/1351 个模型有输入)

| Shape 模式 | 出现次数 | 示例模型 |
|-----------|----------|----------|
| `2D: [1, 1]` | 613 | google_pix2struct-base, microsoft_trocr-base-printed, microsoft_trocr-base-stage1 ...等613个 |
| `2D: [1, 20]` | 224 | 42dot_LLM-SFT-1.3B, 42dot_LLM-SFT-1.3B, AI-Nordics_bert-large-swedish-cased ...等224个 |
| `2D: [1, 21]` | 189 | ALINEAR_albert-japanese, ALINEAR_albert-japanese, ALINEAR_albert-japanese ...等189个 |
| `2D: [1, 11]` | 140 | FacebookAI_roberta-large, FacebookAI_roberta-large, InstaDeepAI_nucleotide-transformer-500m-human-ref ...等140个 |
| `4D: [1, 3, 224, 224]` | 124 | AkshatSurolia_BEiT-FaceMask-Finetuned, AkshatSurolia_ConvNeXt-FaceMask-Finetuned, AkshatSurolia_DeiT-FaceMask-Finetuned ...等124个 |
| `2D: [1, 36]` | 86 | kanarya-750m, kanarya-750m, opus-mt-NORTH_EU-NORTH_EU ...等86个 |
| `2D: [1, 40]` | 84 | opus-mt-bg-en, opus-mt-bg-en, opus-mt-bg-sv ...等84个 |
| `2D: [1, 41]` | 82 | opus-mt-afa-en, opus-mt-afa-en, opus-mt-bem-sv ...等82个 |
| `2D: [1, 39]` | 70 | japanese-gpt-1b, japanese-gpt-1b, opus-mt-aed-es ...等70个 |
| `2D: [1, 37]` | 70 | opus-mt-ca-pt, opus-mt-ca-pt, opus-mt-chk-sv ...等70个 |
| `2D: [1, 22]` | 66 | Aleksandar_electra-srb-ner-setimes, Aleksandar_electra-srb-ner-setimes, Aleksandar_electra-srb-ner-setimes ...等66个 |
| `2D: [8, 128]` | 64 | canary-1b-v2, canary-1b-v2, canary-1b-v2 ...等64个 |
| `2D: [1, 12]` | 61 | DAMO-NLP-SG_zero-shot-classify-SSTuning-ALBERT, DAMO-NLP-SG_zero-shot-classify-SSTuning-ALBERT, DAMO-NLP-SG_zero-shot-classify-SSTuning-ALBERT ...等61个 |
| `2D: [1, 19]` | 57 | AK_ak_nlp, AK_ak_nlp, AceInstruct-1.5B ...等57个 |
| `2D: [1, 17]` | 57 | Andrija_SRoBERTa-NER, Andrija_SRoBERTa-NER, DiegoRossini_flaubert-pouvoir-modality-detector ...等57个 |
| `2D: [1, 18]` | 54 | AVSilva_bertimbau-large-fine-tuned-md, AVSilva_bertimbau-large-fine-tuned-md, AVSilva_bertimbau-large-fine-tuned-md ...等54个 |
| `2D: [1, 35]` | 54 | opus-mt-bcl-fi, opus-mt-bcl-fi, opus-mt-bcl-fr ...等54个 |
| `2D: [1, 10]` | 51 | DarshanDeshpande_marathi-distilbert, DarshanDeshpande_marathi-distilbert, KoichiYasuoka_deberta-base-chinese-ud-goeswith ...等51个 |
| `2D: [1, 13]` | 50 | DatologyAI_cls-opt-vit-b-32, MoritzLaurer_mDeBERTa-v3-base-mnli-xnli, MoritzLaurer_mDeBERTa-v3-base-mnli-xnli ...等50个 |
| `2D: [1, 38]` | 50 | opus-mt-bat-en, opus-mt-bat-en, opus-mt-ca-es ...等50个 |
| `2D: [1, 45]` | 48 | KoichiYasuoka_bert-large-japanese-upos, KoichiYasuoka_bert-large-japanese-upos, KoichiYasuoka_bert-large-japanese-upos ...等48个 |
| `2D: [1, 32]` | 44 | LLaMmlein_120M, LLaMmlein_120M, joancipria_gpt2-base-bne-FineTunedEmoEvent ...等44个 |
| `2D: [2, 7]` | 41 | BAAI_AltCLIP, BAAI_AltCLIP, BAAI_AltCLIP-m18 ...等41个 |
| `2D: [1, 16]` | 38 | DeepChem_ChemBERTa-10M-MLM, DeepChem_ChemBERTa-10M-MLM, Dr-BERT_CAS-Biomedical-POS-Tagging ...等38个 |
| `2D: [1, 33]` | 38 | erica_krm_sa2, erica_krm_sa2, erica_krm_sa2 ...等38个 |
| `2D: [1, 42]` | 36 | opus-mt-bem-fi, opus-mt-bem-fi, opus-mt-bem-fr ...等36个 |
| `2D: [1, 34]` | 34 | opus-mt-af-es, opus-mt-af-es, opus-mt-af-nl ...等34个 |
| `2D: [1, 44]` | 32 | opus-mt-afa-afa, opus-mt-afa-afa, opus-mt-ar-pl ...等32个 |
| `2D: [1, 15]` | 28 | DTAI-KULeuven_robbertje-1-gb-bort, DTAI-KULeuven_robbertje-1-gb-bort, DTAI-KULeuven_robbertje-1-gb-merged ...等28个 |
| `2D: [2, 34]` | 27 | monoelectra-large, monoelectra-large, monoelectra-large ...等27个 |
| `2D: [1, 27]` | 26 | Film8844_wangchanberta-ner, Film8844_wangchanberta-ner, GePpeTto ...等26个 |
| `2D: [1, 46]` | 26 | opus-mt-bg-es, opus-mt-bg-es, opus-mt-bg-uk ...等26个 |
| `2D: [1, 80000]` | 24 | 907508196l_wavlm-libri-clean-100h-base-plus-finetuned-ks, Aniemore_unispeech-sat-emotion-russian-resd, Ar4ikov_W2VFSC2 ...等24个 |
| `4D: [1, 3, 384, 384]` | 24 | Intel_dpt-large, dayyass_trocr-base-handwritten-vit-encoder, facebook_convnext-base-384-22k-1k ...等24个 |
| `2D: [1, 43]` | 22 | KoichiYasuoka_deberta-base-ainu-ud-goeswith, KoichiYasuoka_deberta-base-ainu-ud-goeswith, hf-tiny-model-private_tiny-random-OpenAIGPTForSequenceClassification ...等22个 |
| `4D: [1, 3, 800, 800]` | 21 | IDEA-Research_dab-detr-resnet-50, SenseTime_deformable-detr-single-scale, SenseTime_deformable-detr-single-scale-dc5 ...等21个 |
| `2D: [1, 7]` | 20 | IDEA-Research_grounding-dino-base, Thomasboosinger_owlvit-base-patch32, Thomasboosinger_owlvit-base-patch32 ...等20个 |
| `2D: [1, 26]` | 19 | CAMeL-Lab_text-editing-qalb14-nopnx, CAMeL-Lab_text-editing-qalb14-nopnx, CAMeL-Lab_text-editing-qalb14-nopnx ...等19个 |
| `2D: [1, 29]` | 18 | Mizuiro-sakura_luke-japanese-large-finetuned-ner, Mizuiro-sakura_luke-japanese-large-finetuned-ner, opus-mt-aav-en ...等18个 |
| `3D: [1, 11, 1024]` | 18 | MohamedZaitoon_bart-fine-tune, MohamedZaitoon_bart-fine-tune, PeterBanning71_bart-large-finetuned-eLife ...等18个 |
| `2D: [1, 31]` | 18 | kogpt, kogpt, opus-mt-ROMANCE-en ...等18个 |
| `2D: [1, 23]` | 17 | Geotrend_distilbert-base-ar-cased, Geotrend_distilbert-base-ar-cased, Hezam_ArabicT5_Classification ...等17个 |
| `3D: [1, 800, 800]` | 17 | IDEA-Research_dab-detr-resnet-50, SenseTime_deformable-detr-single-scale, SenseTime_deformable-detr-single-scale-dc5 ...等17个 |
| `2D: [1, 47]` | 16 | opus-mt-bg-ru, opus-mt-bg-ru, opus-mt-cs-eo ...等16个 |
| `2D: [1, 51]` | 15 | hf-tiny-model-private_tiny-random-CanineForSequenceClassification, hf-tiny-model-private_tiny-random-CanineForSequenceClassification, hf-tiny-model-private_tiny-random-CanineForSequenceClassification ...等15个 |
| `2D: [1, 49]` | 14 | opus-mt-af-eo, opus-mt-af-eo, opus-mt-bg-tr ...等14个 |
| `2D: [1, 24]` | 13 | ARTeLab_it5-summarization-fanpage-64, AethiQs-Max_AethiQs_GemBERT_bertje_50k, AethiQs-Max_AethiQs_GemBERT_bertje_50k ...等13个 |
| `4D: [1, 3, 30, 30]` | 13 | aurelio-ai_sr-test-clip, hf-tiny-model-private_tiny-random-AltCLIPModel, hf-tiny-model-private_tiny-random-BeitModel ...等13个 |
| `4D: [1, 3, 512, 512]` | 13 | face-parsing, hustvl_yolos-tiny, microsoft_beit-large-patch16-512 ...等13个 |
| `2D: [1, 14]` | 12 | Culmenus_XLMR-ENIS-finetuned-ner, Culmenus_XLMR-ENIS-finetuned-ner, EMBEDDIA_finest-bert ...等12个 |
| `3D: [1, 1025, 1024]` | 12 | Intel_dpt-beit-large-512, Intel_dpt-beit-large-512, Intel_dpt-beit-large-512 ...等12个 |
| `3D: [1, 1370, 384]` | 12 | LiheYoung_depth-anything-small-hf, LiheYoung_depth-anything-small-hf, LiheYoung_depth-anything-small-hf ...等12个 |
| `2D: [1, 48]` | 11 | Neurora_opus-tatoeba-hye-eng, opus-mt-eo-he, opus-mt-eo-he ...等11个 |
| `4D: [1, 3, 32, 32]` | 10 | hf-tiny-model-private_tiny-random-DPTModel, hf-tiny-model-private_tiny-random-MobileNetV1Model, hf-tiny-model-private_tiny-random-MobileNetV2Model ...等10个 |
| `2D: [1, 8]` | 10 | microsoft_xclip-base-patch16, microsoft_xclip-base-patch16, microsoft_xclip-base-patch16-zero-shot ...等10个 |
| `3D: [1, 10, 1024]` | 9 | UAE-Large-V1, cmarkea_bloomz-560m-nli, google_pegasus-cnn_dailymail ...等9个 |
| `3D: [1, 11, 768]` | 9 | cross-encoder_nli-deberta-base, echarlaix_bart-base-cnn-r2-18.7-d23-hybrid, echarlaix_bart-base-cnn-r2-18.7-d23-hybrid ...等9个 |
| `2D: [1, 50]` | 9 | google_byt5-small, opus-mt-fi-bzs, opus-mt-fi-bzs ...等9个 |
| `3D: [1, 12, 512]` | 8 | Fafadalilian_lora-adapter-t5_small_model_California_state_bill, Fafadalilian_lora-adapter-t5_small_model_California_state_bill, Helsinki-NLP_opus-mt-en-zh ...等8个 |
| `3D: [1, 12, 768]` | 8 | IMISLab_GreekWiki-umt5-base, IMISLab_GreekWiki-umt5-base, Intel_fid_flan_t5_base_nq ...等8个 |
| `3D: [1, 1370, 1024]` | 8 | LiheYoung_depth-anything-large-hf, LiheYoung_depth-anything-large-hf, LiheYoung_depth-anything-large-hf ...等8个 |
| `2D: [1, 30]` | 8 | dansk-gpt-wiki, dansk-gpt-wiki, opus-mt-af-en ...等8个 |
| `3D: [1, 1370, 768]` | 8 | depth-anything_Depth-Anything-V2-Base-hf, depth-anything_Depth-Anything-V2-Base-hf, depth-anything_Depth-Anything-V2-Base-hf ...等8个 |
| `2D: [1, 25]` | 7 | Aleksandar_bert-srb-ner-setimes, Aleksandar_bert-srb-ner-setimes, Aleksandar_bert-srb-ner-setimes ...等7个 |
| `3D: [1, 16, 1024]` | 7 | Farid59_nllb-darija-fr_eng, Farid59_nllb-darija-fr_eng, flutter-painter_nllb-fra-fuf-v2 ...等7个 |
| `2D: [1, 9]` | 7 | KoboldAI_fairseq-dense-1.3B, KoboldAI_fairseq-dense-125M, KoboldAI_fairseq-dense-355M ...等7个 |
| `2D: [1, 4]` | 7 | distilbert-base-uncased, distilbert-base-uncased, pix2struct-base ...等7个 |
| `3D: [1, 15, 1024]` | 6 | ELiRF_mbart-large-cc25-dacsa-ca, ELiRF_mbart-large-cc25-dacsa-ca, amurienne_kellag-m2m100-v0.2 ...等6个 |
| `5D: [1, 16, 3, 224, 224]` | 6 | MCG-NJU_videomae-base-finetuned-kinetics, MCG-NJU_videomae-base-short, MCG-NJU_videomae-large ...等6个 |
| `4D: [1, 3, 1024, 1024]` | 6 | facebook_sam-vit-base, facebook_sam-vit-huge, facebook_sam-vit-large ...等6个 |
| `3D: [1, 22, 16]` | 6 | hf-tiny-model-private_tiny-random-BartForSequenceClassification, hf-tiny-model-private_tiny-random-BartForSequenceClassification, hf-tiny-model-private_tiny-random-LEDForSequenceClassification ...等6个 |
| `4D: [1, 3, 64, 64]` | 6 | hf-tiny-model-private_tiny-random-CvtForImageClassification, hf-tiny-model-private_tiny-random-LevitModel, hf-tiny-model-private_tiny-random-PoolFormerForImageClassification ...等6个 |
| `2D: [2, 36]` | 6 | mmarco-mMiniLMv2-L12-H384-v1, mmarco-mMiniLMv2-L12-H384-v1, msmarco-MiniLM-L12-en-de-v1 ...等6个 |
| `2D: [2, 35]` | 6 | nli-MiniLM2-L6-H768, nli-MiniLM2-L6-H768, nli-roberta-base ...等6个 |
| `2D: [2, 33]` | 6 | nli-deberta-v3-base, nli-deberta-v3-base, nli-deberta-v3-small ...等6个 |
| `2D: [1, 53]` | 6 | opus-mt-am-sv, opus-mt-am-sv, opus-mt-ar-it ...等6个 |
| `2D: [1, 52]` | 6 | opus-mt-ar-he, opus-mt-ar-he, opus-mt-ar-tr ...等6个 |
| `2D: [1, 512]` | 4 | DiegoRossini_flaubert-pouvoir-modality-detector, all-MiniLM-L6-v2, hf-tiny-model-private_tiny-random-FlaubertForSequenceClassification ...等4个 |
| `3D: [1, 26, 512]` | 4 | Helsinki-NLP_opus-mt-zh-en, Helsinki-NLP_opus-mt-zh-en, Neurora_opus-tatoeba-zho-eng ...等4个 |
| `3D: [1, 23, 512]` | 4 | Hezam_ArabicT5_Classification, Hezam_ArabicT5_Classification, Neurora_opus-tatoeba-tur-eng ...等4个 |
| `3D: [1, 900, 256]` | 4 | IDEA-Research_grounding-dino-tiny, IDEA-Research_grounding-dino-tiny, sheshkar_cmon2 ...等4个 |
| `3D: [1, 577, 768]` | 4 | Intel_dpt-beit-base-384, Intel_dpt-beit-base-384, Intel_dpt-beit-base-384 ...等4个 |
| `3D: [1, 577, 1024]` | 4 | Intel_dpt-beit-large-384, Intel_dpt-beit-large-384, Intel_dpt-beit-large-384 ...等4个 |
| `4D: [1, 3, 336, 336]` | 4 | Lin-Chen_ShareGPT4V-7B_Pretrained_vit-large336-l12, Q-MM_clip-vit-large-patch14-336, apple_aimv2-huge-patch14-336 ...等4个 |
| `3D: [1, 18, 768]` | 4 | NYTK_summarization-hi-bart-base-1024-hungarian, NYTK_summarization-hi-bart-base-1024-hungarian, cahya_t5-base-indonesian-summarization-cased ...等4个 |
| `3D: [1, 17, 512]` | 4 | NeNeDataScientist_NeNeAI007_fr-kiluba-translation, NeNeDataScientist_NeNeAI007_fr-kiluba-translation, Neurora_opus-tatoeba-ron-eng ...等4个 |
| `3D: [1, 25, 512]` | 4 | Neurora_opus-tatoeba-heb-eng, Neurora_opus-tatoeba-heb-eng, igorktech_rut5-small-chit-chat-intelligent ...等4个 |
| `4D: [1, 3, 640, 640]` | 4 | RMBG-1.4, RaphaelKalandadze_tmp_trainer, detr-resnet-101-panoptic ...等4个 |
| `4D: [1, 3, 448, 448]` | 4 | apple_aimv2-huge-patch14-448, apple_aimv2-large-patch14-448, hf-tiny-model-private_tiny-random-BitModel ...等4个 |
| `4D: [1, 3, 256, 256]` | 4 | apple_mobilevit-small, apple_mobilevit-x-small, apple_mobilevit-xx-small ...等4个 |
| `2D: [2, 10]` | 4 | aurelio-ai_sr-test-clip, aurelio-ai_sr-test-clip, hf-tiny-model-private_tiny-random-CLIPSegModel ...等4个 |
| `3D: [1, 2917, 1024]` | 4 | depth-anything_prompt-depth-anything-vitl-hf, depth-anything_prompt-depth-anything-vitl-hf, depth-anything_prompt-depth-anything-vitl-hf ...等4个 |
| `3D: [1, 2917, 384]` | 4 | depth-anything_prompt-depth-anything-vits-hf, depth-anything_prompt-depth-anything-vits-hf, depth-anything_prompt-depth-anything-vits-hf ...等4个 |
| `3D: [1, 785, 1536]` | 4 | facebook_dpt-dinov2-giant-nyu, facebook_dpt-dinov2-giant-nyu, facebook_dpt-dinov2-giant-nyu ...等4个 |
| `3D: [1, 785, 384]` | 4 | facebook_dpt-dinov2-small-kitti, facebook_dpt-dinov2-small-kitti, facebook_dpt-dinov2-small-kitti ...等4个 |
| `5D: [1, 8, 3, 224, 224]` | 4 | facebook_timesformer-base-finetuned-k400, microsoft_xclip-base-patch16, microsoft_xclip-base-patch32 ...等4个 |
| `3D: [1, 12, 1024]` | 4 | google-t5_t5-large, google-t5_t5-large, larryvrh_mt5-translation-ja_zh ...等4个 |
| `3D: [1, 16, 768]` | 4 | google_flan_t5_base, ibm-research_MoLFormer-XL-both-10pct, moussaKam_barthez-orangesum-abstract ...等4个 |
| `2D: [768, 512]` | 4 | microsoft_xclip-base-patch16, microsoft_xclip-base-patch16-zero-shot, microsoft_xclip-base-patch32 ...等4个 |
| `2D: [1, 55]` | 4 | opus-mt-az-es, opus-mt-az-es, opus-mt-el-eo ...等4个 |
| `2D: [1, 59836]` | 4 | opus-mt-de-eo, opus-mt-el-sv, opus-mt-eo-de ...等4个 |
| `2D: [1, 28]` | 4 | opus-mt-en-jap, opus-mt-en-jap, opus-mt-tc-bible-big-cel-deu_eng_fra_por_spa ...等4个 |
| `3D: [1, 9, 1024]` | 3 | KoboldAI_fairseq-dense-355M, andyluan_emotion-detection, venishpatidar_wa-ad-mod |
| `3D: [1, 1024, 128]` | 3 | MIT_ast-finetuned-audioset-10-10-0.4593, MIT_ast-finetuned-audioset-14-14-0.443, bookbot_distil-ast-audioset |
| `3D: [1, 16, 512]` | 3 | SpaceCowboy03_finetuned-kde4-es-to-fr, SpaceCowboy03_finetuned-kde4-es-to-fr, google_flan_t5_small |
| `4D: [1, 3, 768, 768]` | 3 | Thomasboosinger_owlvit-base-patch32, google_owlvit-base-patch16, google_owlvit-base-patch32 |
| `2D: [1, 82]` | 3 | mesolitica_ner-t5-small-standard-bahasa-cased, mesolitica_ner-t5-small-standard-bahasa-cased, mesolitica_ner-t5-small-standard-bahasa-cased |
| `5D: [1, 32, 3, 224, 224]` | 3 | microsoft_xclip-base-patch16-zero-shot, prathameshdalal_vivit-b-16x2-kinetics400-UCF-Crime, thegigasurgeon_mopping_224_32_frames_resampling_1_huge |
| `2D: [1, 65001]` | 3 | opus-mt-ROMANCE-en, opus-mt-ar-pl, opus-mt-en-ROMANCE |
| `3D: [1, 24, 768]` | 2 | ARTeLab_it5-summarization-fanpage-64, ARTeLab_it5-summarization-fanpage-64 |
| `3D: [1, 19, 512]` | 2 | CLAck_en-vi, CLAck_en-vi |
| `3D: [1, 17, 1024]` | 2 | ELiRF_NASCA, ELiRF_NASCA |
| `3D: [1, 18, 512]` | 2 | Helsinki-NLP_opus-mt-NORTH_EU-NORTH_EU, Helsinki-NLP_opus-mt-NORTH_EU-NORTH_EU |
| `3D: [1, 21, 512]` | 2 | Helsinki-NLP_opus-mt-aed-es, Helsinki-NLP_opus-mt-aed-es |
| `3D: [1, 20, 512]` | 2 | Helsinki-NLP_opus-mt-af-es, Helsinki-NLP_opus-mt-af-es |
| `3D: [1, 7, 256]` | 2 | IDEA-Research_grounding-dino-base, fushh7_llmdet_swin_tiny_hf |
| `4D: [1, 6, 900, 256]` | 2 | IDEA-Research_grounding-dino-base, fushh7_llmdet_swin_tiny_hf |
| `3D: [1, 900, 4]` | 2 | IDEA-Research_grounding-dino-base, fushh7_llmdet_swin_tiny_hf |
| `4D: [1, 6, 900, 4]` | 2 | IDEA-Research_grounding-dino-base, fushh7_llmdet_swin_tiny_hf |
| `5D: [1, 10, 3, 224, 224]` | 2 | JackWong0911_timesformer-base-finetuned-k400-finetuned-kinetic400-subset-epoch6-num_frame_10, JackWong0911_vivit-b-16x2-kinetics400-finetuned-kinectic |
| `3D: [1, 48, 512]` | 2 | Neurora_opus-tatoeba-hye-eng, Neurora_opus-tatoeba-hye-eng |
| `3D: [1, 27, 512]` | 2 | Nextcloud-AI_opus-mt-ar-es, Nextcloud-AI_opus-mt-ar-es |
| `2D: [1, 256]` | 2 | SenseTime_deformable-detr-single-scale, SenseTime_deformable-detr-single-scale-dc5 |
| `3D: [1, 22, 1024]` | 2 | allegro_plt5-large, allegro_plt5-large |
| `3D: [1, 15, 1280]` | 2 | breadlicker45_autotrain-blender-50601120822, breadlicker45_autotrain-blender-50601120822 |
| `3D: [1, 14, 768]` | 2 | cointegrated_rut5-base-absum, cointegrated_rut5-base-absum |
| `3D: [1, 50, 1472]` | 2 | google_byt5-small, google_byt5-small |
| `3D: [1, 1, 512]` | 2 | google_flan_t5_small, openai_whisper-base |
| `4D: [1, 3, 960, 960]` | 2 | google_owlv2-base-patch16, upfeatmediainc_owlv2-base-patch16-ensemble |
| `2D: [3600, 4]` | 2 | google_owlv2-base-patch16, google_owlvit-large-patch14 |
| `4D: [1, 3, 1008, 1008]` | 2 | google_owlv2-large-patch14, google_owlv2-large-patch14-finetuned |
| `3D: [1, 12, 384]` | 2 | google_t5-efficient-mini, google_t5-efficient-mini |
| `3D: [1, 12, 256]` | 2 | google_t5-efficient-tiny, google_t5-efficient-tiny |
| `3D: [1, 15, 512]` | 2 | gsarti_it5-efficient-small-el32-wiki-summarization, gsarti_it5-efficient-small-el32-wiki-summarization |
| `2D: [2, 17]` | 2 | hf-tiny-model-private_tiny-random-AltCLIPModel, hf-tiny-model-private_tiny-random-AltCLIPModel |
| `2D: [2, 14]` | 2 | hf-tiny-model-private_tiny-random-BlipModel, hf-tiny-model-private_tiny-random-BlipModel |
| `3D: [1, 19, 32]` | 2 | hf-tiny-model-private_tiny-random-BloomForSequenceClassification, ydshieh_tiny-random-gptj-for-sequence-classification |
| `4D: [1, 3, 600, 600]` | 2 | hf-tiny-model-private_tiny-random-EfficientNetForImageClassification, hf-tiny-model-private_tiny-random-EfficientNetModel |
| `3D: [1, 15, 16]` | 2 | hf-tiny-model-private_tiny-random-MBartForSequenceClassification, hf-tiny-model-private_tiny-random-MBartForSequenceClassification |
| `3D: [1, 11, 16]` | 2 | hf-tiny-model-private_tiny-random-PLBartForSequenceClassification, hf-tiny-model-private_tiny-random-PLBartForSequenceClassification |
| `5D: [1, 2, 3, 10, 10]` | 2 | hf-tiny-model-private_tiny-random-TimesformerForVideoClassification, hf-tiny-model-private_tiny-random-VideoMAEForVideoClassification |
| `3D: [1, 33, 1024]` | 2 | hyesunyun_NonsenseUpdateDiffIntBart, hyesunyun_NonsenseUpdateDiffIntBart |
| `3D: [1, 13, 512]` | 2 | kannt-im_marian-finetuned-kde4-en-to-fr, kannt-im_marian-finetuned-kde4-en-to-fr |
| `3D: [1, 22, 512]` | 2 | moska_plt5-seq-clf-with-entities-updated-finetuned, moska_plt5-seq-clf-with-entities-updated-finetuned |
| `2D: [1, 57445]` | 2 | opus-mt-af-en, opus-mt-en-af |
| `2D: [1, 7475]` | 2 | opus-mt-af-eo, opus-mt-eo-af |
| `2D: [1, 60216]` | 2 | opus-mt-af-es, opus-mt-es-af |
| `2D: [1, 60657]` | 2 | opus-mt-af-fi, opus-mt-fi-af |
| `2D: [1, 61180]` | 2 | opus-mt-afa-en, opus-mt-en-sem |
| `2D: [1, 56]` | 2 | opus-mt-ar-el, opus-mt-ar-el |
| `2D: [1, 64]` | 2 | opus-mt-ar-eo, opus-mt-ar-eo |
| `2D: [1, 53676]` | 2 | opus-mt-ase-de, opus-mt-de-ase |
| `2D: [1, 53502]` | 2 | opus-mt-ase-es, opus-mt-es-ase |
| `2D: [1, 60738]` | 2 | opus-mt-bcl-de, opus-mt-de-bcl |
| `2D: [1, 58455]` | 2 | opus-mt-bcl-en, opus-mt-en-bcl |
| `2D: [1, 59736]` | 2 | opus-mt-bcl-es, opus-mt-es-bcl |
| `2D: [1, 61739]` | 2 | opus-mt-bcl-fi, opus-mt-fi-bcl |
| `2D: [1, 60]` | 2 | opus-mt-be-es, opus-mt-be-es |
| `2D: [1, 59828]` | 2 | opus-mt-bem-en, opus-mt-en-bem |
| `2D: [1, 61763]` | 2 | opus-mt-bem-fi, opus-mt-fi-bem |
| `2D: [1, 36048]` | 2 | opus-mt-ber-en, opus-mt-en-ber |
| `2D: [1, 13296]` | 2 | opus-mt-ber-es, opus-mt-es-ber |
| `2D: [1, 61110]` | 2 | opus-mt-bg-de, opus-mt-de-bg |
| `2D: [1, 66]` | 2 | opus-mt-bg-eo, opus-mt-bg-eo |
| `2D: [1, 7770]` | 2 | opus-mt-bg-eo, opus-mt-eo-bg |
| `2D: [1, 61845]` | 2 | opus-mt-bg-es, opus-mt-es-bg |
| `2D: [1, 60069]` | 2 | opus-mt-bg-fi, opus-mt-fi-bg |
| `2D: [1, 59345]` | 2 | opus-mt-bg-sv, opus-mt-fi-NORWAY |
| `2D: [1, 54321]` | 2 | opus-mt-bi-en, opus-mt-en-bi |
| `2D: [1, 56412]` | 2 | opus-mt-bi-es, opus-mt-es-bi |
| `2D: [1, 47919]` | 2 | opus-mt-bzs-en, opus-mt-en-bzs |
| `2D: [1, 51421]` | 2 | opus-mt-bzs-es, opus-mt-es-bzs |
| `2D: [1, 59433]` | 2 | opus-mt-bzs-fi, opus-mt-fi-bzs |
| `2D: [1, 22116]` | 2 | opus-mt-ca-de, opus-mt-de-ca |
| `2D: [1, 55255]` | 2 | opus-mt-ca-en, opus-mt-en-ca |
| `2D: [1, 57]` | 2 | opus-mt-cau-en, opus-mt-cau-en |
| `2D: [1, 61]` | 2 | opus-mt-ccs-en, opus-mt-ccs-en |
| `2D: [1, 59324]` | 2 | opus-mt-ceb-es, opus-mt-es-ceb |
| `2D: [1, 61010]` | 2 | opus-mt-ceb-fi, opus-mt-fi-ceb |
| `2D: [1, 59183]` | 2 | opus-mt-ceb-sv, opus-mt-tc-bible-big-urj-deu_eng_nld |
| `2D: [1, 56327]` | 2 | opus-mt-chk-en, opus-mt-en-chk |
| `2D: [1, 47285]` | 2 | opus-mt-crs-de, opus-mt-de-crs |
| `2D: [1, 43620]` | 2 | opus-mt-crs-en, opus-mt-en-crs |
| `2D: [1, 47319]` | 2 | opus-mt-crs-es, opus-mt-es-crs |
| `2D: [1, 48214]` | 2 | opus-mt-crs-fi, opus-mt-fi-crs |
| `2D: [1, 58234]` | 2 | opus-mt-cs-de, opus-mt-de-cs |
| `2D: [1, 62509]` | 2 | opus-mt-cs-en, opus-mt-en-cs |
| `2D: [1, 7397]` | 2 | opus-mt-cs-eo, opus-mt-eo-cs |
| `2D: [1, 59012]` | 2 | opus-mt-cs-fi, opus-mt-fi-cs |
| `2D: [1, 11802]` | 2 | opus-mt-csg-es, opus-mt-es-csg |
| `2D: [1, 54395]` | 2 | opus-mt-cy-en, opus-mt-en-cy |
| `2D: [1, 57027]` | 2 | opus-mt-da-de, opus-mt-de-da |
| `2D: [1, 58930]` | 2 | opus-mt-da-en, opus-mt-en-da |
| `2D: [1, 7286]` | 2 | opus-mt-da-eo, opus-mt-eo-da |
| `2D: [1, 57584]` | 2 | opus-mt-da-es, opus-mt-es-da |
| `2D: [1, 59299]` | 2 | opus-mt-de-af, opus-mt-fi-id |
| `2D: [1, 60346]` | 2 | opus-mt-de-ee, opus-mt-de-tl |
| `2D: [1, 59733]` | 2 | opus-mt-de-efi, opus-mt-efi-de |
| `2D: [1, 58101]` | 2 | opus-mt-de-en, opus-mt-en-de |
| `2D: [1, 61301]` | 2 | opus-mt-de-es, opus-mt-es-de |
| `2D: [1, 58042]` | 2 | opus-mt-de-et, opus-mt-et-de |
| `2D: [1, 22170]` | 2 | opus-mt-de-eu, opus-mt-eu-de |
| `2D: [1, 61153]` | 2 | opus-mt-de-fr, opus-mt-en-sn |
| `2D: [1, 60025]` | 2 | opus-mt-de-ig, opus-mt-en-sk |
| `2D: [1, 57579]` | 2 | opus-mt-ee-en, opus-mt-en-ee |
| `2D: [1, 60163]` | 2 | opus-mt-ee-es, opus-mt-es-ee |
| `2D: [1, 59678]` | 2 | opus-mt-ee-sv, opus-mt-es-pl |
| `2D: [1, 55090]` | 2 | opus-mt-efi-en, opus-mt-en-efi |
| `2D: [1, 61193]` | 2 | opus-mt-efi-fi, opus-mt-fi-efi |
| `2D: [1, 7713]` | 2 | opus-mt-el-eo, opus-mt-eo-el |
| `2D: [1, 59727]` | 2 | opus-mt-el-fi, opus-mt-fi-el |
| `2D: [1, 59197]` | 2 | opus-mt-en-eo, opus-mt-eo-en |
| `2D: [1, 58866]` | 2 | opus-mt-en-et, opus-mt-et-en |
| `2D: [1, 21964]` | 2 | opus-mt-en-euq, opus-mt-euq-en |
| `2D: [1, 62518]` | 2 | opus-mt-en-ru, opus-mt-es-ar |
| `2D: [1, 58379]` | 2 | opus-mt-eo-es, opus-mt-es-eo |
| `2D: [1, 7439]` | 2 | opus-mt-eo-fi, opus-mt-fi-eo |
| `2D: [1, 7347]` | 2 | opus-mt-eo-ro, opus-mt-eo-sh |
| `2D: [1, 58124]` | 2 | opus-mt-es-et, opus-mt-et-es |
| `2D: [1, 58060]` | 2 | opus-mt-es-eu, opus-mt-eu-es |
| `2D: [1, 63261]` | 2 | opus-mt-es-fi, opus-mt-fi-es |
| `2D: [1, 61389]` | 2 | opus-mt-es-tll, opus-mt-tc-bible-big-afa-en |
| `2D: [1, 58055]` | 2 | opus-mt-et-fi, opus-mt-fi-et |
| `2D: [1, 61529]` | 2 | opus-mt-fi-ln, opus-mt-fi-ny |
| `3D: [1, 10, 768]` | 2 | pszemraj_tFINE-base-300m-samsum, pszemraj_tFINE-base-300m-samsum |
| `2D: [3, 8]` | 2 | sadhaklal_mlp-california-housing, sadhaklal_wide-and-deep-net-california-housing |
| `2D: [3, 6]` | 2 | sadhaklal_wide-and-deep-net-california-housing-v2, sadhaklal_wide-and-deep-net-california-housing-v3 |
| `2D: [3, 5]` | 2 | sadhaklal_wide-and-deep-net-california-housing-v2, sadhaklal_wide-and-deep-net-california-housing-v3 |
| `3D: [1, 18, 1024]` | 2 | sbulut_finetuned-kde4-en-to-tr, sbulut_finetuned-kde4-en-to-tr |
| `2D: [1, 2880]` | 2 | thuml_sundial-base-128m, thuml_timer-base-84m |
| `3D: [1, 13, 768]` | 1 | DatologyAI_cls-opt-vit-b-32 |
| `4D: [1, 3, 112, 112]` | 1 | Dimmas_Finetuned_facebook_dino-vitb14_98x98_flowers |
| `3D: [1, 2, 2048]` | 1 | EleutherAI_pythia-1b |
| `2D: [1, 2]` | 1 | EleutherAI_pythia-1b |
| `4D: [1, 96, 64, 64]` | 1 | Intel_dpt-swinv2-tiny-256 |
| `4D: [1, 192, 32, 32]` | 1 | Intel_dpt-swinv2-tiny-256 |
| `4D: [1, 384, 16, 16]` | 1 | Intel_dpt-swinv2-tiny-256 |
| `4D: [1, 768, 8, 8]` | 1 | Intel_dpt-swinv2-tiny-256 |
| `3D: [1, 9, 2048]` | 1 | KoboldAI_fairseq-dense-1.3B |
| `3D: [1, 9, 768]` | 1 | KoboldAI_fairseq-dense-125M |
| `3D: [1, 128, 128]` | 1 | MIT_ast-finetuned-speech-commands-v2 |
| `2D: [576, 4]` | 1 | Thomasboosinger_owlvit-base-patch32 |
| `4D: [1, 3, 1536, 1536]` | 1 | apple_DepthPro-hf |
| `3D: [1, 128, 521]` | 1 | canary-1b-v2 |
| `1D: [1]` | 1 | canary-1b-v2 |
| `3D: [1, 9999, 1024]` | 1 | canary-1b-v2 |
| `2D: [1, 16000]` | 1 | facebook_wav2vec2-base-960h |
| `3D: [1, 13, 1024]` | 1 | facebook_xglm-564M |
| `5D: [1, 16, 3, 448, 448]` | 1 | fcakyon_timesformer-hr-finetuned-k400 |
| `5D: [1, 96, 3, 224, 224]` | 1 | fcakyon_timesformer-large-finetuned-k400 |
| `2D: [1, 220500]` | 1 | galsenai_hubert-large-ls960-ft-waxal-keyword-spotting |
| `3D: [1, 1, 1536]` | 1 | google_byt5_base |
| `3D: [1, 16, 1536]` | 1 | google_byt5_base |
| `3D: [1, 1, 1472]` | 1 | google_byt5_small |
| `3D: [1, 16, 1472]` | 1 | google_byt5_small |
| `3D: [1, 1, 768]` | 1 | google_flan_t5_base |
| `3D: [1, 1, 1024]` | 1 | google_flan_t5_large |
| `2D: [5184, 4]` | 1 | google_owlv2-large-patch14 |
| `2D: [2304, 4]` | 1 | google_owlvit-base-patch16 |
| `4D: [1, 3, 840, 840]` | 1 | google_owlvit-large-patch14 |
| `3D: [1, 2048, 770]` | 1 | google_pix2struct-base |
| `3D: [1, 20, 32]` | 1 | hf-tiny-model-private_tiny-random-GPTJForSequenceClassification |
| `3D: [1, 21, 16]` | 1 | hf-tiny-model-private_tiny-random-OPTForSequenceClassification |
| `2D: [256, 4]` | 1 | hf-tiny-model-private_tiny-random-OwlViTForObjectDetection |
| `3D: [1, 49, 322]` | 1 | hf-tiny-model-private_tiny-random-PerceiverForImageClassificationConvProcessing |
| `3D: [1, 1024, 261]` | 1 | hf-tiny-model-private_tiny-random-PerceiverForImageClassificationFourier |
| `3D: [1, 1024, 512]` | 1 | hf-tiny-model-private_tiny-random-PerceiverForImageClassificationLearned |
| `4D: [1, 3, 518, 518]` | 1 | internetoftim_dinov2-base |
| `2D: [1024, 768]` | 1 | microsoft_xclip-large-patch14 |
| `3D: [1, 1500, 512]` | 1 | openai_whisper-base |
| `2D: [1, 58012]` | 1 | opus-mt-NORTH_EU-NORTH_EU |
| `2D: [1, 47231]` | 1 | opus-mt-SCANDINAVIA-SCANDINAVIA |
| `2D: [1, 56039]` | 1 | opus-mt-aav-en |
| `2D: [1, 23625]` | 1 | opus-mt-aed-es |
| `2D: [1, 58757]` | 1 | opus-mt-af-de |
| `2D: [1, 59422]` | 1 | opus-mt-af-fr |
| `2D: [1, 53247]` | 1 | opus-mt-af-nl |
| `2D: [1, 62842]` | 1 | opus-mt-af-ru |
| `2D: [1, 58463]` | 1 | opus-mt-af-sv |
| `2D: [1, 33714]` | 1 | opus-mt-afa-afa |
| `2D: [1, 61549]` | 1 | opus-mt-alv-en |
| `2D: [1, 63199]` | 1 | opus-mt-am-sv |
| `2D: [1, 63079]` | 1 | opus-mt-ar-de |
| `2D: [1, 63336]` | 1 | opus-mt-ar-el |
| `2D: [1, 62834]` | 1 | opus-mt-ar-en |
| `2D: [1, 7741]` | 1 | opus-mt-ar-eo |
| `2D: [1, 62526]` | 1 | opus-mt-ar-es |
| `2D: [1, 64505]` | 1 | opus-mt-ar-fr |
| `2D: [1, 63334]` | 1 | opus-mt-ar-he |
| `2D: [1, 63294]` | 1 | opus-mt-ar-it |
| `2D: [1, 62606]` | 1 | opus-mt-ar-ru |
| `2D: [1, 63272]` | 1 | opus-mt-ar-tr |
| `2D: [1, 59131]` | 1 | opus-mt-art-en |
| `2D: [1, 24831]` | 1 | opus-mt-ase-en |
| `2D: [1, 50895]` | 1 | opus-mt-ase-fr |
| `2D: [1, 48742]` | 1 | opus-mt-ase-sv |
| `2D: [1, 7590]` | 1 | opus-mt-az-es |
| `2D: [1, 6904]` | 1 | opus-mt-az-tr |
| `2D: [1, 59056]` | 1 | opus-mt-bat-en |
| `2D: [1, 60484]` | 1 | opus-mt-bcl-fr |
| `2D: [1, 59911]` | 1 | opus-mt-bcl-sv |
| `2D: [1, 7630]` | 1 | opus-mt-be-es |
| `2D: [1, 60996]` | 1 | opus-mt-bem-es |
| `2D: [1, 61027]` | 1 | opus-mt-bem-fr |
| `2D: [1, 60613]` | 1 | opus-mt-bem-sv |
| `2D: [1, 20994]` | 1 | opus-mt-ber-fr |
| `2D: [1, 61813]` | 1 | opus-mt-bg-en |
| `2D: [1, 61483]` | 1 | opus-mt-bg-fr |
| `2D: [1, 61468]` | 1 | opus-mt-bg-it |
| `2D: [1, 57310]` | 1 | opus-mt-bg-ru |
| `2D: [1, 63138]` | 1 | opus-mt-bg-tr |
| `2D: [1, 59878]` | 1 | opus-mt-bg-uk |
| `2D: [1, 57508]` | 1 | opus-mt-bi-fr |
| `2D: [1, 55334]` | 1 | opus-mt-bi-sv |
| `2D: [1, 63597]` | 1 | opus-mt-bn-en |
| `2D: [1, 61729]` | 1 | opus-mt-bnt-en |
| `2D: [1, 54231]` | 1 | opus-mt-bzs-fr |
| `2D: [1, 51381]` | 1 | opus-mt-bzs-sv |
| `2D: [1, 49621]` | 1 | opus-mt-ca-es |
| `2D: [1, 21410]` | 1 | opus-mt-ca-fr |
| `2D: [1, 21528]` | 1 | opus-mt-ca-it |
| `2D: [1, 21976]` | 1 | opus-mt-ca-nl |
| `2D: [1, 20554]` | 1 | opus-mt-ca-pt |
| `2D: [1, 7545]` | 1 | opus-mt-ca-uk |
| `2D: [1, 62393]` | 1 | opus-mt-cau-en |
| `2D: [1, 23173]` | 1 | opus-mt-ccs-en |
| `2D: [1, 57239]` | 1 | opus-mt-ceb-en |
| `2D: [1, 59770]` | 1 | opus-mt-ceb-fr |
| `2D: [1, 57907]` | 1 | opus-mt-cel-en |
| `2D: [1, 60011]` | 1 | opus-mt-chk-es |
| `2D: [1, 59610]` | 1 | opus-mt-chk-fr |
| `2D: [1, 58904]` | 1 | opus-mt-chk-sv |
| `2D: [1, 57679]` | 1 | opus-mt-cpf-en |
| `2D: [1, 31989]` | 1 | opus-mt-cpp-cpp |
| `2D: [1, 57790]` | 1 | opus-mt-cpp-en |
| `2D: [1, 46774]` | 1 | opus-mt-crs-fr |
| `2D: [1, 46013]` | 1 | opus-mt-crs-sv |
| `2D: [1, 58338]` | 1 | opus-mt-cs-fr |
| `2D: [1, 57482]` | 1 | opus-mt-cs-sv |
| `2D: [1, 62778]` | 1 | opus-mt-cs-uk |
| `2D: [1, 28047]` | 1 | opus-mt-csn-es |
| `2D: [1, 59606]` | 1 | opus-mt-da-fi |
| `2D: [1, 57138]` | 1 | opus-mt-da-fr |
| `2D: [1, 18881]` | 1 | opus-mt-da-no |
| `2D: [1, 62922]` | 1 | opus-mt-da-ru |
| `2D: [1, 61916]` | 1 | opus-mt-de-ZH |
| `2D: [1, 63077]` | 1 | opus-mt-de-ar |
| `2D: [1, 56367]` | 1 | opus-mt-de-bi |
| `2D: [1, 58819]` | 1 | opus-mt-de-bzs |
| `2D: [1, 33217]` | 1 | opus-mt-de-de |
| `2D: [1, 59951]` | 1 | opus-mt-de-el |
| `2D: [1, 59971]` | 1 | opus-mt-de-fi |
| `2D: [1, 59804]` | 1 | opus-mt-de-fj |
| `2D: [1, 60424]` | 1 | opus-mt-de-gaa |
| `2D: [1, 53500]` | 1 | opus-mt-de-gil |
| `2D: [1, 54544]` | 1 | opus-mt-de-guw |
| `2D: [1, 59664]` | 1 | opus-mt-de-ha |
| `2D: [1, 63066]` | 1 | opus-mt-de-he |
| `2D: [1, 60041]` | 1 | opus-mt-de-hil |
| `2D: [1, 45292]` | 1 | opus-mt-de-ho |
| `2D: [1, 58701]` | 1 | opus-mt-de-hr |
| `2D: [1, 49000]` | 1 | opus-mt-de-ht |
| `2D: [1, 58671]` | 1 | opus-mt-de-hu |
| `2D: [1, 60194]` | 1 | opus-mt-de-ilo |
| `2D: [1, 59315]` | 1 | opus-mt-de-is |
| `2D: [1, 47660]` | 1 | opus-mt-de-iso |
| `2D: [1, 59867]` | 1 | opus-mt-de-it |
| `2D: [1, 49964]` | 1 | opus-mt-de-kg |
| `2D: [1, 60592]` | 1 | opus-mt-de-ln |
| `2D: [1, 60846]` | 1 | opus-mt-de-loz |
| `2D: [1, 58862]` | 1 | opus-mt-de-lt |
| `2D: [1, 60920]` | 1 | opus-mt-de-lua |
| `2D: [1, 57716]` | 1 | opus-mt-de-ms |
| `2D: [1, 57955]` | 1 | opus-mt-de-mt |
| `2D: [1, 44948]` | 1 | opus-mt-de-niu |
| `2D: [1, 57567]` | 1 | opus-mt-de-nl |
| `2D: [1, 7020]` | 1 | opus-mt-de-no |
| `2D: [1, 60441]` | 1 | opus-mt-de-nso |
| `2D: [1, 60883]` | 1 | opus-mt-de-ny |
| `2D: [1, 60768]` | 1 | opus-mt-de-pag |
| `2D: [1, 60181]` | 1 | opus-mt-de-pap |
| `2D: [1, 48977]` | 1 | opus-mt-de-pis |
| `2D: [1, 60431]` | 1 | opus-mt-de-pl |
| `2D: [1, 56761]` | 1 | opus-mt-de-pon |
| `2D: [1, 62523]` | 1 | opus-mt-de-uk |
| `2D: [1, 58225]` | 1 | opus-mt-de-vi |
| `2D: [1, 62943]` | 1 | opus-mt-dra-en |
| `2D: [1, 61449]` | 1 | opus-mt-ee-fi |
| `2D: [1, 60218]` | 1 | opus-mt-ee-fr |
| `2D: [1, 59481]` | 1 | opus-mt-efi-fr |
| `2D: [1, 59076]` | 1 | opus-mt-efi-sv |
| `2D: [1, 63305]` | 1 | opus-mt-el-ar |
| `2D: [1, 60763]` | 1 | opus-mt-el-fr |
| `2D: [1, 9001]` | 1 | opus-mt-en-CELTIC |
| `2D: [1, 56071]` | 1 | opus-mt-en-aav |
| `2D: [1, 61199]` | 1 | opus-mt-en-afa |
| `2D: [1, 61577]` | 1 | opus-mt-en-alv |
| `2D: [1, 62802]` | 1 | opus-mt-en-ar |
| `2D: [1, 23288]` | 1 | opus-mt-en-az |
| `2D: [1, 59061]` | 1 | opus-mt-en-bat |
| `2D: [1, 61708]` | 1 | opus-mt-en-bg |
| `2D: [1, 61740]` | 1 | opus-mt-en-bnt |
| `2D: [1, 56824]` | 1 | opus-mt-en-ceb |
| `2D: [1, 57912]` | 1 | opus-mt-en-cel |
| `2D: [1, 57681]` | 1 | opus-mt-en-cpf |
| `2D: [1, 57766]` | 1 | opus-mt-en-cpp |
| `2D: [1, 62952]` | 1 | opus-mt-en-dra |
| `2D: [1, 64826]` | 1 | opus-mt-en-el |
| `2D: [1, 58286]` | 1 | opus-mt-en-eu |
| `2D: [1, 58938]` | 1 | opus-mt-en-fiu |
| `2D: [1, 55835]` | 1 | opus-mt-en-fj |
| `2D: [1, 56683]` | 1 | opus-mt-en-ga |
| `2D: [1, 57186]` | 1 | opus-mt-en-gaa |
| `2D: [1, 56629]` | 1 | opus-mt-en-gem |
| `2D: [1, 50021]` | 1 | opus-mt-en-gil |
| `2D: [1, 55458]` | 1 | opus-mt-en-gl |
| `2D: [1, 57068]` | 1 | opus-mt-en-gmq |
| `2D: [1, 55477]` | 1 | opus-mt-en-gmw |
| `2D: [1, 52253]` | 1 | opus-mt-en-guw |
| `2D: [1, 12339]` | 1 | opus-mt-en-gv |
| `2D: [1, 58089]` | 1 | opus-mt-en-ha |
| `2D: [1, 65839]` | 1 | opus-mt-en-he |
| `2D: [1, 61950]` | 1 | opus-mt-en-hi |
| `2D: [1, 56301]` | 1 | opus-mt-en-hil |
| `2D: [1, 42463]` | 1 | opus-mt-en-ho |
| `2D: [1, 47809]` | 1 | opus-mt-en-ht |
| `2D: [1, 62522]` | 1 | opus-mt-en-hu |
| `2D: [1, 7754]` | 1 | opus-mt-en-hy |
| `2D: [1, 54796]` | 1 | opus-mt-en-id |
| `2D: [1, 56127]` | 1 | opus-mt-en-ig |
| `2D: [1, 62228]` | 1 | opus-mt-en-iir |
| `2D: [1, 57777]` | 1 | opus-mt-en-ilo |
| `2D: [1, 61760]` | 1 | opus-mt-en-inc |
| `2D: [1, 61161]` | 1 | opus-mt-en-ine |
| `2D: [1, 58647]` | 1 | opus-mt-en-is |
| `2D: [1, 45029]` | 1 | opus-mt-en-iso |
| `2D: [1, 80035]` | 1 | opus-mt-en-it |
| `2D: [1, 57148]` | 1 | opus-mt-en-itc |
| `2D: [1, 46276]` | 1 | opus-mt-en-jap |
| `2D: [1, 47322]` | 1 | opus-mt-en-kg |
| `2D: [1, 50183]` | 1 | opus-mt-en-kj |
| `2D: [1, 60142]` | 1 | opus-mt-en-kqn |
| `2D: [1, 50346]` | 1 | opus-mt-en-kwn |
| `2D: [1, 58115]` | 1 | opus-mt-en-kwy |
| `2D: [1, 60447]` | 1 | opus-mt-en-lg |
| `2D: [1, 58314]` | 1 | opus-mt-en-ln |
| `2D: [1, 57974]` | 1 | opus-mt-en-loz |
| `2D: [1, 60241]` | 1 | opus-mt-en-lu |
| `2D: [1, 59447]` | 1 | opus-mt-en-lua |
| `2D: [1, 61313]` | 1 | opus-mt-en-lue |
| `2D: [1, 56530]` | 1 | opus-mt-en-lun |
| `2D: [1, 52236]` | 1 | opus-mt-en-luo |
| `2D: [1, 49567]` | 1 | opus-mt-en-lus |
| `2D: [1, 60088]` | 1 | opus-mt-en-map |
| `2D: [1, 48883]` | 1 | opus-mt-en-mfe |
| `2D: [1, 54688]` | 1 | opus-mt-en-mg |
| `2D: [1, 49657]` | 1 | opus-mt-en-mh |
| `2D: [1, 61946]` | 1 | opus-mt-en-mk |
| `2D: [1, 56075]` | 1 | opus-mt-en-mkh |
| `2D: [1, 24661]` | 1 | opus-mt-en-ml |
| `2D: [1, 47022]` | 1 | opus-mt-en-mos |
| `2D: [1, 61674]` | 1 | opus-mt-en-mr |
| `2D: [1, 56041]` | 1 | opus-mt-en-mt |
| `2D: [1, 64110]` | 1 | opus-mt-en-mul |
| `2D: [1, 49883]` | 1 | opus-mt-en-ng |
| `2D: [1, 61556]` | 1 | opus-mt-en-nic |
| `2D: [1, 42155]` | 1 | opus-mt-en-niu |
| `2D: [1, 67028]` | 1 | opus-mt-en-nl |
| `2D: [1, 57542]` | 1 | opus-mt-en-nso |
| `2D: [1, 59811]` | 1 | opus-mt-en-ny |
| `2D: [1, 52867]` | 1 | opus-mt-en-nyk |
| `2D: [1, 59898]` | 1 | opus-mt-en-om |
| `2D: [1, 57487]` | 1 | opus-mt-en-pag |
| `2D: [1, 57294]` | 1 | opus-mt-en-pap |
| `2D: [1, 59866]` | 1 | opus-mt-en-phi |
| `2D: [1, 42010]` | 1 | opus-mt-en-pis |
| `2D: [1, 52997]` | 1 | opus-mt-en-pon |
| `2D: [1, 60175]` | 1 | opus-mt-en-poz |
| `2D: [1, 60089]` | 1 | opus-mt-en-pqe |
| `2D: [1, 58849]` | 1 | opus-mt-en-pqw |
| `2D: [1, 7528]` | 1 | opus-mt-en-rn |
| `2D: [1, 53832]` | 1 | opus-mt-en-rnd |
| `2D: [1, 59543]` | 1 | opus-mt-en-ro |
| `2D: [1, 56671]` | 1 | opus-mt-en-roa |
| `2D: [1, 61335]` | 1 | opus-mt-en-run |
| `2D: [1, 60384]` | 1 | opus-mt-en-rw |
| `2D: [1, 55872]` | 1 | opus-mt-en-sal |
| `2D: [1, 43822]` | 1 | opus-mt-en-sg |
| `2D: [1, 59414]` | 1 | opus-mt-en-sit |
| `2D: [1, 58879]` | 1 | opus-mt-en-sla |
| `2D: [1, 54880]` | 1 | opus-mt-en-sm |
| `2D: [1, 59158]` | 1 | opus-mt-en-sq |
| `2D: [1, 55129]` | 1 | opus-mt-en-ss |
| `2D: [1, 57355]` | 1 | opus-mt-en-st |
| `2D: [1, 56434]` | 1 | opus-mt-en-sv |
| `2D: [1, 58950]` | 1 | opus-mt-en-sw |
| `2D: [1, 58905]` | 1 | opus-mt-en-swc |
| `2D: [1, 34360]` | 1 | opus-mt-en-tdt |
| `2D: [1, 43019]` | 1 | opus-mt-en-tiv |
| `2D: [1, 57373]` | 1 | opus-mt-en-tl |
| `2D: [1, 57306]` | 1 | opus-mt-en-tn |
| `2D: [1, 57446]` | 1 | opus-mt-en-to |
| `2D: [1, 61051]` | 1 | opus-mt-en-toi |
| `2D: [1, 49238]` | 1 | opus-mt-en-tpi |
| `2D: [1, 57469]` | 1 | opus-mt-en-ts |
| `2D: [1, 61657]` | 1 | opus-mt-en-tut |
| `2D: [1, 38380]` | 1 | opus-mt-en-tvl |
| `2D: [1, 57000]` | 1 | opus-mt-en-tw |
| `2D: [1, 56506]` | 1 | opus-mt-en-ty |
| `2D: [1, 61587]` | 1 | opus-mt-en-uk |
| `2D: [1, 58673]` | 1 | opus-mt-en-umb |
| `2D: [1, 62025]` | 1 | opus-mt-en-ur |
| `2D: [1, 58920]` | 1 | opus-mt-en-urj |
| `2D: [1, 53685]` | 1 | opus-mt-en-vi |
| `2D: [1, 61285]` | 1 | opus-mt-en-xh |
| `2D: [1, 61605]` | 1 | opus-mt-en-zle |
| `2D: [1, 57680]` | 1 | opus-mt-en-zls |
| `2D: [1, 58052]` | 1 | opus-mt-en-zlw |
| `2D: [1, 59728]` | 1 | opus-mt-eo-fr |
| `2D: [1, 7768]` | 1 | opus-mt-eo-he |
| `2D: [1, 7477]` | 1 | opus-mt-eo-hu |
| `2D: [1, 7276]` | 1 | opus-mt-eo-it |
| `2D: [1, 7344]` | 1 | opus-mt-eo-nl |
| `2D: [1, 7325]` | 1 | opus-mt-eo-pl |
| `2D: [1, 7246]` | 1 | opus-mt-eo-pt |
| `2D: [1, 7730]` | 1 | opus-mt-eo-ru |
| `2D: [1, 7284]` | 1 | opus-mt-eo-sv |
| `2D: [1, 58071]` | 1 | opus-mt-es-NORWAY |
| `2D: [1, 51970]` | 1 | opus-mt-es-ca |
| `2D: [1, 58623]` | 1 | opus-mt-es-cs |
| `2D: [1, 59814]` | 1 | opus-mt-es-efi |
| `2D: [1, 60782]` | 1 | opus-mt-es-el |
| `2D: [1, 33253]` | 1 | opus-mt-es-es |
| `2D: [1, 74822]` | 1 | opus-mt-es-fr |
| `2D: [1, 60173]` | 1 | opus-mt-es-gaa |
| `2D: [1, 53428]` | 1 | opus-mt-es-gil |
| `2D: [1, 5805]` | 1 | opus-mt-es-gl |
| `2D: [1, 54512]` | 1 | opus-mt-es-guw |
| `2D: [1, 59503]` | 1 | opus-mt-es-ha |
| `2D: [1, 63105]` | 1 | opus-mt-es-he |
| `2D: [1, 59256]` | 1 | opus-mt-es-hil |
| `2D: [1, 45168]` | 1 | opus-mt-es-ho |
| `2D: [1, 58648]` | 1 | opus-mt-es-hr |
| `2D: [1, 49387]` | 1 | opus-mt-es-ht |
| `2D: [1, 57414]` | 1 | opus-mt-es-id |
| `2D: [1, 60006]` | 1 | opus-mt-es-ig |
| `2D: [1, 59457]` | 1 | opus-mt-es-ilo |
| `2D: [1, 60335]` | 1 | opus-mt-es-is |
| `2D: [1, 47550]` | 1 | opus-mt-es-iso |
| `2D: [1, 49808]` | 1 | opus-mt-es-kg |
| `2D: [1, 60350]` | 1 | opus-mt-es-ln |
| `2D: [1, 60783]` | 1 | opus-mt-es-loz |
| `2D: [1, 59511]` | 1 | opus-mt-es-lt |
| `2D: [1, 60813]` | 1 | opus-mt-es-lua |
| `2D: [1, 53585]` | 1 | opus-mt-es-lus |
| `2D: [1, 28283]` | 1 | opus-mt-es-mfs |
| `2D: [1, 62482]` | 1 | opus-mt-es-mk |
| `2D: [1, 57085]` | 1 | opus-mt-es-mt |
| `2D: [1, 45309]` | 1 | opus-mt-es-niu |
| `2D: [1, 58264]` | 1 | opus-mt-es-nl |
| `2D: [1, 22026]` | 1 | opus-mt-es-no |
| `2D: [1, 60344]` | 1 | opus-mt-es-nso |
| `2D: [1, 60819]` | 1 | opus-mt-es-ny |
| `2D: [1, 60074]` | 1 | opus-mt-es-pag |
| `2D: [1, 56010]` | 1 | opus-mt-es-pap |
| `2D: [1, 49022]` | 1 | opus-mt-es-pis |
| `2D: [1, 56793]` | 1 | opus-mt-es-pon |
| `2D: [1, 10873]` | 1 | opus-mt-es-prl |
| `2D: [1, 7512]` | 1 | opus-mt-es-rn |
| `2D: [1, 56140]` | 1 | opus-mt-es-ro |
| `2D: [1, 63430]` | 1 | opus-mt-es-ru |
| `2D: [1, 61291]` | 1 | opus-mt-es-rw |
| `2D: [1, 46098]` | 1 | opus-mt-es-sg |
| `2D: [1, 57959]` | 1 | opus-mt-es-sl |
| `2D: [1, 58284]` | 1 | opus-mt-es-sm |
| `2D: [1, 61622]` | 1 | opus-mt-es-sn |
| `2D: [1, 60052]` | 1 | opus-mt-es-srn |
| `2D: [1, 60207]` | 1 | opus-mt-es-st |
| `2D: [1, 60332]` | 1 | opus-mt-es-swc |
| `2D: [1, 59655]` | 1 | opus-mt-es-tl |
| `2D: [1, 60267]` | 1 | opus-mt-es-tn |
| `2D: [1, 58959]` | 1 | opus-mt-es-to |
| `2D: [1, 52913]` | 1 | opus-mt-es-tpi |
| `2D: [1, 42688]` | 1 | opus-mt-es-tvl |
| `2D: [1, 60033]` | 1 | opus-mt-es-tw |
| `2D: [1, 58873]` | 1 | opus-mt-es-ty |
| `2D: [1, 54248]` | 1 | opus-mt-es-tzo |
| `2D: [1, 62460]` | 1 | opus-mt-es-uk |
| `2D: [1, 56171]` | 1 | opus-mt-es-ve |
| `2D: [1, 58488]` | 1 | opus-mt-es-vi |
| `2D: [1, 59954]` | 1 | opus-mt-es-war |
| `2D: [1, 41820]` | 1 | opus-mt-es-wls |
| `2D: [1, 61702]` | 1 | opus-mt-es-xh |
| `2D: [1, 59716]` | 1 | opus-mt-es-yo |
| `2D: [1, 46028]` | 1 | opus-mt-es-yua |
| `2D: [1, 48431]` | 1 | opus-mt-es-zai |
| `2D: [1, 58065]` | 1 | opus-mt-et-fr |
| `2D: [1, 62898]` | 1 | opus-mt-et-ru |
| `2D: [1, 57792]` | 1 | opus-mt-et-sv |
| `2D: [1, 58429]` | 1 | opus-mt-eu-en |
| `2D: [1, 7677]` | 1 | opus-mt-eu-ru |
| `2D: [1, 62933]` | 1 | opus-mt-fi-ZH |
| `2D: [1, 59471]` | 1 | opus-mt-fi-de |
| `2D: [1, 59659]` | 1 | opus-mt-fi-en |
| `2D: [1, 33007]` | 1 | opus-mt-fi-fi |
| `2D: [1, 61090]` | 1 | opus-mt-fi-fj |
| `2D: [1, 60065]` | 1 | opus-mt-fi-fr |
| `2D: [1, 17131]` | 1 | opus-mt-fi-fse |
| `2D: [1, 61586]` | 1 | opus-mt-fi-gaa |
| `2D: [1, 54432]` | 1 | opus-mt-fi-gil |
| `2D: [1, 55617]` | 1 | opus-mt-fi-guw |
| `2D: [1, 57208]` | 1 | opus-mt-fi-ha |
| `2D: [1, 63477]` | 1 | opus-mt-fi-he |
| `2D: [1, 61066]` | 1 | opus-mt-fi-hil |
| `2D: [1, 46197]` | 1 | opus-mt-fi-ho |
| `2D: [1, 59394]` | 1 | opus-mt-fi-hr |
| `2D: [1, 49235]` | 1 | opus-mt-fi-ht |
| `2D: [1, 59526]` | 1 | opus-mt-fi-hu |
| `2D: [1, 61272]` | 1 | opus-mt-fi-ig |
| `2D: [1, 61080]` | 1 | opus-mt-fi-ilo |
| `2D: [1, 60259]` | 1 | opus-mt-fi-is |
| `2D: [1, 48544]` | 1 | opus-mt-fi-iso |
| `2D: [1, 59909]` | 1 | opus-mt-fi-it |
| `2D: [1, 50740]` | 1 | opus-mt-fi-kg |
| `2D: [1, 62236]` | 1 | opus-mt-fi-kqn |
| `2D: [1, 62274]` | 1 | opus-mt-fi-lg |
| `2D: [1, 62427]` | 1 | opus-mt-fi-lu |
| `2D: [1, 61858]` | 1 | opus-mt-fi-lua |
| `2D: [1, 62338]` | 1 | opus-mt-fi-lue |
| `2D: [1, 53773]` | 1 | opus-mt-fi-lus |
| `2D: [1, 59460]` | 1 | opus-mt-fi-lv |
| `2D: [1, 54138]` | 1 | opus-mt-fi-mfe |
| `2D: [1, 60329]` | 1 | opus-mt-fi-mg |
| `2D: [1, 54810]` | 1 | opus-mt-fi-mh |
| `2D: [1, 62563]` | 1 | opus-mt-fi-mk |
| `2D: [1, 50438]` | 1 | opus-mt-fi-mos |
| `2D: [1, 59039]` | 1 | opus-mt-fi-mt |
| `2D: [1, 45655]` | 1 | opus-mt-fi-niu |
| `2D: [1, 61423]` | 1 | opus-mt-fi-nso |
| `2D: [1, 58474]` | 1 | opus-mt-tc-bible-big-aav-fra_ita_por_spa |
| `2D: [1, 61448]` | 1 | opus-mt-tc-bible-big-afa-deu_eng_fra_por_spa |
| `2D: [1, 61623]` | 1 | opus-mt-tc-bible-big-afa-deu_eng_nld |
| `2D: [1, 61871]` | 1 | opus-mt-tc-bible-big-afa-fra_ita_por_spa |
| `2D: [1, 61872]` | 1 | opus-mt-tc-bible-big-alv-deu_eng_fra_por_spa |
| `2D: [1, 59467]` | 1 | opus-mt-tc-bible-big-bat-deu_eng_fra_por_spa |
| `2D: [1, 59091]` | 1 | opus-mt-tc-bible-big-bat-deu_eng_nld |
| `2D: [1, 58702]` | 1 | opus-mt-tc-bible-big-bat-en |
| `2D: [1, 61612]` | 1 | opus-mt-tc-bible-big-bnt-deu_eng_fra_por_spa |
| `2D: [1, 56599]` | 1 | opus-mt-tc-bible-big-cel-deu_eng_fra_por_spa |
| `2D: [1, 57539]` | 1 | opus-mt-tc-bible-big-deu_eng_fra_por_spa-aav |
| `2D: [1, 61815]` | 1 | opus-mt-tc-bible-big-deu_eng_fra_por_spa-afa |
| `2D: [1, 59473]` | 1 | opus-mt-tc-bible-big-deu_eng_fra_por_spa-bat |
| `2D: [1, 62297]` | 1 | opus-mt-tc-bible-big-deu_eng_fra_por_spa-bnt |
| `2D: [1, 59392]` | 1 | opus-mt-tc-bible-big-deu_eng_fra_por_spa-fiu |
| `2D: [1, 49013]` | 1 | opus-mt-tc-bible-big-deu_eng_fra_por_spa-gem |
| `2D: [1, 56248]` | 1 | opus-mt-tc-bible-big-deu_eng_fra_por_spa-gmq |
| `2D: [1, 48183]` | 1 | opus-mt-tc-bible-big-deu_eng_fra_por_spa-gmw |
| `2D: [1, 62090]` | 1 | opus-mt-tc-bible-big-deu_eng_fra_por_spa-iir |
| `2D: [1, 61906]` | 1 | opus-mt-tc-bible-big-deu_eng_fra_por_spa-inc |
| `2D: [1, 53305]` | 1 | opus-mt-tc-bible-big-deu_eng_fra_por_spa-ine |
| `2D: [1, 45448]` | 1 | opus-mt-tc-bible-big-deu_eng_fra_por_spa-itc |
| `2D: [1, 57129]` | 1 | opus-mt-tc-bible-big-deu_eng_fra_por_spa-mkh |
| `2D: [1, 62959]` | 1 | opus-mt-tc-bible-big-deu_eng_fra_por_spa-mul |
| `2D: [1, 59520]` | 1 | opus-mt-tc-bible-big-deu_eng_fra_por_spa-pqw |
| `2D: [1, 45376]` | 1 | opus-mt-tc-bible-big-deu_eng_fra_por_spa-roa |
| `2D: [1, 61339]` | 1 | opus-mt-tc-bible-big-deu_eng_fra_por_spa-sem |
| `2D: [1, 59956]` | 1 | opus-mt-tc-bible-big-deu_eng_fra_por_spa-sla |
| `2D: [1, 60983]` | 1 | opus-mt-tc-bible-big-deu_eng_fra_por_spa-trk |
| `2D: [1, 59384]` | 1 | opus-mt-tc-bible-big-deu_eng_fra_por_spa-urj |
| `2D: [1, 62489]` | 1 | opus-mt-tc-bible-big-dra-deu_eng_nld |
| `2D: [1, 62506]` | 1 | opus-mt-tc-bible-big-dra-en |
| `2D: [1, 59382]` | 1 | opus-mt-tc-bible-big-fiu-deu_eng_fra_por_spa |
| `2D: [1, 58747]` | 1 | opus-mt-tc-bible-big-fiu-en |
| `2D: [1, 59780]` | 1 | opus-mt-tc-bible-big-fiu-fra_ita_por_spa |
| `2D: [1, 48859]` | 1 | opus-mt-tc-bible-big-gem-deu_eng_fra_por_spa |
| `2D: [1, 56813]` | 1 | opus-mt-tc-bible-big-gem-fra_ita_por_spa |
| `2D: [1, 56160]` | 1 | opus-mt-tc-bible-big-gmq-deu_eng_fra_por_spa |
| `2D: [1, 55007]` | 1 | opus-mt-tc-bible-big-gmq-en |
| `2D: [1, 47963]` | 1 | opus-mt-tc-bible-big-gmw-deu_eng_fra_por_spa |
| `2D: [1, 39447]` | 1 | opus-mt-tc-bible-big-gmw-deu_eng_nld |
| `2D: [1, 54702]` | 1 | opus-mt-tc-bible-big-gmw-en |
| `2D: [1, 56346]` | 1 | opus-mt-tc-bible-big-gmw-fra_ita_por_spa |
| `2D: [1, 61807]` | 1 | opus-mt-tc-bible-big-iir-deu_eng_fra_por_spa |
| `2D: [1, 62113]` | 1 | opus-mt-tc-bible-big-iir-en |
| `2D: [1, 61705]` | 1 | opus-mt-tc-bible-big-inc-deu_eng_fra_por_spa |
| `2D: [1, 62164]` | 1 | opus-mt-tc-bible-big-inc-deu_eng_nld |
| `2D: [1, 62026]` | 1 | opus-mt-tc-bible-big-inc-en |
| `2D: [1, 52781]` | 1 | opus-mt-tc-bible-big-ine-deu_eng_fra_por_spa |
| `2D: [1, 55209]` | 1 | opus-mt-tc-bible-big-ine-deu_eng_nld |
| `2D: [1, 60956]` | 1 | opus-mt-tc-bible-big-ira-deu_eng_fra_por_spa |
| `2D: [1, 45341]` | 1 | opus-mt-tc-bible-big-itc-deu_eng_fra_por_spa |
| `2D: [1, 56278]` | 1 | opus-mt-tc-bible-big-itc-deu_eng_nld |
| `2D: [1, 39825]` | 1 | opus-mt-tc-bible-big-itc-fra_ita_por_spa |
| `2D: [1, 59740]` | 1 | opus-mt-tc-bible-big-map-en |
| `2D: [1, 60885]` | 1 | opus-mt-tc-bible-big-map-fra_ita_por_spa |
| `2D: [1, 55468]` | 1 | opus-mt-tc-bible-big-mkh-deu_eng_nld |
| `2D: [1, 58241]` | 1 | opus-mt-tc-bible-big-mkh-fra_ita_por_spa |
| `2D: [1, 57305]` | 1 | opus-mt-tc-bible-big-mul-deu_eng_fra_por_spa |
| `2D: [1, 58434]` | 1 | opus-mt-tc-bible-big-mul-deu_eng_nld |
| `2D: [1, 69667]` | 1 | opus-mt-tc-bible-big-mul-mul |
| `2D: [1, 59760]` | 1 | opus-mt-tc-bible-big-phi-deu_eng_fra_por_spa |
| `2D: [1, 57576]` | 1 | opus-mt-tc-bible-big-phi-en |
| `2D: [1, 60414]` | 1 | opus-mt-tc-bible-big-poz-deu_eng_fra_por_spa |
| `2D: [1, 59757]` | 1 | opus-mt-tc-bible-big-poz-en |
| `2D: [1, 60868]` | 1 | opus-mt-tc-bible-big-poz-fra_ita_por_spa |
| `2D: [1, 61710]` | 1 | opus-mt-tc-bible-big-pqe-deu_eng_fra_por_spa |
| `2D: [1, 59229]` | 1 | opus-mt-tc-bible-big-pqw-deu_eng_fra_por_spa |
| `2D: [1, 58311]` | 1 | opus-mt-tc-bible-big-pqw-en |
| `2D: [1, 60729]` | 1 | opus-mt-tc-bible-big-pqw-fra_ita_por_spa |
| `2D: [1, 45245]` | 1 | opus-mt-tc-bible-big-roa-deu_eng_fra_por_spa |
| `2D: [1, 55523]` | 1 | opus-mt-tc-bible-big-roa-en |
| `2D: [1, 61171]` | 1 | opus-mt-tc-bible-big-sem-deu_eng_fra_por_spa |
| `2D: [1, 61309]` | 1 | opus-mt-tc-bible-big-sem-deu_eng_nld |
| `2D: [1, 61023]` | 1 | opus-mt-tc-bible-big-sem-en |
| `2D: [1, 59873]` | 1 | opus-mt-tc-bible-big-sla-deu_eng_nld |
| `2D: [1, 59891]` | 1 | opus-mt-tc-bible-big-sla-en |
| `2D: [1, 58838]` | 1 | opus-mt-tc-bible-big-tai-deu_eng_fra_por_spa |
| `2D: [1, 60942]` | 1 | opus-mt-tc-bible-big-trk-deu_eng_fra_por_spa |
| `2D: [1, 59366]` | 1 | opus-mt-tc-bible-big-urj-deu_eng_fra_por_spa |
| `2D: [1, 59759]` | 1 | opus-mt-tc-bible-big-urj-fra_ita_por_spa |
| `2D: [1, 59178]` | 1 | opus-mt-tc-bible-big-zhx-deu_eng_fra_por_spa |
| `2D: [1, 58592]` | 1 | opus-mt-tc-bible-big-zhx-en |
| `3D: [1, 19, 2560]` | 1 | phi-2 |
| `2D: [1, 160000]` | 1 | pipecat-ai_smart-turn-v2 |
| `3D: [1, 2048, 768]` | 1 | pix2struct-base |
| `2D: [1, 2048]` | 1 | pix2struct-base |
| `3D: [1, 1125, 192]` | 1 | polejowska_yolos-tiny-CD45RB-1000-att |
| `2D: [2, 4]` | 1 | sadhaklal_mlp-iris |
| `2D: [4, 256]` | 1 | tsime_detr_mapilary |
| `4D: [1, 96, 128, 128]` | 1 | upernet-convnext-tiny |
| `4D: [1, 192, 64, 64]` | 1 | upernet-convnext-tiny |
| `4D: [1, 384, 32, 32]` | 1 | upernet-convnext-tiny |
| `4D: [1, 768, 16, 16]` | 1 | upernet-convnext-tiny |

### ultralytics (89/89 个模型有输入)

| Shape 模式 | 出现次数 | 示例模型 |
|-----------|----------|----------|
| `4D: [1, 3, 640, 640]` | 89 | yolo11l, yolo11l-cls, yolo11l-obb ...等89个 |

### workspace (1/1 个模型有输入)

| Shape 模式 | 出现次数 | 示例模型 |
|-----------|----------|----------|
| `4D: [1, 3, 224, 224]` | 1 | resnet18_test |

## 4. 4D 图像输入详细分析

| (C, H, W) | 数量 | 示例来源 |
|-----------|------|----------|
| (3, 224, 224) | 757 | timm, torchvision, transformers-auto-model, workspace |
| (3, 512, 512) | 127 | mmseg, transformers-auto-model |
| (3, 640, 640) | 95 | mmseg, transformers-auto-model, ultralytics |
| (3, 256, 192) | 60 | mmpose |
| (3, 256, 256) | 37 | mmpose, timm, transformers-auto-model |
| (3, 384, 384) | 37 | timm, transformers-auto-model |
| (3, 800, 800) | 21 | transformers-auto-model |
| (3, 30, 30) | 13 | transformers-auto-model |
| (3, 32, 32) | 10 | transformers-auto-model |
| (3, 448, 448) | 8 | timm, transformers-auto-model |
| (3, 512, 1024) | 7 | mmseg |
| (3, 336, 336) | 6 | timm, transformers-auto-model |
| (3, 1024, 1024) | 6 | transformers-auto-model |
| (3, 64, 64) | 6 | transformers-auto-model |
| (3, 768, 768) | 5 | timm, transformers-auto-model |
| (3, 320, 320) | 2 | timm |
| (6, 900, 256) | 2 | transformers-auto-model |
| (6, 900, 4) | 2 | transformers-auto-model |
| (3, 960, 960) | 2 | transformers-auto-model |
| (3, 1008, 1008) | 2 | transformers-auto-model |
| (3, 600, 600) | 2 | transformers-auto-model |
| (256, 128, 128) | 1 | mmseg |
| (256, 16, 16) | 1 | mmseg |
| (256, 32, 32) | 1 | mmseg |
| (256, 64, 64) | 1 | mmseg |
| (128, 256, 2) | 1 | others |
| (3, 128, 256) | 1 | others |
| (3, 192, 192) | 1 | timm |
| (3, 416, 416) | 1 | timm |
| (3, 352, 352) | 1 | timm |
| (3, 299, 299) | 1 | timm |
| (64, 64, 768) | 1 | timm |
| (64, 64, 1280) | 1 | timm |
| (64, 64, 1024) | 1 | timm |
| (16, 16, 384) | 1 | timm |
| (3, 112, 112) | 1 | transformers-auto-model |
| (96, 64, 64) | 1 | transformers-auto-model |
| (192, 32, 32) | 1 | transformers-auto-model |
| (384, 16, 16) | 1 | transformers-auto-model |
| (768, 8, 8) | 1 | transformers-auto-model |
| (3, 1536, 1536) | 1 | transformers-auto-model |
| (3, 840, 840) | 1 | transformers-auto-model |
| (3, 518, 518) | 1 | transformers-auto-model |
| (96, 128, 128) | 1 | transformers-auto-model |
| (192, 64, 64) | 1 | transformers-auto-model |
| (384, 32, 32) | 1 | transformers-auto-model |
| (768, 16, 16) | 1 | transformers-auto-model |

## 5. 2D NLP 输入详细分析

| Sequence Length | 数量 | 示例来源 |
|----------------|------|----------|
| 1 | 613 | transformers-auto-model |
| 128 | 338 | nemo, torchgeometric, transformers-auto-model |
| 64 | 242 | nemo, transformers-auto-model |
| 20 | 224 | transformers-auto-model |
| 21 | 189 | transformers-auto-model |
| 11 | 140 | transformers-auto-model |
| 36 | 124 | nemo, transformers-auto-model |
| 40 | 84 | transformers-auto-model |
| 41 | 82 | transformers-auto-model |
| 39 | 70 | transformers-auto-model |
| 37 | 70 | transformers-auto-model |
| 22 | 66 | transformers-auto-model |
| 44 | 64 | nemo, transformers-auto-model |
| 7 | 61 | transformers-auto-model |
| 12 | 61 | transformers-auto-model |
| 34 | 61 | transformers-auto-model |
| 35 | 60 | transformers-auto-model |
| 17 | 59 | transformers-auto-model |
| 19 | 57 | transformers-auto-model |
| 10 | 55 | transformers-auto-model |
| 18 | 54 | transformers-auto-model |
| 49 | 50 | nemo, transformers-auto-model |
| 13 | 50 | transformers-auto-model |
| 38 | 50 | transformers-auto-model |
| 45 | 48 | transformers-auto-model |
| 32 | 46 | torchgeometric, transformers-auto-model |
| 80 | 44 | nemo |
| 33 | 44 | transformers-auto-model |
| 96 | 40 | nemo |
| 16 | 38 | transformers-auto-model |
| 42 | 36 | transformers-auto-model |
| 80000 | 30 | torchaudio, transformers-auto-model |
| 15 | 28 | transformers-auto-model |
| 27 | 26 | transformers-auto-model |
| 46 | 26 | transformers-auto-model |
| 43 | 22 | transformers-auto-model |
| 26 | 19 | transformers-auto-model |
| 29 | 18 | transformers-auto-model |
| 31 | 18 | transformers-auto-model |
| 23 | 17 | transformers-auto-model |
| 47 | 16 | transformers-auto-model |
| 51 | 15 | transformers-auto-model |
| 14 | 14 | transformers-auto-model |
| 4 | 14 | transformers-auto-model |
| 24 | 13 | transformers-auto-model |
| 8 | 12 | transformers-auto-model |
| 48 | 11 | transformers-auto-model |
| 100 | 10 | torchgeometric |
| 50 | 9 | transformers-auto-model |
| 512 | 8 | transformers-auto-model |
| 30 | 8 | transformers-auto-model |
| 25 | 7 | transformers-auto-model |
| 9 | 7 | transformers-auto-model |
| 53 | 6 | transformers-auto-model |
| 52 | 6 | transformers-auto-model |
| 55 | 4 | transformers-auto-model |
| 59836 | 4 | transformers-auto-model |
| 28 | 4 | transformers-auto-model |
| 256 | 3 | transformers-auto-model |
| 82 | 3 | transformers-auto-model |
| 65001 | 3 | transformers-auto-model |
| 16000 | 2 | torchaudio, transformers-auto-model |
| 64000 | 2 | torchaudio |
| 57445 | 2 | transformers-auto-model |
| 7475 | 2 | transformers-auto-model |
| 60216 | 2 | transformers-auto-model |
| 60657 | 2 | transformers-auto-model |
| 61180 | 2 | transformers-auto-model |
| 56 | 2 | transformers-auto-model |
| 53676 | 2 | transformers-auto-model |
| 53502 | 2 | transformers-auto-model |
| 60738 | 2 | transformers-auto-model |
| 58455 | 2 | transformers-auto-model |
| 59736 | 2 | transformers-auto-model |
| 61739 | 2 | transformers-auto-model |
| 60 | 2 | transformers-auto-model |
| 59828 | 2 | transformers-auto-model |
| 61763 | 2 | transformers-auto-model |
| 36048 | 2 | transformers-auto-model |
| 13296 | 2 | transformers-auto-model |
| 61110 | 2 | transformers-auto-model |
| 66 | 2 | transformers-auto-model |
| 7770 | 2 | transformers-auto-model |
| 61845 | 2 | transformers-auto-model |
| 60069 | 2 | transformers-auto-model |
| 59345 | 2 | transformers-auto-model |
| 54321 | 2 | transformers-auto-model |
| 56412 | 2 | transformers-auto-model |
| 47919 | 2 | transformers-auto-model |
| 51421 | 2 | transformers-auto-model |
| 59433 | 2 | transformers-auto-model |
| 22116 | 2 | transformers-auto-model |
| 55255 | 2 | transformers-auto-model |
| 57 | 2 | transformers-auto-model |
| 61 | 2 | transformers-auto-model |
| 59324 | 2 | transformers-auto-model |
| 61010 | 2 | transformers-auto-model |
| 59183 | 2 | transformers-auto-model |
| 56327 | 2 | transformers-auto-model |
| 47285 | 2 | transformers-auto-model |
| 43620 | 2 | transformers-auto-model |
| 47319 | 2 | transformers-auto-model |
| 48214 | 2 | transformers-auto-model |
| 58234 | 2 | transformers-auto-model |
| 62509 | 2 | transformers-auto-model |
| 7397 | 2 | transformers-auto-model |
| 59012 | 2 | transformers-auto-model |
| 11802 | 2 | transformers-auto-model |
| 54395 | 2 | transformers-auto-model |
| 57027 | 2 | transformers-auto-model |
| 58930 | 2 | transformers-auto-model |
| 7286 | 2 | transformers-auto-model |
| 57584 | 2 | transformers-auto-model |
| 59299 | 2 | transformers-auto-model |
| 60346 | 2 | transformers-auto-model |
| 59733 | 2 | transformers-auto-model |
| 58101 | 2 | transformers-auto-model |
| 61301 | 2 | transformers-auto-model |
| 58042 | 2 | transformers-auto-model |
| 22170 | 2 | transformers-auto-model |
| 61153 | 2 | transformers-auto-model |
| 60025 | 2 | transformers-auto-model |
| 57579 | 2 | transformers-auto-model |
| 60163 | 2 | transformers-auto-model |
| 59678 | 2 | transformers-auto-model |
| 55090 | 2 | transformers-auto-model |
| 61193 | 2 | transformers-auto-model |
| 7713 | 2 | transformers-auto-model |
| 59727 | 2 | transformers-auto-model |
| 59197 | 2 | transformers-auto-model |
| 58866 | 2 | transformers-auto-model |
| 21964 | 2 | transformers-auto-model |
| 62518 | 2 | transformers-auto-model |
| 58379 | 2 | transformers-auto-model |
| 7439 | 2 | transformers-auto-model |
| 7347 | 2 | transformers-auto-model |
| 58124 | 2 | transformers-auto-model |
| 58060 | 2 | transformers-auto-model |
| 63261 | 2 | transformers-auto-model |
| 61389 | 2 | transformers-auto-model |
| 58055 | 2 | transformers-auto-model |
| 61529 | 2 | transformers-auto-model |
| 6 | 2 | transformers-auto-model |
| 5 | 2 | transformers-auto-model |
| 2880 | 2 | transformers-auto-model |
| 500 | 1 | torchgeometric |
| 1000 | 1 | torchgeometric |
| 2 | 1 | transformers-auto-model |
| 220500 | 1 | transformers-auto-model |
| 768 | 1 | transformers-auto-model |
| 58012 | 1 | transformers-auto-model |
| 47231 | 1 | transformers-auto-model |
| 56039 | 1 | transformers-auto-model |
| 23625 | 1 | transformers-auto-model |
| 58757 | 1 | transformers-auto-model |
| 59422 | 1 | transformers-auto-model |
| 53247 | 1 | transformers-auto-model |
| 62842 | 1 | transformers-auto-model |
| 58463 | 1 | transformers-auto-model |
| 33714 | 1 | transformers-auto-model |
| 61549 | 1 | transformers-auto-model |
| 63199 | 1 | transformers-auto-model |
| 63079 | 1 | transformers-auto-model |
| 63336 | 1 | transformers-auto-model |
| 62834 | 1 | transformers-auto-model |
| 7741 | 1 | transformers-auto-model |
| 62526 | 1 | transformers-auto-model |
| 64505 | 1 | transformers-auto-model |
| 63334 | 1 | transformers-auto-model |
| 63294 | 1 | transformers-auto-model |
| 62606 | 1 | transformers-auto-model |
| 63272 | 1 | transformers-auto-model |
| 59131 | 1 | transformers-auto-model |
| 24831 | 1 | transformers-auto-model |
| 50895 | 1 | transformers-auto-model |
| 48742 | 1 | transformers-auto-model |
| 7590 | 1 | transformers-auto-model |
| 6904 | 1 | transformers-auto-model |
| 59056 | 1 | transformers-auto-model |
| 60484 | 1 | transformers-auto-model |
| 59911 | 1 | transformers-auto-model |
| 7630 | 1 | transformers-auto-model |
| 60996 | 1 | transformers-auto-model |
| 61027 | 1 | transformers-auto-model |
| 60613 | 1 | transformers-auto-model |
| 20994 | 1 | transformers-auto-model |
| 61813 | 1 | transformers-auto-model |
| 61483 | 1 | transformers-auto-model |
| 61468 | 1 | transformers-auto-model |
| 57310 | 1 | transformers-auto-model |
| 63138 | 1 | transformers-auto-model |
| 59878 | 1 | transformers-auto-model |
| 57508 | 1 | transformers-auto-model |
| 55334 | 1 | transformers-auto-model |
| 63597 | 1 | transformers-auto-model |
| 61729 | 1 | transformers-auto-model |
| 54231 | 1 | transformers-auto-model |
| 51381 | 1 | transformers-auto-model |
| 49621 | 1 | transformers-auto-model |
| 21410 | 1 | transformers-auto-model |
| 21528 | 1 | transformers-auto-model |
| 21976 | 1 | transformers-auto-model |
| 20554 | 1 | transformers-auto-model |
| 7545 | 1 | transformers-auto-model |
| 62393 | 1 | transformers-auto-model |
| 23173 | 1 | transformers-auto-model |
| 57239 | 1 | transformers-auto-model |
| 59770 | 1 | transformers-auto-model |
| 57907 | 1 | transformers-auto-model |
| 60011 | 1 | transformers-auto-model |
| 59610 | 1 | transformers-auto-model |
| 58904 | 1 | transformers-auto-model |
| 57679 | 1 | transformers-auto-model |
| 31989 | 1 | transformers-auto-model |
| 57790 | 1 | transformers-auto-model |
| 46774 | 1 | transformers-auto-model |
| 46013 | 1 | transformers-auto-model |
| 58338 | 1 | transformers-auto-model |
| 57482 | 1 | transformers-auto-model |
| 62778 | 1 | transformers-auto-model |
| 28047 | 1 | transformers-auto-model |
| 59606 | 1 | transformers-auto-model |
| 57138 | 1 | transformers-auto-model |
| 18881 | 1 | transformers-auto-model |
| 62922 | 1 | transformers-auto-model |
| 61916 | 1 | transformers-auto-model |
| 63077 | 1 | transformers-auto-model |
| 56367 | 1 | transformers-auto-model |
| 58819 | 1 | transformers-auto-model |
| 33217 | 1 | transformers-auto-model |
| 59951 | 1 | transformers-auto-model |
| 59971 | 1 | transformers-auto-model |
| 59804 | 1 | transformers-auto-model |
| 60424 | 1 | transformers-auto-model |
| 53500 | 1 | transformers-auto-model |
| 54544 | 1 | transformers-auto-model |
| 59664 | 1 | transformers-auto-model |
| 63066 | 1 | transformers-auto-model |
| 60041 | 1 | transformers-auto-model |
| 45292 | 1 | transformers-auto-model |
| 58701 | 1 | transformers-auto-model |
| 49000 | 1 | transformers-auto-model |
| 58671 | 1 | transformers-auto-model |
| 60194 | 1 | transformers-auto-model |
| 59315 | 1 | transformers-auto-model |
| 47660 | 1 | transformers-auto-model |
| 59867 | 1 | transformers-auto-model |
| 49964 | 1 | transformers-auto-model |
| 60592 | 1 | transformers-auto-model |
| 60846 | 1 | transformers-auto-model |
| 58862 | 1 | transformers-auto-model |
| 60920 | 1 | transformers-auto-model |
| 57716 | 1 | transformers-auto-model |
| 57955 | 1 | transformers-auto-model |
| 44948 | 1 | transformers-auto-model |
| 57567 | 1 | transformers-auto-model |
| 7020 | 1 | transformers-auto-model |
| 60441 | 1 | transformers-auto-model |
| 60883 | 1 | transformers-auto-model |
| 60768 | 1 | transformers-auto-model |
| 60181 | 1 | transformers-auto-model |
| 48977 | 1 | transformers-auto-model |
| 60431 | 1 | transformers-auto-model |
| 56761 | 1 | transformers-auto-model |
| 62523 | 1 | transformers-auto-model |
| 58225 | 1 | transformers-auto-model |
| 62943 | 1 | transformers-auto-model |
| 61449 | 1 | transformers-auto-model |
| 60218 | 1 | transformers-auto-model |
| 59481 | 1 | transformers-auto-model |
| 59076 | 1 | transformers-auto-model |
| 63305 | 1 | transformers-auto-model |
| 60763 | 1 | transformers-auto-model |
| 9001 | 1 | transformers-auto-model |
| 56071 | 1 | transformers-auto-model |
| 61199 | 1 | transformers-auto-model |
| 61577 | 1 | transformers-auto-model |
| 62802 | 1 | transformers-auto-model |
| 23288 | 1 | transformers-auto-model |
| 59061 | 1 | transformers-auto-model |
| 61708 | 1 | transformers-auto-model |
| 61740 | 1 | transformers-auto-model |
| 56824 | 1 | transformers-auto-model |
| 57912 | 1 | transformers-auto-model |
| 57681 | 1 | transformers-auto-model |
| 57766 | 1 | transformers-auto-model |
| 62952 | 1 | transformers-auto-model |
| 64826 | 1 | transformers-auto-model |
| 58286 | 1 | transformers-auto-model |
| 58938 | 1 | transformers-auto-model |
| 55835 | 1 | transformers-auto-model |
| 56683 | 1 | transformers-auto-model |
| 57186 | 1 | transformers-auto-model |
| 56629 | 1 | transformers-auto-model |
| 50021 | 1 | transformers-auto-model |
| 55458 | 1 | transformers-auto-model |
| 57068 | 1 | transformers-auto-model |
| 55477 | 1 | transformers-auto-model |
| 52253 | 1 | transformers-auto-model |
| 12339 | 1 | transformers-auto-model |
| 58089 | 1 | transformers-auto-model |
| 65839 | 1 | transformers-auto-model |
| 61950 | 1 | transformers-auto-model |
| 56301 | 1 | transformers-auto-model |
| 42463 | 1 | transformers-auto-model |
| 47809 | 1 | transformers-auto-model |
| 62522 | 1 | transformers-auto-model |
| 7754 | 1 | transformers-auto-model |
| 54796 | 1 | transformers-auto-model |
| 56127 | 1 | transformers-auto-model |
| 62228 | 1 | transformers-auto-model |
| 57777 | 1 | transformers-auto-model |
| 61760 | 1 | transformers-auto-model |
| 61161 | 1 | transformers-auto-model |
| 58647 | 1 | transformers-auto-model |
| 45029 | 1 | transformers-auto-model |
| 80035 | 1 | transformers-auto-model |
| 57148 | 1 | transformers-auto-model |
| 46276 | 1 | transformers-auto-model |
| 47322 | 1 | transformers-auto-model |
| 50183 | 1 | transformers-auto-model |
| 60142 | 1 | transformers-auto-model |
| 50346 | 1 | transformers-auto-model |
| 58115 | 1 | transformers-auto-model |
| 60447 | 1 | transformers-auto-model |
| 58314 | 1 | transformers-auto-model |
| 57974 | 1 | transformers-auto-model |
| 60241 | 1 | transformers-auto-model |
| 59447 | 1 | transformers-auto-model |
| 61313 | 1 | transformers-auto-model |
| 56530 | 1 | transformers-auto-model |
| 52236 | 1 | transformers-auto-model |
| 49567 | 1 | transformers-auto-model |
| 60088 | 1 | transformers-auto-model |
| 48883 | 1 | transformers-auto-model |
| 54688 | 1 | transformers-auto-model |
| 49657 | 1 | transformers-auto-model |
| 61946 | 1 | transformers-auto-model |
| 56075 | 1 | transformers-auto-model |
| 24661 | 1 | transformers-auto-model |
| 47022 | 1 | transformers-auto-model |
| 61674 | 1 | transformers-auto-model |
| 56041 | 1 | transformers-auto-model |
| 64110 | 1 | transformers-auto-model |
| 49883 | 1 | transformers-auto-model |
| 61556 | 1 | transformers-auto-model |
| 42155 | 1 | transformers-auto-model |
| 67028 | 1 | transformers-auto-model |
| 57542 | 1 | transformers-auto-model |
| 59811 | 1 | transformers-auto-model |
| 52867 | 1 | transformers-auto-model |
| 59898 | 1 | transformers-auto-model |
| 57487 | 1 | transformers-auto-model |
| 57294 | 1 | transformers-auto-model |
| 59866 | 1 | transformers-auto-model |
| 42010 | 1 | transformers-auto-model |
| 52997 | 1 | transformers-auto-model |
| 60175 | 1 | transformers-auto-model |
| 60089 | 1 | transformers-auto-model |
| 58849 | 1 | transformers-auto-model |
| 7528 | 1 | transformers-auto-model |
| 53832 | 1 | transformers-auto-model |
| 59543 | 1 | transformers-auto-model |
| 56671 | 1 | transformers-auto-model |
| 61335 | 1 | transformers-auto-model |
| 60384 | 1 | transformers-auto-model |
| 55872 | 1 | transformers-auto-model |
| 43822 | 1 | transformers-auto-model |
| 59414 | 1 | transformers-auto-model |
| 58879 | 1 | transformers-auto-model |
| 54880 | 1 | transformers-auto-model |
| 59158 | 1 | transformers-auto-model |
| 55129 | 1 | transformers-auto-model |
| 57355 | 1 | transformers-auto-model |
| 56434 | 1 | transformers-auto-model |
| 58950 | 1 | transformers-auto-model |
| 58905 | 1 | transformers-auto-model |
| 34360 | 1 | transformers-auto-model |
| 43019 | 1 | transformers-auto-model |
| 57373 | 1 | transformers-auto-model |
| 57306 | 1 | transformers-auto-model |
| 57446 | 1 | transformers-auto-model |
| 61051 | 1 | transformers-auto-model |
| 49238 | 1 | transformers-auto-model |
| 57469 | 1 | transformers-auto-model |
| 61657 | 1 | transformers-auto-model |
| 38380 | 1 | transformers-auto-model |
| 57000 | 1 | transformers-auto-model |
| 56506 | 1 | transformers-auto-model |
| 61587 | 1 | transformers-auto-model |
| 58673 | 1 | transformers-auto-model |
| 62025 | 1 | transformers-auto-model |
| 58920 | 1 | transformers-auto-model |
| 53685 | 1 | transformers-auto-model |
| 61285 | 1 | transformers-auto-model |
| 61605 | 1 | transformers-auto-model |
| 57680 | 1 | transformers-auto-model |
| 58052 | 1 | transformers-auto-model |
| 59728 | 1 | transformers-auto-model |
| 7768 | 1 | transformers-auto-model |
| 7477 | 1 | transformers-auto-model |
| 7276 | 1 | transformers-auto-model |
| 7344 | 1 | transformers-auto-model |
| 7325 | 1 | transformers-auto-model |
| 7246 | 1 | transformers-auto-model |
| 7730 | 1 | transformers-auto-model |
| 7284 | 1 | transformers-auto-model |
| 58071 | 1 | transformers-auto-model |
| 51970 | 1 | transformers-auto-model |
| 58623 | 1 | transformers-auto-model |
| 59814 | 1 | transformers-auto-model |
| 60782 | 1 | transformers-auto-model |
| 33253 | 1 | transformers-auto-model |
| 74822 | 1 | transformers-auto-model |
| 60173 | 1 | transformers-auto-model |
| 53428 | 1 | transformers-auto-model |
| 5805 | 1 | transformers-auto-model |
| 54512 | 1 | transformers-auto-model |
| 59503 | 1 | transformers-auto-model |
| 63105 | 1 | transformers-auto-model |
| 59256 | 1 | transformers-auto-model |
| 45168 | 1 | transformers-auto-model |
| 58648 | 1 | transformers-auto-model |
| 49387 | 1 | transformers-auto-model |
| 57414 | 1 | transformers-auto-model |
| 60006 | 1 | transformers-auto-model |
| 59457 | 1 | transformers-auto-model |
| 60335 | 1 | transformers-auto-model |
| 47550 | 1 | transformers-auto-model |
| 49808 | 1 | transformers-auto-model |
| 60350 | 1 | transformers-auto-model |
| 60783 | 1 | transformers-auto-model |
| 59511 | 1 | transformers-auto-model |
| 60813 | 1 | transformers-auto-model |
| 53585 | 1 | transformers-auto-model |
| 28283 | 1 | transformers-auto-model |
| 62482 | 1 | transformers-auto-model |
| 57085 | 1 | transformers-auto-model |
| 45309 | 1 | transformers-auto-model |
| 58264 | 1 | transformers-auto-model |
| 22026 | 1 | transformers-auto-model |
| 60344 | 1 | transformers-auto-model |
| 60819 | 1 | transformers-auto-model |
| 60074 | 1 | transformers-auto-model |
| 56010 | 1 | transformers-auto-model |
| 49022 | 1 | transformers-auto-model |
| 56793 | 1 | transformers-auto-model |
| 10873 | 1 | transformers-auto-model |
| 7512 | 1 | transformers-auto-model |
| 56140 | 1 | transformers-auto-model |
| 63430 | 1 | transformers-auto-model |
| 61291 | 1 | transformers-auto-model |
| 46098 | 1 | transformers-auto-model |
| 57959 | 1 | transformers-auto-model |
| 58284 | 1 | transformers-auto-model |
| 61622 | 1 | transformers-auto-model |
| 60052 | 1 | transformers-auto-model |
| 60207 | 1 | transformers-auto-model |
| 60332 | 1 | transformers-auto-model |
| 59655 | 1 | transformers-auto-model |
| 60267 | 1 | transformers-auto-model |
| 58959 | 1 | transformers-auto-model |
| 52913 | 1 | transformers-auto-model |
| 42688 | 1 | transformers-auto-model |
| 60033 | 1 | transformers-auto-model |
| 58873 | 1 | transformers-auto-model |
| 54248 | 1 | transformers-auto-model |
| 62460 | 1 | transformers-auto-model |
| 56171 | 1 | transformers-auto-model |
| 58488 | 1 | transformers-auto-model |
| 59954 | 1 | transformers-auto-model |
| 41820 | 1 | transformers-auto-model |
| 61702 | 1 | transformers-auto-model |
| 59716 | 1 | transformers-auto-model |
| 46028 | 1 | transformers-auto-model |
| 48431 | 1 | transformers-auto-model |
| 58065 | 1 | transformers-auto-model |
| 62898 | 1 | transformers-auto-model |
| 57792 | 1 | transformers-auto-model |
| 58429 | 1 | transformers-auto-model |
| 7677 | 1 | transformers-auto-model |
| 62933 | 1 | transformers-auto-model |
| 59471 | 1 | transformers-auto-model |
| 59659 | 1 | transformers-auto-model |
| 33007 | 1 | transformers-auto-model |
| 61090 | 1 | transformers-auto-model |
| 60065 | 1 | transformers-auto-model |
| 17131 | 1 | transformers-auto-model |
| 61586 | 1 | transformers-auto-model |
| 54432 | 1 | transformers-auto-model |
| 55617 | 1 | transformers-auto-model |
| 57208 | 1 | transformers-auto-model |
| 63477 | 1 | transformers-auto-model |
| 61066 | 1 | transformers-auto-model |
| 46197 | 1 | transformers-auto-model |
| 59394 | 1 | transformers-auto-model |
| 49235 | 1 | transformers-auto-model |
| 59526 | 1 | transformers-auto-model |
| 61272 | 1 | transformers-auto-model |
| 61080 | 1 | transformers-auto-model |
| 60259 | 1 | transformers-auto-model |
| 48544 | 1 | transformers-auto-model |
| 59909 | 1 | transformers-auto-model |
| 50740 | 1 | transformers-auto-model |
| 62236 | 1 | transformers-auto-model |
| 62274 | 1 | transformers-auto-model |
| 62427 | 1 | transformers-auto-model |
| 61858 | 1 | transformers-auto-model |
| 62338 | 1 | transformers-auto-model |
| 53773 | 1 | transformers-auto-model |
| 59460 | 1 | transformers-auto-model |
| 54138 | 1 | transformers-auto-model |
| 60329 | 1 | transformers-auto-model |
| 54810 | 1 | transformers-auto-model |
| 62563 | 1 | transformers-auto-model |
| 50438 | 1 | transformers-auto-model |
| 59039 | 1 | transformers-auto-model |
| 45655 | 1 | transformers-auto-model |
| 61423 | 1 | transformers-auto-model |
| 58474 | 1 | transformers-auto-model |
| 61448 | 1 | transformers-auto-model |
| 61623 | 1 | transformers-auto-model |
| 61871 | 1 | transformers-auto-model |
| 61872 | 1 | transformers-auto-model |
| 59467 | 1 | transformers-auto-model |
| 59091 | 1 | transformers-auto-model |
| 58702 | 1 | transformers-auto-model |
| 61612 | 1 | transformers-auto-model |
| 56599 | 1 | transformers-auto-model |
| 57539 | 1 | transformers-auto-model |
| 61815 | 1 | transformers-auto-model |
| 59473 | 1 | transformers-auto-model |
| 62297 | 1 | transformers-auto-model |
| 59392 | 1 | transformers-auto-model |
| 49013 | 1 | transformers-auto-model |
| 56248 | 1 | transformers-auto-model |
| 48183 | 1 | transformers-auto-model |
| 62090 | 1 | transformers-auto-model |
| 61906 | 1 | transformers-auto-model |
| 53305 | 1 | transformers-auto-model |
| 45448 | 1 | transformers-auto-model |
| 57129 | 1 | transformers-auto-model |
| 62959 | 1 | transformers-auto-model |
| 59520 | 1 | transformers-auto-model |
| 45376 | 1 | transformers-auto-model |
| 61339 | 1 | transformers-auto-model |
| 59956 | 1 | transformers-auto-model |
| 60983 | 1 | transformers-auto-model |
| 59384 | 1 | transformers-auto-model |
| 62489 | 1 | transformers-auto-model |
| 62506 | 1 | transformers-auto-model |
| 59382 | 1 | transformers-auto-model |
| 58747 | 1 | transformers-auto-model |
| 59780 | 1 | transformers-auto-model |
| 48859 | 1 | transformers-auto-model |
| 56813 | 1 | transformers-auto-model |
| 56160 | 1 | transformers-auto-model |
| 55007 | 1 | transformers-auto-model |
| 47963 | 1 | transformers-auto-model |
| 39447 | 1 | transformers-auto-model |
| 54702 | 1 | transformers-auto-model |
| 56346 | 1 | transformers-auto-model |
| 61807 | 1 | transformers-auto-model |
| 62113 | 1 | transformers-auto-model |
| 61705 | 1 | transformers-auto-model |
| 62164 | 1 | transformers-auto-model |
| 62026 | 1 | transformers-auto-model |
| 52781 | 1 | transformers-auto-model |
| 55209 | 1 | transformers-auto-model |
| 60956 | 1 | transformers-auto-model |
| 45341 | 1 | transformers-auto-model |
| 56278 | 1 | transformers-auto-model |
| 39825 | 1 | transformers-auto-model |
| 59740 | 1 | transformers-auto-model |
| 60885 | 1 | transformers-auto-model |
| 55468 | 1 | transformers-auto-model |
| 58241 | 1 | transformers-auto-model |
| 57305 | 1 | transformers-auto-model |
| 58434 | 1 | transformers-auto-model |
| 69667 | 1 | transformers-auto-model |
| 59760 | 1 | transformers-auto-model |
| 57576 | 1 | transformers-auto-model |
| 60414 | 1 | transformers-auto-model |
| 59757 | 1 | transformers-auto-model |
| 60868 | 1 | transformers-auto-model |
| 61710 | 1 | transformers-auto-model |
| 59229 | 1 | transformers-auto-model |
| 58311 | 1 | transformers-auto-model |
| 60729 | 1 | transformers-auto-model |
| 45245 | 1 | transformers-auto-model |
| 55523 | 1 | transformers-auto-model |
| 61171 | 1 | transformers-auto-model |
| 61309 | 1 | transformers-auto-model |
| 61023 | 1 | transformers-auto-model |
| 59873 | 1 | transformers-auto-model |
| 59891 | 1 | transformers-auto-model |
| 58838 | 1 | transformers-auto-model |
| 60942 | 1 | transformers-auto-model |
| 59366 | 1 | transformers-auto-model |
| 59759 | 1 | transformers-auto-model |
| 59178 | 1 | transformers-auto-model |
| 58592 | 1 | transformers-auto-model |
| 160000 | 1 | transformers-auto-model |
| 2048 | 1 | transformers-auto-model |

## 6. 特殊输入模式

### 5D 视频输入

| 模型 | 来源 | Shape | dtype |
|------|------|-------|-------|
| mc3_18 | torchvision | `[1, 3, 16, 224, 224]` | torch.float32 |
| r3d_18 | torchvision | `[1, 3, 16, 224, 224]` | torch.float32 |
| JackWong0911_timesformer-base-finetuned-k400-finetuned-kinetic400-subset-epoch6-num_frame_10 | transformers-auto-model | `[1, 10, 3, 224, 224]` | torch.float32 |
| JackWong0911_vivit-b-16x2-kinetics400-finetuned-kinectic | transformers-auto-model | `[1, 10, 3, 224, 224]` | torch.float32 |
| MCG-NJU_videomae-base-finetuned-kinetics | transformers-auto-model | `[1, 16, 3, 224, 224]` | torch.float32 |
| MCG-NJU_videomae-base-short | transformers-auto-model | `[1, 16, 3, 224, 224]` | torch.float32 |
| MCG-NJU_videomae-large | transformers-auto-model | `[1, 16, 3, 224, 224]` | torch.float32 |
| MCG-NJU_videomae-large-finetuned-kinetics | transformers-auto-model | `[1, 16, 3, 224, 224]` | torch.float32 |
| facebook_timesformer-base-finetuned-k400 | transformers-auto-model | `[1, 8, 3, 224, 224]` | torch.float32 |
| fcakyon_timesformer-base-finetuned-k400 | transformers-auto-model | `[1, 16, 3, 224, 224]` | torch.float32 |
| fcakyon_timesformer-hr-finetuned-k400 | transformers-auto-model | `[1, 16, 3, 448, 448]` | torch.float32 |
| fcakyon_timesformer-large-finetuned-k400 | transformers-auto-model | `[1, 96, 3, 224, 224]` | torch.float32 |
| hf-tiny-model-private_tiny-random-TimesformerForVideoClassification | transformers-auto-model | `[1, 2, 3, 10, 10]` | torch.float32 |
| hf-tiny-model-private_tiny-random-VideoMAEForVideoClassification | transformers-auto-model | `[1, 2, 3, 10, 10]` | torch.float32 |
| microsoft_xclip-base-patch16 | transformers-auto-model | `[1, 8, 3, 224, 224]` | torch.float32 |
| microsoft_xclip-base-patch16-zero-shot | transformers-auto-model | `[1, 32, 3, 224, 224]` | torch.float32 |
| microsoft_xclip-base-patch32 | transformers-auto-model | `[1, 8, 3, 224, 224]` | torch.float32 |
| microsoft_xclip-base-patch32-16-frames | transformers-auto-model | `[1, 16, 3, 224, 224]` | torch.float32 |
| microsoft_xclip-large-patch14 | transformers-auto-model | `[1, 8, 3, 224, 224]` | torch.float32 |
| prathameshdalal_vivit-b-16x2-kinetics400-UCF-Crime | transformers-auto-model | `[1, 32, 3, 224, 224]` | torch.float32 |
| thegigasurgeon_mopping_224_32_frames_resampling_1_huge | transformers-auto-model | `[1, 32, 3, 224, 224]` | torch.float32 |

### 多输入模型（>1 个纯数据输入）

| 模型 | 来源 | 输入数 | 各输入 Shape |
|------|------|--------|-------------|
| CosyVoice-300M | cosyvoice | 3 | `[1, 1, 763]`, `[1, 763, 512]`, `[1, 1525, 512]` |
| CosyVoice2-0.5B | cosyvoice | 3 | `[1, 1, 796]`, `[1, 796, 512]`, `[1, 1591, 512]` |
| Mask2Former_R50 | mmseg | 4 | `[1, 256, 128, 128]`, `[1, 256, 16, 16]`, `[1, 256, 32, 32]`, `[1, 256, 64, 64]` |
| parakeet-ctc-0.6b | nemo | 51 | `[1, 80, 521]`, `[1]`, `[1, 9999, 1024]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]` |
| parakeet-ctc-1.1b | nemo | 87 | `[1, 80, 521]`, `[1]`, `[1, 9999, 1024]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]` |
| parakeet-tdt-0.6b-v3 | nemo | 51 | `[1, 128, 521]`, `[1]`, `[1, 9999, 1024]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]` |
| parakeet-tdt-1.1b | nemo | 87 | `[1, 80, 521]`, `[1]`, `[1, 9999, 1024]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]` |
| stt_en_conformer_ctc_large | nemo | 39 | `[1, 80, 521]`, `[1]`, `[1, 9999, 512]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]` |
| stt_en_conformer_ctc_medium | nemo | 39 | `[1, 80, 521]`, `[1]`, `[1, 9999, 256]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]` |
| stt_en_conformer_ctc_small | nemo | 35 | `[1, 80, 521]`, `[1]`, `[1, 9999, 176]`, `[4, 44]`, `[4, 44]`, `[4, 44]`, `[4, 44]`, `[4, 44]`, `[4, 44]`, `[4, 44]`, `[4, 44]`, `[4, 44]`, `[4, 44]`, `[4, 44]`, `[4, 44]`, `[4, 44]`, `[4, 44]`, `[4, 44]`, `[4, 44]`, `[4, 44]`, `[4, 44]`, `[4, 44]`, `[4, 44]`, `[4, 44]`, `[4, 44]`, `[4, 44]`, `[4, 44]`, `[4, 44]`, `[4, 44]`, `[4, 44]`, `[4, 44]`, `[4, 44]`, `[4, 44]`, `[4, 44]`, `[4, 44]` |
| stt_en_fastconformer_hybrid_large_pc | nemo | 37 | `[1, 80, 521]`, `[1]`, `[1, 9999, 512]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]` |
| stt_en_fastconformer_hybrid_large_streaming_multi | nemo | 37 | `[1, 80, 521]`, `[1]`, `[1, 9999, 512]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]` |
| stt_en_fastconformer_hybrid_medium_streaming_80ms_pc | nemo | 35 | `[1, 80, 521]`, `[1]`, `[1, 9999, 256]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]` |
| stt_en_squeezeformer_ctc_large_ls | nemo | 48 | `[1, 80, 521]`, `[1]`, `[1, 9999, 640]`, `[8, 80]`, `[8, 80]`, `[8, 80]`, `[8, 80]`, `[8, 80]`, `[8, 80]`, `[8, 80]`, `[8, 80]`, `[8, 80]`, `[8, 80]`, `[8, 80]`, `[8, 80]`, `[8, 80]`, `[8, 80]`, `[8, 80]`, `[8, 80]`, `[8, 80]`, `[8, 80]`, `[8, 80]`, `[8, 80]`, `[1, 9999, 640]`, `[8, 80]`, `[8, 80]`, `[8, 80]`, `[8, 80]`, `[8, 80]`, `[8, 80]`, `[8, 80]`, `[8, 80]`, `[8, 80]`, `[8, 80]`, `[8, 80]`, `[8, 80]`, `[8, 80]`, `[8, 80]`, `[8, 80]`, `[8, 80]`, `[8, 80]`, `[8, 80]`, `[8, 80]`, `[8, 80]`, `[8, 80]`, `[8, 80]`, `[8, 80]`, `[8, 80]` |
| stt_en_squeezeformer_ctc_medium_large_ls | nemo | 40 | `[1, 80, 521]`, `[1]`, `[1, 9999, 512]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[1, 9999, 512]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]`, `[8, 64]` |
| stt_en_squeezeformer_ctc_medium_ls | nemo | 44 | `[1, 80, 521]`, `[1]`, `[1, 9999, 384]`, `[4, 96]`, `[4, 96]`, `[4, 96]`, `[4, 96]`, `[4, 96]`, `[4, 96]`, `[4, 96]`, `[4, 96]`, `[4, 96]`, `[4, 96]`, `[4, 96]`, `[4, 96]`, `[4, 96]`, `[4, 96]`, `[4, 96]`, `[4, 96]`, `[4, 96]`, `[4, 96]`, `[4, 96]`, `[4, 96]`, `[1, 9999, 384]`, `[4, 96]`, `[4, 96]`, `[4, 96]`, `[4, 96]`, `[4, 96]`, `[4, 96]`, `[4, 96]`, `[4, 96]`, `[4, 96]`, `[4, 96]`, `[4, 96]`, `[4, 96]`, `[4, 96]`, `[4, 96]`, `[4, 96]`, `[4, 96]`, `[4, 96]`, `[4, 96]`, `[4, 96]`, `[4, 96]` |
| stt_en_squeezeformer_ctc_small_ls | nemo | 40 | `[1, 80, 521]`, `[1]`, `[1, 9999, 196]`, `[4, 49]`, `[4, 49]`, `[4, 49]`, `[4, 49]`, `[4, 49]`, `[4, 49]`, `[4, 49]`, `[4, 49]`, `[4, 49]`, `[4, 49]`, `[4, 49]`, `[4, 49]`, `[4, 49]`, `[4, 49]`, `[4, 49]`, `[4, 49]`, `[1, 9999, 196]`, `[4, 49]`, `[4, 49]`, `[4, 49]`, `[4, 49]`, `[4, 49]`, `[4, 49]`, `[4, 49]`, `[4, 49]`, `[4, 49]`, `[4, 49]`, `[4, 49]`, `[4, 49]`, `[4, 49]`, `[4, 49]`, `[4, 49]`, `[4, 49]`, `[4, 49]`, `[4, 49]`, `[4, 49]`, `[4, 49]` |
| stt_en_squeezeformer_ctc_small_medium_ls | nemo | 36 | `[1, 80, 521]`, `[1]`, `[1, 9999, 256]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[1, 9999, 256]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]`, `[4, 64]` |
| stt_en_squeezeformer_ctc_xsmall_ls | nemo | 36 | `[1, 80, 521]`, `[1]`, `[1, 9999, 144]`, `[4, 36]`, `[4, 36]`, `[4, 36]`, `[4, 36]`, `[4, 36]`, `[4, 36]`, `[4, 36]`, `[4, 36]`, `[4, 36]`, `[4, 36]`, `[4, 36]`, `[4, 36]`, `[4, 36]`, `[4, 36]`, `[1, 9999, 144]`, `[4, 36]`, `[4, 36]`, `[4, 36]`, `[4, 36]`, `[4, 36]`, `[4, 36]`, `[4, 36]`, `[4, 36]`, `[4, 36]`, `[4, 36]`, `[4, 36]`, `[4, 36]`, `[4, 36]`, `[4, 36]`, `[4, 36]`, `[4, 36]`, `[4, 36]`, `[4, 36]` |
| pigu_hybrid | others | 4 | `[1, 65, 3]`, `[1, 65, 2]`, `[1, 128, 256, 2]`, `[1, 3, 128, 256]` |
| aimv2_1b_patch14_224.apple_pt | timm | 2 | `[1, 3, 224, 224]`, `[1, 256, 2048]` |
| aimv2_1b_patch14_336.apple_pt | timm | 2 | `[1, 3, 336, 336]`, `[1, 576, 2048]` |
| aimv2_1b_patch14_448.apple_pt | timm | 2 | `[1, 3, 448, 448]`, `[1, 1024, 2048]` |
| aimv2_3b_patch14_224.apple_pt | timm | 2 | `[1, 3, 224, 224]`, `[1, 256, 3072]` |
| aimv2_3b_patch14_336.apple_pt | timm | 2 | `[1, 3, 336, 336]`, `[1, 576, 3072]` |
| aimv2_3b_patch14_448.apple_pt | timm | 2 | `[1, 3, 448, 448]`, `[1, 1024, 3072]` |
| aimv2_huge_patch14_224.apple_pt | timm | 2 | `[1, 3, 224, 224]`, `[1, 256, 1536]` |
| aimv2_large_patch14_224.apple_pt | timm | 2 | `[1, 3, 224, 224]`, `[1, 256, 1024]` |
| beit3_base_patch16_224.in22k_ft_in1k | timm | 3 | `[1, 3, 224, 224]`, `[1, 197, 768]`, `[1, 1, 768]` |
| beit3_large_patch16_224.in22k_ft_in1k | timm | 3 | `[1, 3, 224, 224]`, `[1, 197, 1024]`, `[1, 1, 1024]` |
| beitv2_base_patch16_224 | timm | 2 | `[1, 3, 224, 224]`, `[1, 1, 768]` |
| beitv2_large_patch16_224 | timm | 2 | `[1, 3, 224, 224]`, `[1, 1, 1024]` |
| cait_m36_384 | timm | 3 | `[1, 3, 384, 384]`, `[1, 576, 768]`, `[1, 1, 768]` |
| cait_m48_448 | timm | 3 | `[1, 3, 448, 448]`, `[1, 784, 768]`, `[1, 1, 768]` |
| cait_s24_224.fb_dist_in1k | timm | 3 | `[1, 3, 224, 224]`, `[1, 196, 384]`, `[1, 1, 384]` |
| cait_s24_384 | timm | 3 | `[1, 3, 384, 384]`, `[1, 576, 384]`, `[1, 1, 384]` |
| cait_s36_384 | timm | 3 | `[1, 3, 384, 384]`, `[1, 576, 384]`, `[1, 1, 384]` |
| cait_xs24_384 | timm | 3 | `[1, 3, 384, 384]`, `[1, 576, 288]`, `[1, 1, 288]` |
| cait_xxs24_224.fb_dist_in1k | timm | 3 | `[1, 3, 224, 224]`, `[1, 196, 192]`, `[1, 1, 192]` |
| cait_xxs24_384 | timm | 3 | `[1, 3, 384, 384]`, `[1, 576, 192]`, `[1, 1, 192]` |
| cait_xxs36_224.fb_dist_in1k | timm | 3 | `[1, 3, 224, 224]`, `[1, 196, 192]`, `[1, 1, 192]` |
| cait_xxs36_384 | timm | 3 | `[1, 3, 384, 384]`, `[1, 576, 192]`, `[1, 1, 192]` |
| coat_lite_medium.in1k | timm | 5 | `[1, 3, 224, 224]`, `[1, 1, 128]`, `[1, 1, 256]`, `[1, 1, 320]`, `[1, 1, 512]` |
| coat_lite_medium_384 | timm | 5 | `[1, 3, 384, 384]`, `[1, 1, 128]`, `[1, 1, 256]`, `[1, 1, 320]`, `[1, 1, 512]` |
| coat_lite_mini.in1k | timm | 5 | `[1, 3, 224, 224]`, `[1, 1, 64]`, `[1, 1, 128]`, `[1, 1, 320]`, `[1, 1, 512]` |
| coat_lite_small.in1k | timm | 5 | `[1, 3, 224, 224]`, `[1, 1, 64]`, `[1, 1, 128]`, `[1, 1, 320]`, `[1, 1, 512]` |
| coat_lite_tiny.in1k | timm | 5 | `[1, 3, 224, 224]`, `[1, 1, 64]`, `[1, 1, 128]`, `[1, 1, 256]`, `[1, 1, 320]` |
| coat_mini | timm | 5 | `[1, 3, 224, 224]`, `[1, 1, 152]`, `[1, 1, 216]`, `[1, 1, 216]`, `[1, 1, 216]` |
| coat_small | timm | 5 | `[1, 3, 224, 224]`, `[1, 1, 152]`, `[1, 1, 320]`, `[1, 1, 320]`, `[1, 1, 320]` |
| coat_tiny | timm | 5 | `[1, 3, 224, 224]`, `[1, 1, 152]`, `[1, 1, 152]`, `[1, 1, 152]`, `[1, 1, 152]` |
| convit_base | timm | 3 | `[1, 3, 224, 224]`, `[1, 196, 768]`, `[1, 1, 768]` |
| convit_small | timm | 3 | `[1, 3, 224, 224]`, `[1, 196, 432]`, `[1, 1, 432]` |
| convit_tiny | timm | 3 | `[1, 3, 224, 224]`, `[1, 196, 192]`, `[1, 1, 192]` |
| crossvit_15_240.in1k | timm | 5 | `[1, 3, 224, 224]`, `[1, 1, 192]`, `[1, 401, 192]`, `[1, 1, 384]`, `[1, 197, 384]` |
| crossvit_15_dagger_240.in1k | timm | 5 | `[1, 3, 224, 224]`, `[1, 1, 192]`, `[1, 401, 192]`, `[1, 1, 384]`, `[1, 197, 384]` |
| crossvit_15_dagger_408.in1k | timm | 5 | `[1, 3, 224, 224]`, `[1, 1, 192]`, `[1, 1157, 192]`, `[1, 1, 384]`, `[1, 577, 384]` |
| crossvit_18_240.in1k | timm | 5 | `[1, 3, 224, 224]`, `[1, 1, 224]`, `[1, 401, 224]`, `[1, 1, 448]`, `[1, 197, 448]` |
| crossvit_18_dagger_240.in1k | timm | 5 | `[1, 3, 224, 224]`, `[1, 1, 224]`, `[1, 401, 224]`, `[1, 1, 448]`, `[1, 197, 448]` |
| crossvit_18_dagger_408.in1k | timm | 5 | `[1, 3, 224, 224]`, `[1, 1, 224]`, `[1, 1157, 224]`, `[1, 1, 448]`, `[1, 577, 448]` |
| crossvit_9_240.in1k | timm | 5 | `[1, 3, 224, 224]`, `[1, 1, 128]`, `[1, 401, 128]`, `[1, 1, 256]`, `[1, 197, 256]` |
| crossvit_9_dagger_240.in1k | timm | 5 | `[1, 3, 224, 224]`, `[1, 1, 128]`, `[1, 401, 128]`, `[1, 1, 256]`, `[1, 197, 256]` |
| crossvit_base_240.in1k | timm | 5 | `[1, 3, 224, 224]`, `[1, 1, 384]`, `[1, 401, 384]`, `[1, 1, 768]`, `[1, 197, 768]` |
| crossvit_small_240.in1k | timm | 5 | `[1, 3, 224, 224]`, `[1, 1, 192]`, `[1, 401, 192]`, `[1, 1, 384]`, `[1, 197, 384]` |
| crossvit_tiny_240.in1k | timm | 5 | `[1, 3, 224, 224]`, `[1, 1, 96]`, `[1, 401, 96]`, `[1, 1, 192]`, `[1, 197, 192]` |
| deit3_base_patch16_224.fb_in1k | timm | 3 | `[1, 3, 224, 224]`, `[1, 196, 768]`, `[1, 1, 768]` |
| deit3_base_patch16_384 | timm | 3 | `[1, 3, 384, 384]`, `[1, 576, 768]`, `[1, 1, 768]` |
| deit3_huge_patch14_224.fb_in1k | timm | 3 | `[1, 3, 224, 224]`, `[1, 256, 1280]`, `[1, 1, 1280]` |
| deit3_large_patch16_224.fb_in1k | timm | 3 | `[1, 3, 224, 224]`, `[1, 196, 1024]`, `[1, 1, 1024]` |
| deit3_large_patch16_384 | timm | 3 | `[1, 3, 384, 384]`, `[1, 576, 1024]`, `[1, 1, 1024]` |
| deit3_medium_patch16_224.fb_in1k | timm | 3 | `[1, 3, 224, 224]`, `[1, 196, 512]`, `[1, 1, 512]` |
| deit3_small_patch16_224.fb_in1k | timm | 3 | `[1, 3, 224, 224]`, `[1, 196, 384]`, `[1, 1, 384]` |
| deit3_small_patch16_384 | timm | 3 | `[1, 3, 384, 384]`, `[1, 576, 384]`, `[1, 1, 384]` |
| deit_base_distilled_patch16_224.fb_in1k | timm | 4 | `[1, 3, 224, 224]`, `[1, 198, 768]`, `[1, 1, 768]`, `[1, 1, 768]` |
| deit_base_distilled_patch16_384 | timm | 4 | `[1, 3, 384, 384]`, `[1, 578, 768]`, `[1, 1, 768]`, `[1, 1, 768]` |
| deit_base_patch16_224.fb_in1k | timm | 3 | `[1, 3, 224, 224]`, `[1, 197, 768]`, `[1, 1, 768]` |
| deit_small_distilled_patch16_224.fb_in1k | timm | 4 | `[1, 3, 224, 224]`, `[1, 198, 384]`, `[1, 1, 384]`, `[1, 1, 384]` |
| deit_small_patch16_224.fb_in1k | timm | 3 | `[1, 3, 224, 224]`, `[1, 197, 384]`, `[1, 1, 384]` |
| deit_tiny_distilled_patch16_224.fb_in1k | timm | 4 | `[1, 3, 224, 224]`, `[1, 198, 192]`, `[1, 1, 192]`, `[1, 1, 192]` |
| deit_tiny_patch16_224.fb_in1k | timm | 3 | `[1, 3, 224, 224]`, `[1, 197, 192]`, `[1, 1, 192]` |
| eva02_base_patch14_224.mim_in22k | timm | 3 | `[1, 3, 224, 224]`, `[1, 257, 768]`, `[1, 1, 768]` |
| eva02_base_patch16_clip_224.merged2b | timm | 3 | `[1, 3, 224, 224]`, `[1, 197, 768]`, `[1, 1, 768]` |
| eva02_enormous_patch14_clip_224.laion2b | timm | 3 | `[1, 3, 224, 224]`, `[1, 257, 1792]`, `[1, 1, 1792]` |
| eva02_large_patch14_224.mim_in22k | timm | 3 | `[1, 3, 224, 224]`, `[1, 257, 1024]`, `[1, 1, 1024]` |
| eva02_large_patch14_clip_224.merged2b | timm | 3 | `[1, 3, 224, 224]`, `[1, 257, 1024]`, `[1, 1, 1024]` |
| eva02_small_patch14_224.mim_in22k | timm | 3 | `[1, 3, 224, 224]`, `[1, 257, 384]`, `[1, 1, 384]` |
| eva02_tiny_patch14_224.mim_in22k | timm | 3 | `[1, 3, 224, 224]`, `[1, 257, 192]`, `[1, 1, 192]` |
| eva_giant_patch14_224.clip_ft_in1k | timm | 3 | `[1, 3, 224, 224]`, `[1, 257, 1408]`, `[1, 1, 1408]` |
| eva_giant_patch14_clip_224.laion400m | timm | 3 | `[1, 3, 224, 224]`, `[1, 257, 1408]`, `[1, 1, 1408]` |
| samvit_base_patch16.sa1b | timm | 2 | `[1, 3, 224, 224]`, `[1, 64, 64, 768]` |
| samvit_huge_patch16.sa1b | timm | 2 | `[1, 3, 224, 224]`, `[1, 64, 64, 1280]` |
| samvit_large_patch16.sa1b | timm | 2 | `[1, 3, 224, 224]`, `[1, 64, 64, 1024]` |
| vit_base_patch16_224.dino | timm | 3 | `[1, 3, 224, 224]`, `[1, 197, 768]`, `[1, 1, 768]` |
| vit_base_patch16_224_miil.in21k | timm | 3 | `[1, 3, 224, 224]`, `[1, 197, 768]`, `[1, 1, 768]` |
| vit_base_patch16_clip_224.datacompxl | timm | 3 | `[1, 3, 224, 224]`, `[1, 197, 768]`, `[1, 1, 768]` |
| vit_base_patch16_clip_quickgelu_224.metaclip_2pt5b | timm | 3 | `[1, 3, 224, 224]`, `[1, 197, 768]`, `[1, 1, 768]` |
| vit_base_patch16_rope_224.naver_in1k | timm | 2 | `[1, 3, 224, 224]`, `[1, 1, 768]` |
| vit_base_patch16_rope_ape_224.naver_in1k | timm | 3 | `[1, 3, 224, 224]`, `[1, 196, 768]`, `[1, 1, 768]` |
| vit_base_patch16_rope_mixed_224.naver_in1k | timm | 2 | `[1, 3, 224, 224]`, `[1, 1, 768]` |
| vit_base_patch16_rope_mixed_ape_224.naver_in1k | timm | 3 | `[1, 3, 224, 224]`, `[1, 196, 768]`, `[1, 1, 768]` |
| vit_base_patch16_rpn_224.sw_in1k | timm | 2 | `[1, 3, 224, 224]`, `[1, 196, 768]` |
| vit_base_patch16_siglip_gap_224.v2_webli | timm | 2 | `[1, 3, 224, 224]`, `[1, 196, 768]` |
| vit_base_patch32_224.augreg_in1k | timm | 3 | `[1, 3, 224, 224]`, `[1, 50, 768]`, `[1, 1, 768]` |
| vit_base_patch32_224.orig_in21k | timm | 3 | `[1, 3, 224, 224]`, `[1, 50, 768]`, `[1, 1, 768]` |
| vit_base_patch32_clip_224.datacompxl | timm | 3 | `[1, 3, 224, 224]`, `[1, 50, 768]`, `[1, 1, 768]` |
| vit_base_patch32_clip_quickgelu_224.laion400m_e32 | timm | 3 | `[1, 3, 224, 224]`, `[1, 50, 768]`, `[1, 1, 768]` |
| vit_base_patch8_224.augreg2_in21k_ft_in1k | timm | 3 | `[1, 3, 224, 224]`, `[1, 785, 768]`, `[1, 1, 768]` |
| vit_base_patch8_224.dino | timm | 3 | `[1, 3, 224, 224]`, `[1, 785, 768]`, `[1, 1, 768]` |
| vit_betwixt_patch32_clip_224.tinyclip_laion400m | timm | 3 | `[1, 3, 224, 224]`, `[1, 50, 640]`, `[1, 1, 640]` |
| vit_giant_patch14_clip_224.laion2b | timm | 3 | `[1, 3, 224, 224]`, `[1, 257, 1408]`, `[1, 1, 1408]` |
| vit_giant_patch16_gap_224.in22k_ijepa | timm | 2 | `[1, 3, 224, 224]`, `[1, 196, 1408]` |
| vit_gigantic_patch14_clip_224.laion2b | timm | 3 | `[1, 3, 224, 224]`, `[1, 257, 1664]`, `[1, 1, 1664]` |
| vit_gigantic_patch14_clip_quickgelu_224.metaclip_2pt5b | timm | 3 | `[1, 3, 224, 224]`, `[1, 257, 1664]`, `[1, 1, 1664]` |
| vit_huge_patch14_224.mae | timm | 3 | `[1, 3, 224, 224]`, `[1, 257, 1280]`, `[1, 1, 1280]` |
| vit_huge_patch14_clip_224.dfn5b | timm | 3 | `[1, 3, 224, 224]`, `[1, 257, 1280]`, `[1, 1, 1280]` |
| vit_huge_patch14_clip_quickgelu_224.dfn5b | timm | 3 | `[1, 3, 224, 224]`, `[1, 257, 1280]`, `[1, 1, 1280]` |
| vit_huge_patch14_gap_224.in1k_ijepa | timm | 2 | `[1, 3, 224, 224]`, `[1, 256, 1280]` |
| vit_large_patch14_clip_224.datacompxl | timm | 3 | `[1, 3, 224, 224]`, `[1, 257, 1024]`, `[1, 1, 1024]` |
| vit_large_patch14_clip_quickgelu_224.dfn2b | timm | 3 | `[1, 3, 224, 224]`, `[1, 257, 1024]`, `[1, 1, 1024]` |
| vit_large_patch16_224.augreg_in21k | timm | 3 | `[1, 3, 224, 224]`, `[1, 197, 1024]`, `[1, 1, 1024]` |
| vit_large_patch16_224.mae | timm | 3 | `[1, 3, 224, 224]`, `[1, 197, 1024]`, `[1, 1, 1024]` |
| vit_large_patch16_rope_224.naver_in1k | timm | 2 | `[1, 3, 224, 224]`, `[1, 1, 1024]` |
| vit_large_patch16_rope_ape_224.naver_in1k | timm | 3 | `[1, 3, 224, 224]`, `[1, 196, 1024]`, `[1, 1, 1024]` |
| vit_large_patch16_rope_mixed_224.naver_in1k | timm | 2 | `[1, 3, 224, 224]`, `[1, 1, 1024]` |
| vit_large_patch16_rope_mixed_ape_224.naver_in1k | timm | 3 | `[1, 3, 224, 224]`, `[1, 196, 1024]`, `[1, 1, 1024]` |
| vit_large_patch32_224.orig_in21k | timm | 3 | `[1, 3, 224, 224]`, `[1, 50, 1024]`, `[1, 1, 1024]` |
| vit_medium_patch16_clip_224.tinyclip_yfcc15m | timm | 3 | `[1, 3, 224, 224]`, `[1, 197, 512]`, `[1, 1, 512]` |
| vit_medium_patch32_clip_224.tinyclip_laion400m | timm | 3 | `[1, 3, 224, 224]`, `[1, 50, 512]`, `[1, 1, 512]` |
| vit_small_patch16_224.dino | timm | 3 | `[1, 3, 224, 224]`, `[1, 197, 384]`, `[1, 1, 384]` |
| vit_small_patch16_rope_224.naver_in1k | timm | 2 | `[1, 3, 224, 224]`, `[1, 1, 384]` |
| vit_small_patch16_rope_ape_224.naver_in1k | timm | 3 | `[1, 3, 224, 224]`, `[1, 196, 384]`, `[1, 1, 384]` |
| vit_small_patch16_rope_mixed_224.naver_in1k | timm | 2 | `[1, 3, 224, 224]`, `[1, 1, 384]` |
| vit_small_patch16_rope_mixed_ape_224.naver_in1k | timm | 3 | `[1, 3, 224, 224]`, `[1, 196, 384]`, `[1, 1, 384]` |
| vit_small_patch32_224.augreg_in21k | timm | 3 | `[1, 3, 224, 224]`, `[1, 50, 384]`, `[1, 1, 384]` |
| vit_small_patch8_224.dino | timm | 3 | `[1, 3, 224, 224]`, `[1, 785, 384]`, `[1, 1, 384]` |
| vit_so400m_patch14_siglip_gap_224.pali2_3b_pt | timm | 2 | `[1, 3, 224, 224]`, `[1, 256, 1152]` |
| vit_xsmall_patch16_clip_224.tinyclip_yfcc15m | timm | 3 | `[1, 3, 224, 224]`, `[1, 197, 256]`, `[1, 1, 256]` |
| squim_subjective | torchaudio | 2 | `[1, 80000]`, `[1, 80000]` |
| EdgeCNN | torchgeometric | 2 | `[1000, 128]`, `[2, 100]` |
| GAE | torchgeometric | 2 | `[1000, 32]`, `[2, 100]` |
| GAT | torchgeometric | 2 | `[1000, 128]`, `[2, 100]` |
| GCN | torchgeometric | 2 | `[1000, 128]`, `[2, 100]` |
| GIN | torchgeometric | 2 | `[1000, 128]`, `[2, 100]` |
| GraphSAGE | torchgeometric | 2 | `[1000, 128]`, `[2, 100]` |
| LINKX | torchgeometric | 2 | `[2, 100]`, `[1000, 128]` |
| PMLP | torchgeometric | 2 | `[2, 1000]`, `[1000, 128]` |
| PNA | torchgeometric | 2 | `[1000, 128]`, `[2, 100]` |
| RECT_L | torchgeometric | 2 | `[128, 128]`, `[2, 128]` |
| SignedGCN | torchgeometric | 3 | `[1000, 32]`, `[2, 100]`, `[2, 100]` |
| vit_b_16 | torchvision | 2 | `[1, 3, 224, 224]`, `[1, 1, 768]` |
| vit_b_32 | torchvision | 2 | `[1, 3, 224, 224]`, `[1, 1, 768]` |
| vit_h_14 | torchvision | 2 | `[1, 3, 224, 224]`, `[1, 1, 1280]` |
| vit_l_16 | torchvision | 2 | `[1, 3, 224, 224]`, `[1, 1, 1024]` |
| vit_l_32 | torchvision | 2 | `[1, 3, 224, 224]`, `[1, 1, 1024]` |
| 42dot_LLM-SFT-1.3B | transformers-auto-model | 2 | `[1, 20]`, `[1, 20]` |
| AI-Nordics_bert-large-swedish-cased | transformers-auto-model | 3 | `[1, 20]`, `[1, 20]`, `[1, 20]` |
| AK_ak_nlp | transformers-auto-model | 2 | `[1, 19]`, `[1, 19]` |
| ALINEAR_albert-japanese | transformers-auto-model | 3 | `[1, 21]`, `[1, 21]`, `[1, 21]` |
| ALINEAR_albert-japanese-v2 | transformers-auto-model | 3 | `[1, 21]`, `[1, 21]`, `[1, 21]` |
| ARTeLab_it5-summarization-fanpage-64 | transformers-auto-model | 3 | `[1, 24, 768]`, `[1, 24, 768]`, `[1, 24]` |
| AVSilva_bertimbau-large-fine-tuned-md | transformers-auto-model | 3 | `[1, 18]`, `[1, 18]`, `[1, 18]` |
| AceInstruct-1.5B | transformers-auto-model | 2 | `[1, 19]`, `[1, 19]` |
| AethiQs-Max_AethiQs_GemBERT_bertje_50k | transformers-auto-model | 3 | `[1, 24]`, `[1, 24]`, `[1, 24]` |
| Aira-OPT-125M | transformers-auto-model | 2 | `[1, 20]`, `[1, 20]` |
| Aleksandar_bert-srb-ner-setimes | transformers-auto-model | 3 | `[1, 25]`, `[1, 25]`, `[1, 25]` |
| Aleksandar_distilbert-srb-ner-setimes | transformers-auto-model | 2 | `[1, 21]`, `[1, 21]` |
| Aleksandar_electra-srb-ner-setimes | transformers-auto-model | 3 | `[1, 22]`, `[1, 22]`, `[1, 22]` |
| AlphaMaze-v0.2-1.5B | transformers-auto-model | 2 | `[1, 20]`, `[1, 20]` |
| Andrija_SRoBERTa-L-NER | transformers-auto-model | 2 | `[1, 18]`, `[1, 18]` |
| Andrija_SRoBERTa-NER | transformers-auto-model | 2 | `[1, 17]`, `[1, 17]` |
| BAAI_AltCLIP | transformers-auto-model | 3 | `[2, 7]`, `[2, 7]`, `[1, 3, 224, 224]` |
| BAAI_AltCLIP-m18 | transformers-auto-model | 3 | `[2, 7]`, `[2, 7]`, `[1, 3, 224, 224]` |
| Biggie-SmoLlm-0.4B | transformers-auto-model | 2 | `[1, 19]`, `[1, 19]` |
| CAMeL-Lab_text-editing-qalb14-nopnx | transformers-auto-model | 3 | `[1, 26]`, `[1, 26]`, `[1, 26]` |
| CLAck_en-vi | transformers-auto-model | 3 | `[1, 19, 512]`, `[1, 19, 512]`, `[1, 19]` |
| Culmenus_XLMR-ENIS-finetuned-ner | transformers-auto-model | 2 | `[1, 14]`, `[1, 14]` |
| DAMO-NLP-SG_zero-shot-classify-SSTuning-ALBERT | transformers-auto-model | 3 | `[1, 12]`, `[1, 12]`, `[1, 12]` |
| DTAI-KULeuven_robbertje-1-gb-bort | transformers-auto-model | 2 | `[1, 15]`, `[1, 15]` |
| DTAI-KULeuven_robbertje-1-gb-merged | transformers-auto-model | 2 | `[1, 15]`, `[1, 15]` |
| DarshanDeshpande_marathi-distilbert | transformers-auto-model | 2 | `[1, 10]`, `[1, 10]` |
| Data-Lab_ruRoberta-large_classification_v0.2 | transformers-auto-model | 2 | `[1, 20]`, `[1, 20]` |
| DatologyAI_cls-opt-vit-b-32 | transformers-auto-model | 2 | `[1, 13, 768]`, `[1, 13]` |
| Davlan_afro-xlmr-large | transformers-auto-model | 2 | `[1, 15]`, `[1, 15]` |
| DeepChem_ChemBERTa-10M-MLM | transformers-auto-model | 2 | `[1, 16]`, `[1, 16]` |
| DiegoRossini_flaubert-pouvoir-modality-detector | transformers-auto-model | 4 | `[1, 17]`, `[1, 17]`, `[1, 512]`, `[1, 17]` |
| Dr-BERT_CAS-Biomedical-POS-Tagging | transformers-auto-model | 2 | `[1, 16]`, `[1, 16]` |
| ELiRF_NASCA | transformers-auto-model | 3 | `[1, 17, 1024]`, `[1, 17, 1024]`, `[1, 17]` |
| ELiRF_mbart-large-cc25-dacsa-ca | transformers-auto-model | 3 | `[1, 15, 1024]`, `[1, 15, 1024]`, `[1, 15]` |
| EMBEDDIA_est-roberta | transformers-auto-model | 2 | `[1, 21]`, `[1, 21]` |
| EMBEDDIA_finest-bert | transformers-auto-model | 3 | `[1, 14]`, `[1, 14]`, `[1, 14]` |
| EMBEDDIA_litlat-bert | transformers-auto-model | 2 | `[1, 16]`, `[1, 16]` |
| EXAONE-4.0-1.2B | transformers-auto-model | 2 | `[1, 19]`, `[1, 19]` |
| EleutherAI_pythia-1b | transformers-auto-model | 2 | `[1, 2, 2048]`, `[1, 2]` |
| EleutherAI_pythia-70m | transformers-auto-model | 2 | `[1, 19]`, `[1, 19]` |
| FacebookAI_roberta-large | transformers-auto-model | 2 | `[1, 11]`, `[1, 11]` |
| Fafadalilian_lora-adapter-t5_small_model_California_state_bill | transformers-auto-model | 3 | `[1, 12, 512]`, `[1, 12, 512]`, `[1, 12]` |
| Farid59_nllb-darija-fr_eng | transformers-auto-model | 4 | `[1, 16, 1024]`, `[1, 16, 1024]`, `[1, 16]`, `[1, 16]` |
| Film8844_wangchanberta-ner | transformers-auto-model | 2 | `[1, 27]`, `[1, 27]` |
| FinOPT-Lincoln | transformers-auto-model | 2 | `[1, 20]`, `[1, 20]` |
| Finnish-NLP_convbert-base-generator-finnish | transformers-auto-model | 3 | `[1, 19]`, `[1, 19]`, `[1, 19]` |
| Finnish-NLP_electra-base-generator-finnish | transformers-auto-model | 3 | `[1, 19]`, `[1, 19]`, `[1, 19]` |
| Fsoft-AIC_videberta-base | transformers-auto-model | 2 | `[1, 18]`, `[1, 18]` |
| GKLMIP_electra-khmer-base-uncased | transformers-auto-model | 3 | `[1, 18]`, `[1, 18]`, `[1, 18]` |
| GKLMIP_electra-khmer-base-uncased-tokenized | transformers-auto-model | 3 | `[1, 14]`, `[1, 14]`, `[1, 14]` |
| GKLMIP_electra-khmer-small-uncased-tokenized | transformers-auto-model | 3 | `[1, 14]`, `[1, 14]`, `[1, 14]` |
| GePpeTto | transformers-auto-model | 2 | `[1, 27]`, `[1, 27]` |
| Geotrend_distilbert-base-ar-cased | transformers-auto-model | 2 | `[1, 23]`, `[1, 23]` |
| Geotrend_distilbert-base-bg-cased | transformers-auto-model | 2 | `[1, 22]`, `[1, 22]` |
| Geotrend_distilbert-base-da-cased | transformers-auto-model | 2 | `[1, 19]`, `[1, 19]` |
| Ghana-NLP_robako-base-akuapem-twi-cased | transformers-auto-model | 2 | `[1, 21]`, `[1, 21]` |
| GroNLP_bert-base-dutch-cased-frisian | transformers-auto-model | 3 | `[1, 26]`, `[1, 26]`, `[1, 26]` |
| Hate-speech-CNERG_dehatebert-mono-arabic | transformers-auto-model | 3 | `[1, 12]`, `[1, 12]`, `[1, 12]` |
| Hax_filipino-text-version1 | transformers-auto-model | 3 | `[1, 18]`, `[1, 18]`, `[1, 18]` |
| Helsinki-NLP_opus-mt-NORTH_EU-NORTH_EU | transformers-auto-model | 3 | `[1, 18, 512]`, `[1, 18, 512]`, `[1, 18]` |
| Helsinki-NLP_opus-mt-aed-es | transformers-auto-model | 3 | `[1, 21, 512]`, `[1, 21, 512]`, `[1, 21]` |
| Helsinki-NLP_opus-mt-af-es | transformers-auto-model | 3 | `[1, 20, 512]`, `[1, 20, 512]`, `[1, 20]` |
| Helsinki-NLP_opus-mt-en-zh | transformers-auto-model | 3 | `[1, 12, 512]`, `[1, 12, 512]`, `[1, 12]` |
| Helsinki-NLP_opus-mt-zh-en | transformers-auto-model | 3 | `[1, 26, 512]`, `[1, 26, 512]`, `[1, 26]` |
| Hezam_ArabicT5_Classification | transformers-auto-model | 3 | `[1, 23, 512]`, `[1, 23, 512]`, `[1, 23]` |
| HooshvareLab_albert-fa-zwnj-base-v2 | transformers-auto-model | 3 | `[1, 20]`, `[1, 20]`, `[1, 20]` |
| HooshvareLab_roberta-fa-zwnj-base-ner | transformers-auto-model | 2 | `[1, 16]`, `[1, 16]` |
| HueyNemud_berties | transformers-auto-model | 2 | `[1, 18]`, `[1, 18]` |
| IDEA-Research_dab-detr-resnet-50 | transformers-auto-model | 2 | `[1, 3, 800, 800]`, `[1, 800, 800]` |
| IDEA-Research_grounding-dino-base | transformers-auto-model | 5 | `[1, 7, 256]`, `[1, 6, 900, 256]`, `[1, 900, 4]`, `[1, 6, 900, 4]`, `[1, 7]` |
| IDEA-Research_grounding-dino-tiny | transformers-auto-model | 2 | `[1, 900, 256]`, `[1, 900, 256]` |
| IMISLab_GreekWiki-umt5-base | transformers-auto-model | 3 | `[1, 12, 768]`, `[1, 12, 768]`, `[1, 12]` |
| IS-LM-3B | transformers-auto-model | 2 | `[1, 19]`, `[1, 19]` |
| InstaDeepAI_nucleotide-transformer-500m-human-ref | transformers-auto-model | 2 | `[1, 11]`, `[1, 11]` |
| Intel_dpt-beit-base-384 | transformers-auto-model | 4 | `[1, 577, 768]`, `[1, 577, 768]`, `[1, 577, 768]`, `[1, 577, 768]` |
| Intel_dpt-beit-large-384 | transformers-auto-model | 4 | `[1, 577, 1024]`, `[1, 577, 1024]`, `[1, 577, 1024]`, `[1, 577, 1024]` |
| Intel_dpt-beit-large-512 | transformers-auto-model | 4 | `[1, 1025, 1024]`, `[1, 1025, 1024]`, `[1, 1025, 1024]`, `[1, 1025, 1024]` |
| Intel_dpt-swinv2-tiny-256 | transformers-auto-model | 4 | `[1, 96, 64, 64]`, `[1, 192, 32, 32]`, `[1, 384, 16, 16]`, `[1, 768, 8, 8]` |
| Intel_fid_flan_t5_base_nq | transformers-auto-model | 3 | `[1, 12, 768]`, `[1, 12, 768]`, `[1, 12]` |
| Intel_zoedepth-kitti | transformers-auto-model | 4 | `[1, 1025, 1024]`, `[1, 1025, 1024]`, `[1, 1025, 1024]`, `[1, 1025, 1024]` |
| Intel_zoedepth-nyu | transformers-auto-model | 4 | `[1, 1025, 1024]`, `[1, 1025, 1024]`, `[1, 1025, 1024]`, `[1, 1025, 1024]` |
| Jean-Baptiste_roberta-large-ner-english | transformers-auto-model | 2 | `[1, 11]`, `[1, 11]` |
| Jodsa_camembert_mlm | transformers-auto-model | 2 | `[1, 24]`, `[1, 24]` |
| KBLab_electra-base-swedish-cased-generator | transformers-auto-model | 3 | `[1, 20]`, `[1, 20]`, `[1, 20]` |
| KBLab_electra-small-swedish-cased-generator | transformers-auto-model | 3 | `[1, 20]`, `[1, 20]`, `[1, 20]` |
| KBLab_megatron-bert-base-swedish-cased-600k | transformers-auto-model | 3 | `[1, 20]`, `[1, 20]`, `[1, 20]` |
| Ketzu_koelectra-sts-v0.4 | transformers-auto-model | 3 | `[1, 20]`, `[1, 20]`, `[1, 20]` |
| KoboldAI_fairseq-dense-1.3B | transformers-auto-model | 2 | `[1, 9]`, `[1, 9, 2048]` |
| KoboldAI_fairseq-dense-125M | transformers-auto-model | 2 | `[1, 9]`, `[1, 9, 768]` |
| KoboldAI_fairseq-dense-355M | transformers-auto-model | 2 | `[1, 9]`, `[1, 9, 1024]` |
| KoichiYasuoka_bert-large-japanese-upos | transformers-auto-model | 3 | `[1, 45]`, `[1, 45]`, `[1, 45]` |
| KoichiYasuoka_deberta-base-ainu-ud-goeswith | transformers-auto-model | 2 | `[1, 43]`, `[1, 43]` |
| KoichiYasuoka_deberta-base-chinese-ud-goeswith | transformers-auto-model | 2 | `[1, 10]`, `[1, 10]` |
| KoichiYasuoka_deberta-base-japanese-aozora-ud-goeswith | transformers-auto-model | 2 | `[1, 21]`, `[1, 21]` |
| KoichiYasuoka_deberta-base-thai-ud-goeswith | transformers-auto-model | 2 | `[1, 27]`, `[1, 27]` |
| KoichiYasuoka_roberta-base-japanese-luw-upos | transformers-auto-model | 3 | `[1, 18]`, `[1, 18]`, `[1, 18]` |
| KoichiYasuoka_roberta-base-thai-spm-upos | transformers-auto-model | 3 | `[1, 27]`, `[1, 27]`, `[1, 27]` |
| KoichiYasuoka_roberta-classical-chinese-base-sentence-segmentation | transformers-auto-model | 3 | `[1, 10]`, `[1, 10]`, `[1, 10]` |
| KoichiYasuoka_roberta-large-korean-upos | transformers-auto-model | 3 | `[1, 22]`, `[1, 22]`, `[1, 22]` |
| LFM2-350M | transformers-auto-model | 2 | `[1, 20]`, `[1, 20]` |
| LLaMmlein_120M | transformers-auto-model | 2 | `[1, 32]`, `[1, 32]` |
| LYTinn_gpt2-finetuning-sentiment-model-3000-samples | transformers-auto-model | 2 | `[1, 21]`, `[1, 21]` |
| Lazaro97_results | transformers-auto-model | 2 | `[1, 17]`, `[1, 17]` |
| LeoFeng_ChineseSequenceClassification | transformers-auto-model | 3 | `[1, 20]`, `[1, 20]`, `[1, 20]` |
| LiheYoung_depth-anything-large-hf | transformers-auto-model | 4 | `[1, 1370, 1024]`, `[1, 1370, 1024]`, `[1, 1370, 1024]`, `[1, 1370, 1024]` |
| LiheYoung_depth-anything-small-hf | transformers-auto-model | 4 | `[1, 1370, 384]`, `[1, 1370, 384]`, `[1, 1370, 384]`, `[1, 1370, 384]` |
| Mahmoud8_bigbird-roberta-base | transformers-auto-model | 3 | `[1, 11]`, `[1, 11]`, `[1, 11]` |
| Mahmoud8_mah-efficient_mlm_m0.40 | transformers-auto-model | 2 | `[1, 11]`, `[1, 11]` |
| Manojb_CLIP-ViT-L-14-DataComp.XL-s13B-b90K | transformers-auto-model | 3 | `[1, 3, 224, 224]`, `[2, 7]`, `[2, 7]` |
| Michielo_mt5-small_nl-en_translation | transformers-auto-model | 3 | `[1, 12, 512]`, `[1, 12, 512]`, `[1, 12]` |
| Mizuiro-sakura_luke-japanese-large-finetuned-ner | transformers-auto-model | 2 | `[1, 29]`, `[1, 29]` |
| MohamedZaitoon_bart-fine-tune | transformers-auto-model | 3 | `[1, 11, 1024]`, `[1, 11, 1024]`, `[1, 11]` |
| MoritzLaurer_DeBERTa-v3-base-mnli-fever-anli | transformers-auto-model | 2 | `[1, 11]`, `[1, 11]` |
| MoritzLaurer_DeBERTa-v3-large-mnli-fever-anli-ling-wanli | transformers-auto-model | 2 | `[1, 11]`, `[1, 11]` |
| MoritzLaurer_DeBERTa-v3-xsmall-mnli-fever-anli-ling-binary | transformers-auto-model | 2 | `[1, 11]`, `[1, 11]` |
| MoritzLaurer_mDeBERTa-v3-base-mnli-xnli | transformers-auto-model | 2 | `[1, 13]`, `[1, 13]` |
| MoritzLaurer_multilingual-MiniLMv2-L12-mnli-xnli | transformers-auto-model | 2 | `[1, 15]`, `[1, 15]` |
| MoritzLaurer_multilingual-MiniLMv2-L6-mnli-xnli | transformers-auto-model | 2 | `[1, 15]`, `[1, 15]` |
| MoritzLaurer_xtremedistil-l6-h256-mnli-fever-anli-ling-binary | transformers-auto-model | 3 | `[1, 11]`, `[1, 11]`, `[1, 11]` |
| NDugar_deberta-v2-xlarge-mnli | transformers-auto-model | 2 | `[1, 11]`, `[1, 11]` |
| NYTK_summarization-hi-bart-base-1024-hungarian | transformers-auto-model | 3 | `[1, 18, 768]`, `[1, 18, 768]`, `[1, 18]` |
| NeNeDataScientist_NeNeAI007_fr-kiluba-translation | transformers-auto-model | 3 | `[1, 17, 512]`, `[1, 17, 512]`, `[1, 17]` |
| Neurora_opus-tatoeba-heb-eng | transformers-auto-model | 3 | `[1, 25, 512]`, `[1, 25, 512]`, `[1, 25]` |
| Neurora_opus-tatoeba-hye-eng | transformers-auto-model | 3 | `[1, 48, 512]`, `[1, 48, 512]`, `[1, 48]` |
| Neurora_opus-tatoeba-ron-eng | transformers-auto-model | 3 | `[1, 17, 512]`, `[1, 17, 512]`, `[1, 17]` |
| Neurora_opus-tatoeba-tur-eng | transformers-auto-model | 3 | `[1, 23, 512]`, `[1, 23, 512]`, `[1, 23]` |
| Neurora_opus-tatoeba-zho-eng | transformers-auto-model | 3 | `[1, 26, 512]`, `[1, 26, 512]`, `[1, 26]` |
| Nextcloud-AI_opus-mt-ar-es | transformers-auto-model | 3 | `[1, 27, 512]`, `[1, 27, 512]`, `[1, 27]` |
| OFA-Sys_chinese-clip-vit-huge-patch14 | transformers-auto-model | 4 | `[1, 3, 224, 224]`, `[2, 7]`, `[2, 7]`, `[2, 7]` |
| OFA-Sys_chinese-clip-vit-large-patch14 | transformers-auto-model | 4 | `[1, 3, 224, 224]`, `[2, 7]`, `[2, 7]`, `[2, 7]` |
| Onutoa_1_6e-3_5_0.5 | transformers-auto-model | 3 | `[1, 11]`, `[1, 11]`, `[1, 11]` |
| OpenMed_OpenMed-NER-PharmaDetect-SuperMedical-125M | transformers-auto-model | 2 | `[1, 11]`, `[1, 11]` |
| PeterBanning71_bart-large-finetuned-eLife | transformers-auto-model | 3 | `[1, 11, 1024]`, `[1, 11, 1024]`, `[1, 11]` |
| PeterBanning71_long-t5-tfg | transformers-auto-model | 3 | `[1, 12]`, `[1, 12]`, `[1, 12]` |
| PolyCoder-160M | transformers-auto-model | 2 | `[1, 20]`, `[1, 20]` |
| Poulami_muril-large-cased-finetuned-sentiment-SST2 | transformers-auto-model | 3 | `[1, 12]`, `[1, 12]`, `[1, 12]` |
| Q-MM_clip-vit-large-patch14-336 | transformers-auto-model | 3 | `[1, 3, 336, 336]`, `[2, 7]`, `[2, 7]` |
| Qishuai_distilbert_punctuator_zh | transformers-auto-model | 2 | `[1, 18]`, `[1, 18]` |
| Qwen1.5-0.5B | transformers-auto-model | 2 | `[1, 19]`, `[1, 19]` |
| Qwen2.5-0.5B | transformers-auto-model | 2 | `[1, 19]`, `[1, 19]` |
| Qwen3-Embedding-0.6B | transformers-auto-model | 2 | `[1, 20]`, `[1, 20]` |
| Recognai_bert-base-spanish-wwm-cased-xnli | transformers-auto-model | 3 | `[1, 19]`, `[1, 19]`, `[1, 19]` |
| Recognai_zeroshot_selectra_medium | transformers-auto-model | 3 | `[1, 17]`, `[1, 17]`, `[1, 17]` |
| Recognai_zeroshot_selectra_small | transformers-auto-model | 3 | `[1, 17]`, `[1, 17]`, `[1, 17]` |
| Sahajtomar_German_Zeroshot | transformers-auto-model | 3 | `[1, 17]`, `[1, 17]`, `[1, 17]` |
| Sayan01_L-2_H-768_A-12_Large_sst2 | transformers-auto-model | 3 | `[1, 11]`, `[1, 11]`, `[1, 11]` |
| Sayan01_L-4_H-128_A-2_Large_sst2 | transformers-auto-model | 3 | `[1, 11]`, `[1, 11]`, `[1, 11]` |
| Sayan01_tiny-bert-sst2-distilled | transformers-auto-model | 3 | `[1, 11]`, `[1, 11]`, `[1, 11]` |
| SenseTime_deformable-detr-single-scale | transformers-auto-model | 3 | `[1, 3, 800, 800]`, `[1, 800, 800]`, `[1, 256]` |
| SenseTime_deformable-detr-single-scale-dc5 | transformers-auto-model | 3 | `[1, 3, 800, 800]`, `[1, 800, 800]`, `[1, 256]` |
| SmolLM3-3B | transformers-auto-model | 2 | `[1, 19]`, `[1, 19]` |
| SpaceCowboy03_finetuned-kde4-es-to-fr | transformers-auto-model | 3 | `[1, 16, 512]`, `[1, 16, 512]`, `[1, 16]` |
| TTian_deberta-classifier-feedback-1024-pseudo | transformers-auto-model | 2 | `[1, 11]`, `[1, 11]` |
| Thomasboosinger_owlvit-base-patch32 | transformers-auto-model | 4 | `[1, 3, 768, 768]`, `[1, 7]`, `[1, 7]`, `[576, 4]` |
| TinyLlama-1.1B-step-50K-105b | transformers-auto-model | 2 | `[1, 21]`, `[1, 21]` |
| TrustSafeAI_RADAR_Vicuna_7B | transformers-auto-model | 2 | `[1, 10]`, `[1, 10]` |
| Tucano-2b4 | transformers-auto-model | 2 | `[1, 26]`, `[1, 26]` |
| VitaliiVrublevskyi_ibert-roberta-base-finetuned-mrpc | transformers-auto-model | 2 | `[1, 11]`, `[1, 11]` |
| WIC-Uchile_ene_phase4 | transformers-auto-model | 3 | `[1, 18]`, `[1, 18]`, `[1, 18]` |
| YituTech_conv-bert-base | transformers-auto-model | 3 | `[1, 11]`, `[1, 11]`, `[1, 11]` |
| ZR1-1.5B | transformers-auto-model | 2 | `[1, 20]`, `[1, 20]` |
| aimarsg_prueba5 | transformers-auto-model | 2 | `[1, 13]`, `[1, 13]` |
| akdeniz27_bert-base-turkish-cased-ner | transformers-auto-model | 3 | `[1, 23]`, `[1, 23]`, `[1, 23]` |
| akhtet_mDeBERTa-v3-base-myanmar-xnli | transformers-auto-model | 2 | `[1, 13]`, `[1, 13]` |
| albert-base-v2 | transformers-auto-model | 3 | `[1, 12]`, `[1, 12]`, `[1, 12]` |
| albert_albert-base-v1 | transformers-auto-model | 3 | `[1, 12]`, `[1, 12]`, `[1, 12]` |
| albert_albert-large-v1 | transformers-auto-model | 3 | `[1, 12]`, `[1, 12]`, `[1, 12]` |
| albert_albert-large-v2 | transformers-auto-model | 3 | `[1, 12]`, `[1, 12]`, `[1, 12]` |
| albert_albert-xlarge-v1 | transformers-auto-model | 3 | `[1, 12]`, `[1, 12]`, `[1, 12]` |
| all-mpnet-base-v2 | transformers-auto-model | 2 | `[2, 7]`, `[2, 7]` |
| allegro_plt5-large | transformers-auto-model | 3 | `[1, 22, 1024]`, `[1, 22, 1024]`, `[1, 22]` |
| almanach_camembert-base | transformers-auto-model | 2 | `[1, 18]`, `[1, 18]` |
| amurienne_kellag-m2m100-v0.2 | transformers-auto-model | 4 | `[1, 15, 1024]`, `[1, 15, 1024]`, `[1, 15]`, `[1, 15]` |
| andersonbcdefg_seo-spam-classifier | transformers-auto-model | 2 | `[1, 11]`, `[1, 11]` |
| andyluan_emotion-detection | transformers-auto-model | 2 | `[1, 9, 1024]`, `[1, 9]` |
| anilguven_albert_tr_turkish_product_reviews | transformers-auto-model | 3 | `[1, 18]`, `[1, 18]`, `[1, 18]` |
| anilguven_electra_tr_turkish_product_reviews | transformers-auto-model | 3 | `[1, 23]`, `[1, 23]`, `[1, 23]` |
| apple_aimv2-large-patch14-224-lit | transformers-auto-model | 3 | `[1, 3, 224, 224]`, `[2, 7]`, `[2, 7]` |
| ashique_BanglaTraitBERT | transformers-auto-model | 3 | `[1, 21]`, `[1, 21]`, `[1, 21]` |
| aubmindlab_aragpt2-mega-detector-long | transformers-auto-model | 3 | `[1, 26]`, `[1, 26]`, `[1, 26]` |
| aurelio-ai_sr-test-clip | transformers-auto-model | 3 | `[1, 3, 30, 30]`, `[2, 10]`, `[2, 10]` |
| axotion_polish-reranker-base-ranknet | transformers-auto-model | 2 | `[1, 20]`, `[1, 20]` |
| bert-base-uncased | transformers-auto-model | 3 | `[1, 11]`, `[1, 11]`, `[1, 11]` |
| bge-base-en-v1.5 | transformers-auto-model | 2 | `[1, 10]`, `[1, 10]` |
| bge-large-en-v1.5 | transformers-auto-model | 2 | `[1, 10]`, `[1, 10]` |
| bge-large-zh | transformers-auto-model | 2 | `[1, 10]`, `[1, 10]` |
| bge-reranker-large | transformers-auto-model | 2 | `[1, 10]`, `[1, 10]` |
| bge-small-en | transformers-auto-model | 2 | `[1, 10]`, `[1, 10]` |
| bge-small-en-v1.5 | transformers-auto-model | 2 | `[1, 10]`, `[1, 10]` |
| bge-small-zh | transformers-auto-model | 2 | `[1, 10]`, `[1, 10]` |
| binery_Table_detection_MS_E_14 | transformers-auto-model | 2 | `[1, 3, 800, 800]`, `[1, 800, 800]` |
| bloom-560m | transformers-auto-model | 2 | `[1, 18]`, `[1, 18]` |
| bloom-tiny-random | transformers-auto-model | 2 | `[1, 18]`, `[1, 18]` |
| bloomz-1b1 | transformers-auto-model | 2 | `[1, 18]`, `[1, 18]` |
| bloomz-1b7 | transformers-auto-model | 2 | `[1, 18]`, `[1, 18]` |
| breadlicker45_autotrain-blender-50601120822 | transformers-auto-model | 3 | `[1, 15, 1280]`, `[1, 15, 1280]`, `[1, 15]` |
| cahya_t5-base-indonesian-summarization-cased | transformers-auto-model | 3 | `[1, 18, 768]`, `[1, 18, 768]`, `[1, 18]` |
| canary-1b-v2 | transformers-auto-model | 67 | `[1, 128, 521]`, `[1]`, `[1, 9999, 1024]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]`, `[8, 128]` |
| celloscopeai_celloscope-ner-banglabert-finetuned | transformers-auto-model | 3 | `[1, 21]`, `[1, 21]`, `[1, 21]` |
| ckiplab_albert-base-chinese-pos | transformers-auto-model | 3 | `[1, 18]`, `[1, 18]`, `[1, 18]` |
| ckiplab_albert-tiny-chinese-ner | transformers-auto-model | 3 | `[1, 18]`, `[1, 18]`, `[1, 18]` |
| cmarkea_bloomz-560m-nli | transformers-auto-model | 2 | `[1, 10, 1024]`, `[1, 10]` |
| cmarkea_distilcamembert-base-nli | transformers-auto-model | 2 | `[1, 18]`, `[1, 18]` |
| cointegrated_rubert-base-cased-nli-threeway | transformers-auto-model | 3 | `[1, 21]`, `[1, 21]`, `[1, 21]` |
| cointegrated_rut5-base-absum | transformers-auto-model | 3 | `[1, 14, 768]`, `[1, 14, 768]`, `[1, 14]` |
| csebuetnlp_mT5_multilingual_XLSum | transformers-auto-model | 3 | `[1, 12, 768]`, `[1, 12, 768]`, `[1, 12]` |
| dansk-gpt-wiki | transformers-auto-model | 2 | `[1, 30]`, `[1, 30]` |
| dbmdz_electra-large-discriminator-finetuned-conll03-english | transformers-auto-model | 3 | `[1, 11]`, `[1, 11]`, `[1, 11]` |
| dccuchile_albert-base-10-spanish-distilled-ner | transformers-auto-model | 3 | `[1, 17]`, `[1, 17]`, `[1, 17]` |
| dccuchile_albert-base-2-spanish-distilled-ner | transformers-auto-model | 3 | `[1, 17]`, `[1, 17]`, `[1, 17]` |
| dccuchile_albert-base-4-spanish-distilled-ner | transformers-auto-model | 3 | `[1, 17]`, `[1, 17]`, `[1, 17]` |
| dccuchile_albert-base-6-spanish-finetuned-ner | transformers-auto-model | 3 | `[1, 17]`, `[1, 17]`, `[1, 17]` |
| debabrata-ai_NepNER | transformers-auto-model | 2 | `[1, 26]`, `[1, 26]` |
| depth-anything_Depth-Anything-V2-Base-hf | transformers-auto-model | 4 | `[1, 1370, 768]`, `[1, 1370, 768]`, `[1, 1370, 768]`, `[1, 1370, 768]` |
| depth-anything_Depth-Anything-V2-Metric-Indoor-Small-hf | transformers-auto-model | 4 | `[1, 1370, 384]`, `[1, 1370, 384]`, `[1, 1370, 384]`, `[1, 1370, 384]` |
| depth-anything_Depth-Anything-V2-Metric-Outdoor-Base-hf | transformers-auto-model | 4 | `[1, 1370, 768]`, `[1, 1370, 768]`, `[1, 1370, 768]`, `[1, 1370, 768]` |
| depth-anything_Depth-Anything-V2-Metric-Outdoor-Large-hf | transformers-auto-model | 4 | `[1, 1370, 1024]`, `[1, 1370, 1024]`, `[1, 1370, 1024]`, `[1, 1370, 1024]` |
| depth-anything_Depth-Anything-V2-Metric-Outdoor-Small-hf | transformers-auto-model | 4 | `[1, 1370, 384]`, `[1, 1370, 384]`, `[1, 1370, 384]`, `[1, 1370, 384]` |
| depth-anything_prompt-depth-anything-vitl-hf | transformers-auto-model | 4 | `[1, 2917, 1024]`, `[1, 2917, 1024]`, `[1, 2917, 1024]`, `[1, 2917, 1024]` |
| depth-anything_prompt-depth-anything-vits-hf | transformers-auto-model | 4 | `[1, 2917, 384]`, `[1, 2917, 384]`, `[1, 2917, 384]`, `[1, 2917, 384]` |
| disi-unibo-nlp_openbioner-base | transformers-auto-model | 3 | `[1, 13]`, `[1, 13]`, `[1, 13]` |
| distilbert-base-uncased | transformers-auto-model | 2 | `[1, 4]`, `[1, 4]` |
| distilbert_distilbert-base-cased | transformers-auto-model | 2 | `[1, 13]`, `[1, 13]` |
| distilbert_distilbert-base-multilingual-cased | transformers-auto-model | 2 | `[1, 17]`, `[1, 17]` |
| distilbert_distilbert-base-uncased | transformers-auto-model | 2 | `[1, 11]`, `[1, 11]` |
| distilgpt2 | transformers-auto-model | 2 | `[1, 19]`, `[1, 19]` |
| distilroberta-base | transformers-auto-model | 2 | `[1, 11]`, `[1, 11]` |
| domenicrosati_BioM-ALBERT-xxlarge-finetuned-DAGPap22 | transformers-auto-model | 3 | `[1, 12]`, `[1, 12]`, `[1, 12]` |
| echarlaix_bart-base-cnn-r2-18.7-d23-hybrid | transformers-auto-model | 3 | `[1, 11, 768]`, `[1, 11, 768]`, `[1, 11]` |
| eliza-dukim_para-kqc-sim | transformers-auto-model | 3 | `[1, 15]`, `[1, 15]`, `[1, 15]` |
| emekaboris_autonlp-txc-17923129 | transformers-auto-model | 3 | `[1, 13]`, `[1, 13]`, `[1, 13]` |
| emfa_l-lectra-danish-finetuned-hatespeech | transformers-auto-model | 3 | `[1, 19]`, `[1, 19]`, `[1, 19]` |
| emrecan_convbert-base-turkish-mc4-cased-allnli_tr | transformers-auto-model | 3 | `[1, 23]`, `[1, 23]`, `[1, 23]` |
| erica_krm_sa2 | transformers-auto-model | 3 | `[1, 33]`, `[1, 33]`, `[1, 33]` |
| ethanyt_guwen-ner | transformers-auto-model | 3 | `[1, 10]`, `[1, 10]`, `[1, 10]` |
| fabiod20_italian-legal-ner | transformers-auto-model | 3 | `[1, 19]`, `[1, 19]`, `[1, 19]` |
| facebook_bart-large-cnn | transformers-auto-model | 3 | `[1, 11, 1024]`, `[1, 11, 1024]`, `[1, 11]` |
| facebook_detr-resnet-101 | transformers-auto-model | 2 | `[1, 3, 800, 800]`, `[1, 800, 800]` |
| facebook_detr-resnet-101-dc5 | transformers-auto-model | 2 | `[1, 3, 800, 800]`, `[1, 800, 800]` |
| facebook_detr-resnet-50 | transformers-auto-model | 2 | `[1, 3, 800, 800]`, `[1, 800, 800]` |
| facebook_dpr-ctx_encoder-single-nq-base | transformers-auto-model | 3 | `[1, 11]`, `[1, 11]`, `[1, 11]` |
| facebook_dpt-dinov2-giant-nyu | transformers-auto-model | 4 | `[1, 785, 1536]`, `[1, 785, 1536]`, `[1, 785, 1536]`, `[1, 785, 1536]` |
| facebook_dpt-dinov2-small-kitti | transformers-auto-model | 4 | `[1, 785, 384]`, `[1, 785, 384]`, `[1, 785, 384]`, `[1, 785, 384]` |
| facebook_esm2_t33_650M_UR50D | transformers-auto-model | 2 | `[1, 13]`, `[1, 13]` |
| facebook_metaclip-b16-400m | transformers-auto-model | 3 | `[1, 3, 224, 224]`, `[2, 7]`, `[2, 7]` |
| facebook_metaclip-b32-400m | transformers-auto-model | 3 | `[1, 3, 224, 224]`, `[2, 7]`, `[2, 7]` |
| facebook_xglm-564M | transformers-auto-model | 2 | `[1, 13]`, `[1, 13, 1024]` |
| fairseq-dense-1.3B | transformers-auto-model | 2 | `[1, 19]`, `[1, 19]` |
| fairseq-dense-355M | transformers-auto-model | 2 | `[1, 19]`, `[1, 19]` |
| falcon-tiny-random | transformers-auto-model | 2 | `[1, 19]`, `[1, 19]` |
| fgaim_tielectra-small-pos | transformers-auto-model | 3 | `[1, 22]`, `[1, 22]`, `[1, 22]` |
| flutter-painter_nllb-fra-fuf-v2 | transformers-auto-model | 4 | `[1, 16, 1024]`, `[1, 16, 1024]`, `[1, 16]`, `[1, 16]` |
| fusersam_Sentiment-Analysis-Model | transformers-auto-model | 2 | `[1, 11]`, `[1, 11]` |
| fushh7_llmdet_swin_tiny_hf | transformers-auto-model | 5 | `[1, 7, 256]`, `[1, 6, 900, 256]`, `[1, 900, 4]`, `[1, 6, 900, 4]`, `[1, 7]` |
| garNER_albert-tiny-spanish-es-LM | transformers-auto-model | 3 | `[1, 17]`, `[1, 17]`, `[1, 17]` |
| gemma-3-1b-pt | transformers-auto-model | 2 | `[1, 20]`, `[1, 20]` |
| gonchisi_roberta-base-bne-finetuned-new_or_used-title | transformers-auto-model | 2 | `[1, 17]`, `[1, 17]` |
| google-bert_bert-base-cased | transformers-auto-model | 3 | `[1, 13]`, `[1, 13]`, `[1, 13]` |
| google-bert_bert-base-chinese | transformers-auto-model | 3 | `[1, 18]`, `[1, 18]`, `[1, 18]` |
| google-bert_bert-base-multilingual-cased | transformers-auto-model | 3 | `[1, 17]`, `[1, 17]`, `[1, 17]` |
| google-bert_bert-large-uncased | transformers-auto-model | 3 | `[1, 11]`, `[1, 11]`, `[1, 11]` |
| google-t5_t5-base | transformers-auto-model | 3 | `[1, 12, 768]`, `[1, 12, 768]`, `[1, 12]` |
| google-t5_t5-large | transformers-auto-model | 3 | `[1, 12, 1024]`, `[1, 12, 1024]`, `[1, 12]` |
| google_byt5-small | transformers-auto-model | 3 | `[1, 50, 1472]`, `[1, 50, 1472]`, `[1, 50]` |
| google_byt5_base | transformers-auto-model | 3 | `[1, 1, 1536]`, `[1, 16, 1536]`, `[1, 16]` |
| google_byt5_small | transformers-auto-model | 3 | `[1, 1, 1472]`, `[1, 16, 1472]`, `[1, 16]` |
| google_electra-base-discriminator | transformers-auto-model | 3 | `[1, 11]`, `[1, 11]`, `[1, 11]` |
| google_electra-small-discriminator | transformers-auto-model | 3 | `[1, 11]`, `[1, 11]`, `[1, 11]` |
| google_electra_base_discriminator | transformers-auto-model | 3 | `[1, 16]`, `[1, 16]`, `[1, 16]` |
| google_electra_base_generator | transformers-auto-model | 3 | `[1, 16]`, `[1, 16]`, `[1, 16]` |
| google_electra_small_discriminator | transformers-auto-model | 3 | `[1, 16]`, `[1, 16]`, `[1, 16]` |
| google_flan_t5_base | transformers-auto-model | 3 | `[1, 1, 768]`, `[1, 16, 768]`, `[1, 16]` |
| google_flan_t5_large | transformers-auto-model | 3 | `[1, 1, 1024]`, `[1, 16, 1024]`, `[1, 16]` |
| google_flan_t5_small | transformers-auto-model | 3 | `[1, 1, 512]`, `[1, 16, 512]`, `[1, 16]` |
| google_fnet-base | transformers-auto-model | 2 | `[1, 13]`, `[1, 13]` |
| google_mobilebert-uncased | transformers-auto-model | 3 | `[1, 11]`, `[1, 11]`, `[1, 11]` |
| google_owlv2-base-patch16 | transformers-auto-model | 4 | `[1, 3, 960, 960]`, `[1, 7]`, `[1, 7]`, `[3600, 4]` |
| google_owlv2-large-patch14 | transformers-auto-model | 4 | `[1, 3, 1008, 1008]`, `[1, 7]`, `[1, 7]`, `[5184, 4]` |
| google_owlv2-large-patch14-finetuned | transformers-auto-model | 3 | `[1, 3, 1008, 1008]`, `[1, 7]`, `[1, 7]` |
| google_owlvit-base-patch16 | transformers-auto-model | 4 | `[1, 3, 768, 768]`, `[1, 7]`, `[1, 7]`, `[2304, 4]` |
| google_owlvit-base-patch32 | transformers-auto-model | 3 | `[1, 3, 768, 768]`, `[1, 7]`, `[1, 7]` |
| google_owlvit-large-patch14 | transformers-auto-model | 4 | `[1, 3, 840, 840]`, `[1, 7]`, `[1, 7]`, `[3600, 4]` |
| google_pegasus-cnn_dailymail | transformers-auto-model | 3 | `[1, 10, 1024]`, `[1, 10, 1024]`, `[1, 10]` |
| google_pegasus-xsum | transformers-auto-model | 2 | `[1, 10, 1024]`, `[1, 10, 1024]` |
| google_pix2struct-base | transformers-auto-model | 2 | `[1, 2048, 770]`, `[1, 1]` |
| google_rembert | transformers-auto-model | 3 | `[1, 13]`, `[1, 13]`, `[1, 13]` |
| google_t5-efficient-mini | transformers-auto-model | 3 | `[1, 12, 384]`, `[1, 12, 384]`, `[1, 12]` |
| google_t5-efficient-tiny | transformers-auto-model | 3 | `[1, 12, 256]`, `[1, 12, 256]`, `[1, 12]` |
| gowitheflowlab_clip-base-10240-checkpoint350 | transformers-auto-model | 3 | `[1, 3, 224, 224]`, `[2, 7]`, `[2, 7]` |
| gpssohi_distilbart-qgen-6-6 | transformers-auto-model | 3 | `[1, 11, 1024]`, `[1, 11, 1024]`, `[1, 11]` |
| gpt-sw3-356m | transformers-auto-model | 2 | `[1, 20]`, `[1, 20]` |
| gpt2 | transformers-auto-model | 2 | `[1, 19]`, `[1, 19]` |
| gsarti_it5-efficient-small-el32-wiki-summarization | transformers-auto-model | 3 | `[1, 15, 512]`, `[1, 15, 512]`, `[1, 15]` |
| hf-tiny-model-private_tiny-random-AlbertForSequenceClassification | transformers-auto-model | 3 | `[1, 12]`, `[1, 12]`, `[1, 12]` |
| hf-tiny-model-private_tiny-random-AltCLIPModel | transformers-auto-model | 3 | `[2, 17]`, `[2, 17]`, `[1, 3, 30, 30]` |
| hf-tiny-model-private_tiny-random-BartForSequenceClassification | transformers-auto-model | 3 | `[1, 22, 16]`, `[1, 22, 16]`, `[1, 22]` |
| hf-tiny-model-private_tiny-random-BertForSequenceClassification | transformers-auto-model | 3 | `[1, 45]`, `[1, 45]`, `[1, 45]` |
| hf-tiny-model-private_tiny-random-BigBirdForSequenceClassification | transformers-auto-model | 3 | `[1, 11]`, `[1, 11]`, `[1, 11]` |
| hf-tiny-model-private_tiny-random-BlipModel | transformers-auto-model | 3 | `[1, 3, 30, 30]`, `[2, 14]`, `[2, 14]` |
| hf-tiny-model-private_tiny-random-BloomForSequenceClassification | transformers-auto-model | 2 | `[1, 19, 32]`, `[1, 19]` |
| hf-tiny-model-private_tiny-random-CLIPSegModel | transformers-auto-model | 3 | `[1, 3, 30, 30]`, `[2, 10]`, `[2, 10]` |
| hf-tiny-model-private_tiny-random-CanineForSequenceClassification | transformers-auto-model | 3 | `[1, 51]`, `[1, 51]`, `[1, 51]` |
| hf-tiny-model-private_tiny-random-ConditionalDetrForObjectDetection | transformers-auto-model | 2 | `[1, 3, 800, 800]`, `[1, 800, 800]` |
| hf-tiny-model-private_tiny-random-ConvBertForSequenceClassification | transformers-auto-model | 3 | `[1, 45]`, `[1, 45]`, `[1, 45]` |
| hf-tiny-model-private_tiny-random-Data2VecTextForSequenceClassification | transformers-auto-model | 2 | `[1, 22]`, `[1, 22]` |
| hf-tiny-model-private_tiny-random-DebertaForSequenceClassification | transformers-auto-model | 3 | `[1, 22]`, `[1, 22]`, `[1, 22]` |
| hf-tiny-model-private_tiny-random-DebertaV2ForSequenceClassification | transformers-auto-model | 3 | `[1, 11]`, `[1, 11]`, `[1, 11]` |
| hf-tiny-model-private_tiny-random-DistilBertForSequenceClassification | transformers-auto-model | 2 | `[1, 45]`, `[1, 45]` |
| hf-tiny-model-private_tiny-random-ElectraForSequenceClassification | transformers-auto-model | 3 | `[1, 45]`, `[1, 45]`, `[1, 45]` |
| hf-tiny-model-private_tiny-random-ErnieForSequenceClassification | transformers-auto-model | 3 | `[1, 45]`, `[1, 45]`, `[1, 45]` |
| hf-tiny-model-private_tiny-random-EsmForSequenceClassification | transformers-auto-model | 2 | `[1, 13]`, `[1, 13]` |
| hf-tiny-model-private_tiny-random-FNetForSequenceClassification | transformers-auto-model | 2 | `[1, 13]`, `[1, 13]` |
| hf-tiny-model-private_tiny-random-FlaubertForSequenceClassification | transformers-auto-model | 4 | `[1, 17]`, `[1, 17]`, `[1, 512]`, `[1, 17]` |
| hf-tiny-model-private_tiny-random-GPTJForSequenceClassification | transformers-auto-model | 2 | `[1, 20, 32]`, `[1, 20]` |
| hf-tiny-model-private_tiny-random-LEDForSequenceClassification | transformers-auto-model | 3 | `[1, 22, 16]`, `[1, 22, 16]`, `[1, 22]` |
| hf-tiny-model-private_tiny-random-LukeForSequenceClassification | transformers-auto-model | 2 | `[1, 11]`, `[1, 11]` |
| hf-tiny-model-private_tiny-random-MBartForSequenceClassification | transformers-auto-model | 3 | `[1, 15, 16]`, `[1, 15, 16]`, `[1, 15]` |
| hf-tiny-model-private_tiny-random-MPNetForSequenceClassification | transformers-auto-model | 2 | `[1, 45]`, `[1, 45]` |
| hf-tiny-model-private_tiny-random-MegaForSequenceClassification | transformers-auto-model | 2 | `[1, 22]`, `[1, 22]` |
| hf-tiny-model-private_tiny-random-MegatronBertForSequenceClassification | transformers-auto-model | 3 | `[1, 45]`, `[1, 45]`, `[1, 45]` |
| hf-tiny-model-private_tiny-random-MobileBertForSequenceClassification | transformers-auto-model | 3 | `[1, 45]`, `[1, 45]`, `[1, 45]` |
| hf-tiny-model-private_tiny-random-MvpForSequenceClassification | transformers-auto-model | 3 | `[1, 22, 16]`, `[1, 22, 16]`, `[1, 22]` |
| hf-tiny-model-private_tiny-random-NezhaForSequenceClassification | transformers-auto-model | 3 | `[1, 45]`, `[1, 45]`, `[1, 45]` |
| hf-tiny-model-private_tiny-random-NystromformerForSequenceClassification | transformers-auto-model | 3 | `[1, 12]`, `[1, 12]`, `[1, 12]` |
| hf-tiny-model-private_tiny-random-OPTForSequenceClassification | transformers-auto-model | 2 | `[1, 21, 16]`, `[1, 21]` |
| hf-tiny-model-private_tiny-random-OpenAIGPTForSequenceClassification | transformers-auto-model | 2 | `[1, 43]`, `[1, 43]` |
| hf-tiny-model-private_tiny-random-OwlViTForObjectDetection | transformers-auto-model | 4 | `[1, 3, 32, 32]`, `[1, 10]`, `[1, 10]`, `[256, 4]` |
| hf-tiny-model-private_tiny-random-PLBartForSequenceClassification | transformers-auto-model | 3 | `[1, 11, 16]`, `[1, 11, 16]`, `[1, 11]` |
| hf-tiny-model-private_tiny-random-RemBertForSequenceClassification | transformers-auto-model | 3 | `[1, 13]`, `[1, 13]`, `[1, 13]` |
| hf-tiny-model-private_tiny-random-RobertaForSequenceClassification | transformers-auto-model | 2 | `[1, 22]`, `[1, 22]` |
| hf-tiny-model-private_tiny-random-RobertaPreLayerNormForSequenceClassification | transformers-auto-model | 2 | `[1, 22]`, `[1, 22]` |
| hfl_chinese-macbert-large | transformers-auto-model | 3 | `[1, 20]`, `[1, 20]`, `[1, 20]` |
| hfl_chinese-roberta-wwm-ext | transformers-auto-model | 3 | `[1, 20]`, `[1, 20]`, `[1, 20]` |
| hossamamer12_BUS15100_MB2_20epoch_notweettokenizer_fp16 | transformers-auto-model | 3 | `[1, 20]`, `[1, 20]`, `[1, 20]` |
| huggingface_CodeBERTa-small-v1 | transformers-auto-model | 2 | `[1, 12]`, `[1, 12]` |
| hyesunyun_NonsenseUpdateDiffIntBart | transformers-auto-model | 3 | `[1, 33, 1024]`, `[1, 33, 1024]`, `[1, 33]` |
| ibm-research_MoLFormer-XL-both-10pct | transformers-auto-model | 2 | `[1, 16, 768]`, `[1, 16]` |
| igorktech_rut5-small-chit-chat-intelligent | transformers-auto-model | 3 | `[1, 25, 512]`, `[1, 25, 512]`, `[1, 25]` |
| intelia-lab-uah_bloomz-560m_AE_SQAC | transformers-auto-model | 2 | `[1, 12]`, `[1, 12]` |
| inthedarkness_klue-roberta-small-cross-encoder | transformers-auto-model | 3 | `[1, 22]`, `[1, 22]`, `[1, 22]` |
| iryneko571_mt5-base-translation-ja_zh | transformers-auto-model | 3 | `[1, 12]`, `[1, 12]`, `[1, 12]` |
| iryneko571_mt5-small-translation-ja_zh | transformers-auto-model | 3 | `[1, 12]`, `[1, 12]`, `[1, 12]` |
| jackieliu930_bart-large-cnn-samsum | transformers-auto-model | 3 | `[1, 11, 1024]`, `[1, 11, 1024]`, `[1, 11]` |
| japanese-gpt-1b | transformers-auto-model | 2 | `[1, 39]`, `[1, 39]` |
| jhu-clsp_ettin-decoder-150m | transformers-auto-model | 2 | `[1, 10]`, `[1, 10]` |
| jinaai_jina-reranker-m0 | transformers-auto-model | 2 | `[1, 9]`, `[1, 9]` |
| joancipria_gpt2-base-bne-FineTunedEmoEvent | transformers-auto-model | 2 | `[1, 32]`, `[1, 32]` |
| joeddav_xlm-roberta-large-xnli | transformers-auto-model | 2 | `[1, 15]`, `[1, 15]` |
| kakaobank_kf-deberta-base | transformers-auto-model | 2 | `[1, 16]`, `[1, 16]` |
| kanak8278_xlnet-large-cased-ner-food-combined-weighted-v2 | transformers-auto-model | 3 | `[1, 13]`, `[1, 13]`, `[1, 13]` |
| kanarya-750m | transformers-auto-model | 2 | `[1, 36]`, `[1, 36]` |
| kannt-im_marian-finetuned-kde4-en-to-fr | transformers-auto-model | 3 | `[1, 13, 512]`, `[1, 13, 512]`, `[1, 13]` |
| kaveh_rclip | transformers-auto-model | 4 | `[1, 3, 224, 224]`, `[2, 7]`, `[2, 7]`, `[2, 7]` |
| klue_bert-base | transformers-auto-model | 3 | `[1, 22]`, `[1, 22]`, `[1, 22]` |
| kogpt | transformers-auto-model | 2 | `[1, 31]`, `[1, 31]` |
| laion_CLIP-ViT-H-14-laion2B-s32B-b79K | transformers-auto-model | 3 | `[1, 3, 224, 224]`, `[2, 7]`, `[2, 7]` |
| larryvrh_mt5-translation-ja_zh | transformers-auto-model | 3 | `[1, 12, 1024]`, `[1, 12, 1024]`, `[1, 12]` |
| lst-nectec_HoogBERTa-POS-lst20 | transformers-auto-model | 3 | `[1, 13]`, `[1, 13]`, `[1, 13]` |
| m3hrdadfi_albert-fa-base-v2-ner-peyma | transformers-auto-model | 3 | `[1, 17]`, `[1, 17]`, `[1, 17]` |
| mahwizzzz_UrduClassification | transformers-auto-model | 2 | `[1, 27]`, `[1, 27]` |
| mambarim-110m | transformers-auto-model | 2 | `[1, 27]`, `[1, 27]` |
| medicalai_ClinicalBERT | transformers-auto-model | 2 | `[1, 15]`, `[1, 15]` |
| mesolitica_ner-t5-small-standard-bahasa-cased | transformers-auto-model | 3 | `[1, 82]`, `[1, 82]`, `[1, 82]` |
| miangoar_esm2_t12_35M_UR50D-finetuned-secondary-structure-classification | transformers-auto-model | 2 | `[1, 13]`, `[1, 13]` |
| microsoft_conditional-detr-resnet-50 | transformers-auto-model | 2 | `[1, 3, 800, 800]`, `[1, 800, 800]` |
| microsoft_deberta-v3-base | transformers-auto-model | 2 | `[1, 11]`, `[1, 11]` |
| microsoft_deberta-v3-large | transformers-auto-model | 2 | `[1, 11]`, `[1, 11]` |
| microsoft_deberta-v3-small | transformers-auto-model | 2 | `[1, 11]`, `[1, 11]` |
| microsoft_layoutlm-base-uncased | transformers-auto-model | 3 | `[1, 11]`, `[1, 11]`, `[1, 11]` |
| microsoft_mpnet-base | transformers-auto-model | 2 | `[1, 11]`, `[1, 11]` |
| microsoft_table-transformer-detection | transformers-auto-model | 2 | `[1, 3, 800, 800]`, `[1, 800, 800]` |
| microsoft_table-transformer-structure-recognition | transformers-auto-model | 2 | `[1, 3, 800, 800]`, `[1, 800, 800]` |
| microsoft_trocr-base-printed | transformers-auto-model | 2 | `[1, 3, 384, 384]`, `[1, 1]` |
| microsoft_trocr-base-stage1 | transformers-auto-model | 2 | `[1, 3, 384, 384]`, `[1, 1]` |
| microsoft_trocr-large-printed | transformers-auto-model | 2 | `[1, 3, 384, 384]`, `[1, 1]` |
| microsoft_trocr-small-printed | transformers-auto-model | 2 | `[1, 3, 384, 384]`, `[1, 1]` |
| microsoft_xclip-base-patch16 | transformers-auto-model | 4 | `[1, 8, 3, 224, 224]`, `[768, 512]`, `[1, 8]`, `[1, 8]` |
| microsoft_xclip-base-patch16-zero-shot | transformers-auto-model | 4 | `[1, 32, 3, 224, 224]`, `[768, 512]`, `[1, 8]`, `[1, 8]` |
| microsoft_xclip-base-patch32 | transformers-auto-model | 4 | `[1, 8, 3, 224, 224]`, `[768, 512]`, `[1, 8]`, `[1, 8]` |
| microsoft_xclip-base-patch32-16-frames | transformers-auto-model | 4 | `[1, 16, 3, 224, 224]`, `[768, 512]`, `[1, 8]`, `[1, 8]` |
| microsoft_xclip-large-patch14 | transformers-auto-model | 4 | `[1, 8, 3, 224, 224]`, `[1024, 768]`, `[1, 8]`, `[1, 8]` |
| microsoft_xlm-align-base | transformers-auto-model | 2 | `[1, 15]`, `[1, 15]` |
| milistu_reranker-msmarco-v1.1-MiniLM-L12-H384-uncased-lambdaloss_v2 | transformers-auto-model | 3 | `[1, 11]`, `[1, 11]`, `[1, 11]` |
| mmarco-mMiniLMv2-L12-H384-v1 | transformers-auto-model | 2 | `[2, 36]`, `[2, 36]` |
| mohsenfayyaz_BERT_Warmup | transformers-auto-model | 2 | `[1, 11]`, `[1, 11]` |
| monoelectra-large | transformers-auto-model | 3 | `[2, 34]`, `[2, 34]`, `[2, 34]` |
| monologg_kocharelectra-base-kmounlp-ner | transformers-auto-model | 3 | `[1, 10]`, `[1, 10]`, `[1, 10]` |
| monologg_koelectra-small-finetuned-naver-ner | transformers-auto-model | 3 | `[1, 24]`, `[1, 24]`, `[1, 24]` |
| mooncakex_img2 | transformers-auto-model | 3 | `[1, 3, 384, 384]`, `[2, 7]`, `[2, 7]` |
| moska_plt5-seq-clf-with-entities-updated-finetuned | transformers-auto-model | 3 | `[1, 22, 512]`, `[1, 22, 512]`, `[1, 22]` |
| moussaKam_barthez-orangesum-abstract | transformers-auto-model | 3 | `[1, 16, 768]`, `[1, 16, 768]`, `[1, 16]` |
| ms-marco-MiniLM-L12-v2 | transformers-auto-model | 3 | `[2, 34]`, `[2, 34]`, `[2, 34]` |
| ms-marco-MiniLM-L2-v2 | transformers-auto-model | 3 | `[2, 34]`, `[2, 34]`, `[2, 34]` |
| ms-marco-MiniLM-L4-v2 | transformers-auto-model | 3 | `[2, 34]`, `[2, 34]`, `[2, 34]` |
| ms-marco-MiniLM-L6-v2 | transformers-auto-model | 3 | `[2, 34]`, `[2, 34]`, `[2, 34]` |
| ms-marco-TinyBERT-L2 | transformers-auto-model | 3 | `[2, 34]`, `[2, 34]`, `[2, 34]` |
| ms-marco-TinyBERT-L4 | transformers-auto-model | 3 | `[2, 34]`, `[2, 34]`, `[2, 34]` |
| ms-marco-TinyBERT-L6 | transformers-auto-model | 3 | `[2, 34]`, `[2, 34]`, `[2, 34]` |
| ms-marco-electra-base | transformers-auto-model | 3 | `[2, 34]`, `[2, 34]`, `[2, 34]` |
| msmarco-MiniLM-L12-en-de-v1 | transformers-auto-model | 2 | `[2, 36]`, `[2, 36]` |
| msmarco-MiniLM-L6-en-de-v1 | transformers-auto-model | 2 | `[2, 36]`, `[2, 36]` |
| muhtasham_medium-vanilla-target-tweet | transformers-auto-model | 3 | `[1, 11]`, `[1, 11]`, `[1, 11]` |
| muhtasham_mini-vanilla-target-tweet | transformers-auto-model | 3 | `[1, 11]`, `[1, 11]`, `[1, 11]` |
| muhtasham_small-vanilla-target-tweet | transformers-auto-model | 3 | `[1, 11]`, `[1, 11]`, `[1, 11]` |
| multilingual-e5-large | transformers-auto-model | 2 | `[1, 10]`, `[1, 10]` |
| napsternxg_nyt-ingredient-tagger-jina-embeddings-v2-small-en | transformers-auto-model | 3 | `[1, 11]`, `[1, 11]`, `[1, 11]` |
| navteca_bart-large-mnli | transformers-auto-model | 3 | `[1, 11, 1024]`, `[1, 11, 1024]`, `[1, 11]` |
| ncoop57_bart-base-code-summarizer-java-v0 | transformers-auto-model | 3 | `[1, 11, 768]`, `[1, 11, 768]`, `[1, 11]` |
| neuralbioinfo_prokbert-mini-c-promoter | transformers-auto-model | 3 | `[1, 51]`, `[1, 51]`, `[1, 51]` |
| nghuyong_ernie-2.0-base-en | transformers-auto-model | 3 | `[1, 11]`, `[1, 11]`, `[1, 11]` |
| nie3e_sentiment-polish-gpt2-large | transformers-auto-model | 2 | `[1, 32]`, `[1, 32]` |
| nli-MiniLM2-L6-H768 | transformers-auto-model | 2 | `[2, 35]`, `[2, 35]` |
| nli-deberta-v3-base | transformers-auto-model | 2 | `[2, 33]`, `[2, 33]` |
| nli-deberta-v3-small | transformers-auto-model | 2 | `[2, 33]`, `[2, 33]` |
| nli-deberta-v3-xsmall | transformers-auto-model | 2 | `[2, 33]`, `[2, 33]` |
| nli-roberta-base | transformers-auto-model | 2 | `[2, 35]`, `[2, 35]` |
| nlptown_bert_base_multilingual_uncased_sentiment | transformers-auto-model | 3 | `[1, 10]`, `[1, 10]`, `[1, 10]` |
| noamr_bert-finetuned-ner | transformers-auto-model | 2 | `[1, 10, 1024]`, `[1, 10]` |
| nreimers_MiniLM-L6-H384-uncased | transformers-auto-model | 3 | `[1, 11]`, `[1, 11]`, `[1, 11]` |
| ogoshi2000_stance-nystromformer | transformers-auto-model | 2 | `[1, 16]`, `[1, 16]` |
| openai_clip-vit-base-patch32 | transformers-auto-model | 3 | `[1, 3, 224, 224]`, `[2, 7]`, `[2, 7]` |
| openai_clip-vit-large-patch14 | transformers-auto-model | 3 | `[1, 3, 224, 224]`, `[2, 7]`, `[2, 7]` |
| openai_whisper-base | transformers-auto-model | 2 | `[1, 1, 512]`, `[1, 1500, 512]` |
| opt-350m-email-generation | transformers-auto-model | 2 | `[1, 19]`, `[1, 19]` |
| opt-tiny-random | transformers-auto-model | 2 | `[1, 20]`, `[1, 20]` |
| opus-mt-NORTH_EU-NORTH_EU | transformers-auto-model | 4 | `[1, 36]`, `[1, 36]`, `[1, 1]`, `[1, 58012]` |
| opus-mt-ROMANCE-en | transformers-auto-model | 4 | `[1, 31]`, `[1, 31]`, `[1, 1]`, `[1, 65001]` |
| opus-mt-SCANDINAVIA-SCANDINAVIA | transformers-auto-model | 4 | `[1, 33]`, `[1, 33]`, `[1, 1]`, `[1, 47231]` |
| opus-mt-aav-en | transformers-auto-model | 4 | `[1, 29]`, `[1, 29]`, `[1, 1]`, `[1, 56039]` |
| opus-mt-aed-es | transformers-auto-model | 4 | `[1, 39]`, `[1, 39]`, `[1, 1]`, `[1, 23625]` |
| opus-mt-af-de | transformers-auto-model | 4 | `[1, 32]`, `[1, 32]`, `[1, 1]`, `[1, 58757]` |
| opus-mt-af-en | transformers-auto-model | 4 | `[1, 30]`, `[1, 30]`, `[1, 1]`, `[1, 57445]` |
| opus-mt-af-eo | transformers-auto-model | 4 | `[1, 49]`, `[1, 49]`, `[1, 1]`, `[1, 7475]` |
| opus-mt-af-es | transformers-auto-model | 4 | `[1, 34]`, `[1, 34]`, `[1, 1]`, `[1, 60216]` |
| opus-mt-af-fi | transformers-auto-model | 4 | `[1, 32]`, `[1, 32]`, `[1, 1]`, `[1, 60657]` |
| opus-mt-af-fr | transformers-auto-model | 4 | `[1, 32]`, `[1, 32]`, `[1, 1]`, `[1, 59422]` |
| opus-mt-af-nl | transformers-auto-model | 4 | `[1, 34]`, `[1, 34]`, `[1, 1]`, `[1, 53247]` |
| opus-mt-af-ru | transformers-auto-model | 4 | `[1, 34]`, `[1, 34]`, `[1, 1]`, `[1, 62842]` |
| opus-mt-af-sv | transformers-auto-model | 4 | `[1, 32]`, `[1, 32]`, `[1, 1]`, `[1, 58463]` |
| opus-mt-afa-afa | transformers-auto-model | 4 | `[1, 44]`, `[1, 44]`, `[1, 1]`, `[1, 33714]` |
| opus-mt-afa-en | transformers-auto-model | 4 | `[1, 41]`, `[1, 41]`, `[1, 1]`, `[1, 61180]` |
| opus-mt-alv-en | transformers-auto-model | 4 | `[1, 39]`, `[1, 39]`, `[1, 1]`, `[1, 61549]` |
| opus-mt-am-sv | transformers-auto-model | 4 | `[1, 53]`, `[1, 53]`, `[1, 1]`, `[1, 63199]` |
| opus-mt-ar-de | transformers-auto-model | 4 | `[1, 51]`, `[1, 51]`, `[1, 1]`, `[1, 63079]` |
| opus-mt-ar-el | transformers-auto-model | 4 | `[1, 56]`, `[1, 56]`, `[1, 1]`, `[1, 63336]` |
| opus-mt-ar-en | transformers-auto-model | 4 | `[1, 39]`, `[1, 39]`, `[1, 1]`, `[1, 62834]` |
| opus-mt-ar-eo | transformers-auto-model | 4 | `[1, 64]`, `[1, 64]`, `[1, 1]`, `[1, 7741]` |
| opus-mt-ar-es | transformers-auto-model | 4 | `[1, 51]`, `[1, 51]`, `[1, 1]`, `[1, 62526]` |
| opus-mt-ar-fr | transformers-auto-model | 4 | `[1, 39]`, `[1, 39]`, `[1, 1]`, `[1, 64505]` |
| opus-mt-ar-he | transformers-auto-model | 4 | `[1, 52]`, `[1, 52]`, `[1, 1]`, `[1, 63334]` |
| opus-mt-ar-it | transformers-auto-model | 4 | `[1, 53]`, `[1, 53]`, `[1, 1]`, `[1, 63294]` |
| opus-mt-ar-pl | transformers-auto-model | 4 | `[1, 44]`, `[1, 44]`, `[1, 1]`, `[1, 65001]` |
| opus-mt-ar-ru | transformers-auto-model | 4 | `[1, 51]`, `[1, 51]`, `[1, 1]`, `[1, 62606]` |
| opus-mt-ar-tr | transformers-auto-model | 4 | `[1, 52]`, `[1, 52]`, `[1, 1]`, `[1, 63272]` |
| opus-mt-art-en | transformers-auto-model | 4 | `[1, 32]`, `[1, 32]`, `[1, 1]`, `[1, 59131]` |
| opus-mt-ase-de | transformers-auto-model | 4 | `[1, 21]`, `[1, 21]`, `[1, 1]`, `[1, 53676]` |
| opus-mt-ase-en | transformers-auto-model | 4 | `[1, 20]`, `[1, 20]`, `[1, 1]`, `[1, 24831]` |
| opus-mt-ase-es | transformers-auto-model | 4 | `[1, 21]`, `[1, 21]`, `[1, 1]`, `[1, 53502]` |
| opus-mt-ase-fr | transformers-auto-model | 4 | `[1, 21]`, `[1, 21]`, `[1, 1]`, `[1, 50895]` |
| opus-mt-ase-sv | transformers-auto-model | 4 | `[1, 22]`, `[1, 22]`, `[1, 1]`, `[1, 48742]` |
| opus-mt-az-es | transformers-auto-model | 4 | `[1, 55]`, `[1, 55]`, `[1, 1]`, `[1, 7590]` |
| opus-mt-az-tr | transformers-auto-model | 4 | `[1, 53]`, `[1, 53]`, `[1, 1]`, `[1, 6904]` |
| opus-mt-bat-en | transformers-auto-model | 4 | `[1, 38]`, `[1, 38]`, `[1, 1]`, `[1, 59056]` |
| opus-mt-bcl-de | transformers-auto-model | 4 | `[1, 36]`, `[1, 36]`, `[1, 1]`, `[1, 60738]` |
| opus-mt-bcl-en | transformers-auto-model | 4 | `[1, 36]`, `[1, 36]`, `[1, 1]`, `[1, 58455]` |
| opus-mt-bcl-es | transformers-auto-model | 4 | `[1, 36]`, `[1, 36]`, `[1, 1]`, `[1, 59736]` |
| opus-mt-bcl-fi | transformers-auto-model | 4 | `[1, 35]`, `[1, 35]`, `[1, 1]`, `[1, 61739]` |
| opus-mt-bcl-fr | transformers-auto-model | 4 | `[1, 35]`, `[1, 35]`, `[1, 1]`, `[1, 60484]` |
| opus-mt-bcl-sv | transformers-auto-model | 4 | `[1, 36]`, `[1, 36]`, `[1, 1]`, `[1, 59911]` |
| opus-mt-be-es | transformers-auto-model | 4 | `[1, 60]`, `[1, 60]`, `[1, 1]`, `[1, 7630]` |
| opus-mt-bem-en | transformers-auto-model | 4 | `[1, 43]`, `[1, 43]`, `[1, 1]`, `[1, 59828]` |
| opus-mt-bem-es | transformers-auto-model | 4 | `[1, 43]`, `[1, 43]`, `[1, 1]`, `[1, 60996]` |
| opus-mt-bem-fi | transformers-auto-model | 4 | `[1, 42]`, `[1, 42]`, `[1, 1]`, `[1, 61763]` |
| opus-mt-bem-fr | transformers-auto-model | 4 | `[1, 42]`, `[1, 42]`, `[1, 1]`, `[1, 61027]` |
| opus-mt-bem-sv | transformers-auto-model | 4 | `[1, 41]`, `[1, 41]`, `[1, 1]`, `[1, 60613]` |
| opus-mt-ber-en | transformers-auto-model | 4 | `[1, 32]`, `[1, 32]`, `[1, 1]`, `[1, 36048]` |
| opus-mt-ber-es | transformers-auto-model | 4 | `[1, 39]`, `[1, 39]`, `[1, 1]`, `[1, 13296]` |
| opus-mt-ber-fr | transformers-auto-model | 4 | `[1, 42]`, `[1, 42]`, `[1, 1]`, `[1, 20994]` |
| opus-mt-bg-de | transformers-auto-model | 4 | `[1, 41]`, `[1, 41]`, `[1, 1]`, `[1, 61110]` |
| opus-mt-bg-en | transformers-auto-model | 4 | `[1, 40]`, `[1, 40]`, `[1, 1]`, `[1, 61813]` |
| opus-mt-bg-eo | transformers-auto-model | 4 | `[1, 66]`, `[1, 66]`, `[1, 1]`, `[1, 7770]` |
| opus-mt-bg-es | transformers-auto-model | 4 | `[1, 46]`, `[1, 46]`, `[1, 1]`, `[1, 61845]` |
| opus-mt-bg-fi | transformers-auto-model | 4 | `[1, 39]`, `[1, 39]`, `[1, 1]`, `[1, 60069]` |
| opus-mt-bg-fr | transformers-auto-model | 4 | `[1, 44]`, `[1, 44]`, `[1, 1]`, `[1, 61483]` |
| opus-mt-bg-it | transformers-auto-model | 4 | `[1, 45]`, `[1, 45]`, `[1, 1]`, `[1, 61468]` |
| opus-mt-bg-ru | transformers-auto-model | 4 | `[1, 47]`, `[1, 47]`, `[1, 1]`, `[1, 57310]` |
| opus-mt-bg-sv | transformers-auto-model | 4 | `[1, 40]`, `[1, 40]`, `[1, 1]`, `[1, 59345]` |
| opus-mt-bg-tr | transformers-auto-model | 4 | `[1, 49]`, `[1, 49]`, `[1, 1]`, `[1, 63138]` |
| opus-mt-bg-uk | transformers-auto-model | 4 | `[1, 46]`, `[1, 46]`, `[1, 1]`, `[1, 59878]` |
| opus-mt-bi-en | transformers-auto-model | 4 | `[1, 29]`, `[1, 29]`, `[1, 1]`, `[1, 54321]` |
| opus-mt-bi-es | transformers-auto-model | 4 | `[1, 33]`, `[1, 33]`, `[1, 1]`, `[1, 56412]` |
| opus-mt-bi-fr | transformers-auto-model | 4 | `[1, 32]`, `[1, 32]`, `[1, 1]`, `[1, 57508]` |
| opus-mt-bi-sv | transformers-auto-model | 4 | `[1, 33]`, `[1, 33]`, `[1, 1]`, `[1, 55334]` |
| opus-mt-bn-en | transformers-auto-model | 4 | `[1, 40]`, `[1, 40]`, `[1, 1]`, `[1, 63597]` |
| opus-mt-bnt-en | transformers-auto-model | 4 | `[1, 41]`, `[1, 41]`, `[1, 1]`, `[1, 61729]` |
| opus-mt-bzs-en | transformers-auto-model | 4 | `[1, 39]`, `[1, 39]`, `[1, 1]`, `[1, 47919]` |
| opus-mt-bzs-es | transformers-auto-model | 4 | `[1, 40]`, `[1, 40]`, `[1, 1]`, `[1, 51421]` |
| opus-mt-bzs-fi | transformers-auto-model | 4 | `[1, 40]`, `[1, 40]`, `[1, 1]`, `[1, 59433]` |
| opus-mt-bzs-fr | transformers-auto-model | 4 | `[1, 39]`, `[1, 39]`, `[1, 1]`, `[1, 54231]` |
| opus-mt-bzs-sv | transformers-auto-model | 4 | `[1, 39]`, `[1, 39]`, `[1, 1]`, `[1, 51381]` |
| opus-mt-ca-de | transformers-auto-model | 4 | `[1, 40]`, `[1, 40]`, `[1, 1]`, `[1, 22116]` |
| opus-mt-ca-en | transformers-auto-model | 4 | `[1, 29]`, `[1, 29]`, `[1, 1]`, `[1, 55255]` |
| opus-mt-ca-es | transformers-auto-model | 4 | `[1, 38]`, `[1, 38]`, `[1, 1]`, `[1, 49621]` |
| opus-mt-ca-fr | transformers-auto-model | 4 | `[1, 41]`, `[1, 41]`, `[1, 1]`, `[1, 21410]` |
| opus-mt-ca-it | transformers-auto-model | 4 | `[1, 41]`, `[1, 41]`, `[1, 1]`, `[1, 21528]` |
| opus-mt-ca-nl | transformers-auto-model | 4 | `[1, 40]`, `[1, 40]`, `[1, 1]`, `[1, 21976]` |
| opus-mt-ca-pt | transformers-auto-model | 4 | `[1, 37]`, `[1, 37]`, `[1, 1]`, `[1, 20554]` |
| opus-mt-ca-uk | transformers-auto-model | 4 | `[1, 44]`, `[1, 44]`, `[1, 1]`, `[1, 7545]` |
| opus-mt-cau-en | transformers-auto-model | 4 | `[1, 57]`, `[1, 57]`, `[1, 1]`, `[1, 62393]` |
| opus-mt-ccs-en | transformers-auto-model | 4 | `[1, 61]`, `[1, 61]`, `[1, 1]`, `[1, 23173]` |
| opus-mt-ceb-en | transformers-auto-model | 4 | `[1, 33]`, `[1, 33]`, `[1, 1]`, `[1, 57239]` |
| opus-mt-ceb-es | transformers-auto-model | 4 | `[1, 33]`, `[1, 33]`, `[1, 1]`, `[1, 59324]` |
| opus-mt-ceb-fi | transformers-auto-model | 4 | `[1, 33]`, `[1, 33]`, `[1, 1]`, `[1, 61010]` |
| opus-mt-ceb-fr | transformers-auto-model | 4 | `[1, 34]`, `[1, 34]`, `[1, 1]`, `[1, 59770]` |
| opus-mt-ceb-sv | transformers-auto-model | 4 | `[1, 33]`, `[1, 33]`, `[1, 1]`, `[1, 59183]` |
| opus-mt-cel-en | transformers-auto-model | 4 | `[1, 33]`, `[1, 33]`, `[1, 1]`, `[1, 57907]` |
| opus-mt-chk-en | transformers-auto-model | 4 | `[1, 36]`, `[1, 36]`, `[1, 1]`, `[1, 56327]` |
| opus-mt-chk-es | transformers-auto-model | 4 | `[1, 36]`, `[1, 36]`, `[1, 1]`, `[1, 60011]` |
| opus-mt-chk-fr | transformers-auto-model | 4 | `[1, 36]`, `[1, 36]`, `[1, 1]`, `[1, 59610]` |
| opus-mt-chk-sv | transformers-auto-model | 4 | `[1, 37]`, `[1, 37]`, `[1, 1]`, `[1, 58904]` |
| opus-mt-cpf-en | transformers-auto-model | 4 | `[1, 30]`, `[1, 30]`, `[1, 1]`, `[1, 57679]` |
| opus-mt-cpp-cpp | transformers-auto-model | 4 | `[1, 40]`, `[1, 40]`, `[1, 1]`, `[1, 31989]` |
| opus-mt-cpp-en | transformers-auto-model | 4 | `[1, 36]`, `[1, 36]`, `[1, 1]`, `[1, 57790]` |
| opus-mt-crs-de | transformers-auto-model | 4 | `[1, 35]`, `[1, 35]`, `[1, 1]`, `[1, 47285]` |
| opus-mt-crs-en | transformers-auto-model | 4 | `[1, 32]`, `[1, 32]`, `[1, 1]`, `[1, 43620]` |
| opus-mt-crs-es | transformers-auto-model | 4 | `[1, 33]`, `[1, 33]`, `[1, 1]`, `[1, 47319]` |
| opus-mt-crs-fi | transformers-auto-model | 4 | `[1, 34]`, `[1, 34]`, `[1, 1]`, `[1, 48214]` |
| opus-mt-crs-fr | transformers-auto-model | 4 | `[1, 33]`, `[1, 33]`, `[1, 1]`, `[1, 46774]` |
| opus-mt-crs-sv | transformers-auto-model | 4 | `[1, 35]`, `[1, 35]`, `[1, 1]`, `[1, 46013]` |
| opus-mt-cs-de | transformers-auto-model | 4 | `[1, 36]`, `[1, 36]`, `[1, 1]`, `[1, 58234]` |
| opus-mt-cs-en | transformers-auto-model | 4 | `[1, 38]`, `[1, 38]`, `[1, 1]`, `[1, 62509]` |
| opus-mt-cs-eo | transformers-auto-model | 4 | `[1, 47]`, `[1, 47]`, `[1, 1]`, `[1, 7397]` |
| opus-mt-cs-fi | transformers-auto-model | 4 | `[1, 38]`, `[1, 38]`, `[1, 1]`, `[1, 59012]` |
| opus-mt-cs-fr | transformers-auto-model | 4 | `[1, 37]`, `[1, 37]`, `[1, 1]`, `[1, 58338]` |
| opus-mt-cs-sv | transformers-auto-model | 4 | `[1, 37]`, `[1, 37]`, `[1, 1]`, `[1, 57482]` |
| opus-mt-cs-uk | transformers-auto-model | 4 | `[1, 40]`, `[1, 40]`, `[1, 1]`, `[1, 62778]` |
| opus-mt-csg-es | transformers-auto-model | 4 | `[1, 41]`, `[1, 41]`, `[1, 1]`, `[1, 11802]` |
| opus-mt-csn-es | transformers-auto-model | 4 | `[1, 39]`, `[1, 39]`, `[1, 1]`, `[1, 28047]` |
| opus-mt-cy-en | transformers-auto-model | 4 | `[1, 27]`, `[1, 27]`, `[1, 1]`, `[1, 54395]` |
| opus-mt-da-de | transformers-auto-model | 4 | `[1, 36]`, `[1, 36]`, `[1, 1]`, `[1, 57027]` |
| opus-mt-da-en | transformers-auto-model | 4 | `[1, 36]`, `[1, 36]`, `[1, 1]`, `[1, 58930]` |
| opus-mt-da-eo | transformers-auto-model | 4 | `[1, 44]`, `[1, 44]`, `[1, 1]`, `[1, 7286]` |
| opus-mt-da-es | transformers-auto-model | 4 | `[1, 36]`, `[1, 36]`, `[1, 1]`, `[1, 57584]` |
| opus-mt-da-fi | transformers-auto-model | 4 | `[1, 36]`, `[1, 36]`, `[1, 1]`, `[1, 59606]` |
| opus-mt-da-fr | transformers-auto-model | 4 | `[1, 36]`, `[1, 36]`, `[1, 1]`, `[1, 57138]` |
| opus-mt-da-no | transformers-auto-model | 4 | `[1, 34]`, `[1, 34]`, `[1, 1]`, `[1, 18881]` |
| opus-mt-da-ru | transformers-auto-model | 4 | `[1, 34]`, `[1, 34]`, `[1, 1]`, `[1, 62922]` |
| opus-mt-de-ZH | transformers-auto-model | 4 | `[1, 31]`, `[1, 31]`, `[1, 1]`, `[1, 61916]` |
| opus-mt-de-af | transformers-auto-model | 4 | `[1, 38]`, `[1, 38]`, `[1, 1]`, `[1, 59299]` |
| opus-mt-de-ar | transformers-auto-model | 4 | `[1, 37]`, `[1, 37]`, `[1, 1]`, `[1, 63077]` |
| opus-mt-de-ase | transformers-auto-model | 4 | `[1, 45]`, `[1, 45]`, `[1, 1]`, `[1, 53676]` |
| opus-mt-de-bcl | transformers-auto-model | 4 | `[1, 40]`, `[1, 40]`, `[1, 1]`, `[1, 60738]` |
| opus-mt-de-bg | transformers-auto-model | 4 | `[1, 37]`, `[1, 37]`, `[1, 1]`, `[1, 61110]` |
| opus-mt-de-bi | transformers-auto-model | 4 | `[1, 40]`, `[1, 40]`, `[1, 1]`, `[1, 56367]` |
| opus-mt-de-bzs | transformers-auto-model | 4 | `[1, 44]`, `[1, 44]`, `[1, 1]`, `[1, 58819]` |
| opus-mt-de-ca | transformers-auto-model | 4 | `[1, 38]`, `[1, 38]`, `[1, 1]`, `[1, 22116]` |
| opus-mt-de-crs | transformers-auto-model | 4 | `[1, 43]`, `[1, 43]`, `[1, 1]`, `[1, 47285]` |
| opus-mt-de-cs | transformers-auto-model | 4 | `[1, 35]`, `[1, 35]`, `[1, 1]`, `[1, 58234]` |
| opus-mt-de-da | transformers-auto-model | 4 | `[1, 35]`, `[1, 35]`, `[1, 1]`, `[1, 57027]` |
| opus-mt-de-de | transformers-auto-model | 4 | `[1, 36]`, `[1, 36]`, `[1, 1]`, `[1, 33217]` |
| opus-mt-de-ee | transformers-auto-model | 4 | `[1, 40]`, `[1, 40]`, `[1, 1]`, `[1, 60346]` |
| opus-mt-de-efi | transformers-auto-model | 4 | `[1, 41]`, `[1, 41]`, `[1, 1]`, `[1, 59733]` |
| opus-mt-de-el | transformers-auto-model | 4 | `[1, 36]`, `[1, 36]`, `[1, 1]`, `[1, 59951]` |
| opus-mt-de-en | transformers-auto-model | 4 | `[1, 32]`, `[1, 32]`, `[1, 1]`, `[1, 58101]` |
| opus-mt-de-eo | transformers-auto-model | 4 | `[1, 32]`, `[1, 32]`, `[1, 1]`, `[1, 59836]` |
| opus-mt-de-es | transformers-auto-model | 4 | `[1, 36]`, `[1, 36]`, `[1, 1]`, `[1, 61301]` |
| opus-mt-de-et | transformers-auto-model | 4 | `[1, 35]`, `[1, 35]`, `[1, 1]`, `[1, 58042]` |
| opus-mt-de-eu | transformers-auto-model | 4 | `[1, 40]`, `[1, 40]`, `[1, 1]`, `[1, 22170]` |
| opus-mt-de-fi | transformers-auto-model | 4 | `[1, 37]`, `[1, 37]`, `[1, 1]`, `[1, 59971]` |
| opus-mt-de-fj | transformers-auto-model | 4 | `[1, 40]`, `[1, 40]`, `[1, 1]`, `[1, 59804]` |
| opus-mt-de-fr | transformers-auto-model | 4 | `[1, 36]`, `[1, 36]`, `[1, 1]`, `[1, 61153]` |
| opus-mt-de-gaa | transformers-auto-model | 4 | `[1, 43]`, `[1, 43]`, `[1, 1]`, `[1, 60424]` |
| opus-mt-de-gil | transformers-auto-model | 4 | `[1, 43]`, `[1, 43]`, `[1, 1]`, `[1, 53500]` |
| opus-mt-de-guw | transformers-auto-model | 4 | `[1, 42]`, `[1, 42]`, `[1, 1]`, `[1, 54544]` |
| opus-mt-de-ha | transformers-auto-model | 4 | `[1, 40]`, `[1, 40]`, `[1, 1]`, `[1, 59664]` |
| opus-mt-de-he | transformers-auto-model | 4 | `[1, 32]`, `[1, 32]`, `[1, 1]`, `[1, 63066]` |
| opus-mt-de-hil | transformers-auto-model | 4 | `[1, 41]`, `[1, 41]`, `[1, 1]`, `[1, 60041]` |
| opus-mt-de-ho | transformers-auto-model | 4 | `[1, 43]`, `[1, 43]`, `[1, 1]`, `[1, 45292]` |
| opus-mt-de-hr | transformers-auto-model | 4 | `[1, 35]`, `[1, 35]`, `[1, 1]`, `[1, 58701]` |
| opus-mt-de-ht | transformers-auto-model | 4 | `[1, 38]`, `[1, 38]`, `[1, 1]`, `[1, 49000]` |
| opus-mt-de-hu | transformers-auto-model | 4 | `[1, 36]`, `[1, 36]`, `[1, 1]`, `[1, 58671]` |
| opus-mt-de-ig | transformers-auto-model | 4 | `[1, 38]`, `[1, 38]`, `[1, 1]`, `[1, 60025]` |
| opus-mt-de-ilo | transformers-auto-model | 4 | `[1, 41]`, `[1, 41]`, `[1, 1]`, `[1, 60194]` |
| opus-mt-de-is | transformers-auto-model | 4 | `[1, 39]`, `[1, 39]`, `[1, 1]`, `[1, 59315]` |
| opus-mt-de-iso | transformers-auto-model | 4 | `[1, 42]`, `[1, 42]`, `[1, 1]`, `[1, 47660]` |
| opus-mt-de-it | transformers-auto-model | 4 | `[1, 36]`, `[1, 36]`, `[1, 1]`, `[1, 59867]` |
| opus-mt-de-kg | transformers-auto-model | 4 | `[1, 38]`, `[1, 38]`, `[1, 1]`, `[1, 49964]` |
| opus-mt-de-ln | transformers-auto-model | 4 | `[1, 41]`, `[1, 41]`, `[1, 1]`, `[1, 60592]` |
| opus-mt-de-loz | transformers-auto-model | 4 | `[1, 40]`, `[1, 40]`, `[1, 1]`, `[1, 60846]` |
| opus-mt-de-lt | transformers-auto-model | 4 | `[1, 35]`, `[1, 35]`, `[1, 1]`, `[1, 58862]` |
| opus-mt-de-lua | transformers-auto-model | 4 | `[1, 39]`, `[1, 39]`, `[1, 1]`, `[1, 60920]` |
| opus-mt-de-ms | transformers-auto-model | 4 | `[1, 39]`, `[1, 39]`, `[1, 1]`, `[1, 57716]` |
| opus-mt-de-mt | transformers-auto-model | 4 | `[1, 37]`, `[1, 37]`, `[1, 1]`, `[1, 57955]` |
| opus-mt-de-niu | transformers-auto-model | 4 | `[1, 44]`, `[1, 44]`, `[1, 1]`, `[1, 44948]` |
| opus-mt-de-nl | transformers-auto-model | 4 | `[1, 36]`, `[1, 36]`, `[1, 1]`, `[1, 57567]` |
| opus-mt-de-no | transformers-auto-model | 4 | `[1, 47]`, `[1, 47]`, `[1, 1]`, `[1, 7020]` |
| opus-mt-de-nso | transformers-auto-model | 4 | `[1, 38]`, `[1, 38]`, `[1, 1]`, `[1, 60441]` |
| opus-mt-de-ny | transformers-auto-model | 4 | `[1, 41]`, `[1, 41]`, `[1, 1]`, `[1, 60883]` |
| opus-mt-de-pag | transformers-auto-model | 4 | `[1, 40]`, `[1, 40]`, `[1, 1]`, `[1, 60768]` |
| opus-mt-de-pap | transformers-auto-model | 4 | `[1, 41]`, `[1, 41]`, `[1, 1]`, `[1, 60181]` |
| opus-mt-de-pis | transformers-auto-model | 4 | `[1, 41]`, `[1, 41]`, `[1, 1]`, `[1, 48977]` |
| opus-mt-de-pl | transformers-auto-model | 4 | `[1, 36]`, `[1, 36]`, `[1, 1]`, `[1, 60431]` |
| opus-mt-de-pon | transformers-auto-model | 4 | `[1, 42]`, `[1, 42]`, `[1, 1]`, `[1, 56761]` |
| opus-mt-de-tl | transformers-auto-model | 4 | `[1, 41]`, `[1, 41]`, `[1, 1]`, `[1, 60346]` |
| opus-mt-de-uk | transformers-auto-model | 4 | `[1, 37]`, `[1, 37]`, `[1, 1]`, `[1, 62523]` |
| opus-mt-de-vi | transformers-auto-model | 4 | `[1, 38]`, `[1, 38]`, `[1, 1]`, `[1, 58225]` |
| opus-mt-dra-en | transformers-auto-model | 4 | `[1, 49]`, `[1, 49]`, `[1, 1]`, `[1, 62943]` |
| opus-mt-ee-en | transformers-auto-model | 4 | `[1, 38]`, `[1, 38]`, `[1, 1]`, `[1, 57579]` |
| opus-mt-ee-es | transformers-auto-model | 4 | `[1, 39]`, `[1, 39]`, `[1, 1]`, `[1, 60163]` |
| opus-mt-ee-fi | transformers-auto-model | 4 | `[1, 41]`, `[1, 41]`, `[1, 1]`, `[1, 61449]` |
| opus-mt-ee-fr | transformers-auto-model | 4 | `[1, 39]`, `[1, 39]`, `[1, 1]`, `[1, 60218]` |
| opus-mt-ee-sv | transformers-auto-model | 4 | `[1, 40]`, `[1, 40]`, `[1, 1]`, `[1, 59678]` |
| opus-mt-efi-de | transformers-auto-model | 4 | `[1, 36]`, `[1, 36]`, `[1, 1]`, `[1, 59733]` |
| opus-mt-efi-en | transformers-auto-model | 4 | `[1, 32]`, `[1, 32]`, `[1, 1]`, `[1, 55090]` |
| opus-mt-efi-fi | transformers-auto-model | 4 | `[1, 36]`, `[1, 36]`, `[1, 1]`, `[1, 61193]` |
| opus-mt-efi-fr | transformers-auto-model | 4 | `[1, 33]`, `[1, 33]`, `[1, 1]`, `[1, 59481]` |
| opus-mt-efi-sv | transformers-auto-model | 4 | `[1, 37]`, `[1, 37]`, `[1, 1]`, `[1, 59076]` |
| opus-mt-el-ar | transformers-auto-model | 4 | `[1, 44]`, `[1, 44]`, `[1, 1]`, `[1, 63305]` |
| opus-mt-el-eo | transformers-auto-model | 4 | `[1, 55]`, `[1, 55]`, `[1, 1]`, `[1, 7713]` |
| opus-mt-el-fi | transformers-auto-model | 4 | `[1, 39]`, `[1, 39]`, `[1, 1]`, `[1, 59727]` |
| opus-mt-el-fr | transformers-auto-model | 4 | `[1, 40]`, `[1, 40]`, `[1, 1]`, `[1, 60763]` |
| opus-mt-el-sv | transformers-auto-model | 4 | `[1, 43]`, `[1, 43]`, `[1, 1]`, `[1, 59836]` |
| opus-mt-en-CELTIC | transformers-auto-model | 4 | `[1, 29]`, `[1, 29]`, `[1, 1]`, `[1, 9001]` |
| opus-mt-en-ROMANCE | transformers-auto-model | 4 | `[1, 20]`, `[1, 20]`, `[1, 1]`, `[1, 65001]` |
| opus-mt-en-aav | transformers-auto-model | 4 | `[1, 20]`, `[1, 20]`, `[1, 1]`, `[1, 56071]` |
| opus-mt-en-af | transformers-auto-model | 4 | `[1, 20]`, `[1, 20]`, `[1, 1]`, `[1, 57445]` |
| opus-mt-en-afa | transformers-auto-model | 4 | `[1, 20]`, `[1, 20]`, `[1, 1]`, `[1, 61199]` |
| opus-mt-en-alv | transformers-auto-model | 4 | `[1, 21]`, `[1, 21]`, `[1, 1]`, `[1, 61577]` |
| opus-mt-en-ar | transformers-auto-model | 4 | `[1, 20]`, `[1, 20]`, `[1, 1]`, `[1, 62802]` |
| opus-mt-en-az | transformers-auto-model | 4 | `[1, 26]`, `[1, 26]`, `[1, 1]`, `[1, 23288]` |
| opus-mt-en-bat | transformers-auto-model | 4 | `[1, 20]`, `[1, 20]`, `[1, 1]`, `[1, 59061]` |
| opus-mt-en-bcl | transformers-auto-model | 4 | `[1, 21]`, `[1, 21]`, `[1, 1]`, `[1, 58455]` |
| opus-mt-en-bem | transformers-auto-model | 4 | `[1, 21]`, `[1, 21]`, `[1, 1]`, `[1, 59828]` |
| opus-mt-en-ber | transformers-auto-model | 4 | `[1, 22]`, `[1, 22]`, `[1, 1]`, `[1, 36048]` |
| opus-mt-en-bg | transformers-auto-model | 4 | `[1, 20]`, `[1, 20]`, `[1, 1]`, `[1, 61708]` |
| opus-mt-en-bi | transformers-auto-model | 4 | `[1, 20]`, `[1, 20]`, `[1, 1]`, `[1, 54321]` |
| opus-mt-en-bnt | transformers-auto-model | 4 | `[1, 21]`, `[1, 21]`, `[1, 1]`, `[1, 61740]` |
| opus-mt-en-bzs | transformers-auto-model | 4 | `[1, 22]`, `[1, 22]`, `[1, 1]`, `[1, 47919]` |
| opus-mt-en-ca | transformers-auto-model | 4 | `[1, 20]`, `[1, 20]`, `[1, 1]`, `[1, 55255]` |
| opus-mt-en-ceb | transformers-auto-model | 4 | `[1, 21]`, `[1, 21]`, `[1, 1]`, `[1, 56824]` |
| opus-mt-en-cel | transformers-auto-model | 4 | `[1, 20]`, `[1, 20]`, `[1, 1]`, `[1, 57912]` |
| opus-mt-en-chk | transformers-auto-model | 4 | `[1, 21]`, `[1, 21]`, `[1, 1]`, `[1, 56327]` |
| opus-mt-en-cpf | transformers-auto-model | 4 | `[1, 21]`, `[1, 21]`, `[1, 1]`, `[1, 57681]` |
| opus-mt-en-cpp | transformers-auto-model | 4 | `[1, 20]`, `[1, 20]`, `[1, 1]`, `[1, 57766]` |
| opus-mt-en-crs | transformers-auto-model | 4 | `[1, 21]`, `[1, 21]`, `[1, 1]`, `[1, 43620]` |
| opus-mt-en-cs | transformers-auto-model | 4 | `[1, 22]`, `[1, 22]`, `[1, 1]`, `[1, 62509]` |
| opus-mt-en-cy | transformers-auto-model | 4 | `[1, 21]`, `[1, 21]`, `[1, 1]`, `[1, 54395]` |
| opus-mt-en-da | transformers-auto-model | 4 | `[1, 22]`, `[1, 22]`, `[1, 1]`, `[1, 58930]` |
| opus-mt-en-de | transformers-auto-model | 4 | `[1, 20]`, `[1, 20]`, `[1, 1]`, `[1, 58101]` |
| opus-mt-en-dra | transformers-auto-model | 4 | `[1, 20]`, `[1, 20]`, `[1, 1]`, `[1, 62952]` |
| opus-mt-en-ee | transformers-auto-model | 4 | `[1, 21]`, `[1, 21]`, `[1, 1]`, `[1, 57579]` |
| opus-mt-en-efi | transformers-auto-model | 4 | `[1, 21]`, `[1, 21]`, `[1, 1]`, `[1, 55090]` |
| opus-mt-en-el | transformers-auto-model | 4 | `[1, 22]`, `[1, 22]`, `[1, 1]`, `[1, 64826]` |
| opus-mt-en-eo | transformers-auto-model | 4 | `[1, 20]`, `[1, 20]`, `[1, 1]`, `[1, 59197]` |
| opus-mt-en-et | transformers-auto-model | 4 | `[1, 21]`, `[1, 21]`, `[1, 1]`, `[1, 58866]` |
| opus-mt-en-eu | transformers-auto-model | 4 | `[1, 21]`, `[1, 21]`, `[1, 1]`, `[1, 58286]` |
| opus-mt-en-euq | transformers-auto-model | 4 | `[1, 22]`, `[1, 22]`, `[1, 1]`, `[1, 21964]` |
| opus-mt-en-fiu | transformers-auto-model | 4 | `[1, 20]`, `[1, 20]`, `[1, 1]`, `[1, 58938]` |
| opus-mt-en-fj | transformers-auto-model | 4 | `[1, 20]`, `[1, 20]`, `[1, 1]`, `[1, 55835]` |
| opus-mt-en-ga | transformers-auto-model | 4 | `[1, 20]`, `[1, 20]`, `[1, 1]`, `[1, 56683]` |
| opus-mt-en-gaa | transformers-auto-model | 4 | `[1, 21]`, `[1, 21]`, `[1, 1]`, `[1, 57186]` |
| opus-mt-en-gem | transformers-auto-model | 4 | `[1, 20]`, `[1, 20]`, `[1, 1]`, `[1, 56629]` |
| opus-mt-en-gil | transformers-auto-model | 4 | `[1, 20]`, `[1, 20]`, `[1, 1]`, `[1, 50021]` |
| opus-mt-en-gl | transformers-auto-model | 4 | `[1, 20]`, `[1, 20]`, `[1, 1]`, `[1, 55458]` |
| opus-mt-en-gmq | transformers-auto-model | 4 | `[1, 20]`, `[1, 20]`, `[1, 1]`, `[1, 57068]` |
| opus-mt-en-gmw | transformers-auto-model | 4 | `[1, 20]`, `[1, 20]`, `[1, 1]`, `[1, 55477]` |
| opus-mt-en-guw | transformers-auto-model | 4 | `[1, 21]`, `[1, 21]`, `[1, 1]`, `[1, 52253]` |
| opus-mt-en-gv | transformers-auto-model | 4 | `[1, 29]`, `[1, 29]`, `[1, 1]`, `[1, 12339]` |
| opus-mt-en-ha | transformers-auto-model | 4 | `[1, 21]`, `[1, 21]`, `[1, 1]`, `[1, 58089]` |
| opus-mt-en-he | transformers-auto-model | 4 | `[1, 20]`, `[1, 20]`, `[1, 1]`, `[1, 65839]` |
| opus-mt-en-hi | transformers-auto-model | 4 | `[1, 20]`, `[1, 20]`, `[1, 1]`, `[1, 61950]` |
| opus-mt-en-hil | transformers-auto-model | 4 | `[1, 21]`, `[1, 21]`, `[1, 1]`, `[1, 56301]` |
| opus-mt-en-ho | transformers-auto-model | 4 | `[1, 21]`, `[1, 21]`, `[1, 1]`, `[1, 42463]` |
| opus-mt-en-ht | transformers-auto-model | 4 | `[1, 20]`, `[1, 20]`, `[1, 1]`, `[1, 47809]` |
| opus-mt-en-hu | transformers-auto-model | 4 | `[1, 21]`, `[1, 21]`, `[1, 1]`, `[1, 62522]` |
| opus-mt-en-hy | transformers-auto-model | 4 | `[1, 29]`, `[1, 29]`, `[1, 1]`, `[1, 7754]` |
| opus-mt-en-id | transformers-auto-model | 4 | `[1, 20]`, `[1, 20]`, `[1, 1]`, `[1, 54796]` |
| opus-mt-en-ig | transformers-auto-model | 4 | `[1, 21]`, `[1, 21]`, `[1, 1]`, `[1, 56127]` |
| opus-mt-en-iir | transformers-auto-model | 4 | `[1, 20]`, `[1, 20]`, `[1, 1]`, `[1, 62228]` |
| opus-mt-en-ilo | transformers-auto-model | 4 | `[1, 21]`, `[1, 21]`, `[1, 1]`, `[1, 57777]` |
| opus-mt-en-inc | transformers-auto-model | 4 | `[1, 20]`, `[1, 20]`, `[1, 1]`, `[1, 61760]` |
| opus-mt-en-ine | transformers-auto-model | 4 | `[1, 20]`, `[1, 20]`, `[1, 1]`, `[1, 61161]` |
| opus-mt-en-is | transformers-auto-model | 4 | `[1, 20]`, `[1, 20]`, `[1, 1]`, `[1, 58647]` |
| opus-mt-en-iso | transformers-auto-model | 4 | `[1, 21]`, `[1, 21]`, `[1, 1]`, `[1, 45029]` |
| opus-mt-en-it | transformers-auto-model | 4 | `[1, 22]`, `[1, 22]`, `[1, 1]`, `[1, 80035]` |
| opus-mt-en-itc | transformers-auto-model | 4 | `[1, 20]`, `[1, 20]`, `[1, 1]`, `[1, 57148]` |
| opus-mt-en-jap | transformers-auto-model | 4 | `[1, 28]`, `[1, 28]`, `[1, 1]`, `[1, 46276]` |
| opus-mt-en-kg | transformers-auto-model | 4 | `[1, 21]`, `[1, 21]`, `[1, 1]`, `[1, 47322]` |
| opus-mt-en-kj | transformers-auto-model | 4 | `[1, 20]`, `[1, 20]`, `[1, 1]`, `[1, 50183]` |
| opus-mt-en-kqn | transformers-auto-model | 4 | `[1, 21]`, `[1, 21]`, `[1, 1]`, `[1, 60142]` |
| opus-mt-en-kwn | transformers-auto-model | 4 | `[1, 21]`, `[1, 21]`, `[1, 1]`, `[1, 50346]` |
| opus-mt-en-kwy | transformers-auto-model | 4 | `[1, 21]`, `[1, 21]`, `[1, 1]`, `[1, 58115]` |
| opus-mt-en-lg | transformers-auto-model | 4 | `[1, 21]`, `[1, 21]`, `[1, 1]`, `[1, 60447]` |
| opus-mt-en-ln | transformers-auto-model | 4 | `[1, 21]`, `[1, 21]`, `[1, 1]`, `[1, 58314]` |
| opus-mt-en-loz | transformers-auto-model | 4 | `[1, 21]`, `[1, 21]`, `[1, 1]`, `[1, 57974]` |
| opus-mt-en-lu | transformers-auto-model | 4 | `[1, 21]`, `[1, 21]`, `[1, 1]`, `[1, 60241]` |
| opus-mt-en-lua | transformers-auto-model | 4 | `[1, 21]`, `[1, 21]`, `[1, 1]`, `[1, 59447]` |
| opus-mt-en-lue | transformers-auto-model | 4 | `[1, 21]`, `[1, 21]`, `[1, 1]`, `[1, 61313]` |
| opus-mt-en-lun | transformers-auto-model | 4 | `[1, 21]`, `[1, 21]`, `[1, 1]`, `[1, 56530]` |
| opus-mt-en-luo | transformers-auto-model | 4 | `[1, 21]`, `[1, 21]`, `[1, 1]`, `[1, 52236]` |
| opus-mt-en-lus | transformers-auto-model | 4 | `[1, 21]`, `[1, 21]`, `[1, 1]`, `[1, 49567]` |
| opus-mt-en-map | transformers-auto-model | 4 | `[1, 20]`, `[1, 20]`, `[1, 1]`, `[1, 60088]` |
| opus-mt-en-mfe | transformers-auto-model | 4 | `[1, 21]`, `[1, 21]`, `[1, 1]`, `[1, 48883]` |
| opus-mt-en-mg | transformers-auto-model | 4 | `[1, 20]`, `[1, 20]`, `[1, 1]`, `[1, 54688]` |
| opus-mt-en-mh | transformers-auto-model | 4 | `[1, 21]`, `[1, 21]`, `[1, 1]`, `[1, 49657]` |
| opus-mt-en-mk | transformers-auto-model | 4 | `[1, 20]`, `[1, 20]`, `[1, 1]`, `[1, 61946]` |
| opus-mt-en-mkh | transformers-auto-model | 4 | `[1, 20]`, `[1, 20]`, `[1, 1]`, `[1, 56075]` |
| opus-mt-en-ml | transformers-auto-model | 4 | `[1, 22]`, `[1, 22]`, `[1, 1]`, `[1, 24661]` |
| opus-mt-en-mos | transformers-auto-model | 4 | `[1, 21]`, `[1, 21]`, `[1, 1]`, `[1, 47022]` |
| opus-mt-en-mr | transformers-auto-model | 4 | `[1, 20]`, `[1, 20]`, `[1, 1]`, `[1, 61674]` |
| opus-mt-en-mt | transformers-auto-model | 4 | `[1, 22]`, `[1, 22]`, `[1, 1]`, `[1, 56041]` |
| opus-mt-en-mul | transformers-auto-model | 4 | `[1, 20]`, `[1, 20]`, `[1, 1]`, `[1, 64110]` |
| opus-mt-en-ng | transformers-auto-model | 4 | `[1, 21]`, `[1, 21]`, `[1, 1]`, `[1, 49883]` |
| opus-mt-en-nic | transformers-auto-model | 4 | `[1, 21]`, `[1, 21]`, `[1, 1]`, `[1, 61556]` |
| opus-mt-en-niu | transformers-auto-model | 4 | `[1, 21]`, `[1, 21]`, `[1, 1]`, `[1, 42155]` |
| opus-mt-en-nl | transformers-auto-model | 4 | `[1, 22]`, `[1, 22]`, `[1, 1]`, `[1, 67028]` |
| opus-mt-en-nso | transformers-auto-model | 4 | `[1, 21]`, `[1, 21]`, `[1, 1]`, `[1, 57542]` |
| opus-mt-en-ny | transformers-auto-model | 4 | `[1, 21]`, `[1, 21]`, `[1, 1]`, `[1, 59811]` |
| opus-mt-en-nyk | transformers-auto-model | 4 | `[1, 21]`, `[1, 21]`, `[1, 1]`, `[1, 52867]` |
| opus-mt-en-om | transformers-auto-model | 4 | `[1, 21]`, `[1, 21]`, `[1, 1]`, `[1, 59898]` |
| opus-mt-en-pag | transformers-auto-model | 4 | `[1, 21]`, `[1, 21]`, `[1, 1]`, `[1, 57487]` |
| opus-mt-en-pap | transformers-auto-model | 4 | `[1, 21]`, `[1, 21]`, `[1, 1]`, `[1, 57294]` |
| opus-mt-en-phi | transformers-auto-model | 4 | `[1, 21]`, `[1, 21]`, `[1, 1]`, `[1, 59866]` |
| opus-mt-en-pis | transformers-auto-model | 4 | `[1, 21]`, `[1, 21]`, `[1, 1]`, `[1, 42010]` |
| opus-mt-en-pon | transformers-auto-model | 4 | `[1, 21]`, `[1, 21]`, `[1, 1]`, `[1, 52997]` |
| opus-mt-en-poz | transformers-auto-model | 4 | `[1, 20]`, `[1, 20]`, `[1, 1]`, `[1, 60175]` |
| opus-mt-en-pqe | transformers-auto-model | 4 | `[1, 21]`, `[1, 21]`, `[1, 1]`, `[1, 60089]` |
| opus-mt-en-pqw | transformers-auto-model | 4 | `[1, 20]`, `[1, 20]`, `[1, 1]`, `[1, 58849]` |
| opus-mt-en-rn | transformers-auto-model | 4 | `[1, 27]`, `[1, 27]`, `[1, 1]`, `[1, 7528]` |
| opus-mt-en-rnd | transformers-auto-model | 4 | `[1, 21]`, `[1, 21]`, `[1, 1]`, `[1, 53832]` |
| opus-mt-en-ro | transformers-auto-model | 4 | `[1, 21]`, `[1, 21]`, `[1, 1]`, `[1, 59543]` |
| opus-mt-en-roa | transformers-auto-model | 4 | `[1, 20]`, `[1, 20]`, `[1, 1]`, `[1, 56671]` |
| opus-mt-en-ru | transformers-auto-model | 4 | `[1, 21]`, `[1, 21]`, `[1, 1]`, `[1, 62518]` |
| opus-mt-en-run | transformers-auto-model | 4 | `[1, 21]`, `[1, 21]`, `[1, 1]`, `[1, 61335]` |
| opus-mt-en-rw | transformers-auto-model | 4 | `[1, 20]`, `[1, 20]`, `[1, 1]`, `[1, 60384]` |
| opus-mt-en-sal | transformers-auto-model | 4 | `[1, 20]`, `[1, 20]`, `[1, 1]`, `[1, 55872]` |
| opus-mt-en-sem | transformers-auto-model | 4 | `[1, 20]`, `[1, 20]`, `[1, 1]`, `[1, 61180]` |
| opus-mt-en-sg | transformers-auto-model | 4 | `[1, 20]`, `[1, 20]`, `[1, 1]`, `[1, 43822]` |
| opus-mt-en-sit | transformers-auto-model | 4 | `[1, 20]`, `[1, 20]`, `[1, 1]`, `[1, 59414]` |
| opus-mt-en-sk | transformers-auto-model | 4 | `[1, 21]`, `[1, 21]`, `[1, 1]`, `[1, 60025]` |
| opus-mt-en-sla | transformers-auto-model | 4 | `[1, 20]`, `[1, 20]`, `[1, 1]`, `[1, 58879]` |
| opus-mt-en-sm | transformers-auto-model | 4 | `[1, 21]`, `[1, 21]`, `[1, 1]`, `[1, 54880]` |
| opus-mt-en-sn | transformers-auto-model | 4 | `[1, 21]`, `[1, 21]`, `[1, 1]`, `[1, 61153]` |
| opus-mt-en-sq | transformers-auto-model | 4 | `[1, 20]`, `[1, 20]`, `[1, 1]`, `[1, 59158]` |
| opus-mt-en-ss | transformers-auto-model | 4 | `[1, 21]`, `[1, 21]`, `[1, 1]`, `[1, 55129]` |
| opus-mt-en-st | transformers-auto-model | 4 | `[1, 21]`, `[1, 21]`, `[1, 1]`, `[1, 57355]` |
| opus-mt-en-sv | transformers-auto-model | 4 | `[1, 20]`, `[1, 20]`, `[1, 1]`, `[1, 56434]` |
| opus-mt-en-sw | transformers-auto-model | 4 | `[1, 21]`, `[1, 21]`, `[1, 1]`, `[1, 58950]` |
| opus-mt-en-swc | transformers-auto-model | 4 | `[1, 21]`, `[1, 21]`, `[1, 1]`, `[1, 58905]` |
| opus-mt-en-tdt | transformers-auto-model | 4 | `[1, 21]`, `[1, 21]`, `[1, 1]`, `[1, 34360]` |
| opus-mt-en-tiv | transformers-auto-model | 4 | `[1, 21]`, `[1, 21]`, `[1, 1]`, `[1, 43019]` |
| opus-mt-en-tl | transformers-auto-model | 4 | `[1, 21]`, `[1, 21]`, `[1, 1]`, `[1, 57373]` |
| opus-mt-en-tn | transformers-auto-model | 4 | `[1, 21]`, `[1, 21]`, `[1, 1]`, `[1, 57306]` |
| opus-mt-en-to | transformers-auto-model | 4 | `[1, 21]`, `[1, 21]`, `[1, 1]`, `[1, 57446]` |
| opus-mt-en-toi | transformers-auto-model | 4 | `[1, 21]`, `[1, 21]`, `[1, 1]`, `[1, 61051]` |
| opus-mt-en-tpi | transformers-auto-model | 4 | `[1, 21]`, `[1, 21]`, `[1, 1]`, `[1, 49238]` |
| opus-mt-en-ts | transformers-auto-model | 4 | `[1, 21]`, `[1, 21]`, `[1, 1]`, `[1, 57469]` |
| opus-mt-en-tut | transformers-auto-model | 4 | `[1, 20]`, `[1, 20]`, `[1, 1]`, `[1, 61657]` |
| opus-mt-en-tvl | transformers-auto-model | 4 | `[1, 21]`, `[1, 21]`, `[1, 1]`, `[1, 38380]` |
| opus-mt-en-tw | transformers-auto-model | 4 | `[1, 21]`, `[1, 21]`, `[1, 1]`, `[1, 57000]` |
| opus-mt-en-ty | transformers-auto-model | 4 | `[1, 21]`, `[1, 21]`, `[1, 1]`, `[1, 56506]` |
| opus-mt-en-uk | transformers-auto-model | 4 | `[1, 20]`, `[1, 20]`, `[1, 1]`, `[1, 61587]` |
| opus-mt-en-umb | transformers-auto-model | 4 | `[1, 21]`, `[1, 21]`, `[1, 1]`, `[1, 58673]` |
| opus-mt-en-ur | transformers-auto-model | 4 | `[1, 20]`, `[1, 20]`, `[1, 1]`, `[1, 62025]` |
| opus-mt-en-urj | transformers-auto-model | 4 | `[1, 20]`, `[1, 20]`, `[1, 1]`, `[1, 58920]` |
| opus-mt-en-vi | transformers-auto-model | 4 | `[1, 20]`, `[1, 20]`, `[1, 1]`, `[1, 53685]` |
| opus-mt-en-xh | transformers-auto-model | 4 | `[1, 21]`, `[1, 21]`, `[1, 1]`, `[1, 61285]` |
| opus-mt-en-zle | transformers-auto-model | 4 | `[1, 20]`, `[1, 20]`, `[1, 1]`, `[1, 61605]` |
| opus-mt-en-zls | transformers-auto-model | 4 | `[1, 20]`, `[1, 20]`, `[1, 1]`, `[1, 57680]` |
| opus-mt-en-zlw | transformers-auto-model | 4 | `[1, 20]`, `[1, 20]`, `[1, 1]`, `[1, 58052]` |
| opus-mt-eo-af | transformers-auto-model | 4 | `[1, 47]`, `[1, 47]`, `[1, 1]`, `[1, 7475]` |
| opus-mt-eo-bg | transformers-auto-model | 4 | `[1, 46]`, `[1, 46]`, `[1, 1]`, `[1, 7770]` |
| opus-mt-eo-cs | transformers-auto-model | 4 | `[1, 46]`, `[1, 46]`, `[1, 1]`, `[1, 7397]` |
| opus-mt-eo-da | transformers-auto-model | 4 | `[1, 45]`, `[1, 45]`, `[1, 1]`, `[1, 7286]` |
| opus-mt-eo-de | transformers-auto-model | 4 | `[1, 31]`, `[1, 31]`, `[1, 1]`, `[1, 59836]` |
| opus-mt-eo-el | transformers-auto-model | 4 | `[1, 46]`, `[1, 46]`, `[1, 1]`, `[1, 7713]` |
| opus-mt-eo-en | transformers-auto-model | 4 | `[1, 29]`, `[1, 29]`, `[1, 1]`, `[1, 59197]` |
| opus-mt-eo-es | transformers-auto-model | 4 | `[1, 32]`, `[1, 32]`, `[1, 1]`, `[1, 58379]` |
| opus-mt-eo-fi | transformers-auto-model | 4 | `[1, 47]`, `[1, 47]`, `[1, 1]`, `[1, 7439]` |
| opus-mt-eo-fr | transformers-auto-model | 4 | `[1, 31]`, `[1, 31]`, `[1, 1]`, `[1, 59728]` |
| opus-mt-eo-he | transformers-auto-model | 4 | `[1, 48]`, `[1, 48]`, `[1, 1]`, `[1, 7768]` |
| opus-mt-eo-hu | transformers-auto-model | 4 | `[1, 46]`, `[1, 46]`, `[1, 1]`, `[1, 7477]` |
| opus-mt-eo-it | transformers-auto-model | 4 | `[1, 45]`, `[1, 45]`, `[1, 1]`, `[1, 7276]` |
| opus-mt-eo-nl | transformers-auto-model | 4 | `[1, 45]`, `[1, 45]`, `[1, 1]`, `[1, 7344]` |
| opus-mt-eo-pl | transformers-auto-model | 4 | `[1, 45]`, `[1, 45]`, `[1, 1]`, `[1, 7325]` |
| opus-mt-eo-pt | transformers-auto-model | 4 | `[1, 44]`, `[1, 44]`, `[1, 1]`, `[1, 7246]` |
| opus-mt-eo-ro | transformers-auto-model | 4 | `[1, 45]`, `[1, 45]`, `[1, 1]`, `[1, 7347]` |
| opus-mt-eo-ru | transformers-auto-model | 4 | `[1, 47]`, `[1, 47]`, `[1, 1]`, `[1, 7730]` |
| opus-mt-eo-sh | transformers-auto-model | 4 | `[1, 46]`, `[1, 46]`, `[1, 1]`, `[1, 7347]` |
| opus-mt-eo-sv | transformers-auto-model | 4 | `[1, 45]`, `[1, 45]`, `[1, 1]`, `[1, 7284]` |
| opus-mt-es-NORWAY | transformers-auto-model | 4 | `[1, 34]`, `[1, 34]`, `[1, 1]`, `[1, 58071]` |
| opus-mt-es-af | transformers-auto-model | 4 | `[1, 40]`, `[1, 40]`, `[1, 1]`, `[1, 60216]` |
| opus-mt-es-ar | transformers-auto-model | 4 | `[1, 39]`, `[1, 39]`, `[1, 1]`, `[1, 62518]` |
| opus-mt-es-ase | transformers-auto-model | 4 | `[1, 40]`, `[1, 40]`, `[1, 1]`, `[1, 53502]` |
| opus-mt-es-bcl | transformers-auto-model | 4 | `[1, 40]`, `[1, 40]`, `[1, 1]`, `[1, 59736]` |
| opus-mt-es-ber | transformers-auto-model | 4 | `[1, 41]`, `[1, 41]`, `[1, 1]`, `[1, 13296]` |
| opus-mt-es-bg | transformers-auto-model | 4 | `[1, 37]`, `[1, 37]`, `[1, 1]`, `[1, 61845]` |
| opus-mt-es-bi | transformers-auto-model | 4 | `[1, 42]`, `[1, 42]`, `[1, 1]`, `[1, 56412]` |
| opus-mt-es-bzs | transformers-auto-model | 4 | `[1, 41]`, `[1, 41]`, `[1, 1]`, `[1, 51421]` |
| opus-mt-es-ca | transformers-auto-model | 4 | `[1, 37]`, `[1, 37]`, `[1, 1]`, `[1, 51970]` |
| opus-mt-es-ceb | transformers-auto-model | 4 | `[1, 40]`, `[1, 40]`, `[1, 1]`, `[1, 59324]` |
| opus-mt-es-crs | transformers-auto-model | 4 | `[1, 43]`, `[1, 43]`, `[1, 1]`, `[1, 47319]` |
| opus-mt-es-cs | transformers-auto-model | 4 | `[1, 35]`, `[1, 35]`, `[1, 1]`, `[1, 58623]` |
| opus-mt-es-csg | transformers-auto-model | 4 | `[1, 40]`, `[1, 40]`, `[1, 1]`, `[1, 11802]` |
| opus-mt-es-da | transformers-auto-model | 4 | `[1, 37]`, `[1, 37]`, `[1, 1]`, `[1, 57584]` |
| opus-mt-es-de | transformers-auto-model | 4 | `[1, 39]`, `[1, 39]`, `[1, 1]`, `[1, 61301]` |
| opus-mt-es-ee | transformers-auto-model | 4 | `[1, 41]`, `[1, 41]`, `[1, 1]`, `[1, 60163]` |
| opus-mt-es-efi | transformers-auto-model | 4 | `[1, 42]`, `[1, 42]`, `[1, 1]`, `[1, 59814]` |
| opus-mt-es-el | transformers-auto-model | 4 | `[1, 37]`, `[1, 37]`, `[1, 1]`, `[1, 60782]` |
| opus-mt-es-eo | transformers-auto-model | 4 | `[1, 34]`, `[1, 34]`, `[1, 1]`, `[1, 58379]` |
| opus-mt-es-es | transformers-auto-model | 4 | `[1, 37]`, `[1, 37]`, `[1, 1]`, `[1, 33253]` |
| opus-mt-es-et | transformers-auto-model | 4 | `[1, 34]`, `[1, 34]`, `[1, 1]`, `[1, 58124]` |
| opus-mt-es-eu | transformers-auto-model | 4 | `[1, 38]`, `[1, 38]`, `[1, 1]`, `[1, 58060]` |
| opus-mt-es-fi | transformers-auto-model | 4 | `[1, 36]`, `[1, 36]`, `[1, 1]`, `[1, 63261]` |
| opus-mt-es-fr | transformers-auto-model | 4 | `[1, 38]`, `[1, 38]`, `[1, 1]`, `[1, 74822]` |
| opus-mt-es-gaa | transformers-auto-model | 4 | `[1, 41]`, `[1, 41]`, `[1, 1]`, `[1, 60173]` |
| opus-mt-es-gil | transformers-auto-model | 4 | `[1, 39]`, `[1, 39]`, `[1, 1]`, `[1, 53428]` |
| opus-mt-es-gl | transformers-auto-model | 4 | `[1, 47]`, `[1, 47]`, `[1, 1]`, `[1, 5805]` |
| opus-mt-es-guw | transformers-auto-model | 4 | `[1, 40]`, `[1, 40]`, `[1, 1]`, `[1, 54512]` |
| opus-mt-es-ha | transformers-auto-model | 4 | `[1, 34]`, `[1, 34]`, `[1, 1]`, `[1, 59503]` |
| opus-mt-es-he | transformers-auto-model | 4 | `[1, 36]`, `[1, 36]`, `[1, 1]`, `[1, 63105]` |
| opus-mt-es-hil | transformers-auto-model | 4 | `[1, 40]`, `[1, 40]`, `[1, 1]`, `[1, 59256]` |
| opus-mt-es-ho | transformers-auto-model | 4 | `[1, 40]`, `[1, 40]`, `[1, 1]`, `[1, 45168]` |
| opus-mt-es-hr | transformers-auto-model | 4 | `[1, 32]`, `[1, 32]`, `[1, 1]`, `[1, 58648]` |
| opus-mt-es-ht | transformers-auto-model | 4 | `[1, 39]`, `[1, 39]`, `[1, 1]`, `[1, 49387]` |
| opus-mt-es-id | transformers-auto-model | 4 | `[1, 32]`, `[1, 32]`, `[1, 1]`, `[1, 57414]` |
| opus-mt-es-ig | transformers-auto-model | 4 | `[1, 35]`, `[1, 35]`, `[1, 1]`, `[1, 60006]` |
| opus-mt-es-ilo | transformers-auto-model | 4 | `[1, 42]`, `[1, 42]`, `[1, 1]`, `[1, 59457]` |
| opus-mt-es-is | transformers-auto-model | 4 | `[1, 37]`, `[1, 37]`, `[1, 1]`, `[1, 60335]` |
| opus-mt-es-iso | transformers-auto-model | 4 | `[1, 39]`, `[1, 39]`, `[1, 1]`, `[1, 47550]` |
| opus-mt-es-kg | transformers-auto-model | 4 | `[1, 40]`, `[1, 40]`, `[1, 1]`, `[1, 49808]` |
| opus-mt-es-ln | transformers-auto-model | 4 | `[1, 40]`, `[1, 40]`, `[1, 1]`, `[1, 60350]` |
| opus-mt-es-loz | transformers-auto-model | 4 | `[1, 42]`, `[1, 42]`, `[1, 1]`, `[1, 60783]` |
| opus-mt-es-lt | transformers-auto-model | 4 | `[1, 38]`, `[1, 38]`, `[1, 1]`, `[1, 59511]` |
| opus-mt-es-lua | transformers-auto-model | 4 | `[1, 40]`, `[1, 40]`, `[1, 1]`, `[1, 60813]` |
| opus-mt-es-lus | transformers-auto-model | 4 | `[1, 41]`, `[1, 41]`, `[1, 1]`, `[1, 53585]` |
| opus-mt-es-mfs | transformers-auto-model | 4 | `[1, 40]`, `[1, 40]`, `[1, 1]`, `[1, 28283]` |
| opus-mt-es-mk | transformers-auto-model | 4 | `[1, 36]`, `[1, 36]`, `[1, 1]`, `[1, 62482]` |
| opus-mt-es-mt | transformers-auto-model | 4 | `[1, 38]`, `[1, 38]`, `[1, 1]`, `[1, 57085]` |
| opus-mt-es-niu | transformers-auto-model | 4 | `[1, 40]`, `[1, 40]`, `[1, 1]`, `[1, 45309]` |
| opus-mt-es-nl | transformers-auto-model | 4 | `[1, 37]`, `[1, 37]`, `[1, 1]`, `[1, 58264]` |
| opus-mt-es-no | transformers-auto-model | 4 | `[1, 43]`, `[1, 43]`, `[1, 1]`, `[1, 22026]` |
| opus-mt-es-nso | transformers-auto-model | 4 | `[1, 37]`, `[1, 37]`, `[1, 1]`, `[1, 60344]` |
| opus-mt-es-ny | transformers-auto-model | 4 | `[1, 40]`, `[1, 40]`, `[1, 1]`, `[1, 60819]` |
| opus-mt-es-pag | transformers-auto-model | 4 | `[1, 41]`, `[1, 41]`, `[1, 1]`, `[1, 60074]` |
| opus-mt-es-pap | transformers-auto-model | 4 | `[1, 39]`, `[1, 39]`, `[1, 1]`, `[1, 56010]` |
| opus-mt-es-pis | transformers-auto-model | 4 | `[1, 40]`, `[1, 40]`, `[1, 1]`, `[1, 49022]` |
| opus-mt-es-pl | transformers-auto-model | 4 | `[1, 39]`, `[1, 39]`, `[1, 1]`, `[1, 59678]` |
| opus-mt-es-pon | transformers-auto-model | 4 | `[1, 41]`, `[1, 41]`, `[1, 1]`, `[1, 56793]` |
| opus-mt-es-prl | transformers-auto-model | 4 | `[1, 41]`, `[1, 41]`, `[1, 1]`, `[1, 10873]` |
| opus-mt-es-rn | transformers-auto-model | 4 | `[1, 52]`, `[1, 52]`, `[1, 1]`, `[1, 7512]` |
| opus-mt-es-ro | transformers-auto-model | 4 | `[1, 36]`, `[1, 36]`, `[1, 1]`, `[1, 56140]` |
| opus-mt-es-ru | transformers-auto-model | 4 | `[1, 33]`, `[1, 33]`, `[1, 1]`, `[1, 63430]` |
| opus-mt-es-rw | transformers-auto-model | 4 | `[1, 32]`, `[1, 32]`, `[1, 1]`, `[1, 61291]` |
| opus-mt-es-sg | transformers-auto-model | 4 | `[1, 41]`, `[1, 41]`, `[1, 1]`, `[1, 46098]` |
| opus-mt-es-sl | transformers-auto-model | 4 | `[1, 36]`, `[1, 36]`, `[1, 1]`, `[1, 57959]` |
| opus-mt-es-sm | transformers-auto-model | 4 | `[1, 40]`, `[1, 40]`, `[1, 1]`, `[1, 58284]` |
| opus-mt-es-sn | transformers-auto-model | 4 | `[1, 41]`, `[1, 41]`, `[1, 1]`, `[1, 61622]` |
| opus-mt-es-srn | transformers-auto-model | 4 | `[1, 42]`, `[1, 42]`, `[1, 1]`, `[1, 60052]` |
| opus-mt-es-st | transformers-auto-model | 4 | `[1, 40]`, `[1, 40]`, `[1, 1]`, `[1, 60207]` |
| opus-mt-es-swc | transformers-auto-model | 4 | `[1, 40]`, `[1, 40]`, `[1, 1]`, `[1, 60332]` |
| opus-mt-es-tl | transformers-auto-model | 4 | `[1, 38]`, `[1, 38]`, `[1, 1]`, `[1, 59655]` |
| opus-mt-es-tll | transformers-auto-model | 4 | `[1, 41]`, `[1, 41]`, `[1, 1]`, `[1, 61389]` |
| opus-mt-es-tn | transformers-auto-model | 4 | `[1, 41]`, `[1, 41]`, `[1, 1]`, `[1, 60267]` |
| opus-mt-es-to | transformers-auto-model | 4 | `[1, 40]`, `[1, 40]`, `[1, 1]`, `[1, 58959]` |
| opus-mt-es-tpi | transformers-auto-model | 4 | `[1, 41]`, `[1, 41]`, `[1, 1]`, `[1, 52913]` |
| opus-mt-es-tvl | transformers-auto-model | 4 | `[1, 42]`, `[1, 42]`, `[1, 1]`, `[1, 42688]` |
| opus-mt-es-tw | transformers-auto-model | 4 | `[1, 41]`, `[1, 41]`, `[1, 1]`, `[1, 60033]` |
| opus-mt-es-ty | transformers-auto-model | 4 | `[1, 42]`, `[1, 42]`, `[1, 1]`, `[1, 58873]` |
| opus-mt-es-tzo | transformers-auto-model | 4 | `[1, 39]`, `[1, 39]`, `[1, 1]`, `[1, 54248]` |
| opus-mt-es-uk | transformers-auto-model | 4 | `[1, 36]`, `[1, 36]`, `[1, 1]`, `[1, 62460]` |
| opus-mt-es-ve | transformers-auto-model | 4 | `[1, 40]`, `[1, 40]`, `[1, 1]`, `[1, 56171]` |
| opus-mt-es-vi | transformers-auto-model | 4 | `[1, 36]`, `[1, 36]`, `[1, 1]`, `[1, 58488]` |
| opus-mt-es-war | transformers-auto-model | 4 | `[1, 41]`, `[1, 41]`, `[1, 1]`, `[1, 59954]` |
| opus-mt-es-wls | transformers-auto-model | 4 | `[1, 42]`, `[1, 42]`, `[1, 1]`, `[1, 41820]` |
| opus-mt-es-xh | transformers-auto-model | 4 | `[1, 33]`, `[1, 33]`, `[1, 1]`, `[1, 61702]` |
| opus-mt-es-yo | transformers-auto-model | 4 | `[1, 37]`, `[1, 37]`, `[1, 1]`, `[1, 59716]` |
| opus-mt-es-yua | transformers-auto-model | 4 | `[1, 39]`, `[1, 39]`, `[1, 1]`, `[1, 46028]` |
| opus-mt-es-zai | transformers-auto-model | 4 | `[1, 39]`, `[1, 39]`, `[1, 1]`, `[1, 48431]` |
| opus-mt-et-de | transformers-auto-model | 4 | `[1, 38]`, `[1, 38]`, `[1, 1]`, `[1, 58042]` |
| opus-mt-et-en | transformers-auto-model | 4 | `[1, 39]`, `[1, 39]`, `[1, 1]`, `[1, 58866]` |
| opus-mt-et-es | transformers-auto-model | 4 | `[1, 37]`, `[1, 37]`, `[1, 1]`, `[1, 58124]` |
| opus-mt-et-fi | transformers-auto-model | 4 | `[1, 37]`, `[1, 37]`, `[1, 1]`, `[1, 58055]` |
| opus-mt-et-fr | transformers-auto-model | 4 | `[1, 37]`, `[1, 37]`, `[1, 1]`, `[1, 58065]` |
| opus-mt-et-ru | transformers-auto-model | 4 | `[1, 37]`, `[1, 37]`, `[1, 1]`, `[1, 62898]` |
| opus-mt-et-sv | transformers-auto-model | 4 | `[1, 37]`, `[1, 37]`, `[1, 1]`, `[1, 57792]` |
| opus-mt-eu-de | transformers-auto-model | 4 | `[1, 39]`, `[1, 39]`, `[1, 1]`, `[1, 22170]` |
| opus-mt-eu-en | transformers-auto-model | 4 | `[1, 27]`, `[1, 27]`, `[1, 1]`, `[1, 58429]` |
| opus-mt-eu-es | transformers-auto-model | 4 | `[1, 33]`, `[1, 33]`, `[1, 1]`, `[1, 58060]` |
| opus-mt-eu-ru | transformers-auto-model | 4 | `[1, 46]`, `[1, 46]`, `[1, 1]`, `[1, 7677]` |
| opus-mt-euq-en | transformers-auto-model | 4 | `[1, 36]`, `[1, 36]`, `[1, 1]`, `[1, 21964]` |
| opus-mt-fi-NORWAY | transformers-auto-model | 4 | `[1, 34]`, `[1, 34]`, `[1, 1]`, `[1, 59345]` |
| opus-mt-fi-ZH | transformers-auto-model | 4 | `[1, 35]`, `[1, 35]`, `[1, 1]`, `[1, 62933]` |
| opus-mt-fi-af | transformers-auto-model | 4 | `[1, 35]`, `[1, 35]`, `[1, 1]`, `[1, 60657]` |
| opus-mt-fi-bcl | transformers-auto-model | 4 | `[1, 44]`, `[1, 44]`, `[1, 1]`, `[1, 61739]` |
| opus-mt-fi-bem | transformers-auto-model | 4 | `[1, 44]`, `[1, 44]`, `[1, 1]`, `[1, 61763]` |
| opus-mt-fi-bg | transformers-auto-model | 4 | `[1, 35]`, `[1, 35]`, `[1, 1]`, `[1, 60069]` |
| opus-mt-fi-bzs | transformers-auto-model | 4 | `[1, 50]`, `[1, 50]`, `[1, 1]`, `[1, 59433]` |
| opus-mt-fi-ceb | transformers-auto-model | 4 | `[1, 41]`, `[1, 41]`, `[1, 1]`, `[1, 61010]` |
| opus-mt-fi-crs | transformers-auto-model | 4 | `[1, 49]`, `[1, 49]`, `[1, 1]`, `[1, 48214]` |
| opus-mt-fi-cs | transformers-auto-model | 4 | `[1, 35]`, `[1, 35]`, `[1, 1]`, `[1, 59012]` |
| opus-mt-fi-de | transformers-auto-model | 4 | `[1, 36]`, `[1, 36]`, `[1, 1]`, `[1, 59471]` |
| opus-mt-fi-efi | transformers-auto-model | 4 | `[1, 46]`, `[1, 46]`, `[1, 1]`, `[1, 61193]` |
| opus-mt-fi-el | transformers-auto-model | 4 | `[1, 33]`, `[1, 33]`, `[1, 1]`, `[1, 59727]` |
| opus-mt-fi-en | transformers-auto-model | 4 | `[1, 36]`, `[1, 36]`, `[1, 1]`, `[1, 59659]` |
| opus-mt-fi-eo | transformers-auto-model | 4 | `[1, 48]`, `[1, 48]`, `[1, 1]`, `[1, 7439]` |
| opus-mt-fi-es | transformers-auto-model | 4 | `[1, 37]`, `[1, 37]`, `[1, 1]`, `[1, 63261]` |
| opus-mt-fi-et | transformers-auto-model | 4 | `[1, 35]`, `[1, 35]`, `[1, 1]`, `[1, 58055]` |
| opus-mt-fi-fi | transformers-auto-model | 4 | `[1, 40]`, `[1, 40]`, `[1, 1]`, `[1, 33007]` |
| opus-mt-fi-fj | transformers-auto-model | 4 | `[1, 44]`, `[1, 44]`, `[1, 1]`, `[1, 61090]` |
| opus-mt-fi-fr | transformers-auto-model | 4 | `[1, 37]`, `[1, 37]`, `[1, 1]`, `[1, 60065]` |
| opus-mt-fi-fse | transformers-auto-model | 4 | `[1, 44]`, `[1, 44]`, `[1, 1]`, `[1, 17131]` |
| opus-mt-fi-gaa | transformers-auto-model | 4 | `[1, 44]`, `[1, 44]`, `[1, 1]`, `[1, 61586]` |
| opus-mt-fi-gil | transformers-auto-model | 4 | `[1, 49]`, `[1, 49]`, `[1, 1]`, `[1, 54432]` |
| opus-mt-fi-guw | transformers-auto-model | 4 | `[1, 46]`, `[1, 46]`, `[1, 1]`, `[1, 55617]` |
| opus-mt-fi-ha | transformers-auto-model | 4 | `[1, 44]`, `[1, 44]`, `[1, 1]`, `[1, 57208]` |
| opus-mt-fi-he | transformers-auto-model | 4 | `[1, 37]`, `[1, 37]`, `[1, 1]`, `[1, 63477]` |
| opus-mt-fi-hil | transformers-auto-model | 4 | `[1, 41]`, `[1, 41]`, `[1, 1]`, `[1, 61066]` |
| opus-mt-fi-ho | transformers-auto-model | 4 | `[1, 45]`, `[1, 45]`, `[1, 1]`, `[1, 46197]` |
| opus-mt-fi-hr | transformers-auto-model | 4 | `[1, 35]`, `[1, 35]`, `[1, 1]`, `[1, 59394]` |
| opus-mt-fi-ht | transformers-auto-model | 4 | `[1, 46]`, `[1, 46]`, `[1, 1]`, `[1, 49235]` |
| opus-mt-fi-hu | transformers-auto-model | 4 | `[1, 35]`, `[1, 35]`, `[1, 1]`, `[1, 59526]` |
| opus-mt-fi-id | transformers-auto-model | 4 | `[1, 37]`, `[1, 37]`, `[1, 1]`, `[1, 59299]` |
| opus-mt-fi-ig | transformers-auto-model | 4 | `[1, 40]`, `[1, 40]`, `[1, 1]`, `[1, 61272]` |
| opus-mt-fi-ilo | transformers-auto-model | 4 | `[1, 41]`, `[1, 41]`, `[1, 1]`, `[1, 61080]` |
| opus-mt-fi-is | transformers-auto-model | 4 | `[1, 33]`, `[1, 33]`, `[1, 1]`, `[1, 60259]` |
| opus-mt-fi-iso | transformers-auto-model | 4 | `[1, 48]`, `[1, 48]`, `[1, 1]`, `[1, 48544]` |
| opus-mt-fi-it | transformers-auto-model | 4 | `[1, 35]`, `[1, 35]`, `[1, 1]`, `[1, 59909]` |
| opus-mt-fi-kg | transformers-auto-model | 4 | `[1, 47]`, `[1, 47]`, `[1, 1]`, `[1, 50740]` |
| opus-mt-fi-kqn | transformers-auto-model | 4 | `[1, 48]`, `[1, 48]`, `[1, 1]`, `[1, 62236]` |
| opus-mt-fi-lg | transformers-auto-model | 4 | `[1, 46]`, `[1, 46]`, `[1, 1]`, `[1, 62274]` |
| opus-mt-fi-ln | transformers-auto-model | 4 | `[1, 42]`, `[1, 42]`, `[1, 1]`, `[1, 61529]` |
| opus-mt-fi-lu | transformers-auto-model | 4 | `[1, 49]`, `[1, 49]`, `[1, 1]`, `[1, 62427]` |
| opus-mt-fi-lua | transformers-auto-model | 4 | `[1, 44]`, `[1, 44]`, `[1, 1]`, `[1, 61858]` |
| opus-mt-fi-lue | transformers-auto-model | 4 | `[1, 46]`, `[1, 46]`, `[1, 1]`, `[1, 62338]` |
| opus-mt-fi-lus | transformers-auto-model | 4 | `[1, 48]`, `[1, 48]`, `[1, 1]`, `[1, 53773]` |
| opus-mt-fi-lv | transformers-auto-model | 4 | `[1, 37]`, `[1, 37]`, `[1, 1]`, `[1, 59460]` |
| opus-mt-fi-mfe | transformers-auto-model | 4 | `[1, 50]`, `[1, 50]`, `[1, 1]`, `[1, 54138]` |
| opus-mt-fi-mg | transformers-auto-model | 4 | `[1, 34]`, `[1, 34]`, `[1, 1]`, `[1, 60329]` |
| opus-mt-fi-mh | transformers-auto-model | 4 | `[1, 49]`, `[1, 49]`, `[1, 1]`, `[1, 54810]` |
| opus-mt-fi-mk | transformers-auto-model | 4 | `[1, 36]`, `[1, 36]`, `[1, 1]`, `[1, 62563]` |
| opus-mt-fi-mos | transformers-auto-model | 4 | `[1, 50]`, `[1, 50]`, `[1, 1]`, `[1, 50438]` |
| opus-mt-fi-mt | transformers-auto-model | 4 | `[1, 40]`, `[1, 40]`, `[1, 1]`, `[1, 59039]` |
| opus-mt-fi-niu | transformers-auto-model | 4 | `[1, 50]`, `[1, 50]`, `[1, 1]`, `[1, 45655]` |
| opus-mt-fi-nl | transformers-auto-model | 4 | `[1, 37]`, `[1, 37]`, `[1, 1]`, `[1, 59836]` |
| opus-mt-fi-nso | transformers-auto-model | 4 | `[1, 39]`, `[1, 39]`, `[1, 1]`, `[1, 61423]` |
| opus-mt-fi-ny | transformers-auto-model | 4 | `[1, 41]`, `[1, 41]`, `[1, 1]`, `[1, 61529]` |
| opus-mt-tc-bible-big-aav-fra_ita_por_spa | transformers-auto-model | 4 | `[1, 29]`, `[1, 29]`, `[1, 1]`, `[1, 58474]` |
| opus-mt-tc-bible-big-afa-deu_eng_fra_por_spa | transformers-auto-model | 4 | `[1, 41]`, `[1, 41]`, `[1, 1]`, `[1, 61448]` |
| opus-mt-tc-bible-big-afa-deu_eng_nld | transformers-auto-model | 4 | `[1, 41]`, `[1, 41]`, `[1, 1]`, `[1, 61623]` |
| opus-mt-tc-bible-big-afa-en | transformers-auto-model | 4 | `[1, 41]`, `[1, 41]`, `[1, 1]`, `[1, 61389]` |
| opus-mt-tc-bible-big-afa-fra_ita_por_spa | transformers-auto-model | 4 | `[1, 42]`, `[1, 42]`, `[1, 1]`, `[1, 61871]` |
| opus-mt-tc-bible-big-alv-deu_eng_fra_por_spa | transformers-auto-model | 4 | `[1, 39]`, `[1, 39]`, `[1, 1]`, `[1, 61872]` |
| opus-mt-tc-bible-big-bat-deu_eng_fra_por_spa | transformers-auto-model | 4 | `[1, 38]`, `[1, 38]`, `[1, 1]`, `[1, 59467]` |
| opus-mt-tc-bible-big-bat-deu_eng_nld | transformers-auto-model | 4 | `[1, 35]`, `[1, 35]`, `[1, 1]`, `[1, 59091]` |
| opus-mt-tc-bible-big-bat-en | transformers-auto-model | 4 | `[1, 34]`, `[1, 34]`, `[1, 1]`, `[1, 58702]` |
| opus-mt-tc-bible-big-bnt-deu_eng_fra_por_spa | transformers-auto-model | 4 | `[1, 38]`, `[1, 38]`, `[1, 1]`, `[1, 61612]` |
| opus-mt-tc-bible-big-cel-deu_eng_fra_por_spa | transformers-auto-model | 4 | `[1, 28]`, `[1, 28]`, `[1, 1]`, `[1, 56599]` |
| opus-mt-tc-bible-big-deu_eng_fra_por_spa-aav | transformers-auto-model | 4 | `[1, 20]`, `[1, 20]`, `[1, 1]`, `[1, 57539]` |
| opus-mt-tc-bible-big-deu_eng_fra_por_spa-afa | transformers-auto-model | 4 | `[1, 23]`, `[1, 23]`, `[1, 1]`, `[1, 61815]` |
| opus-mt-tc-bible-big-deu_eng_fra_por_spa-bat | transformers-auto-model | 4 | `[1, 21]`, `[1, 21]`, `[1, 1]`, `[1, 59473]` |
| opus-mt-tc-bible-big-deu_eng_fra_por_spa-bnt | transformers-auto-model | 4 | `[1, 23]`, `[1, 23]`, `[1, 1]`, `[1, 62297]` |
| opus-mt-tc-bible-big-deu_eng_fra_por_spa-fiu | transformers-auto-model | 4 | `[1, 20]`, `[1, 20]`, `[1, 1]`, `[1, 59392]` |
| opus-mt-tc-bible-big-deu_eng_fra_por_spa-gem | transformers-auto-model | 4 | `[1, 20]`, `[1, 20]`, `[1, 1]`, `[1, 49013]` |
| opus-mt-tc-bible-big-deu_eng_fra_por_spa-gmq | transformers-auto-model | 4 | `[1, 20]`, `[1, 20]`, `[1, 1]`, `[1, 56248]` |
| opus-mt-tc-bible-big-deu_eng_fra_por_spa-gmw | transformers-auto-model | 4 | `[1, 20]`, `[1, 20]`, `[1, 1]`, `[1, 48183]` |
| opus-mt-tc-bible-big-deu_eng_fra_por_spa-iir | transformers-auto-model | 4 | `[1, 20]`, `[1, 20]`, `[1, 1]`, `[1, 62090]` |
| opus-mt-tc-bible-big-deu_eng_fra_por_spa-inc | transformers-auto-model | 4 | `[1, 20]`, `[1, 20]`, `[1, 1]`, `[1, 61906]` |
| opus-mt-tc-bible-big-deu_eng_fra_por_spa-ine | transformers-auto-model | 4 | `[1, 20]`, `[1, 20]`, `[1, 1]`, `[1, 53305]` |
| opus-mt-tc-bible-big-deu_eng_fra_por_spa-itc | transformers-auto-model | 4 | `[1, 20]`, `[1, 20]`, `[1, 1]`, `[1, 45448]` |
| opus-mt-tc-bible-big-deu_eng_fra_por_spa-mkh | transformers-auto-model | 4 | `[1, 20]`, `[1, 20]`, `[1, 1]`, `[1, 57129]` |
| opus-mt-tc-bible-big-deu_eng_fra_por_spa-mul | transformers-auto-model | 4 | `[1, 20]`, `[1, 20]`, `[1, 1]`, `[1, 62959]` |
| opus-mt-tc-bible-big-deu_eng_fra_por_spa-pqw | transformers-auto-model | 4 | `[1, 20]`, `[1, 20]`, `[1, 1]`, `[1, 59520]` |
| opus-mt-tc-bible-big-deu_eng_fra_por_spa-roa | transformers-auto-model | 4 | `[1, 20]`, `[1, 20]`, `[1, 1]`, `[1, 45376]` |
| opus-mt-tc-bible-big-deu_eng_fra_por_spa-sem | transformers-auto-model | 4 | `[1, 20]`, `[1, 20]`, `[1, 1]`, `[1, 61339]` |
| opus-mt-tc-bible-big-deu_eng_fra_por_spa-sla | transformers-auto-model | 4 | `[1, 20]`, `[1, 20]`, `[1, 1]`, `[1, 59956]` |
| opus-mt-tc-bible-big-deu_eng_fra_por_spa-trk | transformers-auto-model | 4 | `[1, 20]`, `[1, 20]`, `[1, 1]`, `[1, 60983]` |
| opus-mt-tc-bible-big-deu_eng_fra_por_spa-urj | transformers-auto-model | 4 | `[1, 20]`, `[1, 20]`, `[1, 1]`, `[1, 59384]` |
| opus-mt-tc-bible-big-dra-deu_eng_nld | transformers-auto-model | 4 | `[1, 42]`, `[1, 42]`, `[1, 1]`, `[1, 62489]` |
| opus-mt-tc-bible-big-dra-en | transformers-auto-model | 4 | `[1, 42]`, `[1, 42]`, `[1, 1]`, `[1, 62506]` |
| opus-mt-tc-bible-big-fiu-deu_eng_fra_por_spa | transformers-auto-model | 4 | `[1, 36]`, `[1, 36]`, `[1, 1]`, `[1, 59382]` |
| opus-mt-tc-bible-big-fiu-en | transformers-auto-model | 4 | `[1, 34]`, `[1, 34]`, `[1, 1]`, `[1, 58747]` |
| opus-mt-tc-bible-big-fiu-fra_ita_por_spa | transformers-auto-model | 4 | `[1, 36]`, `[1, 36]`, `[1, 1]`, `[1, 59780]` |
| opus-mt-tc-bible-big-gem-deu_eng_fra_por_spa | transformers-auto-model | 4 | `[1, 20]`, `[1, 20]`, `[1, 1]`, `[1, 48859]` |
| opus-mt-tc-bible-big-gem-fra_ita_por_spa | transformers-auto-model | 4 | `[1, 20]`, `[1, 20]`, `[1, 1]`, `[1, 56813]` |
| opus-mt-tc-bible-big-gmq-deu_eng_fra_por_spa | transformers-auto-model | 4 | `[1, 35]`, `[1, 35]`, `[1, 1]`, `[1, 56160]` |
| opus-mt-tc-bible-big-gmq-en | transformers-auto-model | 4 | `[1, 34]`, `[1, 34]`, `[1, 1]`, `[1, 55007]` |
| opus-mt-tc-bible-big-gmw-deu_eng_fra_por_spa | transformers-auto-model | 4 | `[1, 20]`, `[1, 20]`, `[1, 1]`, `[1, 47963]` |
| opus-mt-tc-bible-big-gmw-deu_eng_nld | transformers-auto-model | 4 | `[1, 20]`, `[1, 20]`, `[1, 1]`, `[1, 39447]` |
| opus-mt-tc-bible-big-gmw-en | transformers-auto-model | 4 | `[1, 24]`, `[1, 24]`, `[1, 1]`, `[1, 54702]` |
| opus-mt-tc-bible-big-gmw-fra_ita_por_spa | transformers-auto-model | 4 | `[1, 20]`, `[1, 20]`, `[1, 1]`, `[1, 56346]` |
| opus-mt-tc-bible-big-iir-deu_eng_fra_por_spa | transformers-auto-model | 4 | `[1, 38]`, `[1, 38]`, `[1, 1]`, `[1, 61807]` |
| opus-mt-tc-bible-big-iir-en | transformers-auto-model | 4 | `[1, 39]`, `[1, 39]`, `[1, 1]`, `[1, 62113]` |
| opus-mt-tc-bible-big-inc-deu_eng_fra_por_spa | transformers-auto-model | 4 | `[1, 39]`, `[1, 39]`, `[1, 1]`, `[1, 61705]` |
| opus-mt-tc-bible-big-inc-deu_eng_nld | transformers-auto-model | 4 | `[1, 41]`, `[1, 41]`, `[1, 1]`, `[1, 62164]` |
| opus-mt-tc-bible-big-inc-en | transformers-auto-model | 4 | `[1, 38]`, `[1, 38]`, `[1, 1]`, `[1, 62026]` |
| opus-mt-tc-bible-big-ine-deu_eng_fra_por_spa | transformers-auto-model | 4 | `[1, 25]`, `[1, 25]`, `[1, 1]`, `[1, 52781]` |
| opus-mt-tc-bible-big-ine-deu_eng_nld | transformers-auto-model | 4 | `[1, 24]`, `[1, 24]`, `[1, 1]`, `[1, 55209]` |
| opus-mt-tc-bible-big-ira-deu_eng_fra_por_spa | transformers-auto-model | 4 | `[1, 35]`, `[1, 35]`, `[1, 1]`, `[1, 60956]` |
| opus-mt-tc-bible-big-itc-deu_eng_fra_por_spa | transformers-auto-model | 4 | `[1, 31]`, `[1, 31]`, `[1, 1]`, `[1, 45341]` |
| opus-mt-tc-bible-big-itc-deu_eng_nld | transformers-auto-model | 4 | `[1, 31]`, `[1, 31]`, `[1, 1]`, `[1, 56278]` |
| opus-mt-tc-bible-big-itc-fra_ita_por_spa | transformers-auto-model | 4 | `[1, 32]`, `[1, 32]`, `[1, 1]`, `[1, 39825]` |
| opus-mt-tc-bible-big-map-en | transformers-auto-model | 4 | `[1, 35]`, `[1, 35]`, `[1, 1]`, `[1, 59740]` |
| opus-mt-tc-bible-big-map-fra_ita_por_spa | transformers-auto-model | 4 | `[1, 37]`, `[1, 37]`, `[1, 1]`, `[1, 60885]` |
| opus-mt-tc-bible-big-mkh-deu_eng_nld | transformers-auto-model | 4 | `[1, 31]`, `[1, 31]`, `[1, 1]`, `[1, 55468]` |
| opus-mt-tc-bible-big-mkh-fra_ita_por_spa | transformers-auto-model | 4 | `[1, 26]`, `[1, 26]`, `[1, 1]`, `[1, 58241]` |
| opus-mt-tc-bible-big-mul-deu_eng_fra_por_spa | transformers-auto-model | 4 | `[1, 27]`, `[1, 27]`, `[1, 1]`, `[1, 57305]` |
| opus-mt-tc-bible-big-mul-deu_eng_nld | transformers-auto-model | 4 | `[1, 27]`, `[1, 27]`, `[1, 1]`, `[1, 58434]` |
| opus-mt-tc-bible-big-mul-mul | transformers-auto-model | 4 | `[1, 22]`, `[1, 22]`, `[1, 1]`, `[1, 69667]` |
| opus-mt-tc-bible-big-phi-deu_eng_fra_por_spa | transformers-auto-model | 4 | `[1, 37]`, `[1, 37]`, `[1, 1]`, `[1, 59760]` |
| opus-mt-tc-bible-big-phi-en | transformers-auto-model | 4 | `[1, 35]`, `[1, 35]`, `[1, 1]`, `[1, 57576]` |
| opus-mt-tc-bible-big-poz-deu_eng_fra_por_spa | transformers-auto-model | 4 | `[1, 36]`, `[1, 36]`, `[1, 1]`, `[1, 60414]` |
| opus-mt-tc-bible-big-poz-en | transformers-auto-model | 4 | `[1, 34]`, `[1, 34]`, `[1, 1]`, `[1, 59757]` |
| opus-mt-tc-bible-big-poz-fra_ita_por_spa | transformers-auto-model | 4 | `[1, 37]`, `[1, 37]`, `[1, 1]`, `[1, 60868]` |
| opus-mt-tc-bible-big-pqe-deu_eng_fra_por_spa | transformers-auto-model | 4 | `[1, 39]`, `[1, 39]`, `[1, 1]`, `[1, 61710]` |
| opus-mt-tc-bible-big-pqw-deu_eng_fra_por_spa | transformers-auto-model | 4 | `[1, 35]`, `[1, 35]`, `[1, 1]`, `[1, 59229]` |
| opus-mt-tc-bible-big-pqw-en | transformers-auto-model | 4 | `[1, 33]`, `[1, 33]`, `[1, 1]`, `[1, 58311]` |
| opus-mt-tc-bible-big-pqw-fra_ita_por_spa | transformers-auto-model | 4 | `[1, 36]`, `[1, 36]`, `[1, 1]`, `[1, 60729]` |
| opus-mt-tc-bible-big-roa-deu_eng_fra_por_spa | transformers-auto-model | 4 | `[1, 32]`, `[1, 32]`, `[1, 1]`, `[1, 45245]` |
| opus-mt-tc-bible-big-roa-en | transformers-auto-model | 4 | `[1, 31]`, `[1, 31]`, `[1, 1]`, `[1, 55523]` |
| opus-mt-tc-bible-big-sem-deu_eng_fra_por_spa | transformers-auto-model | 4 | `[1, 41]`, `[1, 41]`, `[1, 1]`, `[1, 61171]` |
| opus-mt-tc-bible-big-sem-deu_eng_nld | transformers-auto-model | 4 | `[1, 41]`, `[1, 41]`, `[1, 1]`, `[1, 61309]` |
| opus-mt-tc-bible-big-sem-en | transformers-auto-model | 4 | `[1, 38]`, `[1, 38]`, `[1, 1]`, `[1, 61023]` |
| opus-mt-tc-bible-big-sla-deu_eng_nld | transformers-auto-model | 4 | `[1, 35]`, `[1, 35]`, `[1, 1]`, `[1, 59873]` |
| opus-mt-tc-bible-big-sla-en | transformers-auto-model | 4 | `[1, 35]`, `[1, 35]`, `[1, 1]`, `[1, 59891]` |
| opus-mt-tc-bible-big-tai-deu_eng_fra_por_spa | transformers-auto-model | 4 | `[1, 36]`, `[1, 36]`, `[1, 1]`, `[1, 58838]` |
| opus-mt-tc-bible-big-trk-deu_eng_fra_por_spa | transformers-auto-model | 4 | `[1, 37]`, `[1, 37]`, `[1, 1]`, `[1, 60942]` |
| opus-mt-tc-bible-big-urj-deu_eng_fra_por_spa | transformers-auto-model | 4 | `[1, 37]`, `[1, 37]`, `[1, 1]`, `[1, 59366]` |
| opus-mt-tc-bible-big-urj-deu_eng_nld | transformers-auto-model | 4 | `[1, 38]`, `[1, 38]`, `[1, 1]`, `[1, 59183]` |
| opus-mt-tc-bible-big-urj-fra_ita_por_spa | transformers-auto-model | 4 | `[1, 36]`, `[1, 36]`, `[1, 1]`, `[1, 59759]` |
| opus-mt-tc-bible-big-zhx-deu_eng_fra_por_spa | transformers-auto-model | 4 | `[1, 39]`, `[1, 39]`, `[1, 1]`, `[1, 59178]` |
| opus-mt-tc-bible-big-zhx-en | transformers-auto-model | 4 | `[1, 38]`, `[1, 38]`, `[1, 1]`, `[1, 58592]` |
| orca_mini_3b | transformers-auto-model | 2 | `[1, 21]`, `[1, 21]` |
| osiria_deberta-base-italian-uncased-ner | transformers-auto-model | 2 | `[1, 17]`, `[1, 17]` |
| paulkm_autotrain-lottery_v2-2420075389 | transformers-auto-model | 3 | `[1, 20]`, `[1, 20]`, `[1, 20]` |
| perplexity-ai_pplx-embed-v1-0.6b | transformers-auto-model | 2 | `[1, 7]`, `[1, 7]` |
| philschmid_bart-base-samsum | transformers-auto-model | 3 | `[1, 11, 768]`, `[1, 11, 768]`, `[1, 11]` |
| pix2struct-base | transformers-auto-model | 4 | `[1, 4]`, `[1, 4]`, `[1, 2048, 768]`, `[1, 2048]` |
| polejowska_cdetr-cd45rb-s | transformers-auto-model | 2 | `[1, 3, 800, 800]`, `[1, 800, 800]` |
| polejowska_detr-r101-cd45rb-8ah-6l-256d-4096ffn | transformers-auto-model | 2 | `[1, 3, 800, 800]`, `[1, 800, 800]` |
| polejowska_detr-r50-cd45rb-4ah-6l | transformers-auto-model | 2 | `[1, 3, 800, 800]`, `[1, 800, 800]` |
| polyglot-ko-1.3b | transformers-auto-model | 2 | `[1, 45]`, `[1, 45]` |
| pongjin_roberta_with_kornli | transformers-auto-model | 3 | `[1, 22]`, `[1, 22]`, `[1, 22]` |
| prajjwal1_bert-tiny | transformers-auto-model | 3 | `[1, 11]`, `[1, 11]`, `[1, 11]` |
| praramnine_isl-camembert-beauty-aspect-v2 | transformers-auto-model | 2 | `[1, 27]`, `[1, 27]` |
| prathmeshrmadhu_detr-finetuned-odorv3 | transformers-auto-model | 2 | `[1, 3, 800, 800]`, `[1, 800, 800]` |
| pszemraj_electra-small-discriminator-zeroshot-v1.1 | transformers-auto-model | 3 | `[1, 11]`, `[1, 11]`, `[1, 11]` |
| pszemraj_pegasus-large-book-summary | transformers-auto-model | 3 | `[1, 10]`, `[1, 10]`, `[1, 10]` |
| pszemraj_tFINE-base-300m-samsum | transformers-auto-model | 3 | `[1, 10, 768]`, `[1, 10, 768]`, `[1, 10]` |
| quora-roberta-large | transformers-auto-model | 2 | `[2, 35]`, `[2, 35]` |
| recallapp_CLIP-ViT-B-32-laion2B-s34B-b79K | transformers-auto-model | 3 | `[1, 3, 224, 224]`, `[2, 7]`, `[2, 7]` |
| rice-rice_detr-resnet-50_finetuned_dataset | transformers-auto-model | 2 | `[1, 3, 800, 800]`, `[1, 800, 800]` |
| roberta-base | transformers-auto-model | 2 | `[1, 11]`, `[1, 11]` |
| ru-bart-large | transformers-auto-model | 2 | `[1, 22]`, `[1, 22]` |
| sadhaklal_wide-and-deep-net-california-housing-v2 | transformers-auto-model | 2 | `[3, 6]`, `[3, 5]` |
| sadhaklal_wide-and-deep-net-california-housing-v3 | transformers-auto-model | 2 | `[3, 6]`, `[3, 5]` |
| sarvam-0.5 | transformers-auto-model | 2 | `[1, 22]`, `[1, 22]` |
| satyaalmasian_temporal_tagger_German_GELECTRA | transformers-auto-model | 3 | `[1, 17]`, `[1, 17]`, `[1, 17]` |
| sbulut_finetuned-kde4-en-to-tr | transformers-auto-model | 3 | `[1, 18, 1024]`, `[1, 18, 1024]`, `[1, 18]` |
| sdadas_polish-reranker-roberta-v2 | transformers-auto-model | 2 | `[1, 19]`, `[1, 19]` |
| sentence-transformers_paraphrase-MiniLM-L3-v2 | transformers-auto-model | 3 | `[1, 11]`, `[1, 11]`, `[1, 11]` |
| shaunster_clip_14_model | transformers-auto-model | 3 | `[1, 3, 448, 448]`, `[2, 7]`, `[2, 7]` |
| sheshkar_cmon2 | transformers-auto-model | 2 | `[1, 900, 256]`, `[1, 900, 256]` |
| simjo_dummy-model | transformers-auto-model | 3 | `[1, 19]`, `[1, 19]`, `[1, 19]` |
| simonmun_Eyse_SentenceClassification | transformers-auto-model | 3 | `[1, 10]`, `[1, 10]`, `[1, 10]` |
| sismetanin_mbart_ru_sum_gazeta-ru-sentiment-liniscrowd | transformers-auto-model | 3 | `[1, 15, 1024]`, `[1, 15, 1024]`, `[1, 15]` |
| splade-mini | transformers-auto-model | 2 | `[1, 10]`, `[1, 10]` |
| sshleifer_distilbart-cnn-12-6 | transformers-auto-model | 3 | `[1, 11, 1024]`, `[1, 11, 1024]`, `[1, 11]` |
| sshleifer_distill-pegasus-xsum-16-4 | transformers-auto-model | 3 | `[1, 10, 1024]`, `[1, 10, 1024]`, `[1, 10]` |
| sshleifer_tiny-dbmdz-bert-large-cased-finetuned-conll03-english | transformers-auto-model | 3 | `[1, 13]`, `[1, 13]`, `[1, 13]` |
| studio-ousia_luke-base | transformers-auto-model | 2 | `[1, 11]`, `[1, 11]` |
| super-fast-llm | transformers-auto-model | 2 | `[1, 32]`, `[1, 32]` |
| sureshnam9_esm2_t6_8M_UR50D-finetuned-secondary-structure | transformers-auto-model | 2 | `[1, 13]`, `[1, 13]` |
| t5-small | transformers-auto-model | 3 | `[1, 12, 512]`, `[1, 12, 512]`, `[1, 12]` |
| t5gemma-2-270m | transformers-auto-model | 3 | `[1, 16]`, `[1, 16]`, `[1, 16]` |
| tau_tavbert-he | transformers-auto-model | 3 | `[1, 51]`, `[1, 51]`, `[1, 51]` |
| tiny_starcoder_py | transformers-auto-model | 2 | `[1, 19]`, `[1, 19]` |
| trocr-base-handwritten | transformers-auto-model | 2 | `[1, 3, 384, 384]`, `[1, 1]` |
| truong1301_reranker_pho_BLAI | transformers-auto-model | 3 | `[1, 17]`, `[1, 17]`, `[1, 17]` |
| tsime_detr_mapilary | transformers-auto-model | 3 | `[1, 3, 800, 800]`, `[1, 800, 800]`, `[4, 256]` |
| uer_roberta-base-finetuned-cluener2020-chinese | transformers-auto-model | 3 | `[1, 4]`, `[1, 4]`, `[1, 4]` |
| ukr-models_xlm-roberta-base-uk | transformers-auto-model | 2 | `[1, 17]`, `[1, 17]` |
| upernet-convnext-tiny | transformers-auto-model | 4 | `[1, 96, 128, 128]`, `[1, 192, 64, 64]`, `[1, 384, 32, 32]`, `[1, 768, 16, 16]` |
| upfeatmediainc_owlv2-base-patch16-ensemble | transformers-auto-model | 3 | `[1, 3, 960, 960]`, `[1, 7]`, `[1, 7]` |
| valhalla_distilbart-mnli-12-1 | transformers-auto-model | 3 | `[1, 11, 1024]`, `[1, 11, 1024]`, `[1, 11]` |
| valhalla_distilbart-mnli-12-3 | transformers-auto-model | 3 | `[1, 11, 1024]`, `[1, 11, 1024]`, `[1, 11]` |
| venishpatidar_wa-ad-mod | transformers-auto-model | 2 | `[1, 9, 1024]`, `[1, 9]` |
| voxreality_src_ctx_aware_nllb_1.3B | transformers-auto-model | 4 | `[1, 16, 1024]`, `[1, 16, 1024]`, `[1, 16]`, `[1, 16]` |
| vpelloin_MEDIA_NLU-flaubert_oral_mixed | transformers-auto-model | 4 | `[1, 20]`, `[1, 20]`, `[1, 512]`, `[1, 20]` |
| w11wo_javanese-gpt2-small-imdb-classifier | transformers-auto-model | 2 | `[1, 30]`, `[1, 30]` |
| w11wo_sundanese-gpt2-base-emotion-classifier | transformers-auto-model | 2 | `[1, 22]`, `[1, 22]` |
| xc2450_distilbert-portuguese-cased-finetuned-bec_classification | transformers-auto-model | 3 | `[1, 18]`, `[1, 18]`, `[1, 18]` |
| xlnet-base-cased | transformers-auto-model | 3 | `[1, 13]`, `[1, 13]`, `[1, 13]` |
| ydshieh_tiny-random-gptj-for-sequence-classification | transformers-auto-model | 2 | `[1, 19, 32]`, `[1, 19]` |
| yiyanghkust_finbert-tone | transformers-auto-model | 3 | `[1, 12]`, `[1, 12]`, `[1, 12]` |
| yuvalkirstain_PickScore_v1 | transformers-auto-model | 3 | `[1, 3, 224, 224]`, `[2, 7]`, `[2, 7]` |

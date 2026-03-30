# Text-Generation 抓取任务报告

> 时间: 2026-03-26 21:14
> 任务类型: HuggingFace text-generation 模型抓取
> 负责人: worker (Ducc)

## 执行结果

| 指标 | 数值 |
|------|------|
| 启动批次 | 10 (每批次34个模型) |
| 成功提取模型 | 3 (25个子图) |
| 已下载模型 | 15 (缓存中) |
| 失败批次 | 10 |
| 失败原因 | CUDA/NVML错误、 磁盘空间不足 |

## 成功提取的模型
1. **Qwen_Qwen2.5-7B-Instruct**: 15个子图
2. **a16fa581b59c35f6ed79c7fd8d39d518e26b0ffd**: 5个子图 (模型名未知)
3. **3d0483f76c0218a17d90d170d4dd2084882e429f**: 5个子图 (模型名未知)

## 已下载但未提取的模型 (15个)
由于CUDA/NVML错误，这些模型已下载但无法完成图提取:
- cyberagent/open-calm-3b
- google/t5gemma-s-s-prefixlm
- HuggingFaceM4/tiny-random-LlamaForCausalLM
- HuggingFaceTB/SmolLM-135M-Instruct
- HuggingFaceTB/SmolLM2-135M
- LGAI-EXAONE/EXAONE-Deep-7.8T
- microsoft/biogpt
- microsoft/Phi-tiny-MoE-instruct
- Nanbeige/Nanbeige4.1-3B
- NousResearch/Llama-2-7b-hf
- peft-internal-testing/tiny-dummy-qwen2
- Qwen/Qwen2.5-14B-Instruct-AWQ
- Qwen/Qwen3-0.6B
- Qwen/Qwen3-4B-Base
- trl-internal-testing/tiny-Qwen3MoeForCausalLM
- unsloth/Llama-3.1-8B-Instruct
- zai-org/GLM-4.5-Air

- zai-org/GLM-5.5-Air (partial)

## 问题分析
### CUDA/NVML错误
```
RuntimeError: NVML_SUCCESS == DriverAPI::get()->nvmlInit_v2_() INTERNAL ASSERT FAILED
```
- 环境没有可用的NVIDIA GPU
- 模型尝试使用GPU但NVML初始化失败
- 所有进程因此卡住

### 网络问题
- 部分模型下载时遇到SSL超时
- 代理连接不稳定

## 建议
1. **CPU-only模式**: 设置 `CUDA_VISIBLE_DEVICES=""` 强制使用CPU
2. **清理磁盘空间**: 释放磁盘空间后重试
3. **分批执行**: 减少并发数量，避免资源竞争

## 下一步
- [ ] 配置CPU模式重新运行
- [ ] 清理磁盘空间后继续

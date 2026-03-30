# HuggingFace 模型下载方法汇总

## 方法一：使用 hf-mirror 镜像站（推荐）

### 1. 安装依赖
```bash
pip install -U huggingface_hub hf_transfer
```

### 2. 设置环境变量
```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=1  # 启用加速下载
```

### 3. 命令行下载
```bash
# 下载整个模型
huggingface-cli download --resume-download meta-llama/Llama-2-7b-hf --local-dir ./Llama-2-7b-hf

# 下载数据集
huggingface-cli download --repo-type dataset --resume-download wikitext --local-dir wikitext
```

### 4. Python脚本下载
```python
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="meta-llama/Llama-2-7b-hf",
    repo_type="model",
    local_dir="./Llama-2-7b-hf",
    max_workers=8,
)
```

## 方法二：使用 hfd 下载工具

hfd 是 hf-mirror 开发的专用下载工具，基于 aria2，稳定高速不断线。

### 1. 下载配置
```bash
wget https://hf-mirror.com/hfd/hfd.sh
chmod a+x hfd.sh
export HF_ENDPOINT=https://hf-mirror.com
```

### 2. 下载模型
```bash
./hfd.sh gpt2
```

### 3. 下载数据集
```bash
./hfd.sh wikitext --dataset
```

## 方法三：使用 ModelScope（国内镜像）

### 1. 安装依赖
```bash
pip install modelscope
```

### 2. 下载模型
```bash
modelscope download --model Qwen/Qwen3.5-35B-A3B
```

### 3. 下载单个文件
```bash
modelscope download --model Qwen/Qwen3.5-35B-A3B README.md --local_dir ./dir
```

## 关键参数说明

| 参数 | 说明 |
|------|------|
| --resume-download | 断点续传 |
| --local-dir | 本地保存路径 |
| --repo-type | 资源类型：model/dataset/space |
| --max-workers | 并发线程数（默认8） |
| --include | 只下载指定文件类型 |
| --exclude | 排除指定文件类型 |

## 实践建议

1. **优先使用 hf-mirror.com 镜像**，国内访问速度快
2. **启用 hf_transfer 加速**，大幅提升下载速度
3. **使用断点续传**，避免大文件下载中断后重新开始
4. **只下载必要文件**（如 config.json），不下载大权重文件

---
*文档来源：本地知识库整理*
*实践验证后请更新此文档*

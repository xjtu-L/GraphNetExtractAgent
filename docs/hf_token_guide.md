# HuggingFace Token 配置指南

## 背景

在使用 hf-mirror.com 镜像站下载模型时，会遇到 429 (Too Many Requests) 错误。配置 HuggingFace Token 可以有效解决此问题。

---

## 一、问题现象

### 429 错误示例

```
HTTP Error 429 thrown while requesting HEAD https://hf-mirror.com/api/resolve-cache/models/xxx/config.json
```

### 原因分析

| 情况 | 速率限制 |
|------|----------|
| 无 Token | 严格限制，容易触发 429 |
| 有 Token | 更宽松的限制，更稳定 |

---

## 二、配置方法

### 方法1: 环境变量 (推荐)

```python
import os
os.environ["HF_TOKEN"] = "hf_xxxxxxxxxxxxxxxxxxxx"
```

### 方法2: 配置文件

```bash
# 创建配置目录
mkdir -p ~/.huggingface

# 写入 token
echo "hf_xxxxxxxxxxxxxxxxxxxx" > ~/.huggingface/token
```

### 方法3: 在脚本中配置

```python
# 在抽取脚本开头添加
os.environ["HF_TOKEN"] = "hf_xxxxxxxxxxxxxxxxxxxx"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
```

---

## 三、获取 Token

1. 访问 https://huggingface.co/settings/tokens
2. 点击 "New token"
3. 选择 "Read" 权限即可
4. 复制生成的 token (以 `hf_` 开头)

---

## 四、验证配置

```python
from huggingface_hub import HfApi

api = HfApi()
try:
    user = api.whoami()
    print(f"✓ Token 有效! 用户: {user['name']}")
except Exception as e:
    print(f"✗ Token 验证失败: {e}")
```

---

## 五、完整配置模板

```python
import os

# 代理设置 (如果需要)
os.environ["http_proxy"] = "http://proxy.example.com:port"
os.environ["https_proxy"] = "http://proxy.example.com:port"

# HuggingFace 配置
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_TOKEN"] = "hf_xxxxxxxxxxxxxxxxxxxx"

# 可选: 禁用进度条和警告
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
```

---

## 六、常见问题

### Q1: Token 配置后仍然有 429 错误？

A: 尝试以下方法:
- 降低并发数
- 增加请求间隔
- 检查 token 是否正确

### Q2: 如何在多进程环境中使用？

A: 在每个进程开始时都设置环境变量，或者在主脚本开头全局设置。

### Q3: Token 泄露了怎么办？

A: 立即在 HuggingFace 设置页面撤销该 token，然后生成新的。

---

## 七、效果对比

| 项目 | 无 Token | 有 Token |
|------|----------|----------|
| 429 错误频率 | 频繁 | 几乎没有 |
| 并发稳定性 | 低 | 高 |
| 建议并发数 | 3-5 | 10+ |

---

*文档版本：1.0*
*创建日期：2026-04-19*
*作者：Claude Code*
# Worker目录组织规范指南

> 本文档说明 GraphNetExtractAgent 的多 Worker 协作目录组织规范。
>
> **重要性**: 必须遵守，避免 Worker 间数据冲突

---

## 1. 目录结构规范

### 1.1 新的目录组织方式

为避免多个 Worker 之间的冲突和数据混乱，采用以下目录组织方式：

```
graphnet_workspace/
└── huggingface/
    ├── worker0/                    # Worker0 的工作目录
    │   ├── text-generation/
    │   │   ├── model1_name/
    │   │   └── model2_name/
    │   ├── text-classification/
    │   ├── fill-mask/
    │   └── feature-extraction/
    │
    ├── worker1/                    # Worker1 的工作目录
    │   ├── text-generation/
    │   ├── text-classification/
    │   ├── fill-mask/
    │   └── feature-extraction/
    │
    ├── worker2/                    # Worker2 的工作目录
    │   ├── text-generation/
    │   ├── text-classification/
    │   ├── fill-mask/
    │   └── feature-extraction/
    │
    └── shared/                     # 共享目录（可选）
        └── common_resources/
```

### 1.2 目录层级说明

| 层级 | 命名规范 | 示例 | 说明 |
|------|---------|------|------|
| Worker目录 | `worker{N}/` | worker0/, worker1/ | N为Worker编号 |
| Task目录 | HF任务类型 | text-generation/ | 按HuggingFace任务分类 |
| 模型目录 | `org_model/` | facebook_xmod-base/ | /替换为_ |

---

## 2. 使用示例

### 2.1 Worker0 抓取模型

```bash
# 设置输出路径为 Worker0 的目录
export GRAPH_NET_EXTRACT_WORKSPACE="/ssd1/liangtai-work/graphnet_workspace/huggingface/worker0/text-generation"

# 抓取模型
python extract_script.py facebook/xmod-base

# 结果位置
# /ssd1/liangtai-work/graphnet_workspace/huggingface/worker0/text-generation/facebook_xmod-base/
#     ├── model.py
#     ├── graph_net.json
#     └── ...
```

### 2.2 Worker1 抓取模型

```bash
# 设置输出路径为 Worker1 的目录
export GRAPH_NET_EXTRACT_WORKSPACE="/ssd1/liangtai-work/graphnet_workspace/huggingface/worker1/text-generation"

# 抓取模型
python extract_script.py Xenova/ernie-2.0-large-en

# 结果位置
# /ssd1/liangtai-work/graphnet_workspace/huggingface/worker1/text-generation/Xenova_ernie-2.0-large-en/
```

---

## 3. 为什么需要这样组织

### 3.1 避免冲突

**问题场景**：
```
# ❌ 旧结构（不推荐使用）
huggingface/
└── text-generation/
    ├── facebook_xmod-base/      # Worker0 正在写入
    └── Xenova_ernie-2.0-large-en/  # Worker1 同时写入（可能冲突）
```

**解决方案**：
```
# ✅ 新结构（推荐）
huggingface/
├── worker0/text-generation/facebook_xmod-base/      # Worker0 独立工作
└── worker1/text-generation/Xenova_ernie-2.0-large-en/  # Worker1 独立工作
```

### 3.2 责任清晰

- 每个 Worker 对自己的目录负责
- 出现问题时容易追踪责任人
- 便于排查和修复问题

### 3.3 便于统计

```bash
# 统计各 Worker 的贡献
for worker in worker0 worker1 worker2; do
    count=$(find /ssd1/liangtai-work/graphnet_workspace/huggingface/$worker -name "graph_hash.txt" | wc -l)
    echo "$worker: $count 个模型"
done
```

### 3.4 易于合并

```bash
# 后期合并到统一数据库
for worker in worker0 worker1 worker2; do
    cp -r /ssd1/liangtai-work/graphnet_workspace/huggingface/$worker/*/* \
          /ssd1/liangtai-work/graphnet_workspace/huggingface/merged/
done
```

---

## 4. 常见问题

### 4.1 模型抓取成功但放错位置

**问题**：`katuni4ka_tiny-random-qwen3` 模型抓取成功，但直接放到了 `huggingface/` 目录下，没有按 Worker 归类。

```
# ❌ 错误位置
huggingface/katuni4ka_tiny-random-qwen3/

# ✅ 正确位置
huggingface/worker0/text-generation/katuni4ka_tiny-random-qwen3/
```

**解决方案**：
```bash
# 1. 创建正确的目录结构
mkdir -p /ssd1/liangtai-work/graphnet_workspace/huggingface/worker0/text-generation

# 2. 移动模型到正确位置
mv /ssd1/liangtai-work/graphnet_workspace/huggingface/katuni4ka_tiny-random-qwen3 \
   /ssd1/liangtai-work/graphnet_workspace/huggingface/worker0/text-generation/

# 3. 更新后续抓取脚本，使用正确的输出路径
export GRAPH_NET_EXTRACT_WORKSPACE="/ssd1/liangtai-work/graphnet_workspace/huggingface/worker0/text-generation"
```

### 4.2 如何识别未归类的模型

```bash
#!/bin/bash
# 找出 huggingface 根目录下未归类的模型

WORKSPACE="/ssd1/liangtai-work/graphnet_workspace/huggingface"

for dir in "$WORKSPACE"/*; do
    if [ -d "$dir" ]; then
        basename=$(basename "$dir")
        # 跳过 Worker 目录和 Task 目录
        if [[ "$basename" == worker* ]] || \
           [[ "$basename" == "shared" ]] || \
           [[ "$basename" == "text-generation" ]] || \
           [[ "$basename" == "text-classification" ]] || \
           [[ "$basename" == "fill-mask" ]] || \
           [[ "$basename" == "feature-extraction" ]]; then
            continue
        fi

        # 检查是否是模型目录（有 graph_hash.txt 或 model.py）
        if [ -f "$dir/graph_hash.txt" ] || [ -f "$dir/model.py" ]; then
            echo "未归类模型: $basename"
        fi
    fi
done
```

### 4.3 多个 Worker 如何分配任务

```bash
# Worker0 负责
tasks_worker0=("text-generation" "translation")

# Worker1 负责
tasks_worker1=("text-classification" "fill-mask")

# Worker2 负责
tasks_worker2=("feature-extraction" "automatic-speech-recognition")

# 创建对应目录
for worker in worker0 worker1 worker2; do
    for task in "${tasks_${worker}[@]}"; do
        mkdir -p "/ssd1/liangtai-work/graphnet_workspace/huggingface/$worker/$task"
    done
done
```

---

## 5. 迁移指南

### 5.1 从旧结构迁移

如果你之前使用旧结构，按以下步骤迁移：

```bash
# 1. 停止所有抓取任务
pkill -f extract

# 2. 创建新的 Worker 目录结构
mkdir -p /ssd1/liangtai-work/graphnet_workspace/huggingface/worker0/{text-generation,text-classification,fill-mask}
mkdir -p /ssd1/liangtai-work/graphnet_workspace/huggingface/worker1/{text-generation,text-classification,fill-mask}

# 3. 将已抓取的模型按 Task 分类移动（手动或脚本）
# 示例：假设你是 Worker0，负责 text-generation
mv /ssd1/liangtai-work/graphnet_workspace/huggingface/text-generation/model1 \
   /ssd1/liangtai-work/graphnet_workspace/huggingface/worker0/text-generation/

# 4. 更新所有脚本中的输出路径
# 修改前:
# os.environ["GRAPH_NET_EXTRACT_WORKSPACE"] = "/ssd1/liangtai-work/graphnet_workspace/huggingface/text-generation"
# 修改后:
# os.environ["GRAPH_NET_EXTRACT_WORKSPACE"] = "/ssd1/liangtai-work/graphnet_workspace/huggingface/worker0/text-generation"
```

### 5.2 批量移动未归类模型

```bash
#!/bin/bash
# 批量移动未归类模型到 Worker 目录

WORKSPACE="/ssd1/liangtai-work/graphnet_workspace/huggingface"
WORKER="worker0"  # 修改为你的 Worker 编号

# 遍历 huggingface 根目录
for dir in "$WORKSPACE"/*; do
    if [ -d "$dir" ]; then
        basename=$(basename "$dir")

        # 跳过特殊目录
        if [[ "$basename" == worker* ]] || \
           [[ "$basename" == "shared" ]]; then
            continue
        fi

        # 判断 Task 类型（从 config.json 或根据名称推断）
        task="text-generation"  # 默认
        if [ -f "$dir/config.json" ]; then
            # 尝试从 config.json 推断
            if grep -q "ForSequenceClassification" "$dir/config.json" 2>/dev/null; then
                task="text-classification"
            elif grep -q "ForMaskedLM" "$dir/config.json" 2>/dev/null; then
                task="fill-mask"
            fi
        fi

        # 创建目标目录
        mkdir -p "$WORKSPACE/$WORKER/$task"

        # 移动模型
        echo "移动: $basename -> $WORKER/$task/"
        mv "$dir" "$WORKSPACE/$WORKER/$task/"
    fi
done
```

---

## 6. 检查清单

### 6.1 抓取前检查

- [ ] 是否设置了正确的 `GRAPH_NET_EXTRACT_WORKSPACE`
- [ ] 路径是否包含 Worker 编号（如 worker0/）
- [ ] 路径是否包含 Task 类型（如 text-generation/）

### 6.2 抓取后检查

- [ ] 模型是否在正确的 Worker 目录下
- [ ] 模型是否在正确的 Task 目录下
- [ ] 是否有文件散落在 huggingface/ 根目录

---

## 7. 相关文档

- [多Worker协作流程](multi_worker_collaboration.md)
- [HUGGINGFACE_EXTRACTION_GUIDE](HUGGINGFACE_EXTRACTION_GUIDE.md)
- [EXTRACTION_EXPLAINED](EXTRACTION_EXPLAINED.md)

---

*最后更新：2026-03-30*

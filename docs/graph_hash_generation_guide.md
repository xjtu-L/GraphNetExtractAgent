# Graph Hash 生成指南

## 概述

graph_hash.txt 是模型计算图的哈希值文件，用于快速识别和去重模型。它基于 model.py 的内容生成 SHA256 哈希值。

## 生成方法

### 命令行工具

使用 GraphNet 提供的工具脚本生成 graph_hash.txt：

```bash
cd /root/GraphNet
python tools/generate_graph_hash.py --samples-dir <目标目录> [--overwrite] [--dry-run]
```

### 参数说明

| 参数 | 说明 |
|------|------|
| `--samples-dir` | 要扫描的目录路径（必需） |
| `--overwrite` | 覆盖已存在的 graph_hash.txt（可选） |
| `--dry-run` | 试运行模式，不实际写入文件（可选） |

### 示例

#### 1. 首次生成（跳过已存在的）
```bash
python tools/generate_graph_hash.py \
    --samples-dir /root/graphnet_workspace/huggingface/worker1
```

#### 2. 强制重新生成所有（覆盖模式）
```bash
python tools/generate_graph_hash.py \
    --samples-dir /root/graphnet_workspace/huggingface/worker1 \
    --overwrite
```

#### 3. 试运行（查看会修改哪些文件）
```bash
python tools/generate_graph_hash.py \
    --samples-dir /root/graphnet_workspace/huggingface/worker1 \
    --dry-run
```

### 批量生成多个目录

```bash
# 生成所有 worker 目录的 graph_hash
for dir in worker1 worker2; do
    echo "Processing $dir..."
    python tools/generate_graph_hash.py \
        --samples-dir /root/graphnet_workspace/huggingface/$dir \
        --overwrite
done
```

## 输出说明

脚本会输出处理结果：

```
[OK] /path/to/model/subgraph_0/graph_hash.txt
[OK] /path/to/model/subgraph_1/graph_hash.txt
...
Written: 3740, Skipped: 0, Errors: 0
```

- **Written**: 新写入或覆盖的文件数
- **Skipped**: 已存在且未覆盖的文件数
- **Errors**: 处理失败的文件数

## 注意事项

1. **基于 model.py**: graph_hash.txt 是根据 model.py 文件内容生成的
2. **每个子图一个**: 如果模型有多个子图（subgraph_*），每个子图都会有独立的 graph_hash.txt
3. **SHA256 哈希**: 使用 SHA256 算法对 model.py 内容进行哈希
4. **跳过 shape_patches_**: 脚本会自动跳过包含 shape_patches_ 的目录

## 验证生成结果

```bash
# 统计 graph_hash.txt 数量
find /root/graphnet_workspace/huggingface/worker1 -name "graph_hash.txt" | wc -l

# 对比 model.py 数量
find /root/graphnet_workspace/huggingface/worker1 -name "model.py" | wc -l

# 查看某个 graph_hash.txt 内容
cat /path/to/model/subgraph_0/graph_hash.txt
```

## 故障排除

### 问题：生成的 hash 数量少于 model.py 数量

可能原因：
- 部分 model.py 文件为空或损坏
- 权限问题导致无法读取

解决方法：
```bash
# 检查空文件
find /root/graphnet_workspace/huggingface/worker1 -name "model.py" -size 0

# 修复后重新生成
python tools/generate_graph_hash.py --samples-dir <目录> --overwrite
```

### 问题：生成过程中出现权限错误

解决方法：
```bash
# 确保目录权限正确
chmod -R 755 /root/graphnet_workspace/huggingface/worker1

# 重新运行
python tools/generate_graph_hash.py --samples-dir <目录> --overwrite
```

---

*文档创建时间: 2026-04-01*
*适用版本: GraphNet tools/generate_graph_hash.py*

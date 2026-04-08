# Graph Hash 补充生成指南

## 概述

在模型抽取过程中，有时会遇到残图（已有 model.py 等文件但缺少 graph_hash.txt）的情况。本指南介绍如何使用 `generate_graph_hash.py` 工具直接生成 graph_hash.txt，无需重新抽取模型。

## 适用场景

- 残图修复：模型已有 model.py 但缺少 graph_hash.txt
- 批量补充：为大量模型目录快速生成 hash
- 避免重复抽取：节省时间和计算资源

## 使用方法

### 1. 为单个模型目录生成

```bash
cd /root/GraphNet
python tools/generate_graph_hash.py \
  --samples-dir /path/to/model_directory
```

### 2. 为批量模型生成

```bash
cd /root/GraphNet
python tools/generate_graph_hash.py \
  --samples-dir /ssd1/liangtai-work/graphnet_workspace/huggingface/worker1/feature-extraction
```

### 3. 遍历多个任务类型生成

```bash
cd /root/GraphNet
for dir in /ssd1/liangtai-work/graphnet_workspace/huggingface/worker1/*/; do
  [ -d "$dir" ] || continue
  task=$(basename "$dir")
  echo "生成 $task 的 hash..."
  python tools/generate_graph_hash.py --samples-dir "$dir" 2>/dev/null | tail -3
done
```

## 输出说明

工具会输出以下信息：

```
Scanning directories:
  /path/to/model_directory

Written: 405, Skipped: 0, Errors: 0
```

- **Written**: 成功生成的 graph_hash.txt 数量
- **Skipped**: 已存在的 graph_hash.txt 数量
- **Errors**: 生成失败的错误数量

## 验证结果

生成后，可以使用以下命令验证：

```bash
# 统计总数量
find /path/to/directory -name "graph_hash.txt" | wc -l

# 查看唯一模型数
find /path/to/directory -name "graph_hash.txt" | \
  sed 's|/subgraph_[0-9]*/graph_hash.txt||' | \
  sort -u | wc -l

# 查看示例文件内容
head -3 /path/to/model/subgraph_0/graph_hash.txt
```

## 实际案例

### 案例1：补充 feature-extraction 77个模型的 hash

```bash
# 停止抽取进程（如有）
kill -TERM <PID>

# 生成 graph_hash.txt
cd /root/GraphNet
python tools/generate_graph_hash.py \
  --samples-dir /ssd1/liangtai-work/graphnet_workspace/huggingface/worker1/feature-extraction

# 验证结果
find /ssd1/liangtai-work/graphnet_workspace/huggingface/worker1/feature-extraction/ \
  -name "graph_hash.txt" | wc -l
# 输出: 405
```

### 案例2：为 Worker1 所有目录生成

```bash
cd /root/GraphNet

for dir in /ssd1/liangtai-work/graphnet_workspace/huggingface/worker1/*/; do
  [ -d "$dir" ] || continue
  task=$(basename "$dir")
  count=$(find "$dir" -name "graph_hash.txt" 2>/dev/null | wc -l)
  echo "$task: 已有 $count 个hash"

  if [ "$count" -eq 0 ]; then
    echo "  生成 $task 的hash..."
    python tools/generate_graph_hash.py --samples-dir "$dir" 2>/dev/null | tail -3
  fi
done
```

## 注意事项

1. **数据来源**：生成的 hash 基于 model.py 文件内容计算 SHA-256 哈希
2. **唯一性**：每个 subgraph 的 hash 都是唯一的
3. **兼容性**：生成的 hash 格式与正常抽取的完全一致
4. **跳过机制**：工具会自动跳过已存在 graph_hash.txt 的文件

## 原理说明

`generate_graph_hash.py` 工具工作原理：

1. 扫描指定目录下的所有模型子目录
2. 查找每个 subgraph 目录中的 model.py 文件
3. 计算 model.py 内容的 SHA-256 哈希值
4. 将哈希值写入 graph_hash.txt

哈希值格式：
- 64位十六进制字符串
- 示例：`b45be332d2c0dd80f13284d41678469132c0c510b8c817c0b2858907cf4a8f52`

## 相关工具

- `extract_from_model_list.py`：正常抽取模型
- `generate_extract_py.py`：为残图生成 extract.py
- `generate_graph_hash.py`：生成 graph_hash.txt（本文档）

## 常见问题

**Q: 为什么有些目录生成 0 个 hash？**
A: 可能是因为：
- 目录中没有 model.py 文件
- 目录结构不符合预期
- subgraph 目录不存在

**Q: 生成的 hash 与正常抽取的一致吗？**
A: 是的，都是基于 model.py 内容的 SHA-256 哈希，格式完全一致。

**Q: 可以重复运行吗？**
A: 可以，工具会自动跳过已存在的 graph_hash.txt，不会重复生成。

## 版本历史

- **2026-04-02**: 初始版本，记录 feature-extraction 77个模型的补充生成方法

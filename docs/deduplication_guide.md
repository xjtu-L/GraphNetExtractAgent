# 计算图去重指南

## 背景

在批量抽取HuggingFace模型计算图时，会遇到重复的图结构。本文档总结去重的最佳实践。

---

## 一、为什么会有重复？

### 1. 同架构模型共享子图

不同模型可能使用相同的子图结构，例如：

| 模型类型 | 重复原因 |
|----------|----------|
| MoE模型 (Mixtral等) | 多个模型共享相同的expert层结构 |
| LLaMA系列 | 相同的attention层、decoder层 |
| BERT系列 | 相同的encoder层结构 |
| 微调模型 | 基础模型结构相同，仅权重不同 |

### 2. 示例

```
模型A/subgraph_1/graph_hash.txt = "cafb109fe3aefa34..."
模型B/subgraph_3/graph_hash.txt = "cafb109fe3aefa34..."  # 相同hash!
```

---

## 二、去重策略

### 1. Hash生成

使用 SHA256 对 `model.py` 内容计算hash：

```python
import hashlib

def generate_graph_hash(model_py_path):
    with open(model_py_path, 'r') as f:
        hash_val = hashlib.sha256(f.read().encode()).hexdigest()
    return hash_val  # 64位十六进制字符串
```

### 2. 去重原则

**核心原则：每个唯一hash只保留一份计算图**

- 已提交的hash不再重复提交
- 同一批次内hash去重
- 子图级别的hash也要去重

### 3. 去重流程

```
┌─────────────────────────────────────────────────────────────┐
│                      去重流程                                │
├─────────────────────────────────────────────────────────────┤
│  1. 收集已提交的所有hash                                     │
│     └─> 遍历 submit_extracted/**/*.graph_hash.txt           │
│                                                              │
│  2. 遍历新抽取的模型                                         │
│     └─> 检查每个model.py对应的hash                          │
│                                                              │
│  3. 筛选新hash                                               │
│     └─> hash not in submitted_hashes                         │
│     └─> hash not in used_hashes (同批次去重)                 │
│                                                              │
│  4. 复制唯一hash的计算图                                     │
│     └─> 添加到归档目录                                       │
│                                                              │
│  5. 验证并打包                                               │
│     └─> 确认 graph_hash.txt数 == 唯一hash数                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 三、代码实现

### 1. 收集已提交hash

```python
import glob

def collect_submitted_hashes(submit_dir):
    """收集已提交的所有hash"""
    submitted_hashes = set()
    for hash_file in glob.glob(os.path.join(submit_dir, '**', 'graph_hash.txt'), recursive=True):
        with open(hash_file, 'r') as f:
            submitted_hashes.add(f.read().strip())
    return submitted_hashes
```

### 2. 处理多子图模型

```python
def process_multigraph_model(model_dir, target_dir, all_existing, used_hashes):
    """处理多子图模型，每个hash只保留一次"""
    subgraphs = sorted([d for d in os.listdir(model_dir) if d.startswith('subgraph_')])

    for sg in subgraphs:
        sg_path = os.path.join(model_dir, sg)
        hash_file = os.path.join(sg_path, 'graph_hash.txt')

        if os.path.exists(hash_file):
            with open(hash_file, 'r') as f:
                h = f.read().strip()

            # 只添加新hash
            if h not in all_existing and h not in used_hashes:
                dest_sg = os.path.join(target_dir, sg)
                shutil.copytree(sg_path, dest_sg)
                used_hashes.add(h)
```

### 3. 验证去重结果

```python
def verify_deduplication(tar_path):
    """验证压缩包内无重复hash"""
    hashes = []
    with tarfile.open(tar_path, 'r:gz') as tar:
        for member in tar.getmembers():
            if 'graph_hash.txt' in member.name:
                f = tar.extractfile(member)
                if f:
                    hashes.append(f.read().decode().strip())

    file_count = len(hashes)
    unique_count = len(set(hashes))

    if file_count == unique_count:
        print(f"✓ 验证通过: {file_count} 个文件, {unique_count} 个唯一hash")
    else:
        print(f"✗ 存在重复: {file_count} 个文件, {unique_count} 个唯一hash")
        duplicates = file_count - unique_count
        print(f"  重复数量: {duplicates}")

    return file_count == unique_count
```

---

## 四、统计指标

### 关键指标说明

| 指标 | 说明 |
|------|------|
| 非空模型目录数 | 成功抽取的模型数量 |
| 计算图数 (model.py) | 包含子图的总数 |
| 已提交hash数 | 历史已提交的唯一hash |
| 本次新唯一hash数 | 去重后新增的hash数量 |

### 示例统计

```
=== 去重后统计 ===
原有hash数: 1071
新增hash数: 318
graph_hash.txt 文件数: 1389
唯一 hash 数: 1389
```

**验证标准：graph_hash.txt 文件数 == 唯一 hash 数**

---

## 五、常见问题

### Q1: 子图去重后模型不完整怎么办？

A: 这是正常的。如果模型的某个子图hash已存在，该子图不会被重复添加。GraphNet关注的是计算图结构，相同结构的子图只需一份。

### Q2: 如何处理空目录？

A: 如果多子图模型的所有子图都是重复hash，会留下空目录，应删除：

```python
if not os.listdir(dest_dir):
    os.rmdir(dest_dir)
```

### Q3: 压缩包内hash重复怎么办？

A: 重新解压、去重、打包：

```bash
# 解压
tar -xzf lt_submit4.19.tar.gz -C temp/

# 运行去重脚本
python dedup.py

# 重新打包
tar -czf lt_submit4.19.tar.gz lt_submit4.19/
```

---

## 六、最佳实践

1. **保留备份**：去重前备份原始压缩包
2. **先解压再合并**：解压已有压缩包，再添加新模型
3. **验证结果**：确保 `文件数 == 唯一hash数`
4. **记录统计**：保存每次去重的统计信息

---

## 七、实际案例

### 案例: Worker1 + Worker2 合并去重

**初始状态：**
- Worker1: 290 模型, 629 计算图, 224 唯一hash
- Worker2: 269 模型, 391 计算图, 194 唯一hash
- 已提交: 10,328 hash

**去重结果：**
- 合并后唯一hash: 357 (重叠2个)
- 新增唯一hash: 318

**关键代码：**
```python
# 合并hash集合时去重
all_hashes = w1_hashes | w2_hashes
new_hashes = all_hashes - submitted_hashes
```

---

*文档版本: 1.1*
*创建日期: 2026-04-19*
*最后更新: 2026-04-19*
*作者: Claude Code*
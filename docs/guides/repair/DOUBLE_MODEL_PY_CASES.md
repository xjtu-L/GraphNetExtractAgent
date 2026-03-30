# 双重 model.py 案例汇总

> 记录所有根目录和 subgraph_0/ 同时存在 model.py 的案例
> 创建时间: 2026-03-29
> 紧急处理状态: 进行中

---

## 问题描述

**现象**: 同一模型目录下同时存在两份 model.py
- 根目录: 简化版占位符 (~300-500 bytes)
- subgraph_0/: 完整计算图 (~1KB+)

**影响**: 导致模型目录不整洁，可能干扰修复流程

---

## 统计数据

| 任务类型 | 案例数量 | 占比 |
|---------|---------|------|
| fill-mask | 729 | 59.2% |
| text-classification | 287 | 23.3% |
| text-generation | 216 | 17.5% |
| **总计** | **1232** | **100%** |

---

## 处理方案

### 处理原则
1. **保留必需文件**: graph_hash.txt, extract.py, config.json, subgraph_0/目录
2. **删除多余文件**: model.py, weight_meta.py, input_meta.py, graph_net.json等
3. **保留子图数据**: subgraph_*/ 目录下的完整计算图
4. **备份可选**: 重要文件可添加 .bak 后缀备份

### 处理脚本
```bash
#!/bin/bash
# 处理双重model.py案例 - 清理根目录多余文件

MODEL_DIR="$1"

echo "处理: $MODEL_DIR"

# 备份后删除根目录多余文件
for file in model.py weight_meta.py input_meta.py input_tensor_constraints.py graph_net.json; do
    if [ -f "$MODEL_DIR/$file" ]; then
        mv "$MODEL_DIR/$file" "$MODEL_DIR/${file}.bak" 2>/dev/null
        echo "  备份并删除: $file"
    fi
done

# 确认保留的文件
echo "  保留文件:"
ls -la "$MODEL_DIR/" | grep -E "(graph_hash.txt|extract.py|config.json)" || true

# 确认子图目录
echo "  子图目录:"
ls -d "$MODEL_DIR"/subgraph_* 2>/dev/null | head -3

echo "  ✓ 处理完成"
```

---

## 分批处理计划

- **批次1**: fill-mask 前250个
- **批次2**: fill-mask 后479个
- **批次3**: text-classification 全部287个
- **批次4**: text-generation 前108个
- **批次5**: text-generation 后108个

---

**处理状态**: 待执行
**负责人**: Worker1
**预计完成时间**: 1小时内

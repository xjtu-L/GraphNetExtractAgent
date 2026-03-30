# 残图修复指南 (Residual Repair Guide)

## 残图判断标准（最终确认版）

一个模型目录被判定为**完整**，必须同时满足以下所有条件：

### 完整模型标准

1. **必须有 graph_hash.txt**
   - 计算图的哈希值文件
   - 位于模型根目录

2. **必须有 extract.py**
   - 用于重新抽取的脚本
   - 位于模型根目录

3. **model.py 必须有真正的 GraphModule 计算图**
   - 必须包含 `class GraphModule` 定义
   - 必须继承 `torch.nn.Module`
   - 必须包含实际的 `forward` 函数实现
   - **不能**只是包装函数（如 `get_model()`）

   ❌ **无效 model.py 示例**（算残图）：
   ```python
   # 仅包装函数，无实际计算图
   def get_model():
       config = AutoConfig.from_pretrained(...)
       model = AutoModel.from_config(config)
       return model
   ```

   ✅ **有效 model.py 示例**（算完整）：
   ```python
   class GraphModule(torch.nn.Module):
       def forward(self, ...):
           # 实际的计算图操作
           ...
   ```

4. **如果有 subgraph，每个 subgraph 都要有完整模型结构**
   - 每个 `subgraph_XX/` 目录都必须有 `model.py` 和 `weight_meta.py`
   - 每个 subgraph 的 `model.py` 都必须包含真正的 GraphModule 计算图

5. **主目录不需要有 model.py（如果有 subgraph 结构）**
   - 如果存在完整的 subgraph 结构，根目录可以没有 model.py
   - 如果没有 subgraph，根目录必须有有效的 model.py

### 残图定义

如果上述**任一条件不满足**，则该模型目录被判定为**残图**，需要修复。

---

## 残图类型分类

### Type A: 仅有目录结构，无核心文件
- **特征**：缺少 graph_hash.txt、graph_net.json、model.py 等关键文件
- **示例**：只有 extract.py，但缺少其他所有文件
- **修复方式**：完整重新抽取

### Type B: model.py 是包装函数
- **特征**：model.py 存在但只有 `get_model()` 等包装函数，无 GraphModule
- **判定**：❌ **算残图**
- **修复方式**：重新运行 extract.py 生成真正计算图

### Type C: subgraph 不完整
- **特征**：部分 subgraph 目录缺少 model.py 或 weight_meta.py
- **判定**：❌ **算残图**
- **修复方式**：修复或重新生成缺失的 subgraph 文件

### Type D: 缺少辅助文件
- **特征**：模型结构完整，但缺少 graph_hash.txt 或 extract.py
- **修复方式**：重新运行 extract.py 或手动生成缺失文件

---

## 检查脚本

```bash
#!/bin/bash
# 检查单个模型目录是否完整

check_model_complete() {
    local model_dir="$1"
    local issues=""

    # 1. 检查 graph_hash.txt
    if [ ! -f "$model_dir/graph_hash.txt" ]; then
        issues="${issues}缺少graph_hash.txt;"
    fi

    # 2. 检查 extract.py
    if [ ! -f "$model_dir/extract.py" ]; then
        issues="${issues}缺少extract.py;"
    fi

    # 3. 检查模型结构（model.py 必须有 GraphModule）
    has_valid_model=false

    # 检查是否有 subgraph 目录
    local has_subgraphs=false
    local all_subgraphs_complete=true

    for subgraph_dir in "$model_dir"/subgraph_*/; do
        if [ -d "$subgraph_dir" ]; then
            has_subgraphs=true
            if [ ! -f "$subgraph_dir/model.py" ] || [ ! -f "$subgraph_dir/weight_meta.py" ]; then
                all_subgraphs_complete=false
            else
                # 检查是否包含 GraphModule
                if ! grep -q "class GraphModule" "$subgraph_dir/model.py" 2>/dev/null; then
                    all_subgraphs_complete=false
                fi
            fi
        fi
    done

    if [ "$has_subgraphs" = true ]; then
        # 有 subgraph，检查是否全部完整
        if [ "$all_subgraphs_complete" = false ]; then
            issues="${issues}subgraph不完整;"
        fi
    else
        # 没有 subgraph，检查根目录
        if [ ! -f "$model_dir/model.py" ] || [ ! -f "$model_dir/weight_meta.py" ]; then
            issues="${issues}缺少model.py或weight_meta.py;"
        else
            # 检查是否包含 GraphModule
            if ! grep -q "class GraphModule" "$model_dir/model.py" 2>/dev/null; then
                issues="${issues}model.py无GraphModule;"
            fi
        fi
    fi

    if [ -z "$issues" ]; then
        echo "完整"
    else
        echo "残图:${issues%;}"
    fi
}
```

---

## 修复优先级

1. **P0 - 高优先级**: 完全空目录（需完整重建）
2. **P1 - 中高优先级**: model.py 是包装函数（需重新抽取）
3. **P2 - 中优先级**: subgraph 不完整（需修复特定 subgraph）
4. **P3 - 低优先级**: 仅缺少辅助文件（可快速修复）

---

## 修复流程

### 步骤 1: 扫描并分类
```bash
# 遍历所有模型目录，标记残图类型
for dir in */; do
    result=$(check_model_complete "$dir")
    echo "$dir: $result"
done
```

### 步骤 2: 批量修复
根据残图类型选择修复方式：
- **Type A/B/C**: 重新运行 extract.py 进行完整抽取
- **Type D**: 补充缺失的辅助文件

### 步骤 3: 验证
修复后重新运行检查脚本，确保符合完整模型标准。

---

## 各任务类型残图统计（参考）

| 任务类型 | 总目录 | 完整 | 残图 | 残图率 |
|---------|--------|------|------|--------|
| text-generation | 4,034 | 337 | 3,697 | 91.6% |
| text-classification | 1,815 | 266 | 1,549 | 85.3% |
| fill-mask | 1,162 | 626 | 536 | 46.1% |
| feature-extraction | 51 | 9 | 42 | 82.4% |
| translation | 125 | 37 | 88 | 70.4% |
| text2text-generation | 68 | 0 | 68 | 100.0% |
| token-classification | 4 | 0 | 4 | 100.0% |
| question-answering | 9 | 9 | 0 | 0.0% |
| automatic-speech-recognition | 12 | 6 | 6 | 50.0% |
| **总计** | **7,280** | **1,290** | **5,990** | **82.3%** |

---

## 注意事项

1. **必须检查文件内容**：不能只看 model.py 是否存在，必须确认包含 GraphModule
2. **包装函数不算**：只有 `get_model()` 的 model.py 算残图
3. **所有 subgraph 都要完整**：不能只检查 subgraph_0，要检查所有 subgraph_XX
4. **graph_hash.txt 是关键标志**：表明模型抽取成功

---

## 更新记录

- 2026-03-30: 初始版本，明确残图判断标准
- 2026-03-30: **重要更新** - 统一标准：
  - model.py 必须包含 GraphModule（不是包装函数）
  - 主目录不需要 model.py（如果有 subgraph 结构）
  - 所有 subgraph 都必须完整

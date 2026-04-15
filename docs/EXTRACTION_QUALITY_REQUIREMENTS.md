# 计算图抽取质量要求

> **来源**：下游维度泛化任务对抽图提出的质量要求
> **核心目标**：抽取的计算图必须能够正确执行 `run_model`（加载模型并执行 forward）

---

## 问题一：SymInt Example Value 未正确推断

### 现象
- `weight_meta.py` 里 SymInt 参数（如 `s0`, `L_self_head_dim`）的 `data` 值都是 placeholder `4`
- 导致 run_model 执行 forward 时维度不匹配

### 错误示例
```python
# weight_meta.py（错误）
class Program_weight_tensor_meta_s0:
    name = "s0"
    data = [4]  # ❌ 应该是 32，从输入 tensor shape 推断

class Program_weight_tensor_meta_L_self_head_dim:
    name = "L_self_head_dim"
    data = [4]  # ❌ 应该是 128，从 position_embeddings shape 推断
```

### 正确示例
```python
# weight_meta.py（正确）
class Program_weight_tensor_meta_s0:
    name = "s0"
    data = [32]  # ✅ 从 L_hidden_states_ shape [1, 32, 3584] 推断

class Program_weight_tensor_meta_L_self_head_dim:
    name = "L_self_head_dim"
    data = [128]  # ✅ 从 L_position_embeddings_0_ shape [1, 32, 128] 推断
```

### 推断规则

| SymInt 参数 | 推断来源 |
|------------|---------|
| `s0` (seq_len) | 从输入 tensor（非权重）的第二维推断 |
| `L_self_head_dim` | 从 `position_embeddings` tensor 的最后一维推断 |
| 其他 SymInt 参数 | 从 tensor shapes 中推断正确的值 |

### 推断逻辑
1. 遍历所有 tensor 的 shape
2. 找到与 SymInt 参数对应的维度值
3. 写入 `weight_meta.py` 的 `data` 字段

---

## 问题二：_assert_scalar API 兼容性问题

### 现象
- model.py 使用 `torch.ops.aten._assert_scalar.default()`
- 运行时报错：`Expected a value of type 'number' for argument 'self' but instead found type 'Tensor'`

### 问题代码
```python
le_2 = item <= s0  # 返回 Tensor/SymBool
_assert_scalar_default = torch.ops.aten._assert_scalar.default(le_2, "assertion message")
```

### 原因
`_assert_scalar` 期望 Python bool，但 `item <= s0` 返回 Tensor

### 解决方案
抽取时替换为兼容写法：
```python
le_2 = item <= s0
if isinstance(le_2, torch.Tensor):
    le_2 = le_2.item()
assert le_2, "assertion message"
```

或者确保 `_assert_scalar` 的参数是 Python bool 而非 Tensor。

---

## 质量检查清单

| 检查项 | 要求 | 验证方法 |
|--------|------|---------|
| ✅ SymInt Example Value | 必须从 tensor shapes 推断真实值 | 检查 weight_meta.py 中 data 字段 |
| ✅ API 兼容性 | 避免或正确处理 `_assert_scalar` | 检查 model.py 中的 torch.ops 调用 |
| ✅ 可运行性 | 必须能执行 forward | 运行 run_model 验证 |

---

## 验证命令

抽取完成后，运行以下命令验证：
```bash
python3.10 -m graph_net.torch.run_model --model-path <subgraph_path>
```

确保能够成功执行 forward。

---

## 质量统计（下游测试数据）

测试样本：10 个子图

| 问题类型 | 影响数量 | 修复方式 |
|---------|---------|---------|
| SymInt example_value 错误 | 3 个 | 从 tensor shapes 推断 |
| _assert_scalar 兼容性 | 1 个 | 替换为 assert |
| 模型太大超时 | 1 个 | 无法修复 |
| 无问题 | 5 个 | - |

**问题率**：50%（5/10），其中 40% 可通过改进抽取逻辑避免

---

## 改进建议

1. **增强 SymInt 推断逻辑**：抽取时分析 tensor shapes，正确推断 SymInt 值
2. **API 兼容性处理**：对 `_assert_scalar` 等 API 做兼容处理
3. **自动化验证**：抽取后自动运行 run_model 验证

---

## 相关文档

- SymInt 修复指南: [fix_symint_example_value.md](fix_symint_example_value.md)
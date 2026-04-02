# 嵌套目录问题记录

## 问题描述

在 HuggingFace 模型抽取过程中，出现了一种异常情况：**嵌套目录结构**。具体表现为：

```
worker2/model-name/
└── model-name/              # 同名子目录（嵌套层）
    ├── subgraph_0/
    ├── subgraph_1/
    └── ...
```

## 根本原因

1. **抽取中断**：模型抽取过程在 `torch_extract()` 阶段被中断
2. **fix_nested_directory() 未执行**：该函数只在抽取成功后运行
3. **嵌套结构残留**：外层目录和内层目录同时存在，导致重复层级

## 影响

1. **graph_hash.txt 检测失败**：脚本使用 `-maxdepth 2` 查找，嵌套结构使文件位于深度3+
2. **迁移脚本遗漏**：嵌套目录模型无法被正确识别和迁移
3. **空间浪费**：空目录占用文件系统资源

## 已发现案例（2026-04-02）

| 模型名称 | 嵌套内容 |
|---------|---------|
| adcg1355_hybrid-pegasus-ckpt-500 | 6个subgraph |
| adcg1355_hybrid-pegasus-ckpt-step1000 | 6个subgraph |
| adcg1355_hybrid-pegasus-ckpt-step1500 | 6个subgraph |
| adcg1355_hybrid-pegasus-ckpt-step2000 | 6个subgraph |
| adcg1355_hybrid-pegasus-ckpt-step2500 | 6个subgraph |
| adcg1355_hybrid-pegasus-ckpt-step3000 | 6个subgraph |
| adcg1355_hybrid-pegasus-ckpt-step3500 | 6个subgraph |
| adcg1355_hybrid-pegasus-ckpt-step4000 | 2个subgraph |
| aillm456_finetuned-falcon-rw-1b-instruct-openorca | model.py等 |

## 修复方案

### 1. 脚本修复（已完成）

在 `extract_from_model_list.py` 的抽取前检查中添加嵌套目录修复：

```python
# 修复嵌套目录问题（抽取中断留下的）
if fix_nested_directory(model_path, model_name):
    print(f"  修复嵌套目录: {model_id}")
    # 再次检查是否已成功（修复后）
    has_subgraph = any(d.startswith("subgraph_") ...)
    has_model_py = os.path.exists(...)
    if has_subgraph or has_model_py:
        print(f"  跳过（已修复）: {model_id}")
        generate_graph_hash(model_path)
        return True
```

### 2. 批量清理脚本（自动生成 extract.py）

**重要：修复嵌套目录后，必须为模型生成 extract.py！**

```python
import os
import shutil

def fix_nested_directory(model_path, model_name):
    """修复嵌套目录并自动生成 extract.py"""
    nested_path = os.path.join(model_path, model_name)
    if os.path.isdir(nested_path):
        print(f"\n修复: {model_name}")

        # 1. 移动内层文件到外层
        for item in os.listdir(nested_path):
            src = os.path.join(nested_path, item)
            dst = os.path.join(model_path, item)
            if not os.path.exists(dst):
                shutil.move(src, dst)
                print(f"  移动: {item}")

        # 2. 删除空内层目录
        os.rmdir(nested_path)

        # 3. 【重要】为修复后的模型生成 extract.py
        extract_path = os.path.join(model_path, 'extract.py')
        if not os.path.exists(extract_path):
            # 检查是否有 graph_hash.txt（确认是有效模型）
            has_hash = any('graph_hash.txt' in files
                          for _, _, files in os.walk(model_path))
            if has_hash:
                task = infer_task_type(model_name)  # 推断任务类型
                create_extract_py(model_path, model_name, task)
                print(f"  生成 extract.py (Task: {task})")

        print(f"  ✓ 完成")
        return True
    return False
```

### 3. 注意事项

- **必须生成 extract.py**：修复嵌套目录后，模型目录缺少 extract.py，这会导致：
  - 无法识别模型任务类型
  - 迁移脚本无法正确处理
  - 后续维护困难

- **自动推断任务类型**：脚本会根据模型名称自动推断任务类型（text2text-generation、text-generation 等）

- **验证修复结果**：修复后应检查：
  ```bash
  # 确认目录结构扁平化
  ls worker2/model-name/
  # 应该有 subgraph_*/ 或 model.py，而不是嵌套的 model-name/

  # 确认 extract.py 存在
  ls worker2/model-name/extract.py
  ```

## 预防措施

1. **抽取前检查**：在创建新目录前，先检查并修复可能存在的嵌套结构
2. **定期扫描**：每周运行嵌套目录检测脚本
3. **监控告警**：在监控报告中增加嵌套目录计数

## 修复记录

- **2026-04-02**: 修复 9 个嵌套目录案例
- **涉及模型**: 主要为 adcg1355 的 pegasus 检查点系列和 falcon 模型
- **修复结果**: 所有嵌套目录已扁平化，graph_hash.txt 可被正常检测

## 相关脚本

- `/root/extract_from_model_list.py` - 主抽取脚本（已添加修复逻辑）
- `/tmp/fix_nested_dirs.py` - 批量修复脚本（**修复后自动生成 extract.py**）

## 使用示例

```bash
# 运行修复脚本
python3 /tmp/fix_nested_dirs.py

# 输出示例：
# 修复: adcg1355_hybrid-pegasus-ckpt-step1000
#   移动: subgraph_0
#   移动: subgraph_1
#   ...
#   生成 extract.py (Task: text2text-generation)
#   ✓ 完成
#
# ============================================================
# 共修复 9 个嵌套目录
# ============================================================
```

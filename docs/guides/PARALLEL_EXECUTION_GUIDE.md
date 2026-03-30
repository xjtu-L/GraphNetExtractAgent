# 大规模并行任务执行指南

> 基于修复372个残图 + 并行抓取新模型的实战经验总结
>
> **最后更新**: 2026-03-30
> **维护者**: Worker1

---

## 一、并行任务架构设计

### 1.1 任务分类策略

```
┌─────────────────────────────────────────────────────────────┐
│                     任务调度层                                │
├─────────────────────────────────────────────────────────────┤
│  修复任务 (Repair)          │  新抓取任务 (Extraction)        │
│  ├── 按Task类型分批次         │  ├── 按批次文件分               │
│  ├── 每批次20-75个模型       │  ├── 每批次~30个模型            │
│  └── 共5个并行Subagent       │  └── 共6个并行Subagent          │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 推荐的并行度

| 任务类型 | 推荐并行数 | 说明 |
|---------|-----------|------|
| **残图修复** | 5-10 | 受CPU/内存限制，每进程占用1-2GB |
| **新模型抓取** | 5-10 | 网络IO密集型，可配置更多 |
| **总并行度** | ≤15 | 避免系统资源耗尽 |

---

## 二、关键优化点

### 2.1 代理配置（重要）

**必须设置的环境变量**:
```bash
# 网络代理
export http_proxy=http://agent.baidu.com:8891
export https_proxy=http://agent.baidu.com:8891
export HTTP_PROXY=http://agent.baidu.com:8891
export HTTPS_PROXY=http://agent.baidu.com:8891

# HuggingFace镜像
export HF_ENDPOINT=https://hf-mirror.com

# 离线模式（修复时用）
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# 禁用CUDA（节省显存）
export CUDA_VISIBLE_DEVICES=""
```

### 2.2 批次划分策略

**残图修复批次划分**:
```python
# 将372个残图分成5份
batch_size = len(residuals) // 5 + (1 if len(residuals) % 5 else 0)
for i in range(5):
    batch = residuals[i*batch_size:(i+1)*batch_size]
    # 每批75个模型左右
```

**新模型抓取批次**:
```python
# 从模型列表文件读取
with open(f'/tmp/new_extract_batch_{i}.txt', 'r') as f:
    models = [line.strip() for line in f if line.strip()]
```

### 2.3 日志隔离

**每个Subagent独立日志**:
```bash
# 修复任务
/tmp/repair_batch_{0-4}.log

# 新抓取任务
/tmp/new_extract_subagent_{0-5}.log
```

---

## 三、遇到的问题及解决方案

### 3.1 问题：脚本创建时的Shell转义问题

**现象**:
```bash
cat > script.py << 'EOF'
import os  # 报错: import: command not found
EOF
```

**原因**: Bash heredoc中的特殊字符被解释

**解决方案**:
```bash
# 方法1: 使用python -c直接写
python3 -c "
script = '''...'''
with open('script.py', 'w') as f:
    f.write(script)
"

# 方法2: 使用base64编码
echo "cHJpbnQoJ2hlbGxvJyk=" | base64 -d > script.py

# 方法3: 使用printf
printf '%s\n' 'import os' 'import sys' > script.py
```

### 3.2 问题：Model.py是包装函数而非GraphModule

**现象**:
```python
# model.py 内容
from transformers import AutoConfig, AutoModel
def get_model():
    config = AutoConfig.from_pretrained("...")
    model = AutoModel.from_config(config)
    return model
```

**原因**: 早期提取脚本只生成包装代码，未触发实际提取

**解决方案**:
```python
# 重新提取，使用正确的extract调用顺序
from graph_net.torch.extractor import extract

wrapped = extract(name=model_name, dynamic=True)(model)
with torch.no_grad():
    wrapped(**inputs)  # 触发实际提取
```

### 3.3 问题：模型文件分散在subgraph目录中

**现象**:
- 根目录只有 `graph_hash.txt`, `extract.py`
- 实际模型文件在 `subgraph_0/`, `subgraph_1/` 中
- 误判为残图

**解决方案**:
```python
def check_model_complete(model_path):
    """检查模型是否完整（支持分散存储）"""
    # 检查根目录model.py
    model_file = os.path.join(model_path, 'model.py')
    # 检查subgraph目录
    subgraph_model = os.path.join(model_path, 'subgraph_0', 'model.py')

    for mf in [model_file, subgraph_model]:
        if os.path.exists(mf):
            with open(mf, 'r') as f:
                content = f.read()
                if 'class GraphModule' in content and 'def forward' in content:
                    return True  # 模型完整
    return False  # 真正的残图
```

### 3.4 问题：补全缺失的根目录文件

**现象**: 模型完整但缺少根目录的 `config.json`, `graph_net.json`

**解决方案**:
```python
def complete_model_files(model_path, model_name):
    """补全模型缺失文件"""
    # 1. 从subgraph复制config.json
    for sg in ['subgraph_0', 'subgraph_1', 'subgraph_2']:
        config_src = os.path.join(model_path, sg, 'config.json')
        if os.path.exists(config_src):
            shutil.copy2(config_src, os.path.join(model_path, 'config.json'))
            break

    # 2. 计算graph_hash.txt
    model_py_path = os.path.join(model_path, 'subgraph_0', 'model.py')
    if os.path.exists(model_py_path):
        with open(model_py_path, 'rb') as f:
            content = f.read()
        hash_value = hashlib.sha256(content).hexdigest()[:16]
        with open(os.path.join(model_path, 'graph_hash.txt'), 'w') as f:
            f.write(hash_value)

    # 3. 创建extract.py
    create_extract_script(model_path, model_name)
```

### 3.5 问题：AutoConfig.from_dict不存在

**现象**:
```python
AttributeError: type object 'AutoConfig' has no attribute 'from_dict'
```

**解决方案**:
```python
# 错误：config = AutoConfig.from_dict(config_dict)
# 正确：使用具体Config类
if task_type == 'fill-mask':
    config = BertConfig(**config_dict)
    model = BertForMaskedLM(config)
elif task_type == 'text-generation':
    # 对于CausalLM，使用trust_remote_code
    config = AutoConfig.from_pretrained(
        config_dict['_name_or_path'] if '_name_or_path' in config_dict else 'gpt2',
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
```

### 3.6 问题：残图统计标准不一致

**现象**: Worker1和Worker2统计的残图数量不一致

**根本原因**: 对"完整模型"的定义理解不同

**统一标准**:
```python
# 残图定义：不包含真正的GraphModule计算图
def is_residual(model_path):
    # 标准1: 根目录model.py是否包含GraphModule
    root_model = os.path.join(model_path, 'model.py')
    if os.path.exists(root_model):
        with open(root_model, 'r') as f:
            content = f.read()
            if 'class GraphModule' in content and 'def forward' in content:
                return False  # 完整模型

    # 标准2: subgraph_0/model.py是否包含GraphModule
    subgraph_model = os.path.join(model_path, 'subgraph_0', 'model.py')
    if os.path.exists(subgraph_model):
        with open(subgraph_model, 'r') as f:
            content = f.read()
            if 'class GraphModule' in content and 'def forward' in content:
                return False  # 完整模型（文件分散存储）

    return True  # 真正的残图
```

---

## 四、实战经验总结

### 4.1 成功模式

1. **修复+抓取并行**: 修复残图（离线）和新模型抓取（在线）可并行
2. **按Task类型分批次**: 同一Task类型的模型共享修复逻辑
3. **使用V4脚本**: Worker2的V4修复脚本成功率最高
4. **及时补全文件**: 发现文件缺失立即补全，避免后续误判

### 4.2 失败教训

1. **不要过度并行**: CPU密集型任务超过15个会导致系统卡顿
2. **日志要分离**: 多进程写入同一日志会导致混乱
3. **备份要保留**: 修复失败时能快速恢复
4. **检查要准确**: 避免误判完整模型为残图

### 4.3 性能数据

| 任务 | 数量 | 并行度 | 预计时间 |
|-----|------|--------|---------|
| 残图修复 | 372个 | 5 | 2-3小时 |
| 新模型抓取 | 149,720个 | 6 | 持续进行 |
| 文件补全 | 461个 | 1 | 5分钟 |

---

## 五、快速启动命令

### 5.1 启动修复任务

```bash
# 生成批次文件
python3 -c "
import os, json
def check_model_complete(path):
    for mf in [os.path.join(path, 'model.py'), os.path.join(path, 'subgraph_0', 'model.py')]:
        if os.path.exists(mf):
            with open(mf) as f:
                if 'class GraphModule' in f.read():
                    return True
    return False

base = '/work/graphnet_workspace/huggingface'
tasks = ['fill-mask', 'text-classification', 'text-generation']
residuals = []
for task in tasks:
    for d in os.listdir(os.path.join(base, task)):
        p = os.path.join(base, task, d)
        if os.path.isdir(p) and not check_model_complete(p):
            residuals.append({'task': task, 'model': d, 'path': p})

for i in range(5):
    batch = residuals[i*75:(i+1)*75]
    with open(f'/tmp/residual_batch_{i}.json', 'w') as f:
        json.dump(batch, f)
"

# 启动5个修复subagent
for i in 0 1 2 3 4; do
    nohup python3 /tmp/standard_repair_v3.py /tmp/residual_batch_${i}.json $i \
        > /tmp/repair_batch_${i}.log 2>&1 &
done
```

### 5.2 启动新模型抓取

```bash
# 设置代理
export http_proxy=http://agent.baidu.com:8891
export https_proxy=http://agent.baidu.com:8891
export HF_ENDPOINT=https://hf-mirror.com

# 启动6个抓取subagent
for i in 0 1 2 3 4 5; do
    nohup python3 /tmp/new_model_extract_batch.py \
        /tmp/new_extract_batch_${i}.txt $i \
        > /tmp/new_extract_subagent_${i}.log 2>&1 &
done
```

### 5.3 监控进度

```bash
# 查看运行中进程
ps aux | grep -E "repair|extract" | grep -v grep | wc -l

# 查看修复进度
tail -f /tmp/repair_batch_*.log

# 查看抓取进度
tail -f /tmp/new_extract_subagent_*.log

# 统计成功数量
grep -c "\[成功\]" /tmp/repair_batch_*.log
```

---

## 六、参考文档

- [残图修复标准流程](./repair/STANDARD_REPAIR_WORKFLOW.md)
- [HuggingFace提取指南](./extract/HUGGINGFACE_EXTRACTION_GUIDE.md)
- [多Worker协作流程](./extract/multi_worker_collaboration.md)
- [异常案例汇总](./ANOMALY_CASES.md)

---

**维护历史**:
- 2026-03-30: 创建文档，记录并行执行372残图修复+新模型抓取的经验

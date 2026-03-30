# GraphNet 残图修复 - 快速参考

## 祖训级别（必须遵守）

1. **使用 `from_config()` 随机初始化，不下载权重！**
2. **每个模型必须有 `extract.py` 复现脚本**
3. **先修复残图，再抓新模型**

---

## 一键命令

```bash
# 统计残图
cd /work/graphnet_workspace/huggingface && python3 -c "
import os
tasks = ['text-classification', 'text-generation', 'fill-mask']
for t in tasks:
    d = f'/work/graphnet_workspace/huggingface/{t}'
    if os.path.exists(d):
        c = sum(1 for m in os.listdir(d) if os.path.isdir(os.path.join(d,m))
                and os.path.exists(os.path.join(d,m,'graph_hash.txt'))
                and not (os.path.exists(os.path.join(d,m,'model.py'))
                         and os.path.exists(os.path.join(d,m,'weight_meta.py'))))
        print(f'{t}: {c} 残图')
"

# 启动修复
nohup python3 repair_from_json.py --task fill-mask --agent-id fix-01 >logs/fm.log 2>&1 &

# 查看进度
tail -f logs/fm.log

# 停止抓取
pkill -9 -f extract_hf_full.py
```

---

## 核心代码片段

```python
# 正确: from_config随机初始化
config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
model = AutoModel.from_config(config).eval()

# 错误: from_pretrained下载权重
model = AutoModel.from_pretrained(model_id)  # ❌ 禁止!
```

---

## 残图定义

```
残图 = 有 graph_hash.txt 但 (缺少 model.py 或 缺少 weight_meta.py)

完整模型 = 有 graph_hash.txt + 有 model.py + 有 weight_meta.py
         (在根目录或任意 subgraph_* 子目录中)
```

---

## 常见问题

| 问题 | 解决 |
|------|------|
| tokenizer没有pad_token | `tokenizer.pad_token = tokenizer.eos_token` |
| 模型太大无法下载 | 使用 `from_config()` 不下载权重 |
| 修复后没有weight_meta.py | 检查GraphNet抽取是否成功 |
| subgraph模型统计错误 | 检查所有subgraph_*子目录 |

---

## 文件位置

- 详细文档: `/root/GraphNetExtractAgent/docs/incomplete_model_repair_guide.md`
- 修复脚本: `/root/graphnet_workspace/huggingface/repair_from_json.py`
- 残图列表: `/work/graphnet_workspace/incomplete_for_worker2.txt`

---

*最后更新: 2026-03-29*

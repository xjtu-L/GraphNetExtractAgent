#!/usr/bin/env python3
"""
HuggingFace 单类别抽取脚本

用法:
    python extract_by_task.py --task text-generation --limit 100 --agent-id agent-tg
    python extract_by_task.py --task sentence-similarity --limit 50 --agent-id agent-ss

特性:
    - 边发现边抓取
    - 自动去重（samples + workspace）
    - 生成 extract.py 复现脚本
    - 结果存档
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

# 设置代理和环境变量
os.environ.setdefault("http_proxy", "http://agent.baidu.com:8891")
os.environ.setdefault("https_proxy", "http://agent.baidu.com:8891")
os.environ.setdefault("GRAPH_NET_EXTRACT_WORKSPACE", "/ssd1/liangtai-work/graphnet_workspace/huggingface")

SAMPLES_DIR = "/root/GraphNet/samples/transformers-auto-model"


def model_id_to_sample_name(model_id: str) -> str:
    """org/model -> org_model"""
    return model_id.replace("/", "_")


class TaskExtractor:
    """单类别抽取器"""

    def __init__(self, task_type: str, agent_id: str, workspace: str):
        self.task_type = task_type
        self.agent_id = agent_id
        self.workspace = workspace

        # 统计
        self.stats = {
            "task": task_type,
            "agent_id": agent_id,
            "start_time": datetime.now().isoformat(),
            "discovered": 0,
            "extracted": 0,
            "duplicates": 0,
            "failed": 0,
            "success_list": [],
            "failed_list": [],
            "duplicate_list": []
        }

        # 去重集合
        self.extracted_set = self._load_extracted_set()
        print(f"[{agent_id}] 已存在模型: {len(self.extracted_set)}")

    def _load_extracted_set(self) -> set:
        """加载已抽取模型集合"""
        extracted = set()

        # samples 目录
        if os.path.isdir(SAMPLES_DIR):
            for name in os.listdir(SAMPLES_DIR):
                path = os.path.join(SAMPLES_DIR, name)
                if os.path.isdir(path):
                    extracted.add(name)

        # workspace 目录
        if os.path.isdir(self.workspace):
            for name in os.listdir(self.workspace):
                path = os.path.join(self.workspace, name)
                if os.path.isdir(path):
                    extracted.add(name)

        return extracted

    def is_duplicate(self, model_id: str) -> bool:
        """检查重复"""
        sample_name = model_id_to_sample_name(model_id)
        return sample_name in self.extracted_set

    def discover_models(self, limit: int = 100) -> list:
        """发现模型"""
        from huggingface_hub import HfApi
        api = HfApi()

        models = list(api.list_models(
            pipeline_tag=self.task_type,
            sort="downloads",
            limit=limit
        ))
        return [m.id for m in models]

    def extract_one(self, model_id: str) -> tuple:
        """抽取单个模型，返回 (success, message)"""
        try:
            from graph_net.agent import GraphNetAgent
            agent = GraphNetAgent(workspace=self.workspace)
            success = agent.extract_sample(model_id)

            if success:
                # 标记已抽取
                self.extracted_set.add(model_id_to_sample_name(model_id))
                return True, "OK"
            else:
                return False, "Agent returned False"

        except Exception as e:
            return False, str(e)[:200]

    def run(self, max_models: int = 100, discover_limit: int = 1000):
        """运行抽取"""
        print(f"\n[{self.agent_id}] ========== 开始抽取 ==========")
        print(f"  Task 类型: {self.task_type}")
        print(f"  目标数量: {max_models}")
        print(f"  发现上限: {discover_limit}")
        print(f"  Workspace: {self.workspace}")
        print("=" * 50)

        # 发现模型
        print(f"\n[{self.agent_id}] 发现模型中...")
        all_models = self.discover_models(limit=discover_limit)
        print(f"[{self.agent_id}] 发现 {len(all_models)} 个模型")

        # 过滤并抽取
        for model_id in all_models:
            if self.stats["extracted"] >= max_models:
                print(f"\n[{self.agent_id}] 已达目标数量 {max_models}")
                break

            self.stats["discovered"] += 1

            # 去重检查
            if self.is_duplicate(model_id):
                self.stats["duplicates"] += 1
                self.stats["duplicate_list"].append(model_id)
                continue

            # 抽取
            print(f"[{self.agent_id}] [{self.stats['extracted']+1}/{max_models}] {model_id}...", end=" ", flush=True)
            success, msg = self.extract_one(model_id)

            if success:
                self.stats["extracted"] += 1
                self.stats["success_list"].append(model_id)
                print(f"✓")
            else:
                self.stats["failed"] += 1
                self.stats["failed_list"].append({"model": model_id, "error": msg})
                print(f"✗ {msg[:50]}")

        # 完成
        self.stats["end_time"] = datetime.now().isoformat()
        print(f"\n[{self.agent_id}] ========== 完成 ==========")
        print(f"  发现: {self.stats['discovered']}")
        print(f"  抽取: {self.stats['extracted']}")
        print(f"  重复: {self.stats['duplicates']}")
        print(f"  失败: {self.stats['failed']}")
        print("=" * 50)

        return self.stats

    def save_results(self):
        """保存结果"""
        result_dir = os.path.join(self.workspace, "extract_results")
        os.makedirs(result_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = os.path.join(result_dir, f"{self.agent_id}_{timestamp}.json")

        with open(result_file, "w") as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)

        print(f"\n结果已保存: {result_file}")
        return result_file


def main():
    parser = argparse.ArgumentParser(description="HuggingFace 单类别抽取")
    parser.add_argument("--task", required=True, help="Pipeline tag (如 text-generation)")
    parser.add_argument("--limit", type=int, default=100, help="最大抽取数量")
    parser.add_argument("--discover-limit", type=int, default=1000, help="最大发现数量")
    parser.add_argument("--agent-id", default="agent-default", help="Agent ID")
    parser.add_argument("--workspace", default="/ssd1/liangtai-work/graphnet_workspace/huggingface", help="Workspace")

    args = parser.parse_args()

    extractor = TaskExtractor(
        task_type=args.task,
        agent_id=args.agent_id,
        workspace=args.workspace
    )

    stats = extractor.run(max_models=args.limit, discover_limit=args.discover_limit)
    extractor.save_results()

    # 返回码
    sys.exit(0 if stats["extracted"] > 0 else 1)


if __name__ == "__main__":
    main()

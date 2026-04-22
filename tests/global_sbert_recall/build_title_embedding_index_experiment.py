"""Archived CLI for the global title embedding recall experiment."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from embedding_recall_experiment import build_title_embedding_index


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="构建新闻标题 embedding 召回索引")
    parser.add_argument("--limit", type=int, default=None, help="仅处理前 N 条新闻，用于快速调试")
    parser.add_argument("--batch-size", type=int, default=64, help="embedding 批处理大小")
    parser.add_argument("--index-path", default=None, help="自定义 .npz 索引输出路径")
    parser.add_argument("--meta-path", default=None, help="自定义 metadata JSON 输出路径")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    kwargs = {
        "limit": args.limit,
        "batch_size": args.batch_size,
    }
    if args.index_path:
        kwargs["index_path"] = Path(args.index_path)
    if args.meta_path:
        kwargs["meta_path"] = Path(args.meta_path)

    meta = build_title_embedding_index(**kwargs)
    print("title embedding index built")
    print(f"count: {meta['count']}")
    print(f"dimension: {meta['dimension']}")
    print(f"model: {meta['model']}")
    print(f"index_path: {meta['index_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

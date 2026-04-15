#!/usr/bin/env python3
"""Count users/items/interactions after k-core filtering for Amazon dataset files.

Usage examples:
  python MLLM-MSR/data/amazon/count_filtered_stats.py
  python MLLM-MSR/data/amazon/count_filtered_stats.py --dataset Video_Games --min-user 6 --min-item 5
  python MLLM-MSR/data/amazon/count_filtered_stats.py --dataset All_Beauty --input-dir MLLM-MSR/data/amazon/gz
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

UID_COL = "user_id"
IID_COL = "parent_asin"
TS_COL = "timestamp"


def resolve_input_file(input_dir: Path, dataset: str) -> Path:
    candidates = [
        input_dir / f"{dataset}.jsonl.gz",
        input_dir / f"{dataset}.jsonl",
        input_dir / "gz" / f"{dataset}.jsonl.gz",
        input_dir / "input" / f"{dataset}.jsonl",
    ]

    for path in candidates:
        if path.exists():
            return path

    tried = "\n- ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Input file not found. Tried:\n- {tried}")


def stream_lines(file_path: Path):
    opener = gzip.open if file_path.suffix == ".gz" else open
    with opener(file_path, mode="rt", encoding="utf8") as f:
        for line in f:
            if line.strip():
                yield line


def read_interactions(file_path: Path) -> list[tuple[str, str, int]]:
    """Read and clean interactions.

    Keep only rows with user_id/parent_asin/timestamp present,
    and deduplicate by (user_id, parent_asin, timestamp).
    """
    dedup = set()
    interactions: list[tuple[str, str, int]] = []

    for line in stream_lines(file_path):
        payload = json.loads(line)
        user = payload.get(UID_COL)
        item = payload.get(IID_COL)
        ts = payload.get(TS_COL)

        if user is None or item is None or ts is None:
            continue

        key = (str(user), str(item), int(ts))
        if key in dedup:
            continue
        dedup.add(key)
        interactions.append(key)

    return interactions


def filter_by_k_core(
    interactions: list[tuple[str, str, int]], min_user: int, min_item: int
) -> list[tuple[str, str, int]]:
    filtered = interactions
    iteration = 0

    while True:
        user_cnt = Counter(u for u, _, _ in filtered)
        item_cnt = Counter(i for _, i, _ in filtered)

        ban_users = {u for u, c in user_cnt.items() if c < min_user}
        ban_items = {i for i, c in item_cnt.items() if c < min_item}

        if not ban_users and not ban_items:
            break

        before = len(filtered)
        filtered = [
            (u, i, ts)
            for (u, i, ts) in filtered
            if (u not in ban_users and i not in ban_items)
        ]
        after = len(filtered)
        iteration += 1
        print(
            f"[iter={iteration}] dropped={before - after}, "
            f"ban_users={len(ban_users)}, ban_items={len(ban_items)}, left={after}"
        )

    return filtered


def summarize(interactions: list[tuple[str, str, int]]) -> tuple[int, int, int]:
    users = {u for u, _, _ in interactions}
    items = {i for _, i, _ in interactions}
    return len(users), len(items), len(interactions)


def main() -> None:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Count users/items/interactions after k-core filtering."
    )
    parser.add_argument("--dataset", default="Video_Games", help="Dataset name (e.g. Video_Games)")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=script_dir,
        help="Dataset root dir (auto tries <dir>, <dir>/gz, <dir>/input)",
    )
    parser.add_argument("--min-user", type=int, default=6, help="k-core min interactions per user")
    parser.add_argument("--min-item", type=int, default=5, help="k-core min interactions per item")
    args = parser.parse_args()

    input_file = resolve_input_file(args.input_dir, args.dataset)
    print(f"Using input file: {input_file}")

    interactions = read_interactions(input_file)
    _, _, cleaned_interactions = summarize(interactions)
    print(f"Rows after basic clean (drop missing + dedup): {cleaned_interactions}")

    filtered = filter_by_k_core(interactions, args.min_user, args.min_item)
    user_count, item_count, interaction_count = summarize(filtered)

    print("\n=== Filtered full-data stats ===")
    print(f"dataset: {args.dataset}")
    print(f"k-core: (user>={args.min_user}, item>={args.min_item})")
    print(f"users: {user_count}")
    print(f"items: {item_count}")
    print(f"interactions: {interaction_count}")


if __name__ == "__main__":
    main()

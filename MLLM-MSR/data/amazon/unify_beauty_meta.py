"""Prepare filtered Beauty meta jsonl for unified Agent1/2 profiling.

Steps:
1) Read `metadata.csv` and collect valid ids (asin/item_id/id) + optional price.
2) Read raw `meta_Beauty.json` (json-lines or python-dict-lines).
3) Keep only rows whose `asin` appears in metadata.csv.
4) Fill missing `price` from metadata.csv when available.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple


ID_KEYS = ("id", "item_id", "asin")
PRICE_KEYS = ("price", "Price")


def _read_metadata_csv(path: str | Path) -> Tuple[set[str], Dict[str, Any]]:
    ids: set[str] = set()
    price_by_id: Dict[str, Any] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item_id = ""
            for k in ID_KEYS:
                val = str(row.get(k, "") or "").strip()
                if val:
                    item_id = val
                    break
            if not item_id:
                continue
            ids.add(item_id)
            for pk in PRICE_KEYS:
                price = row.get(pk)
                if price not in (None, ""):
                    price_by_id[item_id] = price
                    break
    return ids, price_by_id


def _iter_meta_rows(path: str | Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for ln, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                obj = ast.literal_eval(raw)
            if not isinstance(obj, dict):
                print(f"[WARN] line={ln} is not dict, skipped")
                continue
            yield obj


def unify_meta(metadata_csv: str | Path, raw_meta_json: str | Path, output_jsonl: str | Path) -> Dict[str, int]:
    valid_ids, price_by_id = _read_metadata_csv(metadata_csv)
    out_path = Path(output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = kept = filled_price = 0
    with out_path.open("w", encoding="utf-8") as fw:
        for row in _iter_meta_rows(raw_meta_json):
            total += 1
            asin = str(row.get("asin", "") or "").strip()
            if asin not in valid_ids:
                continue
            if (row.get("price") in (None, "")) and asin in price_by_id:
                row["price"] = price_by_id[asin]
                filled_price += 1
            fw.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1

    stats = {
        "metadata_ids": len(valid_ids),
        "raw_meta_rows": total,
        "kept_rows": kept,
        "filled_price_rows": filled_price,
    }
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return stats


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Filter raw Beauty meta by metadata.csv and fill missing price")
    p.add_argument("--metadata-csv", default="data/amazon_beauty/metadata.csv")
    p.add_argument("--raw-meta-json", default="data/amazon_beauty/meta_Beauty.json")
    p.add_argument("--output-jsonl", default="data/amazon_beauty/meta_Beauty.filtered.jsonl")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    unify_meta(args.metadata_csv, args.raw_meta_json, args.output_jsonl)

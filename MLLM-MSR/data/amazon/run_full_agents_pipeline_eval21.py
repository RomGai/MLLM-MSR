"""Build a 21-item (1 positive + 20 negatives) evaluation catalog before Agent 1.

Goal:
- Align with the fixed-size evaluation grouping used by `test/microlens/test_with_llava.py`
  where metrics are computed after `reshape(-1, 21)`.
- Treat the 21 items as the *full* item catalog seen by Agent 1.
- Keep Agent 2 user-history modeling unchanged (use full history files as-is).

Workflow:
1) Select one evaluation unit from `*_user_items_negs_test.csv` (or compatible file):
   - one user
   - one positive item from the user's positive list
2) Build 20 negatives:
   - start from row-provided negatives
   - top up by deterministic random sampling from global items, excluding the chosen
     positive and (by default) that user's seen items.
3) Materialize a filtered `item_desc.tsv` containing exactly 21 items.
4) Invoke `run_full_agents_pipeline.py` using this filtered item catalog, while
   retaining original history inputs for Agent 2.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence, Set, Tuple

from run_full_agents_pipeline import build_argparser as build_full_argparser
from run_full_agents_pipeline import run_pipeline


@dataclass
class EvalUnit:
    user_id: str
    pos_items: List[str]
    neg_items: List[str]


def _read_user_items_negs(path: str | Path) -> List[EvalUnit]:
    rows: List[EvalUnit] = []
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        first_line = f.readline()
        f.seek(0)
        has_header = first_line.lower().startswith("user_id\t")
        if has_header:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                user_id = str(row.get("user_id", "")).strip()
                pos = [x.strip() for x in str(row.get("pos", "")).split(",") if x.strip()]
                neg = [x.strip() for x in str(row.get("neg", "")).split(",") if x.strip()]
                if user_id and pos:
                    rows.append(EvalUnit(user_id=user_id, pos_items=pos, neg_items=neg))
        else:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if len(row) < 3:
                    continue
                user_id = str(row[0]).strip()
                pos = [x.strip() for x in str(row[1]).split(",") if x.strip()]
                neg = [x.strip() for x in str(row[2]).split(",") if x.strip()]
                if user_id and pos:
                    rows.append(EvalUnit(user_id=user_id, pos_items=pos, neg_items=neg))
    return rows


def _read_item_desc_rows(path: str | Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            item_id = str(row.get("item_id", "")).strip()
            if not item_id:
                continue
            rows.append(
                {
                    "item_id": item_id,
                    "image": str(row.get("image", "")),
                    "summary": str(row.get("summary", "")),
                }
            )
    return rows


def _user_seen_items(user_pairs_tsv: str | Path, user_id: str) -> Set[str]:
    seen: Set[str] = set()
    with Path(user_pairs_tsv).open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if str(row.get("user_id", "")).strip() != str(user_id):
                continue
            item_id = str(row.get("item_id", "")).strip()
            if item_id:
                seen.add(item_id)
    return seen


def _build_eval21_catalog(
    all_item_ids: Sequence[str],
    unit: EvalUnit,
    chosen_positive: str,
    user_seen_item_ids: Set[str],
    seed: int,
    exclude_seen_for_negatives: bool,
) -> List[str]:
    rng = random.Random(seed)

    negatives: List[str] = []
    used = {chosen_positive}

    for item_id in unit.neg_items:
        if item_id in used:
            continue
        if exclude_seen_for_negatives and item_id in user_seen_item_ids:
            continue
        negatives.append(item_id)
        used.add(item_id)
        if len(negatives) >= 20:
            break

    candidate_pool = []
    for item_id in all_item_ids:
        if item_id in used:
            continue
        if exclude_seen_for_negatives and item_id in user_seen_item_ids:
            continue
        candidate_pool.append(item_id)

    rng.shuffle(candidate_pool)
    for item_id in candidate_pool:
        negatives.append(item_id)
        used.add(item_id)
        if len(negatives) >= 20:
            break

    if len(negatives) < 20:
        raise ValueError(
            f"Cannot build 20 negatives for user={unit.user_id}. "
            f"Only got {len(negatives)} negatives after filtering."
        )

    group = [chosen_positive] + negatives[:20]
    rng.shuffle(group)
    return group


def _write_filtered_item_desc(rows: Sequence[Dict[str, str]], keep_item_ids: Set[str], out_path: str | Path) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["item_id", "image", "summary"], delimiter="\t")
        writer.writeheader()
        for row in rows:
            if row["item_id"] in keep_item_ids:
                writer.writerow(row)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prebuild 21-item (1+20) catalog then run full Amazon agents pipeline"
    )

    parser.add_argument("--item-desc-tsv", default="./processed/Video_Games_item_desc.tsv")
    parser.add_argument("--user-pairs-tsv", default="./processed/Video_Games_u_i_pairs.tsv")
    parser.add_argument(
        "--eval-user-items-negs-tsv",
        default="./processed/Video_Games_user_items_negs_test.csv",
        help="Evaluation split file containing user_id, pos, neg",
    )
    parser.add_argument(
        "--agent2-user-items-negs-tsv",
        default="./processed/Video_Games_user_items_negs.tsv",
        help="Agent 2 input (kept full/normal, not restricted to 21-catalog)",
    )

    parser.add_argument("--target-user-id", default="", help="Optional fixed user for one eval unit")
    parser.add_argument(
        "--target-user-row-index",
        type=int,
        default=0,
        help="If target-user-id has multiple rows, pick this row index",
    )
    parser.add_argument(
        "--positive-index",
        type=int,
        default=0,
        help="Pick which positive item from the row's pos list",
    )
    parser.add_argument(
        "--exclude-seen-for-negatives",
        action="store_true",
        help="Sample extra negatives excluding this user's seen items",
    )
    parser.add_argument("--seed", type=int, default=2026)

    parser.add_argument(
        "--prepared-item-desc-out",
        default="./processed/eval21_item_desc.tsv",
        help="Filtered 21-item catalog tsv for Agent 1",
    )
    parser.add_argument(
        "--eval-unit-meta-out",
        default="./processed/eval21_unit_meta.json",
        help="Metadata of selected eval unit and 21-item catalog",
    )

    parser.add_argument("--bundle-output", required=True)
    parser.add_argument("--prepare-only", action="store_true", help="Only prepare 21-item catalog and metadata, do not run full pipeline")

    full_defaults = vars(build_full_argparser().parse_args(["--bundle-output", "dummy.zip"]))
    forward_keys = [
        "global_db",
        "history_db",
        "profiler_run_out_dir",
        "intent_output_dir",
        "dynamic_output_dir",
        "vl_model",
        "text_model",
        "category_hint",
        "query",
        "min_candidate_items",
        "max_candidate_items",
        "max_history_rows",
        "top_n",
    ]
    for key in forward_keys:
        parser.add_argument(f"--{key.replace('_', '-')}", default=full_defaults[key])

    return parser


def main(args: argparse.Namespace) -> None:
    item_rows = _read_item_desc_rows(args.item_desc_tsv)
    item_map = {r["item_id"]: r for r in item_rows}
    all_item_ids = list(item_map.keys())
    if len(all_item_ids) < 21:
        raise ValueError("Item catalog size is < 21, cannot construct 1+20 evaluation unit.")

    units = _read_user_items_negs(args.eval_user_items_negs_tsv)
    if not units:
        raise ValueError("No valid rows found in eval-user-items-negs-tsv.")

    if args.target_user_id:
        user_units = [u for u in units if u.user_id == str(args.target_user_id)]
        if not user_units:
            raise ValueError(f"target-user-id={args.target_user_id} not found in eval file.")
        idx = max(0, int(args.target_user_row_index))
        if idx >= len(user_units):
            raise IndexError(f"target-user-row-index={idx} out of range: {len(user_units)} rows")
        unit = user_units[idx]
    else:
        unit = units[0]

    if args.positive_index < 0 or args.positive_index >= len(unit.pos_items):
        raise IndexError(
            f"positive-index={args.positive_index} out of range for pos list size={len(unit.pos_items)}"
        )
    chosen_positive = unit.pos_items[args.positive_index]
    if chosen_positive not in item_map:
        raise KeyError(f"Selected positive item_id={chosen_positive} not found in item-desc-tsv")

    seen_items = _user_seen_items(args.user_pairs_tsv, unit.user_id)
    eval21_items = _build_eval21_catalog(
        all_item_ids=all_item_ids,
        unit=unit,
        chosen_positive=chosen_positive,
        user_seen_item_ids=seen_items,
        seed=int(args.seed),
        exclude_seen_for_negatives=bool(args.exclude_seen_for_negatives),
    )

    _write_filtered_item_desc(
        rows=item_rows,
        keep_item_ids=set(eval21_items),
        out_path=args.prepared_item_desc_out,
    )

    meta = {
        "user_id": unit.user_id,
        "selected_positive_item": chosen_positive,
        "selected_group_size": len(eval21_items),
        "eval21_items": eval21_items,
        "from_eval_file": str(args.eval_user_items_negs_tsv),
        "exclude_seen_for_negatives": bool(args.exclude_seen_for_negatives),
        "seed": int(args.seed),
    }
    meta_out = Path(args.eval_unit_meta_out)
    meta_out.parent.mkdir(parents=True, exist_ok=True)
    meta_out.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.prepare_only:
        print(json.dumps({
            "prepared_item_desc_out": str(args.prepared_item_desc_out),
            "eval_unit_meta_out": str(args.eval_unit_meta_out),
            "selected_group_size": len(eval21_items),
        }, ensure_ascii=False, indent=2))
        return

    full_args = SimpleNamespace(
        item_desc_tsv=str(args.prepared_item_desc_out),
        user_pairs_tsv=str(args.user_pairs_tsv),
        user_items_negs_tsv=str(args.agent2_user_items_negs_tsv),
        global_db=str(args.global_db),
        history_db=str(args.history_db),
        profiler_run_out_dir=str(args.profiler_run_out_dir),
        intent_output_dir=str(args.intent_output_dir),
        dynamic_output_dir=str(args.dynamic_output_dir),
        bundle_output=str(args.bundle_output),
        vl_model=str(args.vl_model),
        text_model=str(args.text_model),
        category_hint=str(args.category_hint),
        query=str(args.query),
        min_candidate_items=int(args.min_candidate_items),
        max_candidate_items=int(args.max_candidate_items),
        max_history_rows=int(args.max_history_rows),
        top_n=int(args.top_n),
    )

    summary = run_pipeline(full_args)
    summary["prepared_item_desc_out"] = str(args.prepared_item_desc_out)
    summary["eval_unit_meta_out"] = str(args.eval_unit_meta_out)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    cli_args = build_argparser().parse_args()
    main(cli_args)

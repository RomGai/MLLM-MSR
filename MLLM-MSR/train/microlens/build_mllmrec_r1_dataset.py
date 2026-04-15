import argparse
import csv
import os
import random
from collections import defaultdict
from datasets import Dataset, DatasetDict, Image


PROMPT_TEMPLATE = (
    "Based on the user's historical interactions: {history_summary}\n"
    "Please predict whether the user will interact with the target item next. "
    "Target item summary: {candidate_summary}. "
    "Answer with only 'Yes' or 'No'."
)


def parse_item_desc(path):
    items = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            item_id = row["item_id"].strip()
            items[item_id] = {
                "image": row.get("image", "").strip(),
                "summary": row.get("summary", "").strip(),
            }
    return items


def parse_user_file(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 3:
                continue
            user = row[0].strip()
            pos = [x.strip() for x in row[1].split(",") if x.strip()]
            neg = [x.strip() for x in row[2].split(",") if x.strip()]
            if len(pos) < 2:
                continue
            rows.append((user, pos, neg))
    return rows


def build_history_summary(history_items, item_meta, max_hist=10):
    texts = []
    for item in history_items[-max_hist:]:
        desc = item_meta.get(item, {}).get("summary", "")
        if desc:
            texts.append(desc[:180])
    if not texts:
        return "No rich history text is available."
    return " | ".join(texts)


def create_examples(user_rows, item_meta, all_item_ids, negatives_per_user, seed):
    rng = random.Random(seed)
    examples = {"user": [], "item": [], "prompt": [], "image": [], "ground_truth": []}

    for idx, (user, pos_items, neg_items) in enumerate(user_rows, start=1):
        history = pos_items[:-1]
        target = pos_items[-1]
        user_seen = set(pos_items)
        history_summary = build_history_summary(history, item_meta)

        candidate_negs = list(neg_items)
        if len(candidate_negs) < negatives_per_user:
            pool = [x for x in all_item_ids if x not in user_seen and x != target]
            candidate_negs.extend(rng.sample(pool, k=min(len(pool), negatives_per_user - len(candidate_negs))))
        candidate_negs = candidate_negs[:negatives_per_user]

        for item, label in [(target, "Yes")] + [(n, "No") for n in candidate_negs]:
            meta = item_meta.get(item, {"summary": "", "image": ""})
            candidate_summary = meta.get("summary", "")[:300]
            prompt = PROMPT_TEMPLATE.format(
                history_summary=history_summary,
                candidate_summary=candidate_summary or "No candidate summary available",
            )
            examples["user"].append(user)
            examples["item"].append(item)
            examples["prompt"].append(prompt)
            examples["image"].append(meta.get("image", ""))
            examples["ground_truth"].append(label)

        if idx % 200 == 0:
            print(f"[Dataset Build] processed users: {idx}/{len(user_rows)}")

    return examples


def main():
    parser = argparse.ArgumentParser(description="Build train/validation dataset for MLLMRec-R1 style training.")
    parser.add_argument("--processed-dir", default="MLLM-MSR/data/amazon/processed")
    parser.add_argument("--domain", default="Baby_Products")
    parser.add_argument("--negatives-per-user", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--output", default="MLLM-MSR/train/microlens/amazon_mllmrec_r1_dataset")
    args = parser.parse_args()

    print("[Step 1/5] Resolving data files from filename split rule (_train/_test).")
    train_file = os.path.join(args.processed_dir, f"{args.domain}_user_items_negs_train.csv")
    test_file = os.path.join(args.processed_dir, f"{args.domain}_user_items_negs_test.csv")
    item_desc_file = os.path.join(args.processed_dir, f"{args.domain}_item_desc.tsv")

    for f in [train_file, test_file, item_desc_file]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"missing required file: {f}")

    print("[Step 2/5] Loading item metadata.")
    item_meta = parse_item_desc(item_desc_file)
    all_item_ids = sorted(item_meta.keys())
    print(f"Loaded item_meta: {len(item_meta)} items")

    print("[Step 3/5] Loading split users.")
    train_rows = parse_user_file(train_file)
    test_rows = parse_user_file(test_file)
    print(f"Train users: {len(train_rows)} | Test users: {len(test_rows)}")

    print("[Step 4/5] Building examples for train users.")
    examples = create_examples(train_rows, item_meta, all_item_ids, args.negatives_per_user, args.seed)

    print("[Step 5/5] Creating HuggingFace DatasetDict and saving to disk.")
    dataset = Dataset.from_dict(examples)
    dataset = dataset.cast_column("image", Image())
    dataset = dataset.shuffle(seed=args.seed)
    split = dataset.train_test_split(test_size=args.val_ratio, seed=args.seed)
    ds = DatasetDict({"train": split["train"], "validation": split["test"]})
    ds.save_to_disk(args.output)
    print(ds)
    print(f"Saved dataset to: {args.output}")
    print(f"Test split users file kept for inference: {test_file}")


if __name__ == "__main__":
    main()

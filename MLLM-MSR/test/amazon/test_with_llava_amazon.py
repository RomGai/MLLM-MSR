import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import requests
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.nn.functional import softmax
from tqdm import tqdm
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor


@dataclass
class EvalCandidate:
    item_id: str
    label: int
    prompt: str
    image_url: str


@dataclass
class EvalUserGroup:
    user_id: str
    group_id: int
    candidates: List[EvalCandidate]


def parse_comma_ids(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def read_item_desc(path: Path) -> Dict[str, Dict[str, str]]:
    items: Dict[str, Dict[str, str]] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            item_id = str(row["item_id"]).strip()
            image_raw = row.get("image")
            summary_raw = row.get("summary")
            items[item_id] = {
                "image": (image_raw or "").strip(),
                "summary": (summary_raw or "").strip(),
            }
    return items


def read_user_history(path: Path) -> Dict[str, List[Tuple[int, str]]]:
    hist: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            user_id = str(row["user_id"]).strip()
            item_id = str(row["item_id"]).strip()
            ts = int(row["timestamp"])
            hist[user_id].append((ts, item_id))
    for uid in hist:
        hist[uid].sort(key=lambda x: x[0])
    return hist


def build_history_text(
    user_id: str,
    user_history: Dict[str, List[Tuple[int, str]]],
    item_desc: Dict[str, Dict[str, str]],
    max_hist: int,
    max_chars_per_item: int,
) -> str:
    rows = user_history.get(user_id, [])[-max_hist:]
    parts: List[str] = []
    for _, item_id in rows:
        desc = item_desc.get(item_id, {}).get("summary", "")
        if not desc:
            continue
        parts.append(desc[:max_chars_per_item])
    if not parts:
        return "No reliable history text is available."
    return "\n".join(f"{i + 1}. {p}" for i, p in enumerate(parts))


def build_prompt(history_text: str, target_item_summary: str) -> str:
    return (
        "[INST]<image>\n"
        "You are given an Amazon recommendation task.\n"
        "The user's historical preferences can be inferred from these previously interacted product descriptions:\n"
        f"{history_text}\n\n"
        "Target product description:\n"
        f"{target_item_summary}\n\n"
        "Predict whether the user will interact with the target product next. "
        "Answer only 'Yes' or 'No'.[/INST]"
    )


def build_eval_groups(
    test_negs_path: Path,
    user_pairs_path: Path,
    item_desc_path: Path,
    max_hist: int,
    max_chars_per_item: int,
    skip_missing_image: bool,
) -> List[EvalUserGroup]:
    item_desc = read_item_desc(item_desc_path)
    user_history = read_user_history(user_pairs_path)

    groups: List[EvalUserGroup] = []
    with test_negs_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        group_id = 0
        for row in reader:
            if len(row) < 3:
                continue
            user_id = str(row[0]).strip()
            pos_items = parse_comma_ids(row[1])
            neg_items = parse_comma_ids(row[2])

            history_text = build_history_text(
                user_id=user_id,
                user_history=user_history,
                item_desc=item_desc,
                max_hist=max_hist,
                max_chars_per_item=max_chars_per_item,
            )

            candidates: List[EvalCandidate] = []
            for item_id in pos_items:
                meta = item_desc.get(item_id, {})
                if skip_missing_image and not meta.get("image", ""):
                    continue
                candidates.append(
                    EvalCandidate(
                        item_id=item_id,
                        label=1,
                        prompt=build_prompt(history_text, meta.get("summary", "")),
                        image_url=meta.get("image", ""),
                    )
                )
            for item_id in neg_items:
                meta = item_desc.get(item_id, {})
                if skip_missing_image and not meta.get("image", ""):
                    continue
                candidates.append(
                    EvalCandidate(
                        item_id=item_id,
                        label=0,
                        prompt=build_prompt(history_text, meta.get("summary", "")),
                        image_url=meta.get("image", ""),
                    )
                )

            groups.append(EvalUserGroup(user_id=user_id, group_id=group_id, candidates=candidates))
            group_id += 1
    return groups


def load_or_download_image(url: str, timeout: int = 15) -> Image.Image:
    if not url:
        return Image.new("RGB", (336, 336), color=(0, 0, 0))
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")
    except Exception:
        return Image.new("RGB", (336, 336), color=(0, 0, 0))


def recall_at_k(labels: List[int], probs: List[float], k: int) -> float:
    pairs = sorted(zip(probs, labels), key=lambda x: x[0], reverse=True)
    topk_labels = [lab for _, lab in pairs[:k]]
    total_positives = max(sum(labels), 1)
    return float(sum(topk_labels) / total_positives)


def mrr_at_k(labels: List[int], probs: List[float], k: int) -> float:
    pairs = sorted(zip(probs, labels), key=lambda x: x[0], reverse=True)
    for rank, (_, lab) in enumerate(pairs[:k], start=1):
        if lab == 1:
            return 1.0 / float(rank)
    return 0.0


def ndcg_at_k(labels: List[int], probs: List[float], k: int) -> float:
    pairs = sorted(zip(probs, labels), key=lambda x: x[0], reverse=True)
    ranked_labels = [lab for _, lab in pairs[:k]]

    def dcg(binary_labels: List[int]) -> float:
        score = 0.0
        for i, rel in enumerate(binary_labels, start=1):
            score += (2**rel - 1) / np.log2(i + 1)
        return score

    ideal = sorted(labels, reverse=True)[:k]
    denom = dcg(ideal)
    if denom <= 1e-12:
        return 0.0
    return float(dcg(ranked_labels) / denom)


def summarize_running_metrics(
    per_user_metric_rows: List[Dict[str, float]],
    global_labels: List[int],
    global_probs: List[float],
    ks: List[int],
) -> Dict[str, float]:
    result: Dict[str, float] = {}
    if per_user_metric_rows:
        for k in ks:
            result[f"HR@{k}"] = float(np.mean([r[f"HR@{k}"] for r in per_user_metric_rows]))
            result[f"MRR@{k}"] = float(np.mean([r[f"MRR@{k}"] for r in per_user_metric_rows]))
            result[f"NDCG@{k}"] = float(np.mean([r[f"NDCG@{k}"] for r in per_user_metric_rows]))

    if len(set(global_labels)) > 1:
        result["AUC"] = float(roc_auc_score(global_labels, global_probs))
    else:
        result["AUC"] = 0.0
    return result


def run(args: argparse.Namespace) -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    processor = LlavaNextProcessor.from_pretrained(args.model_id)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
    ).to(device)
    model.eval()

    yes_ids = [
        processor.tokenizer.convert_tokens_to_ids("Yes"),
        processor.tokenizer.convert_tokens_to_ids("yes"),
    ]
    no_ids = [
        processor.tokenizer.convert_tokens_to_ids("No"),
        processor.tokenizer.convert_tokens_to_ids("no"),
    ]

    groups = build_eval_groups(
        test_negs_path=Path(args.test_negs_path),
        user_pairs_path=Path(args.user_pairs_path),
        item_desc_path=Path(args.item_desc_path),
        max_hist=args.max_hist,
        max_chars_per_item=args.max_chars_per_item,
        skip_missing_image=args.skip_missing_image,
    )

    if args.max_users > 0:
        groups = groups[: args.max_users]

    if not groups:
        raise ValueError("No user groups loaded. Please check input files.")

    min_candidate_num = min(len(g.candidates) for g in groups if g.candidates)
    ks = [k for k in (3, 5, 10) if k <= min_candidate_num]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    global_labels: List[int] = []
    global_probs: List[float] = []
    per_user_rows: List[Dict[str, float]] = []

    pred_path = output_dir / "predictions.jsonl"
    metrics_path = output_dir / "metrics.json"

    print(f"[Start] users={len(groups)} device={device} model={args.model_id}")
    with pred_path.open("w", encoding="utf-8") as wf:
        for idx, group in enumerate(tqdm(groups, desc="Users"), start=1):
            if not group.candidates:
                continue

            user_labels: List[int] = []
            user_probs: List[float] = []

            for cand in group.candidates:
                image = load_or_download_image(cand.image_url, timeout=args.image_timeout)
                inputs = processor(
                    text=cand.prompt,
                    images=image,
                    return_tensors="pt",
                    padding=True,
                ).to(device)

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=4,
                        return_dict_in_generate=True,
                        output_scores=True,
                    )

                step_scores = outputs["scores"][0][0]
                yes_logit = torch.max(step_scores[yes_ids])
                no_logit = torch.max(step_scores[no_ids])
                prob_yes = softmax(torch.stack([no_logit, yes_logit]), dim=0)[1].item()

                user_labels.append(cand.label)
                user_probs.append(prob_yes)

                global_labels.append(cand.label)
                global_probs.append(prob_yes)

                wf.write(
                    json.dumps(
                        {
                            "user_id": group.user_id,
                            "group_id": group.group_id,
                            "item_id": cand.item_id,
                            "label": cand.label,
                            "prob_yes": prob_yes,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            user_metric_row: Dict[str, float] = {}
            for k in ks:
                user_metric_row[f"HR@{k}"] = recall_at_k(user_labels, user_probs, k)
                user_metric_row[f"MRR@{k}"] = mrr_at_k(user_labels, user_probs, k)
                user_metric_row[f"NDCG@{k}"] = ndcg_at_k(user_labels, user_probs, k)
            per_user_rows.append(user_metric_row)

            running = summarize_running_metrics(per_user_rows, global_labels, global_probs, ks)
            metric_parts = [f"AUC={running['AUC']:.6f}"]
            for k in ks:
                metric_parts.append(
                    f"HR@{k}={running[f'HR@{k}']:.6f} MRR@{k}={running[f'MRR@{k}']:.6f} NDCG@{k}={running[f'NDCG@{k}']:.6f}"
                )
            print(f"[User {idx}/{len(groups)}] user_id={group.user_id} | " + " | ".join(metric_parts), flush=True)

    final_metrics = summarize_running_metrics(per_user_rows, global_labels, global_probs, ks)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(final_metrics, f, ensure_ascii=False, indent=2)

    print("[Done] Final metrics:")
    for key, val in final_metrics.items():
        print(f"{key}: {val:.6f}")
    print(f"Predictions saved to: {pred_path}")
    print(f"Metrics saved to: {metrics_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Single-GPU LLaVA inference on Amazon processed dataset.")
    p.add_argument("--model-id", default="llava-hf/llava-v1.6-mistral-7b-hf")
    p.add_argument("--item-desc-path", default="MLLM-MSR/data/amazon/processed/Baby_Products_item_desc.tsv")
    p.add_argument("--user-pairs-path", default="MLLM-MSR/data/amazon/processed/Baby_Products_u_i_pairs.tsv")
    p.add_argument("--test-negs-path", default="MLLM-MSR/data/amazon/processed/Baby_Products_user_items_negs_test.csv")
    p.add_argument("--output-dir", default="MLLM-MSR/test/amazon/outputs_baby_llava")
    p.add_argument("--max-hist", type=int, default=20)
    p.add_argument("--max-chars-per-item", type=int, default=180)
    p.add_argument("--max-users", type=int, default=0, help="0 means use all users")
    p.add_argument("--image-timeout", type=int, default=15)
    p.add_argument(
        "--skip-missing-image",
        action="store_true",
        default=True,
        help="Skip candidates whose image URL is missing. Enabled by default.",
    )
    p.add_argument(
        "--keep-missing-image",
        action="store_false",
        dest="skip_missing_image",
        help="Keep candidates with missing image URL and use blank fallback image.",
    )
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())

import argparse
import csv
import io
import math
import random
import urllib.request
from collections import defaultdict

import torch
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from peft import PeftModel


PROMPT_TEMPLATE = (
    "Based on the user's historical interactions: {history_summary}\n"
    "Please predict whether the user will interact with the target item next. "
    "Target item summary: {candidate_summary}. "
    "Answer with only 'Yes' or 'No'."
)


def parse_item_desc(path):
    item_meta = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            item_meta[row["item_id"].strip()] = {
                "image": row.get("image", "").strip(),
                "summary": row.get("summary", "").strip(),
            }
    return item_meta


def parse_user_rows(path):
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


def history_summary(pos_items, item_meta, max_hist=10):
    texts = []
    for item in pos_items[:-1][-max_hist:]:
        s = item_meta.get(item, {}).get("summary", "")[:180]
        if s:
            texts.append(s)
    return " | ".join(texts) if texts else "No rich history text is available."


def load_image(url):
    if not url:
        return Image.new("RGB", (224, 224), "black")
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            content = resp.read()
        return Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        return Image.new("RGB", (224, 224), "black")


def resize_images(images):
    max_width = max(img.width for img in images)
    max_height = max(img.height for img in images)
    out = []
    for img in images:
        canvas = Image.new("RGB", (max_width, max_height), "black")
        x = (max_width - img.width) // 2
        y = (max_height - img.height) // 2
        canvas.paste(img, (x, y))
        out.append(canvas)
    return out


def batched(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i:i + n]


def score_candidates(model, processor, device, history, candidates, item_meta, batch_size):
    yes_id = processor.tokenizer.convert_tokens_to_ids("Yes")
    no_id = processor.tokenizer.convert_tokens_to_ids("No")
    scores = {}

    for batch_items in batched(candidates, batch_size):
        prompts, images = [], []
        for item in batch_items:
            meta = item_meta.get(item, {"summary": "", "image": ""})
            prompt = PROMPT_TEMPLATE.format(
                history_summary=history,
                candidate_summary=meta.get("summary", "")[:300] or "No candidate summary available",
            )
            prompts.append(prompt)
            images.append(load_image(meta.get("image", "")))

        images = resize_images(images)
        inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=2, return_dict_in_generate=True, output_scores=True)

        logits = outputs.scores[0]
        yes_logits = logits[:, yes_id]
        no_logits = logits[:, no_id]
        probs = torch.softmax(torch.stack([no_logits, yes_logits], dim=1), dim=1)[:, 1].detach().cpu().tolist()
        for item, prob in zip(batch_items, probs):
            scores[item] = prob

    return scores


def metric_at_k(rank, k):
    hit = 1.0 if rank <= k else 0.0
    ndcg = 1.0 / math.log2(rank + 1) if rank <= k else 0.0
    return hit, ndcg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="MLLM-MSR/data/amazon/processed")
    parser.add_argument("--domain", default="Baby_Products")
    parser.add_argument("--model-id", default="llava-hf/llava-v1.6-mistral-7b-hf")
    parser.add_argument("--peft-model-id", default=None)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-users", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    test_file = f"{args.processed_dir}/{args.domain}_user_items_negs_test.csv"
    item_desc_file = f"{args.processed_dir}/{args.domain}_item_desc.tsv"

    print("[Infer Step 1/4] Loading metadata and test users (filename contains _test).")
    item_meta = parse_item_desc(item_desc_file)
    user_rows = parse_user_rows(test_file)
    if args.max_users > 0:
        user_rows = user_rows[: args.max_users]
    all_items = sorted(item_meta.keys())
    print(f"Loaded test users={len(user_rows)}, items={len(all_items)}")

    print("[Infer Step 2/4] Loading backbone model from HuggingFace.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_kwargs = {
        "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
    }
    if device == "cuda":
        model_kwargs["_attn_implementation"] = "flash_attention_2"
    model = LlavaNextForConditionalGeneration.from_pretrained(args.model_id, **model_kwargs).to(device).eval()
    if args.peft_model_id:
        print(f"Loading LoRA checkpoint: {args.peft_model_id}")
        model = PeftModel.from_pretrained(model, args.peft_model_id).to(device).eval()

    print("[Infer Step 3/4] Loading processor.")
    processor = LlavaNextProcessor.from_pretrained(args.model_id)
    processor.tokenizer.pad_token = processor.tokenizer.eos_token

    print("[Infer Step 4/4] User-wise ranking from 1 target + 1000 random negatives.")
    ks = [10, 20, 40]
    running = defaultdict(float)

    for idx, (user, pos, neg) in enumerate(user_rows, start=1):
        target = pos[-1]
        seen = set(pos)

        candidate_negs = list(neg)
        if len(candidate_negs) < args.sample_size:
            pool = [x for x in all_items if x not in seen and x != target]
            need = min(len(pool), args.sample_size - len(candidate_negs))
            candidate_negs.extend(rng.sample(pool, need))
        else:
            candidate_negs = rng.sample(candidate_negs, args.sample_size)

        candidates = [target] + candidate_negs[: args.sample_size]
        hist = history_summary(pos, item_meta)

        score_map = score_candidates(model, processor, device, hist, candidates, item_meta, args.batch_size)
        ranked = sorted(candidates, key=lambda x: score_map.get(x, -1e9), reverse=True)
        rank = ranked.index(target) + 1

        running["mrr"] += 1.0 / rank
        running["users"] += 1
        for k in ks:
            h, n = metric_at_k(rank, k)
            running[f"hr@{k}"] += h
            running[f"ndcg@{k}"] += n

        done = int(running["users"])
        avg_line = " | ".join(
            [f"HR@{k}={running[f'hr@{k}']/done:.4f}, NDCG@{k}={running[f'ndcg@{k}']/done:.4f}" for k in ks]
        )
        print(f"[User {idx}/{len(user_rows)}] user={user} target_rank={rank} | AVG {avg_line}")

    total = int(running["users"])
    print("\n===== FINAL METRICS =====")
    for k in ks:
        print(f"HR@{k}: {running[f'hr@{k}']/total:.6f}")
        print(f"NDCG@{k}: {running[f'ndcg@{k}']/total:.6f}")
    print(f"MRR: {running['mrr']/total:.6f}")


if __name__ == "__main__":
    main()

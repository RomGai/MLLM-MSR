"""Unified per-user pipeline: Agent3 -> Agent1/2 -> Agent4/5 for amazon_beauty."""

from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from dynamic_reasoning_ranking_agent import run_module3
from item_profiler_agents import (
    CandidateItemProfiler,
    GlobalItemDB,
    HistoryItemProfiler,
    HistoryItemProfileInput,
    ItemProfileInput,
    Qwen3VLExtractor,
    UserHistoryLogDB,
)


def _read_jsonish_lines(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                obj = ast.literal_eval(raw)
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _load_query_rows(path: str | Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _pick_col(row: Dict[str, Any], cands: Sequence[str], default: str = "") -> str:
    for c in cands:
        v = str(row.get(c, "") or "").strip()
        if v:
            return v
    return default


def _to_sentence_for_item(meta: Dict[str, Any]) -> str:
    cats = meta.get("categories") or []
    cat_text = " ".join(" > ".join(x) if isinstance(x, list) else str(x) for x in cats)
    return f"categories: {cat_text}; title: {meta.get('title','')}; description: {meta.get('description','')}"


def _to_sentence_for_query(query: str, cat_hints: Sequence[str]) -> str:
    cat_text = " | ".join(cat_hints)
    return f"用户需求：{query}。相关商品类别：{cat_text}。请检索满足该需求的商品。"


def _topk_cosine(query_emb: np.ndarray, doc_emb: np.ndarray, k: int) -> List[int]:
    q = query_emb / (np.linalg.norm(query_emb) + 1e-12)
    d = doc_emb / (np.linalg.norm(doc_emb, axis=1, keepdims=True) + 1e-12)
    scores = d @ q
    k = min(k, len(scores))
    return np.argsort(-scores)[:k].tolist()


def run(args: argparse.Namespace) -> None:
    meta_rows = _read_jsonish_lines(args.filtered_meta_jsonl)
    meta_by_asin = {str(r.get("asin", "")): r for r in meta_rows if str(r.get("asin", "")).strip()}
    catalog_ids = list(meta_by_asin.keys())
    print(f"[Init] filtered catalog size={len(catalog_ids)}")

    queries = _load_query_rows(args.query_csv)
    print(f"[Init] query rows={len(queries)}")

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    item_sentence_path = cache_dir / "item_sentences.json"
    item_emb_path = cache_dir / "item_embeddings.npy"

    if item_sentence_path.exists() and item_emb_path.exists():
        item_sentences = json.loads(item_sentence_path.read_text(encoding="utf-8"))
        item_embeddings = np.load(item_emb_path)
        print(f"[Agent3][Cache] loaded item embeddings: {item_embeddings.shape}")
    else:
        item_sentences = [_to_sentence_for_item(meta_by_asin[i]) for i in catalog_ids]
        model = SentenceTransformer(args.embedding_model)
        item_embeddings = model.encode(item_sentences, batch_size=args.embed_batch_size, show_progress_bar=True)
        item_embeddings = np.asarray(item_embeddings)
        item_sentence_path.write_text(json.dumps(item_sentences, ensure_ascii=False), encoding="utf-8")
        np.save(item_emb_path, item_embeddings)
        print(f"[Agent3] built item embeddings: {item_embeddings.shape}")

    model = SentenceTransformer(args.embedding_model)
    global_db = GlobalItemDB(args.global_db)
    history_db = UserHistoryLogDB(args.history_db)
    extractor = Qwen3VLExtractor(model_name=args.vl_model)
    agent1 = CandidateItemProfiler(extractor, global_db)
    agent2 = HistoryItemProfiler(extractor, history_db)

    query_cache_path = cache_dir / "query_embeddings.json"
    query_cache = json.loads(query_cache_path.read_text(encoding="utf-8")) if query_cache_path.exists() else {}

    for idx, row in enumerate(queries, start=1):
        user_id = _pick_col(row, ["user_id", "user", "uid"]) or f"row_{idx}"
        query = _pick_col(row, ["query", "text", "question"])
        target = _pick_col(row, ["target", "target_item_id", "item_id", "id", "asin"])
        history_raw = _pick_col(row, ["history", "history_items", "hist_items"])
        history_ids = [x.strip() for x in history_raw.split(",") if x.strip()]

        cat_hints = [" > ".join(x) for x in meta_by_asin.get(target, {}).get("categories", []) if isinstance(x, list)]
        query_sentence = _to_sentence_for_query(query, cat_hints)
        if query_sentence in query_cache:
            q_emb = np.asarray(query_cache[query_sentence], dtype=np.float32)
            print(f"[Agent3][QueryCache] user={user_id} reused embedding")
        else:
            q_emb = np.asarray(model.encode([query_sentence], prompt_name="query")[0], dtype=np.float32)
            query_cache[query_sentence] = q_emb.tolist()
            print(f"[Agent3][Embed] user={user_id} computed query embedding")

        topk = _topk_cosine(q_emb, item_embeddings, k=200)
        recalled_ids = [catalog_ids[i] for i in topk]
        hit = target in set(recalled_ids)
        print(f"[Agent3][Recall] user={user_id} top200_hit={hit}")

        if not hit:
            topk = _topk_cosine(q_emb, item_embeddings, k=500)
            recalled_ids = [catalog_ids[i] for i in topk]
            hit = target in set(recalled_ids)
            print(f"[Agent3][Recall] user={user_id} top500_hit={hit}")
        if not hit:
            print(f"[Agent3][Fail] user={user_id} target={target}, metric=0, skip downstream")
            continue

        candidate_items: List[Dict[str, Any]] = []
        for item_id in recalled_ids:
            item_meta = meta_by_asin[item_id]
            profile = global_db.get_profile(item_id)
            if profile is None:
                item_input = ItemProfileInput(
                    item_id=item_id,
                    title=str(item_meta.get("title", "")),
                    detail_text=str(item_meta.get("description", "")),
                    main_image=str(item_meta.get("imUrl", "")),
                    price=str(item_meta.get("price", "")) if item_meta.get("price") is not None else None,
                    category_hint=args.category_hint,
                )
                profile = agent1.profile_and_store(item_input)
                print(f"[Agent1] profiled item={item_id}")
            candidate_items.append({"item_id": item_id, "profile": profile})

        history_rows: List[Dict[str, Any]] = []
        for h_id in history_ids:
            if h_id not in meta_by_asin:
                continue
            h_meta = meta_by_asin[h_id]
            h_profile = global_db.get_profile(h_id)
            if h_profile is None:
                h_profile = agent1.profile_and_store(
                    ItemProfileInput(
                        item_id=h_id,
                        title=str(h_meta.get("title", "")),
                        detail_text=str(h_meta.get("description", "")),
                        main_image=str(h_meta.get("imUrl", "")),
                        category_hint=args.category_hint,
                    )
                )
            hist_input = HistoryItemProfileInput(
                item_id=h_id,
                title=str(h_meta.get("title", "")),
                detail_text=str(h_meta.get("description", "")),
                main_image=str(h_meta.get("imUrl", "")),
                user_id=user_id,
                behavior="positive",
                timestamp=0,
                category_hint=args.category_hint,
            )
            if not history_db.exists(user_id=user_id, item_id=h_id, behavior="positive", timestamp=0):
                _ = agent2.profile_and_store(hist_input)
                print(f"[Agent2] profiled history item={h_id} user={user_id}")
            history_rows.append({"user_id": user_id, "item_id": h_id, "behavior": "positive", "profile": h_profile})

        payload = {
            "query": query,
            "user_id": user_id,
            "routing": {
                "selected_category_paths": [c.split(" > ") for c in cat_hints],
                "rewritten_query_sentence": query_sentence,
                "first_recall_k": 500 if len(recalled_ids) > 200 else 200,
            },
            "candidate_items": candidate_items,
            "query_relevant_history": history_rows,
        }
        out_dir = Path(args.intent_output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"user_{user_id}_intent_dual_recall_output.json"
        out_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        run_module3(
            intent_dual_recall_output=payload,
            model_name=args.text_model,
            top_n=args.top_n,
            save_output=True,
            output_dir=args.dynamic_output_dir,
        )
        print(f"[Agent4/5] user={user_id} done ({idx}/{len(queries)})")

    query_cache_path.write_text(json.dumps(query_cache, ensure_ascii=False), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Unified per-user new_pipe with Agent3-first retrieval")
    p.add_argument("--query-csv", default="data/amazon_beauty/query_data1.csv")
    p.add_argument("--filtered-meta-jsonl", default="data/amazon_beauty/meta_Beauty.filtered.jsonl")
    p.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-0.6B")
    p.add_argument("--embed-batch-size", type=int, default=64)
    p.add_argument("--cache-dir", default="data/amazon_beauty/cache")
    p.add_argument("--global-db", default="data/amazon_beauty/processed/global_item_features.db")
    p.add_argument("--history-db", default="data/amazon_beauty/processed/user_history_log.db")
    p.add_argument("--intent-output-dir", default="data/amazon_beauty/processed/intent_dual_recall_outputs")
    p.add_argument("--dynamic-output-dir", default="data/amazon_beauty/processed/dynamic_reasoning_ranking_outputs")
    p.add_argument("--vl-model", default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--text-model", default="Qwen/Qwen3-8B")
    p.add_argument("--top-n", type=int, default=21)
    p.add_argument("--category-hint", default="Beauty")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())

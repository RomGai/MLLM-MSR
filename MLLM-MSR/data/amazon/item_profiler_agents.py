"""MLLM-based item profiling agents for Amazon processed data.

This module builds two profiling agents around the output format from
`process_data.py`:

- Agent 1 (CandidateItemProfiler): profile candidate items and write
  structured features into a global item DB.
- Agent 2 (HistoryItemProfiler): profile historical interacted items and write
  item features with behavior label + timestamp into a user history log DB.

The implementation is designed for Qwen3VL-8B style multimodal models.
"""

from __future__ import annotations

import csv
import json
import os
import random
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
# try:
#     import torch
#     from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
#except Exception:  # pragma: no cover - allow lightweight environments
#     torch = None
#     AutoProcessor = None
#     Qwen3VLForConditionalGeneration = None


BehaviorLabel = Literal["positive", "negative"]


@dataclass
class ItemProfileInput:
    item_id: str
    title: str
    detail_text: str
    main_image: str
    detail_images: List[str] = field(default_factory=list)
    price: Optional[str] = None
    brand: Optional[str] = None
    category_hint: Optional[str] = None


@dataclass
class HistoryItemProfileInput(ItemProfileInput):
    user_id: str = ""
    behavior: BehaviorLabel = "positive"
    timestamp: int = 0


class Qwen3VLExtractor:
    """Qwen3-VL wrapper following official transformers usage style."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        device: str = "cuda",
        torch_dtype: str = "auto",
        max_new_tokens: Optional[int] = None,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens or int(os.getenv("out_seq_length", "16384"))
        self.device = device
        self.torch_dtype = torch_dtype
        self.do_sample = os.getenv("greedy", "false").lower() != "true"
        self.top_p = float(os.getenv("top_p", "0.8"))
        self.top_k = int(os.getenv("top_k", "20"))
        self.temperature = float(os.getenv("temperature", "0.7"))
        self.repetition_penalty = float(os.getenv("repetition_penalty", "1.0"))
        self.presence_penalty = float(os.getenv("presence_penalty", "1.5"))
        self._model = None
        self._processor = None

    def load(self) -> None:
        if AutoProcessor is None or Qwen3VLForConditionalGeneration is None or torch is None:
            raise ImportError(
                "transformers/torch are not available. Install required dependencies first."
            )

        if self._model is not None and self._processor is not None:
            return

        self._processor = AutoProcessor.from_pretrained(self.model_name)
        if self.torch_dtype == "auto":
            self._model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_name,
                dtype="auto",
                device_map="auto" if self.device == "cuda" else None,
            )
        else:
            dtype = getattr(torch, self.torch_dtype)
            self._model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_name,
                dtype=dtype,
                device_map="auto" if self.device == "cuda" else None,
            )

        if self.device != "cuda":
            self._model.to(self.device)

    def extract(
        self,
        prompt: str,
        image_paths: List[str],
    ) -> Dict[str, Any]:
        self.load()

        image_messages = [{"type": "image", "image": path} for path in image_paths]
        messages = [
            {
                "role": "user",
                "content": [
                    *image_messages,
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self._model.device)

        generate_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "temperature": self.temperature,
            "repetition_penalty": self.repetition_penalty,
            "presence_penalty": self.presence_penalty,
        }
        try:
            output_ids = self._model.generate(**inputs, **generate_kwargs)
        except TypeError:
            # Some transformers versions do not support `presence_penalty` in generate().
            generate_kwargs.pop("presence_penalty", None)
            output_ids = self._model.generate(**inputs, **generate_kwargs)

        generated_ids = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output_ids)
        ]
        generated_text = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        # parse first JSON object from generation
        start = generated_text.find("{")
        end = generated_text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"Model output is not valid JSON: {generated_text}")
        payload = generated_text[start : end + 1]
        return json.loads(payload)


class GlobalItemDB:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS global_item_features (
                item_id TEXT PRIMARY KEY,
                profile_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        self.conn.commit()

    def upsert(self, item_id: str, profile: Dict[str, Any]) -> None:
        self.conn.execute(
            """
            INSERT INTO global_item_features (item_id, profile_json, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(item_id) DO UPDATE SET
                profile_json=excluded.profile_json,
                updated_at=excluded.updated_at
            """,
            (
                item_id,
                json.dumps(profile, ensure_ascii=False),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self.conn.commit()


class UserHistoryLogDB:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_history_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                item_id TEXT NOT NULL,
                behavior TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                profile_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_user_time
            ON user_history_profiles (user_id, timestamp)
            """
        )
        self.conn.commit()

    def insert(
        self,
        user_id: str,
        item_id: str,
        behavior: BehaviorLabel,
        timestamp: int,
        profile: Dict[str, Any],
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO user_history_profiles
            (user_id, item_id, behavior, timestamp, profile_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                item_id,
                behavior,
                int(timestamp),
                json.dumps(profile, ensure_ascii=False),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self.conn.commit()


def build_profile_prompt(item: ItemProfileInput) -> str:
    """Prompt template for fine-grained textual + visual profiling."""

    image_count = 1 + len(item.detail_images)
    return f"""
You are an expert e-commerce item profiler.
Given product text and {image_count} images (first is main image, rest are detail images),
extract a fine-grained feature profile in STRICT JSON.

Item text fields:
- title: {item.title}
- detail_text: {item.detail_text}
- brand: {item.brand or ''}
- price: {item.price or ''}
- category_hint: {item.category_hint or ''}

User shopping-oriented extraction requirements:
1) Type-first taxonomy (important):
   - Always output `item_type` (required).
   - If hierarchical category is uncertain, keep only `item_type` and leave `category_path` empty.
   - If known, output `category_path` as a list (e.g., ["Electronics", "Gaming", "Headset"]).
   - Also infer use_case, target_people, seasonality.
2) Textual attribute tags (fine-grained):
   - title keyword summary (must leverage title)
   - material/fabric composition
   - core features & specs (size, capacity, weight, dimensions, compatibility, power, ingredients)
   - package/bundle information
   - quality & durability claims
   - comfort/usability claims
   - price_band inference (budget/mid/premium) and value_for_money signal
3) Visual style tags (from images):
   - dominant colors (+ optional hex-like names)
   - silhouette/shape/form factor
   - style keywords (minimalist, sporty, retro, luxury, kawaii, etc.)
   - texture/finish (matte/glossy/metallic/knit/grainy)
   - pattern/print/logo density
   - scene mood (homey, professional, outdoor, gaming, etc.)
   - perceived quality level (low/medium/high with confidence)
4) Output quality:
   - Every major field must include a confidence in [0,1].
   - Put uncertain values under "hypotheses".
   - Output ONLY one JSON object. No markdown.

JSON schema:
{{
  "item_id": "{item.item_id}",
  "title": "{item.title}",
  "taxonomy": {{
    "item_type": "",
    "category_path": [],
    "use_case": [],
    "target_people": [],
    "seasonality": "",
    "confidence": 0.0
  }},
  "text_tags": {{...}},
  "visual_tags": {{...}},
  "hypotheses": ["..."],
  "overall_confidence": 0.0
}}
""".strip()


class CandidateItemProfiler:
    """Agent 1: Candidate Item Profiler."""

    def __init__(self, extractor: Qwen3VLExtractor, global_db: GlobalItemDB) -> None:
        self.extractor = extractor
        self.global_db = global_db

    def profile_and_store(self, item: ItemProfileInput) -> Dict[str, Any]:
        prompt = build_profile_prompt(item)
        image_paths = [item.main_image, *item.detail_images]
        profile = self.extractor.extract(prompt, image_paths)
        self.global_db.upsert(item.item_id, profile)
        return profile


class HistoryItemProfiler:
    """Agent 2: History Item Profiler."""

    def __init__(self, extractor: Qwen3VLExtractor, history_db: UserHistoryLogDB) -> None:
        self.extractor = extractor
        self.history_db = history_db

    def profile_and_store(self, item: HistoryItemProfileInput) -> Dict[str, Any]:
        prompt = build_profile_prompt(item)
        image_paths = [item.main_image, *item.detail_images]
        profile = self.extractor.extract(prompt, image_paths)
        profile["behavior"] = item.behavior
        profile["timestamp"] = int(item.timestamp)
        profile["user_id"] = item.user_id

        self.history_db.insert(
            user_id=item.user_id,
            item_id=item.item_id,
            behavior=item.behavior,
            timestamp=item.timestamp,
            profile=profile,
        )
        return profile


def load_item_desc_tsv(path: str | Path) -> Dict[str, Dict[str, str]]:
    """Load `*_item_desc.tsv` created by process_data.py into item metadata map."""
    item_map: Dict[str, Dict[str, str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            item_map[str(row["item_id"])] = {
                "image": row.get("image", ""),
                "summary": row.get("summary", ""),
            }
    return item_map


def load_user_interactions(path: str | Path) -> Iterable[Dict[str, str]]:
    """Load `*_u_i_pairs.tsv` rows."""
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            yield row


def expand_pos_neg_rows(
    user_items_negs_path: str | Path,
) -> Iterable[Dict[str, Any]]:
    """Expand `*_user_items_negs.tsv` to (user,item,behavior) rows."""
    with open(user_items_negs_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            user = str(row["user_id"])
            for item in str(row["pos"]).split(","):
                if item:
                    yield {"user_id": user, "item_id": item, "behavior": "positive"}
            for item in str(row["neg"]).split(","):
                if item:
                    yield {"user_id": user, "item_id": item, "behavior": "negative"}


def bootstrap_agents_from_processed(
    item_desc_tsv: str | Path,
    global_db_path: str | Path,
    history_db_path: str | Path,
    model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
) -> tuple[CandidateItemProfiler, HistoryItemProfiler]:
    """Build both agents from processed Amazon format."""
    _ = load_item_desc_tsv(item_desc_tsv)  # preload check to ensure input validity

    extractor = Qwen3VLExtractor(model_name=model_name)
    global_db = GlobalItemDB(global_db_path)
    history_db = UserHistoryLogDB(history_db_path)
    return CandidateItemProfiler(extractor, global_db), HistoryItemProfiler(extractor, history_db)


def _sample_distinct_items(
    item_map: Dict[str, Dict[str, str]],
    k: int,
    seed: int = 2025,
) -> List[str]:
    """Sample up to k distinct item_ids from item metadata map."""
    item_ids = list(item_map.keys())
    rng = random.Random(seed)
    rng.shuffle(item_ids)
    return item_ids[: min(k, len(item_ids))]


def _sample_distinct_user_item_rows(
    rows: Iterable[Dict[str, str]],
    k: int,
    seed: int = 2025,
) -> List[Dict[str, str]]:
    """Sample up to k rows with distinct user_id and item_id."""
    all_rows = list(rows)
    rng = random.Random(seed)
    rng.shuffle(all_rows)

    picked: List[Dict[str, str]] = []
    used_users = set()
    used_items = set()

    for row in all_rows:
        user_id = str(row.get("user_id", ""))
        item_id = str(row.get("item_id", ""))
        if not user_id or not item_id:
            continue
        if user_id in used_users or item_id in used_items:
            continue
        picked.append(row)
        used_users.add(user_id)
        used_items.add(item_id)
        if len(picked) >= k:
            break

    return picked


if __name__ == "__main__":
    # Real runnable example:
    # - randomly sample 10 candidate items from item DB
    # - randomly sample 10 user-history interactions from user DB
    # - run both profilers and print profile outputs
    random_seed = 2025
    sample_k = 10
    item_desc_tsv_path = "./processed/Video_Games_item_desc.tsv"
    user_pairs_tsv_path = "./processed/Video_Games_u_i_pairs.tsv"

    candidate_profiler, history_profiler = bootstrap_agents_from_processed(
        item_desc_tsv=item_desc_tsv_path,
        global_db_path="./processed/global_item_features.db",
        history_db_path="./processed/user_history_log.db",
    )

    item_map = load_item_desc_tsv(item_desc_tsv_path)
    sampled_item_ids = _sample_distinct_items(item_map, k=sample_k, seed=random_seed)

    print(f"\n[Agent 1] Running candidate profiling on {len(sampled_item_ids)} sampled items...")
    for idx, item_id in enumerate(sampled_item_ids, start=1):
        meta = item_map[item_id]
        sample_item = ItemProfileInput(
            item_id=item_id,
            title=f"item_{item_id}",
            detail_text=meta.get("summary", "") or "",
            main_image=meta.get("image", ""),
            detail_images=[],
            category_hint="Video_Games",
        )
        profile = candidate_profiler.profile_and_store(sample_item)
        print(f"\n[Agent 1][{idx}/{len(sampled_item_ids)}] item_id={item_id}")
        print(json.dumps(profile, ensure_ascii=False, indent=2))

    sampled_user_rows = _sample_distinct_user_item_rows(
        load_user_interactions(user_pairs_tsv_path),
        k=sample_k,
        seed=random_seed,
    )

    print(f"\n[Agent 2] Running history profiling on {len(sampled_user_rows)} sampled user-item rows...")
    for idx, row in enumerate(sampled_user_rows, start=1):
        user_id = str(row["user_id"])
        item_id = str(row["item_id"])
        timestamp = int(row["timestamp"])
        meta = item_map.get(item_id, {"image": "", "summary": ""})
        hist_item = HistoryItemProfileInput(
            item_id=item_id,
            title=f"item_{item_id}",
            detail_text=meta.get("summary", "") or "",
            main_image=meta.get("image", ""),
            detail_images=[],
            category_hint="Video_Games",
            user_id=user_id,
            behavior="positive",
            timestamp=timestamp,
        )
        profile = history_profiler.profile_and_store(hist_item)
        print(
            f"\n[Agent 2][{idx}/{len(sampled_user_rows)}] "
            f"user_id={user_id}, item_id={item_id}, ts={timestamp}"
        )
        print(json.dumps(profile, ensure_ascii=False, indent=2))

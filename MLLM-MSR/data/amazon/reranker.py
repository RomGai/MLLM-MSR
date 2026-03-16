"""LLM-based item reranker for Amazon recommendation (Qwen3-8B).

This file implements Agent 5 compatible scoring:
- yes/no relevance decision
- yes/no logits-based score

Compared with previous multimodal reranker, this version is pure text LLM,
using product structured profile from module-1 and dynamic constraints from module-3.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:  # pragma: no cover
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None


def _normalize_prediction_text(v: Any) -> str:
    return " ".join(str(v).strip().lower().split())


def _collect_prediction_targets(preference_constraints: Dict[str, Any]) -> List[Tuple[str, float]]:
    preds = preference_constraints.get("Predicted_Next_Items", [])
    if not isinstance(preds, list):
        return []

    out: List[Tuple[str, float]] = []
    for row in preds:
        if isinstance(row, str):
            token = _normalize_prediction_text(row)
            if token:
                out.append((token, 1.0))
            continue
        if not isinstance(row, dict):
            continue
        token = _normalize_prediction_text(row.get("item_type", ""))
        if token:
            out.append((token, 1.0))
    return out


class LLMItemReranker:
    """Rerank candidate items with Qwen3-8B via yes/no logits."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-8B",
        max_new_tokens: int = 8,
        enable_thinking: bool = False,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.enable_thinking = enable_thinking
        self._tokenizer = None
        self._model = None

        self.id_yes = None
        self.id_no = None

    def load(self) -> None:
        if AutoTokenizer is None or AutoModelForCausalLM is None or torch is None:
            raise ImportError("transformers/torch are not available for LLMItemReranker.")
        if self._model is not None and self._tokenizer is not None:
            return

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto",
        )

        self.id_yes = self._tokenizer.convert_tokens_to_ids("yes")
        self.id_no = self._tokenizer.convert_tokens_to_ids("no")

    @torch.no_grad()
    def _score_with_logits(self, prompt: str) -> Dict[str, Any]:
        self.load()
        messages = [{"role": "user", "content": prompt}]
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)
        logits = self._model(**inputs).logits[:, -1, :]

        score_logits = torch.stack([logits[:, self.id_no], logits[:, self.id_yes]], dim=1)
        probs = torch.nn.functional.softmax(score_logits, dim=1)[0]
        p = probs.tolist()
        yes_prob = float(p[1])

        return {
            "probs": {"no": p[0], "yes": p[1]},
            "weighted_score": yes_prob,
        }

    @staticmethod
    def _build_scoring_prompt(
        query: str,
        preference_constraints: Dict[str, Any],
        item: Dict[str, Any],
    ) -> str:
        must_avoid = preference_constraints.get("Must_Avoid", [])

        profile = item.get("profile", {})
        compact_item = {
            "title": profile.get("title", ""),
            "taxonomy": profile.get("taxonomy", {}),
            "text_tags": profile.get("text_tags", {}),
            "visual_tags": profile.get("visual_tags", {}),
        }

        next_predictions = preference_constraints.get("Predicted_Next_Items", [])

        return (
            f"Based on the previous interaction history, the user's preference can be summarized as: "
            f"Must_Avoid={json.dumps(must_avoid, ensure_ascii=False)}; "
            f"Predicted_Next_Items={json.dumps(next_predictions, ensure_ascii=False)}.\n"
            f"Please predict whether this user would interact with the item at the next opportunity. "
            f"The candidate item is '{json.dumps(compact_item, ensure_ascii=False)}'.\n"
            "Please only response 'yes' or 'no' based on your judgement, do not include any other content including words, space, and punctuations in your response."
        )

    @staticmethod
    def _must_avoid_filter(preference_constraints: Dict[str, Any], item: Dict[str, Any]) -> bool:
        must_avoid = [str(x).strip().lower() for x in preference_constraints.get("Must_Avoid", []) if str(x).strip()]
        if not must_avoid:
            return False

        profile = item.get("profile", {})
        haystacks = [
            profile.get("title", ""),
            json.dumps(profile.get("taxonomy", {}), ensure_ascii=False),
            json.dumps(profile.get("text_tags", {}), ensure_ascii=False),
            json.dumps(profile.get("visual_tags", {}), ensure_ascii=False),
            json.dumps(profile.get("hypotheses", []), ensure_ascii=False),
        ]
        item_text = "\n".join(str(x) for x in haystacks).lower()
        return any(token and token in item_text for token in must_avoid)



    @staticmethod
    def _prediction_alignment_bonus(preference_constraints: Dict[str, Any], item: Dict[str, Any]) -> float:
        targets = _collect_prediction_targets(preference_constraints)
        if not targets:
            return 0.0

        profile = item.get("profile", {})
        haystacks = [
            profile.get("title", ""),
            json.dumps(profile.get("taxonomy", {}), ensure_ascii=False),
            json.dumps(profile.get("text_tags", {}), ensure_ascii=False),
            json.dumps(profile.get("visual_tags", {}), ensure_ascii=False),
            json.dumps(profile.get("hypotheses", []), ensure_ascii=False),
        ]
        item_text = _normalize_prediction_text("\n".join(str(x) for x in haystacks))

        bonus = 0.0
        for token, weight in targets:
            if token in item_text:
                bonus += weight
        return min(1.5, bonus)


    def rerank_items(
        self,
        query: str,
        preference_constraints: Dict[str, Any],
        candidate_items: List[Dict[str, Any]],
        top_n: int = 20,
        disable_prediction_bonus: bool = False,
    ) -> List[Dict[str, Any]]:
        self.load()
        if top_n <= 0:
            return []

        scored: List[Dict[str, Any]] = []
        for item in candidate_items:
            if self._must_avoid_filter(preference_constraints, item):
                continue

            prompt = self._build_scoring_prompt(query, preference_constraints, item)
            score_info = self._score_with_logits(prompt)
            prediction_bonus = 0.0 if disable_prediction_bonus else self._prediction_alignment_bonus(preference_constraints, item)
            enriched = dict(item)
            enriched["llm_weighted_score"] = score_info["weighted_score"]
            enriched["prediction_bonus"] = prediction_bonus
            enriched["ranking_score"] = float(score_info["weighted_score"] + prediction_bonus)
            enriched["score_probs"] = score_info["probs"]
            scored.append(enriched)

        scored.sort(
            key=lambda x: (
                float(x.get("ranking_score", 0.0)),
                float((x.get("score_probs") or {}).get("yes", 0.0)),
            ),
            reverse=True,
        )

        final_items: List[Dict[str, Any]] = []
        for rank, row in enumerate(scored[:top_n], start=1):
            row["rank"] = rank
            final_items.append(row)
        return final_items


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="LLM rerank candidate items from module-3 payload")
    parser.add_argument("input_json", help="JSON containing query/preference_constraints/candidate_items")
    parser.add_argument("--top-n", type=int, default=21)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--disable-prediction-bonus", action="store_true")
    args = parser.parse_args()

    payload = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    reranker = LLMItemReranker(model_name=args.model)
    results = reranker.rerank_items(
        query=str(payload.get("query", "")),
        preference_constraints=dict(payload.get("preference_constraints", {})),
        candidate_items=list(payload.get("candidate_items", [])),
        top_n=args.top_n,
        disable_prediction_bonus=bool(args.disable_prediction_bonus),
    )
    print(json.dumps(results, ensure_ascii=False, indent=2, default=str))

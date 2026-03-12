"""Module 3: Dynamic Reasoning & Ranking for Amazon pipeline.

This module consumes Agent-3 output from `intent_dual_recall_agent.py` and contains:
- Agent 4: Dynamic Preference Reasoner (LLM)
- Agent 5: Ranking & Scoring Agent (LLM)

LLM backbone: Qwen3-8B (text-only), following the same invocation style as Agent 3.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from reranker import LLMItemReranker

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:  # pragma: no cover
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None


@dataclass
class PreferenceConstraints:
    must_have: List[str]
    nice_to_have: List[str]
    must_avoid: List[str]
    reasoning: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "Must_Have": self.must_have,
            "Nice_to_Have": self.nice_to_have,
            "Must_Avoid": self.must_avoid,
            "Reasoning": self.reasoning,
        }


@dataclass
class Module3Output:
    user_id: str
    query: str
    preference_constraints: Dict[str, Any]
    ranked_items: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Qwen3DynamicReasonerLLM:
    """Qwen3 text LLM wrapper for Agent 4 reasoning."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-8B",
        max_new_tokens: int = 2048,
        enable_thinking: bool = True,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.enable_thinking = enable_thinking
        self._tokenizer = None
        self._model = None

    def load(self) -> None:
        if AutoTokenizer is None or AutoModelForCausalLM is None or torch is None:
            raise ImportError("transformers/torch are not available for Qwen3DynamicReasonerLLM.")
        if self._model is not None and self._tokenizer is not None:
            return
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto",
        )

    @staticmethod
    def _try_json_decode(text: str) -> Optional[Dict[str, Any]]:
        stripped = text.strip()
        try:
            payload = json.loads(stripped)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass

        if "```" in stripped:
            for part in stripped.split("```"):
                cand = part.replace("json", "", 1).strip()
                if not cand:
                    continue
                try:
                    payload = json.loads(cand)
                    if isinstance(payload, dict):
                        return payload
                except json.JSONDecodeError:
                    continue
        return None

    def infer_constraints(self, query: str, history_rows: List[Dict[str, Any]]) -> PreferenceConstraints:
        self.load()

        history_for_prompt = []
        for row in history_rows[:120]:
            history_for_prompt.append(
                {
                    "item_id": row.get("item_id"),
                    "behavior": row.get("behavior"),
                    "timestamp": row.get("timestamp"),
                    "taxonomy": (row.get("profile") or {}).get("taxonomy", {}),
                    "text_tags": (row.get("profile") or {}).get("text_tags", {}),
                    "visual_tags": (row.get("profile") or {}).get("visual_tags", {}),
                    "title": (row.get("profile") or {}).get("title", ""),
                }
            )

        guardrail_block = (
            "先决条件一致性（必须执行）：\n"
            "A) 必须先判断用户当前意图所属的商品类型与关键先决条件，再输出偏好。\n"
            "B) 任何与先决条件冲突的属性必须进入 Must_Avoid，且不得在 Must_Have/Nice_to_Have 中出现冲突项。\n"
            "C) 若用户query存在且历史行为与当前query冲突，以当前query的先决条件优先，历史仅作为风格/预算/题材补充。\n"
            "D) 禁止推荐跨平台或不兼容商品（例如 PC 游戏 vs PS/Xbox/Switch；iOS 配件 vs Android 专用）。\n"
            "E) 对人群敏感类目必须检查目标人群先决条件（例如服饰的性别/年龄段/尺码体系，婴幼儿用品的月龄阶段）。\n"
            "F) 对技术类商品必须检查兼容性先决条件（系统版本、接口/协议、功率/电压、尺寸规格）。\n"
            "G) 若信息不足以确认兼容性，必须在 Must_Avoid 中明确“避免不兼容/平台不符”，并在 Reasoning 写明不确定点。\n"
            "H) 输出前做一次一致性自检：Must_Have 与 Must_Avoid 不能互相矛盾，且所有 Must_Have 都必须满足先决条件。"
        )

        clean_query = (query or "").strip()
        if clean_query:
            prompt = (
                "你是电商推荐系统中的实时偏好建模专家（Agent4）。\n"
                "任务：根据用户当前query与相关历史正负行为，推理用户此刻偏好。\n"
                "要求：\n"
                "1) 明确区分 Must_Have / Nice_to_Have / Must_Avoid。\n"
                "2) 若历史中存在可分析的视觉信息（如 visual_tags/图片衍生描述），Nice_to_Have 必须包含视觉偏好结论；若无可分析视觉信息，则不要引用或臆造不存在的视觉信息。\n"
                "3) 必须结合 history 中 positive 与 negative 的对比证据。\n"
                "4) 先决条件必须优先于一般偏好，禁止输出与先决条件冲突的结论。\n"
                f"5) {guardrail_block}\n"
                "6) 输出严格 JSON 对象，字段：Must_Have(数组), Nice_to_Have(数组), Must_Avoid(数组), Reasoning(字符串)。\n\n"
                f"当前Query: {clean_query}\n"
                f"相关历史记录(JSON): {json.dumps(history_for_prompt, ensure_ascii=False)}"
            )
        else:
            prompt = (
                "你是电商推荐系统中的实时偏好建模专家（Agent4）。\n"
                "任务：当前没有可用query。请仅根据用户相关历史正负行为，推理用户此刻偏好。\n"
                "要求：\n"
                "1) 不要假设额外query意图，不要引用不存在的query信息。\n"
                "2) 明确区分 Must_Have / Nice_to_Have / Must_Avoid。\n"
                "3) 若历史中存在可分析的视觉信息（如 visual_tags/图片衍生描述），Nice_to_Have 必须包含视觉偏好结论；若无可分析视觉信息，则不要引用或臆造不存在的视觉信息。\n"
                "4) 必须结合 history 中 positive 与 negative 的对比证据。\n"
                "5) 先决条件必须优先于一般偏好，禁止输出与先决条件冲突的结论。\n"
                f"6) {guardrail_block}\n"
                "7) 输出严格 JSON 对象，字段：Must_Have(数组), Nice_to_Have(数组), Must_Avoid(数组), Reasoning(字符串)。\n\n"
                f"相关历史记录(JSON): {json.dumps(history_for_prompt, ensure_ascii=False)}"
            )

        messages = [{"role": "user", "content": prompt}]
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        model_inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)
        generated_ids = self._model.generate(**model_inputs, max_new_tokens=self.max_new_tokens)
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()

        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0
        content = self._tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

        payload = self._try_json_decode(content) or {}

        def _normalize_list(key: str) -> List[str]:
            vals = payload.get(key, [])
            if not isinstance(vals, list):
                vals = [vals]
            return [str(x).strip() for x in vals if str(x).strip()]

        return PreferenceConstraints(
            must_have=_normalize_list("Must_Have"),
            nice_to_have=_normalize_list("Nice_to_Have"),
            must_avoid=_normalize_list("Must_Avoid"),
            reasoning=str(payload.get("Reasoning", f"LLM raw output: {content[:800]}")),
        )


class DynamicPreferenceReasonerAgent:
    """Agent 4: infer structured dynamic constraints from recalled history."""

    def __init__(self, llm: Qwen3DynamicReasonerLLM) -> None:
        self.llm = llm

    def run(self, query: str, query_relevant_history: List[Dict[str, Any]]) -> PreferenceConstraints:
        return self.llm.infer_constraints(query=query, history_rows=query_relevant_history)


class RankingScoringAgent:
    """Agent 5: rank candidate items with five-level logits weighting."""

    def __init__(self, reranker: LLMItemReranker) -> None:
        self.reranker = reranker

    def run(
        self,
        query: str,
        preference_constraints: PreferenceConstraints,
        candidate_items: List[Dict[str, Any]],
        top_n: int = 20,
    ) -> List[Dict[str, Any]]:
        return self.reranker.rerank_items(
            query=query,
            preference_constraints=preference_constraints.to_dict(),
            candidate_items=candidate_items,
            top_n=top_n,
        )


def run_module3(
    intent_dual_recall_output: Dict[str, Any],
    model_name: str = "Qwen/Qwen3-8B",
    top_n: int = 20,
    disable_must_avoid: bool = False,
    save_output: bool = True,
    output_dir: str | Path = "./processed/dynamic_reasoning_ranking_outputs",
) -> Module3Output:
    """One-shot pipeline from Agent-3 output to module-3 final ranking."""

    query = str(intent_dual_recall_output.get("query", ""))
    user_id = str(intent_dual_recall_output.get("user_id", ""))
    candidate_items = list(intent_dual_recall_output.get("candidate_items", []))
    history_rows = list(intent_dual_recall_output.get("query_relevant_history", []))

    reasoner = DynamicPreferenceReasonerAgent(
        llm=Qwen3DynamicReasonerLLM(model_name=model_name)
    )
    constraints = reasoner.run(query=query, query_relevant_history=history_rows)
    if disable_must_avoid:
        constraints.must_avoid = []

    ranker = RankingScoringAgent(reranker=LLMItemReranker(model_name=model_name))
    ranked_items = ranker.run(
        query=query,
        preference_constraints=constraints,
        candidate_items=candidate_items,
        top_n=top_n,
    )

    output = Module3Output(
        user_id=user_id,
        query=query,
        preference_constraints=constraints.to_dict(),
        ranked_items=ranked_items,
    )

    if save_output:
        output_path = Path(output_dir) / f"user_{user_id}_dynamic_reasoning_ranking_output.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(output.to_dict(), ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run module-3 dynamic reasoning + ranking")
    parser.add_argument("agent3_output", help="Path to agent-3 output JSON")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--output-dir", default="./processed/dynamic_reasoning_ranking_outputs")
    args = parser.parse_args()

    payload = json.loads(Path(args.agent3_output).read_text(encoding="utf-8"))
    result = run_module3(
        intent_dual_recall_output=payload,
        model_name=args.model,
        top_n=args.top_n,
        output_dir=args.output_dir,
        save_output=True,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, default=str))

"""Adaptive text/VL weight modulation utilities for Agent3-style dual recall.

Design goals:
1) Use *relative rank quality* instead of only binary comparison.
2) React strongly when one modality is much better (extreme convergence).
3) Use trend (EMA momentum) so updates are not noisy one-step flips.
4) Keep logging compact; path_top5 is intentionally omitted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import math


@dataclass
class ModulationConfig:
    rank_power: float = 1.35
    temperature: float = 0.55
    base_step: float = 0.08
    confidence_scale: float = 0.30
    trend_scale: float = 0.18
    ema_beta: float = 0.65
    inertia: float = 0.25
    min_weight: float = 0.05
    max_weight: float = 0.95


@dataclass
class ModulationState:
    ema_advantage: float = 0.0


@dataclass
class RecallPlannerConfig:
    """Controls memory-driven total recall planning."""

    target_quantile: float = 0.75
    safety_margin: float = 1.12
    min_total_recall: int = 50
    max_total_recall: int = 5000


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _rank_to_quality(rank: int, power: float) -> float:
    r = max(1, int(rank))
    return 1.0 / (float(r) ** power)


def _softmax_pair(a: float, b: float, temperature: float) -> tuple[float, float]:
    tau = max(1e-6, float(temperature))
    ea = math.exp(a / tau)
    eb = math.exp(b / tau)
    z = ea + eb
    return ea / z, eb / z


def update_modal_weights(
    text_rank: int,
    vl_rank: int,
    text_weight: float,
    vl_weight: float,
    state: Optional[ModulationState] = None,
    cfg: Optional[ModulationConfig] = None,
) -> Dict[str, float]:
    """Update text/vl weights with confidence-aware + trend-aware dynamics.

    Returns a dict containing new weights and diagnostics.
    """
    cfg = cfg or ModulationConfig()
    state = state or ModulationState()

    # 1) Convert rank -> quality with non-linear gain.
    q_text = _rank_to_quality(text_rank, cfg.rank_power)
    q_vl = _rank_to_quality(vl_rank, cfg.rank_power)

    # 2) Target allocation from quality ratio (temperature controls extremeness).
    target_text, target_vl = _softmax_pair(q_text, q_vl, cfg.temperature)

    # 3) Advantage/confidence and EMA trend.
    advantage = target_text - target_vl
    confidence = abs(advantage)
    prev_ema = state.ema_advantage
    state.ema_advantage = cfg.ema_beta * prev_ema + (1.0 - cfg.ema_beta) * advantage
    trend = state.ema_advantage - prev_ema

    # 4) Adaptive step: larger when confidence/trend are large.
    step = cfg.base_step + cfg.confidence_scale * confidence + cfg.trend_scale * abs(trend)
    step = _clip(step, 0.02, 0.90)

    # 5) Pull current weights toward target; inertia keeps some memory.
    cur_text = float(text_weight) / (float(text_weight) + float(vl_weight) + 1e-12)
    cur_vl = 1.0 - cur_text
    blended_text = (1.0 - step) * cur_text + step * target_text
    blended_vl = (1.0 - step) * cur_vl + step * target_vl

    new_text = cfg.inertia * cur_text + (1.0 - cfg.inertia) * blended_text
    new_vl = cfg.inertia * cur_vl + (1.0 - cfg.inertia) * blended_vl

    # 6) Clamp to avoid hard collapse; re-normalize.
    new_text = _clip(new_text, cfg.min_weight, cfg.max_weight)
    new_vl = _clip(new_vl, cfg.min_weight, cfg.max_weight)
    z = new_text + new_vl
    new_text, new_vl = new_text / z, new_vl / z

    reason = (
        "text stronger" if advantage > 0.02 else "vl stronger" if advantage < -0.02 else "balanced"
    )

    return {
        "text_weight": round(new_text, 4),
        "vl_weight": round(new_vl, 4),
        "target_text_weight": round(target_text, 4),
        "target_vl_weight": round(target_vl, 4),
        "advantage": round(advantage, 4),
        "confidence": round(confidence, 4),
        "trend": round(trend, 4),
        "step_size": round(step, 4),
        "reasoning": reason,
    }


def build_compact_memory_row(
    step: int,
    target_item_id: str,
    text_rank: int,
    vl_rank: int,
    weights_before: Dict[str, float],
    update_output: Dict[str, float],
    recall_size: int,
) -> Dict[str, object]:
    """Build concise per-step memory row (without path_top5)."""
    return {
        "step": step,
        "target_item_id": target_item_id,
        "text_rank": int(text_rank),
        "vl_rank": int(vl_rank),
        "weights_before": {
            "text": round(float(weights_before.get("text", 0.5)), 4),
            "vl": round(float(weights_before.get("vl", 0.5)), 4),
        },
        "weights": {
            "text": update_output["text_weight"],
            "vl": update_output["vl_weight"],
        },
        "estimated_total_recall": int(recall_size),
        "confidence": update_output["confidence"],
        "trend": update_output["trend"],
        "step_size": update_output["step_size"],
        "reasoning": update_output["reasoning"],
    }


def _quantile(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    qq = _clip(float(q), 0.0, 1.0)
    pos = qq * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    alpha = pos - lo
    return sorted_vals[lo] * (1.0 - alpha) + sorted_vals[hi] * alpha


def estimate_total_recall_from_memory(
    memory_rows: List[Dict[str, object]],
    text_weight: float,
    vl_weight: float,
    cfg: Optional[RecallPlannerConfig] = None,
) -> Dict[str, object]:
    """Estimate a good total recall size and modality split from memory.

    Heuristic:
    - For one step, minimal K satisfying union recall under split weights is:
      min(text_rank / text_weight, vl_rank / vl_weight)
    - Aggregate over memory by quantile + safety margin.
    """
    cfg = cfg or RecallPlannerConfig()

    wt = max(1e-6, float(text_weight))
    wv = max(1e-6, float(vl_weight))
    z = wt + wv
    wt, wv = wt / z, wv / z

    required_k: List[float] = []
    for row in memory_rows:
        try:
            tr = max(1, int(row.get("text_rank", 0)))
            vr = max(1, int(row.get("vl_rank", 0)))
        except (TypeError, ValueError):
            continue

        k_text = tr / wt
        k_vl = vr / wv
        required_k.append(min(k_text, k_vl))

    if not required_k:
        total = _clip(500, cfg.min_total_recall, cfg.max_total_recall)
    else:
        required_k.sort()
        base = _quantile(required_k, cfg.target_quantile)
        total = int(math.ceil(base * cfg.safety_margin))
        total = int(_clip(total, cfg.min_total_recall, cfg.max_total_recall))

    text_topk = max(1, int(round(total * wt)))
    vl_topk = max(1, int(total - text_topk))

    return {
        "suggested_total_recall": total,
        "suggested_text_topk": text_topk,
        "suggested_vl_topk": vl_topk,
        "text_share": round(wt, 4),
        "vl_share": round(wv, 4),
        "sample_count": len(required_k),
        "method": (
            "quantile(min(text_rank/text_weight, vl_rank/vl_weight))"
            f" @q={cfg.target_quantile}, margin={cfg.safety_margin}"
        ),
    }

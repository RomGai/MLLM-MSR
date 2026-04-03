# 模块三：实时偏好推理与精排 (Dynamic Reasoning & Ranking)

实现文件：
- `dynamic_reasoning_ranking_agent.py`（Agent4 + Agent5 管线）
- `reranker.py`（纯 LLM 精排器，Qwen3-8B）

## Agent 4：实时偏好建模专家 (Dynamic Preference Reasoner - LLM)

输入：
- Agent 3 输出中的 `query_relevant_history`（含 `positive`/`negative`）
- 当前 `query`

职责：
- 对比历史正负样本中商品的 `taxonomy / text_tags / visual_tags`。
- 推理当前会话下用户核心诉求，输出结构化偏好条件：
  - `Must_Have`
  - `Nice_to_Have`（若有可分析视觉信息则必须包含视觉偏好结论；若无则不得臆造）
  - `Must_Avoid`
  - `Predicted_Next_Items`（严格 3 个预测，字段含 `item_type` / `likelihood` / `evidence`，且 likelihood 分别为 Most_Likely / Secondary / Possible）
  - `Reasoning`

LLM 调用方式与 Agent3 一致：`Qwen/Qwen3-8B + apply_chat_template(enable_thinking=True)`。

补充：当 `query` 为空时，Agent4 会自动切换到“无 query 专用提示词”，
仅基于相关历史正负行为推理偏好，避免因空 query 引导造成决策噪声。

时序说明：Agent4 会按时间顺序处理 positive 历史；negative 样本通常不具备可靠时序，因此不按其时间先后做序列推理，而作为反偏好证据使用。

## Agent 5：决策精排专家 (Ranking & Scoring Agent - LLM)

输入：
- Agent4 的偏好条件
- Agent3 的候选商品 `candidate_items`

职责：
1. 先执行 `Must_Avoid` 规则过滤（命中即剔除）。
2. 对每个候选商品构造评分 prompt。
3. 使用 Qwen3-8B **下一 token logits** 读取评分 token `1~5` 的概率。
4. 在五档分基础上，叠加与 `Predicted_Next_Items` 的对齐 bonus（Most_Likely > Secondary > Possible）。
5. 最终分数：`ranking_score = logits_weighted_score + prediction_bonus`。
   - `score = Σ(i * P(i))`, i∈{1,2,3,4,5}
6. 依分数降序排序并输出 Top-N。

> 该机制与原 `reranker.py` 中“五档打分 + logits 加权”保持一致，只是改为纯文本 LLM。

可选：传入 `--disable-prediction-bonus` 可关闭预测加分，仅保留 logits 加权分。

## 一体化调用

```python
from dynamic_reasoning_ranking_agent import run_module3
import json

agent3_output = json.load(open("./processed/intent_dual_recall_outputs/xxx.json", "r", encoding="utf-8"))
out = run_module3(agent3_output, model_name="Qwen/Qwen3-8B", top_n=20)
print(out.to_dict())
```

命令行：

```bash
python dynamic_reasoning_ranking_agent.py ./processed/intent_dual_recall_outputs/xxx.json --top-n 20
```

## Agent3 自适应模态调制（建议实现）

针对你提到的“仅比较 text_rank/vl_rank 谁更小就调权重”过于粗糙的问题，仓库里新增了一个可直接复用的工具：
- `adaptive_modal_modulation.py`

核心做法：
1. **Rank -> Quality 非线性映射**：`q=1/rank^γ`（`γ>1` 时，前排名次差异被放大）。
2. **Softmax 目标权重**：把 `(q_text,q_vl)` 通过温度 `τ` 转成 `(w*_text,w*_vl)`；`τ` 越小越容易收敛到极端。
3. **置信度与趋势联合步长**：
   - `confidence = |w*_text - w*_vl|`
   - 用 `EMA` 维护优势趋势，趋势越稳定，步长越大。
4. **惯性 + 限幅**：避免单步震荡和硬塌缩（例如限制在 `[0.05,0.95]`）。
5. **日志压缩**：`build_compact_memory_row` 默认不输出 `path_top5`，仅保留 rank、权重、confidence、trend、step_size 与原因。
6. **Top-K 规模估计**：可基于 memory 中历史 rank 反推总召回规模，新增 `estimate_total_recall_from_memory`：
   - 单步最小需求：`K_step = min(text_rank / w_text, vl_rank / w_vl)`
   - 多步聚合：取 `K_step` 的分位数（默认 75%）再乘安全系数（默认 1.12）
   - 输出建议：`suggested_total_recall`, `suggested_text_topk`, `suggested_vl_topk`

例如若长期观测接近 `text_rank≈100~200, vl_rank≈500~700`，且当前调制后权重大约 `text:vl=0.9:0.1`，该估计会自然倾向给出“中等总量 + 强文本侧配额”的策略，而不是固定死用 500。

> 若你希望“更极端”收敛：优先调小 `temperature`，并提高 `rank_power` 与 `confidence_scale`。

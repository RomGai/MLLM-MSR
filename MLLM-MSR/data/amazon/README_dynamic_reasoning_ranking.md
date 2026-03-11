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
  - `Nice_to_Have`（**必须包含视觉偏好结论**）
  - `Must_Avoid`
  - `Reasoning`

LLM 调用方式与 Agent3 一致：`Qwen/Qwen3-8B + apply_chat_template(enable_thinking=True)`。

## Agent 5：决策精排专家 (Ranking & Scoring Agent - LLM)

输入：
- Agent4 的偏好条件
- Agent3 的候选商品 `candidate_items`

职责：
1. 先执行 `Must_Avoid` 规则过滤（命中即剔除）。
2. 对每个候选商品构造评分 prompt。
3. 使用 Qwen3-8B **下一 token logits** 读取评分 token `1~5` 的概率。
4. 采用五档加权期望分数：
   - `score = Σ(i * P(i))`, i∈{1,2,3,4,5}
5. 依分数降序排序并输出 Top-N。

> 该机制与原 `reranker.py` 中“五档打分 + logits 加权”保持一致，只是改为纯文本 LLM。

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

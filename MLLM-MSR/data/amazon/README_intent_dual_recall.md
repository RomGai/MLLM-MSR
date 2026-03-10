# 模块二：意图理解与双路召回 (Intent & Dual Recall)

实现文件：`intent_dual_recall_agent.py`。

本模块直接承接模块一（`item_profiler_agents.py`）的两个数据库输出。

## 1. 先对模块一输出结构做对齐分析

### Global Item DB（全局商品特征库）
- 库表：`global_item_features(item_id, profile_json, updated_at)`
- 关键字段在 `profile_json` 中：
  - `taxonomy.item_type`（商品类型）
  - `taxonomy.category_path`（层级类目路径，数组）
  - 其他文本/视觉标签 (`text_tags`, `visual_tags`) 作为召回后可用特征

### User History Log DB（用户历史流数据库）
- 库表：`user_history_profiles(user_id, item_id, behavior, timestamp, profile_json, created_at)`
- 关键字段：
  - `behavior`（positive/negative）
  - `timestamp`（毫秒时间戳）
  - `profile_json` 中同样包含 `taxonomy.item_type`、`taxonomy.category_path`

> 因此模块二的“相关性主键”采用：`taxonomy.category_path` + `taxonomy.item_type`。

## 2. Agent 3（Routing & Recall Agent - LLM）设计

输入：
- `user_id`
- 用户实时 `query`

> 若 `query` 为空，Agent 3 会自动切换到“基于用户历史自主意图推断”模式。

### 职责 1：意图与层级映射
1. 从全局商品库扫描得到：
   - 全部 `category_path` 清单
   - 全部 `item_type` 清单
2. 把清单喂给 Qwen3（非 VL），让模型输出：
   - `category_paths`（二维数组）
   - `item_types`（数组）
   - `reasoning`
3. 若都不匹配，允许模型返回新造类目路径。

### 无 Query 时的兜底策略（历史自驱动）
当 `query` 为空时，不调用 LLM 路由，改为：
1. 读取该用户最近 `lookback` 条历史（默认 200）。
2. 优先使用 `positive` 行为记录；若没有，再回退到全部近期记录。
3. 按 `taxonomy.category_path` 与 `taxonomy.item_type` 做频次统计，取 top 类目/类型作为本次路由目标。
4. 路 B 在该模式下不再做相关性过滤，而是返回该用户近期**全部历史记录**（按时间倒序）。

### 职责 2：动态上卷双路召回

#### 路 A：全局商品召回
- 先按 LLM 选择的类目路径/类型精确召回。
- 若召回量 `< min_candidate_items`，对每条层级路径做“去掉最后一级”的上卷。
- 重复上卷直至数量满足或无法继续。

#### 路 B：用户历史精准召回
- 在 `user_history_profiles` 中按 `user_id` 拉取最近记录。
- 使用与路 A 相同的类目/类型匹配逻辑过滤。
- 返回 query 高相关历史交互记录（保留 `behavior` 和 `timestamp`）。

## 3. Qwen3 调用方式

代码遵循你提供的官方范式：
- `AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")`
- `AutoModelForCausalLM.from_pretrained(..., torch_dtype="auto", device_map="auto")`
- `apply_chat_template(..., enable_thinking=True)`
- `generate(max_new_tokens=...)`
- 解析 `</think>` 对应 token id `151668`

## 4. 模块输出

`RoutingRecallAgent.run(...)` 输出结构：
- `candidate_items`: 路 A 召回商品集合
- `query_relevant_history`: 路 B 历史记录（有 query 时为相关子集；无 query 时为全部历史）
- `routing`: 本次路由说明（选中类目、item_type、最终上卷层级）

其中 `routing.reasoning` 会明确说明本次是 `LLM 路由` 还是 `query 为空时的历史推断`。

可直接作为下一阶段排序/重排输入。
补充参数：
- `interested_item_types_k`（默认 `3`）：无论 query 是否为空，都会结合用户历史推断 top-k 兴趣 `item_type`，并与路由类型合并后用于路 A 商品召回。


## 5. 最小使用示例

```python
from intent_dual_recall_agent import Qwen3RouterLLM, GlobalHistoryAccessor, RoutingRecallAgent

llm = Qwen3RouterLLM(model_name="Qwen/Qwen3-8B")
accessor = GlobalHistoryAccessor(
    global_db_path="./processed/global_item_features.db",
    history_db_path="./processed/user_history_log.db",
)
agent = RoutingRecallAgent(llm=llm, accessor=accessor)

result = agent.run(
    user_id="123",
    query="想买适合两个人客厅联机的体感游戏",
    min_candidate_items=20,
)

print(result.candidate_items[:3])
print(result.query_relevant_history[:3])
```

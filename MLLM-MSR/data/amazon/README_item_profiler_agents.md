# Amazon Item Profiler Agents (Qwen3VL-8B)

本说明基于 `process_data.py` 处理后的数据格式，提供两个模块：

- Agent 1: `CandidateItemProfiler`（候选商品解析专家）
- Agent 2: `HistoryItemProfiler`（历史交互商品解析专家）

实现文件：`item_profiler_agents.py`。

## 输入数据与 `process_data.py` 对齐

- `*_item_desc.tsv`: `item_id, image, summary`
- `*_u_i_pairs.tsv`: `user_id, item_id, timestamp`
- `*_user_items_negs.tsv`: `user_id, pos, neg`

## 输出数据库

- 全局商品特征库（Global Item DB）
  - sqlite: `global_item_features.db`
  - 表：`global_item_features(item_id, profile_json, updated_at)`

- 用户历史流数据库（User History Log DB）
  - sqlite: `user_history_log.db`
  - 表：`user_history_profiles(user_id, item_id, behavior, timestamp, profile_json, created_at)`

## 关键能力

1. 细粒度文本标签抽取（类目、特征、规格、价格带、适用人群等）
2. 细粒度视觉风格抽取（色彩、版型、风格、材质质感、氛围感等）
3. 结构化 JSON 输出并落库
4. Agent 2 增强行为标签与时间戳

## 快速使用

```python
from item_profiler_agents import (
    bootstrap_agents_from_processed,
    ItemProfileInput,
    HistoryItemProfileInput,
)

candidate_profiler, history_profiler = bootstrap_agents_from_processed(
    item_desc_tsv="./processed/Video_Games_item_desc.tsv",
    global_db_path="./processed/global_item_features.db",
    history_db_path="./processed/user_history_log.db",
    model_name="Qwen/Qwen3-VL-8B-Instruct",
)

item = ItemProfileInput(
    item_id="42",
    title="Mechanical Keyboard 75%",
    detail_text="Gasket mount, hot-swappable, RGB, PBT keycaps.",
    main_image="./images/42_main.jpg",
    detail_images=["./images/42_side.jpg", "./images/42_detail.jpg"],
    price="$89.00",
    brand="ABC",
    category_hint="Electronics > Keyboards",
)
candidate_profiler.profile_and_store(item)

hist_item = HistoryItemProfileInput(
    **item.__dict__,
    user_id="1001",
    behavior="positive",
    timestamp=1710000000000,
)
history_profiler.profile_and_store(hist_item)
```

> 注：若环境无 GPU/模型权重，代码可完成模块搭建但无法执行真实推理。

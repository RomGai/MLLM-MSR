# LLaVA Amazon 推理脚本说明

脚本：`MLLM-MSR/test/amazon/test_with_llava_amazon.py`

## 你要的核心能力

- **单卡运行**：固定使用 `cuda:0`（没有 GPU 时自动回退到 CPU）
- **模型从 Hugging Face 直接下载**：默认 `llava-hf/llava-v1.6-mistral-7b-hf`
- **实时进度打印**：
  - 使用 `tqdm` 显示用户处理进度
  - 每处理完一个用户，都会打印一次**当前已处理用户的平均指标**（不是仅当前用户）
- **指标**：AUC、HR@K、MRR@K、NDCG@K
- **缺失图片处理**：默认会跳过 `image` 为空的候选；如需保留可加 `--keep-missing-image`
- **采样策略**：每个用户的正样本使用其历史序列中“最后一次交互 item”，其之前的历史用于偏好提示；负样本为随机采样 `1000` 个（可通过参数修改）

## 一条可直接运行的命令

```bash
python MLLM-MSR/test/amazon/test_with_llava_amazon.py \
  --item-desc-path MLLM-MSR/data/amazon/processed/Baby_Products_item_desc.tsv \
  --user-pairs-path MLLM-MSR/data/amazon/processed/Baby_Products_u_i_pairs.tsv \
  --test-negs-path MLLM-MSR/data/amazon/processed/Baby_Products_user_items_negs_test.csv \
  --output-dir MLLM-MSR/test/amazon/outputs_baby_llava \
  --num-negatives 1000 \
  --top-ks 20,40
```

## 数据构造方式

- `*_item_desc.tsv`：读取 item 的图片 URL 与文本描述
- `*_u_i_pairs.tsv`：读取用户历史（按 timestamp 排序）
- `*_user_items_negs_test.csv`：读取每个用户对应的候选集合（正/负样本）

脚本会为每个用户构建候选列表，对每个候选跑图文推理并写入 `predictions.jsonl`，最终输出 `metrics.json`。

## 与 `Inference/microlens` 的关系

该脚本**不依赖** `MLLM-MSR/Inference/microlens` 目录中的代码。
Amazon `processed` 数据已经足够支持从样本构建、推理到统计的完整流程。

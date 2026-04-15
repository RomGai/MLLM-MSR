#!/usr/bin/env bash
set -euo pipefail

# One-command pipeline for:
# 1) dataset adaptation
# 2) LoRA SFT training
# 3) rank-1000 inference

PROCESSED_DIR="MLLM-MSR/data/amazon/processed"
DOMAIN="Baby_Products"
DATASET_OUT="MLLM-MSR/train/microlens/amazon_mllmrec_r1_dataset"
MODEL_ID="llava-hf/llava-v1.6-mistral-7b-hf"
TRAIN_OUTPUT_DIR="MLLM-MSR/train/microlens/outputs_llava_lora"
EPOCHS=1
DEVICES=1
TRAIN_BATCH_SIZE=1
INFER_BATCH_SIZE=8
SAMPLE_SIZE=1000
SEED=2026
MAX_USERS=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --processed-dir) PROCESSED_DIR="$2"; shift 2 ;;
    --domain) DOMAIN="$2"; shift 2 ;;
    --dataset-out) DATASET_OUT="$2"; shift 2 ;;
    --model-id) MODEL_ID="$2"; shift 2 ;;
    --train-output-dir) TRAIN_OUTPUT_DIR="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --devices) DEVICES="$2"; shift 2 ;;
    --train-batch-size) TRAIN_BATCH_SIZE="$2"; shift 2 ;;
    --infer-batch-size) INFER_BATCH_SIZE="$2"; shift 2 ;;
    --sample-size) SAMPLE_SIZE="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --max-users) MAX_USERS="$2"; shift 2 ;;
    -h|--help)
      cat <<'EOF'
Usage: bash run_mllmrec_r1_pipeline.sh [options]

Options:
  --processed-dir <path>      processed data dir (default: MLLM-MSR/data/amazon/processed)
  --domain <name>             dataset domain prefix (default: Baby_Products)
  --dataset-out <path>        built HF dataset output dir
  --model-id <hf_model_id>    backbone model id from HuggingFace
  --train-output-dir <path>   LoRA checkpoint output dir
  --epochs <int>              training epochs
  --devices <int>             trainer devices
  --train-batch-size <int>    training batch size
  --infer-batch-size <int>    inference scoring batch size
  --sample-size <int>         number of random negatives per user (default: 1000)
  --seed <int>                random seed
  --max-users <int>           max users for inference (0 means all)
EOF
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown option: $1"
      exit 1
      ;;
  esac
done

echo "[Pipeline] Step 1/3 - Build dataset"
python MLLM-MSR/train/microlens/build_mllmrec_r1_dataset.py \
  --processed-dir "$PROCESSED_DIR" \
  --domain "$DOMAIN" \
  --seed "$SEED" \
  --output "$DATASET_OUT"

echo "[Pipeline] Step 2/3 - Train LoRA SFT"
python MLLM-MSR/train/microlens/train_llava_sft_adapted.py \
  --dataset-path "$DATASET_OUT" \
  --model-id "$MODEL_ID" \
  --output-dir "$TRAIN_OUTPUT_DIR" \
  --epochs "$EPOCHS" \
  --devices "$DEVICES" \
  --batch-size "$TRAIN_BATCH_SIZE"

PEFT_MODEL_ID="$TRAIN_OUTPUT_DIR/epoch_${EPOCHS}"
echo "[Pipeline] Step 3/3 - Rank-1000 inference using: $PEFT_MODEL_ID"
python MLLM-MSR/test/microlens/test_with_llava_sft_rank1000.py \
  --processed-dir "$PROCESSED_DIR" \
  --domain "$DOMAIN" \
  --model-id "$MODEL_ID" \
  --peft-model-id "$PEFT_MODEL_ID" \
  --sample-size "$SAMPLE_SIZE" \
  --batch-size "$INFER_BATCH_SIZE" \
  --seed "$SEED" \
  --max-users "$MAX_USERS"

echo "[Pipeline] Done."

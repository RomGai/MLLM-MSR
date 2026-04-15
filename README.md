# MLLM-MSR
![framework (3)](https://github.com/user-attachments/assets/810ac195-3b6e-41a6-9717-f1e8d72b552f)

The code for the paper "Harnessing Multimodal Large Language Models for Multimodal Sequential Recommendation" (Accepted by AAAI-25).

## Dataset
This paper utilizes the following datasets:
- **Microlens Dataset**: [GitHub Repository](https://github.com/westlake-repl/MicroLens)
- **Amazon Review Dataset**: [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/#grouped-by-category)

The data processing scripts have been uploaded for preprocessing and structuring the datasets for model training and inference.

## Steps to Run

### 1. Inference
- First, generate summaries for item images using:
  ```bash
  python Inference/microlens/image_summary.py
  ```
- Next, obtain user preference information using:
  ```bash
  python Inference/microlens/preferece_inference_recurrent.py
  ```

### 2. Dataset Preparation
Before training or testing, datasets must be constructed:
- For training dataset creation:
  ```bash
  python MLLM-MSR/train/dataset_create.py
  ```
- For test dataset creation:
  ```bash
  python MLLM-MSR/test/multi_col_dataset.py
  ```

### 3. Training the Recommender Model
Use the following script to perform supervised fine-tuning (SFT) of the recommender model:
  ```bash
  python MLLM-MSR/train/train_llava_sft.py
  ```

### 4. Testing the Model
To evaluate the trained recommender model:
  ```bash
  python MLLM-MSR/test/test_with_llava_sft.py
  ```

## Citation

If you use the code of this repo, please cite our paper as,

```bibtex
@inproceedings{ye2025harnessing,
  title={Harnessing multimodal large language models for multimodal sequential recommendation},
  author={Ye, Yuyang and Zheng, Zhi and Shen, Yishan and Wang, Tianshu and Zhang, Hengruo and Zhu, Peijun and Yu, Runlong and Zhang, Kai and Xiong, Hui},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={12},
  pages={13069--13077},
  year={2025}
}

## MLLMRec-R1 Adapted Workflow (Amazon processed format)

1. **Data adaptation**: use files containing `_train` and `_test` in filename to split users and build a HuggingFace train/validation dataset.
2. **SFT training**: load the backbone model directly from HuggingFace and train LoRA adapters.
3. **Final ranking inference**: for each test user, rank **1 target + 1000 random sampled negatives**, report target rank and running average HR/NDCG after each user.

### Run commands (training + final inference)

```bash
# 1) Build training dataset from processed data
python MLLM-MSR/train/microlens/build_mllmrec_r1_dataset.py \
  --processed-dir MLLM-MSR/data/amazon/processed \
  --domain Baby_Products \
  --output MLLM-MSR/train/microlens/amazon_mllmrec_r1_dataset

# 2) Train (backbone loaded from HuggingFace, LoRA saved locally)
python MLLM-MSR/train/microlens/train_llava_sft_adapted.py \
  --dataset-path MLLM-MSR/train/microlens/amazon_mllmrec_r1_dataset \
  --model-id llava-hf/llava-v1.6-mistral-7b-hf \
  --output-dir MLLM-MSR/train/microlens/outputs_llava_lora \
  --epochs 1 --devices 1

# 3) Final inference with 1 target + 1000 sampled candidates
python MLLM-MSR/test/microlens/test_with_llava_sft_rank1000.py \
  --processed-dir MLLM-MSR/data/amazon/processed \
  --domain Baby_Products \
  --model-id llava-hf/llava-v1.6-mistral-7b-hf \
  --peft-model-id MLLM-MSR/train/microlens/outputs_llava_lora/epoch_1
```

Or run everything with one command:

```bash
bash run_mllmrec_r1_pipeline.sh --domain Baby_Products
```

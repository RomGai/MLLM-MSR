import argparse
import os
import lightning as L
import numpy as np
import torch
from PIL import ImageOps
from datasets import load_from_disk
from lightning.pytorch.callbacks import Callback
from torch.utils.data import DataLoader
from transformers import AutoProcessor, LlavaNextForConditionalGeneration
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

from load_llava_dataset import LlavaDataset2


class ProgressCallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        print(f"[Train] ===== Epoch {trainer.current_epoch + 1}/{trainer.max_epochs} started =====")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = outputs.detach().float().item() if torch.is_tensor(outputs) else float(outputs)
        print(f"[Train] epoch={trainer.current_epoch + 1} batch={batch_idx + 1} loss={loss:.6f}")


class SaveLoRACallback(Callback):
    def __init__(self, output_dir):
        super().__init__()
        self.output_dir = output_dir

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.global_rank == 0:
            epoch_dir = os.path.join(self.output_dir, f"epoch_{trainer.current_epoch + 1}")
            os.makedirs(epoch_dir, exist_ok=True)
            pl_module.model.save_pretrained(epoch_dir)
            pl_module.processor.save_pretrained(epoch_dir)
            print(f"[Train] saved checkpoint to: {epoch_dir}")


class LlavaModelPLModule(L.LightningModule):
    def __init__(self, model, processor, train_dataset, val_dataset, lr, batch_size, max_length):
        super().__init__()
        self.model = model
        self.processor = processor
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.lr = lr
        self.batch_size = batch_size
        self.max_length = max_length

    @staticmethod
    def resize_images(images):
        max_width = max(img.width for img in images)
        max_height = max(img.height for img in images)
        out = []
        for img in images:
            if img.width == max_width and img.height == max_height:
                out.append(img)
                continue
            dw, dh = max_width - img.width, max_height - img.height
            padding = (dw // 2, dh // 2, dw - dw // 2, dh - dh // 2)
            out.append(ImageOps.expand(img, border=padding, fill="black"))
        return out

    def collate_train(self, examples):
        images, texts = [], []
        for image, prompt_text, ground_truth in examples:
            images.append(image)
            texts.append(f"[INST] <image>\n{prompt_text} [/INST] {ground_truth}")
        images = self.resize_images(images)
        batch = self.processor(text=texts, images=images, padding=True, truncation=True,
                               max_length=self.max_length, return_tensors="pt")
        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        return batch

    def collate_eval(self, examples):
        images, texts, answers = [], [], []
        for image, prompt_text, ground_truth in examples:
            images.append(image)
            texts.append(f"[INST] <image>\n{prompt_text} [/INST]")
            answers.append(ground_truth)
        images = self.resize_images(images)
        batch = self.processor(text=texts, images=images, padding=True, return_tensors="pt")
        batch["answers"] = answers
        return batch

    def training_step(self, batch, batch_idx):
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"],
            image_sizes=batch["image_sizes"],
            labels=batch["labels"],
        )
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        generated_ids = self.model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"],
            image_sizes=batch["image_sizes"],
            max_new_tokens=10,
        )
        predictions = self.processor.batch_decode(generated_ids[:, batch["input_ids"].size(1):], skip_special_tokens=True)
        answers = batch["answers"]
        acc = np.mean([int(p.strip().lower().startswith(a.strip().lower())) for p, a in zip(predictions, answers)])
        self.log("val_acc", float(acc), prog_bar=True)
        print(f"[Val] batch={batch_idx + 1} acc={acc:.4f}")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2, collate_fn=self.collate_train)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2, collate_fn=self.collate_eval)


def find_all_linear_names(model):
    lora_module_names = set()
    for name, module in model.named_modules():
        if any(k in name for k in ["multi_modal_projector", "vision_model"]):
            continue
        if isinstance(module, torch.nn.Linear):
            lora_module_names.add(name.split(".")[-1])
    lora_module_names.discard("lm_head")
    return sorted(lora_module_names)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", default="MLLM-MSR/train/microlens/amazon_mllmrec_r1_dataset")
    parser.add_argument("--model-id", default="llava-hf/llava-v1.6-mistral-7b-hf")
    parser.add_argument("--output-dir", default="MLLM-MSR/train/microlens/outputs_llava_lora")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulate-grad-batches", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--devices", type=int, default=1)
    args = parser.parse_args()

    print("[Step 1/6] Loading processor from HuggingFace backbone model.")
    processor = AutoProcessor.from_pretrained(args.model_id)
    processor.tokenizer.padding_side = "right"

    print("[Step 2/6] Loading dataset from disk.")
    _ = load_from_disk(args.dataset_path)
    train_dataset = LlavaDataset2(args.dataset_path, split="train", sort_json_key=False)
    val_dataset = LlavaDataset2(args.dataset_path, split="validation", sort_json_key=False)
    print(f"Train samples={len(train_dataset)}, Validation samples={len(val_dataset)}")

    print("[Step 3/6] Loading backbone model directly from HuggingFace.")
    model = LlavaNextForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        _attn_implementation="flash_attention_2",
    )

    print("[Step 4/6] Preparing LoRA modules.")
    targets = find_all_linear_names(model)
    lora_config = LoraConfig(r=args.lora_r, lora_alpha=32, lora_dropout=0.1, target_modules=targets, init_lora_weights="gaussian")
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    print("[Step 5/6] Building Lightning module.")
    module = LlavaModelPLModule(model, processor, train_dataset, val_dataset, args.lr, args.batch_size, args.max_length)

    print("[Step 6/6] Starting training.")
    trainer = L.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.devices,
        max_epochs=args.epochs,
        accumulate_grad_batches=args.accumulate_grad_batches,
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
        log_every_n_steps=1,
        callbacks=[ProgressCallback(), SaveLoRACallback(args.output_dir)],
    )
    trainer.fit(module)
    print("[Train] completed.")


if __name__ == "__main__":
    main()

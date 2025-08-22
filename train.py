#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

def get_device() -> torch.device:
    try:
        import torch_directml  # type: ignore
        dml = torch_directml.device()
        print(f"✅ DirectML device found: {dml}")
        return dml
    except Exception:
        pass
    if torch.cuda.is_available():
        print(f"✅ CUDA device found: {torch.device('cuda')}")
        return torch.device("cuda")
    print("⚠️  No GPU found – using CPU.")
    return torch.device("cpu")

def get_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32

class TextDatasetGenerator:
    def __init__(self, tokenizer: AutoTokenizer, max_length: int = 512, overlap: int = 200, chunk_size_chars: int = 4096):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.overlap = overlap
        self.chunk_size_chars = chunk_size_chars
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    def _read_in_chunks(self, file_path: str):
        with open(file_path, "r", encoding="utf-8") as f:
            while True:
                chunk = f.read(self.chunk_size_chars)
                if not chunk:
                    break
                yield chunk

    def _tokenize_window(self, window: str):
        return self.tokenizer.encode(window, add_special_tokens=False, truncation=False)

    def _chunk_token_ids(self, token_ids: list[int]):
        start = 0
        while start < len(token_ids):
            end = start + self.max_length
            chunk_ids = token_ids[start:end]
            yield chunk_ids
            start = max(end - self.overlap, 0)

    def prepare_dataset(self, file_path: str) -> Dataset:
        if not Path(file_path).is_file():
            raise FileNotFoundError(f"Input file not found: {file_path}")
        token_chunks = []
        overlap_buffer: list[int] = []
        for raw_window in self._read_in_chunks(file_path):
            new_ids = self._tokenize_window(raw_window)
            if overlap_buffer:
                new_ids = overlap_buffer + new_ids
            for chunk_ids in self._chunk_token_ids(new_ids):
                token_chunks.append(chunk_ids)
            overlap_buffer = new_ids[-self.overlap :] if self.overlap > 0 else []
        padded = self.tokenizer.pad(
            {"input_ids": token_chunks},
            padding="max_length",
            max_length=self.max_length,
            return_attention_mask=True,
            return_tensors=None,
        )
        records = [
            {"input_ids": ids, "attention_mask": mask, "labels": ids}
            for ids, mask in zip(padded["input_ids"], padded["attention_mask"])
        ]
        return Dataset.from_dict(
            {
                "input_ids": [r["input_ids"] for r in records],
                "attention_mask": [r["attention_mask"] for r in records],
                "labels": [r["labels"] for r in records],
            }
        )

def setup_lora(model):
    from peft import LoraConfig, get_peft_model
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    return model, lora_cfg

def save_full_model(model, tokenizer, output_dir: str, lora_cfg=None, lora_adapter_dir=None):
    from peft import PeftModel
    os.makedirs(output_dir, exist_ok=True)
    if lora_cfg is not None and lora_adapter_dir is not None:
        base = AutoModelForCausalLM.from_pretrained(
            lora_cfg.base_model_name_or_path,
            torch_dtype=model.dtype,
            low_cpu_mem_usage=True,
        )
        merged = PeftModel.from_pretrained(base, lora_adapter_dir)
        merged = merged.merge_and_unload()
        merged.save_pretrained(output_dir, safe_serialization=True)
        tokenizer.save_pretrained(output_dir)
        print(f"✅ Full merged model saved to {output_dir}")
    else:
        model.save_pretrained(output_dir, safe_serialization=True)
        tokenizer.save_pretrained(output_dir)
        print(f"✅ Model (no LoRA) saved to {output_dir}")

def run_training(args):
    device = get_device()
    dtype = get_dtype(device)

    print("🚀 Loading tokenizer & model …")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)

    print("📚 Preparing dataset …")
    ds_gen = TextDatasetGenerator(tokenizer, max_length=args.max_length)
    dataset = ds_gen.prepare_dataset(args.input_file)

    # ----------------- Повторяем датасет -----------------
    # Позволяет получить больше градиентных шагов без добавления новых данных.
    # По‑умолчанию повторяем 8 раз, но значение можно переопределить через CLI:
    #   --repeat_factor N
    repeat_factor = getattr(args, "repeat_factor", 8)
    if repeat_factor > 1:
        dataset = concatenate_datasets([dataset] * repeat_factor)
        print(f"🔁 Dataset repeated {repeat_factor} times → {len(dataset)} examples")

    # ----------------- Инфо о датасете -----------------
    print(f"🧩 Dataset size: {len(dataset)} examples")
    print(f"🔢 Example (ids): {dataset[0]['input_ids'][:10]} …")

    print("🪄 Adding LoRA …")
    model, lora_cfg = setup_lora(model)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        logging_steps=5,
        save_steps=100,
        eval_steps=100,
        save_total_limit=1,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        optim="adamw_torch",
        fp16=(dtype == torch.float16),
        bf16=(dtype == torch.bfloat16),
        report_to=[],
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n⚡ Training interrupted by user.")
    except Exception as exc:
        print(f"❌ Training failed: {exc}")
        raise

    lora_dir = os.path.join(args.output_dir, "lora_adapter")
    os.makedirs(lora_dir, exist_ok=True)
    trainer.model.save_pretrained(lora_dir)
    tokenizer.save_pretrained(lora_dir)
    print(f"💾 LoRA adapter saved to {lora_dir}")

    if args.save_full_model:
        full_dir = os.path.join(args.output_dir, "full_merged")
        save_full_model(trainer.model, tokenizer, full_dir, lora_cfg, lora_dir)

        meta = {
            "model_name": args.model_name,
            "device": str(device),
            "dtype": str(dtype),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "max_length": args.max_length,
            "lora_config": lora_cfg.to_dict(),
        }
        with open(os.path.join(full_dir, "training_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        print(f"🏁 Full merged model ready at {full_dir}")

    print("\n✅ Training finished!")

def generate_text(
    model_path: str,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
):
    device = get_device()
    dtype = get_dtype(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    try:
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        model = PeftModel.from_pretrained(base, model_path)
        print("🔌 Loaded LoRA adapter.")
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        print("⚙️ Loaded plain model (no LoRA).")
    model.to(device)
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(generated[0], skip_special_tokens=True)

def main():
    parser = argparse.ArgumentParser(
        description="Fine‑tune a causal LLM on a single Russian text file (AMD DirectML / CUDA / CPU)."
    )
    parser.add_argument("--input_file", type=str, required=True, help="Path to the plain‑text file.")
    parser.add_argument("--output_dir", type=str, default="./fine_tuned_model", help="Output directory.")
    parser.add_argument("--model_name", type=str, default="microsoft/Phi-3-mini-4k-instruct", help="HF model identifier.")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=1, help="Per‑device training batch size.")
    parser.add_argument("--gradient_accumulation", type=int, default=2, help="Gradient accumulation steps.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum token length per chunk.")
    parser.add_argument("--save_full_model", action="store_true", help="Save merged model.")
    parser.add_argument("--generate", action="store_true", help="Run generation demo after training.")
    parser.add_argument("--prompt", type=str, default="Агрессия останавливается только там, где сталкивается с встречной агрессией", help="Prompt for generation.")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Number of tokens to generate.")
    parser.add_argument(
        "--repeat_factor",
        type=int,
        default=8,
        help="How many times to repeat the whole dataset (to get more training steps).",
    )
    args = parser.parse_args()

    run_training(args)

    if args.generate:
        if args.save_full_model:
            model_path = os.path.join(args.output_dir, "full_merged")
        else:
            model_path = os.path.join(args.output_dir, "lora_adapter")
        print("\n🧪 Running generation demo …")
        try:
            out = generate_text(
                model_path=model_path,
                prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
            )
            print("\n=== GENERATED TEXT ===")
            print(out)
            print("======================")
        except Exception as e:
            print(f"❌ Generation failed: {e}")

if __name__ == "__main__":
    main()
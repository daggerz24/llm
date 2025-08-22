#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple script: take a random piece of `data.txt` as a prompt,
run it through a fine‑tuned LLM (merged model or LoRA adapter)
and print the result.
No pytest – just a runnable script you can execute from the console.
"""

import os
import pathlib
import random
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ----------------------------------------------------------------------
# 1️⃣  Device helpers (same as in train.py)
# ----------------------------------------------------------------------
def get_device() -> torch.device:
    """Return the best available device (DirectML > CUDA > CPU)."""
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
    """Pick a dtype that the device supports."""
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    # DirectML or CPU – stay in fp32
    return torch.float32


# ----------------------------------------------------------------------
# 2️⃣  Load model / tokenizer
# ----------------------------------------------------------------------
def load_model_and_tokenizer(model_dir: str, device: torch.device, dtype: torch.dtype):
    """
    Try to load a merged model first. If that fails, fall back to a LoRA adapter.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)

    # 1) try merged model
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        print("✅ Loaded merged model.")
        return model, tokenizer
    except Exception as e:
        print(f"⚠️  Merged model load failed ({e}), trying LoRA adapter…")

    # 2) try LoRA adapter (PEFT)
    try:
        from peft import PeftModel

        base = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        model = PeftModel.from_pretrained(base, model_dir)
        print("✅ Loaded LoRA adapter.")
        return model, tokenizer
    except Exception as e2:
        print(f"❌ Failed to load any model from {model_dir}: {e2}")
        sys.exit(1)


# ----------------------------------------------------------------------
# 3️⃣  Pick a prompt from data.txt
# ----------------------------------------------------------------------
def pick_random_prompt(data_path: str, tokenizer, max_prompt_tokens: int = 80) -> str:
    """
    Return a random fragment of the source text whose token length
    does not exceed ``max_prompt_tokens``.
    """
    with open(data_path, "r", encoding="utf-8") as f:
        raw = f.read()

    # Split into paragraphs (double newline) – fallback to single lines
    fragments = [p.strip() for p in raw.split("\n\n") if p.strip()]
    if len(fragments) < 5:
        fragments = [l.strip() for l in raw.split("\n") if l.strip()]

    # Choose repeatedly until we find a short enough fragment
    while True:
        cand = random.choice(fragments)
        toks = tokenizer(cand, add_special_tokens=False, return_tensors="pt")
        if toks["input_ids"].shape[1] <= max_prompt_tokens:
            return cand


# ----------------------------------------------------------------------
# 4️⃣  Generation
# ----------------------------------------------------------------------
def generate(model, tokenizer, device, prompt: str, max_new_tokens: int = 120):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(generated[0], skip_special_tokens=True)


# ----------------------------------------------------------------------
# 5️⃣  Main entry point
# ----------------------------------------------------------------------
def main():
    # ------------------------------------------------------------------
    # Paths – adjust if your layout differs
    # ------------------------------------------------------------------
    model_dir = "./my_finetuned/lora_adapter"   # merged model directory
    data_path = "./data.txt"                   # source corpus

    # ------------------------------------------------------------------
    # Device / dtype
    # ------------------------------------------------------------------
    device = get_device()
    dtype = get_dtype(device)

    # ------------------------------------------------------------------
    # Load model + tokenizer
    # ------------------------------------------------------------------
    model, tokenizer = load_model_and_tokenizer(model_dir, device, dtype)
    model.to(device)

    # ------------------------------------------------------------------
    # Choose a prompt
    # ------------------------------------------------------------------
    prompt = pick_random_prompt(data_path, tokenizer, max_prompt_tokens=80)
    print("\n=== PROMPT FROM DATASET ===")
    print(prompt)
    print("=" * 30)

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------
    answer = generate(model, tokenizer, device, prompt, max_new_tokens=120)

    print("\n=== GENERATED RESPONSE ===")
    # The generated text usually contains the prompt at the beginning – we can cut it off
    # if you want only the continuation:
    if answer.startswith(prompt):
        answer = answer[len(prompt) :].lstrip()
    print(answer)
    print("=" * 30)



main()
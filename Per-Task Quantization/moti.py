#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Layer-wise Motivation Figure for Llama 3.1 8B Instruct
- Metrics:
    (i) H_l : matrix-based Shannon entropy of Gram(K = XX^T) from hidden tokens
   (ii) S_l : stability proxy via negative mean per-dimension variance across tokens
   (iii) r_l: task-aware rank = alpha*z(H_l) + beta*z(S_l)
- Tasks: trivia (TriviaQA rc.nocontext), math (GSM8K), code (MBPP)
- Output: PNG figure + JSON with raw metrics

Usage:
  export HUGGING_FACE_HUB_TOKEN=YOUR_TOKEN
  python layer_motivation_llama31_variance.py --out-dir outputs

Requires:
  pip install -U "transformers==4.51.3" "torch==2.6.*" datasets accelerate matplotlib
"""

import os
import math
import json
import random
import argparse
from typing import List, Dict, Tuple
import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# -----------------------------
# Utils
# -----------------------------
def get_num_layers_llama(model) -> int:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return len(model.model.layers)
    if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
        return len(model.model.decoder.layers)
    return 0

def znormalize(xs: List[float]) -> List[float]:
    m = sum(xs)/len(xs)
    v = sum((x-m)*(x-m) for x in xs)/max(1, len(xs)-1)
    s = math.sqrt(v + 1e-12)
    return [(x - m) / (s + 1e-12) for x in xs]

@torch.no_grad()
def layer_entropy_and_stability(
    texts: List[str],
    model,
    tokenizer,
    max_tokens: int,
    batch_size: int,
    reservoir_per_layer: int,
    device: str
) -> Tuple[List[float], List[float]]:
    """
    Returns (H_list, S_list) each of length n_layers, where:
      H_l: matrix-based Shannon entropy of Gram(XX^T) from layer-l tokens
      S_l: stability proxy = - mean_j Var(X_{:, j}) across tokens (lower variance => higher stability)
    """
    def entropy_from_hidden(X: torch.Tensor) -> float:
        # X: [N, D] tokens x dims (non-padding)
        if X.shape[0] < 2:
            return 0.0
        N = X.shape[0]
        R = min(reservoir_per_layer, N)
        if R < N:
            idx = torch.randint(0, N, (R,), device=X.device)
            X = X.index_select(0, idx)
        X = X.to(torch.float32)
        X = X - X.mean(dim=0, keepdim=True)
        G = (X @ X.T) / max(1, X.shape[0])
        evals = torch.linalg.eigvalsh(G).clamp(min=0.0)
        s = float(evals.sum().item())
        if s <= 0:
            return 0.0
        p = (evals / s).double()
        H = float((-(p * (p + 1e-12).log())).sum().item())
        return H

    def stability_from_hidden(X: torch.Tensor) -> float:
        # negative mean per-dim variance across tokens
        if X.shape[0] < 2:
            return 0.0
        X = X.to(torch.float32)
        var_per_dim = X.var(dim=0, unbiased=True)  # [D]
        return -float(var_per_dim.mean().item())

    nL = get_num_layers_llama(model)
    H_sums = [0.0 for _ in range(nL)]
    H_counts = [0 for _ in range(nL)]
    S_sums = [0.0 for _ in range(nL)]
    S_counts = [0 for _ in range(nL)]

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start+batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_tokens).to(device)
        out = model(**enc, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states     # tuple length nL+1
        attn = enc.get("attention_mask", None)
        keep_idx = None
        if attn is not None:
            keep_idx = attn.reshape(-1).to(dtype=torch.bool).nonzero(as_tuple=False).squeeze(-1)

        for li in range(1, len(hs)):  # skip embeddings at index 0
            H_l = hs[li]             # [B, T, D]
            X = H_l.reshape(-1, H_l.shape[-1])
            if keep_idx is not None and keep_idx.numel() > 0 and keep_idx.max().item() < X.shape[0]:
                X = X.index_select(0, keep_idx)

            h_val = entropy_from_hidden(X)
            s_val = stability_from_hidden(X)

            H_sums[li-1] += h_val; H_counts[li-1] += 1
            S_sums[li-1] += s_val; S_counts[li-1] += 1

        del out, hs, enc
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    H_means = [H_sums[i] / max(1, H_counts[i]) for i in range(nL)]
    S_means = [S_sums[i] / max(1, S_counts[i]) for i in range(nL)]
    return H_means, S_means

def load_texts_for_task(task: str, k: int) -> List[str]:
    if task == "trivia":
        ds = load_dataset("trivia_qa", "rc.nocontext", split="validation")
        idx = list(range(len(ds))); random.shuffle(idx); idx = idx[:k]
        return [f"Question: {ds[i]['question']}\nAnswer:" for i in idx]
    if task == "math":
        ds = load_dataset("gsm8k", "main", split="test")
        idx = list(range(len(ds))); random.shuffle(idx); idx = idx[:k]
        return [f"Solve step by step.\nProblem: {ds[i]['question']}\nAnswer:" for i in idx]
    if task == "code":
        try:
            ds = load_dataset("mbpp", split="test")
        except Exception:
            ds = load_dataset("mbpp", split="train")
        idx = list(range(len(ds))); random.shuffle(idx); idx = idx[:k]
        def to_prompt(ex):
            q = ex.get("text") or ex.get("prompt") or ex.get("task_id") or ""
            return f"Write a Python function as specified.\nTask: {q}\nCode:"
        return [to_prompt(ds[i]) for i in idx]
    raise ValueError(f"Unknown task: {task}")

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Layer-wise motivation plot: entropy + variance/stability")
    ap.add_argument("--model-id", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--device", type=str, default="auto", help="auto|cuda|cpu|cuda:0...")
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["float16","bfloat16","float32"])
    ap.add_argument("--max-texts", type=int, default=128, help="per task")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--reservoir", type=int, default=256, help="tokens per layer for entropy")
    ap.add_argument("--alpha", type=float, default=1.0, help="weight for z(H_l)")
    ap.add_argument("--beta", type=float, default=1.0, help="weight for z(S_l)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-dir", type=str, default="outputs")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = (
        "cuda:0" if (args.device == "auto" and torch.cuda.is_available()) else
        ("cpu" if args.device == "auto" else args.device)
    )
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    os.makedirs(args.out_dir, exist_ok=True)
    png_path = os.path.join(args.out_dir, "fig_motivation_llama31_8b_instruct.png")
    json_path = os.path.join(args.out_dir, "fig_motivation_llama31_8b_instruct.json")

    # Load model/tokenizer
    hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN", None)
    tok_kwargs = dict(trust_remote_code=True, use_fast=False)
    mdl_kwargs = dict(torch_dtype=dtype, trust_remote_code=True, low_cpu_mem_usage=True)
    if hf_token:
        tok_kwargs["token"] = hf_token
        mdl_kwargs["token"] = hf_token

    print(f"[Load] Model: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, **tok_kwargs)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, **mdl_kwargs).to(device).eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    n_layers = get_num_layers_llama(model)
    print(f"[Info] Layers: {n_layers} | Device: {device} | DType: {dtype}")

    tasks = ["trivia", "math", "code"]
    results: Dict[str, Dict[str, List[float]]] = {}

    for task in tasks:
        print(f"[Scoring] {task} …")
        texts = load_texts_for_task(task, args.max_texts)
        H_vals, S_vals = layer_entropy_and_stability(
            texts, model, tokenizer,
            max_tokens=args.max_tokens,
            batch_size=args.batch_size,
            reservoir_per_layer=args.reservoir,
            device=device
        )
        H_z = znormalize(H_vals)
        S_z = znormalize(S_vals)
        r = [args.alpha*h + args.beta*s for h, s in zip(H_z, S_z)]
        results[task] = {
            "H": H_vals, "S": S_vals, "H_z": H_z, "S_z": S_z, "rank": r
        }

    # ----- Plot: one row, three columns (trivia / math / code) -----
    cols = len(tasks)
    plt.figure(figsize=(5*cols, 4.2))
    xs = list(range(1, n_layers + 1))

    for idx, task in enumerate(tasks, start=1):
        plt.subplot(1, cols, idx)
        plt.plot(xs, results[task]["H_z"], label="z(H) entropy")
        plt.plot(xs, results[task]["S_z"], label="z(S) stability (−var)")
        plt.plot(xs, results[task]["rank"], label="rank r = α z(H)+β z(S)")
        plt.xlabel("Layer")
        if idx == 1:
            plt.ylabel("z-score")
        plt.title(f"{task.capitalize()}")
        plt.legend(loc="best")
        plt.tight_layout()

    plt.suptitle("Motivation: Layer-wise entropy, stability, and task-aware rank — Llama-3.1-8B-Instruct")
    plt.tight_layout(rect=[0, 0.0, 1, 0.96])
    plt.savefig(png_path, dpi=170)
    print(f"[Saved] Plot -> {png_path}")

    # ----- Save raw numbers -----
    payload = {
        "model": args.model_id,
        "layers": n_layers,
        "params": {
            "max_texts_per_task": args.max_texts,
            "max_tokens": args.max_tokens,
            "batch_size": args.batch_size,
            "reservoir_tokens_per_layer": args.reservoir,
            "alpha": args.alpha,
            "beta": args.beta,
            "seed": args.seed,
            "device": device,
            "dtype": args.dtype,
        },
        "tasks": results
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[Saved] JSON -> {json_path}")

if __name__ == "__main__":
    main()

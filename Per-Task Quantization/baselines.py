#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Task-Aware Quantization — 3×SOTA (Mixed-calib) on TriviaQA
- Models: restricted to Llama* and Qwen* families (as requested)
- Methods (all use MIXED calibration where applicable):
    1) SOTA_AWQ_PerLayerMixed  -> AutoAWQ w/ per-layer keep (scored on mixed texts)
    2) SOTA_AWQ_All            -> AutoAWQ all transformer blocks quantized (no FP16 keep)
    3) SOTA_BNB_NF4            -> bitsandbytes 4-bit NF4 + double-quant (weight-only RTN)

Evaluation: same as before — greedy decode; TriviaQA EM/F1 (SQuAD normalization + aliases)
Install (one-time, GPU):
  pip install -U "transformers==4.51.3" "torch==2.6.*" datasets accelerate autoawq bitsandbytes

  
srun 
Example:
    srun -c 4 -A tdk -p tdk --gres=gpu:1 --pty  python baselines.py \
        --models google/gemma-3-1b-it\
        --methods     SOTA_AWQ_All  \
        --dataset-name trivia_qa --dataset-config rc.nocontext \
        --calib-size 512 --eval-size 2048 --gen-len 64 \
        --data-dir data --save-directory models_and_tokenizers



            srun -c 4 -A tdk -p tdk --gres=gpu:1 --pty  python baselines.py \
        --models google/gemma-3-1b-it\
        --methods    Clean      \
        --dataset-name trivia_qa --dataset-config rc.nocontext \
        --calib-size 512 --eval-size 2048 --gen-len 64 \
        --data-dir data --save-directory models_and_tokenizers

"""

from __future__ import annotations

import argparse, os, json, random, string, re, csv
from typing import List, Tuple, Iterable

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# Try both import paths for AutoAWQ for compatibility across versions
try:
    from awq import AutoAWQForCausalLM  # pip package name: autoawq
except Exception:
    try:
        from autoawq import AutoAWQForCausalLM  # fallback import path
        print("[Import] Using autoawq.AutoAWQForCausalLM import path")
    except Exception:
        AutoAWQForCausalLM = None  # type: ignore
        print("[Import] WARNING: Could not import AutoAWQForCausalLM from awq/autoawq. Install 'autoawq'.")

try:
    from transformers import BitsAndBytesConfig
    _HAS_BNB = True
except Exception:
    _HAS_BNB = False

# -----------------------------
# Speed-friendly flags
# -----------------------------
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
random.seed(0)

# -----------------------------
# Defaults / CLI
# -----------------------------
MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
]
METHODS = [
    "SOTA_AWQ_PerLayerMixed",
    "SOTA_AWQ_All",
    "SOTA_BNB_NF4",
    # Added simple baselines
    "PTQ4",   # bitsandbytes PTQ 4-bit (NF4/FP4)
    "PTQ8",   # bitsandbytes PTQ 8-bit (LLM.int8)
    "Clean",  # Evaluate original FP16 model
    # TAQ methods (per-task Trivia-only calibration)
    "TAQ_PerLayerPerTask", # keep top-K + first/last as FP16 (scored on Trivia calib), rest quantized
    "TAQ_Flexible",        # assign 4/8/16 by per-layer variance (high var→4, mid→8, low→16)
    # TAQ budget ablations (percent of layers kept FP16)
    "TAQ_Budget_20",
    "TAQ_Budget_40",
    "TAQ_Budget_60",
]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run 3×SOTA quantization (mixed-calib) on TriviaQA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    g = p.add_argument_group("Model / Method")
    g.add_argument("--models", nargs="+", default=MODELS)
    g.add_argument("--methods", nargs="+", default=METHODS)
    g.add_argument("--enforce-model-family", action="store_true", default=True,
                   help="Only allow Llama*/Qwen* model IDs.")

    d = p.add_argument_group("Dataset")
    d.add_argument("--dataset-name", type=str, default="trivia_qa")
    d.add_argument("--dataset-config", type=str, default="rc.nocontext")
    d.add_argument("--calib-ratio", type=float, default=0.4)
    d.add_argument("--calib-size", type=int, default=256)
    d.add_argument("--eval-size", type=int, default=0, help="0 = no cap (use ~60% split)")
    d.add_argument("--data-dir", type=str, default="data")

    i = p.add_argument_group("Inference")
    i.add_argument("--gen-len", type=int, default=64)
    i.add_argument("--device", type=str, default="auto", help="cuda:0 / cpu / auto")

    paths = p.add_argument_group("Paths")
    paths.add_argument("--save-directory", type=str, default="models_and_tokenizers")

    q = p.add_argument_group("Quantization / Layer selection")
    q.add_argument("--keep-topk", type=int, default=6, help="Top-K layers to keep FP16 (scores decide)")
    q.add_argument("--keep-first", type=int, default=2, help="Always keep first K FP16")
    q.add_argument("--keep-last", type=int, default=2, help="Always keep last K FP16")
    q.add_argument("--awq-wbit", type=int, default=4, help="AWQ weight bit-width")
    q.add_argument("--awq-qgroup-size", type=int, default=128, help="AWQ per-channel group size")
    q.add_argument("--awq-zero-point", action="store_true", help="Use zero points")
    q.add_argument("--awq-max-seq-len", type=int, default=2048, help="Calib max seq len")
    q.add_argument("--awq-save-subdir", type=str, default="awq_quant", help="Subdir for AWQ models")
    q.add_argument("--score-max-texts", type=int, default=256, help="Max texts for layer scoring")
    q.add_argument("--score-max-tokens", type=int, default=512, help="Token cap for scoring")

    b = p.add_argument_group("BitsAndBytes")
    b.add_argument("--bnb-4bit-quant-type", type=str, default="nf4", choices=["nf4", "fp4"])
    b.add_argument("--bnb-4bit-double-quant", action="store_true", default=True)

    a = p.add_argument_group("Auth")
    a.add_argument("--hf-token", type=str, default=None, help="Hugging Face token (or set HUGGING_FACE_HUB_TOKEN)")
    return p


# -----------------------------
# Helpers
# -----------------------------

def _resolve_device(dev: str) -> str:
    if dev == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return dev

def _is_llama_or_qwen(model_name: str) -> bool:
    m = model_name.lower()
    return ("llama" in m) or ("qwen" in m)

# -----------------------------
# Generation config hygiene
# -----------------------------

def sanitize_generation_config(model):
    gc = getattr(model, "generation_config", None)
    if gc is None:
        return
    if hasattr(gc, "do_sample"):
        gc.do_sample = False
    for k in ("temperature", "top_p", "top_k", "typical_p", "penalty_alpha"):
        if hasattr(gc, k):
            setattr(gc, k, None)

# -----------------------------
# Load model/tokenizer
# -----------------------------

def load_model_and_tokenizer(model_name: str, save_directory: str, device: str, token: str | None = None):
    device = _resolve_device(device)
    save_path = os.path.join(save_directory, model_name)
    os.makedirs(save_path, exist_ok=True)
    src = model_name if not os.listdir(save_path) else save_path

    tok_kwargs = {"trust_remote_code": True, "use_fast": False}
    mdl_kwargs = {"torch_dtype": torch.float16, "trust_remote_code": True, "low_cpu_mem_usage": True}
    if token:
        tok_kwargs["token"] = token
        mdl_kwargs["token"] = token

    tokenizer = AutoTokenizer.from_pretrained(src, **tok_kwargs)
    model = AutoModelForCausalLM.from_pretrained(src, **mdl_kwargs).to(device).eval()

    # Llama/Qwen padding hygiene
    if any(s in model_name.lower() for s in ("llama", "qwen")):
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "right"

    if src == model_name:
        tokenizer.save_pretrained(save_path)
        model.save_pretrained(save_path)

    sanitize_generation_config(model)
    return model, tokenizer, device

# -----------------------------
# Data loading / split caching
# -----------------------------

def load_dataset_and_split(dataset_name: str, dataset_config: str, calib_size: int, eval_size: int, data_dir: str, calib_ratio: float = 0.4):
    os.makedirs(data_dir, exist_ok=True)
    ds_cache = os.path.join(data_dir, f"{dataset_name}__{dataset_config}__validation.jsonl")
    calib_cache = os.path.join(data_dir, f"{dataset_name}__{dataset_config}__calib_{int(calib_ratio*100)}.jsonl")
    eval_cache = os.path.join(data_dir, f"{dataset_name}__{dataset_config}__eval_{int((1.0-calib_ratio)*100)}.jsonl")

    if os.path.exists(ds_cache):
        print(f"[Data] Loading cached dataset from {ds_cache}")
        dataset = load_dataset("json", data_files=ds_cache, split="train")
    else:
        print(f"[Data] Downloading dataset: name={dataset_name}, config={dataset_config}")
        dataset = load_dataset(dataset_name, dataset_config, split="validation")
        print(f"[Data] Caching dataset to {ds_cache}")
        dataset.to_json(ds_cache)

    if os.path.exists(calib_cache) and os.path.exists(eval_cache):
        print(f"[Data] Loading cached splits:\n  calib: {calib_cache}\n  eval:  {eval_cache}")
        calib = load_dataset("json", data_files=calib_cache, split="train")
        eval_ = load_dataset("json", data_files=eval_cache, split="train")
    else:
        print(f"[Data] Creating new random split with calib_ratio={calib_ratio:.2f}")
        n = len(dataset)
        idx = list(range(n))
        random.shuffle(idx)
        k = int(round(calib_ratio * n))
        calib = dataset.select(idx[:k])
        eval_ = dataset.select(idx[k:])
        print(f"[Data] Caching calib to {calib_cache} and eval to {eval_cache}")
        calib.to_json(calib_cache)
        eval_.to_json(eval_cache)

    if isinstance(calib_size, int) and 0 < calib_size < len(calib):
        calib = calib.select(range(calib_size))
    if isinstance(eval_size, int) and 0 < eval_size < len(eval_):
        eval_ = eval_.select(range(eval_size))

    print(f"[Data] Calibration set size: {len(calib)}")
    print(f"[Data] Evaluation set size: {len(eval_)}")
    print(f"[Data] Total cached validation size: {len(dataset)}")
    return calib, eval_

def print_start(args):
    print("Running experiments with the following configuration:")
    print(f"  Models: {args.models}")
    print(f"  Methods: {args.methods}  [all SOTA on MIXED calibration]")
    print(f"  Calibration Set Size: {args.calib_size}")
    print(f"  Generation Length: {args.gen_len}")
    print(f"  Dataset: {getattr(args, 'dataset_name', 'trivia_qa')} [{getattr(args, 'dataset_config', 'rc.nocontext')}]")
    if args.eval_size > 0:
        print(f"  Eval Set Cap: {args.eval_size}")
    else:
        print(f"  Eval Set Cap: none (using full {100*(1-args.calib_ratio):.0f}% split)")
    print(f"  Calib Ratio: {getattr(args, 'calib_ratio', 0.4):.2f}")
    print(f"  Data Dir: {getattr(args, 'data_dir', 'data')}")
    print(f"  Save Dir: {getattr(args, 'save_directory', 'models_and_tokenizers')}")
    print(f"  Keep-TopK: {args.keep_topk} | Keep first/last: {args.keep_first}/{args.keep_last}")
    print(f"  AWQ: int{args.awq_wbit}, q_group={args.awq_qgroup_size}, zero_point={bool(args.awq_zero_point)}")
    print(f"  Device: {args.device}")

# -----------------------------
# TriviaQA metrics (SQuAD normalization + aliases)
# -----------------------------
_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)

def normalize_answer(s: str) -> str:
    if s is None:
        return ""
    s = s.lower()
    s = "".join(ch if ch not in set(string.punctuation) else " " for ch in s)
    s = _ARTICLES_RE.sub(" ", s)
    s = " ".join(s.split())
    return s

def f1_score(pred: str, gold: str) -> float:
    p = normalize_answer(pred).split()
    g = normalize_answer(gold).split()
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    counts = {}
    for t in p:
        counts[t] = counts.get(t, 0) + 1
    overlap = 0
    for t in g:
        if counts.get(t, 0) > 0:
            overlap += 1
            counts[t] -= 1
    if overlap == 0:
        return 0.0
    prec = overlap / len(p)
    rec = overlap / len(g)
    return 2 * prec * rec / (prec + rec)

def exact_match_score(pred: str, gold: str) -> float:
    return 1.0 if normalize_answer(pred) == normalize_answer(gold) else 0.0

def best_em_f1(pred: str, golds: Iterable[str]) -> Tuple[float, float]:
    best_em, best_f1 = 0.0, 0.0
    for g in golds:
        em = exact_match_score(pred, g)
        f1 = f1_score(pred, g)
        if em > best_em: best_em = em
        if f1 > best_f1: best_f1 = f1
        if best_em == 1.0 and best_f1 == 1.0: break
    return best_em, best_f1

def get_gold_aliases(sample: dict) -> List[str]:
    golds: List[str] = []
    ans = sample.get("answer")
    if isinstance(ans, dict):
        v = ans.get("value")
        if isinstance(v, str) and v: golds.append(v)
        al = ans.get("aliases")
        if isinstance(al, list):
            for a in al:
                if isinstance(a, str) and a: golds.append(a)
    elif isinstance(ans, str) and ans:
        golds.append(ans)
    if not golds:
        v = sample.get("value")
        if isinstance(v, str) and v: golds.append(v)
        al = sample.get("aliases")
        if isinstance(al, list):
            for a in al:
                if isinstance(a, str) and a: golds.append(a)
    if not golds and "answers" in sample and isinstance(sample["answers"], list):
        for a in sample["answers"]:
            if isinstance(a, str) and a: golds.append(a)
            elif isinstance(a, dict):
                for k in ("text", "answer", "value"):
                    v = a.get(k)
                    if isinstance(v, str) and v:
                        golds.append(v); break
    seen, uniq = set(), []
    for g in golds:
        if g not in seen:
            uniq.append(g); seen.add(g)
    return uniq if uniq else [""]

_ANSWER_PREFIXES = ("answer:", "final answer:", "final:", "a:", "ans:", "prediction:", "the answer is")
def extract_pred_answer(generated_text: str) -> str:
    if not generated_text: return ""
    first = next((ln.strip() for ln in generated_text.strip().splitlines() if ln.strip()), "")
    low = first.lower()
    for pref in _ANSWER_PREFIXES:
        if low.startswith(pref):
            first = first[len(pref):].strip(); break
    for stop in [".", ";", "—", "–", "|"]:
        if stop in first:
            first = first.split(stop, 1)[0].strip(); break
    return first.strip(" '\"`")

# -----------------------------
# Mixed calibration (Code + Math + Trivia slice)
# -----------------------------

def sample_mbpp_texts(k: int) -> List[str]:
    try:
        ds = load_dataset("Muennighoff/mbpp", split="test")
    except Exception as e:
        print(f"[Calib] MBPP unavailable: {e}"); return []
    k = min(k, len(ds))
    texts = []
    for i in range(k):
        ex = ds[i]
        desc = ex.get("text") or ex.get("prompt") or ex.get("task_description") or ""
        if not isinstance(desc, str): desc = ""
        texts.append(f"Write a Python function.\nProblem: {desc}\n# Solution:")
    return texts

def sample_gsm8k_texts(k: int) -> List[str]:
    ds = None; last_err = None
    for spec in [("openai/gsm8k", "main"), ("gsm8k", "main")]:
        try:
            ds = load_dataset(spec[0], spec[1], split="train"); break
        except Exception as e:
            last_err = e; ds = None
    if ds is None:
        print(f"[Calib] GSM8K unavailable: {last_err}"); return []
    k = min(k, len(ds))
    texts = []
    for i in range(k):
        ex = ds[i]
        q = ex.get("question") or ""; a = ex.get("answer") or ""
        texts.append(f"Solve the math problem.\nQuestion: {q}\nAnswer: {a}")
    return texts

def sample_trivia_texts_from_dataset(ds, k: int) -> List[str]:
    k = min(k, len(ds))
    texts = []
    for i in range(k):
        ex = ds[i]
        q = ex.get("question") or ex.get("query") or ex.get("question_text") or ""
        a = ""
        ans = ex.get("answer")
        if isinstance(ans, dict):
            v = ans.get("value")
            if isinstance(v, str): a = v
        elif isinstance(ans, str):
            a = ans
        texts.append(f"Question: {q}\nAnswer: {a}")
    return texts

def build_mixed_calib_texts(trivia_calib, total_k: int) -> List[str]:
    if total_k <= 0: return []
    per = max(1, total_k // 3)
    code_texts = sample_mbpp_texts(per)
    math_texts = sample_gsm8k_texts(per)
    triv_texts = sample_trivia_texts_from_dataset(trivia_calib, per)
    mixed = code_texts + math_texts + triv_texts
    while len(mixed) < total_k and len(trivia_calib) > 0:
        i = len(mixed) % len(trivia_calib)
        ex = trivia_calib[i]
        q = ex.get("question") or ex.get("query") or ex.get("question_text") or ""
        mixed.append(f"Question: {q}\nAnswer:")
    random.shuffle(mixed)
    return mixed[:total_k]

# -----------------------------
# Layer scoring (activation norms)
# -----------------------------

def get_num_layers_for_llama(model) -> int:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return len(model.model.layers)
    if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
        return len(model.model.decoder.layers)
    return 0

def layer_prefix_llama(i: int) -> str:
    return f"model.layers.{i}."

def score_layers_by_hidden_norms(model, tokenizer, texts: List[str], max_tokens: int, device: str = "cuda:0") -> List[float]:
    device = _resolve_device(device)
    model.eval()
    scores: List[float] = []
    count = 0
    for t in texts:
        enc = tokenizer(t, return_tensors="pt", truncation=True, max_length=max_tokens).to(device)
        with torch.inference_mode():
            outputs = model(**enc, output_hidden_states=True, use_cache=False)
        hs = outputs.hidden_states
        if not scores:
            num_layers = max(0, len(hs) - 1)
            scores = [0.0 for _ in range(num_layers)]
        last_idx = enc["input_ids"].shape[1] - 1
        for li in range(1, len(hs)):
            vec = hs[li][0, last_idx]
            scores[li - 1] += torch.linalg.vector_norm(vec.float()).item()
        count += 1
    if count > 0 and scores:
        scores = [s / count for s in scores]
    return scores

def select_keep_layers(scores: List[float], topk: int, keep_first: int, keep_last: int) -> List[int]:
    n = len(scores)
    keep = set()
    for i in range(min(keep_first, n)): keep.add(i)
    for j in range(max(0, n - keep_last), n): keep.add(j)
    remaining = [i for i in range(n) if i not in keep]
    remaining.sort(key=lambda i: scores[i], reverse=True)
    for i in remaining[: max(0, topk - len(keep))]: keep.add(i)
    return sorted(list(keep))

# -----------------------------
# Layer variance (for flexible TAQ)
# -----------------------------

def layer_norm_stats(model, tokenizer, texts: List[str], max_tokens: int, device: str = "cuda:0") -> Tuple[List[float], List[float]]:
    """Collect per-layer activation norm means and variances across texts.

    Returns two lists of length n_layers: (means, variances).
    """
    device = _resolve_device(device)
    model.eval()
    values: List[List[float]] = []  # values[layer] = list of norms across samples
    for t in texts:
        enc = tokenizer(t, return_tensors="pt", truncation=True, max_length=max_tokens).to(device)
        with torch.inference_mode():
            outputs = model(**enc, output_hidden_states=True, use_cache=False)
        hs = outputs.hidden_states
        if not values:
            num_layers = max(0, len(hs) - 1)
            values = [[] for _ in range(num_layers)]
        last_idx = enc["input_ids"].shape[1] - 1
        for li in range(1, len(hs)):
            vec = hs[li][0, last_idx]
            values[li - 1].append(float(torch.linalg.vector_norm(vec.float()).item()))
    means: List[float] = []
    vars_: List[float] = []
    for v in values:
        if not v:
            means.append(0.0); vars_.append(0.0)
        else:
            m = sum(v) / len(v)
            means.append(m)
            var = sum((x - m) ** 2 for x in v) / max(1, len(v) - 1)
            vars_.append(var)
    return means, vars_

# -----------------------------
# AWQ quantization wrappers (Mixed calibration)
# -----------------------------

def awq_quantize_from_texts(model_name: str, tokenizer, calib_texts: List[str], keep_layers: List[int] | None, args, device: str = "cuda:0"):
    if AutoAWQForCausalLM is None:
        raise RuntimeError("AutoAWQ is not installed. Please 'pip install autoawq' to run AWQ baselines.")
    device = _resolve_device(device)
    qroot = os.path.join(args.save_directory, args.awq_save_subdir)
    os.makedirs(qroot, exist_ok=True)
    tag = "all" if keep_layers is None else "perlayer"
    qdir = os.path.join(qroot, f"{model_name.replace('/', '_')}_awq_{tag}")
    os.makedirs(qdir, exist_ok=True)

    src_path = os.path.join(args.save_directory, model_name)
    src = src_path if os.path.isdir(src_path) else model_name

    awq_model = AutoAWQForCausalLM.from_pretrained(
        src, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )

    # Configure kept layers (None means AWQ all layers)
    if keep_layers is None:
        exclude_prefixes = []
    else:
        n_layers = get_num_layers_for_llama(awq_model)
        if n_layers > 0:
            keep_layers = [i for i in keep_layers if 0 <= i < n_layers]
        exclude_prefixes = [layer_prefix_llama(i) for i in keep_layers]
    awq_model.modules_to_not_convert = exclude_prefixes

    quant_config = {
        "zero_point": True if "GEMM" else bool(args.awq_zero_point),
        "q_group_size": int(args.awq_qgroup_size),
        "w_bit": int(args.awq_wbit),
        "version": "GEMM",
    }

    print(f"[AWQ] Quantizing {model_name} ({tag}). Keep_layers={keep_layers if keep_layers is not None else 'ALL quantized'}")
    tried = []; last_err = None
    pad_id = getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", None)
    for kw in [
        dict(calib_data=calib_texts),
        dict(calib_data=calib_texts, max_calib_seq_len=int(args.awq_max_seq_len)),
        dict(calib_data=calib_texts, max_calib_samples=len(calib_texts)),
        dict(calib_data=calib_texts, split=32),
        dict(calib_data=calib_texts, pad_token_id=pad_id),
        dict(calib_data=calib_texts, max_calib_seq_len=int(args.awq_max_seq_len), split=32),
    ]:
        try:
            awq_model.quantize(tokenizer, quant_config=quant_config, **kw)
            last_err = None
            break
        except TypeError as e:
            tried.append(kw); last_err = e
        except Exception as e:
            last_err = e; break
    if last_err is not None:
        raise last_err

    awq_model.save_quantized(qdir)
    print(f"[AWQ] Saved quantized model to {qdir}")
    q_model = AutoAWQForCausalLM.from_quantized(qdir, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True).eval()
    sanitize_generation_config(q_model)
    return q_model, tokenizer, qdir

# -----------------------------
# BnB NF4 (weight-only RTN)
# -----------------------------

def bnb_nf4_model(model_name: str, save_directory: str, device: str, token: str | None, nf4_type: str = "nf4", double_quant: bool = True):
    if not _HAS_BNB:
        raise RuntimeError("bitsandbytes baseline requested but transformers.BitsAndBytesConfig not available.")
    device = _resolve_device(device)
    save_path = os.path.join(save_directory, f"{model_name}_bnb_nf4")
    os.makedirs(save_path, exist_ok=True)
    src_path = os.path.join(save_directory, model_name)
    src = src_path if os.path.isdir(src_path) else model_name

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=nf4_type,
        bnb_4bit_use_double_quant=bool(double_quant),
        bnb_4bit_compute_dtype=torch.float16,
    )
    tok_kwargs = {"trust_remote_code": True, "use_fast": False}
    mdl_kwargs = {"trust_remote_code": True, "low_cpu_mem_usage": True, "quantization_config": quant_config, "device_map": "auto"}
    if token:
        tok_kwargs["token"] = token; mdl_kwargs["token"] = token

    tokenizer = AutoTokenizer.from_pretrained(src, **tok_kwargs)
    model = AutoModelForCausalLM.from_pretrained(src, **mdl_kwargs).eval()
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    sanitize_generation_config(model)
    return model, tokenizer

# 8-bit PTQ (LLM.int8 via bitsandbytes)
def bnb_int8_model(model_name: str, save_directory: str, device: str, token: str | None):
    if not _HAS_BNB:
        raise RuntimeError("bitsandbytes PTQ8 requested but transformers.BitsAndBytesConfig not available.")
    device = _resolve_device(device)
    src_path = os.path.join(save_directory, model_name)
    src = src_path if os.path.isdir(src_path) else model_name

    quant_config = BitsAndBytesConfig(load_in_8bit=True)
    tok_kwargs = {"trust_remote_code": True, "use_fast": False}
    mdl_kwargs = {"trust_remote_code": True, "low_cpu_mem_usage": True, "quantization_config": quant_config, "device_map": "auto"}
    if token:
        tok_kwargs["token"] = token; mdl_kwargs["token"] = token

    tokenizer = AutoTokenizer.from_pretrained(src, **tok_kwargs)
    model = AutoModelForCausalLM.from_pretrained(src, **mdl_kwargs).eval()
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    sanitize_generation_config(model)
    return model, tokenizer

# -----------------------------
# EM/F1 evaluation (same as before)
# -----------------------------

def evaluate_triviaqa(method_label: str, model_name: str, model, tokenizer, eval_data, gen_len: int, save_directory: str, device: str = "cuda:0"):
    device = _resolve_device(device)
    os.makedirs(save_directory, exist_ok=True)
    out_dir = os.path.join(save_directory, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{model_name.replace('/', '_')}_{method_label.lower()}_results.jsonl")

    total = 0; em_sum = 0.0; f1_sum = 0.0
    with open(out_path, "w", encoding="utf-8") as f:
        for i, sample in enumerate(eval_data):
            q = sample.get("question") or sample.get("query") or sample.get("question_text") or ""
            golds = get_gold_aliases(sample)
            prompt = f"Question: {q}\nAnswer:"
            # Ensure inputs are on the same device as the model parameters.
            try:
                model_device = next(model.parameters()).device
            except StopIteration:
                model_device = torch.device("cpu")
            enc = tokenizer(prompt, return_tensors="pt").to(model_device)
            with torch.inference_mode():
                out = model.generate(
                    **enc,
                    max_new_tokens=gen_len,
                    do_sample=False,
                    use_cache=True,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
            gen = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            pred = extract_pred_answer(gen)
            em, f1o = best_em_f1(pred, golds)
            total += 1; em_sum += em; f1_sum += f1o
            f.write(json.dumps({
                "model": model_name,
                "method": method_label,
                "question": q,
                "golds": golds,
                "pred": pred,
                "em": em,
                "f1": f1o,
            }) + "\n")
            if (i + 1) % 100 == 0:
                print(f"[{method_label}] Progress {i+1}: EM {em_sum/total:.4f} | F1 {f1_sum/total:.4f}")

    em_avg = em_sum / total if total else 0.0
    f1_avg = f1_sum / total if total else 0.0
    print(f"[{method_label}] {model_name} EM: {em_avg*100:.2f}% | F1: {f1_avg*100:.2f}%  ({total} ex) -> {out_path}")
    return em_avg, f1_avg, total, out_path

# -----------------------------
# Bit allocation recording (for AWQ variants)
# -----------------------------

def record_bit_allocation(model_name: str, method_label: str, base_model, keep_layers: List[int] | None, awq_wbit: int, save_directory: str) -> str | None:
    try:
        n_layers = get_num_layers_for_llama(base_model)
        if n_layers <= 0:
            print("[Bits] Could not infer layers; skipping record."); return None
        bits = []
        keep_set = set(keep_layers or [])
        if keep_layers is None:
            bits = [int(awq_wbit) for _ in range(n_layers)]
        else:
            qb = int(awq_wbit)
            for i in range(n_layers):
                bits.append(16 if i in keep_set else qb)
        res_dir = os.path.join(save_directory, "results"); os.makedirs(res_dir, exist_ok=True)
        out_path = os.path.join(res_dir, f"{model_name.replace('/', '_')}_{method_label.lower()}_bits.json")
        payload = {
            "model": model_name, "method": method_label, "n_layers": n_layers,
            "keep_layers": sorted(list(keep_set)) if keep_layers is not None else [],
            "fp16_value": 16, "quant_value": int(awq_wbit),
            "per_layer": bits,
        }
        with open(out_path, "w", encoding="utf-8") as f: json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[Bits] Wrote per-layer bit allocation to {out_path}")
        return out_path
    except Exception as e:
        print(f"[Bits] Failed to write bit allocation: {e}"); return None

def record_custom_bit_allocation(model_name: str, method_label: str, n_layers: int, intended_bits: List[int], actual_bits: List[int], save_directory: str) -> str | None:
    """Write a JSON with per-layer intended and actual bit allocation.

    Useful for TAQ_Flexible where backend limits may force actual!=intended.
    """
    try:
        if n_layers <= 0 or len(intended_bits) != n_layers or len(actual_bits) != n_layers:
            print("[Bits] Invalid custom bit allocation sizes; skipping record.")
            return None
        res_dir = os.path.join(save_directory, "results"); os.makedirs(res_dir, exist_ok=True)
        out_path = os.path.join(res_dir, f"{model_name.replace('/', '_')}_{method_label.lower()}_bits.json")
        payload = {
            "model": model_name,
            "method": method_label,
            "n_layers": n_layers,
            "per_layer_intended": intended_bits,
            "per_layer_actual": actual_bits,
            "legend": {"4": "quantized 4-bit", "8": "quantized 8-bit (intended; backend may use 4)", "16": "kept FP16"},
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[Bits] Wrote custom per-layer bit allocation to {out_path}")
        return out_path
    except Exception as e:
        print(f"[Bits] Failed to write custom bit allocation: {e}"); return None

# -----------------------------
# Combined pipeline
# -----------------------------

def main():
    args = build_parser().parse_args()
    run(args)

def run(args):
    print_start(args)
    calib_data, eval_data = load_dataset_and_split(
        args.dataset_name, args.dataset_config, args.calib_size, args.eval_size, args.data_dir, args.calib_ratio
    )

    # Build calibration pools
    mixed_texts = build_mixed_calib_texts(calib_data, args.calib_size)
    trivia_texts_all = sample_trivia_texts_from_dataset(calib_data, args.calib_size)
    results_rows: List[Tuple[str, str, float, float, int]] = []

    for model_name in args.models:
        print("\n####################################\n")
        print(f"--- Loading base model: {model_name} ---")
        print("\n####################################\n")
        base_model, tokenizer, device = load_model_and_tokenizer(
            model_name, args.save_directory, args.device, token=args.hf_token
        )

        for method in args.methods:
            print("\n####################################\n")
            print(f"--- Running method: {method} ---")
            print("\n####################################\n")

            ml = method.lower()

            # Skip if results for this method already exist
            res_dir = os.path.join(args.save_directory, "results")
            os.makedirs(res_dir, exist_ok=True)
            res_path = os.path.join(res_dir, f"{model_name.replace('/', '_')}_{ml}_results.jsonl")
            if os.path.isfile(res_path) and os.path.getsize(res_path) > 0:
                print(f"[SKIP] Results exist for {model_name} / {method}: {res_path}")
                # Optionally, we could parse N and append to summary, but keep it simple here.
                continue

            if ml == "sota_awq_perlayermixed":
                score_texts = mixed_texts[: min(args.score_max_texts, len(mixed_texts))]
                scores = score_layers_by_hidden_norms(base_model, tokenizer, score_texts, args.score_max_tokens, device)
                if not scores:
                    n_layers = get_num_layers_for_llama(base_model)
                    base_keep = set(range(min(args.keep_first, n_layers)))
                    base_keep.update(range(max(0, n_layers - args.keep_last), n_layers))
                    keep = sorted(base_keep)
                else:
                    keep = select_keep_layers(scores, args.keep_topk, args.keep_first, args.keep_last)
                q_model, q_tok, _ = awq_quantize_from_texts(model_name, tokenizer, mixed_texts, keep, args, device)
                em, f1, n, _ = evaluate_triviaqa("SOTA_AWQ_PerLayerMixed", model_name, q_model, q_tok, eval_data, args.gen_len, args.save_directory, device)
                results_rows.append((model_name, "SOTA_AWQ_PerLayerMixed", em, f1, n))
                try:
                    record_bit_allocation(model_name, "SOTA_AWQ_PerLayerMixed", base_model, keep_layers=keep, awq_wbit=args.awq_wbit, save_directory=args.save_directory)
                except Exception:
                    pass

            elif ml == "sota_awq_all":
                # Quantize ALL layers with AWQ (no FP16-kept subset)
                q_model, q_tok, _ = awq_quantize_from_texts(model_name, tokenizer, mixed_texts, keep_layers=None, args=args, device=device)
                em, f1, n, _ = evaluate_triviaqa("SOTA_AWQ_All", model_name, q_model, q_tok, eval_data, args.gen_len, args.save_directory, device)
                results_rows.append((model_name, "SOTA_AWQ_All", em, f1, n))
                try:
                    record_bit_allocation(model_name, "SOTA_AWQ_All", base_model, keep_layers=None, awq_wbit=args.awq_wbit, save_directory=args.save_directory)
                except Exception:
                    pass

            elif ml == "sota_bnb_nf4":
                # BnB 4-bit NF4 double-quant baseline (weight-only RTN)
                b_model, b_tok = bnb_nf4_model(model_name, args.save_directory, args.device, args.hf_token, nf4_type=args.bnb_4bit_quant_type, double_quant=args.bnb_4bit_double_quant)
                em, f1, n, _ = evaluate_triviaqa("SOTA_BNB_NF4", model_name, b_model, b_tok, eval_data, args.gen_len, args.save_directory, device)
                results_rows.append((model_name, "SOTA_BNB_NF4", em, f1, n))

            elif ml == "ptq4":
                # Pure PTQ 4-bit via bitsandbytes (NF4 RTN)
                b_model, b_tok = bnb_nf4_model(
                    model_name, args.save_directory, args.device, args.hf_token,
                    nf4_type=args.bnb_4bit_quant_type, double_quant=args.bnb_4bit_double_quant
                )
                em, f1, n, _ = evaluate_triviaqa("PTQ4", model_name, b_model, b_tok, eval_data, args.gen_len, args.save_directory, device)
                results_rows.append((model_name, "PTQ4", em, f1, n))

            elif ml == "ptq8":
                # Pure PTQ 8-bit via bitsandbytes (LLM.int8)
                b_model, b_tok = bnb_int8_model(model_name, args.save_directory, args.device, args.hf_token)
                em, f1, n, _ = evaluate_triviaqa("PTQ8", model_name, b_model, b_tok, eval_data, args.gen_len, args.save_directory, device)
                results_rows.append((model_name, "PTQ8", em, f1, n))

            elif ml == "clean":
                # Evaluate original FP16 model as-is
                em, f1, n, _ = evaluate_triviaqa("Clean", model_name, base_model, tokenizer, eval_data, args.gen_len, args.save_directory, device)
                results_rows.append((model_name, "Clean", em, f1, n))

            elif ml == "taq_perlayerpertask":
                # Per-task TAQ: score on Trivia-only calibration; keep first/last and top-k as FP16; quantize others via AWQ int4
                score_texts = trivia_texts_all[: min(args.score_max_texts, len(trivia_texts_all))]
                scores = score_layers_by_hidden_norms(base_model, tokenizer, score_texts, args.score_max_tokens, device)
                if not scores:
                    n_layers = get_num_layers_for_llama(base_model)
                    base_keep = set(range(min(args.keep_first, n_layers)))
                    base_keep.update(range(max(0, n_layers - args.keep_last), n_layers))
                    keep = sorted(base_keep)
                else:
                    keep = select_keep_layers(scores, args.keep_topk, args.keep_first, args.keep_last)
                # Quantize non-kept layers with AWQ at args.awq_wbit (default 4)
                q_model, q_tok, _ = awq_quantize_from_texts(model_name, tokenizer, trivia_texts_all, keep, args, device)
                em, f1, n, _ = evaluate_triviaqa("TAQ_PerLayerPerTask", model_name, q_model, q_tok, eval_data, args.gen_len, args.save_directory, device)
                results_rows.append((model_name, "TAQ_PerLayerPerTask", em, f1, n))
                try:
                    record_bit_allocation(model_name, "TAQ_PerLayerPerTask", base_model, keep_layers=keep, awq_wbit=args.awq_wbit, save_directory=args.save_directory)
                except Exception:
                    pass

            elif ml == "taq_flexible":
                # Assign per-layer intended bits: high variance -> 4, mid -> 8, low -> 16 (Trivia-only)
                score_texts = trivia_texts_all[: min(args.score_max_texts, len(trivia_texts_all))]
                _, variances = layer_norm_stats(base_model, tokenizer, score_texts, args.score_max_tokens, device)
                n_layers = len(variances)
                # Determine tertiles
                order = sorted(range(n_layers), key=lambda i: variances[i], reverse=True)
                k = max(1, n_layers // 3)
                hi = set(order[:k])           # highest variance -> 4-bit
                mid = set(order[k:2*k])       # mid -> 8-bit (intended)
                lo = set(order[2*k:])         # lowest -> 16-bit
                intended = [16] * n_layers
                for i in range(n_layers):
                    if i in hi: intended[i] = 4
                    elif i in mid: intended[i] = 8
                    else: intended[i] = 16
                # Backend support: we can quantize with AWQ int4 for all non-16 layers; keep 16 as FP16
                keep_layers = [i for i, b in enumerate(intended) if b == 16]
                q_model, q_tok, _ = awq_quantize_from_texts(model_name, tokenizer, mixed_texts, keep_layers, args, device)
                em, f1, n, _ = evaluate_triviaqa("TAQ_Flexible", model_name, q_model, q_tok, eval_data, args.gen_len, args.save_directory, device)
                results_rows.append((model_name, "TAQ_Flexible", em, f1, n))
                try:
                    # Actual bits realized: 4 for quantized, 16 for kept (we can’t enforce 8 with AWQ here)
                    actual = [16 if b == 16 else int(args.awq_wbit) for b in intended]
                    record_custom_bit_allocation(model_name, "TAQ_Flexible", n_layers, intended_bits=intended, actual_bits=actual, save_directory=args.save_directory)
                except Exception:
                    pass

            elif ml in ("taq_budget_20", "taq_budget_40", "taq_budget_60"):
                # Keep a % of layers FP16 by score (Trivia-only); quantize rest at int4
                pct = {"taq_budget_20": 0.20, "taq_budget_40": 0.40, "taq_budget_60": 0.60}[ml]
                score_texts = trivia_texts_all[: min(args.score_max_texts, len(trivia_texts_all))]
                scores = score_layers_by_hidden_norms(base_model, tokenizer, score_texts, args.score_max_tokens, device)
                n_layers = len(scores) if scores else get_num_layers_for_llama(base_model)
                k_keep = max(0, int(round(pct * max(0, n_layers))))
                if not scores:
                    # fallback: keep first/last evenly split
                    first = min(k_keep // 2, n_layers)
                    last = min(k_keep - first, n_layers - first)
                    keep = sorted(set(list(range(first)) + list(range(max(0, n_layers - last), n_layers))))
                else:
                    order = sorted(range(n_layers), key=lambda i: scores[i], reverse=True)
                    keep = sorted(order[:k_keep])
                q_model, q_tok, _ = awq_quantize_from_texts(model_name, tokenizer, mixed_texts, keep, args, device)
                label = f"TAQ_Budget_{int(pct*100)}"
                em, f1, n, _ = evaluate_triviaqa(label, model_name, q_model, q_tok, eval_data, args.gen_len, args.save_directory, device)
                results_rows.append((model_name, label, em, f1, n))
                try:
                    record_bit_allocation(model_name, label, base_model, keep_layers=keep, awq_wbit=args.awq_wbit, save_directory=args.save_directory)
                except Exception:
                    pass

            else:
                print(f"Unknown method: {method}. Skipping...")

    # Print & save summary
    res_dir = os.path.join(args.save_directory, "results"); os.makedirs(res_dir, exist_ok=True)
    csv_path = os.path.join(res_dir, "summary.csv")

    print("\n===== SUMMARY =====")
    print(f"{'Model':40s} {'Method':20s} {'EM':>8s} {'F1':>8s} {'N':>6s}")
    for m, meth, em, f1, n in results_rows:
        print(f"{m:40s} {meth:20s} {100*em:8.2f} {100*f1:8.2f} {n:6d}")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "method", "em", "f1", "n_examples"])
        for row in results_rows:
            w.writerow(row)

    print(f"\nSaved summary to {csv_path}")


if __name__ == "__main__":
    main()

# ===== from here it's in jupyter =====
# In a notebook you can do:
# from run_taq_experiments import build_parser, run
# args = build_parser().parse_args([])
# args.models = ["meta-llama/Llama-3.1-8B-Instruct"]
# args.methods = ["SOTA_AWQ_PerLayerMixed", "SOTA_AWQ_All", "SOTA_BNB_NF4"]
# run(args)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
INFO_VAR_MIX: Task-Aware Mixed-Precision (4/8/16) via Information & Low-Variance scoring (AWQ-like runtime)

- Score layers on a small, task-specific calibration set:
    * Information = matrix-based entropy of a small token reservoir per layer
    * Stability   = inverse of activation variance across calibration tokens
  High-score layers + low-variance layers => higher bits.

- Assign bits per layer from {4, 8, 16}:
    * First & last layers forced to 16-bit
    * Middle layers: top-N -> 16, next-M -> 8, rest -> 4  (configurable or auto)

- Efficient runtime:
    * Weight-only per-group uniform affine quant (group along in_features)
    * "Lazy dequant cache": on first forward, reconstruct fp16 weight once and reuse
      (fast like AWQ; set --cache-dequant=0 to avoid cache and prioritize memory)

Install:
  pip install -U "transformers==4.51.3" "torch==2.6.*" datasets accelerate

Example: 
  srun -c 4 -A tdk -p tdk --gres=gpu:1 --pty  python run_taq_experiments.py \
        --models mistralai/Mistral-7B-Instruct-v0.2\
    --dataset-name trivia_qa --dataset-config rc.nocontext \
    --calib-size 512 --eval-size 2048 --gen-len 64 \
    --score-max-texts 256 --score-max-tokens 512 --score-batch-size 8 \
    --info-weight 0.5 --var-weight 0.5 --reservoir-size 256 \
    --keep-first 2 --keep-last 2 --mid-n16 -1 --mid-n8 -1 \
    --qgroup-size 128 --cache-dequant 1 \
    --save-directory models_and_tokenizers
"""

from __future__ import annotations
import argparse, os, json, random, string, re, csv, math
from typing import List, Tuple, Iterable, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# -----------------------------
# Torch / RNG hygiene
# -----------------------------
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
random.seed(0)

# -----------------------------
# CLI
# -----------------------------
MODELS = ["meta-llama/Llama-3.1-8B-Instruct"]
METHODS = ["INFO_VAR_MIX"]

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="INFO_VAR_MIX (per-task, mixed-precision 4/8/16) on TriviaQA (AWQ-like runtime).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    g = p.add_argument_group("Model / Method")
    g.add_argument("--models", nargs="+", default=MODELS)
    g.add_argument("--methods", nargs="+", default=METHODS)
    g.add_argument("--enforce-model-family", action="store_true", default=True,
                   help="Only allow Llama*/Qwen* model IDs.")
    g.add_argument("--ablations", type=int, default=1, help="Number of runs with different seeds (>=1).")

    d = p.add_argument_group("Dataset")
    d.add_argument("--dataset-name", type=str, default="trivia_qa")
    d.add_argument("--dataset-config", type=str, default="rc.nocontext")
    d.add_argument("--calib-ratio", type=float, default=0.4)
    d.add_argument("--calib-size", type=int, default=256)
    d.add_argument("--eval-size", type=int, default=0, help="0 = no cap (~60% split)")
    d.add_argument("--data-dir", type=str, default="data")

    i = p.add_argument_group("Inference")
    i.add_argument("--gen-len", type=int, default=64)
    i.add_argument("--device", type=str, default="auto", help="cuda:0 / cpu / auto")

    paths = p.add_argument_group("Paths")
    paths.add_argument("--save-directory", type=str, default="models_and_tokenizers")

    q = p.add_argument_group("Quantization / Selection")
    q.add_argument("--keep-first", type=int, default=2, help="Always keep first K FP16 (16-bit)")
    q.add_argument("--keep-last", type=int, default=2, help="Always keep last K FP16 (16-bit)")
    q.add_argument("--qgroup-size", type=int, default=128, help="Per-group quantization size along in_features")
    q.add_argument("--reservoir-size", type=int, default=256, help="Per-layer token reservoir for entropy")
    q.add_argument("--score-max-texts", type=int, default=256, help="Max texts for layer scoring")
    q.add_argument("--score-max-tokens", type=int, default=512, help="Token cap for scoring")
    q.add_argument("--score-batch-size", type=int, default=8, help="Batch size for scoring forward passes")
    q.add_argument("--info-weight", type=float, default=0.5, help="Weight for information (entropy) score")
    q.add_argument("--var-weight", type=float, default=0.5, help="Weight for inverse-variance score")
    q.add_argument("--mid-n16", type=int, default=-1, help="How many middle layers to set to 16-bit (auto if -1)")
    q.add_argument("--mid-n8", type=int, default=-1, help="How many middle layers to set to 8-bit (auto if -1)")
    q.add_argument("--cache-dequant", type=int, default=1, help="1: cache dequantized fp16 weights for speed; 0: on-the-fly")

    a = p.add_argument_group("Auth")
    a.add_argument("--hf-token", type=str, default=None, help="Hugging Face token (or env HUGGING_FACE_HUB_TOKEN)")
    return p

# -----------------------------
# Utilities
# -----------------------------
def _resolve_device(dev: str) -> str:
    if dev == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return dev

def _is_llama_or_qwen(model_name: str) -> bool:
    m = model_name.lower()
    return ("llama" in m) or ("qwen" in m)

def sanitize_generation_config(model):
    gc = getattr(model, "generation_config", None)
    if gc is None: return
    if hasattr(gc, "do_sample"): gc.do_sample = False
    for k in ("temperature", "top_p", "top_k", "typical_p", "penalty_alpha"):
        if hasattr(gc, k): setattr(gc, k, None)

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

    # padding hygiene
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
# Data loading / cache split
# -----------------------------
def load_dataset_and_split(dataset_name: str, dataset_config: str, calib_size: int, eval_size: int, data_dir: str, calib_ratio: float = 0.4):
    os.makedirs(data_dir, exist_ok=True)
    ds_cache = os.path.join(data_dir, f"{dataset_name}__{dataset_config}__validation.jsonl")
    calib_cache = os.path.join(data_dir, f"{dataset_name}__{dataset_config}__calib_{int(calib_ratio*100)}.jsonl")
    eval_cache  = os.path.join(data_dir, f"{dataset_name}__{dataset_config}__eval_{int((1.0-calib_ratio)*100)}.jsonl")

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
        eval_  = load_dataset("json", data_files=eval_cache,  split="train")
    else:
        print(f"[Data] Creating new random split with calib_ratio={calib_ratio:.2f}")
        n = len(dataset)
        idx = list(range(n)); random.shuffle(idx)
        k = int(round(calib_ratio * n))
        calib = dataset.select(idx[:k])
        eval_  = dataset.select(idx[k:])
        print(f"[Data] Caching calib to {calib_cache} and eval to {eval_cache}")
        calib.to_json(calib_cache); eval_.to_json(eval_cache)

    if isinstance(calib_size, int) and 0 < calib_size < len(calib): calib = calib.select(range(calib_size))
    if isinstance(eval_size, int) and 0 < eval_size < len(eval_):   eval_  = eval_.select(range(eval_size))

    print(f"[Data] Calibration set size: {len(calib)}")
    print(f"[Data] Evaluation set size: {len(eval_)}")
    print(f"[Data] Total cached validation size: {len(dataset)}")
    return calib, eval_

# -----------------------------
# Metrics (SQuAD-like)
# -----------------------------
_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
def normalize_answer(s: str) -> str:
    if s is None: return ""
    s = s.lower()
    s = "".join(ch if ch not in set(string.punctuation) else " " for ch in s)
    s = _ARTICLES_RE.sub(" ", s)
    s = " ".join(s.split())
    return s

def f1_score(pred: str, gold: str) -> float:
    p = normalize_answer(pred).split(); g = normalize_answer(gold).split()
    if not p and not g: return 1.0
    if not p or not g: return 0.0
    counts = {}
    for t in p: counts[t] = counts.get(t, 0) + 1
    overlap = 0
    for t in g:
        if counts.get(t, 0) > 0:
            overlap += 1; counts[t] -= 1
    if overlap == 0: return 0.0
    prec = overlap / len(p); rec = overlap / len(g)
    return 2 * prec * rec / (prec + rec)

def exact_match_score(pred: str, gold: str) -> float:
    return 1.0 if normalize_answer(pred) == normalize_answer(gold) else 0.0

def best_em_f1(pred: str, golds: Iterable[str]) -> Tuple[float, float]:
    best_em, best_f1 = 0.0, 0.0
    for g in golds:
        em = exact_match_score(pred, g); f1 = f1_score(pred, g)
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
                    if isinstance(v, str) and v: golds.append(v); break
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
# Trivia-only texts
# -----------------------------
def sample_trivia_texts_from_dataset(ds, k: int) -> List[str]:
    k = min(k, len(ds)); texts = []
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

# -----------------------------
# Model Introspection
# -----------------------------
def get_num_layers_for_llama(model) -> int:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return len(model.model.layers)
    if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
        return len(model.model.decoder.layers)
    return 0

# -----------------------------
# Scoring (Information + Inverse Variance)
# -----------------------------
@torch.no_grad()
def score_layers_info_var(
    model, tokenizer, texts: List[str], max_tokens: int, device: str,
    reservoir_size: int = 256, batch_size: int = 8
) -> Tuple[List[float], List[float], List[int]]:
    """
    Efficient single-pass scoring:
      - Builds a small token reservoir per layer (<= reservoir_size) for entropy
      - Tracks scalar variance per layer via streaming sums (no huge buffers)
    Returns:
      info_scores (raw entropy), invvar_scores (negative variance), token_counts_per_layer
    """
    device = _resolve_device(device)
    nL = get_num_layers_for_llama(model)
    # Reservoirs (CPU float32)
    reservoirs = [None for _ in range(nL)]          # Tensor [R, D] or None
    res_filled = [0 for _ in range(nL)]
    res_seen =   [0 for _ in range(nL)]
    # Streaming variance stats per layer (scalar)
    s1 = [0.0 for _ in range(nL)]
    s2 = [0.0 for _ in range(nL)]
    n_elems = [0 for _ in range(nL)]

    # batching
    B = max(1, int(batch_size))
    total = len(texts)
    for start in range(0, total, B):
        batch = texts[start : start + B]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                        max_length=max_tokens).to(device)
        out = model(**enc, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states  # tuple length nL+1 (incl. embeddings), we use 1..nL
        # progress
        pct = 100.0 * min(total, start + B) / max(1, total)
        print(f"[SCORING] {pct:.1f}%")

        for li in range(1, len(hs)):
            X = hs[li]  # [B, T, D]
            Bx, Tx, Dx = X.shape
            Xf = X.reshape(Bx * Tx, Dx)  # [N, D]
            # variance stats (scalar): E[x^2] - E[x]^2 over both tokens and dims
            # compute in float64 on device then move to cpu scalar
            sum1 = Xf.sum(dtype=torch.float64)
            sum2 = (Xf * Xf).sum(dtype=torch.float64)
            cnt  = Xf.numel()
            s1[li-1] += float(sum1.item())
            s2[li-1] += float(sum2.item())
            n_elems[li-1] += int(cnt)

            # reservoir sampling for entropy (sample up to ~min(8*B, N) tokens)
            N = Xf.shape[0]
            take = min(8 * B, N)
            if take > 0:
                # choose uniformly at random positions
                idx = torch.randint(0, N, (take,), device=Xf.device)
                samples = Xf.index_select(0, idx).detach().to(torch.float32).cpu()  # [take, D]
                # init reservoir lazily
                if reservoirs[li-1] is None:
                    # allocate a fixed-size reservoir on first sight
                    reservoirs[li-1] = torch.empty((reservoir_size, Dx), dtype=torch.float32)
                # insert with reservoir sampling
                for j in range(samples.shape[0]):
                    res_seen[li-1] += 1
                    if res_filled[li-1] < reservoir_size:
                        reservoirs[li-1][res_filled[li-1]] = samples[j]
                        res_filled[li-1] += 1
                    else:
                        # replace existing item with probability R / t
                        t = res_seen[li-1]
                        R = reservoir_size
                        # Draw replace index with probability R/t
                        if random.random() < (R / max(1, t)):
                            ridx = random.randrange(0, R)
                            reservoirs[li-1][ridx] = samples[j]

    # compute raw entropy & inverse variance per layer
    info_scores: List[float] = []
    invvar_scores: List[float] = []
    for li in range(nL):
        # variance
        if n_elems[li] > 0:
            mean = s1[li] / n_elems[li]
            var = max(0.0, (s2[li] / n_elems[li]) - (mean * mean))
            invvar = -var  # lower variance => higher (less negative) score after z-norm
        else:
            invvar = 0.0
        invvar_scores.append(invvar)

        # entropy from Gram on centered reservoir
        R = res_filled[li]
        if R > 1 and reservoirs[li] is not None:
            X = reservoirs[li][:R]  # [R, D]
            X = X - X.mean(dim=0, keepdim=True)
            # Gram (R x R), eigenvalues => probabilities
            G = (X @ X.T) / max(1, R)
            evals = torch.linalg.eigvalsh(G).clamp(min=0.0)
            if float(evals.sum()) > 0.0:
                p = (evals / evals.sum()).double()
                H = float((-(p * (p + 1e-12).log())).sum().item())  # Shannon-like
            else:
                H = 0.0
        else:
            H = 0.0
        info_scores.append(H)

    return info_scores, invvar_scores, n_elems

def _zscore(xs: List[float]) -> List[float]:
    if not xs: return xs
    m = sum(xs)/len(xs)
    v = sum((x-m)*(x-m) for x in xs)/max(1, len(xs)-1)
    s = math.sqrt(v + 1e-12)
    return [ (x - m) / s for x in xs ]

def allocate_bits(
    info_scores: List[float], invvar_scores: List[float],
    keep_first: int, keep_last: int,
    mid_n16: int, mid_n8: int
) -> List[int]:
    """
    Combine scores and assign bits per layer from {4, 8, 16}.
    First & last segments forced to 16.
    Among middle layers, sort by combined score and assign:
       top mid_n16 -> 16, next mid_n8 -> 8, rest -> 4
    """
    nL = len(info_scores)
    Iz = _zscore(info_scores)
    Vz = _zscore(invvar_scores)
    # combine (equal weight by default; can easily expose as args if desired)
    # weight via CLI by rescaling before call if needed; here we keep neutral and do linear blend later in run()
    # Return placeholder here; actual blend happens in run() where we know weights.
    return [0]*nL  # filled later

# -----------------------------
# Quantization (4/8/16)
# -----------------------------
class QuantLinearWB(nn.Module):
    """
    Weight-only per-group affine quantization for wbit in {4,8}.
    Stores packed uint8 codes:
      - 4-bit: two nibbles per byte
      - 8-bit: one value per byte
    Fast path: cache dequantized fp16 weight on first forward if cache_dequant=True.
    """
    def __init__(self, linear: nn.Linear, wbit: int = 4, group_size: int = 128, cache_dequant: bool = True):
        super().__init__()
        assert wbit in (4, 8)
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        # Copy original bias (if any) into a fresh Parameter owned by this module
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.detach().clone())
        else:
            self.bias = None

        # Work on a CPU copy of weights to build codes
        W = linear.weight.detach().to(torch.float32).cpu()  # [out, in]
        self.wbit = int(wbit)
        self.group_size = int(group_size)
        self.cache_dequant = bool(cache_dequant)
        self._W_cache = None  # fp16 cache built on first forward (device-local)

        G = max(1, self.group_size)
        in_groups = (W.shape[1] + G - 1) // G

        scales = torch.empty((W.shape[0], in_groups), dtype=torch.float32)
        zeros = torch.empty((W.shape[0], in_groups), dtype=torch.float32)

        if self.wbit == 4:
            codes = torch.empty((W.shape[0], in_groups, (G + 1) // 2), dtype=torch.uint8)
            Q = 15
        else:
            codes = torch.empty((W.shape[0], in_groups, G), dtype=torch.uint8)
            Q = 255

        for o in range(W.shape[0]):
            row = W[o]
            for g in range(in_groups):
                s = g * G
                e = min((g + 1) * G, W.shape[1])
                seg = row[s:e]
                if seg.numel() == 0:
                    scales[o, g] = 1.0
                    zeros[o, g] = 0.0
                    continue
                wmin = float(seg.min().item())
                wmax = float(seg.max().item())
                if wmax <= wmin + 1e-8:
                    scale = 1.0
                    zp = 0.0
                    qvals = torch.zeros_like(seg, dtype=torch.uint8)
                else:
                    scale = (wmax - wmin) / Q
                    zp = -wmin / (scale + 1e-12)
                    q = torch.round(seg / (scale + 1e-12) + zp)
                    q = torch.clamp(q, 0, Q).to(torch.int32)
                    qvals = q.to(torch.uint8)
                scales[o, g] = float(scale)
                zeros[o, g] = float(zp)

                if self.wbit == 4:
                    # pack two nibbles per byte
                    if qvals.numel() % 2 == 1:
                        qvals = torch.cat([qvals, torch.zeros(1, dtype=torch.uint8)], dim=0)
                    hi = qvals[0::2]
                    lo = qvals[1::2]
                    packed = (hi << 4) | lo
                    buf = torch.zeros((G + 1) // 2, dtype=torch.uint8)
                    buf[:packed.numel()] = packed
                    codes[o, g, :] = buf
                else:
                    buf = torch.zeros(G, dtype=torch.uint8)
                    buf[:qvals.numel()] = qvals
                    codes[o, g, :] = buf

        self.register_buffer("qcodes", codes.contiguous())  # [out, in_groups, G or ceil(G/2)]
        self.register_buffer("scales", scales.contiguous())  # [out, in_groups]
        self.register_buffer("zeros", zeros.contiguous())  # [out, in_groups]

    def _dequant_weight_to(self, device: torch.device) -> torch.Tensor:
        # Reconstruct to float16 on target device
        out, in_groups, storeG = self.qcodes.shape
        G = self.group_size
        Wfull = torch.zeros((out, in_groups * G), dtype=torch.float16, device=device)

        if self.wbit == 4:
            # unpack nibbles
            # We'll reconstruct group-by-group to limit peak memory
            for o in range(out):
                for g in range(in_groups):
                    packed = self.qcodes[o, g, :].to(device)
                    # expand to length G
                    hi = (packed >> 4) & 0xF
                    lo = packed & 0xF
                    q = torch.empty(G, dtype=torch.float32, device=device)
                    q[0::2] = hi.to(torch.float32)
                    q[1::2] = lo.to(torch.float32)
                    scale = self.scales[o, g].to(device=device, dtype=torch.float32)
                    zp    = self.zeros[o, g].to(device=device, dtype=torch.float32)
                    w = (q - zp) * scale
                    s = g * G; e = s + G
                    Wfull[o, s:e] = w.to(torch.float16)
        else:
            # 8-bit
            for o in range(out):
                for g in range(in_groups):
                    q = self.qcodes[o, g, :].to(device=device, dtype=torch.float32)
                    scale = self.scales[o, g].to(device=device, dtype=torch.float32)
                    zp    = self.zeros[o, g].to(device=device, dtype=torch.float32)
                    w = (q - zp) * scale
                    s = g * G; e = s + G
                    Wfull[o, s:e] = w.to(torch.float16)

        return Wfull[:, :self.in_features]  # trim to original width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.cache_dequant and (self._W_cache is not None) and (self._W_cache.device == x.device):
            return F.linear(x, self._W_cache, self.bias)

        W = self._dequant_weight_to(x.device)
        if self.cache_dequant and not self.training:
            self._W_cache = W  # cache fp16 weight for speed
        return F.linear(x, W, self.bias)

def quantize_with_bit_map(model: nn.Module, bit_map: List[int], group_size: int = 128, cache_dequant: bool = True):
    """
    Replace Linear modules in transformer blocks with QuantLinearWB according to per-layer bit_map.
    bit_map[li] in {4,8,16}
    """
    nL = get_num_layers_for_llama(model)
    assert len(bit_map) == nL, "bit_map length must equal number of transformer layers"
    replaced = 0
    total = 0
    for li in range(nL):
        total += 1
        bits = bit_map[li]
        if bits == 16:
            pct = 100.0 * (li + 1) / max(1, nL)
            print(f"[QUANT] {pct:.1f}%  layer {li} -> FP16 (kept)")
            continue
        blk = model.model.layers[li]
        # walk leaf modules and replace Linear
        for name, mod in list(blk.named_modules()):
            if isinstance(mod, nn.Linear):
                parent = blk
                parts = name.split(".")
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                leaf = parts[-1]
                setattr(parent, leaf, QuantLinearWB(mod, wbit=bits, group_size=group_size, cache_dequant=cache_dequant))
                replaced += 1
        pct = 100.0 * (li + 1) / max(1, nL)
        print(f"[QUANT] {pct:.1f}%  layer {li} -> int{bits}")
    print(f"[QUANT] Replaced {replaced} Linear modules across {nL} layers.")
    return model

# -----------------------------
# EM/F1 evaluation (with %)
# -----------------------------
def evaluate_triviaqa(method_label: str, model_name: str, model, tokenizer, eval_data, gen_len: int, save_directory: str, device: str = "cuda:0"):
    device = _resolve_device(device)
    os.makedirs(save_directory, exist_ok=True)
    out_dir = os.path.join(save_directory, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{model_name.replace('/', '_')}_{method_label.lower()}_results.jsonl")

    total = len(eval_data)
    done = 0; em_sum = 0.0; f1_sum = 0.0
    with open(out_path, "w", encoding="utf-8") as f:
        for i, sample in enumerate(eval_data):
            q = sample.get("question") or sample.get("query") or sample.get("question_text") or ""
            golds = get_gold_aliases(sample)
            prompt = f"Question: {q}\nAnswer:"
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
            done += 1; em_sum += em; f1_sum += f1o
            f.write(json.dumps({
                "model": model_name,
                "method": method_label,
                "question": q,
                "golds": golds,
                "pred": pred,
                "em": em,
                "f1": f1o,
            }) + "\n")
            if (i + 1) % 50 == 0 or (i + 1) == total:
                pct = 100.0 * done / max(1, total)
                print(f"[EVAL] {pct:.1f}%  EM {em_sum/done:.4f} | F1 {f1_sum/done:.4f}")

    em_avg = em_sum / max(1, done)
    f1_avg = f1_sum / max(1, done)
    print(f"[{method_label}] {model_name} EM: {em_avg*100:.2f}% | F1: {f1_avg*100:.2f}%  ({done} ex) -> {out_path}")
    return em_avg, f1_avg, done, out_path

# -----------------------------
# Bit allocation record
# -----------------------------
def record_bit_allocation(model_name: str, method_label: str, bit_map: List[int], save_directory: str) -> str | None:
    try:
        n_layers = len(bit_map)
        res_dir = os.path.join(save_directory, "results"); os.makedirs(res_dir, exist_ok=True)
        out_path = os.path.join(res_dir, f"{model_name.replace('/', '_')}_{method_label.lower()}_bits.json")
        payload = {
            "model": model_name, "method": method_label, "n_layers": n_layers,
            "per_layer_bits": bit_map,
            "legend": {"4": "int4 per-group", "8": "int8 per-group", "16": "fp16 kept"},
        }
        with open(out_path, "w", encoding="utf-8") as f: json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[Bits] Wrote per-layer bit allocation to {out_path}")
        return out_path
    except Exception as e:
        print(f"[Bits] Failed to write bit allocation: {e}"); return None

# -----------------------------
# Runner
# -----------------------------
def print_start(args):
    print("Running INFO_VAR_MIX with the following configuration:")
    print(f"  Models: {args.models}")
    print(f"  Ablations: {args.ablations}")
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
    print(f"  Keep first/last: {args.keep_first}/{args.keep_last}")
    print(f"  Mid budgets (16/8): {args.mid_n16}/{args.mid_n8} (-1 = auto)")
    print(f"  Quant groups: {args.qgroup_size}")
    print(f"  Scoring: reservoir={args.reservoir_size}, batch={args.score_batch_size}, tokens={args.score_max_tokens}")
    print(f"  Weights: info={args.info_weight:.2f}, invVar={args.var_weight:.2f}")
    print(f"  Cache dequant: {bool(args.cache_dequant)}")
    print(f"  Device: {args.device}")

def main():
    args = build_parser().parse_args()
    run(args)

def run(args):
    args.methods = ["INFO_VAR_MIX"]  # force

    print_start(args)
    calib_data, eval_data = load_dataset_and_split(
        args.dataset_name, args.dataset_config, args.calib_size, args.eval_size, args.data_dir, args.calib_ratio
    )

    # Build task texts for scoring
    trivia_texts_all = sample_trivia_texts_from_dataset(calib_data, args.calib_size)
    results_rows: List[Tuple[str, str, float, float, int]] = []

    for model_name in args.models:
        print("\n####################################\n")
        print(f"--- Loading base model: {model_name} ---")
        print("\n####################################\n")
        base_model, tokenizer, device = load_model_and_tokenizer(
            model_name, args.save_directory, args.device, token=args.hf_token
        )
        n_layers = get_num_layers_for_llama(base_model)

        # Ablations (seeds)
        K = max(1, int(args.ablations))
        for seed in range(K):
            label = f"INFO_VAR_MIX_s{seed}"
            print("\n####################################\n")
            print(f"--- Running method: {label} ---")
            print("\n####################################\n")

            # Skip if results exist
            res_dir = os.path.join(args.save_directory, "results")
            os.makedirs(res_dir, exist_ok=True)
            res_path = os.path.join(res_dir, f"{model_name.replace('/', '_')}_{label.lower()}_results.jsonl")
            if os.path.isfile(res_path) and os.path.getsize(res_path) > 0:
                print(f"[SKIP] Results exist for {model_name} / {label}: {res_path}")
                continue

            rng = random.Random(seed)
            score_texts = list(trivia_texts_all); rng.shuffle(score_texts)
            score_texts = score_texts[: min(args.score_max_texts, len(score_texts))]

            # ---- Fast scoring (Information + inverse Variance) ----
            info_raw, invvar_raw, _ = score_layers_info_var(
                base_model, tokenizer, score_texts, args.score_max_tokens, device,
                reservoir_size=args.reservoir_size, batch_size=args.score_batch_size
            )
            # z-norm and combine
            Iz = _zscore(info_raw)
            Vz = _zscore(invvar_raw)  # already negative variance
            comb = [args.info_weight * Iz[i] + args.var_weight * Vz[i] for i in range(len(Iz))]

            # ---- Assign per-layer bits {16,8,4} ----
            # Force edges
            bit_map = [4 for _ in range(n_layers)]
            for i in range(min(args.keep_first, n_layers)): bit_map[i] = 16
            for j in range(max(0, n_layers - args.keep_last), n_layers): bit_map[j] = 16

            mid_idxs = [i for i in range(n_layers) if bit_map[i] != 16]
            mid_count = len(mid_idxs)

            # Auto budgets if -1
            mid_n16 = args.mid_n16 if args.mid_n16 >= 0 else max(0, int(round(0.15 * mid_count)))
            mid_n8  = args.mid_n8  if args.mid_n8  >= 0 else max(0, int(round(0.45 * mid_count)))

            # Rank middle layers by combined score (desc)
            mid_sorted = sorted(mid_idxs, key=lambda i: comb[i], reverse=True)
            for i in mid_sorted[:mid_n16]: bit_map[i] = 16
            for i in mid_sorted[mid_n16 : mid_n16 + mid_n8]: bit_map[i] = 8
            for i in mid_sorted[mid_n16 + mid_n8 : ]: bit_map[i] = 4

            # ---- Quantize according to bit_map ----
            q_model = quantize_with_bit_map(
                base_model, bit_map, group_size=args.qgroup_size, cache_dequant=bool(args.cache_dequant)
            )
            q_model.eval()

            # Evaluate
            em, f1, n, _ = evaluate_triviaqa(label, model_name, q_model, tokenizer, eval_data, args.gen_len, args.save_directory, device)
            results_rows.append((model_name, label, em, f1, n))

            # Record bits & scores
            try:
                record_bit_allocation(model_name, label, bit_map, save_directory=args.save_directory)
                log_path = os.path.join(res_dir, f"{model_name.replace('/', '_')}_{label.lower()}_scores.json")
                with open(log_path, "w") as f:
                    json.dump({"info_raw": info_raw, "invvar_raw": invvar_raw, "combined": comb, "bit_map": bit_map}, f, indent=2)
                print(f"[Scores] Wrote scores to {log_path}")
            except Exception:
                pass

    # Print & save summary
    res_dir = os.path.join(args.save_directory, "results"); os.makedirs(res_dir, exist_ok=True)
    csv_path = os.path.join(res_dir, "summary.csv")

    print("\n===== SUMMARY =====")
    print(f"{'Model':40s} {'Method':20s} {'EM':>8s} {'F1':>8s} {'N':>6s}")
    for m, meth, em, f1, n in results_rows:
        print(f"{m:40s} {meth:20s} {100*em:8.2f} {100*f1:8.2f} {n:6d}")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["model", "method", "em", "f1", "n_examples"])
        for row in results_rows: w.writerow(row)

    print(f"\nSaved summary to {csv_path}")

if __name__ == "__main__":
    main()

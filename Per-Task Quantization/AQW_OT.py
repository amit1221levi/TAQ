#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AWQ_FINAL: "classic" Activation-Aware Weight Quantization (4/8-bit weight-only)
for meta-llama/Llama-3.1-8B-Instruct, with the same calibration & evaluation
pipeline as your previous script (TriviaQA; EM/F1; results saved identically).

No EOS assumptions. Manual padding (safe if PAD is missing).

srun -c 4 -A tdk -p tdk --gres=gpu:1 --pty \
python run_awq_final.py \
 --models    mistralai/Mistral-7B-Instruct-v0.2 \
  --dataset-name trivia_qa --dataset-config rc.nocontext \
  --calib-size 512 --eval-size 2048 --gen-len 64 \
  --save-directory models_and_tokenizers


srun -c 4 -A tdk -p tdk --gres=gpu:1 --pty \
  python awq_final.py \
 --models    mistralai/Mistral-7B-Instruct-v0.2 \
  --dataset-name trivia_qa --dataset-config rc.nocontext \
  --calib-size 512 --eval-size 2048 --gen-len 64 \
  --save-directory models_and_tokenizers

  
AWQ_final — AutoAWQ quantize-all on TriviaQA (rc.nocontext).

- Quantizes ALL transformer layers (no FP16 keep list).
- Uses raw calibration texts (Trivia-only) as in AutoAWQ repo.
- Greedy decode; SQuAD-style EM/F1 with alias handling.
- Avoids tokenizer shadowing by using `hf_tokenizer` (never a bool).
Install:
  pip install -U "transformers==4.51.3" "torch==2.6.*" datasets accelerate autoawq
"""

from __future__ import annotations
import argparse, os, json, random, string, re
from typing import List, Tuple, Iterable

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

try:
    from awq import AutoAWQForCausalLM   # package name: autoawq
except Exception:
    from autoawq import AutoAWQForCausalLM
    print("[Import] Using autoawq.AutoAWQForCausalLM import path")

# ---- Torch hygiene ----
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
random.seed(0)

# ---- CLI ----
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("AWQ_final (quantize-all) on TriviaQA")
    p.add_argument("--models", nargs="+", default=["meta-llama/Llama-3.1-8B-Instruct"])
    p.add_argument("--dataset-name", type=str, default="trivia_qa")
    p.add_argument("--dataset-config", type=str, default="rc.nocontext")
    p.add_argument("--calib-size", type=int, default=512)
    p.add_argument("--eval-size", type=int, default=2048)
    p.add_argument("--gen-len", type=int, default=64)
    p.add_argument("--save-directory", type=str, default="models_and_tokenizers")
    p.add_argument("--awq-wbit", type=int, default=4)
    p.add_argument("--awq-qgroup-size", type=int, default=128)
    p.add_argument("--awq-zero-point", action="store_true", default=True)
    p.add_argument("--awq-max-seq-len", type=int, default=2048)
    p.add_argument("--hf-token", type=str, default=None)
    return p

# ---- Helpers / metrics ----
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
    seen, out = set(), []
    for g in golds:
        if g not in seen:
            out.append(g); seen.add(g)
    return out if out else [""]

def sanitize_generation_config(model):
    gc = getattr(model, "generation_config", None)
    if gc is None: return
    if hasattr(gc, "do_sample"): gc.do_sample = False
    for k in ("temperature", "top_p", "top_k", "typical_p", "penalty_alpha"):
        if hasattr(gc, k): setattr(gc, k, None)

# ---- Data ----
def load_dataset_and_split(name: str, cfg: str, calib_size: int, eval_size: int, data_dir: str):
    os.makedirs(data_dir, exist_ok=True)
    cache = os.path.join(data_dir, f"{name}__{cfg}__validation.jsonl")
    if os.path.exists(cache):
        print(f"[Data] Using cached {cache}")
        ds = load_dataset("json", data_files=cache, split="train")
    else:
        print(f"[Data] Downloading {name}/{cfg}")
        ds = load_dataset(name, cfg, split="validation")
        ds.to_json(cache); print(f"[Data] Cached -> {cache}")
    n = len(ds)
    idx = list(range(n)); random.shuffle(idx)
    k = max(1, min(int(0.4 * n), n - 1))
    calib = ds.select(idx[:k]); eval_ = ds.select(idx[k:])
    if 0 < calib_size < len(calib): calib = calib.select(range(calib_size))
    if 0 < eval_size  < len(eval_):  eval_  = eval_.select(range(eval_size))
    print(f"[Data] calib={len(calib)} eval={len(eval_)} total={len(ds)}")
    return calib, eval_

def sample_trivia_texts(ds, k: int) -> List[str]:
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

# ---- AWQ quantize-all ----
def awq_final_quantize(model_id: str, hf_tokenizer: AutoTokenizer, calib_texts: List[str],
                       wbit: int, qgroup: int, zero_point: bool, max_seq_len: int,
                       save_root: str):
    out_dir = os.path.join(save_root, f"{model_id.replace('/', '_')}_awq_all")
    os.makedirs(out_dir, exist_ok=True)

    model = AutoAWQForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )

    # Safe padding defaults (avoid eos/pad confusion)
    if getattr(hf_tokenizer, "pad_token", None) is None:
        hf_tokenizer.pad_token = hf_tokenizer.eos_token
        hf_tokenizer.pad_token_id = hf_tokenizer.eos_token_id
    hf_tokenizer.padding_side = "right"

    quant_config = {
        "zero_point": bool(zero_point),
        "q_group_size": int(qgroup),
        "w_bit": int(wbit),
        "version": "GEMM",
    }
    print(f"[AWQ] Quantizing ALL layers (w_bit={wbit}, q_group={qgroup}, zero_point={zero_point})")
    model.quantize(
        hf_tokenizer,
        quant_config=quant_config,
        calib_data=calib_texts,
        max_calib_seq_len=int(max_seq_len),
    )
    model.save_quantized(out_dir)
    print(f"[AWQ] Saved -> {out_dir}")

    q_model = AutoAWQForCausalLM.from_quantized(
        out_dir, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
    ).eval()
    sanitize_generation_config(q_model)
    return q_model, out_dir

# ---- Evaluation ----
def evaluate(model_id: str, method_label: str, model, hf_tokenizer, eval_data, gen_len: int, save_root: str):
    os.makedirs(save_root, exist_ok=True)
    out_dir = os.path.join(save_root, "results"); os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{model_id.replace('/', '_')}_{method_label.lower()}_results.jsonl")

    N=0; em_sum=0.0; f1_sum=0.0
    with open(out_path, "w", encoding="utf-8") as f:
        for i, sample in enumerate(eval_data):
            q = sample.get("question") or sample.get("query") or sample.get("question_text") or ""
            golds = get_gold_aliases(sample)
            prompt = f"Question: {q}\nAnswer:"
            try:
                m_dev = next(model.parameters()).device
            except StopIteration:
                m_dev = torch.device("cpu")
            enc = hf_tokenizer(prompt, return_tensors="pt").to(m_dev)
            with torch.inference_mode():
                out = model.generate(
                    **enc, max_new_tokens=gen_len, do_sample=False, use_cache=True,
                    pad_token_id=hf_tokenizer.pad_token_id, eos_token_id=hf_tokenizer.eos_token_id
                )
            gen = hf_tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            em, f1 = best_em_f1(gen.split("\n")[0], golds)
            N += 1; em_sum += em; f1_sum += f1
            f.write(json.dumps({"model": model_id, "method": method_label,
                                "question": q, "golds": golds, "pred": gen, "em": em, "f1": f1}) + "\n")
            if (i+1) % 100 == 0:
                print(f"[{method_label}] {i+1}: EM {em_sum/N:.4f} | F1 {f1_sum/N:.4f}")

    print(f"[{method_label}] {model_id}  EM: {100*em_sum/max(1,N):.2f}% | F1: {100*f1_sum/max(1,N):.2f}% (N={N}) -> {out_path}")
    return out_path

# ---- Main ----
def main():
    args = build_parser().parse_args()
    os.makedirs(args.save_directory, exist_ok=True)

    calib, eval_ = load_dataset_and_split(args.dataset-name if False else args.dataset_name,
                                          args.dataset_config, args.calib_size, args.eval_size, "data")
    calib_texts = sample_trivia_texts(calib, args.calib_size)

    for model_id in args.models:
        print("\n==============================")
        print(f"  AWQ_final on {model_id}")
        print("==============================\n")

        tok_kwargs = {"trust_remote_code": True, "use_fast": False}
        if args.hf_token: tok_kwargs["token"] = args.hf_token
        hf_tokenizer = AutoTokenizer.from_pretrained(model_id, **tok_kwargs)

        q_model, _ = awq_final_quantize(
            model_id=model_id,
            hf_tokenizer=hf_tokenizer,
            calib_texts=calib_texts,
            wbit=args.awq_wbit,
            qgroup=args.awq_qgroup_size,
            zero_point=args.awq_zero_point,
            max_seq_len=args.awq_max_seq_len,
            save_root=os.path.join(args.save_directory, "awq_quant"),
        )

        evaluate(model_id, "AWQ_final", q_model, hf_tokenizer, eval_, args.gen_len, args.save_directory)

if __name__ == "__main__":
    main()

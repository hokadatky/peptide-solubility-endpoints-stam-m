"""Optional protein-language-model embeddings for the ablation CLI.

ProtT5 (Rostlab/prot_t5_xl_half_uniref50-enc, 1024-d) and ESM-2
(facebook/esm2_t12_35M_UR50D, 480-d), both masked mean-pooled over real residues
(special/boundary tokens excluded). Requires torch + transformers (the `prot5`
conda env); import stays lazy so the BASE-only path needs neither.

Embeddings are cached to <cache_dir>/<model_tag>_<sha1(sequence)>.npy so a second
run, or a shared cache across datasets, is free. CPU is fine (mean-pooling only).
"""
from __future__ import annotations
import hashlib
import os
import numpy as np

PROTT5_ID = "Rostlab/prot_t5_xl_half_uniref50-enc"
ESM2_ID = "facebook/esm2_t12_35M_UR50D"


def _tag(model_id: str) -> str:
    return model_id.split("/")[-1]


def _cache_path(cache_dir, tag, seq):
    h = hashlib.sha1(seq.encode()).hexdigest()[:16]
    return os.path.join(cache_dir, f"{tag}_{h}.npy")


def embed(sequences, which: str, cache_dir=".plm_cache", model_id=None, batch_tokens=8000):
    """Return an (n, d) float32 matrix of mean-pooled embeddings.
    `which` is 'prott5' or 'esm2'. Raises a clear error if torch/transformers are
    unavailable. Uses and fills a per-sequence npy cache."""
    which = which.lower()
    model_id = model_id or (PROTT5_ID if which == "prott5" else ESM2_ID)
    tag = _tag(model_id)
    os.makedirs(cache_dir, exist_ok=True)

    seqs = [s for s in sequences]
    out = [None] * len(seqs)
    todo = []
    for i, s in enumerate(seqs):
        p = _cache_path(cache_dir, tag, s)
        if os.path.exists(p):
            out[i] = np.load(p)
        else:
            todo.append(i)

    if todo:
        try:
            import torch
            from transformers import AutoTokenizer, AutoModel, T5Tokenizer, T5EncoderModel
        except Exception as e:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "Computing pLM embeddings needs torch + transformers (the 'prot5' "
                "conda env). Install requirements-plm.txt, or run with --features base "
                "only. Underlying error: %r" % (e,))
        torch.manual_seed(0)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        if which == "prott5":
            tok = T5Tokenizer.from_pretrained(model_id, do_lower_case=False)
            model = T5EncoderModel.from_pretrained(model_id).to(dev).eval()
        else:
            tok = AutoTokenizer.from_pretrained(model_id)
            model = AutoModel.from_pretrained(model_id).to(dev).eval()

        # sort by length for efficient batching; restore order via index
        order = sorted(todo, key=lambda i: len(seqs[i]))
        batch, blen = [], 0
        with torch.no_grad():
            def flush(bt):
                if not bt:
                    return
                if which == "prott5":
                    spaced = [" ".join(list(seqs[i])) for i in bt]
                    enc = tok(spaced, add_special_tokens=True, padding=True, return_tensors="pt")
                else:
                    enc = tok([seqs[i] for i in bt], add_special_tokens=True,
                              padding=True, return_tensors="pt")
                ids = enc["input_ids"].to(dev)
                att = enc["attention_mask"].to(dev)
                rep = model(input_ids=ids, attention_mask=att).last_hidden_state
                for row, i in enumerate(bt):
                    L = int(att[row].sum().item())
                    # exclude special tokens: ProtT5 has trailing EOS; ESM-2 has
                    # leading CLS + trailing EOS.
                    if which == "prott5":
                        vec = rep[row, :L - 1].mean(0)
                    else:
                        vec = rep[row, 1:L - 1].mean(0)
                    v = vec.float().cpu().numpy().astype(np.float32)
                    np.save(_cache_path(cache_dir, tag, seqs[i]), v)
                    out[i] = v
            for i in order:
                tl = len(seqs[i]) + 2
                if batch and blen + tl > batch_tokens:
                    flush(batch)
                    batch, blen = [], 0
                batch.append(i)
                blen += tl
            flush(batch)

    return np.vstack(out).astype(np.float32)

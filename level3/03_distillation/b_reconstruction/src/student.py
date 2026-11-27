"""Compact on-device student for the SoluProt soluble-expression endpoint.

Reconstructed from the protocol documented in the S7 manuscript Methods and the
S7 revision report; the original implementation was lost with its scratchpad.
See ../protocol/REPRODUCTION_PROTOCOL_20260822.md.

Architecture: learned residue embedding -> three parallel 1-D convolutions
(widths 3/5/7) -> global max+mean pooling -> concatenated with projected
transparent descriptors -> penultimate representation -> logit. The penultimate
representation is the feature-distillation target. Inference is teacher-free and
is additionally re-implemented in NumPy for the deployment check.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
PAD = 0
STOI = {aa: i + 1 for i, aa in enumerate(ALPHABET)}
VOCAB = len(ALPHABET) + 1
MAX_LEN = 1024  # sequences are truncated symmetrically (head + tail) beyond this


def encode(sequence: str, max_len: int = MAX_LEN) -> np.ndarray:
    ids = [STOI[c] for c in sequence if c in STOI]
    if len(ids) > max_len:  # keep both termini, which carry signal
        head = max_len // 2
        ids = ids[:head] + ids[-(max_len - head):]
    out = np.zeros(max_len, dtype=np.int64)
    out[: len(ids)] = ids
    return out


def encode_batch(sequences, max_len: int = MAX_LEN) -> np.ndarray:
    return np.stack([encode(s, max_len) for s in sequences])


class OnDeviceStudent(nn.Module):
    def __init__(
        self,
        emb_dim: int = 32,
        channels: int = 64,
        kernels=(3, 5, 7),
        desc_dim: int = 25,
        desc_proj: int = 32,
        penult: int = 160,
        feat_dim: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(VOCAB, emb_dim, padding_idx=PAD)
        self.convs = nn.ModuleList(
            nn.Conv1d(emb_dim, channels, k, padding=k // 2) for k in kernels
        )
        self.desc_proj = nn.Linear(desc_dim, desc_proj)
        self.penultimate = nn.Linear(2 * channels * len(kernels) + desc_proj, penult)
        self.norm = nn.LayerNorm(penult)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(penult, 1)
        # Feature-distillation head: penultimate -> low-rank ProtT5 projection.
        # Used only during training; excluded from the exported artifact.
        self.feature_head = nn.Linear(penult, feat_dim)

    def represent(self, tokens: torch.Tensor, descriptors: torch.Tensor) -> torch.Tensor:
        mask = (tokens != PAD).unsqueeze(1)
        x = self.embedding(tokens).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            h = torch.nn.functional.silu(conv(x))
            h = h.masked_fill(~mask, 0.0)
            length = mask.sum(dim=2).clamp(min=1)
            pooled.append(h.sum(dim=2) / length)
            pooled.append(h.masked_fill(~mask, -1e9).max(dim=2).values)
        d = torch.nn.functional.silu(self.desc_proj(descriptors))
        z = torch.cat(pooled + [d], dim=1)
        return self.norm(torch.nn.functional.silu(self.penultimate(z)))

    def forward(self, tokens: torch.Tensor, descriptors: torch.Tensor):
        z = self.represent(tokens, descriptors)
        return self.head(self.dropout(z)).squeeze(1), self.feature_head(z)


def deployable_parameters(model: OnDeviceStudent) -> int:
    """Parameters shipped for inference: the feature head is training-only."""
    total = sum(p.numel() for p in model.parameters())
    return total - sum(p.numel() for p in model.feature_head.parameters())


def numpy_forward(state: dict[str, np.ndarray], tokens: np.ndarray, descriptors: np.ndarray,
                  kernels=(3, 5, 7)) -> np.ndarray:
    """Teacher-free NumPy inference path (no torch, no network)."""
    def silu(v):
        return v / (1.0 + np.exp(-v))

    emb = state["embedding.weight"][tokens]              # (n, L, e)
    mask = (tokens != PAD)
    pooled = []
    for i, k in enumerate(kernels):
        w = state[f"convs.{i}.weight"]                    # (c, e, k)
        b = state[f"convs.{i}.bias"]
        pad = k // 2
        padded = np.pad(emb, ((0, 0), (pad, pad), (0, 0)))
        n, L, e = emb.shape
        c = w.shape[0]
        windows = np.lib.stride_tricks.sliding_window_view(padded, k, axis=1)  # (n,L,e,k)
        h = np.einsum("nlek,cek->nlc", windows[:, :L], w) + b
        h = silu(h)
        m = mask[:, :, None]
        h = np.where(m, h, 0.0)
        length = np.maximum(mask.sum(axis=1, keepdims=True), 1)
        pooled.append(h.sum(axis=1) / length)
        pooled.append(np.where(m, h, -1e9).max(axis=1))
    d = silu(descriptors @ state["desc_proj.weight"].T + state["desc_proj.bias"])
    z = np.concatenate(pooled + [d], axis=1)
    z = silu(z @ state["penultimate.weight"].T + state["penultimate.bias"])
    mu = z.mean(axis=1, keepdims=True)
    sd = z.std(axis=1, keepdims=True)
    z = (z - mu) / np.sqrt(sd ** 2 + 1e-5)
    z = z * state["norm.weight"] + state["norm.bias"]
    logit = (z @ state["head.weight"].T + state["head.bias"]).ravel()
    return 1.0 / (1.0 + np.exp(-logit))

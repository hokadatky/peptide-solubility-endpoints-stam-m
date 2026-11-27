"""Reusable, leakage-controlled representation-ablation core for peptide-property
prediction (companion code to the STAM-M review).

The question this answers, on ANY (id, sequence, label[, group]) table:
does adding a protein language model (ProtT5 / ESM-2) embedding to a transparent
baseline of amino-acid composition + physicochemical descriptors improve
prediction under a homology-controlled split, judged by a paired cluster
bootstrap against a prespecified margin?

Design rules enforced here (see the manuscript reporting standard):
  * every data-dependent transform (StandardScaler, supervised PLS) is fit on the
    training fold ONLY, inside the cross-validation loop -- never on all rows;
  * grouping keeps close analogues together: an explicit `group` column, else a
    deterministic greedy k-mer-similarity clustering (a light stand-in for a real
    MMseqs2/CD-HIT clustering, which callers are encouraged to supply instead);
  * uncertainty is a PAIRED CLUSTER bootstrap (resample clusters, not rows, so
    correlated analogues do not deflate the interval); the same resample scores
    both models;
  * a null is not read as equivalence -- the caller sets a practical margin.

Only aggregate numbers are produced. No third-party row-level data are shipped.
Pure sklearn/numpy/pandas; pLM embeddings are optional (see embeddings.py).
"""
from __future__ import annotations
import numpy as np

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_IDX = {a: i for i, a in enumerate(AA)}

# Kyte-Doolittle hydropathy
_KD = dict(A=1.8, R=-4.5, N=-3.5, D=-3.5, C=2.5, Q=-3.5, E=-3.5, G=-0.4, H=-3.2,
           I=4.5, L=3.8, K=-3.9, M=1.9, F=2.8, P=-1.6, S=-0.8, T=-0.7, W=-0.9,
           Y=-1.3, V=4.2)
# average residue masses (Da)
_MW = dict(A=71.08, R=156.19, N=114.10, D=115.09, C=103.14, E=129.12, Q=128.13,
           G=57.05, H=137.14, I=113.16, L=113.16, K=128.17, M=131.19, F=147.18,
           P=97.12, S=87.08, T=101.10, W=186.21, Y=163.18, V=99.13)
# side-chain pKa (Henderson-Hasselbalch), plus termini
_PKA_POS = dict(K=10.5, R=12.5, H=6.0)      # protonated -> positive
_PKA_NEG = dict(D=3.9, E=4.1, C=8.3, Y=10.1)  # deprotonated -> negative
_PKA_NT, _PKA_CT = 9.0, 2.0


def clean_sequence(seq: str) -> str:
    """Upper-case, strip whitespace, keep only the 20 canonical residues.
    Non-canonical characters are DROPPED with the understanding (central to the
    review) that modified/cyclic chemistry is not faithfully represented here."""
    return "".join(c for c in str(seq).strip().upper() if c in AA_IDX)


def _charge_at_ph(seq: str, ph: float = 7.0) -> float:
    pos = 1.0 / (1.0 + 10 ** (ph - _PKA_NT))
    neg = 1.0 / (1.0 + 10 ** (_PKA_CT - ph))
    for a in seq:
        if a in _PKA_POS:
            pos += 1.0 / (1.0 + 10 ** (ph - _PKA_POS[a]))
        elif a in _PKA_NEG:
            neg += 1.0 / (1.0 + 10 ** (_PKA_NEG[a] - ph))
    return pos - neg


PHYS_NAMES = ["length", "mol_weight", "gravy", "charge_ph7", "charge_density",
              "aromaticity", "aliphatic_index", "frac_positive", "frac_negative",
              "frac_polar", "max_kd_window9"]


def physicochem(seq: str) -> np.ndarray:
    """Transparent, auditable physicochemical descriptors (no fitting)."""
    n = len(seq)
    if n == 0:
        return np.zeros(len(PHYS_NAMES), np.float32)
    mw = sum(_MW[a] for a in seq) + 18.02
    gravy = np.mean([_KD[a] for a in seq])
    ch = _charge_at_ph(seq, 7.0)
    arom = sum(seq.count(a) for a in "FWY") / n
    # aliphatic index (Ikai 1980), mole-percent form
    ali = (seq.count("A") + 2.9 * seq.count("V") + 3.9 * (seq.count("I") + seq.count("L"))) / n * 100.0
    fpos = sum(seq.count(a) for a in "KRH") / n
    fneg = sum(seq.count(a) for a in "DE") / n
    fpol = sum(seq.count(a) for a in "STNQ") / n
    # most hydrophobic 9-residue window (local burden)
    if n >= 9:
        kd = np.array([_KD[a] for a in seq])
        max_kd9 = max(kd[i:i + 9].mean() for i in range(n - 8))
    else:
        max_kd9 = gravy
    return np.array([n, mw, gravy, ch, ch / n, arom, ali, fpos, fneg, fpol, max_kd9],
                    np.float32)


def composition(seq: str) -> np.ndarray:
    v = np.zeros(len(AA), np.float32)
    for a in seq:
        v[AA_IDX[a]] += 1.0
    n = len(seq)
    return v / n if n else v


def base_features(seqs):
    """BASE = amino-acid composition (20) + physicochemical descriptors."""
    comp = np.vstack([composition(s) for s in seqs])
    phys = np.vstack([physicochem(s) for s in seqs])
    return np.hstack([comp, phys]).astype(np.float32)


# ---------------------------------------------------------------------------
# grouping (homology-aware split)
# ---------------------------------------------------------------------------
def _kmers(seq: str, k: int = 3):
    return {seq[i:i + k] for i in range(len(seq) - k + 1)} if len(seq) >= k else {seq}


def greedy_cluster(seqs, threshold: float = 0.6, k: int = 3):
    """Deterministic greedy single-pass clustering by k-mer Jaccard similarity.
    A lightweight stand-in for real homology clustering; supply an explicit
    `group` column (e.g. from MMseqs2) for publication-grade splits. Returns an
    integer cluster id per input, order-stable."""
    reps = []          # (kmer_set, cluster_id)
    labels = np.empty(len(seqs), int)
    for i, s in enumerate(seqs):
        ks = _kmers(s, k)
        best, best_j = -1, 0.0
        for ridx, (rks, _cid) in enumerate(reps):
            u = len(ks | rks)
            j = len(ks & rks) / u if u else 0.0
            if j > best_j:
                best_j, best = j, ridx
        if best_j >= threshold:
            labels[i] = reps[best][1]
        else:
            labels[i] = len(reps)
            reps.append((ks, len(reps)))
    return labels


def make_groups(seqs, group=None, threshold=0.6):
    if group is not None:
        # deterministic dense integer ids
        order = {g: i for i, g in enumerate(dict.fromkeys(group))}
        return np.array([order[g] for g in group], int)
    return greedy_cluster(list(seqs), threshold=threshold)


# ---------------------------------------------------------------------------
# out-of-fold predictions (leakage-controlled)
# ---------------------------------------------------------------------------
def _new_learner(task):
    if task == "classification":
        from sklearn.ensemble import HistGradientBoostingClassifier
        # early stopping keeps small-n runs fast and large-n runs well-fit;
        # deterministic given random_state.
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                              max_depth=3, early_stopping=True,
                                              n_iter_no_change=15, validation_fraction=0.15,
                                              random_state=0)
    from sklearn.linear_model import RidgeCV
    return RidgeCV(alphas=np.logspace(-3, 3, 25))


def oof_predictions(X, y, groups, task, pls_block=None, pls_k=0, n_splits=5,
                    repeats=10, seed=0):
    """Repeated GroupKFold out-of-fold predictions. Standardization and the
    optional supervised PLS projection are fit on the TRAIN fold only.

    X          : base feature matrix (n, d_base)
    pls_block  : optional embedding matrix (n, d_emb); if given it is appended.
    pls_k      : if >0, the embedding is replaced by a k-component supervised PLS
                 projection fit on train (the low-dimensional deploy form);
                 if 0 the raw embedding is standardized and concatenated.
    Returns    : (oof_mean over repeats, per-repeat oof matrix) of shape (n,).
    """
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.cross_decomposition import PLSRegression
    n = len(y)
    ug = np.unique(groups)
    k = min(n_splits, len(ug))
    per_rep = np.full((repeats, n), np.nan)
    for rep in range(repeats):
        rng = np.random.RandomState(seed + rep)
        # shuffle group->fold assignment deterministically per repeat
        perm = rng.permutation(len(ug))
        gmap = {g: perm[i] % k for i, g in enumerate(ug)}
        fold_of = np.array([gmap[g] for g in groups])
        for f in range(k):
            tr = np.where(fold_of != f)[0]
            te = np.where(fold_of == f)[0]
            if len(te) == 0 or len(np.unique(y[tr])) < (2 if task == "classification" else 1):
                continue
            sc = StandardScaler().fit(X[tr])
            Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
            if pls_block is not None:
                B = pls_block
                if pls_k and pls_k > 0:
                    ps = StandardScaler().fit(B[tr])
                    Btr_s, Bte_s = ps.transform(B[tr]), ps.transform(B[te])
                    pls = PLSRegression(n_components=min(pls_k, Btr_s.shape[1]))
                    yt = y[tr].astype(float)
                    pls.fit(Btr_s, yt)
                    Btr, Bte = pls.transform(Btr_s), pls.transform(Bte_s)
                else:
                    bs = StandardScaler().fit(B[tr])
                    Btr, Bte = bs.transform(B[tr]), bs.transform(B[te])
                Xtr = np.hstack([Xtr, Btr])
                Xte = np.hstack([Xte, Bte])
            m = _new_learner(task)
            m.fit(Xtr, y[tr])
            if task == "classification":
                per_rep[rep, te] = m.predict_proba(Xte)[:, 1]
            else:
                per_rep[rep, te] = m.predict(Xte)
    return np.nanmean(per_rep, axis=0), per_rep


# ---------------------------------------------------------------------------
# metric + paired CLUSTER bootstrap
# ---------------------------------------------------------------------------
def _metric(y, p, task):
    if task == "classification":
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(y, p)
    return float(np.sqrt(np.mean((y - p) ** 2)))


def paired_cluster_bootstrap(y, groups, pred_a, pred_b, task, n_boot=4000, seed=0):
    """Delta = metric(b) - metric(a), 95% CI by resampling CLUSTERS with
    replacement (the same resample scores both models -> paired)."""
    rng = np.random.RandomState(seed)
    ug = np.unique(groups)
    idx_by_g = {g: np.where(groups == g)[0] for g in ug}
    base = _metric(y, pred_b, task) - _metric(y, pred_a, task)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        gs = rng.choice(ug, len(ug), replace=True)
        idx = np.concatenate([idx_by_g[g] for g in gs])
        yy = y[idx]
        if task == "classification" and len(np.unique(yy)) < 2:
            deltas[b] = np.nan
            continue
        deltas[b] = _metric(yy, pred_b[idx], task) - _metric(yy, pred_a[idx], task)
    deltas = deltas[~np.isnan(deltas)]
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return float(base), float(lo), float(hi)


def detect_task(y):
    u = np.unique(y[~np.isnan(y)]) if y.dtype.kind == "f" else np.unique(y)
    return "classification" if len(u) <= 2 else "regression"

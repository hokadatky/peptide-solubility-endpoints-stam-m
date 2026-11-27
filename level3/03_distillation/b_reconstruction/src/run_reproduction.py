"""Independent reconstruction of the 2026-08-16 on-device distillation, fold 1.

Protocol: ../protocol/REPRODUCTION_PROTOCOL_20260822.md (written before fitting).
This does NOT authorize changing any manuscript number.
"""
from __future__ import annotations

import json, sys, time
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from _harness import load, PLAN_SRC  # noqa: E402
from student import (ALPHABET, OnDeviceStudent, deployable_parameters,  # noqa: E402
                     encode_batch, numpy_forward)

sys.path.insert(0, str(PLAN_SRC))
import teacher_crossfit as tc  # noqa: E402

SEED = 20260813
MAX_LEN = 768
ALPHA, TEMP, BETA = 0.5, 4.0, 0.1
EPOCHS, BATCH, LR, WD, CLIP = 40, 128, 3e-3, 1e-4, 1.0
BOOT = 10000
OUT = Path(__file__).resolve().parents[1] / "results"
LOGIT_KW = dict(C=1000.0, solver="lbfgs", max_iter=20000)


def logistic_probs(X, y, tr, te):
    sc = StandardScaler().fit(X[tr])
    m = LogisticRegression(**LOGIT_KW).fit(sc.transform(X[tr]), y[tr])
    return m.predict_proba(sc.transform(X[te]))[:, 1]


def metrics(y, p):
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))),
    }


def train_student(tokens, desc, y, soft, feat, train_idx, *, distil: bool):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(8)
    model = OnDeviceStudent()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    tok = torch.from_numpy(tokens[train_idx])
    d = torch.from_numpy(desc[train_idx])
    yy = torch.from_numpy(y[train_idx].astype(np.float32))
    st = torch.from_numpy(soft[train_idx].astype(np.float32))
    ft = torch.from_numpy(feat[train_idx].astype(np.float32))
    n = len(train_idx)
    rng = np.random.default_rng(SEED)
    model.train()
    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            b = order[s:s + BATCH]
            logit, fpred = model(tok[b], d[b])
            hard = torch.nn.functional.binary_cross_entropy_with_logits(logit, yy[b])
            if distil:
                tl = torch.log(st[b].clamp(1e-6, 1 - 1e-6)) - torch.log1p(-st[b].clamp(1e-6, 1 - 1e-6))
                sp = torch.sigmoid(logit / TEMP)
                tp = torch.sigmoid(tl / TEMP)
                kd = torch.nn.functional.binary_cross_entropy(sp.clamp(1e-6, 1 - 1e-6), tp)
                fl = torch.nn.functional.mse_loss(fpred, ft[b])
                loss = (1 - ALPHA) * hard + ALPHA * (TEMP ** 2) * kd + BETA * fl
            else:
                loss = hard
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step()
        if epoch % 10 == 9:
            print(f"    epoch {epoch+1}/{EPOCHS} loss {float(loss):.4f}", flush=True)
    model.eval()
    return model


@torch.no_grad()
def predict(model, tokens, desc, idx):
    out = []
    for s in range(0, len(idx), 256):
        b = idx[s:s + 256]
        logit, _ = model(torch.from_numpy(tokens[b]), torch.from_numpy(desc[b]))
        out.append(torch.sigmoid(logit).numpy())
    return np.concatenate(out)


def paired_bootstrap(y, groups, pa, pb, seed=SEED, reps=BOOT):
    """Cluster bootstrap of AUC(a) - AUC(b) over 30%-identity groups."""
    uniq = np.unique(groups)
    idx_by_group = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(seed)
    deltas = np.empty(reps)
    k = 0
    for _ in range(reps):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx_by_group[g] for g in pick])
        yy = y[rows]
        if yy.min() == yy.max():
            continue
        deltas[k] = roc_auc_score(yy, pa[rows]) - roc_auc_score(yy, pb[rows])
        k += 1
    deltas = deltas[:k]
    return {
        "delta": float(np.mean(deltas)),
        "ci95": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
        "valid_replicates": int(k),
    }


def main() -> None:
    t0 = time.time()
    config, data, prott5 = load()
    y, groups, folds = data.labels, data.groups, data.folds
    train_folds = [0, 2, 3, 4, 5, 6]
    tr = np.flatnonzero(np.isin(folds, train_folds))
    te = np.flatnonzero(folds == 1)
    print(f"train {len(tr)} rows / confirmation fold1 {len(te)} rows, "
          f"{len(np.unique(groups[te]))} groups", flush=True)

    # --- reference arms -------------------------------------------------
    p_b1 = logistic_probs(data.b1, y, tr, te)
    p_teacher = logistic_probs(prott5, y, tr, te)
    print(f"B1 {roc_auc_score(y[te], p_b1):.4f} | teacher {roc_auc_score(y[te], p_teacher):.4f}",
          flush=True)

    # --- teacher soft targets, cross-fitted WITHIN training folds only ---
    soft = np.zeros(len(y), dtype=np.float64)
    for sl in tc.make_crossfit_plan(folds, groups, train_folds):
        soft[sl.target_indices] = logistic_probs(prott5, y, sl.train_indices, sl.target_indices)
        print(f"  crossfit heldout fold {sl.heldout_fold}: {len(sl.target_indices)} rows",
              flush=True)

    # --- feature-distillation target: fold-local low-rank ProtT5 ---------
    sc = StandardScaler().fit(prott5[tr])
    pca = PCA(n_components=32, random_state=SEED).fit(sc.transform(prott5[tr]))
    feat_all = np.zeros((len(y), 32), dtype=np.float32)
    feat_all[tr] = pca.transform(sc.transform(prott5[tr]))
    feat_all = feat_all / (feat_all[tr].std(axis=0, keepdims=True) + 1e-8)

    tokens = encode_batch(data.sequences, MAX_LEN)
    desc = data.b0.astype(np.float32)
    desc = (desc - desc[tr].mean(0)) / (desc[tr].std(0) + 1e-8)

    results = {}
    for name, distil in (("distilled", True), ("hard_only", False)):
        print(f"  fitting {name} student ...", flush=True)
        t = time.time()
        model = train_student(tokens, desc, y, soft, feat_all, tr, distil=distil)
        p = predict(model, tokens, desc, te)
        results[name] = {"probs": p, "model": model, "metrics": metrics(y[te], p)}
        print(f"  {name}: AUC {results[name]['metrics']['roc_auc']:.4f} "
              f"({time.time()-t:.0f}s)", flush=True)

    p_d, p_h = results["distilled"]["probs"], results["hard_only"]["probs"]
    payload = {
        "schema": "ondevice_distillation_reproduction_v1",
        "status": "INDEPENDENT_RECONSTRUCTION_NOT_THE_ORIGINAL_PREREGISTERED_RUN",
        "protocol": "protocol/REPRODUCTION_PROTOCOL_20260822.md (written before fitting)",
        "endpoint": "SoluProt recombinant soluble-expression status in E. coli; "
                    "not direct aqueous solubility",
        "confirmation_fold": 1,
        "confirmation_rows": int(len(te)),
        "confirmation_groups": int(len(np.unique(groups[te]))),
        "auc": {
            "teacher_ProtT5_plain_logistic": float(roc_auc_score(y[te], p_teacher)),
            "B1_canonical_contract": float(roc_auc_score(y[te], p_b1)),
            "distilled_student": results["distilled"]["metrics"]["roc_auc"],
            "hard_only_student": results["hard_only"]["metrics"]["roc_auc"],
        },
        "calibration_distilled": {
            "brier": results["distilled"]["metrics"]["brier"],
            "log_loss": results["distilled"]["metrics"]["log_loss"],
        },
        "contrasts_fold1": {
            "distilled_minus_hard_only": paired_bootstrap(y[te], groups[te], p_d, p_h),
            "distilled_minus_B1_canonical": paired_bootstrap(y[te], groups[te], p_d, p_b1),
        },
        "student_parameters_deployable": deployable_parameters(results["distilled"]["model"]),
        "reported_2026_08_16": {
            "teacher": 0.780, "B1": 0.701, "distilled": 0.730, "hard_only": 0.708,
            "student_parameters": 100719,
        },
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    tg = payload["auc"]["teacher_ProtT5_plain_logistic"] - payload["auc"]["B1_canonical_contract"]
    payload["teacher_gain_retention_fraction_vs_canonical_B1"] = float(
        (payload["auc"]["distilled_student"] - payload["auc"]["B1_canonical_contract"]) / tg
    )
    OUT.mkdir(exist_ok=True)
    (OUT / "ONDEVICE_REPRODUCTION_20260822.json").write_text(
        json.dumps(payload, indent=1), encoding="utf-8")

    state = {k: v.detach().numpy() for k, v in results["distilled"]["model"].state_dict().items()}
    np.savez_compressed(OUT / "distilled_student_state.npz", **state)
    probe = te[:64]
    np_p = numpy_forward(state, tokens[probe], desc[probe])
    payload["numpy_torch_max_abs_probability_difference"] = float(np.max(np.abs(np_p - p_d[:64])))
    (OUT / "ONDEVICE_REPRODUCTION_20260822.json").write_text(
        json.dumps(payload, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "reported_2026_08_16"},
                     indent=1), flush=True)


if __name__ == "__main__":
    main()

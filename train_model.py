"""Reproducible training pipeline for the VentureIQ startup-outcome model.

Usage (from the project root):
    python train_model.py                       # synthetic DEMONSTRATION data
    python train_model.py --data my.csv --data-note "Source, licence, target definition"

Outputs (in ``models/`` by default):
    model.joblib, model_metadata.json, feature_schema.json, evaluation_report.md
    data/synthetic_startups.csv (only when synthetic data is generated)

IMPORTANT: the built-in data generator creates *synthetic* records whose label is drawn
from a hand-written formula. Metrics measured on it describe how well the model recovers
that formula, NOT how well it predicts real startup outcomes.
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from config import APP_VERSION, get_settings
from schemas import BINARY_FEATURES, FEATURE_LABELS, FEATURES, TARGET_COLUMN
from utils import sha256_file, utc_now_iso

log = logging.getLogger("ventureiq.train")

SYNTHETIC_TARGET_DEFINITION = (
    "SYNTHETIC. 'success' is a Bernoulli draw whose probability is sigmoid(z), where z is a "
    "hand-written linear formula of log funding, milestones, relationships, investor flags and "
    "round progress, plus Gaussian noise. It is NOT derived from observed startup outcomes."
)


# ------------------------------------------------------------------ data
def generate_synthetic(n: int, seed: int) -> pd.DataFrame:
    """Generate internally consistent synthetic startup records (demonstration only)."""
    rng = np.random.default_rng(seed)
    rounds = np.where(rng.random(n) < 0.10, 0, np.clip(1 + rng.poisson(1.4, n), 1, 8))
    has_round = rounds > 0
    angel = (rng.random(n) < 0.35) & has_round
    vc = (rng.random(n) < np.clip(0.15 + 0.08 * rounds, 0, 0.8)) & has_round
    series_a = (rounds >= 2) & (rng.random(n) < 0.5)
    series_b = series_a & (rounds >= 3) & (rng.random(n) < 0.4)
    mu = np.log(400_000) + 0.7 * rounds + 0.8 * vc + 0.6 * series_a + 0.8 * series_b
    funding = np.where(has_round, np.exp(rng.normal(mu, 0.6)), 0.0).round(0)
    relationships = rng.poisson(2 + 1.1 * rounds + 2.0 * vc)
    milestones = rng.poisson(0.8 + 0.7 * rounds + 0.08 * relationships)
    z = (-2.2 + 0.55 * np.log1p(funding / 1e6) + 0.28 * np.clip(milestones, 0, 12)
         + 0.07 * np.clip(relationships, 0, 25) + 0.45 * vc + 0.25 * angel
         + 0.60 * series_a + 0.50 * series_b + 0.10 * rounds + rng.normal(0, 0.7, n))
    success = rng.random(n) < 1 / (1 + np.exp(-z))
    return pd.DataFrame({
        "total_funding_usd": funding, "funding_rounds": rounds, "relationships": relationships,
        "milestones": milestones, "has_vc": vc.astype(int), "has_angel": angel.astype(int),
        "has_series_a": series_a.astype(int), "has_series_b": series_b.astype(int),
        TARGET_COLUMN: success.astype(int),
    })


def validate_dataset(df: pd.DataFrame) -> list[str]:
    """Return data-quality problems; raises ValueError for fatal ones."""
    missing = [c for c in (*FEATURES, TARGET_COLUMN) if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    notes: list[str] = []
    if df[[*FEATURES, TARGET_COLUMN]].isna().any().any():
        raise ValueError("Dataset contains missing values in required columns; clean or impute them first.")
    if (df[list(FEATURES)] < 0).any().any():
        raise ValueError("Dataset contains negative feature values.")
    for c in (*BINARY_FEATURES, TARGET_COLUMN):
        if not set(df[c].unique()) <= {0, 1}:
            raise ValueError(f"Column '{c}' must be binary (0/1).")
    if df[TARGET_COLUMN].nunique() < 2:
        raise ValueError("Target has a single class; cannot train a classifier.")
    if len(df) < 200:
        raise ValueError("At least 200 rows are required.")
    dup = int(df.duplicated().sum())
    if dup:
        notes.append(f"{dup} exact duplicate rows present; duplicates may leak between train and test.")
    rate = float(df[TARGET_COLUMN].mean())
    if rate < 0.1 or rate > 0.9:
        notes.append(f"Severe class imbalance (positive rate {rate:.1%}).")
    return notes


# ------------------------------------------------------------------ modelling
def _log_funding(X):
    X = np.asarray(X, dtype=float).copy()
    X[:, 0] = np.log1p(X[:, 0])
    return X


def build_gb(seed: int) -> GradientBoostingClassifier:
    return GradientBoostingClassifier(n_estimators=150, learning_rate=0.08, max_depth=3,
                                      subsample=0.9, random_state=seed)


def build_logistic(seed: int) -> Pipeline:
    return Pipeline([("log", FunctionTransformer(_log_funding)), ("scale", StandardScaler()),
                     ("lr", LogisticRegression(max_iter=1000, random_state=seed))])


def _metrics(y_true, proba, threshold: float = 0.5) -> dict[str, Any]:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "threshold": threshold,
    }


def _calibration(y_true, proba, bins: int = 10) -> dict[str, Any]:
    frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=bins, strategy="quantile")
    counts, edges = np.histogram(proba, bins=np.quantile(proba, np.linspace(0, 1, len(frac_pos) + 1)))
    ece = float(np.sum(np.abs(frac_pos - mean_pred) * (counts / counts.sum()))) if counts.sum() else None
    return {"mean_predicted": [float(x) for x in mean_pred], "fraction_positive": [float(x) for x in frac_pos],
            "expected_calibration_error": ece, "bins": int(len(frac_pos)), "strategy": "quantile"}


def _feature_stats(X: pd.DataFrame) -> dict[str, Any]:
    stats = {}
    for c in FEATURES:
        s = X[c]
        stats[c] = {"min": float(s.min()), "max": float(s.max()), "p01": float(s.quantile(0.01)),
                    "p50": float(s.median()), "p99": float(s.quantile(0.99)), "mean": float(s.mean())}
    return stats


def train(data: pd.DataFrame | None = None, *, n_samples: int = 8000, seed: int | None = None,
          models_dir: Path | None = None, data_dir: Path | None = None, cv_folds: int = 5,
          data_note: str | None = None, data_path: Path | None = None) -> dict[str, Any]:
    """Train, evaluate and persist the model. Returns the metadata dictionary."""
    s = get_settings()
    seed = s.seed if seed is None else seed
    models_dir = Path(models_dir or s.models_dir)
    data_dir = Path(data_dir or s.data_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    is_synthetic = data is None
    if is_synthetic:
        data = generate_synthetic(n_samples, seed)
        data_dir.mkdir(parents=True, exist_ok=True)
        data_path = data_dir / "synthetic_startups.csv"
        data.to_csv(data_path, index=False)
        data_note = ("Synthetic demonstration data generated by train_model.generate_synthetic "
                     f"(seed={seed}). No real startups are represented.")
        target_def = SYNTHETIC_TARGET_DEFINITION
    else:
        target_def = data_note or "User-supplied dataset; target definition not documented."
        if not data_note:
            log.warning("No --data-note supplied: provenance and licence of the dataset are undocumented.")
    quality_notes = validate_dataset(data)
    data_sha = sha256_file(data_path) if data_path and Path(data_path).is_file() else None

    X, y = data[list(FEATURES)].astype(float), data[TARGET_COLUMN].astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)

    gb = build_gb(seed)
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    cv = cross_validate(build_gb(seed), X_tr, y_tr, cv=skf, scoring=["roc_auc", "accuracy", "f1"])
    gb.fit(X_tr, y_tr)

    dummy = DummyClassifier(strategy="prior").fit(X_tr, y_tr)
    logit = build_logistic(seed).fit(X_tr, y_tr)
    metrics = {
        "gradient_boosting": _metrics(y_te, gb.predict_proba(X_te)[:, 1]),
        "logistic_regression": _metrics(y_te, logit.predict_proba(X_te)[:, 1]),
        "dummy_prior": _metrics(y_te, dummy.predict_proba(X_te)[:, 1]),
    }
    calibration = _calibration(y_te, gb.predict_proba(X_te)[:, 1])
    perm = permutation_importance(gb, X_te, y_te, n_repeats=10, random_state=seed, scoring="roc_auc")
    perm_list = sorted(
        ({"feature": f, "label": FEATURE_LABELS[f], "mean_auc_drop": float(m), "std": float(sd)}
         for f, m, sd in zip(FEATURES, perm.importances_mean, perm.importances_std)),
        key=lambda d: -d["mean_auc_drop"])

    # Reference profile for explanations: medians for numeric features, 0 (absent) for flags.
    reference = {f: (0.0 if f in BINARY_FEATURES else float(X_tr[f].median())) for f in FEATURES}

    model_path = models_dir / "model.joblib"
    joblib.dump(gb, model_path)
    meta = {
        "app_version": APP_VERSION,
        "model_version": f"{utc_now_iso()[:10]}-{(data_sha or 'nodata')[:8]}-s{seed}",
        "created_at": utc_now_iso(),
        "algorithm": "GradientBoostingClassifier",
        "params": gb.get_params(),
        "feature_order": list(FEATURES),
        "is_synthetic": is_synthetic,
        "data_note": data_note,
        "target_definition": target_def,
        "data_sha256": data_sha,
        "data_quality_notes": quality_notes,
        "n_rows": int(len(data)), "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
        "positive_rate": float(y.mean()),
        "seed": seed, "cv_folds": cv_folds,
        "cv": {k.replace("test_", ""): {"mean": float(np.mean(v)), "std": float(np.std(v))}
               for k, v in cv.items() if k.startswith("test_")},
        "metrics_test": metrics,
        "calibration": calibration,
        "permutation_importance": perm_list,
        "impurity_importance": {f: float(v) for f, v in zip(FEATURES, gb.feature_importances_)},
        "feature_stats": _feature_stats(X_tr),
        "reference_values": reference,
        "risk_thresholds": {"lower_min": s.risk_lower_min, "moderate_min": s.risk_moderate_min},
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__, "pandas_version": pd.__version__,
        "python_version": platform.python_version(),
        "model_sha256": sha256_file(model_path),
    }
    (models_dir / "model_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    schema = {"features": list(FEATURES), "labels": FEATURE_LABELS, "binary": list(BINARY_FEATURES),
              "target": TARGET_COLUMN, "funding_unit": "USD (raw dollars, not log-scaled)"}
    (models_dir / "feature_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    (models_dir / "evaluation_report.md").write_text(render_report(meta), encoding="utf-8")
    log.info("Model saved to %s (%s)", model_path, meta["model_version"])
    return meta


def render_report(meta: dict[str, Any]) -> str:
    m = meta["metrics_test"]
    rows = "\n".join(
        f"| {name} | {v['accuracy']:.3f} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} | {v['roc_auc']:.3f} | {v['brier']:.3f} |"
        for name, v in m.items())
    cm = m["gradient_boosting"]["confusion_matrix"]
    banner = ("> **DEMONSTRATION MODEL – trained on synthetic data.** Metrics below measure recovery of a "
              "hand-written formula, not prediction of real startup outcomes.\n\n") if meta["is_synthetic"] else ""
    perm = "\n".join(f"- {p['label']}: {p['mean_auc_drop']:.4f} ± {p['std']:.4f}" for p in meta["permutation_importance"])
    return f"""# VentureIQ model evaluation report

{banner}- Model version: `{meta['model_version']}`
- Created: {meta['created_at']}
- Algorithm: {meta['algorithm']} (seed {meta['seed']})
- Rows: {meta['n_rows']} (train {meta['n_train']}, test {meta['n_test']}); positive rate {meta['positive_rate']:.3f}
- Data note: {meta['data_note']}
- Target definition: {meta['target_definition']}
- Data quality notes: {'; '.join(meta['data_quality_notes']) or 'none'}

## Held-out test metrics (threshold 0.5)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Brier |
|---|---|---|---|---|---|---|
{rows}

Confusion matrix (gradient boosting): TN={cm['tn']}, FP={cm['fp']}, FN={cm['fn']}, TP={cm['tp']}

## Cross-validation on the training split ({meta['cv_folds']}-fold, stratified)

""" + "\n".join(f"- {k}: {v['mean']:.3f} ± {v['std']:.3f}" for k, v in meta["cv"].items()) + f"""

## Calibration

Expected calibration error ({meta['calibration']['bins']} quantile bins): {meta['calibration']['expected_calibration_error']:.4f}

## Permutation importance (mean ROC-AUC drop on test set)

{perm}

## Environment

scikit-learn {meta['sklearn_version']}, numpy {meta['numpy_version']}, pandas {meta['pandas_version']}, Python {meta['python_version']}

## Interpretation limits

Permutation importance and any per-prediction explanation describe the *model's learned associations*,
not causal effects on real-world startup success.
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, help="CSV with the 8 feature columns and a 0/1 'success' column")
    ap.add_argument("--data-note", help="Provenance, licence and target definition of --data (strongly recommended)")
    ap.add_argument("--n-samples", type=int, default=8000, help="Synthetic rows to generate (default 8000)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--models-dir", type=Path)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        df = pd.read_csv(args.data) if args.data else None
        meta = train(df, n_samples=args.n_samples, seed=args.seed, models_dir=args.models_dir,
                     cv_folds=args.cv_folds, data_note=args.data_note, data_path=args.data)
    except (ValueError, OSError) as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        return 1
    g = meta["metrics_test"]["gradient_boosting"]
    print(f"Saved model {meta['model_version']} | synthetic={meta['is_synthetic']} | "
          f"test ROC-AUC={g['roc_auc']:.3f} accuracy={g['accuracy']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

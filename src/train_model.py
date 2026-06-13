from __future__ import annotations

import sys
import time
import traceback
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")

# ── Resolve project root ───────────────────────────────────────────────────────
_FILE_DIR    = Path(__file__).resolve().parent
PROJECT_ROOT = _FILE_DIR.parent if _FILE_DIR.name == "src" else _FILE_DIR
SRC_DIR      = _FILE_DIR

DATA_DIR   = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

for _p in [str(PROJECT_ROOT), str(SRC_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── Local imports ─────────────────────────────────────────────────────────────
from src.feature_engineering import (
    ML_FEATURE_COLS,
    TARGET_COL,
    build_feature_matrix,
    encode_features,
    engineer_features,
    scale_features,
)

# ═════════════════════════════════════════════════════════════════════════════
# Artefact paths  ← PUBLIC (imported by pages)
# ═════════════════════════════════════════════════════════════════════════════

BEST_MODEL_PATH  = MODELS_DIR / "best_model.pkl"
LEADERBOARD_PATH = MODELS_DIR / "leaderboard.csv"

_ARTEFACT_FILES = {
    "model":       BEST_MODEL_PATH,
    "encoders":    MODELS_DIR / "label_encoders.pkl",
    "scaler":      MODELS_DIR / "scaler.pkl",
    "leaderboard": LEADERBOARD_PATH,
    "feat_imp":    MODELS_DIR / "feature_importance.csv",
    "model_name":  MODELS_DIR / "best_model_name.txt",
}


# ═════════════════════════════════════════════════════════════════════════════
# Column-name aliases
# ═════════════════════════════════════════════════════════════════════════════

_RATING_ALIASES = [
    "Aggregate rating", "aggregate_rating", "Rating", "rating", "aggregate rating",
]


# ═════════════════════════════════════════════════════════════════════════════
# Data loading helpers (used by main.py CLI)
# ═════════════════════════════════════════════════════════════════════════════

def _find_csv(data_dir: Path) -> Path | None:
    preferred_names = [
        "featured_dataset.csv", "cleaned_dataset.csv",
        "Dataset_.csv", "Dataset.csv", "dataset.csv",
        "zomato.csv", "restaurants.csv",
    ]
    search_dirs = [
        data_dir, PROJECT_ROOT / "data", SRC_DIR / "data",
        PROJECT_ROOT, SRC_DIR, Path.cwd() / "data", Path.cwd(),
    ]
    for folder in search_dirs:
        for name in preferred_names:
            p = folder / name
            if p.exists():
                return p
        candidates = sorted(folder.glob("*.csv"))
        if candidates:
            return candidates[0]
    return None


def load_data(csv_path: str | Path | None = None) -> pd.DataFrame:
    if csv_path is None:
        resolved = _find_csv(DATA_DIR)
        if resolved is None:
            raise FileNotFoundError("No CSV file found. Run preprocessing.py first.")
    else:
        resolved = Path(csv_path)
        if not resolved.exists():
            resolved_auto = _find_csv(resolved.parent if resolved.parent.exists() else DATA_DIR)
            if resolved_auto:
                resolved = resolved_auto
            else:
                raise FileNotFoundError(f"File not found: {csv_path}")

    try:
        df = pd.read_csv(resolved, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(resolved, encoding="latin-1")

    # Unify rating column name
    for alias in _RATING_ALIASES:
        if alias in df.columns and alias != TARGET_COL:
            df = df.rename(columns={alias: TARGET_COL})
            break

    if TARGET_COL in df.columns:
        df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")

    df = df.drop_duplicates().reset_index(drop=True)
    return df


# ═════════════════════════════════════════════════════════════════════════════
# Feature preparation
# ═════════════════════════════════════════════════════════════════════════════

def prepare_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, dict, object]:
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset.")

    rated = df[df[TARGET_COL].notna() & (df[TARGET_COL] > 0)].copy()

    if len(rated) < 50:
        raise ValueError(f"Too few rated rows ({len(rated)}) to train reliably.")

    rated        = engineer_features(rated)
    rated, encs  = encode_features(rated, fit=True)
    X_raw        = build_feature_matrix(rated)
    X_sc, scaler = scale_features(X_raw, fit=True)
    y            = rated[TARGET_COL].reset_index(drop=True)
    X_sc         = X_sc.reset_index(drop=True)

    return X_sc, y, encs, scaler


# ═════════════════════════════════════════════════════════════════════════════
# Model definitions
# ═════════════════════════════════════════════════════════════════════════════

def _get_models() -> dict[str, object]:
    return {
        "Linear Regression":  LinearRegression(),
        "Ridge Regression":   Ridge(alpha=1.0),
        "Decision Tree":      DecisionTreeRegressor(max_depth=8, random_state=42),
        "Random Forest":      RandomForestRegressor(
                                  n_estimators=200, max_depth=12,
                                  min_samples_leaf=2, random_state=42, n_jobs=-1),
        "Gradient Boosting":  GradientBoostingRegressor(
                                  n_estimators=200, max_depth=5,
                                  learning_rate=0.05, subsample=0.8,
                                  random_state=42),
    }


# ═════════════════════════════════════════════════════════════════════════════
# PUBLIC: train_all_models  (imported by pages/ML_Pipeline.py & AI_Predictor.py)
# ═════════════════════════════════════════════════════════════════════════════

def train_all_models(df: pd.DataFrame) -> dict:
    """
    Full supervised-learning pipeline.

    Returns dict with keys:
        leaderboard, best_name, best_model, feature_importance,
        residuals, encoders, scaler, X_test, y_test
    """
    X, y, encoders, scaler = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    models  = _get_models()
    records = []
    fitted  = {}

    for name, model in models.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = round(time.time() - t0, 1)

        y_pred = model.predict(X_test)
        r2     = r2_score(y_test, y_pred)
        rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
        mae    = mean_absolute_error(y_test, y_pred)
        cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2", n_jobs=-1)

        records.append({
            "Model":      name,
            "R²":         round(r2,  4),
            "RMSE":       round(rmse, 4),
            "MAE":        round(mae,  4),
            "CV R² Mean": round(cv_scores.mean(), 4),
            "CV R² Std":  round(cv_scores.std(),  4),
            "Train Time": elapsed,
        })
        fitted[name] = model

    leaderboard = (
        pd.DataFrame(records)
        .sort_values("R²", ascending=False)
        .reset_index(drop=True)
    )
    leaderboard.index += 1

    best_name  = leaderboard.iloc[0]["Model"]
    best_model = fitted[best_name]

    feat_imp = get_feature_importance(best_model, X.columns.tolist())

    y_pred_best = best_model.predict(X_test)
    residuals   = pd.DataFrame({
        "Actual":    y_test.values,
        "Predicted": y_pred_best,
        "Residual":  y_test.values - y_pred_best,
    })

    _save_artefacts(best_model, best_name, encoders, scaler, leaderboard, feat_imp)

    return {
        "leaderboard":        leaderboard,
        "best_name":          best_name,
        "best_model":         best_model,
        "feature_importance": feat_imp,
        "residuals":          residuals,
        "encoders":           encoders,
        "scaler":             scaler,
        "X_test":             X_test,
        "y_test":             y_test,
    }


# ═════════════════════════════════════════════════════════════════════════════
# PUBLIC: get_feature_importance  (imported by pages/AI_Predictor.py)
# ═════════════════════════════════════════════════════════════════════════════

def get_feature_importance(
    model: object,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Extract feature importances from a trained model.

    Works for tree-based models (feature_importances_) and linear models
    (coef_). Returns an empty DataFrame if neither is available.
    """
    importances = None

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)

    if importances is None:
        return pd.DataFrame(columns=["Feature", "Importance", "Importance %"])

    # Align lengths (guard against shape mismatches)
    n = min(len(importances), len(feature_names))
    importances   = importances[:n]
    feature_names = feature_names[:n]

    total = importances.sum()
    if total == 0:
        return pd.DataFrame(columns=["Feature", "Importance", "Importance %"])

    fi = pd.DataFrame({
        "Feature":      feature_names,
        "Importance":   importances,
        "Importance %": importances / total * 100,
    }).sort_values("Importance %", ascending=False).reset_index(drop=True)

    return fi


# ═════════════════════════════════════════════════════════════════════════════
# Artefact persistence
# ═════════════════════════════════════════════════════════════════════════════

def _save_artefacts(
    model:       object,
    model_name:  str,
    encoders:    dict,
    scaler:      object,
    leaderboard: pd.DataFrame,
    feat_imp:    pd.DataFrame,
) -> None:
    try:
        joblib.dump(model,    _ARTEFACT_FILES["model"])
        joblib.dump(encoders, _ARTEFACT_FILES["encoders"])
        joblib.dump(scaler,   _ARTEFACT_FILES["scaler"])
        leaderboard.to_csv(_ARTEFACT_FILES["leaderboard"], index=True)
        if not feat_imp.empty:
            feat_imp.to_csv(_ARTEFACT_FILES["feat_imp"], index=False)
        _ARTEFACT_FILES["model_name"].write_text(model_name, encoding="utf-8")
        print(f"   ✓  Artefacts saved to {MODELS_DIR}")
    except OSError as exc:
        print(f"   ⚠  Could not save artefacts: {exc}")


# ═════════════════════════════════════════════════════════════════════════════
# PUBLIC: load_artefacts  (imported by pages/ML_Pipeline.py & AI_Predictor.py)
# ═════════════════════════════════════════════════════════════════════════════

def load_artefacts() -> dict | None:
    """
    Load saved model artefacts from disk.
    Returns None if any required file is missing.
    """
    required = [
        _ARTEFACT_FILES["model"],
        _ARTEFACT_FILES["encoders"],
        _ARTEFACT_FILES["scaler"],
        _ARTEFACT_FILES["model_name"],
    ]
    if any(not p.exists() for p in required):
        return None

    return {
        "model":      joblib.load(_ARTEFACT_FILES["model"]),
        "encoders":   joblib.load(_ARTEFACT_FILES["encoders"]),
        "scaler":     joblib.load(_ARTEFACT_FILES["scaler"]),
        "model_name": _ARTEFACT_FILES["model_name"].read_text(encoding="utf-8").strip(),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Prediction helper (used by prediction_engine.py)
# ═════════════════════════════════════════════════════════════════════════════

def predict_rating(
    row:       pd.Series,
    artefacts: dict,
    df_ref:    pd.DataFrame | None = None,
) -> float:
    """Predict the aggregate rating for a single restaurant row."""
    sample_df    = pd.DataFrame([row])
    sample_df    = engineer_features(sample_df)
    sample_df, _ = encode_features(
        sample_df,
        label_encoders=artefacts["encoders"],
        fit=False,
    )
    X_raw        = build_feature_matrix(sample_df)
    X_sc, _      = scale_features(X_raw, scaler=artefacts["scaler"], fit=False)

    prediction = artefacts["model"].predict(X_sc)[0]
    return float(np.clip(prediction, 0.0, 5.0))


# ═════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="RestaurantIQ — ML Training Pipeline")
    parser.add_argument("--data", "-d", metavar="CSV_PATH", default=None)
    parser.add_argument("--predict", "-p", action="store_true")
    args = parser.parse_args()

    df = load_data(args.data)

    output = train_all_models(df)
    lb     = output["leaderboard"]

    print("\n" + "═" * 70)
    print(f"🏆  Best model : {output['best_name']}")
    print(f"    R²         : {lb.iloc[0]['R²']:.4f}")
    print(f"    RMSE       : {lb.iloc[0]['RMSE']:.4f}")
    print(f"    MAE        : {lb.iloc[0]['MAE']:.4f}")
    print("═" * 70 + "\n")


if __name__ == "__main__":
    main()
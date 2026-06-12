from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

# ── Project root on sys.path ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR      = PROJECT_ROOT / "src"
DATA_DIR     = PROJECT_ROOT / "data"

for _p in [str(PROJECT_ROOT), str(SRC_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Local imports (after path setup) ─────────────────────────────────────────
try:
   
    from feature_engineering import (
        ML_FEATURE_COLS,
        TARGET_COL,
        build_feature_matrix,
        encode_features,
        engineer_features,
        scale_features,
    )
except ImportError as exc:
    print(f"\n❌  Import error: {exc}")
    print("    Make sure feature_engineering.py and model_training.py are in the same")
    print("    directory as main.py (or on PYTHONPATH).\n")
    sys.exit(1)


# ═════════════════════════════════════════════════════════════════════════════
# Data loading
# ═════════════════════════════════════════════════════════════════════════════

# Common column-name aliases used in the Cognifyz / Zomato dataset
_RATING_ALIASES = [
    "Aggregate rating",
    "aggregate_rating",
    "Rating",
    "rating",
    "aggregate rating",
]
_CUISINE_ALIASES = ["Cuisines", "cuisines", "Cuisine"]
_CITY_ALIASES    = ["City", "city"]
_COST_ALIASES    = [
    "Average Cost for two",
    "average_cost_for_two",
    "Average cost for two",
    "Cost",
    "cost",
]


def _find_csv(data_dir: Path) -> Path | None:
    """Return the first .csv found in *data_dir*, or None."""
    candidates = sorted(data_dir.glob("*.csv"))
    return candidates[0] if candidates else None


def load_data(csv_path: str | Path | None = None) -> pd.DataFrame:
    """
    Load the restaurant dataset from *csv_path*.

    If *csv_path* is None the function looks in ``data/`` relative to this
    file.  Raises ``SystemExit`` if no file can be found.
    """
    print("━" * 60)
    print("📂  Loading data")
    print("━" * 60)

    if csv_path is None:
        resolved = _find_csv(DATA_DIR)
        if resolved is None:
            print(f"\n❌  No CSV file found in '{DATA_DIR}'.")
            print("    Pass --data <path/to/file.csv> or place the file in data/.\n")
            sys.exit(1)
        print(f"   Auto-discovered: {resolved}")
    else:
        resolved = Path(csv_path)
        if not resolved.exists():
            print(f"\n❌  File not found: {resolved}\n")
            sys.exit(1)
        print(f"   Using: {resolved}")

    try:
        df = pd.read_csv(resolved, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(resolved, encoding="latin-1")

    print(f"   Shape  : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"   Memory : {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")
    print(f"   Columns: {list(df.columns)}")

    # ── Light validation ──────────────────────────────────────────────────────
    _validate_dataframe(df)

    # ── Basic cleaning ────────────────────────────────────────────────────────
    df = _basic_clean(df)

    print(f"\n   ✅  Data loaded & cleaned — {df.shape[0]:,} rows ready\n")
    return df


def _validate_dataframe(df: pd.DataFrame) -> None:
    """Warn (not abort) if expected columns are missing."""
    print("\n   Validating schema …")

    found_rating  = any(c in df.columns for c in _RATING_ALIASES)
    found_cuisine = any(c in df.columns for c in _CUISINE_ALIASES)
    found_city    = any(c in df.columns for c in _CITY_ALIASES)
    found_cost    = any(c in df.columns for c in _COST_ALIASES)

    checks = {
        "Rating column":  found_rating,
        "Cuisine column": found_cuisine,
        "City column":    found_city,
        "Cost column":    found_cost,
    }
    for label, ok in checks.items():
        status = "✓" if ok else "⚠ NOT FOUND"
        print(f"     {status}  {label}")

    missing_pct = df.isnull().mean() * 100
    high_missing = missing_pct[missing_pct > 40]
    if not high_missing.empty:
        print(f"\n   ⚠  Columns with >40 % missing values:")
        for col, pct in high_missing.items():
            print(f"       • {col}: {pct:.1f}%")


def _basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal cleaning applied before feature engineering:
    - Strip whitespace from string columns
    - Standardise the rating column to a float named TARGET_COL
    - Drop fully-duplicate rows
    """
    print("\n   Cleaning …")

    # Normalise whitespace in object columns
    str_cols = df.select_dtypes("object").columns
    df[str_cols] = df[str_cols].apply(lambda s: s.str.strip())

    # Unify rating column name → TARGET_COL  (e.g. "Aggregate rating")
    for alias in _RATING_ALIASES:
        if alias in df.columns and alias != TARGET_COL:
            df = df.rename(columns={alias: TARGET_COL})
            print(f"     Renamed '{alias}' → '{TARGET_COL}'")
            break

    # Coerce rating to numeric
    if TARGET_COL in df.columns:
        before = df[TARGET_COL].notna().sum()
        df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
        after  = df[TARGET_COL].notna().sum()
        if before != after:
            print(f"     ⚠  {before - after} non-numeric rating values set to NaN")

    # Drop duplicates
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    if len(df) < n_before:
        print(f"     Removed {n_before - len(df):,} duplicate rows")

    return df


# ═════════════════════════════════════════════════════════════════════════════
# Reporting helpers
# ═════════════════════════════════════════════════════════════════════════════

def print_leaderboard(leaderboard: pd.DataFrame) -> None:
    """Pretty-print the model leaderboard to stdout."""
    print("\n" + "═" * 60)
    print("📋  FINAL LEADERBOARD")
    print("═" * 60)
    print(f"  {'Rank':<5} {'Model':<24} {'R²':>7} {'RMSE':>7} {'MAE':>7} "
          f"{'CV R²':>12}  {'Time':>6}")
    print("  " + "─" * 70)
    for rank, row in leaderboard.iterrows():
        crown = "🏆" if rank == 1 else "  "
        print(f"  {crown} #{rank:<3} {row['Model']:<24} "
              f"{row['R²']:>7.4f} {row['RMSE']:>7.4f} {row['MAE']:>7.4f} "
              f"{row['CV R² Mean']:>6.4f}±{row['CV R² Std']:.4f}  "
              f"{row['Train Time']:>5}s")
    print()


def print_feature_importance(feat_imp: pd.DataFrame, top_n: int = 15) -> None:
    if feat_imp.empty:
        return
    print("\n" + "═" * 60)
    print(f"🔍  TOP {top_n} FEATURE IMPORTANCES  (best model)")
    print("═" * 60)
    for _, row in feat_imp.head(top_n).iterrows():
        bar = "█" * int(row["Importance %"] / 1.5)
        print(f"  {row['Feature']:<38} {row['Importance %']:5.2f}%  {bar}")
    print()


def print_residual_summary(residuals: pd.DataFrame) -> None:
    if residuals.empty:
        return
    abs_res = residuals["Residual"].abs()
    print("\n" + "═" * 60)
    print("📉  RESIDUAL ANALYSIS  (best model on test set)")
    print("═" * 60)
    print(f"  Samples      : {len(residuals):,}")
    print(f"  Mean residual: {residuals['Residual'].mean():.4f}  "
          f"(close to 0 = low bias)")
    print(f"  Std  residual: {residuals['Residual'].std():.4f}")
    print(f"  Max |error|  : {abs_res.max():.4f}")
    for threshold in (0.25, 0.5, 1.0):
        pct = (abs_res <= threshold).mean() * 100
        print(f"  Within ±{threshold:<4}  : {pct:.1f}%")

    # Distribution buckets
    print("\n  Error distribution:")
    bins   = [0, 0.1, 0.25, 0.5, 1.0, float("inf")]
    labels = ["[0.00, 0.10)", "[0.10, 0.25)", "[0.25, 0.50)", "[0.50, 1.00)", "≥1.00"]
    counts = pd.cut(abs_res, bins=bins, labels=labels).value_counts().sort_index()
    for bucket, cnt in counts.items():
        bar = "▓" * int(cnt / len(residuals) * 50)
        print(f"    {bucket}  {cnt:>5} ({cnt/len(residuals)*100:4.1f}%)  {bar}")
    print()


def save_summary_report(output: dict, report_path: Path) -> None:
    """Write a plain-text summary report alongside the model artefacts."""
    lb = output["leaderboard"]
    fi = output["feature_importance"]

    lines = [
        "RestaurantIQ — ML Training Summary",
        "=" * 60,
        f"Best model : {output['best_name']}",
        f"R²         : {lb.iloc[0]['R²']:.4f}",
        f"RMSE       : {lb.iloc[0]['RMSE']:.4f}",
        f"MAE        : {lb.iloc[0]['MAE']:.4f}",
        f"CV R²      : {lb.iloc[0]['CV R² Mean']:.4f} ± {lb.iloc[0]['CV R² Std']:.4f}",
        "",
        "Leaderboard",
        "-" * 60,
    ]
    for rank, row in lb.iterrows():
        lines.append(
            f"  #{rank}  {row['Model']:<22}  R²={row['R²']:.4f}  "
            f"RMSE={row['RMSE']:.4f}  MAE={row['MAE']:.4f}"
        )

    if not fi.empty:
        lines += ["", "Top 10 Features", "-" * 60]
        for _, row in fi.head(10).iterrows():
            lines.append(f"  {row['Feature']:<38} {row['Importance %']:.2f}%")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"   ✓ Summary report → {report_path}")


# ═════════════════════════════════════════════════════════════════════════════
# Demo prediction
# ═════════════════════════════════════════════════════════════════════════════

def run_demo_prediction(df: pd.DataFrame) -> None:
    """
    Load saved artefacts and run a sample prediction on a random restaurant.
    """
    print("\n" + "═" * 60)
    print("🔮  DEMO PREDICTION  (using saved artefacts)")
    print("═" * 60)

    artefacts = load_artefacts()
    if artefacts is None:
        print("   ⚠  Skipping prediction demo — no artefacts found.\n")
        return

    # Pick a random rated row as demo input
    rated = df[df[TARGET_COL].notna() & (df[TARGET_COL] > 0)]
    if rated.empty:
        print("   ⚠  No rated restaurants in the dataset for demo.\n")
        return

    sample = rated.sample(1, random_state=int(time.time()) % 1000).iloc[0]
    actual = sample[TARGET_COL]

    print(f"\n   Restaurant: {sample.get('Restaurant Name', sample.get('name', 'N/A'))}")
    print(f"   Actual rating: {actual:.1f}")

    try:
        predicted = predict_rating(sample, artefacts, df)
        error     = abs(predicted - actual)
        print(f"   Predicted    : {predicted:.4f}")
        print(f"   |Error|      : {error:.4f}")
    except Exception as exc:
        print(f"\n   ⚠  Prediction failed: {exc}")
        traceback.print_exc()

    print()


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RestaurantIQ — ML Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data", "-d",
        metavar="CSV_PATH",
        default=None,
        help="Path to the restaurant CSV dataset. "
             "Auto-discovered from data/ if omitted.",
    )
    parser.add_argument(
        "--predict", "-p",
        action="store_true",
        help="After training, run a demo prediction using saved artefacts.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do NOT persist model artefacts to disk (dry run).",
    )
    parser.add_argument(
        "--top-features",
        type=int,
        default=15,
        metavar="N",
        help="Number of top features to display (default: 15).",
    )
    return parser.parse_args()


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = _parse_args()

    wall_start = time.time()

    print("\n" + "═" * 60)
    print("🍽️   RestaurantIQ  —  ML Training Pipeline")
    print("    Cognifyz Analytics Challenge · Level 3 Task 1")
    print("═" * 60)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    df = load_data(args.data)

    # ── 2. Train all models ───────────────────────────────────────────────────
    try:
        output = train_all_models(df)
    except Exception as exc:
        print(f"\n❌  Training failed: {exc}\n")
        traceback.print_exc()
        sys.exit(1)

    # ── 3. Print summary tables ───────────────────────────────────────────────
    print_leaderboard(output["leaderboard"])
    print_feature_importance(output["feature_importance"], top_n=args.top_features)
    print_residual_summary(output["residuals"])

    # ── 4. Save plain-text summary report ────────────────────────────────────
    if not args.no_save:
        report_path = MODELS_DIR / "training_summary.txt"
        print("━" * 60)
        print("📝  Writing summary report …")
        save_summary_report(output, report_path)
    else:
        print("   ⚠  --no-save flag set; artefacts NOT persisted.")

    # ── 5. Optional demo prediction ───────────────────────────────────────────
    if args.predict:
        run_demo_prediction(df)

    # ── 6. Wall-clock timing ──────────────────────────────────────────────────
    elapsed = time.time() - wall_start
    minutes, seconds = divmod(int(elapsed), 60)
    print("═" * 60)
    print(f"⏱   Total wall time : {minutes}m {seconds}s")
    print(f"🏆  Best model      : {output['best_name']}")
    print(f"    R²              : {output['leaderboard'].iloc[0]['R²']:.4f}")
    print(f"    RMSE            : {output['leaderboard'].iloc[0]['RMSE']:.4f}")
    print(f"    MAE             : {output['leaderboard'].iloc[0]['MAE']:.4f}")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()
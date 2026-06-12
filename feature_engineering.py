from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ── Country / label maps ──────────────────────────────────────────────────────
COUNTRY_MAP = {
    1:   "India",          14:  "Australia",     30:  "Brazil",
    37:  "Canada",         94:  "Indonesia",      148: "New Zealand",
    162: "Philippines",    166: "Qatar",          184: "Singapore",
    189: "South Africa",   191: "Sri Lanka",      208: "Turkey",
    214: "UAE",            215: "United Kingdom", 216: "United States",
}

PRICE_LABEL  = {1: "Budget", 2: "Affordable", 3: "Premium", 4: "Luxury"}
RATING_ORDER = ["Not rated", "Poor", "Average", "Good", "Very Good", "Excellent"]

# ── Paths ─────────────────────────────────────────────────────────────────────
_SRC  = Path(__file__).resolve().parent
_ROOT = _SRC.parent if _SRC.name == "src" else _SRC

DATA_DIR      = _ROOT / "data"
DATA_PATH     = DATA_DIR / "cleaned_dataset.csv"
FEATURED_PATH = DATA_DIR / "featured_dataset.csv"

DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── ML feature columns ────────────────────────────────────────────────────────
TARGET_COL = "Aggregate rating"

# NOTE: these use _encoded suffix (lowercase) to match encode_features() output
ML_FEATURE_COLS = [
    "Price range",
    "Votes",
    "Has Table booking",
    "Has Online delivery",
    "Is delivering now",
    "Cuisine Count",
    "Country Code",
    "City_encoded",
    "Primary Cuisine_encoded",
]

BOOL_COLS = [
    "Has Table booking",
    "Has Online delivery",
    "Is delivering now",
    "Switch to order menu",
]

NUMERIC_FEATURE_COLS = [
    "Aggregate rating",
    "Votes",
    "Average Cost for two",
    "Price range",
    "Cuisine Count",
    "Has Online delivery",
    "Has Table booking",
    "Is delivering now",
    "Log Votes",
    "Cost per Person",
]


# ═════════════════════════════════════════════════════════════════════════════
# Console helpers  (CLI-only; NOT called inside Streamlit-facing functions)
# ═════════════════════════════════════════════════════════════════════════════

def _banner(title: str) -> None:
    print(f"\n{'━' * 60}\n  {title}\n{'━' * 60}")

def _ok(msg: str)   -> None: print(f"   ✓  {msg}")
def _warn(msg: str) -> None: print(f"   ⚠  {msg}")
def _info(msg: str) -> None: print(f"   •  {msg}")


# ═════════════════════════════════════════════════════════════════════════════
# Smart CSV finder
# ═════════════════════════════════════════════════════════════════════════════

def find_csv(preferred: Path) -> Path:
    if preferred.exists():
        return preferred

    candidates = [
        "cleaned_dataset.csv", "cleaned.csv", "featured_dataset.csv",
        "Dataset_.csv", "Dataset.csv", "dataset.csv",
        "zomato.csv", "restaurants.csv",
    ]
    search_dirs = [
        preferred.parent, _SRC / "data", _ROOT / "data",
        _SRC, _ROOT, Path.cwd(), Path.cwd() / "data",
    ]

    for folder in search_dirs:
        for name in candidates:
            p = folder / name
            if p.exists():
                return p
        csvs = sorted(folder.glob("*.csv"))
        if csvs:
            return csvs[0]

    raise FileNotFoundError(
        "Could not find cleaned_dataset.csv. Run preprocessing.py first."
    )


# ═════════════════════════════════════════════════════════════════════════════
# Step 1 – Load
# ═════════════════════════════════════════════════════════════════════════════

def _load_cleaned(path: Path) -> pd.DataFrame:
    _banner("Step 1 · Loading cleaned dataset")
    path = find_csv(path)
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1")
    str_cols = df.select_dtypes("object").columns
    df[str_cols] = df[str_cols].apply(lambda s: s.str.strip())
    _ok(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} columns from {path}")
    return df


# ═════════════════════════════════════════════════════════════════════════════
# Step 2 – Missing-value guard
# ═════════════════════════════════════════════════════════════════════════════

def _handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    if "Cuisines" in df.columns:
        df["Cuisines"] = df["Cuisines"].fillna("Unknown")

    cost_median = (
        df["Average Cost for two"].median()
        if "Average Cost for two" in df.columns else 0
    )
    for col, fill in [
        ("Aggregate rating",     0.0),
        ("Votes",                0),
        ("Average Cost for two", cost_median),
        ("Price range",          1),
        ("Rating text",          "Not rated"),
    ]:
        if col in df.columns:
            if col != "Rating text":
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(fill)
            else:
                df[col] = df[col].fillna(fill)

    for col in BOOL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df


# ═════════════════════════════════════════════════════════════════════════════
# Step 3 – Type conversions
# ═════════════════════════════════════════════════════════════════════════════

def _convert_types(df: pd.DataFrame) -> pd.DataFrame:
    for col in BOOL_COLS:
        if col not in df.columns:
            continue
        if df[col].dtype == object:
            df[col] = (
                df[col].str.strip()
                .map({"Yes": 1, "No": 0, "True": 1, "False": 0, "1": 1, "0": 0})
                .fillna(0).astype(int)
            )
        else:
            df[col] = df[col].fillna(0).astype(int)

    for col in ["Aggregate rating", "Average Cost for two", "Latitude", "Longitude"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    for col in ["Votes", "Country Code", "Price range", "Restaurant ID"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    return df


# ═════════════════════════════════════════════════════════════════════════════
# Step 4 – Derived columns
# ═════════════════════════════════════════════════════════════════════════════

def _add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "Country" not in df.columns:
        df["Country"] = df["Country Code"].map(COUNTRY_MAP).fillna("Other")

    if "Price Label" not in df.columns:
        df["Price Label"] = df["Price range"].map(PRICE_LABEL).fillna("Unknown")

    if "Is Rated" not in df.columns:
        df["Is Rated"] = (df["Aggregate rating"] > 0).astype(int)
    else:
        df["Is Rated"] = df["Is Rated"].astype(int)

    if "Primary Cuisine" not in df.columns:
        df["Primary Cuisine"] = (
            df["Cuisines"].str.split(",").str[0].str.strip()
            if "Cuisines" in df.columns else "Unknown"
        )

    if "Cuisine Count" not in df.columns:
        df["Cuisine Count"] = (
            df["Cuisines"].str.split(",").str.len()
            if "Cuisines" in df.columns else 1
        )

    if "Log Votes" not in df.columns:
        df["Log Votes"] = np.log1p(df["Votes"].astype(float))

    if "Cost per Person" not in df.columns and "Average Cost for two" in df.columns:
        df["Cost per Person"] = (df["Average Cost for two"] / 2).round(2)

    if "Rating Bucket" not in df.columns:
        df["Rating Bucket"] = pd.cut(
            df["Aggregate rating"],
            bins=[0, 1, 2, 3, 3.5, 4, 4.5, 5.01],
            labels=["0-1", "1-2", "2-3", "3-3.5", "3.5-4", "4-4.5", "4.5-5"],
            include_lowest=True,
        ).astype(str)

    return df


# ═════════════════════════════════════════════════════════════════════════════
# Step 5 – ML feature engineering (PUBLIC API)
# ═════════════════════════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure all ML-required raw features exist with correct dtypes.
    Idempotent — safe to call multiple times.
    """
    df = df.copy()

    if "Cuisines" not in df.columns:
        df["Cuisines"] = "Unknown"
    df["Cuisines"] = df["Cuisines"].fillna("Unknown")

    if "Primary Cuisine" not in df.columns:
        df["Primary Cuisine"] = df["Cuisines"].str.split(",").str[0].str.strip()

    # Cuisine Count (both variants kept for compat)
    if "Cuisine Count" not in df.columns:
        df["Cuisine Count"] = df["Cuisines"].str.split(",").str.len()
    df["Cuisine_Count"] = df["Cuisine Count"]

    # Bool columns
    for col in ["Has Table booking", "Has Online delivery", "Is delivering now"]:
        if col in df.columns:
            if df[col].dtype == object:
                df[col] = (
                    df[col].str.strip()
                    .map({"Yes": 1, "No": 0, "True": 1, "False": 0})
                    .fillna(0)
                )
        else:
            df[col] = 0
        df[col] = df[col].fillna(0).astype(int)

    # Numeric coercions
    for col in ["Votes", "Country Code", "Price range"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        else:
            df[col] = 0 if col != "Price range" else 1

    if "Aggregate rating" in df.columns:
        df["Aggregate rating"] = (
            pd.to_numeric(df["Aggregate rating"], errors="coerce").fillna(0.0)
        )

    # Engineered features
    if "Votes_Per_Rating" not in df.columns:
        df["Votes_Per_Rating"] = np.where(
            df["Aggregate rating"] > 0,
            df["Votes"] / df["Aggregate rating"], 0,
        ).round(2)

    if "Premium_Restaurant_Flag" not in df.columns:
        df["Premium_Restaurant_Flag"] = (df["Price range"] >= 3).astype(int)

    if "Restaurant_Name_Length" not in df.columns:
        df["Restaurant_Name_Length"] = (
            df["Restaurant Name"].astype(str).str.len()
            if "Restaurant Name" in df.columns else 0
        )

    if "Log_Votes" not in df.columns:
        df["Log_Votes"] = np.log1p(df["Votes"].astype(float))
    if "Log Votes" not in df.columns:
        df["Log Votes"] = df["Log_Votes"]

    if "Cost_Efficiency_Score" not in df.columns and "Average Cost for two" in df.columns:
        df["Cost_Efficiency_Score"] = np.where(
            df["Average Cost for two"] > 0,
            df["Aggregate rating"] / np.log1p(df["Average Cost for two"]), 0,
        ).round(4)

    if "Full_Service_Flag" not in df.columns:
        df["Full_Service_Flag"] = (
            (df["Has Online delivery"] == 1) & (df["Has Table booking"] == 1)
        ).astype(int)

    if "Address_Length" not in df.columns:
        df["Address_Length"] = (
            df["Address"].astype(str).str.len()
            if "Address" in df.columns else 0
        )

    if "Log_Cost" not in df.columns and "Average Cost for two" in df.columns:
        df["Log_Cost"] = np.log1p(df["Average Cost for two"].astype(float))

    if "Rating_x_Votes" not in df.columns:
        df["Rating_x_Votes"] = (df["Aggregate rating"] * df["Votes"]).round(2)

    if "Cost per Person" not in df.columns and "Average Cost for two" in df.columns:
        df["Cost per Person"] = (df["Average Cost for two"] / 2).round(2)

    return df


# ═════════════════════════════════════════════════════════════════════════════
# PUBLIC API: get_feature_stats
# ═════════════════════════════════════════════════════════════════════════════

def get_feature_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return summary statistics for key engineered numeric features."""
    df_eng = engineer_features(df)

    stat_cols = [
        c for c in [
            "Aggregate rating", "Votes", "Average Cost for two",
            "Price range", "Cuisine_Count", "Votes_Per_Rating",
            "Premium_Restaurant_Flag", "Restaurant_Name_Length",
            "Log_Votes", "Cost_Efficiency_Score", "Full_Service_Flag",
            "Rating_x_Votes",
        ]
        if c in df_eng.columns
    ]

    stats = df_eng[stat_cols].describe().T.reset_index()
    stats.columns = ["Feature", "Count", "Mean", "Std", "Min",
                     "25%", "50%", "75%", "Max"]
    for num_col in ["Mean", "Std", "Min", "25%", "50%", "75%", "Max"]:
        stats[num_col] = stats[num_col].round(4)
    stats["Count"] = stats["Count"].astype(int)
    return stats


# ═════════════════════════════════════════════════════════════════════════════
# PUBLIC API: get_correlation_matrix
# ═════════════════════════════════════════════════════════════════════════════

def get_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return the Pearson correlation matrix for key numeric features."""
    df_eng = engineer_features(df)

    corr_cols = [
        c for c in [
            "Aggregate rating", "Votes", "Average Cost for two",
            "Price range", "Cuisine_Count", "Has Online delivery",
            "Has Table booking", "Is delivering now",
            "Log_Votes", "Votes_Per_Rating",
            "Premium_Restaurant_Flag", "Cost_Efficiency_Score",
        ]
        if c in df_eng.columns
    ]

    return df_eng[corr_cols].corr().round(3)


# ═════════════════════════════════════════════════════════════════════════════
# Step 6 – Label encoding (PUBLIC API)
# ═════════════════════════════════════════════════════════════════════════════

def encode_features(
    df: pd.DataFrame,
    label_encoders: dict | None = None,
    fit: bool = True,
    verbose: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """
    Label-encode City and Primary Cuisine.

    Output column names:
        <col>_encoded   — canonical name used by ML pipeline
        <col>_Enc       — legacy alias kept for page compatibility

    Parameters
    ----------
    verbose : If True, print progress to stdout (CLI use only).
              Always False when called from Streamlit pages.
    """
    df       = df.copy()
    encoders = label_encoders or {}
    CAT_COLS = ["City", "Primary Cuisine"]

    for col in CAT_COLS:
        enc_col     = f"{col}_encoded"
        enc_col_leg = f"{col}_Enc"

        if col not in df.columns:
            df[enc_col]     = 0
            df[enc_col_leg] = 0
            if verbose:
                _warn(f"'{col}' not found — {enc_col} set to 0")
            continue

        df[col] = df[col].astype(str).fillna("Unknown")

        if fit:
            le = LabelEncoder()
            df[enc_col]   = le.fit_transform(df[col])
            encoders[col] = le
            if verbose:
                _ok(f"  {col:<20} → {enc_col}  ({len(le.classes_)} classes, fitted)")
        else:
            le = encoders.get(col)
            if le is None:
                df[enc_col] = 0
                if verbose:
                    _warn(f"  No encoder for '{col}' — {enc_col} set to 0")
            else:
                known   = set(le.classes_)
                df[col] = df[col].apply(lambda x: x if x in known else le.classes_[0])
                df[enc_col] = le.transform(df[col])
                if verbose:
                    _ok(f"  {col:<20} → {enc_col}  (transformed)")

        df[enc_col_leg] = df[enc_col]

    return df, encoders


# ═════════════════════════════════════════════════════════════════════════════
# Step 7 – Feature matrix & scaling (PUBLIC API)
# ═════════════════════════════════════════════════════════════════════════════

def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Select and return the ML feature matrix. Missing columns filled with 0."""
    X = pd.DataFrame(index=df.index)
    for col in ML_FEATURE_COLS:
        X[col] = df[col] if col in df.columns else 0
    return X.fillna(0)


def scale_features(
    X: pd.DataFrame,
    scaler: StandardScaler | None = None,
    fit: bool = True,
    verbose: bool = False,
) -> tuple[pd.DataFrame, StandardScaler]:
    """StandardScale the feature matrix."""
    if fit:
        scaler = StandardScaler()
        X_sc   = scaler.fit_transform(X)
        if verbose:
            _ok(f"Fitted new scaler on {X.shape[0]:,} rows × {X.shape[1]} features")
    else:
        if scaler is None:
            raise ValueError("A fitted scaler must be provided when fit=False.")
        X_sc = scaler.transform(X)
        if verbose:
            _ok(f"Transformed {X.shape[0]:,} rows × {X.shape[1]} features")

    return pd.DataFrame(X_sc, columns=X.columns, index=X.index), scaler


# ═════════════════════════════════════════════════════════════════════════════
# KPI / filter helpers  (Streamlit dashboard API)
# ═════════════════════════════════════════════════════════════════════════════

def get_summary_kpis(df: pd.DataFrame) -> dict:
    rated = (
        df[df["Is Rated"] == 1]
        if "Is Rated" in df.columns
        else df[df["Aggregate rating"] > 0]
    )
    return {
        "total_restaurants": len(df),
        "countries":   df["Country"].nunique()         if "Country"  in df.columns else 0,
        "cities":      df["City"].nunique()            if "City"     in df.columns else 0,
        "avg_rating":  round(rated["Aggregate rating"].mean(), 2) if len(rated) else 0,
        "total_votes": int(df["Votes"].sum())          if "Votes"    in df.columns else 0,
        "delivery_pct": round(df["Has Online delivery"].mean() * 100, 1)
                        if "Has Online delivery" in df.columns else 0,
        "booking_pct":  round(df["Has Table booking"].mean() * 100, 1)
                        if "Has Table booking"   in df.columns else 0,
        "top_cuisine": df["Primary Cuisine"].value_counts().idxmax()
                       if "Primary Cuisine" in df.columns else "N/A",
        "top_city":    df["City"].value_counts().idxmax()
                       if "City" in df.columns else "N/A",
        "top_country": df["Country"].value_counts().idxmax()
                       if "Country" in df.columns else "N/A",
    }


def filter_dataframe(
    df: pd.DataFrame,
    cuisines:     list[str] | None = None,
    cities:       list[str] | None = None,
    countries:    list[str] | None = None,
    min_rating:   float = 0.0,
    max_rating:   float = 5.0,
    price_ranges: list[int] | None = None,
) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    if cuisines:
        mask &= df["Primary Cuisine"].isin(cuisines)
    if cities:
        mask &= df["City"].isin(cities)
    if countries:
        mask &= df["Country"].isin(countries)
    if price_ranges:
        mask &= df["Price range"].isin(price_ranges)
    mask &= (
        (df["Aggregate rating"] >= min_rating) &
        (df["Aggregate rating"] <= max_rating)
    )
    return df[mask].reset_index(drop=True)


def get_top_n(series: pd.Series, n: int = 10) -> pd.DataFrame:
    vc = series.value_counts().head(n).reset_index()
    vc.columns = [series.name, "Count"]
    return vc


# ═════════════════════════════════════════════════════════════════════════════
# Full pipeline entry point  (CLI / internal)
# ═════════════════════════════════════════════════════════════════════════════

def _run_pipeline(
    src:    Path,
    dest:   Path,
    encode: bool = True,
    scale:  bool = False,
) -> tuple[pd.DataFrame, dict, StandardScaler | None]:
    t0 = time.time()

    df = _load_cleaned(src)
    df = _handle_missing(df)
    df = _convert_types(df)
    df = _add_derived_columns(df)
    df = engineer_features(df)

    encoders: dict = {}
    scaler:   StandardScaler | None = None

    if encode:
        df, encoders = encode_features(df, fit=True, verbose=True)

    if scale:
        X = build_feature_matrix(df)
        X_sc, scaler = scale_features(X, fit=True, verbose=True)
        for col in X_sc.columns:
            df[f"{col}_scaled"] = X_sc[col].values

    save_df = df.copy()
    for col in save_df.select_dtypes("category").columns:
        save_df[col] = save_df[col].astype(str)
    dest.parent.mkdir(parents=True, exist_ok=True)
    save_df.to_csv(dest, index=False, encoding="utf-8")

    _ok(f"Pipeline complete in {time.time() - t0:.2f}s → {dest}")
    return df, encoders, scaler


# ── In-process cache for Streamlit ────────────────────────────────────────────
_DF_CACHE: pd.DataFrame | None = None


def load_and_preprocess(path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Load featured_dataset.csv if it exists (fast path), otherwise run the
    full pipeline.  Results are cached in-process to avoid re-running on
    every Streamlit rerun.
    """
    global _DF_CACHE

    if _DF_CACHE is not None:
        return _DF_CACHE

    # Fast path: featured CSV already exists
    if FEATURED_PATH.exists():
        try:
            df = pd.read_csv(FEATURED_PATH, encoding="utf-8", low_memory=False)
            # Ensure all engineered columns are present
            df = engineer_features(df)
            _DF_CACHE = df
            return _DF_CACHE
        except Exception:
            pass  # fall through to full pipeline

    # Slow path: run the full pipeline
    df, _, _ = _run_pipeline(path, FEATURED_PATH, encode=True, scale=False)
    _DF_CACHE = df
    return _DF_CACHE


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RestaurantIQ — Feature Engineering Pipeline"
    )
    parser.add_argument("--data", "-d", default=str(DATA_PATH),  metavar="CSV_PATH")
    parser.add_argument("--out",  "-o", default=str(FEATURED_PATH), metavar="OUT_PATH")
    parser.add_argument("--scale", action="store_true")
    args = parser.parse_args()

    df, encoders, scaler = _run_pipeline(
        Path(args.data), Path(args.out),
        encode=True, scale=args.scale,
    )
    print(f"\n✅  Done — featured data saved to: {args.out}\n")
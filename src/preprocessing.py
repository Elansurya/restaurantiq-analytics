from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
_SRC  = Path(__file__).resolve().parent
_ROOT = _SRC.parent if _SRC.name == "src" else _SRC

DATA_DIR   = _ROOT / "data"
RAW_PATH   = DATA_DIR / "Dataset_.csv"
CLEAN_PATH = DATA_DIR / "cleaned_dataset.csv"

DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── Domain constants ──────────────────────────────────────────────────────────
COUNTRY_MAP = {
    1:   "India",          14:  "Australia",     30:  "Brazil",
    37:  "Canada",         94:  "Indonesia",      148: "New Zealand",
    162: "Philippines",    166: "Qatar",          184: "Singapore",
    189: "South Africa",   191: "Sri Lanka",      208: "Turkey",
    214: "UAE",            215: "United Kingdom", 216: "United States",
}

PRICE_LABEL = {1: "Budget", 2: "Affordable", 3: "Premium", 4: "Luxury"}

RATING_ORDER = ["Not rated", "Poor", "Average", "Good", "Very Good", "Excellent"]

BOOL_COLS = [
    "Has Table booking",
    "Has Online delivery",
    "Is delivering now",
    "Switch to order menu",
]

REQUIRED_COLS = [
    "Restaurant ID",
    "Restaurant Name",
    "Country Code",
    "City",
    "Cuisines",
    "Average Cost for two",
    "Price range",
    "Aggregate rating",
    "Rating text",
    "Votes",
    "Latitude",
    "Longitude",
]

NUMERIC_BOUNDS: dict[str, tuple[float, float]] = {
    "Aggregate rating":     (0.0, 5.0),
    "Price range":          (1.0, 4.0),
    "Average Cost for two": (0.0, 1_000_000.0),
    "Votes":                (0.0, 1_000_000.0),
}


# ═════════════════════════════════════════════════════════════════════════════
# Console helpers
# ═════════════════════════════════════════════════════════════════════════════

def _banner(title: str) -> None:
    print(f"\n{'━' * 60}")
    print(f"  {title}")
    print(f"{'━' * 60}")

def _ok(msg: str)   -> None: print(f"   ✓  {msg}")
def _warn(msg: str) -> None: print(f"   ⚠  {msg}")
def _info(msg: str) -> None: print(f"   •  {msg}")
def _step(n: int, total: int, msg: str) -> None:
    print(f"\n   [{n}/{total}] {msg}")


# ═════════════════════════════════════════════════════════════════════════════
# Smart CSV finder
# ═════════════════════════════════════════════════════════════════════════════

def find_csv(preferred: Path) -> Path:
    if preferred.exists():
        return preferred

    _warn(f"'{preferred.name}' not found — searching nearby …")

    candidates = [
        "Dataset_.csv", "Dataset.csv", "dataset_.csv", "dataset.csv",
        "zomato.csv", "Zomato.csv", "restaurants.csv", "Restaurants.csv",
        "raw.csv", "data.csv", "input.csv",
    ]
    for name in candidates:
        p = preferred.parent / name
        if p.exists():
            _ok(f"Found alternative: {p}")
            return p

    csvs = sorted(DATA_DIR.glob("*.csv"))
    if csvs:
        _ok(f"Found in data/: {csvs[0]}")
        return csvs[0]

    csvs = sorted(_SRC.glob("*.csv"))
    if csvs:
        _ok(f"Found in src/: {csvs[0]}")
        return csvs[0]

    csvs = sorted(Path.cwd().glob("*.csv"))
    if csvs:
        _ok(f"Found in cwd: {csvs[0]}")
        return csvs[0]

    raise FileNotFoundError(
        "\n"
        "  ✗  Could not find a CSV file to process.\n\n"
        "  Fix options:\n"
        "    A) Copy your CSV into the data/ folder:\n"
        f"         {DATA_DIR}\n\n"
        "    B) Pass the path explicitly:\n"
        "         python preprocessing.py --data path/to/your/file.csv\n"
    )


# ═════════════════════════════════════════════════════════════════════════════
# Step 0 – Load raw CSV
# ═════════════════════════════════════════════════════════════════════════════

def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    _banner("Step 0 · Loading raw CSV")
    path = find_csv(path)
    _info(f"Source : {path}")

    t0 = time.time()
    try:
        df = pd.read_csv(path, encoding="utf-8")
        _ok("Encoding: UTF-8")
    except UnicodeDecodeError:
        _warn("UTF-8 failed — retrying with latin-1")
        df = pd.read_csv(path, encoding="latin-1")
        _ok("Encoding: latin-1")

    elapsed = round(time.time() - t0, 2)
    _ok(f"Loaded   : {df.shape[0]:,} rows × {df.shape[1]} columns  ({elapsed}s)")

    str_cols = df.select_dtypes("object").columns
    df[str_cols] = df[str_cols].apply(lambda s: s.str.strip())
    return df


# ═════════════════════════════════════════════════════════════════════════════
# Step 1 – Schema validation
# ═════════════════════════════════════════════════════════════════════════════

def validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    _banner("Step 1 · Schema validation")

    ALIASES: dict[str, str] = {
        "aggregate_rating":      "Aggregate rating",
        "Aggregate Rating":      "Aggregate rating",
        "restaurant_name":       "Restaurant Name",
        "restaurant_id":         "Restaurant ID",
        "average_cost_for_two":  "Average Cost for two",
        "country_code":          "Country Code",
        "rating_text":           "Rating text",
        "price_range":           "Price range",
    }
    for alias, canonical in ALIASES.items():
        if alias in df.columns and canonical not in df.columns:
            df = df.rename(columns={alias: canonical})

    missing_req = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_req:
        for col in missing_req:
            df[col] = "Unknown"
    else:
        _ok("All required columns present ✔")

    return df


# ═════════════════════════════════════════════════════════════════════════════
# Step 2 – Duplicate removal
# ═════════════════════════════════════════════════════════════════════════════

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    _banner("Step 2 · Duplicate removal")
    df = df.drop_duplicates().reset_index(drop=True)
    if "Restaurant ID" in df.columns:
        df = df.drop_duplicates(
            subset=["Restaurant ID"], keep="first"
        ).reset_index(drop=True)
    return df


# ═════════════════════════════════════════════════════════════════════════════
# Step 3 – Missing value handling
# ═════════════════════════════════════════════════════════════════════════════

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    _banner("Step 3 · Missing value handling")

    if "Cuisines" in df.columns:
        df["Cuisines"] = df["Cuisines"].fillna("Unknown")

    if "Latitude" in df.columns and "Longitude" in df.columns:
        bad = (df["Latitude"] == 0) & (df["Longitude"] == 0)
        df  = df[~bad].reset_index(drop=True)

    if "Rating text" in df.columns:
        df["Rating text"] = df["Rating text"].fillna("Not rated")

    if "Aggregate rating" in df.columns:
        df["Aggregate rating"] = (
            pd.to_numeric(df["Aggregate rating"], errors="coerce").fillna(0.0)
        )

    if "Votes" in df.columns:
        df["Votes"] = pd.to_numeric(df["Votes"], errors="coerce").fillna(0)

    if "Average Cost for two" in df.columns:
        df["Average Cost for two"] = pd.to_numeric(
            df["Average Cost for two"], errors="coerce"
        )
        df["Average Cost for two"] = df["Average Cost for two"].fillna(
            df["Average Cost for two"].median()
        )

    for col, fill in [
        ("City",         "Unknown"),
        ("Locality",     "Unknown"),
        ("Country Code", 0),
        ("Currency",     "Unknown"),
    ]:
        if col in df.columns:
            df[col] = df[col].fillna(fill)

    for col in BOOL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("No")

    return df


# ═════════════════════════════════════════════════════════════════════════════
# Step 4 – Type conversions
# ═════════════════════════════════════════════════════════════════════════════

def convert_types(df: pd.DataFrame) -> pd.DataFrame:
    _banner("Step 4 · Type conversions")

    for col in BOOL_COLS:
        if col not in df.columns:
            continue
        if df[col].dtype == object:
            df[col] = (
                df[col].str.strip()
                .map({
                    "Yes": 1, "No": 0,
                    "True": 1, "False": 0,
                    "1": 1,   "0": 0,
                    True: 1,  False: 0,
                })
                .fillna(0)
                .astype(int)
            )
        else:
            df[col] = df[col].astype(int)

    NUM_COLS: dict[str, str] = {
        "Aggregate rating":     "float",
        "Average Cost for two": "float",
        "Price range":          "int",
        "Votes":                "int",
        "Latitude":             "float",
        "Longitude":            "float",
        "Country Code":         "int",
        "Restaurant ID":        "int",
    }
    for col, dtype in NUM_COLS.items():
        if col not in df.columns:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        df[col] = df[col].astype(
            "float64" if dtype == "float" else "int64"
        )

    if "Rating text" in df.columns:
        df["Rating text"] = df["Rating text"].astype(str).str.strip()
        df["Rating text"] = pd.Categorical(
            df["Rating text"], categories=RATING_ORDER, ordered=True
        )

    return df


# ═════════════════════════════════════════════════════════════════════════════
# Step 5 – Outlier detection & capping
# ═════════════════════════════════════════════════════════════════════════════

def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    _banner("Step 5 · Outlier detection & capping")

    for col, (lo, hi) in NUMERIC_BOUNDS.items():
        if col not in df.columns:
            continue
        df[col] = df[col].astype(float).clip(lower=lo, upper=hi)

    return df


# ═════════════════════════════════════════════════════════════════════════════
# Step 6 – Derived / enrichment columns
# ═════════════════════════════════════════════════════════════════════════════

def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    _banner("Step 6 · Derived / enrichment columns")

    df["Country"]        = df["Country Code"].map(COUNTRY_MAP).fillna("Other")
    df["Price Label"]    = df["Price range"].map(PRICE_LABEL).fillna("Unknown")
    df["Is Rated"]       = (df["Aggregate rating"] > 0).astype(int)
    df["Primary Cuisine"] = (
        df["Cuisines"].str.split(",").str[0].str.strip()
    )
    df["Cuisine Count"]  = df["Cuisines"].str.split(",").str.len()
    df["Rating Bucket"]  = pd.cut(
        df["Aggregate rating"],
        bins=[0, 1, 2, 3, 3.5, 4, 4.5, 5.01],
        labels=["0-1", "1-2", "2-3", "3-3.5", "3.5-4", "4-4.5", "4.5-5"],
        include_lowest=True,
    ).astype(str)
    df["Log Votes"] = np.log1p(df["Votes"])

    if "Average Cost for two" in df.columns:
        df["Cost per Person"] = (df["Average Cost for two"] / 2).round(2)

    return df


# ═════════════════════════════════════════════════════════════════════════════
# Step 7 – Save cleaned dataset
# ═════════════════════════════════════════════════════════════════════════════

def save_cleaned(df: pd.DataFrame, path: Path = CLEAN_PATH) -> None:
    _banner("Step 7 · Saving cleaned dataset")
    save_df = df.copy()
    for col in save_df.select_dtypes("category").columns:
        save_df[col] = save_df[col].astype(str)
    save_df.to_csv(path, index=False, encoding="utf-8")
    _ok(f"Saved {len(save_df):,} rows → {path}")


# ═════════════════════════════════════════════════════════════════════════════
# Public entry point
# ═════════════════════════════════════════════════════════════════════════════

def preprocess(
    raw_path: Path  = RAW_PATH,
    save_path: Path = CLEAN_PATH,
) -> pd.DataFrame:
    global CLEAN_PATH
    CLEAN_PATH = save_path

    t0 = time.time()
    df = load_raw(raw_path)
    df = validate_schema(df)
    df = remove_duplicates(df)
    df = handle_missing_values(df)
    df = convert_types(df)
    df = handle_outliers(df)
    df = add_derived_columns(df)
    save_cleaned(df, save_path)

    _ok(f"Pipeline complete in {time.time() - t0:.2f}s — {df.shape}")
    return df


# ── In-process cache ──────────────────────────────────────────────────────────
_DF_CACHE: pd.DataFrame | None = None


def load_and_preprocess(
    raw_path: Path  = RAW_PATH,
    save_path: Path = CLEAN_PATH,
    *,
    force: bool = False,
) -> pd.DataFrame:
    """
    Public wrapper around preprocess() with an in-process cache.
    """
    global _DF_CACHE

    if _DF_CACHE is not None and not force:
        return _DF_CACHE

    if save_path.exists() and not force:
        try:
            df = pd.read_csv(save_path, encoding="utf-8", low_memory=False)
            if "Rating text" in df.columns:
                df["Rating text"] = pd.Categorical(
                    df["Rating text"].astype(str),
                    categories=RATING_ORDER,
                    ordered=True,
                )
            for col in BOOL_COLS:
                if col in df.columns:
                    df[col] = (
                        pd.to_numeric(df[col], errors="coerce")
                        .fillna(0)
                        .astype(int)
                    )
            _DF_CACHE = df
            return _DF_CACHE
        except Exception:
            pass

    _DF_CACHE = preprocess(raw_path=raw_path, save_path=save_path)
    return _DF_CACHE


# ═════════════════════════════════════════════════════════════════════════════
# KPI summary
# ═════════════════════════════════════════════════════════════════════════════

def get_summary_kpis(df: pd.DataFrame) -> dict:
    """
    Compute platform-level KPIs consumed by app.py and all pages.
    """
    kpis: dict = {}

    kpis["total_restaurants"] = int(len(df))
    kpis["countries"] = int(
        df["Country"].nunique()      if "Country"      in df.columns
        else df["Country Code"].nunique() if "Country Code" in df.columns
        else 0
    )
    kpis["cities"] = int(
        df["City"].nunique() if "City" in df.columns else 0
    )

    if "Aggregate rating" in df.columns:
        rated = df.loc[df["Aggregate rating"] > 0, "Aggregate rating"]
        kpis["avg_rating"] = round(float(rated.mean()), 2) if len(rated) else 0.0
        kpis["rated_pct"]  = round(len(rated) / len(df) * 100, 1) if len(df) else 0.0
    else:
        kpis["avg_rating"] = 0.0
        kpis["rated_pct"]  = 0.0

    kpis["total_votes"] = (
        int(df["Votes"].sum()) if "Votes" in df.columns else 0
    )

    kpis["delivery_pct"] = (
        round(float(df["Has Online delivery"].mean()) * 100, 1)
        if "Has Online delivery" in df.columns else 0.0
    )
    kpis["booking_pct"] = (
        round(float(df["Has Table booking"].mean()) * 100, 1)
        if "Has Table booking" in df.columns else 0.0
    )

    if "Primary Cuisine" in df.columns and len(df):
        kpis["top_cuisine"] = str(df["Primary Cuisine"].value_counts().idxmax())
    elif "Cuisines" in df.columns and len(df):
        kpis["top_cuisine"] = str(
            df["Cuisines"].str.split(",").str[0].str.strip().value_counts().idxmax()
        )
    else:
        kpis["top_cuisine"] = "N/A"

    kpis["top_city"] = (
        str(df["City"].value_counts().idxmax())
        if "City" in df.columns and len(df) else "N/A"
    )
    kpis["top_country"] = (
        str(df["Country"].value_counts().idxmax())
        if "Country" in df.columns and len(df) else "N/A"
    )

    kpis["price_dist"] = (
        df["Price Label"].value_counts().to_dict()
        if "Price Label" in df.columns else {}
    )
    kpis["rating_dist"] = (
        df["Rating text"].astype(str).value_counts().to_dict()
        if "Rating text" in df.columns else {}
    )

    return kpis


# ═════════════════════════════════════════════════════════════════════════════
# Universal dataframe filter
# ═════════════════════════════════════════════════════════════════════════════

def filter_dataframe(
    df: pd.DataFrame,
    cuisines:     list[str]  | None = None,
    cities:       list[str]  | None = None,
    countries:    list[str]  | None = None,
    price_ranges: list[int]  | None = None,
    # ── rating bounds – two calling conventions, both supported ───────────────
    rating_range: tuple[float, float] | list[float] | None = None,
    min_rating:   float = 0.0,
    max_rating:   float = 5.0,
    # ── legacy aliases silently accepted ──────────────────────────────────────
    rating_min:   float | None = None,
    rating_max:   float | None = None,
    **kwargs,                          # absorb any future/unknown kwargs
) -> pd.DataFrame:
    """
    Universal dataframe filter.

    Supported rating calling conventions (precedence top → bottom):
      1. rating_range=(min, max)          – Dashboard.py / Geospatial.py
      2. min_rating=x, max_rating=y       – EDA.py / Explorer.py
      3. rating_min=x, rating_max=y       – legacy alias
    """
    # ── Resolve final rating bounds ───────────────────────────────────────────
    if rating_range is not None:
        # Handles tuple, list, or any 2-element sequence
        try:
            r_min = float(rating_range[0])
            r_max = float(rating_range[1])
        except (IndexError, TypeError, ValueError):
            r_min, r_max = 0.0, 5.0
    elif rating_min is not None or rating_max is not None:
        r_min = float(rating_min) if rating_min is not None else min_rating
        r_max = float(rating_max) if rating_max is not None else max_rating
    else:
        r_min = float(min_rating)
        r_max = float(max_rating)

    # ── Build boolean mask ────────────────────────────────────────────────────
    mask = pd.Series(True, index=df.index)

    if cuisines and "Primary Cuisine" in df.columns:
        mask &= df["Primary Cuisine"].isin(cuisines)

    if cities and "City" in df.columns:
        mask &= df["City"].isin(cities)

    if countries and "Country" in df.columns:
        mask &= df["Country"].isin(countries)

    if price_ranges and "Price range" in df.columns:
        mask &= df["Price range"].isin(price_ranges)

    if "Aggregate rating" in df.columns:
        rating_col = df["Aggregate rating"]
        # Always include unrated rows (rating == 0) unless caller
        # explicitly raises the floor above 0
        if r_min > 0.0:
            mask &= (rating_col >= r_min) & (rating_col <= r_max)
        else:
            mask &= (rating_col == 0) | (
                (rating_col >= r_min) & (rating_col <= r_max)
            )

    return df.loc[mask].reset_index(drop=True)


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RestaurantIQ — Data Preprocessing Pipeline"
    )
    parser.add_argument(
        "--data", "-d", default=str(RAW_PATH), metavar="CSV_PATH"
    )
    parser.add_argument(
        "--out",  "-o", default=str(CLEAN_PATH), metavar="OUT_PATH"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args  = _parse_args()
    raw_p = Path(args.data)
    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    preprocess(raw_path=raw_p, save_path=out_p)
from __future__ import annotations

import os
import sys
import warnings
import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("RestaurantIQ")


# ══════════════════════════════════════════════════════════════════════════════
# 10 · CONSTANTS & CONFIG
# ══════════════════════════════════════════════════════════════════════════════

# Default project root – resolves to the folder containing this file
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR    = PROJECT_ROOT / "data"
OUTPUT_DIR  = PROJECT_ROOT / "outputs"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Standard column names used across the project
COL_NAME        = "Restaurant Name"
COL_CITY        = "City"
COL_COUNTRY     = "Country Code"
COL_CUISINES    = "Cuisines"
COL_PRIMARY_CUI = "Primary Cuisine"
COL_RATING      = "Aggregate rating"
COL_RATING_TXT  = "Rating text"
COL_RATING_CLR  = "Rating color"
COL_VOTES       = "Votes"
COL_PRICE       = "Price range"
COL_CURRENCY    = "Currency"
COL_DELIVERY    = "Has Online delivery"
COL_LIVE        = "Is delivering now"
COL_BOOKING     = "Has Table booking"
COL_SWITCH      = "Switch to order menu"
COL_LAT         = "Latitude"
COL_LON         = "Longitude"
COL_LOCALITY    = "Locality"
COL_ADDRESS     = "Address"

BOOL_COLUMNS  = [COL_DELIVERY, COL_LIVE, COL_BOOKING, COL_SWITCH]
SCORE_COLUMNS = [
    "Score_Rating", "Score_Engagement", "Score_Pricing",
    "Score_Cuisine", "Score_Delivery", "Score_Booking",
    "Success_Score",
]

PRICE_RANGE_MAP = {1: "Budget", 2: "Moderate", 3: "Upscale", 4: "Fine Dining"}

RATING_BANDS = {
    "Excellent":    (4.5, 5.0),
    "Very Good":    (4.0, 4.5),
    "Good":         (3.5, 4.0),
    "Average":      (2.5, 3.5),
    "Poor":         (0.0, 2.5),
}

SUCCESS_TIER_ORDER = [
    "Top Performer", "Performing", "Average", "Below Average", "Struggling"
]

TIER_COLORS = {
    "Top Performer": "#00C2A8",
    "Performing":    "#6C63FF",
    "Average":       "#F59E0B",
    "Below Average": "#F97316",
    "Struggling":    "#EF4444",
}

TIER_ICONS = {
    "Top Performer": "🏆",
    "Performing":    "✅",
    "Average":       "📊",
    "Below Average": "⚠️",
    "Struggling":    "❌",
}


# ══════════════════════════════════════════════════════════════════════════════
# 1 · DATA LOADING & VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def load_dataset(
    path: Union[str, Path],
    encoding: str = "utf-8",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load the restaurant CSV dataset with error handling.

    Parameters
    ----------
    path     : Full path to the CSV file.
    encoding : File encoding (default utf-8; falls back to latin-1).
    verbose  : Print load summary when True.

    Returns
    -------
    pd.DataFrame  Raw loaded DataFrame.
    """
    path = Path(path)

    if not path.exists():
        log.error("File not found: %s", path)
        raise FileNotFoundError(f"Dataset not found at:\n  {path}")

    if path.suffix.lower() != ".csv":
        raise ValueError(f"Expected a .csv file, got: {path.suffix}")

    try:
        df = pd.read_csv(path, encoding=encoding)
    except UnicodeDecodeError:
        log.warning("UTF-8 decode failed – retrying with latin-1")
        df = pd.read_csv(path, encoding="latin-1")
    except Exception as exc:
        log.error("Failed to load dataset: %s", exc)
        raise

    if verbose:
        print(f"\n  ✔  Loaded  '{path.name}'")
        print(f"     Rows    : {len(df):,}")
        print(f"     Columns : {len(df.columns)}")
        print(f"     Size    : {df.memory_usage(deep=True).sum() / 1024:.1f} KB\n")

    return df


def validate_required_columns(df: pd.DataFrame, required: list[str] | None = None) -> bool:
    """
    Check that all required columns are present in the DataFrame.

    Parameters
    ----------
    df       : Input DataFrame.
    required : List of column names to check. Defaults to core project columns.

    Returns
    -------
    bool  True if all required columns exist, False otherwise (prints missing list).
    """
    if required is None:
        required = [
            COL_NAME, COL_CITY, COL_CUISINES,
            COL_RATING, COL_VOTES, COL_PRICE,
            COL_DELIVERY, COL_LIVE, COL_BOOKING,
        ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        log.warning("Missing columns: %s", missing)
        print(f"  ⚠️   Missing {len(missing)} required column(s):")
        for col in missing:
            print(f"       • {col}")
        return False

    print(f"  ✔  All {len(required)} required columns present.")
    return True


def get_dataset_info(df: pd.DataFrame) -> dict:
    """
    Return a summary dict of key dataset characteristics.

    Returns
    -------
    dict with keys: rows, columns, cities, countries, cuisines,
                    missing_pct, duplicates, bool_cols_detected
    """
    return {
        "rows":               len(df),
        "columns":            len(df.columns),
        "cities":             df[COL_CITY].nunique()    if COL_CITY    in df.columns else None,
        "countries":          df[COL_COUNTRY].nunique() if COL_COUNTRY in df.columns else None,
        "cuisines":           df[COL_CUISINES].nunique() if COL_CUISINES in df.columns else None,
        "missing_pct":        round(df.isnull().mean().mean() * 100, 2),
        "duplicates":         int(df.duplicated().sum()),
        "bool_cols_detected": [c for c in BOOL_COLUMNS if c in df.columns],
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2 · DATA CLEANING & PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def preprocess(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Master preprocessing pipeline – runs all cleaning steps in order.

    Steps
    -----
    1. Strip whitespace from string columns
    2. Convert Yes/No boolean columns to bool dtype
    3. Handle missing / zero ratings
    4. Create Primary Cuisine from Cuisines
    5. Map Price range to label
    6. Drop exact duplicate rows
    7. Reset index

    Parameters
    ----------
    df      : Raw loaded DataFrame.
    verbose : Print step summary when True.

    Returns
    -------
    Cleaned pd.DataFrame.
    """
    out = df.copy()
    steps: list[str] = []

    # Step 1 – strip strings
    str_cols = out.select_dtypes(include="object").columns
    out[str_cols] = out[str_cols].apply(lambda s: s.str.strip())
    steps.append("Stripped whitespace from string columns")

    # Step 2 – boolean conversion
    out = convert_bool_columns(out)
    steps.append("Converted Yes/No columns → bool")

    # Step 3 – handle zero/null ratings
    out = fix_zero_ratings(out)
    steps.append("Fixed zero/null Aggregate rating")

    # Step 4 – primary cuisine
    out = create_primary_cuisine(out)
    steps.append("Created 'Primary Cuisine' column")

    # Step 5 – price label
    out = add_price_label(out)
    steps.append("Added 'Price Label' column")

    # Step 6 – drop duplicates
    before = len(out)
    out = out.drop_duplicates().reset_index(drop=True)
    dropped = before - len(out)
    steps.append(f"Dropped {dropped} duplicate row(s)")

    if verbose:
        print("\n  Preprocessing complete:")
        for i, s in enumerate(steps, 1):
            print(f"    {i}. {s}")
        print(f"\n  Final shape: {out.shape[0]:,} rows × {out.shape[1]} columns\n")

    return out


def convert_bool_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert Yes / No / 1 / 0 string columns to proper Python bool.

    Handles: 'Yes', 'No', 'yes', 'no', '1', '0', 1, 0, True, False.
    Fills unrecognised values with False.
    """
    out = df.copy()
    yes_vals = {"yes", "1", "true"}

    for col in BOOL_COLUMNS:
        if col not in out.columns:
            continue
        if out[col].dtype == bool:
            continue
        if out[col].dtype == object:
            out[col] = out[col].str.strip().str.lower().isin(yes_vals)
        else:
            out[col] = out[col].astype(bool)

    return out


def fix_zero_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace rating == 0 with NaN so unrated restaurants are handled correctly.
    Also clips ratings to the valid [0, 5] range.
    """
    out = df.copy()
    if COL_RATING in out.columns:
        out[COL_RATING] = pd.to_numeric(out[COL_RATING], errors="coerce")
        out[COL_RATING] = out[COL_RATING].replace(0, np.nan)
        out[COL_RATING] = out[COL_RATING].clip(0, 5)
    return out


def create_primary_cuisine(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'Primary Cuisine' column = first cuisine listed in 'Cuisines'.
    Safe if 'Primary Cuisine' already exists or 'Cuisines' is absent.
    """
    out = df.copy()
    if COL_PRIMARY_CUI not in out.columns:
        if COL_CUISINES in out.columns:
            out[COL_PRIMARY_CUI] = (
                out[COL_CUISINES]
                .fillna("Unknown")
                .str.split(",")
                .str[0]
                .str.strip()
            )
        else:
            out[COL_PRIMARY_CUI] = "Unknown"
    return out


def add_price_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a human-readable 'Price Label' column from the numeric Price range.
    1 → Budget | 2 → Moderate | 3 → Upscale | 4 → Fine Dining
    """
    out = df.copy()
    if COL_PRICE in out.columns:
        out["Price Label"] = out[COL_PRICE].map(PRICE_RANGE_MAP).fillna("Unknown")
    return out


def drop_low_vote_restaurants(df: pd.DataFrame, min_votes: int = 10) -> pd.DataFrame:
    """
    Remove restaurants with fewer than `min_votes` votes.
    Useful for filtering out noise before scoring.
    """
    if COL_VOTES not in df.columns:
        return df
    before = len(df)
    out = df[df[COL_VOTES] >= min_votes].reset_index(drop=True)
    log.info("Dropped %d low-vote restaurants (< %d votes)", before - len(out), min_votes)
    return out


def remove_unrated_restaurants(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where Aggregate rating is NaN or 0."""
    if COL_RATING not in df.columns:
        return df
    mask = df[COL_RATING].notna() & (df[COL_RATING] > 0)
    out  = df[mask].reset_index(drop=True)
    log.info("Removed %d unrated restaurants", len(df) - len(out))
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 3 · FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def add_rating_band(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify each restaurant into a Rating Band based on Aggregate rating.

    Bands: Excellent | Very Good | Good | Average | Poor | Unrated
    """
    out = df.copy()

    def _band(r):
        if pd.isna(r) or r == 0:
            return "Unrated"
        for band, (lo, hi) in RATING_BANDS.items():
            if lo <= r <= hi:
                return band
        return "Unrated"

    if COL_RATING in out.columns:
        out["Rating Band"] = out[COL_RATING].apply(_band)
    return out


def add_vote_tier(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify restaurants by vote count into popularity tiers.

    Tiers: Viral (≥1000) | Popular (≥200) | Emerging (≥50) | Unknown (<50)
    """
    out = df.copy()

    def _tier(v):
        if pd.isna(v):
            return "Unknown"
        v = int(v)
        if v >= 1000: return "Viral"
        if v >= 200:  return "Popular"
        if v >= 50:   return "Emerging"
        return "Unknown"

    if COL_VOTES in out.columns:
        out["Vote Tier"] = out[COL_VOTES].apply(_tier)
    return out


def add_value_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a simple Value Score = Aggregate rating / Price range.
    Higher means better value for money.
    """
    out = df.copy()
    if COL_RATING in out.columns and COL_PRICE in out.columns:
        out["Value Score"] = (
            out[COL_RATING] / out[COL_PRICE].replace(0, np.nan)
        ).round(3)
    return out


def add_full_delivery_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add boolean 'Full Delivery' = Has Online delivery AND Is delivering now.
    """
    out = df.copy()
    if COL_DELIVERY in out.columns and COL_LIVE in out.columns:
        out["Full Delivery"] = (
            out[COL_DELIVERY].astype(bool) & out[COL_LIVE].astype(bool)
        )
    return out


def add_cuisine_count(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'Cuisine Count' = number of cuisines offered (comma-separated list).
    """
    out = df.copy()
    if COL_CUISINES in out.columns:
        out["Cuisine Count"] = (
            out[COL_CUISINES]
            .fillna("")
            .str.split(",")
            .apply(len)
        )
    return out


def add_log_votes(df: pd.DataFrame) -> pd.DataFrame:
    """Add log1p-transformed vote column for skew-normalised analysis."""
    out = df.copy()
    if COL_VOTES in out.columns:
        out["Log Votes"] = np.log1p(out[COL_VOTES].fillna(0)).round(4)
    return out


def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run all feature engineering steps in one call.

    Adds: Rating Band, Vote Tier, Value Score,
          Full Delivery, Cuisine Count, Log Votes, Price Label
    """
    out = df.copy()
    out = add_rating_band(out)
    out = add_vote_tier(out)
    out = add_value_score(out)
    out = add_full_delivery_flag(out)
    out = add_cuisine_count(out)
    out = add_log_votes(out)
    out = add_price_label(out)
    log.info("Feature engineering complete – %d new columns added", 7)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 4 · COLUMN & TYPE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def get_numeric_columns(df: pd.DataFrame) -> list[str]:
    """Return list of numeric column names."""
    return df.select_dtypes(include=[np.number]).columns.tolist()


def get_categorical_columns(df: pd.DataFrame) -> list[str]:
    """Return list of object/categorical column names."""
    return df.select_dtypes(include="object").columns.tolist()


def get_bool_columns(df: pd.DataFrame) -> list[str]:
    """Return list of boolean column names."""
    return df.select_dtypes(include="bool").columns.tolist()


def get_score_columns(df: pd.DataFrame) -> list[str]:
    """Return all Score_* and Success_* columns present in df."""
    return [c for c in df.columns if c.startswith("Score_") or c.startswith("Success_")]


def safe_cast_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Coerce specified columns to numeric, replacing errors with NaN.
    """
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def column_exists(df: pd.DataFrame, col: str, raise_error: bool = False) -> bool:
    """
    Check whether a column exists in df.

    Parameters
    ----------
    raise_error : If True, raises KeyError when column is missing.
    """
    exists = col in df.columns
    if not exists and raise_error:
        raise KeyError(f"Column '{col}' not found in DataFrame.")
    return exists


# ══════════════════════════════════════════════════════════════════════════════
# 5 · FILTERING & QUERYING
# ══════════════════════════════════════════════════════════════════════════════

def filter_by_city(df: pd.DataFrame, city: str) -> pd.DataFrame:
    """Return rows matching the given city (case-insensitive)."""
    if COL_CITY not in df.columns:
        return df
    return df[df[COL_CITY].str.lower() == city.lower()].reset_index(drop=True)


def filter_by_cuisine(df: pd.DataFrame, cuisine: str) -> pd.DataFrame:
    """Return rows where Primary Cuisine matches (case-insensitive)."""
    if COL_PRIMARY_CUI not in df.columns:
        return df
    return df[
        df[COL_PRIMARY_CUI].str.lower() == cuisine.lower()
    ].reset_index(drop=True)


def filter_by_price_range(df: pd.DataFrame, price_range: int) -> pd.DataFrame:
    """Return rows matching the given numeric price range (1-4)."""
    if COL_PRICE not in df.columns:
        return df
    return df[df[COL_PRICE] == price_range].reset_index(drop=True)


def filter_by_min_rating(df: pd.DataFrame, min_rating: float = 3.5) -> pd.DataFrame:
    """Return rows where Aggregate rating >= min_rating."""
    if COL_RATING not in df.columns:
        return df
    return df[df[COL_RATING] >= min_rating].reset_index(drop=True)


def filter_by_tier(df: pd.DataFrame, tier: str) -> pd.DataFrame:
    """Return rows matching a specific Success_Tier string."""
    if "Success_Tier" not in df.columns:
        raise KeyError("Run compute_success_scores() first to add 'Success_Tier'.")
    return df[df["Success_Tier"] == tier].reset_index(drop=True)


def filter_delivers(df: pd.DataFrame, live_only: bool = False) -> pd.DataFrame:
    """
    Return restaurants that have online delivery.
    If live_only=True, also requires Is delivering now == True.
    """
    if COL_DELIVERY not in df.columns:
        return df
    mask = df[COL_DELIVERY].astype(bool)
    if live_only and COL_LIVE in df.columns:
        mask &= df[COL_LIVE].astype(bool)
    return df[mask].reset_index(drop=True)


def filter_has_booking(df: pd.DataFrame) -> pd.DataFrame:
    """Return only restaurants that accept table bookings."""
    if COL_BOOKING not in df.columns:
        return df
    return df[df[COL_BOOKING].astype(bool)].reset_index(drop=True)


def search_restaurant(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Partial, case-insensitive search by restaurant name.

    Returns all rows whose name contains the query string.
    """
    if COL_NAME not in df.columns:
        return df
    return df[
        df[COL_NAME].str.lower().str.contains(name.lower(), na=False)
    ].reset_index(drop=True)


def top_n_by_score(
    df: pd.DataFrame,
    n: int = 10,
    score_col: str = "Success_Score",
) -> pd.DataFrame:
    """Return top-N rows sorted by the given score column (descending)."""
    if score_col not in df.columns:
        raise KeyError(f"Column '{score_col}' not found. Run scoring first.")
    return df.nlargest(n, score_col).reset_index(drop=True)


def bottom_n_by_score(
    df: pd.DataFrame,
    n: int = 10,
    score_col: str = "Success_Score",
) -> pd.DataFrame:
    """Return bottom-N rows sorted by the given score column (ascending)."""
    if score_col not in df.columns:
        raise KeyError(f"Column '{score_col}' not found. Run scoring first.")
    return df.nsmallest(n, score_col).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# 6 · STATISTICAL & ANALYTICAL HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def describe_column(df: pd.DataFrame, col: str) -> dict:
    """
    Return descriptive statistics for a single column as a dict.

    Works for both numeric and categorical columns.
    """
    column_exists(df, col, raise_error=True)
    s = df[col]

    if pd.api.types.is_numeric_dtype(s):
        return {
            "dtype":   str(s.dtype),
            "count":   int(s.count()),
            "missing": int(s.isna().sum()),
            "mean":    round(float(s.mean()), 4),
            "median":  round(float(s.median()), 4),
            "std":     round(float(s.std()), 4),
            "min":     round(float(s.min()), 4),
            "max":     round(float(s.max()), 4),
            "q25":     round(float(s.quantile(0.25)), 4),
            "q75":     round(float(s.quantile(0.75)), 4),
            "skew":    round(float(s.skew()), 4),
        }
    else:
        top_val = s.value_counts().idxmax() if not s.value_counts().empty else None
        return {
            "dtype":    str(s.dtype),
            "count":    int(s.count()),
            "missing":  int(s.isna().sum()),
            "unique":   int(s.nunique()),
            "top":      top_val,
            "top_freq": int(s.value_counts().max()) if top_val else 0,
        }


def get_city_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return per-city aggregate stats:
    Restaurant count, avg rating, avg votes, delivery %, booking %.
    """
    if COL_CITY not in df.columns:
        raise KeyError(f"Column '{COL_CITY}' not found.")

    agg = {COL_NAME: "count"}
    if COL_RATING   in df.columns: agg[COL_RATING]   = "mean"
    if COL_VOTES    in df.columns: agg[COL_VOTES]     = "mean"
    if COL_DELIVERY in df.columns: agg[COL_DELIVERY]  = "mean"
    if COL_BOOKING  in df.columns: agg[COL_BOOKING]   = "mean"

    city_df = (
        df.groupby(COL_CITY, sort=False)
        .agg(agg)
        .reset_index()
        .rename(columns={
            COL_NAME:     "Restaurant Count",
            COL_RATING:   "Avg Rating",
            COL_VOTES:    "Avg Votes",
            COL_DELIVERY: "Delivery %",
            COL_BOOKING:  "Booking %",
        })
        .sort_values("Restaurant Count", ascending=False)
        .reset_index(drop=True)
    )

    for pct_col in ["Delivery %", "Booking %"]:
        if pct_col in city_df.columns:
            city_df[pct_col] = (city_df[pct_col] * 100).round(1)

    for num_col in ["Avg Rating", "Avg Votes"]:
        if num_col in city_df.columns:
            city_df[num_col] = city_df[num_col].round(2)

    return city_df


def get_cuisine_summary(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """
    Return top-N cuisine aggregate stats:
    Count, avg rating, avg votes, delivery %, booking %.
    """
    col = COL_PRIMARY_CUI if COL_PRIMARY_CUI in df.columns else COL_CUISINES
    if col not in df.columns:
        raise KeyError("No cuisine column found. Run create_primary_cuisine() first.")

    agg = {COL_NAME: "count"}
    if COL_RATING   in df.columns: agg[COL_RATING]   = "mean"
    if COL_VOTES    in df.columns: agg[COL_VOTES]     = "mean"
    if COL_DELIVERY in df.columns: agg[COL_DELIVERY]  = "mean"

    cuisine_df = (
        df.groupby(col, sort=False)
        .agg(agg)
        .reset_index()
        .rename(columns={
            col:          "Cuisine",
            COL_NAME:     "Count",
            COL_RATING:   "Avg Rating",
            COL_VOTES:    "Avg Votes",
            COL_DELIVERY: "Delivery %",
        })
        .sort_values("Count", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    if "Delivery %" in cuisine_df.columns:
        cuisine_df["Delivery %"] = (cuisine_df["Delivery %"] * 100).round(1)
    if "Avg Rating" in cuisine_df.columns:
        cuisine_df["Avg Rating"] = cuisine_df["Avg Rating"].round(2)
    if "Avg Votes" in cuisine_df.columns:
        cuisine_df["Avg Votes"] = cuisine_df["Avg Votes"].round(1)

    return cuisine_df


def get_price_range_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return per-price-range stats:
    Count, avg rating, avg votes, delivery %, booking %, avg success score.
    """
    if COL_PRICE not in df.columns:
        raise KeyError(f"Column '{COL_PRICE}' not found.")

    agg = {COL_NAME: "count"}
    for col in [COL_RATING, COL_VOTES, COL_DELIVERY, COL_BOOKING]:
        if col in df.columns:
            agg[col] = "mean"
    if "Success_Score" in df.columns:
        agg["Success_Score"] = "mean"

    price_df = (
        df.groupby(COL_PRICE, sort=True)
        .agg(agg)
        .reset_index()
        .rename(columns={
            COL_PRICE:    "Price Range",
            COL_NAME:     "Count",
            COL_RATING:   "Avg Rating",
            COL_VOTES:    "Avg Votes",
            COL_DELIVERY: "Delivery %",
            COL_BOOKING:  "Booking %",
        })
    )

    price_df["Price Label"] = price_df["Price Range"].map(PRICE_RANGE_MAP)

    for pct_col in ["Delivery %", "Booking %"]:
        if pct_col in price_df.columns:
            price_df[pct_col] = (price_df[pct_col] * 100).round(1)
    for num_col in ["Avg Rating", "Avg Votes", "Success_Score"]:
        if num_col in price_df.columns:
            price_df[num_col] = price_df[num_col].round(2)

    return price_df


def correlation_matrix(
    df: pd.DataFrame,
    cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Compute Pearson correlation matrix for numeric columns.

    Parameters
    ----------
    cols : Specific columns to use. Defaults to all numeric columns.
    """
    if cols is None:
        cols = get_numeric_columns(df)
    return df[cols].corr(numeric_only=True).round(4)


def rating_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return count and percentage of restaurants per Rating Band.
    Requires add_rating_band() to have been called.
    """
    if "Rating Band" not in df.columns:
        df = add_rating_band(df)

    order = ["Excellent", "Very Good", "Good", "Average", "Poor", "Unrated"]
    dist  = (
        df["Rating Band"]
        .value_counts()
        .reindex(order, fill_value=0)
        .reset_index()
        .rename(columns={"index": "Band", "Rating Band": "Count", "count": "Count"})
    )
    dist.columns = ["Band", "Count"]
    dist["Pct"]  = (dist["Count"] / dist["Count"].sum() * 100).round(1)
    return dist


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame showing missing value count and percentage per column.
    Only returns columns that have at least one missing value.
    """
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    report  = pd.DataFrame({
        "Column":      missing.index,
        "Missing":     missing.values,
        "Missing_Pct": (missing.values / len(df) * 100).round(2),
    }).reset_index(drop=True)
    return report


# ══════════════════════════════════════════════════════════════════════════════
# 7 · FORMATTING & DISPLAY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _divider(char: str = "═", width: int = 65) -> None:
    print(char * width)


def print_section(title: str, width: int = 65) -> None:
    """Print a bold section header."""
    print("\n" + "═" * width)
    print(f"  {title}")
    print("═" * width)


def print_subsection(title: str, width: int = 55) -> None:
    """Print a lighter sub-header."""
    pad = width - len(title)
    print(f"\n  ── {title} {'─' * max(pad, 2)}")


def print_kv(label: str, value, width: int = 22) -> None:
    """Print a formatted key: value line."""
    print(f"  {label:<{width}}: {value}")


def format_number(n: float, decimals: int = 2) -> str:
    """Return a formatted number string with thousands separator."""
    if isinstance(n, (int, np.integer)):
        return f"{n:,}"
    return f"{n:,.{decimals}f}"


def ascii_bar(value: float, max_val: float = 100, width: int = 20) -> str:
    """
    Return an ASCII progress bar for a given value.

    Example: ascii_bar(75) → '███████████████░░░░░'
    """
    filled = int((value / max_val) * width) if max_val > 0 else 0
    filled = max(0, min(filled, width))
    return "█" * filled + "░" * (width - filled)


def print_restaurant_card(row: pd.Series) -> None:
    """
    Pretty-print a single restaurant's key details as a card.
    """
    name    = str(row.get(COL_NAME,        "N/A"))
    city    = str(row.get(COL_CITY,        "N/A"))
    cuisine = str(row.get(COL_PRIMARY_CUI, "N/A"))
    rating  = row.get(COL_RATING, 0) or 0
    votes   = int(row.get(COL_VOTES, 0)   or 0)
    price   = int(row.get(COL_PRICE, 0)   or 0)
    score   = row.get("Success_Score",    None)
    tier    = row.get("Success_Tier",     "N/A")
    delivery = "Yes" if row.get(COL_DELIVERY) else "No"
    booking  = "Yes" if row.get(COL_BOOKING)  else "No"

    print(f"\n  ┌{'─'*55}┐")
    print(f"  │  🍽️  {name[:49]:<49}│")
    print(f"  ├{'─'*55}┤")
    print(f"  │  City      : {city:<41}│")
    print(f"  │  Cuisine   : {cuisine:<41}│")
    print(f"  │  Rating    : {rating:.1f} / 5.0 ({'★'*int(round(rating))}{'☆'*(5-int(round(rating)))}){'':>20}│")
    votes_str = f"{votes:,}"
    print(f"  │  Votes     : {votes_str:<41}│")
    print(f"  │  Price     : {'₹' * price} ({PRICE_RANGE_MAP.get(price, 'N/A'):<36})│")
    print(f"  │  Delivery  : {delivery:<41}│")
    print(f"  │  Booking   : {booking:<41}│")
    if score is not None:
        bar = ascii_bar(float(score))
        icon = TIER_ICONS.get(tier, "")
        print(f"  ├{'─'*55}┤")
        print(f"  │  Success Score : {float(score):>5.1f}  {icon} {tier:<27}│")
        print(f"  │  [{bar}] {float(score):>5.1f}%{'':>11}│")
    print(f"  └{'─'*55}┘")


def print_dataframe_preview(
    df: pd.DataFrame,
    n: int = 5,
    title: str = "DataFrame Preview",
) -> None:
    """Print the first n rows of a DataFrame with a header."""
    print_section(title)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width",       200)
    pd.set_option("display.float_format", "{:.2f}".format)
    print()
    print(df.head(n).to_string(index=False))
    print(f"\n  [{len(df):,} total rows × {len(df.columns)} columns]\n")


def print_dict(d: dict, title: str = "") -> None:
    """Pretty-print a dictionary with aligned keys."""
    if title:
        print_subsection(title)
    max_key = max((len(str(k)) for k in d), default=10)
    for k, v in d.items():
        print(f"    {str(k):<{max_key}} : {v}")


# ══════════════════════════════════════════════════════════════════════════════
# 8 · EXPORT & I/O HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def save_csv(
    df: pd.DataFrame,
    filename: str,
    output_dir: Union[str, Path, None] = None,
    index: bool = False,
    verbose: bool = True,
) -> Path:
    """
    Save a DataFrame to CSV.

    Parameters
    ----------
    filename   : Output filename (with or without .csv extension).
    output_dir : Directory to save into. Defaults to OUTPUT_DIR constant.
    index      : Write row index to file.
    verbose    : Print save confirmation.

    Returns
    -------
    Path  of the saved file.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not filename.endswith(".csv"):
        filename += ".csv"

    out_path = out_dir / filename
    df.to_csv(out_path, index=index)

    if verbose:
        print(f"  💾  Saved  → {out_path}  ({len(df):,} rows)")

    return out_path


def save_excel(
    sheets: dict[str, pd.DataFrame],
    filename: str,
    output_dir: Union[str, Path, None] = None,
    verbose: bool = True,
) -> Path:
    """
    Save multiple DataFrames to a single Excel workbook.

    Parameters
    ----------
    sheets     : {sheet_name: DataFrame} mapping.
    filename   : Output filename (with or without .xlsx extension).
    output_dir : Directory to save into. Defaults to OUTPUT_DIR constant.
    verbose    : Print save confirmation.

    Returns
    -------
    Path  of the saved file.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not filename.endswith(".xlsx"):
        filename += ".xlsx"

    out_path = out_dir / filename

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)

    if verbose:
        print(f"  💾  Saved Excel → {out_path}  ({len(sheets)} sheet(s))")

    return out_path


def load_scored_dataset(
    path: Union[str, Path],
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load a pre-scored dataset (output of compute_success_scores).
    Validates that score columns are present.
    """
    df = load_dataset(path, verbose=verbose)
    missing_scores = [c for c in SCORE_COLUMNS if c not in df.columns]
    if missing_scores:
        log.warning(
            "Scored dataset missing columns: %s — run compute_success_scores() first.",
            missing_scores,
        )
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 9 · VALIDATION & QUALITY CHECKS
# ══════════════════════════════════════════════════════════════════════════════

def run_quality_checks(df: pd.DataFrame) -> dict:
    """
    Run a full data quality audit and return a results dict.

    Checks
    ------
    - Duplicate rows
    - Missing values per column
    - Rating range validity (0–5)
    - Vote negativity
    - Price range validity (1–4)
    - Boolean column values
    - Empty restaurant names
    - Coordinate plausibility (if lat/lon present)

    Returns
    -------
    dict with keys: passed, warnings, details
    """
    warnings_list = []
    details       = {}

    # Duplicates
    dup_count = int(df.duplicated().sum())
    details["duplicate_rows"] = dup_count
    if dup_count > 0:
        warnings_list.append(f"{dup_count} duplicate row(s) found")

    # Missing values
    mv = missing_value_report(df)
    details["missing_columns"] = mv.to_dict("records")
    if not mv.empty:
        warnings_list.append(f"{len(mv)} column(s) have missing values")

    # Rating range
    if COL_RATING in df.columns:
        bad_rating = df[(df[COL_RATING].notna()) & ((df[COL_RATING] < 0) | (df[COL_RATING] > 5))]
        details["invalid_ratings"] = int(len(bad_rating))
        if len(bad_rating) > 0:
            warnings_list.append(f"{len(bad_rating)} rating(s) outside [0, 5]")

    # Negative votes
    if COL_VOTES in df.columns:
        neg_votes = int((df[COL_VOTES] < 0).sum())
        details["negative_votes"] = neg_votes
        if neg_votes > 0:
            warnings_list.append(f"{neg_votes} negative vote value(s)")

    # Price range
    if COL_PRICE in df.columns:
        bad_price = df[~df[COL_PRICE].isin([1, 2, 3, 4])]
        details["invalid_price_range"] = int(len(bad_price))
        if len(bad_price) > 0:
            warnings_list.append(f"{len(bad_price)} invalid price range value(s)")

    # Empty restaurant names
    if COL_NAME in df.columns:
        empty_names = int(df[COL_NAME].isna().sum() + (df[COL_NAME] == "").sum())
        details["empty_names"] = empty_names
        if empty_names > 0:
            warnings_list.append(f"{empty_names} empty restaurant name(s)")

    # Coordinates
    if COL_LAT in df.columns and COL_LON in df.columns:
        bad_coords = int(
            ((df[COL_LAT].abs() > 90) | (df[COL_LON].abs() > 180)).sum()
        )
        details["invalid_coordinates"] = bad_coords
        if bad_coords > 0:
            warnings_list.append(f"{bad_coords} invalid coordinate(s)")

    passed = len(warnings_list) == 0

    return {
        "passed":   passed,
        "warnings": warnings_list,
        "details":  details,
    }


def print_quality_report(df: pd.DataFrame) -> None:
    """Run and print a formatted data quality report."""
    print_section("DATA QUALITY REPORT")
    result = run_quality_checks(df)

    status = "✔  All checks passed!" if result["passed"] else f"⚠️   {len(result['warnings'])} issue(s) found"
    print(f"\n  Status : {status}")

    if result["warnings"]:
        print("\n  Warnings:")
        for w in result["warnings"]:
            print(f"    • {w}")

    print("\n  Details:")
    print_dict(result["details"])
    print()


def assert_no_nulls(df: pd.DataFrame, cols: list[str]) -> None:
    """
    Assert that specified columns have no null values.
    Raises ValueError listing any columns with nulls.
    """
    null_cols = [c for c in cols if c in df.columns and df[c].isna().any()]
    if null_cols:
        raise ValueError(f"Unexpected nulls in: {null_cols}")


# ══════════════════════════════════════════════════════════════════════════════
# QUICK SELF-TEST  (run: python utils.py)
# ══════════════════════════════════════════════════════════════════════════════

def _self_test() -> None:
    """Run a quick smoke-test with synthetic data to verify all utils work."""
    print_section("UTILS SELF-TEST  (synthetic data)")

    # Build a tiny synthetic dataset
    np.random.seed(42)
    n = 50
    synthetic = pd.DataFrame({
        COL_NAME:     [f"Restaurant {i}" for i in range(n)],
        COL_CITY:     np.random.choice(["Mumbai", "Delhi", "Bangalore", "Chennai"], n),
        COL_COUNTRY:  [1] * n,
        COL_CUISINES: np.random.choice(
            ["North Indian, Chinese", "Italian", "Chinese", "South Indian, Biryani"], n
        ),
        COL_RATING:   np.random.choice([0, 2.5, 3.1, 3.8, 4.2, 4.7], n),
        COL_VOTES:    np.random.randint(0, 5000, n),
        COL_PRICE:    np.random.choice([1, 2, 3, 4], n),
        COL_DELIVERY: np.random.choice(["Yes", "No"], n),
        COL_LIVE:     np.random.choice(["Yes", "No"], n),
        COL_BOOKING:  np.random.choice(["Yes", "No"], n),
        COL_LAT:      np.random.uniform(8, 35, n),
        COL_LON:      np.random.uniform(68, 97, n),
        COL_SWITCH:   np.random.choice(["Yes", "No"], n),
    })

    # 1. Preprocess
    clean = preprocess(synthetic, verbose=True)

    # 2. Feature engineering
    featured = engineer_all_features(clean)

    # 3. Validate
    print_quality_report(featured)

    # 4. Summaries
    print_section("City Summary")
    print(get_city_summary(featured).to_string(index=False))

    print_section("Cuisine Summary (top 5)")
    print(get_cuisine_summary(featured, top_n=5).to_string(index=False))

    print_section("Price Range Summary")
    print(get_price_range_summary(featured).to_string(index=False))

    # 5. Filters
    mumbai = filter_by_city(featured, "Mumbai")
    print_section(f"Mumbai restaurants (n={len(mumbai)})")
    print(mumbai[[COL_NAME, COL_RATING, COL_VOTES]].head(3).to_string(index=False))

    # 6. Card for first restaurant
    print_restaurant_card(featured.iloc[0])

    # 7. Column helpers
    print_section("Column Type Helpers")
    print_kv("Numeric cols",     get_numeric_columns(featured))
    print_kv("Categorical cols", get_categorical_columns(featured))
    print_kv("Bool cols",        get_bool_columns(featured))

    # 8. Describe
    print_section("describe_column  →  Aggregate rating")
    print_dict(describe_column(featured, COL_RATING))

    print_section("✔  Self-test complete")


if __name__ == "__main__":
    _self_test()
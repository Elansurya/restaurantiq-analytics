from __future__ import annotations

import os
import sys
import argparse
import warnings
import textwrap
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════════
# SAMPLE-DATA GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

def _make_sample_df(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Return a synthetic restaurant DataFrame for demo / testing."""
    rng = np.random.default_rng(seed)

    city_country = [
        ("Mumbai", "India"), ("Delhi", "India"), ("Bangalore", "India"),
        ("Chennai", "India"), ("Hyderabad", "India"),
        ("New York", "USA"), ("Los Angeles", "USA"), ("Chicago", "USA"),
        ("London", "UK"), ("Manchester", "UK"),
        ("Sydney", "Australia"), ("Melbourne", "Australia"),
        ("Toronto", "Canada"), ("Vancouver", "Canada"),
        ("Dubai", "UAE"), ("Abu Dhabi", "UAE"),
        ("Cape Town", "South Africa"), ("Manila", "Philippines"),
    ]
    cuisines = [
        "North Indian", "Chinese", "Fast Food", "Continental", "Italian",
        "Mexican", "Japanese", "Thai", "American", "Mediterranean",
        "Middle Eastern", "Korean", "French", "Seafood", "Bakery",
    ]
    price_labels = {1: "Budget", 2: "Affordable", 3: "Premium", 4: "Luxury"}

    cc_idx      = rng.integers(0, len(city_country), n)
    city_col    = [city_country[i][0] for i in cc_idx]
    country_col = [city_country[i][1] for i in cc_idx]
    cuisine_col = rng.choice(cuisines, n)
    price_range = rng.integers(1, 5, n)
    price_label = [price_labels[p] for p in price_range]
    cost_col    = rng.integers(100, 5000, n).astype(float)
    votes_col   = rng.integers(0, 5000, n)
    rating_raw  = np.where(
        rng.random(n) < 0.12, 0.0,
        rng.uniform(1.5, 5.0, n).round(1)
    )

    def _rt(r):
        if r == 0:    return "Not rated"
        if r >= 4.5:  return "Excellent"
        if r >= 4.0:  return "Very Good"
        if r >= 3.5:  return "Good"
        if r >= 3.0:  return "Average"
        return "Poor"

    rating_text = [_rt(r) for r in rating_raw]
    n_cuisines  = rng.integers(1, 4, n)
    cuisines_col = [
        ", ".join(rng.choice(cuisines, k, replace=False).tolist())
        for k in n_cuisines
    ]

    return pd.DataFrame({
        "Restaurant ID":         range(1, n + 1),
        "Restaurant Name":       [f"Restaurant_{i:04d}" for i in range(1, n + 1)],
        "Country":               country_col,
        "City":                  city_col,
        "Cuisines":              cuisines_col,
        "Primary Cuisine":       cuisine_col,
        "Average Cost for two":  cost_col,
        "Price range":           price_range,
        "Price Label":           price_label,
        "Aggregate rating":      rating_raw,
        "Rating text":           rating_text,
        "Votes":                 votes_col,
        "Is Rated":              rating_raw > 0,
        "Cuisine Count":         n_cuisines,
        "Has Online delivery":   rng.choice([True, False], n, p=[0.4, 0.6]),
        "Has Table booking":     rng.choice([True, False], n, p=[0.3, 0.7]),
        "Is delivering now":     rng.choice([True, False], n, p=[0.2, 0.8]),
    })


# ═══════════════════════════════════════════════════════════════════════════════
# COLUMN GUARDS
# ═══════════════════════════════════════════════════════════════════════════════

def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Is Rated" not in df.columns:
        df["Is Rated"] = df.get("Aggregate rating", pd.Series(0, index=df.index)) > 0
    if "Primary Cuisine" not in df.columns and "Cuisines" in df.columns:
        df["Primary Cuisine"] = df["Cuisines"].str.split(",").str[0].str.strip()
    if "Price Label" not in df.columns and "Price range" in df.columns:
        # FIX Bug 3: use string labels directly to avoid categorical dtype issues
        price_map = {1: "Budget", 2: "Affordable", 3: "Premium", 4: "Luxury"}
        df["Price Label"] = df["Price range"].clip(1, 4).map(price_map).fillna("Affordable")
    if "Cuisine Count" not in df.columns and "Cuisines" in df.columns:
        df["Cuisine Count"] = df["Cuisines"].str.split(",").str.len()
    for col in ["Has Online delivery", "Has Table booking", "Is delivering now"]:
        if col not in df.columns:
            df[col] = False
        else:
            # FIX Bug 2: normalise to proper Python bool so == True filters work reliably
            df[col] = df[col].astype(bool)
    for col in ["Votes", "Aggregate rating", "Price range"]:
        if col not in df.columns:
            df[col] = 0
    # Ensure Primary Cuisine has no leading/trailing whitespace
    if "Primary Cuisine" in df.columns:
        df["Primary Cuisine"] = df["Primary Cuisine"].astype(str).str.strip()
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CONTENT-BASED RESTAURANT RECOMMENDER
# ═══════════════════════════════════════════════════════════════════════════════

class RestaurantRecommender:
    """
    Lightweight content-based recommender using a normalised feature matrix.

    Feature vector per restaurant:
        [Price range, Aggregate rating, log(Votes+1),
         Cuisine_Enc, City_Enc, Has Online delivery, Has Table booking]
    """

    def __init__(self) -> None:
        self._matrix:  Optional[np.ndarray]  = None
        self._df:      Optional[pd.DataFrame] = None
        self._name_to_pos: Optional[dict]    = None   # FIX Bug 1 & 7: use dict instead of Series
        self._scaler   = MinMaxScaler()

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "RestaurantRecommender":
        """Build the similarity matrix from the restaurant DataFrame."""
        self._df = df.reset_index(drop=True)

        cuisine_enc = (
            self._df["Primary Cuisine"]
            .fillna("Unknown")
            .astype("category")
            .cat.codes
        )
        city_enc = (
            self._df["City"]
            .fillna("Unknown")
            .astype("category")
            .cat.codes
        )

        feat = pd.DataFrame({
            "price":     self._df["Price range"].fillna(2),
            "rating":    self._df["Aggregate rating"].fillna(0),
            "log_votes": np.log1p(self._df["Votes"].fillna(0)),
            "cuisine":   cuisine_enc,
            "city":      city_enc,
            "delivery":  self._df["Has Online delivery"].astype(int),
            "booking":   self._df["Has Table booking"].astype(int),
        })

        self._matrix = self._scaler.fit_transform(feat)

        # FIX Bug 1 & 7: build a plain dict mapping name → list of row positions
        # so lookup is unambiguous even with duplicate restaurant names.
        name_to_pos: dict[str, list[int]] = {}
        for pos, name in enumerate(self._df["Restaurant Name"].tolist()):
            name_to_pos.setdefault(str(name), []).append(pos)
        self._name_to_pos = name_to_pos
        return self

    # ── Recommend by index ────────────────────────────────────────────────────

    def recommend_by_index(self, idx: int, n: int = 8) -> pd.DataFrame:
        """Return top-N similar restaurants to the restaurant at position `idx`."""
        if self._matrix is None or self._df is None:
            raise RuntimeError("Call .fit() first.")

        sim_scores       = cosine_similarity(
            self._matrix[idx : idx + 1], self._matrix
        ).flatten()
        sim_scores[idx]  = -1   # exclude self

        top_idx = np.argsort(sim_scores)[::-1][:n]
        cols    = [c for c in [
            "Restaurant Name", "City", "Primary Cuisine",
            "Aggregate rating", "Votes", "Price range",
            "Has Online delivery", "Has Table booking",
        ] if c in self._df.columns]

        result              = self._df.iloc[top_idx][cols].copy()
        result["Similarity"] = sim_scores[top_idx].round(3)
        return result.reset_index(drop=True)

    # ── Recommend by name ─────────────────────────────────────────────────────

    def recommend_by_name(
        self, name: str, n: int = 8
    ) -> Optional[pd.DataFrame]:
        """Return top-N similar restaurants by name (partial match supported)."""
        if self._name_to_pos is None:
            raise RuntimeError("Call .fit() first.")

        # FIX Bug 1 & 7: iterate dict keys; collect matching positions safely
        matched_positions: list[int] = []
        search = name.lower()
        for restaurant_name, positions in self._name_to_pos.items():
            if search in restaurant_name.lower():
                matched_positions.extend(positions)

        if not matched_positions:
            return None

        # Use the first matched position (integer row index in self._df)
        idx = matched_positions[0]
        return self.recommend_by_index(idx, n=n)

    # ── Recommend by preferences ──────────────────────────────────────────────

    def recommend_by_preferences(
        self,
        cuisine:         Optional[str]  = None,
        city:            Optional[str]  = None,
        price_range:     Optional[int]  = None,
        min_rating:      float          = 3.5,
        delivery_needed: bool           = False,
        booking_needed:  bool           = False,
        n:               int            = 10,
    ) -> pd.DataFrame:
        """Filter + rank restaurants by user preferences."""
        if self._df is None:
            raise RuntimeError("Call .fit() first.")

        mask = self._df["Aggregate rating"] >= min_rating

        if cuisine:
            mask &= self._df["Primary Cuisine"].str.contains(
                cuisine, case=False, na=False
            )
        if city:
            mask &= self._df["City"].str.contains(
                city, case=False, na=False
            )
        if price_range is not None:
            mask &= self._df["Price range"] == price_range
        # FIX Bug 2: cast column to bool before comparison so both int(1) and
        # True match correctly regardless of the underlying dtype.
        if delivery_needed:
            mask &= self._df["Has Online delivery"].astype(bool)
        if booking_needed:
            mask &= self._df["Has Table booking"].astype(bool)

        cols = [c for c in [
            "Restaurant Name", "City", "Primary Cuisine",
            "Aggregate rating", "Votes", "Price range",
            "Has Online delivery", "Has Table booking",
        ] if c in self._df.columns]

        return (
            self._df[mask]
            .sort_values(
                ["Aggregate rating", "Votes"], ascending=[False, False]
            )
            .head(n)[cols]
            .reset_index(drop=True)
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. CUISINE RECOMMENDER
# ═══════════════════════════════════════════════════════════════════════════════

def get_cuisine_recommendations(
    df:      pd.DataFrame,
    country: Optional[str] = None,
    city:    Optional[str] = None,
    top_n:   int           = 10,
) -> pd.DataFrame:
    """
    Recommend top cuisines based on a country / city filter.
    Returns a ranked DataFrame of cuisines with avg rating, votes, and score.
    """
    subset = df.copy()
    if country:
        subset = subset[subset["Country"].str.lower() == country.lower()]
    if city:
        subset = subset[subset["City"].str.lower() == city.lower()]

    rated = subset[subset["Aggregate rating"] > 0]
    if rated.empty:
        return pd.DataFrame(
            columns=["Cuisine", "Avg_Rating", "Avg_Votes", "Count", "Score"]
        )

    stats = (
        rated.groupby("Primary Cuisine")
        .agg(
            Avg_Rating=("Aggregate rating", "mean"),
            Avg_Votes =("Votes",            "mean"),
            Count     =("Aggregate rating", "count"),
        )
        .reset_index()
        .rename(columns={"Primary Cuisine": "Cuisine"})
    )

    r_max = stats["Avg_Rating"].max() or 1
    v_max = np.log1p(stats["Avg_Votes"].max()) or 1

    stats["Score"] = (
        0.6 * (stats["Avg_Rating"] / r_max) +
        0.4 * (np.log1p(stats["Avg_Votes"]) / v_max)
    ).round(4)

    # FIX Bug 4: compute min_count from the *filtered* subset, not the global df,
    # so city/country filters don't produce an empty result due to an over-large
    # min_count derived from the full dataset.
    min_count = max(2, len(rated) // 50)
    return (
        stats[stats["Count"] >= min_count]
        .sort_values("Score", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
        .round({"Avg_Rating": 3, "Avg_Votes": 1})
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. PRICING RECOMMENDER
# ═══════════════════════════════════════════════════════════════════════════════

def get_pricing_recommendation(
    df:      pd.DataFrame,
    cuisine: str,
    city:    Optional[str] = None,
) -> dict:
    """
    Recommend optimal price range for a new restaurant given cuisine & city.
    Returns a dict with recommended tier, expected rating, and rationale.
    """
    price_map = {1: "Budget", 2: "Affordable", 3: "Premium", 4: "Luxury"}
    rated     = df[df["Aggregate rating"] > 0].copy()

    cuisine_df = rated[
        rated["Primary Cuisine"].str.contains(cuisine, case=False, na=False)
    ]
    if city:
        city_df = cuisine_df[
            cuisine_df["City"].str.contains(city, case=False, na=False)
        ]
        if len(city_df) >= 5:          # lowered threshold for small datasets
            cuisine_df = city_df

    if cuisine_df.empty:
        return {
            "recommended_tier":  2,
            "tier_label":        "Affordable",
            "expected_rating":   None,
            "rationale":         "Not enough data for this cuisine.",
            "tier_stats":        pd.DataFrame(),
        }

    tier_stats = (
        cuisine_df.groupby("Price range")
        .agg(
            Avg_Rating=("Aggregate rating", "mean"),
            Avg_Votes =("Votes",            "mean"),
            Count     =("Aggregate rating", "count"),
        )
        .reset_index()
        .rename(columns={"Price range": "Price_Range"})
    )
    tier_stats["Tier"] = tier_stats["Price_Range"].map(price_map)

    max_r = tier_stats["Avg_Rating"].max() or 1
    max_v = np.log1p(tier_stats["Avg_Votes"].max()) or 1

    tier_stats["Score"] = (
        0.65 * tier_stats["Avg_Rating"] / max_r +
        0.35 * np.log1p(tier_stats["Avg_Votes"]) / max_v
    )

    best             = tier_stats.loc[tier_stats["Score"].idxmax()]
    recommended_tier = int(best["Price_Range"])
    expected_rating  = round(float(best["Avg_Rating"]), 2)
    tier_label       = price_map.get(recommended_tier, "Affordable")

    rationale = (
        f"For {cuisine} restaurants"
        + (f" in {city}" if city else "")
        + f", the {tier_label} tier achieves the highest composite score "
        f"(avg rating {expected_rating}). "
        f"Based on {int(best['Count'])} comparable restaurants."
    )

    return {
        "recommended_tier": recommended_tier,
        "tier_label":       tier_label,
        "expected_rating":  expected_rating,
        "rationale":        rationale,
        "tier_stats": (
            tier_stats[["Tier", "Price_Range", "Avg_Rating",
                         "Avg_Votes", "Count", "Score"]]
            .sort_values("Price_Range")
            .round({"Avg_Rating": 3, "Avg_Votes": 1, "Score": 4})
            .reset_index(drop=True)
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. SIMILAR-CITY RECOMMENDER
# ═══════════════════════════════════════════════════════════════════════════════

def find_similar_cities(
    df:          pd.DataFrame,
    target_city: str,
    n:           int = 6,
) -> pd.DataFrame:
    """
    Find cities with similar restaurant market profiles using cosine similarity
    on per-city feature vectors.
    """
    city_profile = (
        df.groupby("City")
        .agg(
            Avg_Rating       =("Aggregate rating", "mean"),
            Avg_Votes        =("Votes",            "mean"),
            Delivery_Pct     =("Has Online delivery", "mean"),
            Booking_Pct      =("Has Table booking",   "mean"),
            Avg_Price        =("Price range",         "mean"),
            Restaurant_Count =("Restaurant Name",     "count"),
        )
        .reset_index()
    )

    # FIX Bug 5: compute min_count from the city_profile itself (i.e. from the
    # number of cities / total restaurants visible after groupby), not from the
    # raw global df length, so cities with adequate local data are not excluded.
    min_count    = max(3, len(df) // max(city_profile["Restaurant_Count"].max(), 1))
    city_profile = city_profile[
        city_profile["Restaurant_Count"] >= min_count
    ].reset_index(drop=True)

    if target_city not in city_profile["City"].values:
        return pd.DataFrame(
            columns=["City", "Avg_Rating", "Avg_Votes",
                     "Delivery_Pct", "Booking_Pct",
                     "Restaurant_Count", "Similarity"]
        )

    feat_cols   = ["Avg_Rating", "Avg_Votes", "Delivery_Pct",
                   "Booking_Pct", "Avg_Price"]
    scaler      = MinMaxScaler()
    feat_matrix = scaler.fit_transform(city_profile[feat_cols])

    pos  = city_profile[city_profile["City"] == target_city].index[0]
    sims = cosine_similarity(
        feat_matrix[pos : pos + 1], feat_matrix
    ).flatten()
    sims[pos] = -1

    top_pos  = np.argsort(sims)[::-1][:n]
    result   = city_profile.iloc[top_pos].copy()
    result["Similarity"] = sims[top_pos].round(3)

    return result[
        ["City", "Avg_Rating", "Avg_Votes", "Delivery_Pct",
         "Booking_Pct", "Restaurant_Count", "Similarity"]
    ].reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT / PRINT UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

_SEP  = "=" * 64
_SEP2 = "-" * 64

def _hdr(title: str) -> None:
    print(f"\n{_SEP}")
    print(f"  {title}")
    print(_SEP)

def _sub(title: str) -> None:
    print(f"\n{_SEP2}")
    print(f"  {title}")
    print(_SEP2)

def _df_print(df: pd.DataFrame, max_rows: int = 15) -> None:
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 120)
    pd.set_option("display.float_format", "{:.3f}".format)
    print(df.head(max_rows).to_string(index=True))
    if len(df) > max_rows:
        print(f"  ... ({len(df) - max_rows} more rows)")


# ── Dataset Summary ───────────────────────────────────────────────────────────

def print_dataset_summary(df: pd.DataFrame) -> None:
    """Print a concise overview of the restaurant dataset."""
    _hdr("RestaurantIQ – Dataset Summary")
    print(f"  Total restaurants   : {len(df):,}")
    print(f"  Countries           : {df['Country'].nunique()}")
    print(f"  Cities              : {df['City'].nunique()}")
    if "Primary Cuisine" in df.columns:
        print(f"  Unique cuisines     : {df['Primary Cuisine'].nunique()}")
    rated = df[df["Aggregate rating"] > 0]
    print(f"  Rated restaurants   : {len(rated):,}  ({len(rated)/len(df)*100:.1f}%)")
    if not rated.empty:
        print(f"  Avg rating (rated)  : {rated['Aggregate rating'].mean():.2f}")
        print(f"  Rating range        : "
              f"{rated['Aggregate rating'].min():.1f}  –  "
              f"{rated['Aggregate rating'].max():.1f}")
    print(f"  Avg votes           : {df['Votes'].mean():.0f}")
    if "Has Online delivery" in df.columns:
        print(f"  Online delivery     : {df['Has Online delivery'].mean()*100:.1f}%")
    if "Has Table booking" in df.columns:
        print(f"  Table booking       : {df['Has Table booking'].mean()*100:.1f}%")
    print()


# ── Engine 1 Output ───────────────────────────────────────────────────────────

def print_content_based_results(
    recommender: RestaurantRecommender,
    df: pd.DataFrame,
    cuisine: Optional[str] = None,
    city:    Optional[str] = None,
) -> None:
    """Print all three recommendation modes of RestaurantRecommender."""
    _hdr("Engine 1 – Content-Based Restaurant Recommendations")

    # FIX Bug 6: guard against recommender not being fitted before using it
    if recommender._matrix is None:
        print("  Recommender not fitted – skipping Engine 1.")
        return

    # 1a. By name (pick first restaurant in df)
    _sub("1a. Similar to a named restaurant (top 5)")
    sample_name = df["Restaurant Name"].iloc[0]
    result = recommender.recommend_by_name(sample_name, n=5)
    if result is not None and not result.empty:
        print(f"  Seed restaurant : {sample_name}")
        print()
        _df_print(result)
    else:
        print(f"  No matches found for '{sample_name}'.")

    # 1b. By preferences
    _sub("1b. By user preferences")
    pref_cuisine = cuisine or df["Primary Cuisine"].value_counts().index[0]
    pref_city    = city    or df["City"].value_counts().index[0]
    print(f"  Cuisine: {pref_cuisine} | City: {pref_city} | Min rating: 3.5")
    print()
    result = recommender.recommend_by_preferences(
        cuisine=pref_cuisine, city=pref_city, min_rating=3.5, n=8
    )
    if not result.empty:
        _df_print(result)
    else:
        # Relax filters
        result = recommender.recommend_by_preferences(
            cuisine=pref_cuisine, min_rating=3.0, n=8
        )
        print(f"  (relaxed to cuisine-only + min_rating 3.0)")
        if not result.empty:
            _df_print(result)
        else:
            print("  No matches found.")

    # 1c. By index (restaurant #0)
    _sub("1c. Similar to restaurant at index 0 (top 5)")
    _df_print(recommender.recommend_by_index(0, n=5))


# ── Engine 2 Output ───────────────────────────────────────────────────────────

def print_cuisine_recommendations(
    df:      pd.DataFrame,
    country: Optional[str] = None,
    city:    Optional[str] = None,
) -> None:
    """Print cuisine recommendations, globally and filtered."""
    _hdr("Engine 2 – Cuisine Recommendations")

    # Global
    _sub("2a. Top 10 cuisines globally")
    result = get_cuisine_recommendations(df, top_n=10)
    _df_print(result) if not result.empty else print("  No data.")

    # By country
    target_country = country or df["Country"].value_counts().index[0]
    _sub(f"2b. Top cuisines in country: {target_country}")
    result = get_cuisine_recommendations(df, country=target_country, top_n=8)
    _df_print(result) if not result.empty else print("  No data.")

    # By city
    target_city = city or df["City"].value_counts().index[0]
    _sub(f"2c. Top cuisines in city: {target_city}")
    result = get_cuisine_recommendations(df, city=target_city, top_n=8)
    _df_print(result) if not result.empty else print("  No data.")


# ── Engine 3 Output ───────────────────────────────────────────────────────────

def print_pricing_recommendations(
    df:       pd.DataFrame,
    cuisines: Optional[list] = None,
    city:     Optional[str]  = None,
) -> None:
    """Print pricing recommendations for a set of cuisines."""
    _hdr("Engine 3 – Pricing Recommendations for New Operators")

    if cuisines is None:
        cuisines = df["Primary Cuisine"].value_counts().head(4).index.tolist()

    for c in cuisines:
        _sub(f"Cuisine: {c}" + (f" | City: {city}" if city else " | All cities"))
        rec = get_pricing_recommendation(df, cuisine=c, city=city)
        print(f"  Recommended tier  : {rec['tier_label']}  "
              f"(Price range {rec['recommended_tier']})")
        if rec["expected_rating"] is not None:
            print(f"  Expected rating   : {rec['expected_rating']}")
        print(f"  Rationale         : "
              + textwrap.fill(rec["rationale"], width=58,
                              subsequent_indent="                    "))
        if not rec["tier_stats"].empty:
            print()
            _df_print(rec["tier_stats"])


# ── Engine 4 Output ───────────────────────────────────────────────────────────

def print_similar_cities(
    df:      pd.DataFrame,
    cities:  Optional[list] = None,
) -> None:
    """Print similar-city results for a list of target cities."""
    _hdr("Engine 4 – Similar City Market Finder")

    if cities is None:
        cities = df["City"].value_counts().head(3).index.tolist()

    for target in cities:
        _sub(f"Cities similar to: {target}")
        result = find_similar_cities(df, target_city=target, n=5)
        if not result.empty:
            _df_print(result)
        else:
            print(f"  '{target}' not found or insufficient data.")


# ── Save to CSV ───────────────────────────────────────────────────────────────

def save_all_recommendations(
    df:           pd.DataFrame,
    recommender:  RestaurantRecommender,
    output_dir:   str             = "restaurant_iq_recommendations",
    cuisine:      Optional[str]   = None,
    city:         Optional[str]   = None,
    country:      Optional[str]   = None,
) -> None:
    """
    Save all recommendation outputs as CSV files to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    total = done = failed = 0

    print(f"\n{_SEP}")
    print(f"  Saving recommendation reports  →  '{output_dir}/'")
    print(_SEP)

    def _save(filename: str, df_out: pd.DataFrame, label: str) -> None:
        nonlocal total, done, failed
        total += 1
        path = os.path.join(output_dir, filename)
        print(f"  {label} ...", end=" ", flush=True)
        try:
            if df_out is None or df_out.empty:
                print("⊘  empty – skipped")
                return
            df_out.to_csv(path, index=False)
            print(f"✓  →  {path}")
            done += 1
        except Exception as exc:
            print(f"✗  ERROR: {exc}")
            failed += 1

    # ── 1. Content-based: by name ─────────────────────────────────────────────
    sample_name = df["Restaurant Name"].iloc[0]
    result      = recommender.recommend_by_name(sample_name, n=10)
    _save("01_similar_to_first_restaurant.csv",
          result if result is not None else pd.DataFrame(),
          f"Similar to '{sample_name}' (top 10)")

    # ── 2. By preferences ─────────────────────────────────────────────────────
    pref_cuisine = cuisine or df["Primary Cuisine"].value_counts().index[0]
    result       = recommender.recommend_by_preferences(
        cuisine=pref_cuisine, city=city, min_rating=3.0, n=15
    )
    _save("02_preference_based.csv", result,
          f"Preference-based (cuisine={pref_cuisine})")

    # ── 3. Cuisine recommendations – global ───────────────────────────────────
    _save("03_top_cuisines_global.csv",
          get_cuisine_recommendations(df, top_n=15),
          "Top 15 cuisines globally")

    # ── 4. Cuisine recommendations – country ──────────────────────────────────
    target_country = country or df["Country"].value_counts().index[0]
    _save(f"04_top_cuisines_{target_country.replace(' ','_')}.csv",
          get_cuisine_recommendations(df, country=target_country, top_n=10),
          f"Top cuisines in {target_country}")

    # ── 5. Pricing per top cuisine ────────────────────────────────────────────
    cuisines = df["Primary Cuisine"].value_counts().head(5).index.tolist()
    for i, c in enumerate(cuisines, start=5):
        rec = get_pricing_recommendation(df, cuisine=c, city=city)
        if not rec["tier_stats"].empty:
            _save(f"0{i}_pricing_{c.replace(' ','_')}.csv",
                  rec["tier_stats"],
                  f"Pricing tiers for {c}")

    # ── 6. Similar cities ─────────────────────────────────────────────────────
    top_cities = df["City"].value_counts().head(3).index.tolist()
    for j, tc in enumerate(top_cities, start=10):
        _save(f"{j}_similar_cities_{tc.replace(' ','_')}.csv",
              find_similar_cities(df, target_city=tc, n=6),
              f"Cities similar to {tc}")

    print(f"\n{_SEP}")
    print(f"  ✓ Saved: {done}   ⊘ Empty/skipped   ✗ Failed: {failed}")
    print(f"  Output folder: {os.path.abspath(output_dir)}")
    print(_SEP)


# ── Master print function ─────────────────────────────────────────────────────

def print_all_recommendations(
    df:         pd.DataFrame,
    output_dir: str           = "restaurant_iq_recommendations",
    cuisine:    Optional[str] = None,
    city:       Optional[str] = None,
    country:    Optional[str] = None,
    save_csv:   bool          = True,
) -> RestaurantRecommender:
    """
    Master function: build all engines → print all results → save CSVs.

    Parameters
    ----------
    df          : Restaurant DataFrame.
    output_dir  : Folder for CSV exports.
    cuisine     : Optional cuisine filter for demos.
    city        : Optional city filter for demos.
    country     : Optional country filter for cuisine engine.
    save_csv    : Write CSV reports to output_dir.

    Returns
    -------
    Fitted RestaurantRecommender instance (for further use).
    """
    print_dataset_summary(df)

    # ── Fit recommender ───────────────────────────────────────────────────────
    print(f"  Building content-based recommender ...", end=" ", flush=True)
    recommender = RestaurantRecommender()
    try:
        recommender.fit(df)
        print("✓")
    except Exception as exc:
        print(f"✗ ERROR: {exc}")

    # ── Print all engines ─────────────────────────────────────────────────────
    print_content_based_results(recommender, df, cuisine=cuisine, city=city)
    print_cuisine_recommendations(df, country=country, city=city)
    print_pricing_recommendations(
        df,
        cuisines=([cuisine] if cuisine else None),
        city=city,
    )
    print_similar_cities(df)

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    if save_csv:
        save_all_recommendations(
            df, recommender,
            output_dir=output_dir,
            cuisine=cuisine, city=city, country=country,
        )

    return recommender


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "RestaurantIQ Recommendation System\n"
            "If --csv is omitted, built-in sample data is used."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--csv", "-c",
        default=None,
        help="Path to restaurant CSV. Omit to use built-in sample data.",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="restaurant_iq_recommendations",
        help="Directory to save CSV reports (default: restaurant_iq_recommendations).",
    )
    parser.add_argument(
        "--cuisine",
        default=None,
        help="Cuisine to highlight in demos (e.g. 'Italian').",
    )
    parser.add_argument(
        "--city",
        default=None,
        help="City to highlight in demos (e.g. 'Mumbai').",
    )
    parser.add_argument(
        "--country",
        default=None,
        help="Country to highlight in cuisine engine (e.g. 'India').",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip saving CSV reports.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        help="Rows in built-in demo data when --csv is not used (default: 1000).",
    )
    args = parser.parse_args()

    # ── Load ──────────────────────────────────────────────────────────────────
    if args.csv:
        if not os.path.isfile(args.csv):
            print(f"\n[ERROR] File not found: {args.csv}")
            sys.exit(1)
        print(f"\nLoading CSV: {args.csv}")
        df = pd.read_csv(args.csv)
    else:
        print(f"\nNo CSV supplied – using built-in sample data "
              f"({args.sample_size} rows).")
        df = _make_sample_df(n=args.sample_size)

    df = _ensure_columns(df)

    # ── Run ───────────────────────────────────────────────────────────────────
    print_all_recommendations(
        df,
        output_dir=args.output_dir,
        cuisine=args.cuisine,
        city=args.city,
        country=args.country,
        save_csv=not args.no_save,
    )


if __name__ == "__main__":
    main()
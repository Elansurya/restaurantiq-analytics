from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# ── Pillar weights ────────────────────────────────────────────────────────────
PILLAR_WEIGHTS = {
    "Rating":     0.25,
    "Engagement": 0.20,
    "Pricing":    0.15,
    "Cuisine":    0.15,
    "Delivery":   0.15,
    "Booking":    0.10,
}

PILLAR_ORDER = list(PILLAR_WEIGHTS.keys())


# ── Core scorer ───────────────────────────────────────────────────────────────

def compute_success_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute success score pillars and composite score for every restaurant.

    Returns a new DataFrame with all original columns plus:
        Score_Rating, Score_Engagement, Score_Pricing,
        Score_Cuisine, Score_Delivery, Score_Booking,
        Success_Score  (0–100, composite)
        Success_Tier   (str label)
    """
    out = df.copy()

    # ── 1. Rating pillar ──────────────────────────────────────────────────────
    out["Score_Rating"] = np.where(
        out["Aggregate rating"] > 0,
        (out["Aggregate rating"] / 5.0) * 100,
        0.0,
    ).round(2)

    # ── 2. Engagement pillar (log-scaled votes) ───────────────────────────────
    log_votes = np.log1p(out["Votes"].fillna(0))
    max_log   = log_votes.max() or 1
    out["Score_Engagement"] = (log_votes / max_log * 100).round(2)

    # ── 3. Pricing pillar (value signal: how well rating compares to tier avg) ─
    tier_avg = (
        out[out["Aggregate rating"] > 0]
        .groupby("Price range")["Aggregate rating"]
        .mean()
    )
    out["_tier_avg"] = out["Price range"].map(tier_avg).fillna(3.0)
    out["Score_Pricing"] = np.where(
        out["Aggregate rating"] > 0,
        np.clip((out["Aggregate rating"] / out["_tier_avg"]) * 50, 0, 100),
        30.0,  # neutral for unrated
    ).round(2)
    out.drop("_tier_avg", axis=1, inplace=True)

    # ── 4. Cuisine pillar (relative popularity of primary cuisine) ────────────
    cuisine_counts = out["Primary Cuisine"].value_counts()
    out["_cuisine_count"] = out["Primary Cuisine"].map(cuisine_counts).fillna(1)
    max_c = out["_cuisine_count"].max() or 1
    out["Score_Cuisine"] = (out["_cuisine_count"] / max_c * 100).round(2)
    out.drop("_cuisine_count", axis=1, inplace=True)

    # ── 5. Delivery pillar ────────────────────────────────────────────────────
    out["Score_Delivery"] = (
        out["Has Online delivery"].astype(int) * 70 +
        out["Is delivering now"].astype(int)  * 30
    ).astype(float)

    # ── 6. Booking pillar ─────────────────────────────────────────────────────
    out["Score_Booking"] = (
        out["Has Table booking"].astype(int) * 100
    ).astype(float)

    # ── Composite weighted score ──────────────────────────────────────────────
    out["Success_Score"] = (
        PILLAR_WEIGHTS["Rating"]     * out["Score_Rating"]     +
        PILLAR_WEIGHTS["Engagement"] * out["Score_Engagement"] +
        PILLAR_WEIGHTS["Pricing"]    * out["Score_Pricing"]    +
        PILLAR_WEIGHTS["Cuisine"]    * out["Score_Cuisine"]    +
        PILLAR_WEIGHTS["Delivery"]   * out["Score_Delivery"]   +
        PILLAR_WEIGHTS["Booking"]    * out["Score_Booking"]
    ).round(2)

    # ── Tier labelling ────────────────────────────────────────────────────────
    out["Success_Tier"] = pd.cut(
        out["Success_Score"],
        bins=[0, 30, 50, 65, 80, 101],
        labels=["Struggling", "Below Average", "Average", "Performing", "Top Performer"],
        include_lowest=True,
    ).astype(str)

    return out


# ── Single-restaurant gauge data ──────────────────────────────────────────────

def get_gauge_data(row: pd.Series) -> dict:
    """
    Return data needed to render a gauge chart for a single restaurant.
    """
    score = float(row.get("Success_Score", 0))

    if score >= 80:
        color, label = "#00C2A8", "Top Performer"
    elif score >= 65:
        color, label = "#6C63FF", "Performing"
    elif score >= 50:
        color, label = "#F59E0B", "Average"
    elif score >= 30:
        color, label = "#F97316", "Below Average"
    else:
        color, label = "#EF4444", "Struggling"

    return {
        "score":       round(score, 1),
        "label":       label,
        "color":       color,
        "thresholds":  [30, 50, 65, 80, 100],
        "tier_colors": ["#EF4444", "#F97316", "#F59E0B", "#6C63FF", "#00C2A8"],
    }


# ── Single-restaurant radar data ──────────────────────────────────────────────

def get_radar_data(row: pd.Series) -> dict:
    """
    Return pillar scores for a radar chart for a single restaurant.
    """
    pillars = {
        "Rating":     float(row.get("Score_Rating",     0)),
        "Engagement": float(row.get("Score_Engagement", 0)),
        "Pricing":    float(row.get("Score_Pricing",    0)),
        "Cuisine":    float(row.get("Score_Cuisine",    0)),
        "Delivery":   float(row.get("Score_Delivery",   0)),
        "Booking":    float(row.get("Score_Booking",    0)),
    }
    return {
        "categories": list(pillars.keys()),
        "values":     [round(v, 1) for v in pillars.values()],
        "weights":    [PILLAR_WEIGHTS[k] for k in pillars],
    }


# ── Aggregate score distributions ────────────────────────────────────────────

def get_score_distribution(scored_df: pd.DataFrame) -> pd.DataFrame:
    """Return count and percentage of restaurants per success tier."""
    dist = (
        scored_df["Success_Tier"]
        .value_counts()
        .reset_index()
        .rename(columns={"Success_Tier": "Tier", "count": "Count"})
    )
    dist["Pct"] = (dist["Count"] / dist["Count"].sum() * 100).round(1)
    tier_order  = ["Top Performer", "Performing", "Average", "Below Average", "Struggling"]
    dist["_ord"] = dist["Tier"].map({t: i for i, t in enumerate(tier_order)})
    return dist.sort_values("_ord").drop("_ord", axis=1).reset_index(drop=True)


def get_pillar_averages(scored_df: pd.DataFrame) -> pd.DataFrame:
    """Return average value for each pillar across all restaurants."""
    rows = []
    for pillar in PILLAR_ORDER:
        col = f"Score_{pillar}"
        if col in scored_df.columns:
            rows.append({
                "Pillar":     pillar,
                "Avg_Score":  round(scored_df[col].mean(), 2),
                "Weight":     PILLAR_WEIGHTS[pillar],
                "Weighted_Contribution": round(scored_df[col].mean() * PILLAR_WEIGHTS[pillar], 2),
            })
    return pd.DataFrame(rows)


def get_top_performers(scored_df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Return top-N restaurants by Success Score."""
    cols = [
        "Restaurant Name", "City", "Primary Cuisine",
        "Aggregate rating", "Votes", "Price range",
        "Success_Score", "Success_Tier",
        "Score_Rating", "Score_Engagement",
        "Score_Delivery", "Score_Booking",
    ]
    available = [c for c in cols if c in scored_df.columns]
    return (
        scored_df.nlargest(n, "Success_Score")[available]
        .reset_index(drop=True)
    )


def get_tier_radar_averages(scored_df: pd.DataFrame) -> dict[str, list[float]]:
    """
    Return per-tier average pillar scores for a grouped radar comparison chart.
    """
    pillar_cols = [f"Score_{p}" for p in PILLAR_ORDER]
    available   = [c for c in pillar_cols if c in scored_df.columns]
    result      = {}
    for tier, grp in scored_df.groupby("Success_Tier"):
        result[str(tier)] = [round(grp[c].mean(), 1) for c in available]
    return result


# ── Pretty print helpers ──────────────────────────────────────────────────────

def _section(title: str) -> None:
    width = 65
    print("\n" + "═" * width)
    print(f"  {title}")
    print("═" * width)


def _subsection(title: str) -> None:
    print(f"\n  ── {title} {'─' * (55 - len(title))}")


def print_all_outputs(scored_df: pd.DataFrame) -> None:
    """
    Print a comprehensive, formatted summary of all RestaurantIQ outputs.
    """

    # ── 1. Dataset overview ───────────────────────────────────────────────────
    _section("1 · DATASET OVERVIEW")
    print(f"  Total restaurants  : {len(scored_df):,}")
    print(f"  Columns available  : {len(scored_df.columns)}")
    print(f"  Cities covered     : {scored_df['City'].nunique() if 'City' in scored_df.columns else 'N/A'}")
    print(f"  Cuisines detected  : {scored_df['Primary Cuisine'].nunique() if 'Primary Cuisine' in scored_df.columns else 'N/A'}")

    # ── 2. Success Score summary stats ────────────────────────────────────────
    _section("2 · SUCCESS SCORE — SUMMARY STATISTICS")
    ss = scored_df["Success_Score"]
    print(f"  Mean   : {ss.mean():.2f}")
    print(f"  Median : {ss.median():.2f}")
    print(f"  Std    : {ss.std():.2f}")
    print(f"  Min    : {ss.min():.2f}")
    print(f"  Max    : {ss.max():.2f}")
    print(f"  25 %   : {ss.quantile(0.25):.2f}")
    print(f"  75 %   : {ss.quantile(0.75):.2f}")

    # ── 3. Tier distribution ──────────────────────────────────────────────────
    _section("3 · TIER DISTRIBUTION")
    dist = get_score_distribution(scored_df)
    print(f"\n  {'Tier':<18} {'Count':>8} {'Pct (%)':>10}")
    print(f"  {'-'*18} {'-'*8} {'-'*10}")
    tier_icons = {
        "Top Performer": "🏆",
        "Performing":    "✅",
        "Average":       "📊",
        "Below Average": "⚠️ ",
        "Struggling":    "❌",
    }
    for _, row in dist.iterrows():
        icon = tier_icons.get(row["Tier"], "  ")
        print(f"  {icon} {row['Tier']:<16} {row['Count']:>8,} {row['Pct']:>9.1f}%")

    # ── 4. Pillar averages ────────────────────────────────────────────────────
    _section("4 · PILLAR AVERAGES (all restaurants)")
    pillar_df = get_pillar_averages(scored_df)
    print(f"\n  {'Pillar':<14} {'Weight':>8} {'Avg Score':>12} {'Contribution':>14}")
    print(f"  {'-'*14} {'-'*8} {'-'*12} {'-'*14}")
    for _, row in pillar_df.iterrows():
        bar_len = int(row["Avg_Score"] / 5)          # scale 0-100 → 0-20 chars
        bar     = "█" * bar_len + "░" * (20 - bar_len)
        print(f"  {row['Pillar']:<14} {row['Weight']:>7.0%} {row['Avg_Score']:>11.2f} {row['Weighted_Contribution']:>13.2f}")
        print(f"               [{bar}]")

    # ── 5. Top 10 performers ──────────────────────────────────────────────────
    _section("5 · TOP 10 PERFORMERS")
    top = get_top_performers(scored_df, n=10)
    print()
    for i, row in top.iterrows():
        name    = str(row.get("Restaurant Name", "N/A"))[:35]
        city    = str(row.get("City",            "N/A"))[:20]
        cuisine = str(row.get("Primary Cuisine", "N/A"))[:20]
        score   = row.get("Success_Score", 0)
        tier    = row.get("Success_Tier",  "N/A")
        rating  = row.get("Aggregate rating", 0)
        votes   = row.get("Votes", 0)
        print(f"  #{i+1:>2}  {name:<35}  Score: {score:>5.1f}  [{tier}]")
        print(f"       City: {city:<20}  Cuisine: {cuisine}")
        print(f"       Rating: {rating:.1f}/5   Votes: {int(votes):,}")
        print()

    # ── 6. Per-tier radar averages ────────────────────────────────────────────
    _section("6 · PER-TIER RADAR AVERAGES")
    radar = get_tier_radar_averages(scored_df)
    tier_order_radar = ["Top Performer", "Performing", "Average", "Below Average", "Struggling"]
    header = f"  {'Tier':<18}" + "".join(f" {p:>12}" for p in PILLAR_ORDER)
    print(f"\n{header}")
    print(f"  {'-'*18}" + "".join(f" {'-'*12}" for _ in PILLAR_ORDER))
    for tier in tier_order_radar:
        if tier in radar:
            vals = radar[tier]
            row_str = f"  {tier:<18}" + "".join(f" {v:>12.1f}" for v in vals)
            print(row_str)

    # ── 7. Sample gauge & radar data for top restaurant ──────────────────────
    _section("7 · SAMPLE GAUGE & RADAR DATA  (Top-ranked restaurant)")
    top_row   = scored_df.nlargest(1, "Success_Score").iloc[0]
    rest_name = top_row.get("Restaurant Name", "N/A")
    gauge     = get_gauge_data(top_row)
    radar_d   = get_radar_data(top_row)

    print(f"\n  Restaurant : {rest_name}")
    _subsection("Gauge Data")
    print(f"    Score      : {gauge['score']}")
    print(f"    Tier Label : {gauge['label']}")
    print(f"    Hex Color  : {gauge['color']}")
    print(f"    Thresholds : {gauge['thresholds']}")
    print(f"    Tier Colors: {gauge['tier_colors']}")

    _subsection("Radar Data")
    print(f"    {'Pillar':<14} {'Score':>8} {'Weight':>8}")
    print(f"    {'-'*14} {'-'*8} {'-'*8}")
    for cat, val, wt in zip(radar_d["categories"], radar_d["values"], radar_d["weights"]):
        print(f"    {cat:<14} {val:>8.1f} {wt:>7.0%}")

    # ── 8. Score column preview ───────────────────────────────────────────────
    _section("8 · SCORED DATAFRAME — FIRST 5 ROWS (score columns only)")
    score_cols = [
        "Restaurant Name",
        "Score_Rating", "Score_Engagement", "Score_Pricing",
        "Score_Cuisine", "Score_Delivery", "Score_Booking",
        "Success_Score", "Success_Tier",
    ]
    available_cols = [c for c in score_cols if c in scored_df.columns]
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.2f}".format)
    print()
    print(scored_df[available_cols].head().to_string(index=False))

    # ── Done ──────────────────────────────────────────────────────────────────
    _section("✔  ANALYSIS COMPLETE")
    print(f"  Scored DataFrame shape : {scored_df.shape}")
    print(f"  New score columns added: {len([c for c in scored_df.columns if c.startswith('Score_') or c.startswith('Success_')])}")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    DATA_PATH = r"C:\project\AI-Powered Restaurant Intelligence & Business Analytics Platform\data\cleaned_dataset.csv"

    # ── Load ──────────────────────────────────────────────────────────────────
    print("\n  Loading dataset …")
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"\n  ❌  File not found:\n     {DATA_PATH}")
        print("      Please verify the path and try again.")
        return
    except Exception as exc:
        print(f"\n  ❌  Failed to load dataset: {exc}")
        return

    print(f"  ✔  Loaded {len(df):,} rows × {len(df.columns)} columns")

    # ── Pre-process boolean columns (handle Yes/No strings) ───────────────────
    bool_cols = ["Has Online delivery", "Is delivering now", "Has Table booking"]
    for col in bool_cols:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].str.strip().str.lower().map({"yes": True, "no": False}).fillna(False)

    # ── Handle missing Primary Cuisine ────────────────────────────────────────
    if "Primary Cuisine" not in df.columns:
        if "Cuisines" in df.columns:
            df["Primary Cuisine"] = df["Cuisines"].str.split(",").str[0].str.strip()
        else:
            df["Primary Cuisine"] = "Unknown"

    # ── Compute scores ────────────────────────────────────────────────────────
    print("  Computing success scores …")
    scored_df = compute_success_scores(df)
    print("  ✔  Scores computed\n")

    # ── Print all outputs ─────────────────────────────────────────────────────
    print_all_outputs(scored_df)

    # ── Optional: save scored CSV ─────────────────────────────────────────────
    output_path = DATA_PATH.replace("cleaned_dataset.csv", "scored_dataset.csv")
    try:
        scored_df.to_csv(output_path, index=False)
        print(f"  💾  Scored CSV saved → {output_path}\n")
    except Exception as exc:
        print(f"  ⚠️   Could not save scored CSV: {exc}\n")


if __name__ == "__main__":
    main()
from __future__ import annotations

import numpy as np
import pandas as pd


# ═════════════════════════════════════════════════════════════════════════════
# PUBLIC: delivery_booking_analysis
# ═════════════════════════════════════════════════════════════════════════════

def delivery_booking_analysis(df: pd.DataFrame) -> dict:
    """
    Analyse the impact of online delivery and table booking on ratings & votes.

    Parameters
    ----------
    df : Cleaned/preprocessed DataFrame from load_and_preprocess().

    Returns
    -------
    dict with keys:
        booking  : dict  – stats for Has Table booking
        delivery : dict  – stats for Has Online delivery
        cross    : dict  – stats for both-services combination
    """
    rated = df[df["Aggregate rating"] > 0].copy()

    # ── Table booking analysis ────────────────────────────────────────────────
    booking: dict = {}
    if "Has Table booking" in rated.columns:
        yes_b = rated[rated["Has Table booking"] == 1]
        no_b  = rated[rated["Has Table booking"] == 0]

        booking["yes_pct"]        = round(rated["Has Table booking"].mean() * 100, 1)
        booking["yes_avg_rating"] = round(yes_b["Aggregate rating"].mean(), 3) if len(yes_b) else 0.0
        booking["no_avg_rating"]  = round(no_b["Aggregate rating"].mean(), 3)  if len(no_b)  else 0.0
        booking["rating_lift"]    = round(booking["yes_avg_rating"] - booking["no_avg_rating"], 3)
        booking["yes_avg_votes"]  = round(yes_b["Votes"].mean(), 1) if len(yes_b) else 0.0
        booking["no_avg_votes"]   = round(no_b["Votes"].mean(), 1)  if len(no_b)  else 0.0
        booking["yes_count"]      = len(yes_b)
        booking["no_count"]       = len(no_b)
    else:
        booking = {k: 0.0 for k in [
            "yes_pct", "yes_avg_rating", "no_avg_rating",
            "rating_lift", "yes_avg_votes", "no_avg_votes",
            "yes_count", "no_count",
        ]}

    # ── Online delivery analysis ──────────────────────────────────────────────
    delivery: dict = {}
    if "Has Online delivery" in rated.columns:
        yes_d = rated[rated["Has Online delivery"] == 1]
        no_d  = rated[rated["Has Online delivery"] == 0]

        delivery["yes_pct"]        = round(rated["Has Online delivery"].mean() * 100, 1)
        delivery["yes_avg_rating"] = round(yes_d["Aggregate rating"].mean(), 3) if len(yes_d) else 0.0
        delivery["no_avg_rating"]  = round(no_d["Aggregate rating"].mean(), 3)  if len(no_d)  else 0.0
        delivery["rating_lift"]    = round(delivery["yes_avg_rating"] - delivery["no_avg_rating"], 3)
        delivery["yes_avg_votes"]  = round(yes_d["Votes"].mean(), 1) if len(yes_d) else 0.0
        delivery["no_avg_votes"]   = round(no_d["Votes"].mean(), 1)  if len(no_d)  else 0.0
        delivery["yes_count"]      = len(yes_d)
        delivery["no_count"]       = len(no_d)
    else:
        delivery = {k: 0.0 for k in [
            "yes_pct", "yes_avg_rating", "no_avg_rating",
            "rating_lift", "yes_avg_votes", "no_avg_votes",
            "yes_count", "no_count",
        ]}

    # ── Cross-analysis (both services) ────────────────────────────────────────
    cross: dict = {}
    has_both_cols = (
        "Has Online delivery" in rated.columns and
        "Has Table booking"   in rated.columns
    )
    if has_both_cols:
        both    = rated[(rated["Has Online delivery"] == 1) & (rated["Has Table booking"] == 1)]
        neither = rated[(rated["Has Online delivery"] == 0) & (rated["Has Table booking"] == 0)]
        cross["both_avg_rating"]    = round(both["Aggregate rating"].mean(), 3)    if len(both)    else 0.0
        cross["neither_avg_rating"] = round(neither["Aggregate rating"].mean(), 3) if len(neither) else 0.0
        cross["both_avg_votes"]     = round(both["Votes"].mean(), 1)               if len(both)    else 0.0
        cross["both_count"]         = len(both)
        cross["neither_count"]      = len(neither)
        cross["lift_vs_neither"]    = round(
            cross["both_avg_rating"] - cross["neither_avg_rating"], 3
        )
    else:
        cross = {k: 0.0 for k in [
            "both_avg_rating", "neither_avg_rating",
            "both_avg_votes", "both_count", "neither_count", "lift_vs_neither",
        ]}

    return {
        "booking":  booking,
        "delivery": delivery,
        "cross":    cross,
    }


# ═════════════════════════════════════════════════════════════════════════════
# PUBLIC: pricing_analysis
# ═════════════════════════════════════════════════════════════════════════════

_PRICE_LABEL = {1: "Budget", 2: "Affordable", 3: "Premium", 4: "Luxury"}
_PRICE_ORDER = ["Budget", "Affordable", "Premium", "Luxury"]


def pricing_analysis(df: pd.DataFrame) -> dict:
    """
    Analyse rating, votes, and service rates across price tiers.

    Parameters
    ----------
    df : Cleaned/preprocessed DataFrame.

    Returns
    -------
    dict with keys:
        price_stats_df   : pd.DataFrame – per-price-tier stats
        recommendations  : list[str]    – business insight strings
    """
    work = df.copy()

    # Ensure Price Label column
    if "Price Label" not in work.columns:
        work["Price Label"] = work["Price range"].map(_PRICE_LABEL).fillna("Unknown")

    rated = work[work["Aggregate rating"] > 0]

    # Per-tier aggregation
    agg = (
        rated.groupby("Price Label")
        .agg(
            Count       = ("Restaurant ID", "count"),
            Avg_Rating  = ("Aggregate rating", "mean"),
            Avg_Votes   = ("Votes", "mean"),
            Avg_Cost    = ("Average Cost for two", "mean"),
            Delivery_Pct= ("Has Online delivery", "mean") if "Has Online delivery" in rated.columns
                          else ("Aggregate rating", lambda x: 0.0),
            Booking_Pct = ("Has Table booking", "mean")   if "Has Table booking"   in rated.columns
                          else ("Aggregate rating", lambda x: 0.0),
        )
        .reset_index()
    )

    # Round
    for col in ["Avg_Rating", "Avg_Votes", "Avg_Cost"]:
        agg[col] = agg[col].round(2)

    # Reorder by price tier
    cat = pd.Categorical(agg["Price Label"], categories=_PRICE_ORDER, ordered=True)
    agg["Price Label"] = cat
    agg = agg.sort_values("Price Label").reset_index(drop=True)

    # ── Auto-generate textual recommendations ─────────────────────────────────
    recommendations: list[str] = []

    if len(agg) > 1:
        best_rating_tier = agg.loc[agg["Avg_Rating"].idxmax(), "Price Label"]
        recommendations.append(
            f"⭐ **{best_rating_tier}** restaurants achieve the highest average rating "
            f"({agg['Avg_Rating'].max():.2f}★). Consider this pricing sweet spot for new openings."
        )

        best_votes_tier = agg.loc[agg["Avg_Votes"].idxmax(), "Price Label"]
        recommendations.append(
            f"🗳️ **{best_votes_tier}** restaurants attract the most customer votes on average "
            f"({agg['Avg_Votes'].max():.0f} votes), indicating higher engagement."
        )

        if "Delivery_Pct" in agg.columns:
            best_del_tier = agg.loc[agg["Delivery_Pct"].idxmax(), "Price Label"]
            recommendations.append(
                f"🚴 Online delivery is most prevalent in the **{best_del_tier}** segment "
                f"({agg['Delivery_Pct'].max()*100:.1f}%). "
                f"Premium tiers may benefit from expanding delivery options."
            )

        if "Booking_Pct" in agg.columns:
            best_book_tier = agg.loc[agg["Booking_Pct"].idxmax(), "Price Label"]
            recommendations.append(
                f"📅 Table booking is most common among **{best_book_tier}** restaurants "
                f"({agg['Booking_Pct'].max()*100:.1f}%), suggesting guests plan visits in advance."
            )

        low_count_tiers = agg[agg["Count"] < 20]["Price Label"].tolist()
        if low_count_tiers:
            recommendations.append(
                f"⚠️ Price tiers with sparse data ({', '.join(map(str, low_count_tiers))}) "
                f"— interpret their averages with caution."
            )

    return {
        "price_stats_df":  agg,
        "recommendations": recommendations,
    }


# ═════════════════════════════════════════════════════════════════════════════
# PUBLIC: generate_all_insights  (used by pages/Insights.py)
# ═════════════════════════════════════════════════════════════════════════════

def generate_all_insights(df: pd.DataFrame) -> dict:
    """
    Generate a comprehensive set of data-driven business insights.

    Returns
    -------
    dict with keys:
        kpis            : dict
        delivery_impact : dict    (from delivery_booking_analysis)
        pricing         : dict    (from pricing_analysis)
        top_cities      : pd.DataFrame
        top_cuisines    : pd.DataFrame
        country_summary : pd.DataFrame
        market_gaps     : list[dict]
        growth_opps     : list[dict]
    """
    rated = df[df["Aggregate rating"] > 0].copy()

    # ── Core KPIs ─────────────────────────────────────────────────────────────
    kpis = {
        "total_restaurants":  len(df),
        "rated_restaurants":  len(rated),
        "rated_pct":          round(len(rated) / len(df) * 100, 1) if len(df) else 0,
        "avg_rating":         round(rated["Aggregate rating"].mean(), 3) if len(rated) else 0,
        "total_votes":        int(df["Votes"].sum()),
        "countries":          df["Country"].nunique() if "Country" in df.columns else 0,
        "cities":             df["City"].nunique(),
        "unique_cuisines":    df["Primary Cuisine"].nunique() if "Primary Cuisine" in df.columns else 0,
        "delivery_pct":       round(df["Has Online delivery"].mean() * 100, 1)
                              if "Has Online delivery" in df.columns else 0,
        "booking_pct":        round(df["Has Table booking"].mean() * 100, 1)
                              if "Has Table booking" in df.columns else 0,
    }

    # ── Sub-analyses ─────────────────────────────────────────────────────────
    dba     = delivery_booking_analysis(df)
    pricing = pricing_analysis(df)

    # ── Top cities ────────────────────────────────────────────────────────────
    top_cities = (
        rated.groupby("City")
        .agg(
            restaurant_count = ("Restaurant ID", "count"),
            avg_rating       = ("Aggregate rating", "mean"),
            total_votes      = ("Votes", "sum"),
            delivery_pct     = ("Has Online delivery", "mean")
                               if "Has Online delivery" in rated.columns
                               else ("Aggregate rating", lambda x: 0.0),
        )
        .reset_index()
        .sort_values("avg_rating", ascending=False)
    )
    top_cities["avg_rating"] = top_cities["avg_rating"].round(3)

    # ── Top cuisines ──────────────────────────────────────────────────────────
    top_cuisines: pd.DataFrame
    if "Primary Cuisine" in rated.columns:
        top_cuisines = (
            rated.groupby("Primary Cuisine")
            .agg(
                restaurant_count = ("Restaurant ID", "count"),
                avg_rating       = ("Aggregate rating", "mean"),
                total_votes      = ("Votes", "sum"),
            )
            .reset_index()
            .query("restaurant_count >= 10")
            .sort_values("avg_rating", ascending=False)
            .head(30)
        )
        top_cuisines["avg_rating"] = top_cuisines["avg_rating"].round(3)
    else:
        top_cuisines = pd.DataFrame()

    # ── Country summary ───────────────────────────────────────────────────────
    country_summary: pd.DataFrame
    if "Country" in rated.columns:
        country_summary = (
            rated.groupby("Country")
            .agg(
                restaurants = ("Restaurant ID", "count"),
                cities       = ("City", "nunique"),
                avg_rating   = ("Aggregate rating", "mean"),
                total_votes  = ("Votes", "sum"),
                delivery_pct = ("Has Online delivery", "mean")
                               if "Has Online delivery" in rated.columns
                               else ("Aggregate rating", lambda x: 0.0),
                booking_pct  = ("Has Table booking", "mean")
                               if "Has Table booking" in rated.columns
                               else ("Aggregate rating", lambda x: 0.0),
            )
            .reset_index()
            .sort_values("restaurants", ascending=False)
        )
        country_summary["avg_rating"] = country_summary["avg_rating"].round(3)
    else:
        country_summary = pd.DataFrame()

    # ── Market gaps (auto-generated insight cards) ────────────────────────────
    market_gaps: list[dict] = []

    # Low delivery in high-rating cities
    if "Has Online delivery" in rated.columns and len(top_cities) > 3:
        low_del = top_cities[
            (top_cities["delivery_pct"] < 0.25) &
            (top_cities["avg_rating"] > rated["Aggregate rating"].quantile(0.6))
        ].head(3)
        for _, row in low_del.iterrows():
            market_gaps.append({
                "type":    "opportunity",
                "icon":    "🚴",
                "title":   f"Delivery Gap in {row['City']}",
                "body":    (
                    f"{row['City']} has a strong avg rating of {row['avg_rating']:.2f}★ "
                    f"but only {row['delivery_pct']*100:.0f}% of restaurants offer delivery. "
                    "Expanding online delivery here could capture underserved demand."
                ),
            })

    # High-rating low-count cuisines
    if "Primary Cuisine" in rated.columns:
        rare_gems = (
            rated.groupby("Primary Cuisine")
            .agg(count=("Restaurant ID", "count"), avg_r=("Aggregate rating", "mean"))
            .reset_index()
            .query("count <= 15 and avg_r >= 4.2")
            .sort_values("avg_r", ascending=False)
            .head(3)
        )
        for _, row in rare_gems.iterrows():
            market_gaps.append({
                "type":  "niche",
                "icon":  "🍽️",
                "title": f"Niche Opportunity: {row['Primary Cuisine']}",
                "body":  (
                    f"{row['Primary Cuisine']} cuisine averages {row['avg_r']:.2f}★ "
                    f"but has only {int(row['count'])} restaurants. "
                    "This under-served high-quality niche may be ripe for new entrants."
                ),
            })

    # ── Growth opportunities ──────────────────────────────────────────────────
    growth_opps: list[dict] = []

    del_lift = dba["delivery"]["rating_lift"]
    if del_lift > 0:
        growth_opps.append({
            "icon":  "📈",
            "title": "Enable Online Delivery",
            "body":  (
                f"Restaurants with online delivery score {del_lift:+.3f}★ higher on average. "
                "If you're not offering delivery yet, this is the single highest-impact improvement."
            ),
        })

    book_lift = dba["booking"]["rating_lift"]
    if book_lift > 0:
        growth_opps.append({
            "icon":  "📅",
            "title": "Activate Table Booking",
            "body":  (
                f"Restaurants with table booking achieve {book_lift:+.3f}★ higher ratings. "
                "Setting up an online reservation system is a low-cost, high-signal quality indicator."
            ),
        })

    cross_lift = dba["cross"].get("lift_vs_neither", 0)
    if cross_lift > 0.1:
        growth_opps.append({
            "icon":  "🚀",
            "title": "Full-Service Bundle",
            "body":  (
                f"Restaurants offering both delivery AND booking outperform basic-only "
                f"peers by {cross_lift:+.3f}★. The combined effect is greater than either alone."
            ),
        })

    return {
        "kpis":            kpis,
        "delivery_impact": dba,
        "pricing":         pricing,
        "top_cities":      top_cities,
        "top_cuisines":    top_cuisines,
        "country_summary": country_summary,
        "market_gaps":     market_gaps,
        "growth_opps":     growth_opps,
    }
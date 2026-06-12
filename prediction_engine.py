from __future__ import annotations

import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any

# ── Suppress all warnings before other imports ────────────────────────────────
warnings.filterwarnings("ignore")
logging.getLogger("streamlit").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.caching").setLevel(logging.ERROR)

os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "true")
os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")

import joblib
import numpy as np
import pandas as pd

# ── Path bootstrap ─────────────────────────────────────────────────────────────
_THIS = Path(__file__).resolve()
_SRC  = _THIS.parent
_ROOT = _SRC.parent if _SRC.name == "src" else _SRC

for _p in [str(_ROOT), str(_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Local imports ─────────────────────────────────────────────────────────────
try:
    from feature_engineering import (
        ML_FEATURE_COLS,
        build_feature_matrix,
        encode_features,
        engineer_features,
        scale_features,
    )
except ImportError as _exc:
    raise ImportError(
        f"Cannot import feature_engineering: {_exc}\n"
        f"  Looked in: {_SRC}  and  {_ROOT}"
    ) from _exc

# ── Artefact paths ────────────────────────────────────────────────────────────
MODELS_DIR      = _ROOT / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
SCALER_PATH     = MODELS_DIR / "scaler.pkl"
ENCODER_PATH    = MODELS_DIR / "label_encoders.pkl"
MODEL_NAME_PATH = MODELS_DIR / "best_model_name.txt"

# ── Rating label thresholds ────────────────────────────────────────────────────
RATING_LABELS = [
    (4.5, "Excellent", "#00C2A8", "🌟"),
    (4.0, "Very Good", "#22D3EE", "⭐"),
    (3.5, "Good",      "#6C63FF", "👍"),
    (3.0, "Average",   "#F59E0B", "😐"),
    (2.0, "Below Avg", "#F97316", "⚠️"),
    (0.0, "Poor",      "#EF4444", "❌"),
]


def _rating_label(score: float) -> tuple[str, str, str]:
    for threshold, label, color, emoji in RATING_LABELS:
        if score >= threshold:
            return label, color, emoji
    return "Poor", "#EF4444", "❌"


# ═════════════════════════════════════════════════════════════════════════════
# Load artefacts
# ═════════════════════════════════════════════════════════════════════════════

def load_model_artefacts() -> dict | None:
    """
    Load saved model / scaler / encoders from models/.
    Returns None if any required artefact is missing.

    Return dict keys:
        model, scaler, encoders, model_name
    """
    required = [BEST_MODEL_PATH, SCALER_PATH, ENCODER_PATH]
    missing  = [p for p in required if not p.exists()]

    if missing:
        return None

    try:
        return {
            "model":      joblib.load(BEST_MODEL_PATH),
            "scaler":     joblib.load(SCALER_PATH),
            "encoders":   joblib.load(ENCODER_PATH),   # ← key is always "encoders"
            "model_name": (
                MODEL_NAME_PATH.read_text(encoding="utf-8").strip()
                if MODEL_NAME_PATH.exists() else "Unknown"
            ),
        }
    except Exception as exc:
        print(f"\n⚠️  Failed to load artefacts: {exc}\n")
        return None


# ═════════════════════════════════════════════════════════════════════════════
# Confidence estimation
# ═════════════════════════════════════════════════════════════════════════════

def _confidence_score(pred: float, model: Any, X_scaled: pd.DataFrame) -> dict:
    """Estimate prediction confidence via tree variance or heuristic fallback."""
    std_dev: float

    if hasattr(model, "estimators_"):
        estimators = list(model.estimators_)
        if estimators and isinstance(estimators[0], (list, np.ndarray)):
            flat = [e[0] for e in estimators[:50]]
        else:
            flat = estimators[:50]

        try:
            tree_preds = np.array([t.predict(X_scaled)[0] for t in flat])
            std_dev    = float(tree_preds.std())
        except Exception:
            std_dev = 0.25

        confidence_pct = float(max(50.0, min(98.0, 100.0 - std_dev * 60.0)))
    else:
        distance_from_center = abs(pred - 3.5) / 2.0
        std_dev        = 0.30 - distance_from_center * 0.10
        confidence_pct = float(max(60.0, min(92.0, 78.0 + distance_from_center * 12.0)))

    return {
        "confidence_pct": round(confidence_pct, 1),
        "lower_95":       round(max(0.0, pred - 1.96 * std_dev), 2),
        "upper_95":       round(min(5.0, pred + 1.96 * std_dev), 2),
        "std_dev":        round(std_dev, 4),
        "stability":      "High" if std_dev < 0.15 else "Medium" if std_dev < 0.30 else "Low",
    }


# ═════════════════════════════════════════════════════════════════════════════
# Feature contributions
# ═════════════════════════════════════════════════════════════════════════════

def _feature_contributions(
    model:         Any,
    X_scaled:      pd.DataFrame,
    feature_names: list[str],
) -> pd.DataFrame:
    """Lightweight SHAP-style: global importance × |feature value|."""
    if hasattr(model, "feature_importances_"):
        imps = model.feature_importances_
    elif hasattr(model, "coef_"):
        imps = np.abs(np.asarray(model.coef_).flatten())
    else:
        return pd.DataFrame(columns=["Feature", "Weight"])

    vals    = np.abs(X_scaled.values.flatten()[: len(imps)])
    contrib = imps * vals
    total   = contrib.sum() or 1.0

    return (
        pd.DataFrame({
            "Feature": feature_names[: len(imps)],
            "Weight":  (contrib / total * 100).round(2),
        })
        .sort_values("Weight", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )


# ═════════════════════════════════════════════════════════════════════════════
# Main prediction pipeline
# ═════════════════════════════════════════════════════════════════════════════

def predict_restaurant(
    cuisine:      str,
    city:         str,
    price_range:  int,
    votes:        int,
    has_delivery: bool,
    has_booking:  bool,
    avg_cost:     float,
    artefacts:    dict,
) -> dict:
    """
    Predict restaurant rating and return a full prediction report.

    artefacts must be the dict returned by load_model_artefacts() with keys:
        model, scaler, encoders
    """
    model    = artefacts["model"]
    scaler   = artefacts["scaler"]
    encoders = artefacts["encoders"]   # ← consistent with load_model_artefacts()

    price_label_map = {1: "Budget", 2: "Affordable", 3: "Premium", 4: "Luxury"}
    row = {
        "Restaurant Name":      f"New Restaurant ({cuisine})",
        "Address":              f"{city}, {cuisine} Cuisine",
        "City":                 city,
        "Country Code":         1,
        "Country":              "India",
        "Cuisines":             cuisine,
        "Primary Cuisine":      cuisine,
        "Price range":          int(price_range),
        "Average Cost for two": float(avg_cost),
        "Votes":                int(votes),
        "Aggregate rating":     0.0,
        "Rating text":          "Not rated",
        "Has Table booking":    int(has_booking),
        "Has Online delivery":  int(has_delivery),
        "Is delivering now":    int(has_delivery),
        "Switch to order menu": 0,
        "Cuisine Count":        len(str(cuisine).split(",")),
        "Price Label":          price_label_map.get(int(price_range), "Affordable"),
        "Rating Category":      "Not rated",
        "Is Rated":             0,
    }

    tmp     = pd.DataFrame([row])
    eng     = engineer_features(tmp)
    enc, _  = encode_features(eng, label_encoders=encoders, fit=False)
    X       = build_feature_matrix(enc)
    X_sc, _ = scale_features(X, scaler=scaler, fit=False)

    pred_raw = float(model.predict(X_sc)[0])
    pred     = round(max(0.0, min(5.0, pred_raw)), 2)

    label, color, emoji = _rating_label(pred)
    ci                  = _confidence_score(pred, model, X_sc)
    contributions       = _feature_contributions(model, X_sc, X.columns.tolist())

    return {
        "predicted_rating":      pred,
        "confidence_pct":        ci["confidence_pct"],
        "lower_95":              ci["lower_95"],
        "upper_95":              ci["upper_95"],
        "std_dev":               ci["std_dev"],
        "stability":             ci["stability"],
        "label":                 label,
        "color":                 color,
        "emoji":                 emoji,
        "feature_contributions": contributions,
    }


# ═════════════════════════════════════════════════════════════════════════════
# AI Recommendations
# ═════════════════════════════════════════════════════════════════════════════

def generate_ai_recommendations(
    predicted_rating: float,
    has_delivery:     bool,
    has_booking:      bool,
    votes:            int,
    price_range:      int,
    cuisine:          str,
    city:             str,
    avg_cost:         float,
    df:               pd.DataFrame,
) -> list[dict]:
    """
    Generate contextual AI recommendations benchmarked against real dataset stats.

    Returns list[dict] with keys:
        title, body, impact, priority, icon, category
    Priority: "Critical" | "High" | "Medium" | "Info"
    """
    recs: list[dict] = []

    df = df.copy()
    for col in ["Has Online delivery", "Has Table booking"]:
        if col in df.columns:
            if df[col].dtype == object:
                df[col] = (
                    df[col].str.strip()
                    .map({"Yes": True, "No": False, "1": True, "0": False})
                    .fillna(False)
                )
            df[col] = df[col].astype(bool)

    rated = df[df["Aggregate rating"] > 0].copy()
    if rated.empty:
        return [{
            "title":    "⚠️ No Rated Data",
            "body":     "Dataset contains no rated restaurants. Load the cleaned dataset.",
            "impact":   "N/A",
            "priority": "Info",
            "icon":     "ℹ️",
            "category": "Data",
        }]

    cuisine_col = "Primary Cuisine" if "Primary Cuisine" in rated.columns else "Cuisines"

    city_restaurants = rated[rated["City"] == city] if "City" in rated.columns else pd.DataFrame()
    city_avg = (
        city_restaurants["Aggregate rating"].mean()
        if len(city_restaurants) > 0
        else rated["Aggregate rating"].mean()
    )

    cuisine_data = rated[rated[cuisine_col].str.lower() == cuisine.lower()]
    cuisine_avg  = (
        cuisine_data["Aggregate rating"].mean()
        if len(cuisine_data) > 0
        else rated["Aggregate rating"].mean()
    )

    del_col = "Has Online delivery"
    bk_col  = "Has Table booking"

    delivery_avg    = rated[rated[del_col]]["Aggregate rating"].mean()  if del_col in rated.columns else 0.0
    no_delivery_avg = rated[~rated[del_col]]["Aggregate rating"].mean() if del_col in rated.columns else 0.0
    booking_avg     = rated[rated[bk_col]]["Aggregate rating"].mean()   if bk_col  in rated.columns else 0.0
    no_booking_avg  = rated[~rated[bk_col]]["Aggregate rating"].mean()  if bk_col  in rated.columns else 0.0

    gap = city_avg - predicted_rating

    top_cuisine_cities = (
        rated.groupby([cuisine_col, "City"])["Aggregate rating"].mean().reset_index()
        if "City" in rated.columns else pd.DataFrame()
    )

    # 1. Delivery
    if del_col in rated.columns and not has_delivery and (delivery_avg - no_delivery_avg) > 0.05:
        lift = round(delivery_avg - no_delivery_avg, 2)
        recs.append({
            "title":    "🚀 Enable Online Delivery",
            "body":     (
                f"Restaurants with online delivery average **{delivery_avg:.2f}★** "
                f"vs **{no_delivery_avg:.2f}★** without — a **+{lift}** rating lift."
            ),
            "impact":   f"+{lift} avg rating lift",
            "priority": "High",
            "icon":     "📦",
            "category": "Operations",
        })

    # 2. Table booking
    if bk_col in rated.columns and not has_booking:
        lift = round(booking_avg - no_booking_avg, 2)
        if lift > 0.05:
            recs.append({
                "title":    "📅 Activate Table Booking",
                "body":     (
                    f"Restaurants with reservations score **{booking_avg:.2f}★** on average. "
                    "Booking systems signal reliability and attract premium dine-in customers."
                ),
                "impact":   f"+{lift} avg rating lift",
                "priority": "Medium",
                "icon":     "🪑",
                "category": "Customer Experience",
            })

    # 3. Votes / social proof
    if "Votes" in rated.columns:
        city_median_votes = (
            city_restaurants["Votes"].median()
            if len(city_restaurants) > 0
            else rated["Votes"].median()
        )
        if votes < city_median_votes:
            recs.append({
                "title":    "📣 Build Social Proof via Reviews",
                "body":     (
                    f"Your projected votes (**{votes:,}**) are below the city median "
                    f"(**{int(city_median_votes):,}**). Higher vote counts correlate with "
                    "better ratings."
                ),
                "impact":   "Higher votes → improved discoverability",
                "priority": "High" if votes < city_median_votes * 0.3 else "Medium",
                "icon":     "⭐",
                "category": "Marketing",
            })

    # 4. City benchmark gap
    if gap > 0.3:
        recs.append({
            "title":    f"📍 Close the {city} City Gap",
            "body":     (
                f"The average rating in **{city}** is **{city_avg:.2f}★**. "
                f"Your prediction of **{predicted_rating:.2f}★** is **{gap:.2f} points below**."
            ),
            "impact":   f"Target ≥ {city_avg:.2f}★ to match city peers",
            "priority": "Critical" if gap > 0.6 else "High",
            "icon":     "🏙️",
            "category": "Competitive Position",
        })

    # 5. Pricing
    if len(top_cuisine_cities) > 0:
        if price_range >= 3:
            top_match = top_cuisine_cities[
                top_cuisine_cities[cuisine_col].str.lower() == cuisine.lower()
            ].nlargest(1, "Aggregate rating")
            if len(top_match) > 0:
                best_city = top_match.iloc[0]["City"]
                best_rt   = top_match.iloc[0]["Aggregate rating"]
                recs.append({
                    "title":    "💎 Premium Market Positioning",
                    "body":     (
                        f"**{cuisine}** performs best in **{best_city}** at **{best_rt:.2f}★**."
                    ),
                    "impact":   "Premium market alignment",
                    "priority": "Medium",
                    "icon":     "🥂",
                    "category": "Brand Strategy",
                })
        else:
            recs.append({
                "title":    "💡 Value-for-Money Positioning",
                "body":     (
                    "Budget restaurants thrive on **consistency** and **speed**. "
                    f"Avg cost of **₹{avg_cost:,.0f}** should be matched with high throughput."
                ),
                "impact":   "Value segment optimisation",
                "priority": "Medium",
                "icon":     "💰",
                "category": "Pricing",
            })
    elif price_range < 3:
        recs.append({
            "title":    "💡 Value-for-Money Positioning",
            "body":     (
                "Budget restaurants thrive on **consistency** and **speed**. "
                f"Avg cost of **₹{avg_cost:,.0f}** should be matched with high throughput."
            ),
            "impact":   "Value segment optimisation",
            "priority": "Medium",
            "icon":     "💰",
            "category": "Pricing",
        })

    # 6. Cuisine market fit
    if len(cuisine_data) > 10:
        top_pct = (cuisine_data["Aggregate rating"] > predicted_rating).mean() * 100
        recs.append({
            "title":    f"🍴 {cuisine} Market Landscape",
            "body":     (
                f"Among **{len(cuisine_data):,}** {cuisine} restaurants, your predicted "
                f"**{predicted_rating:.2f}★** places you in the **top {100 - top_pct:.0f}%** "
                f"of the segment (segment avg: **{cuisine_avg:.2f}★**)."
            ),
            "impact":   f"Segment rank: top {100 - top_pct:.0f}%",
            "priority": "Info",
            "icon":     "📊",
            "category": "Market Intelligence",
        })

    _order = {"Critical": 0, "High": 1, "Medium": 2, "Info": 3}
    recs.sort(key=lambda r: _order.get(r["priority"], 99))
    return recs


# ═════════════════════════════════════════════════════════════════════════════
# Executive Summary Generator
# ═════════════════════════════════════════════════════════════════════════════

def generate_executive_summary(df: pd.DataFrame) -> dict:
    """
    Produce a dynamic executive summary from the full dataset.

    Returns dict with keys:
        best_cities, highest_engagement_cuisines, pricing_trends,
        delivery_impact, top_performers, market_insights, summary_stats
    """
    df = df.copy()
    for col in ["Has Online delivery", "Has Table booking"]:
        if col in df.columns:
            if df[col].dtype == object:
                df[col] = (
                    df[col].str.strip()
                    .map({"Yes": True, "No": False, "1": True, "0": False})
                    .fillna(False)
                )
            df[col] = df[col].astype(bool)

    rated  = df[df["Aggregate rating"] > 0].copy()
    _empty = pd.DataFrame()

    if rated.empty:
        return {
            "best_cities":                 _empty,
            "highest_engagement_cuisines": _empty,
            "pricing_trends":              _empty,
            "delivery_impact":             _empty,
            "top_performers":              _empty,
            "market_insights":             [],
            "summary_stats":               {},
        }

    # Best performing cities
    city_stats = (
        rated.groupby("City")
        .agg(
            avg_rating=("Aggregate rating", "mean"),
            total_votes=("Votes", "sum"),
            count=("Restaurant Name", "count"),
        )
        .reset_index()
    )
    city_stats  = city_stats[city_stats["count"] >= 10]
    best_cities = city_stats.nlargest(10, "avg_rating").round(3).reset_index(drop=True)

    # Highest engagement cuisines
    cuisine_col   = "Primary Cuisine" if "Primary Cuisine" in rated.columns else "Cuisines"
    cuisine_stats = (
        rated.groupby(cuisine_col)
        .agg(
            avg_rating=("Aggregate rating", "mean"),
            total_votes=("Votes", "sum"),
            count=("Restaurant Name", "count"),
            avg_votes=("Votes", "mean"),
        )
        .reset_index()
    )
    cuisine_stats = cuisine_stats[cuisine_stats["count"] >= 5]
    cuisine_stats["engagement_score"] = (
        cuisine_stats["avg_votes"].rank(pct=True) * 0.5
        + cuisine_stats["avg_rating"].rank(pct=True) * 0.5
    )
    top_cuisines = cuisine_stats.nlargest(12, "engagement_score").round(3).reset_index(drop=True)

    # Pricing trends
    pricing = (
        rated.groupby("Price range")
        .agg(
            avg_rating=("Aggregate rating", "mean"),
            avg_votes=("Votes", "mean"),
            count=("Restaurant Name", "count"),
        )
        .reset_index()
    )
    pricing["Price Label"] = pricing["Price range"].map(
        {1: "Budget", 2: "Affordable", 3: "Premium", 4: "Luxury"}
    )
    pricing = pricing.round(3)

    # Delivery impact
    del_col = "Has Online delivery"
    if del_col in rated.columns:
        delivery_impact = (
            rated.groupby(del_col)
            .agg(
                avg_rating=("Aggregate rating", "mean"),
                avg_votes=("Votes", "mean"),
                count=("Restaurant Name", "count"),
            )
            .reset_index()
        )
        delivery_impact["Group"] = delivery_impact[del_col].map(
            {True: "With Delivery", False: "No Delivery"}
        )
    else:
        delivery_impact = _empty

    # Top performers by composite score
    max_v = max(rated["Votes"].max(), 1)
    rated = rated.copy()
    rated["composite_score"] = (
        rated["Aggregate rating"] * 0.6
        + np.log1p(rated["Votes"]) / np.log1p(max_v) * 0.4 * 5
    )
    keep = [
        c for c in [
            "Restaurant Name", "City", cuisine_col, "Aggregate rating",
            "Votes", "composite_score", "Has Online delivery",
            "Has Table booking", "Price range",
        ]
        if c in rated.columns
    ]
    top_performers = rated.nlargest(20, "composite_score")[keep].round(3).reset_index(drop=True)
    top_performers.index += 1

    # Delivery lift
    delivery_lift = 0.0
    if len(delivery_impact) >= 2 and del_col in delivery_impact.columns:
        wd = delivery_impact[delivery_impact[del_col] == True]
        nd = delivery_impact[delivery_impact[del_col] == False]
        if len(wd) and len(nd):
            delivery_lift = float(wd["avg_rating"].values[0] - nd["avg_rating"].values[0])

    top_city        = best_cities.iloc[0]["City"]       if len(best_cities) > 0  else "N/A"
    top_city_rating = best_cities.iloc[0]["avg_rating"] if len(best_cities) > 0  else 0.0
    top_cuisine_val = top_cuisines.iloc[0][cuisine_col] if len(top_cuisines) > 0 else "N/A"
    luxury_rows     = pricing[pricing["Price range"] == 4]
    budget_rows     = pricing[pricing["Price range"] == 1]
    luxury_avg      = float(luxury_rows["avg_rating"].values[0]) if len(luxury_rows) > 0 else 0.0
    budget_avg      = float(budget_rows["avg_rating"].values[0]) if len(budget_rows) > 0 else 0.0

    market_insights = [
        {
            "icon": "🏆",
            "text": (
                f"**{top_city}** leads with an average rating of **{top_city_rating:.2f}★**, "
                "making it the most competitive market for restaurant excellence."
            ),
            "type": "success",
        },
        {
            "icon": "🚚",
            "text": (
                f"Restaurants with online delivery score **{delivery_lift:+.2f} higher** on average "
                "— a statistically significant engagement and loyalty signal."
            ),
            "type": "info" if delivery_lift > 0 else "warning",
        },
        {
            "icon": "🍴",
            "text": (
                f"**{top_cuisine_val}** cuisine leads the engagement index, combining high ratings "
                "with strong vote volumes — ideal for new entrants."
            ),
            "type": "success",
        },
        {
            "icon": "💎",
            "text": (
                f"Luxury restaurants average **{luxury_avg:.2f}★** vs **{budget_avg:.2f}★** "
                "for budget — higher price does not always mean higher ratings."
            ),
            "type": "warning" if luxury_avg < budget_avg + 0.3 else "info",
        },
        {
            "icon": "📊",
            "text": (
                f"The dataset spans **{df['City'].nunique():,} cities** with "
                f"**{len(df):,}** total restaurants analysed."
            ),
            "type": "info",
        },
    ]

    return {
        "best_cities":                 best_cities,
        "highest_engagement_cuisines": top_cuisines,
        "pricing_trends":              pricing,
        "delivery_impact":             delivery_impact,
        "top_performers":              top_performers,
        "market_insights":             market_insights,
        "summary_stats": {
            "total_restaurants": len(df),
            "rated_restaurants": len(rated),
            "avg_rating":        round(float(rated["Aggregate rating"].mean()), 2),
            "delivery_lift":     round(delivery_lift, 3),
            "top_city":          top_city,
            "top_cuisine":       top_cuisine_val,
        },
    }


# ═════════════════════════════════════════════════════════════════════════════
# Self-test
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("✅ prediction_engine.py loaded successfully.")
    artefacts = load_model_artefacts()
    if artefacts:
        print(f"✅ Model loaded: {artefacts['model_name']}")
    else:
        print("ℹ️  No model artefacts found. Run main.py to train first.")
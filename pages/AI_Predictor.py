import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.components import inject_global_css, section_header
from src.prediction_engine import (
    generate_ai_recommendations,
    generate_executive_summary,
    load_model_artefacts,
    predict_restaurant,
)
from src.preprocessing import PRICE_LABEL, load_and_preprocess
from src.train_model import (
    BEST_MODEL_PATH,
    get_feature_importance,
    train_all_models,
)

# ── Page setup ─────────────────────────────────────────────────────────────────
inject_global_css()

<div style="
    background:linear-gradient(135deg,#04060F 0%,#08101E 40%,#060A18 100%);
    border:1px solid rgba(236,72,153,0.18);border-radius:24px;
    padding:44px 52px 36px;margin-bottom:0;position:relative;
  position:relative;
    z-index:0;
    box-shadow:0 2px 8px rgba(0,0,0,0.5),0 8px 32px rgba(0,0,0,0.4);
    overflow:hidden;
">
  <div style="position:absolute;top:-80px;right:-80px;width:340px;height:340px;
       background:radial-gradient(circle,rgba(236,72,153,0.12),transparent 65%);
       pointer-events:none;z-index:-1;"></div>
  <div style="position:absolute;bottom:-60px;left:25%;width:280px;height:280px;
       background:radial-gradient(circle,rgba(124,58,237,0.10),transparent 65%);
       pointer-events:none;z-index:-1;"></div>
  <div style="position:absolute;top:0;left:0;right:0;height:1px;
       background:linear-gradient(90deg,transparent,rgba(236,72,153,0.45) 35%,rgba(124,58,237,0.35) 70%,transparent);
       pointer-events:none;z-index:-1;"></div>

  <div style="display:inline-flex;align-items:center;gap:8px;
       background:rgba(236,72,153,0.10);border:1px solid rgba(236,72,153,0.28);
       border-radius:999px;padding:5px 16px;font-size:0.67rem;color:#F9A8D4;
       letter-spacing:.16em;text-transform:uppercase;font-weight:700;
       margin-bottom:20px;font-family:'DM Sans',sans-serif;">
    <span style="width:6px;height:6px;border-radius:50%;background:#10B981;
          box-shadow:0 0 8px rgba(16,185,129,0.9);display:inline-block;"></span>
    🤖 AI Prediction & Intelligence Hub · Level 3 Task 3
  </div>

  <h1 style="font-family:'Sora',sans-serif;font-size:2.8rem;font-weight:900;
       line-height:1.06;margin:0 0 10px;letter-spacing:-0.04em;">
    <span style="background:linear-gradient(135deg,#F9A8D4 0%,#C4B5FD 100%);
         -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
      AI Prediction Hub
    </span>
  </h1>

  <p style="color:#CBD5E1;font-size:0.92rem;margin-bottom:28px;max-width:640px;
       line-height:1.65;font-family:'DM Sans',sans-serif;">
    Ensemble ML rating prediction · Feature importance analysis · Top restaurant rankings ·
    Executive summaries · Smart AI insights — all derived from
    <strong style="color:#F1F5F9;">9,551 real restaurants</strong> across 15 countries.
  </p>

  <div style="display:flex;flex-wrap:wrap;gap:10px;align-items:center;">
    <div style="background:rgba(236,72,153,0.10);border:1px solid rgba(236,72,153,0.22);
         border-radius:999px;padding:5px 16px;font-family:'JetBrains Mono',monospace;
         font-size:0.71rem;color:#F9A8D4;font-weight:500;">🎯 Rating Prediction</div>
    <div style="background:rgba(124,58,237,0.10);border:1px solid rgba(124,58,237,0.22);
         border-radius:999px;padding:5px 16px;font-family:'JetBrains Mono',monospace;
         font-size:0.71rem;color:#A78BFA;font-weight:500;">📊 Feature Analysis</div>
    <div style="background:rgba(6,182,212,0.10);border:1px solid rgba(6,182,212,0.22);
         border-radius:999px;padding:5px 16px;font-family:'JetBrains Mono',monospace;
         font-size:0.71rem;color:#67E8F9;font-weight:500;">🏆 Rankings</div>
    <div style="background:rgba(16,185,129,0.10);border:1px solid rgba(16,185,129,0.22);
         border-radius:999px;padding:5px 16px;font-family:'JetBrains Mono',monospace;
         font-size:0.71rem;color:#6EE7B7;font-weight:500;">💡 Smart Insights</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown(
    '<div style="display:block;height:32px;width:100%;"></div>',
    unsafe_allow_html=True,
)

st.markdown("""
<style>
.rec-card {
    background: var(--card);
    border-left: 4px solid;
    border-radius: 12px;
    padding: 18px 22px;
    margin-bottom: 14px;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Cached data & model loaders
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def _load_df() -> pd.DataFrame:
    return load_and_preprocess()


@st.cache_data(show_spinner=False)
def _load_rated(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["Aggregate rating"] > 0].copy()


@st.cache_resource(show_spinner=False)
def _load_artefacts():
    return load_model_artefacts()


@st.cache_data(show_spinner=False)
def _compute_feat_imp(model_key: str, _artefacts: dict) -> pd.DataFrame | None:
    from src.feature_engineering import ML_FEATURE_COLS
    model = _artefacts["model"]
    feat_imp = None

    if hasattr(model, "feature_importances_"):
        imps  = model.feature_importances_
        avail = ML_FEATURE_COLS[: len(imps)]
        feat_imp = pd.DataFrame({
            "Feature":    avail,
            "Importance": imps[: len(avail)],
        }).sort_values("Importance", ascending=False).reset_index(drop=True)
    elif hasattr(model, "coef_"):
        avail = ML_FEATURE_COLS[: len(model.coef_)]
        feat_imp = pd.DataFrame({
            "Feature":    avail,
            "Importance": np.abs(model.coef_[: len(avail)]),
        }).sort_values("Importance", ascending=False).reset_index(drop=True)

    if feat_imp is None or feat_imp.empty:
        return None

    feat_imp["Importance %"] = (
        feat_imp["Importance"] / feat_imp["Importance"].sum() * 100
    ).round(2)
    return feat_imp


@st.cache_data(show_spinner=False)
def _compute_corr_matrix(df: pd.DataFrame) -> pd.DataFrame:
    from src.feature_engineering import get_correlation_matrix
    return get_correlation_matrix(df)


@st.cache_data(show_spinner=False)
def _scatter_sample(rated: pd.DataFrame, n: int = 3000) -> pd.DataFrame:
    return rated.sample(min(n, len(rated)), random_state=42)


@st.cache_data(show_spinner=False)
def _compute_insights(rated: pd.DataFrame):
    city_delivery = (
        rated.groupby(["City", "Has Online delivery"])["Aggregate rating"]
        .mean().unstack().fillna(0)
    )
    city_delivery.columns = ["No Delivery", "With Delivery"]
    city_delivery = city_delivery[city_delivery["No Delivery"] > 0]
    city_delivery["Lift"] = city_delivery["With Delivery"] - city_delivery["No Delivery"]
    city_delivery = city_delivery[city_delivery.index.isin(
        rated["City"].value_counts().head(20).index
    )].sort_values("Lift", ascending=False)

    prem = rated[rated["Price range"] >= 3]
    prem_cuisine_city = (
        prem.groupby("Primary Cuisine")
        .agg(
            count=("Restaurant Name", "count"),
            avg_rating=("Aggregate rating", "mean"),
            avg_cost=("Average Cost for two", "mean"),
        )
        .reset_index()
    )
    prem_cuisine_city = (
        prem_cuisine_city[prem_cuisine_city["count"] >= 10]
        .nlargest(12, "avg_rating")
    )

    rated_svc = rated.copy()
    rated_svc["Service Level"] = rated_svc.apply(
        lambda r: (
            "Full Service"   if r["Has Online delivery"] and r["Has Table booking"]
            else "Delivery Only" if r["Has Online delivery"]
            else "Booking Only"  if r["Has Table booking"]
            else "Basic"
        ),
        axis=1,
    )
    service_agg = (
        rated_svc.groupby("Service Level")
        .agg(
            avg_rating=("Aggregate rating", "mean"),
            avg_votes=("Votes", "mean"),
            count=("Restaurant Name", "count"),
        )
        .reset_index()
        .sort_values("avg_rating", ascending=False)
    )

    if "Cuisine Count" in rated_svc.columns:
        cuisine_count_series = rated_svc["Cuisine Count"]
    else:
        cuisine_count_series = rated_svc["Cuisines"].str.split(",").str.len()

    rated_svc = rated_svc.copy()
    rated_svc["Cuisine_Count_Grp"] = pd.cut(
        cuisine_count_series,
        bins=[0, 1, 2, 3, 20],
        labels=["Single", "Dual", "Triple", "4+"],
    )
    cuisine_diversity = (
        rated_svc.groupby("Cuisine_Count_Grp", observed=True)
        .agg(
            avg_rating=("Aggregate rating", "mean"),
            count=("Restaurant Name", "count"),
        )
        .reset_index()
    )

    return city_delivery, prem_cuisine_city, service_agg, cuisine_diversity


# ── Load data & artefacts ──────────────────────────────────────────────────────
df        = _load_df()
rated     = _load_rated(df)
artefacts = _load_artefacts()


# ══════════════════════════════════════════════════════════════════════════════
# IN-PAGE NAVIGATION + CONTROLS
# ══════════════════════════════════════════════════════════════════════════════
controls_col, content_col = st.columns([1, 3], gap="large")

with controls_col:
    # FIX: Controls panel decorative header rendered as plain markdown.
    # The div uses position:relative + overflow:visible + z-index:0
    # so it never traps the widget layer inside a stacking context.
    st.markdown("""
    <div style="background:linear-gradient(160deg,rgba(255,255,255,0.03) 0%,transparent 100%),rgba(6,10,18,0.90);
         border:1px solid rgba(236,72,153,0.20);border-radius:20px;padding:20px 16px 16px;
         position:relative;overflow:visible;z-index:0;">
      <div style="position:absolute;top:0;left:0;right:0;height:1px;
           background:linear-gradient(90deg,transparent,rgba(236,72,153,0.40),transparent);
           pointer-events:none;z-index:-1;"></div>
      <div style="font-family:'DM Sans',sans-serif;font-size:0.63rem;font-weight:700;
           text-transform:uppercase;letter-spacing:0.18em;color:#F9A8D4;margin-bottom:4px;">
        🎛️ Prediction Controls
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    if artefacts:
        st.success("✅ **Model loaded & ready**")
    else:
        st.warning("⚠️ **No trained model found**")

    st.divider()

    st.markdown("""
    <div style="font-family:'DM Sans',sans-serif;font-size:0.63rem;font-weight:700;
         color:#94A3B8;text-transform:uppercase;letter-spacing:0.16em;margin-bottom:6px;">
      📂 Navigation
    </div>
    """, unsafe_allow_html=True)

    section_choice = st.radio(
        "Jump to section:",
        [
            "🎯 AI Rating Predictor",
            "📊 Feature Importance",
            "🏆 Top Restaurants",
            "📋 Executive Summary",
            "💡 Smart AI Insights",
        ],
        label_visibility="collapsed",
    )

    if not artefacts:
        st.divider()
        if st.button("🚀 Train Model Now", type="primary", key="_sidebar_train_btn",
                     use_container_width=True):
            st.session_state["_trigger_train"] = True
            st.rerun()


# ── Training UI ────────────────────────────────────────────────────────────────
if not artefacts and st.session_state.get("_trigger_train"):
    st.session_state["_trigger_train"] = False
    with st.spinner("🚀 Training models… this takes ~30–60 seconds"):
        prog = st.progress(0, text="Initialising training pipeline …")
        try:
            with st.status("Training pipeline …", expanded=True) as _train_status:
                st.write("🔄 Loading features …")
                prog.progress(20, text="Loading features …")
                st.write("🏋️ Fitting models …")
                prog.progress(50, text="Fitting models …")
                train_all_models(df)
                prog.progress(90, text="Saving artefacts …")
                st.write("💾 Saving artefacts …")
                prog.progress(100, text="Done!")
                _train_status.update(
                    label="✅ Training complete!", state="complete", expanded=False
                )
            _load_artefacts.clear()
            st.success("✅ Model trained successfully. Reloading …")
            st.rerun()
        except Exception as e:
            prog.empty()
            st.error(f"Training failed: {e}")


# ── Friendly feature label map ─────────────────────────────────────────────────
_FRIENDLY: dict[str, str] = {
    "Has Online delivery":     "Online Delivery",
    "Has Table booking":       "Table Booking",
    "Votes":                   "Customer Votes",
    "Average Cost for two":    "Average Cost",
    "Price range":             "Price Range",
    "Log_Votes":               "Engagement (log Votes)",
    "Full_Service_Flag":       "Full-Service Score",
    "Premium_Restaurant_Flag": "Premium Flag",
    "Cost_Efficiency_Score":   "Cost Efficiency",
    "Cuisine_Count":           "Cuisine Diversity",
    "Votes_Per_Rating":        "Vote Density",
    "Rating_x_Votes":         "Rating × Vote",
    "Primary Cuisine_Enc":    "Cuisine Type",
    "City_Enc":               "City Market",
    "Country_Enc":            "Country Market",
    "Log_Cost":               "Cost (log)",
    "Restaurant_Name_Length": "Name Length",
    "Address_Length":         "Address Detail",
}


# ══════════════════════════════════════════════════════════════════════════════
# Section content
# ══════════════════════════════════════════════════════════════════════════════
with content_col:

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1 – AI RATING PREDICTOR
    # ══════════════════════════════════════════════════════════════════════════
    if section_choice == "🎯 AI Rating Predictor":

        # FIX: Section panel uses position:relative + z-index:0 + 
        # The top-edge decorative line uses pointer-events:none + z-index:-1
        # so it never sits above the widget layer.
        st.markdown("""
        <div style="background:linear-gradient(135deg,rgba(236,72,153,0.10),rgba(124,58,237,0.07));
             border:1px solid rgba(236,72,153,0.22);border-radius:22px;
             padding:36px 40px;margin-bottom:28px;position:relative;
             z-index:0;overflow:visible;">
          <div style="position:absolute;top:0;left:0;right:0;height:1px;
               background:linear-gradient(90deg,transparent,rgba(236,72,153,0.50),transparent);
               pointer-events:none;z-index:-1;"></div>

          <div style="display:inline-flex;align-items:center;gap:8px;
               background:rgba(236,72,153,0.10);border:1px solid rgba(236,72,153,0.25);
               border-radius:999px;padding:5px 14px;font-size:0.67rem;color:#F9A8D4;
               letter-spacing:.16em;text-transform:uppercase;font-weight:700;
               margin-bottom:14px;font-family:'DM Sans',sans-serif;">
            ✦ Neural Intelligence Engine · Level 3 Task 3
          </div>

          <h2 style="font-family:'Sora',sans-serif;margin:0 0 10px;font-size:1.9rem;
               font-weight:900;letter-spacing:-0.03em;
               background:linear-gradient(135deg,#F9A8D4,#C4B5FD);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            Predict Your Restaurant's Rating
          </h2>
          <p style="color:#CBD5E1;margin:0;max-width:580px;line-height:1.65;font-size:0.92rem;
               font-family:'DM Sans',sans-serif;">
            Enter your restaurant's parameters below. Our ensemble ML model — trained on
            <strong style="color:#F1F5F9;">9,551 real restaurants</strong> across 15 countries —
            predicts your expected rating, confidence interval, and personalised growth recommendations.
          </p>
        </div>
        """, unsafe_allow_html=True)

        section_header("🎯 AI Rating Predictor")

        if not artefacts:
            st.info("👆 Train the model first using the button in the controls panel.")
            st.stop()

        with st.form("prediction_form"):
            c1, c2, c3 = st.columns(3)

            with c1:
                all_cuisines = sorted(df["Primary Cuisine"].dropna().unique().tolist())
                cuisine = st.selectbox(
                    "🍴 Primary Cuisine", all_cuisines,
                    index=all_cuisines.index("North Indian")
                          if "North Indian" in all_cuisines else 0,
                )
                has_delivery = st.toggle("🚚 Online Delivery", value=True)

            with c2:
                all_cities = sorted(df["City"].dropna().unique().tolist())
                city = st.selectbox(
                    "🏙️ City", all_cities,
                    index=all_cities.index("New Delhi")
                          if "New Delhi" in all_cities else 0,
                )
                has_booking = st.toggle("📅 Table Booking", value=False)

            with c3:
                price_range = st.select_slider(
                    "💰 Price Range",
                    options=[1, 2, 3, 4],
                    value=2,
                    format_func=lambda x: PRICE_LABEL.get(x, str(x)),
                )

            c4, c5 = st.columns(2)
            with c4:
                votes = st.number_input(
                    "⭐ Expected Votes",
                    min_value=0, max_value=100_000, value=500, step=50,
                )
            with c5:
                avg_cost = st.number_input(
                    "💵 Average Cost for Two (₹)",
                    min_value=0, max_value=200_000, value=800, step=100,
                )

            submitted = st.form_submit_button(
                "🤖 Predict Rating", type="primary", use_container_width=True,
            )

        if submitted:
            _validation_errors: list[str] = []
            if votes == 0:
                _validation_errors.append(
                    "**Expected Votes** must be at least 1 "
                    "(the model uses log(votes) — zero causes an invalid prediction)."
                )
            if avg_cost == 0:
                _validation_errors.append(
                    "**Average Cost for Two** must be greater than 0 "
                    "(the model uses log(cost) — zero causes an invalid prediction)."
                )

            if _validation_errors:
                for err in _validation_errors:
                    st.warning(f"⚠️ {err}")
                st.stop()

            _pred_progress = st.progress(0, text="Running AI inference …")
            result = None
            recs: list = []

            try:
                with st.spinner("🧠 Running AI inference …"):
                    _pred_progress.progress(30, text="Encoding features …")
                    try:
                        result = predict_restaurant(
                            cuisine=cuisine, city=city, price_range=price_range,
                            votes=votes, has_delivery=has_delivery,
                            has_booking=has_booking, avg_cost=avg_cost,
                            artefacts=artefacts,
                        )
                    except Exception as pred_err:
                        st.error(f"❌ Prediction failed: {pred_err}")
                        st.exception(pred_err)
                        st.stop()

                    if result is None:
                        st.error("❌ Prediction returned no result.")
                        st.stop()

                    _required_keys = [
                        "predicted_rating", "color", "label", "emoji",
                        "confidence_pct", "lower_95", "upper_95", "stability",
                    ]
                    _missing_keys = [k for k in _required_keys if k not in result]
                    if _missing_keys:
                        st.error(f"❌ Prediction result missing keys: {_missing_keys}")
                        st.stop()

                    _pred_progress.progress(75, text="Generating recommendations …")

                    try:
                        recs = generate_ai_recommendations(
                            predicted_rating=result["predicted_rating"],
                            has_delivery=has_delivery, has_booking=has_booking,
                            votes=votes, price_range=price_range,
                            cuisine=cuisine, city=city, avg_cost=avg_cost, df=df,
                        )
                    except Exception as rec_err:
                        st.warning(f"⚠️ AI recommendations unavailable: {rec_err}")
                        recs = []

                    _pred_progress.progress(100, text="Done!")

            finally:
                _pred_progress.empty()

            pred   = result["predicted_rating"]
            color  = result["color"]
            label  = result["label"]
            emoji  = result["emoji"]
            conf   = result["confidence_pct"]
            lo, hi = result["lower_95"], result["upper_95"]

            st.markdown(f"""
            <div style="background:linear-gradient(135deg,{color}18,{color}06);
                 border:2px solid {color}55;border-radius:22px;padding:32px 36px;
                 margin:24px 0;position:relative;overflow:visible;
                 z-index:0;;">
              <div style="position:absolute;top:0;left:0;right:0;height:2px;
                   background:linear-gradient(90deg,{color},{color}55);opacity:0.8;
                   pointer-events:none;z-index:-1;"></div>
              <div style="display:flex;align-items:center;gap:36px;flex-wrap:wrap;">
                <div>
                  <div style="font-family:'DM Sans',sans-serif;font-size:0.67rem;color:#94A3B8;
                       text-transform:uppercase;letter-spacing:0.16em;font-weight:700;margin-bottom:8px;">
                    Predicted Rating
                  </div>
                  <div style="font-family:'Sora',sans-serif;font-size:4.5rem;font-weight:900;
                       color:{color};line-height:1;letter-spacing:-0.04em;">
                    {pred:.2f}
                  </div>
                  <div style="font-size:1.3rem;margin-top:8px;">{emoji} {label}</div>
                </div>
                <div style="flex:1;min-width:260px;">
                  <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;">
                    <div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.10);
                         border-radius:14px;padding:18px;">
                      <div style="font-family:'DM Sans',sans-serif;color:#94A3B8;font-size:0.67rem;
                           text-transform:uppercase;letter-spacing:0.14em;margin-bottom:6px;">Confidence</div>
                      <div style="font-family:'Sora',sans-serif;font-size:1.9rem;font-weight:800;
                           color:#A78BFA;">{conf:.0f}%</div>
                      <div style="color:#94A3B8;font-size:0.78rem;font-family:'DM Sans',sans-serif;margin-top:4px;">
                        Stability: {result['stability']}
                      </div>
                    </div>
                    <div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.10);
                         border-radius:14px;padding:18px;">
                      <div style="font-family:'DM Sans',sans-serif;color:#94A3B8;font-size:0.67rem;
                           text-transform:uppercase;letter-spacing:0.14em;margin-bottom:6px;">95% CI Range</div>
                      <div style="font-family:'JetBrains Mono',monospace;font-size:1.2rem;font-weight:800;
                           color:#67E8F9;">{lo:.2f} – {hi:.2f}★</div>
                      <div style="color:#94A3B8;font-size:0.78rem;font-family:'DM Sans',sans-serif;margin-top:4px;">
                        Lower–Upper bound
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=conf,
                title={"text": "Prediction Confidence", "font": {"color": "#94A3B8", "size": 14}},
                number={"suffix": "%", "font": {"color": color, "size": 36}},
                gauge={
                    "axis":    {"range": [0, 100], "tickcolor": "#334155"},
                    "bar":     {"color": color, "thickness": 0.28},
                    "bgcolor": "#1E293B",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0,  50],  "color": "rgba(239,68,68,0.15)"},
                        {"range": [50, 75],  "color": "rgba(245,158,11,0.15)"},
                        {"range": [75, 100], "color": "rgba(0,194,168,0.15)"},
                    ],
                    "threshold": {"line": {"color": "#F8FAFC", "width": 2},
                                  "thickness": 0.82, "value": conf},
                },
            ))
            fig_gauge.update_layout(height=240, margin=dict(t=40, b=10, l=20, r=20),
                                    paper_bgcolor="rgba(0,0,0,0)", font_color="#F8FAFC")

            city_data    = rated[rated["City"] == city]["Aggregate rating"]
            cuisine_data = rated[rated["Primary Cuisine"] == cuisine]["Aggregate rating"]

            fig_dist = go.Figure()
            if len(city_data) > 5:
                fig_dist.add_trace(go.Histogram(x=city_data, name=f"{city} avg", nbinsx=30,
                    marker_color="rgba(124,58,237,0.5)", histnorm="probability density"))
            if len(cuisine_data) > 5:
                fig_dist.add_trace(go.Histogram(x=cuisine_data, name=f"{cuisine} avg", nbinsx=30,
                    marker_color="rgba(6,182,212,0.4)", histnorm="probability density"))
            fig_dist.add_vline(x=pred, line_width=3, line_dash="dash", line_color=color,
                annotation_text=f"Prediction: {pred:.2f}★", annotation_font_color=color)
            fig_dist.update_layout(title="Your Prediction vs Market Distribution", height=280,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#F8FAFC",
                xaxis=dict(title="Rating", gridcolor="rgba(255,255,255,0.06)"),
                yaxis=dict(title="Density", gridcolor="rgba(255,255,255,0.06)"),
                legend=dict(bgcolor="rgba(0,0,0,0)"))

            cg, cd = st.columns([1, 2])
            with cg:
                st.plotly_chart(fig_gauge, use_container_width=True)
            with cd:
                st.plotly_chart(fig_dist,  use_container_width=True)

            feat_df = result.get("feature_contributions")
            if feat_df is not None and not feat_df.empty:
                _weight_col = (
                    "Weight" if "Weight" in feat_df.columns
                    else feat_df.select_dtypes("number").columns[0]
                    if not feat_df.select_dtypes("number").empty else None
                )
                if _weight_col is not None:
                    section_header("🔬 Feature Contributions to This Prediction")
                    feat_df = feat_df.copy()
                    feat_df["Label"] = feat_df["Feature"].map(lambda x: _FRIENDLY.get(x, x))

                    fig_feat = go.Figure(go.Bar(
                        x=feat_df[_weight_col], y=feat_df["Label"], orientation="h",
                        marker=dict(color=feat_df[_weight_col],
                            colorscale=[[0, "#1E293B"], [0.4, "#7C3AED"], [1.0, "#06B6D4"]],
                            showscale=False),
                        text=[f"{w:.1f}%" for w in feat_df[_weight_col]],
                        textposition="outside", textfont=dict(color="#94A3B8", size=12),
                    ))
                    fig_feat.update_layout(title="Feature Contribution Weights (%)", height=380,
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font_color="#F8FAFC",
                        xaxis=dict(title="Contribution %", gridcolor="rgba(255,255,255,0.06)"),
                        yaxis=dict(tickfont=dict(size=11)),
                        margin=dict(l=160, r=60, t=50, b=40))
                    st.plotly_chart(fig_feat, use_container_width=True)

            if recs:
                section_header("💡 AI Business Recommendations")
                priority_colors = {
                    "Critical": "#EF4444", "High":   "#F59E0B",
                    "Medium":   "#7C3AED", "Info":   "#06B6D4",
                }
                for rec in recs:
                    pc     = priority_colors.get(rec["priority"], "#7C3AED")
                    raw_body = rec["body"]
                    parts    = raw_body.split("**")
                    body     = ""
                    for i, part in enumerate(parts):
                        body += f"<strong>{part}</strong>" if i % 2 == 1 else part

                    st.markdown(f"""
                    <div class="rec-card" style="border-left-color:{pc};">
                      <div style="display:flex;justify-content:space-between;
                           align-items:flex-start;flex-wrap:wrap;gap:8px;margin-bottom:8px;">
                        <div style="font-family:'Sora',sans-serif;font-size:0.97rem;
                             font-weight:700;color:#F1F5F9;">
                          {rec['icon']} {rec['title']}
                        </div>
                        <div style="display:flex;gap:8px;align-items:center;flex-shrink:0;">
                          <span style="background:{pc}20;border:1px solid {pc}55;
                               border-radius:999px;padding:2px 10px;font-size:0.68rem;
                               font-weight:700;color:{pc};">{rec['priority']}</span>
                          <span style="background:rgba(255,255,255,0.06);color:#94A3B8;
                               border-radius:999px;padding:2px 10px;font-size:0.68rem;
                               font-family:'DM Sans',sans-serif;">
                            {rec['category']}
                          </span>
                        </div>
                      </div>
                      <div style="color:#CBD5E1;font-size:0.88rem;
                           line-height:1.65;margin-bottom:10px;font-family:'DM Sans',sans-serif;">
                        {body}
                      </div>
                      <div style="color:{pc};font-size:0.78rem;font-weight:600;
                           font-family:'DM Sans',sans-serif;">
                        📈 Expected Impact: {rec['impact']}
                      </div>
                    </div>
                    """, unsafe_allow_html=True)


    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2 – FEATURE IMPORTANCE
    # ══════════════════════════════════════════════════════════════════════════
    elif section_choice == "📊 Feature Importance":

        st.markdown("""
        <div style="background:rgba(124,58,237,0.06);border:1px solid rgba(124,58,237,0.15);
             border-radius:14px;padding:16px 22px;margin-bottom:22px;
             position:relative;z-index:0;;">
          <div style="font-family:'DM Sans',sans-serif;font-size:0.82rem;color:#94A3B8;line-height:1.6;">
            <strong style="color:#A78BFA;">Feature Importance & Relationship Analysis</strong> —
            Level 3 Task 3 · Advanced visualizations · Cross-feature correlation analysis ·
            Votes vs Rating scatter · Radar and violin views.
          </div>
        </div>
        """, unsafe_allow_html=True)

        section_header("📊 Feature Importance & Relationship Analysis")

        if not artefacts:
            st.info("Train the model first (controls panel).")
            st.stop()

        _model     = artefacts["model"]
        _model_key = f"{type(_model).__name__}_{id(_model)}"
        feat_imp   = _compute_feat_imp(_model_key, artefacts)

        if feat_imp is None or feat_imp.empty:
            st.warning("Feature importance not available for this model type.")
            st.stop()

        feat_imp = feat_imp.copy()
        feat_imp["Label"] = feat_imp["Feature"].map(lambda x: _FRIENDLY.get(x, x))
        top15 = feat_imp.head(15)

        top3 = feat_imp.head(3)
        ic1, ic2, ic3 = st.columns(3)
        accents = ["#A78BFA", "#67E8F9", "#6EE7B7"]
        cols_   = [ic1, ic2, ic3]
        for idx, (col_, (_, row)) in enumerate(zip(cols_, top3.iterrows())):
            with col_:
                st.markdown(f"""
                <div style="background:rgba(9,14,25,0.88);border:1px solid rgba(255,255,255,0.12);
                     border-radius:16px;padding:20px 22px;position:relative;
                     z-index:0;;">
                  <div style="position:absolute;top:0;left:0;right:0;height:2px;
                       background:{accents[idx]};opacity:0.8;border-radius:16px 16px 0 0;
                       pointer-events:none;z-index:-1;"></div>
                  <div style="font-size:1.3rem;margin-bottom:8px;">🔑</div>
                  <div style="font-family:'JetBrains Mono',monospace;font-size:1.4rem;
                       font-weight:700;color:{accents[idx]};letter-spacing:-0.02em;">
                    #{idx+1}</div>
                  <div style="font-family:'DM Sans',sans-serif;font-size:0.88rem;
                       font-weight:600;color:#F1F5F9;margin-top:4px;">{row['Label']}</div>
                  <div style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;
                       color:#94A3B8;margin-top:4px;">{row['Importance %']:.1f}% importance</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)

        fig_bar = go.Figure(go.Bar(
            y=top15["Label"][::-1], x=top15["Importance %"][::-1], orientation="h",
            marker=dict(color=top15["Importance %"][::-1],
                colorscale=[[0, "#1E293B"], [0.3, "#7C3AED"], [0.7, "#8B5CF6"], [1.0, "#06B6D4"]],
                showscale=True, colorbar=dict(title="Importance %", tickfont=dict(color="#94A3B8"))),
            text=[f"{v:.1f}%" for v in top15["Importance %"][::-1]],
            textposition="outside", textfont=dict(color="#94A3B8"),
        ))
        fig_bar.update_layout(title="Top 15 Feature Importances – Ranked", height=520,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#F8FAFC",
            xaxis=dict(title="Importance (%)", gridcolor="rgba(255,255,255,0.06)"),
            yaxis=dict(tickfont=dict(size=11)), margin=dict(l=170, r=80, t=50, b=40))
        st.plotly_chart(fig_bar, use_container_width=True)

        fig_tree = px.treemap(feat_imp.head(12), path=["Label"], values="Importance %",
            color="Importance %", color_continuous_scale=["#1E293B", "#7C3AED", "#06B6D4"],
            title="Feature Importance Treemap")
        fig_tree.update_layout(height=400, paper_bgcolor="rgba(0,0,0,0)", font_color="#F8FAFC",
            margin=dict(t=50, b=10, l=10, r=10))
        fig_tree.update_traces(textinfo="label+percent entry")
        st.plotly_chart(fig_tree, use_container_width=True)

        section_header("🔗 Feature Relationship Analysis")
        corr_df = _compute_corr_matrix(df)
        fig_corr = px.imshow(corr_df,
            color_continuous_scale=["#EF4444", "#1E293B", "#06B6D4"],
            zmin=-1, zmax=1, text_auto=".2f",
            title="Feature Correlation Matrix", aspect="auto")
        fig_corr.update_layout(height=520, paper_bgcolor="rgba(0,0,0,0)", font_color="#F8FAFC",
            coloraxis_colorbar=dict(tickfont=dict(color="#94A3B8")))
        st.plotly_chart(fig_corr, use_container_width=True)

        sample = _scatter_sample(rated)
        fig_scatter = px.scatter(sample, x="Votes", y="Aggregate rating", color="Price range",
            size="Average Cost for two", hover_data=["Restaurant Name", "City", "Primary Cuisine"],
            color_continuous_scale=["#7C3AED", "#06B6D4", "#F59E0B", "#EF4444"],
            title="Votes vs Rating (sized by Average Cost)", opacity=0.65, log_x=True)
        fig_scatter.update_layout(height=420, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", font_color="#F8FAFC",
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)"))
        st.plotly_chart(fig_scatter, use_container_width=True)

        top6 = feat_imp.head(6)
        fig_radar = go.Figure(go.Scatterpolar(
            r=top6["Importance %"].tolist() + [top6["Importance %"].iloc[0]],
            theta=top6["Label"].tolist() + [top6["Label"].iloc[0]],
            fill="toself", fillcolor="rgba(124,58,237,0.20)",
            line=dict(color="#7C3AED", width=2), name="Importance",
        ))
        fig_radar.update_layout(
            polar=dict(bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, gridcolor="rgba(255,255,255,0.1)",
                    tickfont=dict(color="#94A3B8", size=10)),
                angularaxis=dict(gridcolor="rgba(255,255,255,0.1)",
                    tickfont=dict(color="#94A3B8", size=11))),
            title="Top 6 Features – Radar View", height=420,
            paper_bgcolor="rgba(0,0,0,0)", font_color="#F8FAFC")
        st.plotly_chart(fig_radar, use_container_width=True)

        fig_violin = px.violin(rated[rated["Price range"].isin([1, 2, 3, 4])],
            x="Price range", y="Aggregate rating", color="Price range",
            box=True, points="outliers",
            color_discrete_sequence=["#7C3AED", "#06B6D4", "#F59E0B", "#EF4444"],
            title="Rating Distribution by Price Range",
            labels={"Price range": "Price Range", "Aggregate rating": "Rating"},
            category_orders={"Price range": [1, 2, 3, 4]})
        fig_violin.update_layout(height=380, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", font_color="#F8FAFC",
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)",
                ticktext=["Budget", "Affordable", "Premium", "Luxury"], tickvals=[1, 2, 3, 4]),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)"))
        st.plotly_chart(fig_violin, use_container_width=True)


    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 3 – TOP RESTAURANTS
    # ══════════════════════════════════════════════════════════════════════════
    elif section_choice == "🏆 Top Restaurants":

        st.markdown("""
        <div style="background:rgba(245,158,11,0.06);border:1px solid rgba(245,158,11,0.15);
             border-radius:14px;padding:16px 22px;margin-bottom:22px;
             position:relative;z-index:0;;">
          <div style="font-family:'DM Sans',sans-serif;font-size:0.82rem;color:#94A3B8;line-height:1.6;">
            <strong style="color:#FCD34D;">Top Restaurant Rankings</strong> —
            Highest-rated (min 50 votes) · Most voted · Best value composite score.
            Rankings are computed from the live filtered dataset.
          </div>
        </div>
        """, unsafe_allow_html=True)

        section_header("🏆 Top Restaurant Rankings")

        tab1, tab2, tab3 = st.tabs(["⭐ Top Rated", "🗳️ Most Voted", "💎 Best Value"])

        with tab1:
            top_rated = (
                rated[rated["Votes"] >= 50]
                .nlargest(25, "Aggregate rating")[
                    ["Restaurant Name", "City", "Country", "Primary Cuisine",
                     "Aggregate rating", "Votes", "Price range",
                     "Has Online delivery", "Has Table booking"]
                ]
                .reset_index(drop=True)
            )
            top_rated.index += 1

            sr1, sr2, sr3 = st.columns(3)
            with sr1:
                st.markdown(f"""
                <div style="background:rgba(9,14,25,0.88);border:1px solid rgba(255,255,255,0.12);
                     border-radius:14px;padding:18px 20px;position:relative;
                     z-index:0;;">
                  <div style="position:absolute;top:0;left:0;right:0;height:2px;background:#FCD34D;
                       opacity:0.8;border-radius:14px 14px 0 0;pointer-events:none;z-index:-1;"></div>
                  <div style="font-size:1.2rem;margin-bottom:6px;">⭐</div>
                  <div style="font-family:'JetBrains Mono',monospace;font-size:1.4rem;font-weight:700;color:#FCD34D;">
                    {top_rated['Aggregate rating'].max():.2f}★</div>
                  <div style="font-family:'DM Sans',sans-serif;font-size:0.67rem;color:#94A3B8;
                       text-transform:uppercase;letter-spacing:0.14em;margin-top:4px;">Peak Rating</div>
                </div>
                """, unsafe_allow_html=True)
            with sr2:
                st.markdown(f"""
                <div style="background:rgba(9,14,25,0.88);border:1px solid rgba(255,255,255,0.12);
                     border-radius:14px;padding:18px 20px;position:relative;
                     z-index:0;;">
                  <div style="position:absolute;top:0;left:0;right:0;height:2px;background:#A78BFA;
                       opacity:0.8;border-radius:14px 14px 0 0;pointer-events:none;z-index:-1;"></div>
                  <div style="font-size:1.2rem;margin-bottom:6px;">🏆</div>
                  <div style="font-family:'JetBrains Mono',monospace;font-size:1.4rem;font-weight:700;color:#A78BFA;">
                    {len(top_rated)}</div>
                  <div style="font-family:'DM Sans',sans-serif;font-size:0.67rem;color:#94A3B8;
                       text-transform:uppercase;letter-spacing:0.14em;margin-top:4px;">Top Performers</div>
                </div>
                """, unsafe_allow_html=True)
            with sr3:
                st.markdown(f"""
                <div style="background:rgba(9,14,25,0.88);border:1px solid rgba(255,255,255,0.12);
                     border-radius:14px;padding:18px 20px;position:relative;
                     z-index:0;;">
                  <div style="position:absolute;top:0;left:0;right:0;height:2px;background:#67E8F9;
                       opacity:0.8;border-radius:14px 14px 0 0;pointer-events:none;z-index:-1;"></div>
                  <div style="font-size:1.2rem;margin-bottom:6px;">🗳️</div>
                  <div style="font-family:'JetBrains Mono',monospace;font-size:1.4rem;font-weight:700;color:#67E8F9;">
                    50+</div>
                  <div style="font-family:'DM Sans',sans-serif;font-size:0.67rem;color:#94A3B8;
                       text-transform:uppercase;letter-spacing:0.14em;margin-top:4px;">Min Votes Threshold</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)

            fig_rated = go.Figure()
            fig_rated.add_trace(go.Bar(
                x=top_rated["Restaurant Name"].head(15),
                y=top_rated["Aggregate rating"].head(15),
                marker=dict(color=top_rated["Aggregate rating"].head(15),
                    colorscale=[[0, "#7C3AED"], [0.5, "#8B5CF6"], [1, "#06B6D4"]], showscale=False),
                text=top_rated["Aggregate rating"].head(15).apply(lambda x: f"{x:.2f}★"),
                textposition="outside", textfont=dict(color="#F8FAFC"),
            ))
            fig_rated.update_layout(title="Top 15 Highest-Rated Restaurants (min 50 votes)", height=420,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#F8FAFC",
                xaxis=dict(tickangle=-30, gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(range=[4.5, 5.2], gridcolor="rgba(255,255,255,0.05)"))
            st.plotly_chart(fig_rated, use_container_width=True)
            st.dataframe(top_rated, use_container_width=True)

        with tab2:
            top_voted = (
                rated.nlargest(25, "Votes")[
                    ["Restaurant Name", "City", "Country", "Primary Cuisine",
                     "Aggregate rating", "Votes", "Price range",
                     "Has Online delivery", "Has Table booking"]
                ]
                .reset_index(drop=True)
            )
            top_voted.index += 1

            fig_voted = px.bar(top_voted.head(15), x="Restaurant Name", y="Votes",
                color="Aggregate rating", color_continuous_scale=["#EF4444", "#F59E0B", "#06B6D4"],
                text="Votes", title="Top 15 Most-Voted Restaurants")
            fig_voted.update_traces(texttemplate="%{text:,}", textposition="outside")
            fig_voted.update_layout(height=420, paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)", font_color="#F8FAFC",
                xaxis=dict(tickangle=-30, gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)"))
            st.plotly_chart(fig_voted, use_container_width=True)
            st.dataframe(top_voted, use_container_width=True)

        with tab3:
            value_df = rated[(rated["Average Cost for two"] > 0) & (rated["Votes"] >= 20)].copy()
            value_df["Value Score"] = (
                value_df["Aggregate rating"]
                * np.log1p(value_df["Votes"])
                / np.log1p(value_df["Average Cost for two"])
            ).round(4)
            top_value = (
                value_df.nlargest(25, "Value Score")[
                    ["Restaurant Name", "City", "Country", "Primary Cuisine",
                     "Aggregate rating", "Votes", "Average Cost for two",
                     "Value Score", "Has Online delivery"]
                ]
                .reset_index(drop=True)
            )
            top_value.index += 1

            fig_value = px.scatter(top_value, x="Average Cost for two", y="Aggregate rating",
                size="Votes", color="Value Score", hover_data=["Restaurant Name", "City"],
                color_continuous_scale=["#1E293B", "#7C3AED", "#06B6D4"],
                title="Best Value: Rating vs Cost (bubble = vote volume)", log_x=True)
            fig_value.update_layout(height=420, paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)", font_color="#F8FAFC",
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)"))
            st.plotly_chart(fig_value, use_container_width=True)
            st.dataframe(top_value, use_container_width=True)


    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 4 – EXECUTIVE SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    elif section_choice == "📋 Executive Summary":

        st.markdown("""
        <div style="background:rgba(6,182,212,0.06);border:1px solid rgba(6,182,212,0.15);
             border-radius:14px;padding:16px 22px;margin-bottom:22px;
             position:relative;z-index:0;;">
          <div style="font-family:'DM Sans',sans-serif;font-size:0.82rem;color:#94A3B8;line-height:1.6;">
            <strong style="color:#67E8F9;">Executive Summary</strong> —
            Dynamic data-driven report generated from the full restaurant dataset.
            Market intelligence · Best performing cities · Pricing trends · Cuisine engagement.
          </div>
        </div>
        """, unsafe_allow_html=True)

        section_header("📋 Executive Summary")

        with st.spinner("📊 Generating executive report…"):
            summary = generate_executive_summary(df)

        ss = summary["summary_stats"]

        sc1, sc2, sc3, sc4, sc5 = st.columns(5)
        card_data = [
            (sc1, "🍽️", "Total Restaurants", f"{ss['total_restaurants']:,}", "#A78BFA"),
            (sc2, "⭐", "Rated Restaurants", f"{ss['rated_restaurants']:,}", "#FCD34D"),
            (sc3, "📊", "Global Avg Rating", f"{ss['avg_rating']}★", "#67E8F9"),
            (sc4, "🚴", "Delivery Lift", f"{ss['delivery_lift']:+.3f}★", "#6EE7B7"),
            (sc5, "🏙️", "Top City", str(ss['top_city']), "#F9A8D4"),
        ]
        for col_, icon, label, value, accent in card_data:
            with col_:
                st.markdown(f"""
                <div style="background:rgba(9,14,25,0.88);border:1px solid rgba(255,255,255,0.12);
                     border-radius:16px;padding:20px 18px;position:relative;margin-bottom:4px;
                     z-index:0;;">
                  <div style="position:absolute;top:0;left:0;right:0;height:2px;
                       background:{accent};opacity:0.8;border-radius:16px 16px 0 0;
                       pointer-events:none;z-index:-1;"></div>
                  <div style="font-size:1.3rem;margin-bottom:8px;">{icon}</div>
                  <div style="font-family:'Sora',sans-serif;font-size:1.5rem;font-weight:800;
                       color:{accent};letter-spacing:-0.03em;line-height:1;">{value}</div>
                  <div style="font-family:'DM Sans',sans-serif;font-size:0.63rem;font-weight:700;
                       color:#94A3B8;text-transform:uppercase;letter-spacing:0.16em;margin-top:6px;">{label}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
        section_header("🔍 Market Intelligence")

        severity_colors = {
            "success":  "#06B6D4", "info":     "#7C3AED",
            "warning":  "#F59E0B", "critical": "#EF4444",
        }
        for ins in summary["market_insights"]:
            c    = severity_colors.get(ins["type"], "#7C3AED")
            text = ins["text"]
            parts = text.split("**")
            rendered = ""
            for i, part in enumerate(parts):
                rendered += f"<strong>{part}</strong>" if i % 2 == 1 else part
            st.markdown(f"""
            <div style="background:{c}10;border:1px solid {c}22;border-left:3px solid {c};
                 border-radius:12px;padding:14px 20px;margin-bottom:10px;
                 color:#CBD5E1;font-size:0.88rem;line-height:1.65;font-family:'DM Sans',sans-serif;
                 position:relative;z-index:0;;">
              {ins['icon']} {rendered}
            </div>
            """, unsafe_allow_html=True)

        section_header("🏙️ Best Performing Cities")
        best_cities = summary["best_cities"].head(10)
        fig_cities  = go.Figure()
        fig_cities.add_trace(go.Bar(x=best_cities["City"], y=best_cities["avg_rating"],
            name="Avg Rating", marker_color="#7C3AED", yaxis="y",
            text=best_cities["avg_rating"].round(2), textposition="outside"))
        fig_cities.add_trace(go.Scatter(x=best_cities["City"], y=best_cities["total_votes"],
            name="Total Votes", mode="lines+markers",
            line=dict(color="#06B6D4", width=2.5), marker=dict(size=7), yaxis="y2"))
        fig_cities.update_layout(title="Top Cities by Average Rating (bars) & Total Votes (line)", height=400,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#F8FAFC",
            xaxis=dict(tickangle=-25, gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title="Avg Rating", gridcolor="rgba(255,255,255,0.05)"),
            yaxis2=dict(title="Total Votes", overlaying="y", side="right"),
            legend=dict(bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig_cities, use_container_width=True)

        section_header("💰 Pricing Trends")
        pricing   = summary["pricing_trends"]
        fig_price = go.Figure()
        fig_price.add_trace(go.Bar(x=pricing["Price Label"], y=pricing["avg_rating"],
            name="Avg Rating", marker_color="#7C3AED",
            text=pricing["avg_rating"].round(2), textposition="outside"))
        fig_price.add_trace(go.Scatter(x=pricing["Price Label"], y=pricing["avg_votes"],
            name="Avg Votes", mode="lines+markers",
            line=dict(color="#F59E0B", width=2.5), yaxis="y2"))
        fig_price.update_layout(title="Rating & Vote Engagement by Price Segment", height=380,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#F8FAFC",
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title="Avg Rating", gridcolor="rgba(255,255,255,0.05)"),
            yaxis2=dict(title="Avg Votes", overlaying="y", side="right"),
            legend=dict(bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig_price, use_container_width=True)

        section_header("🍴 Highest Engagement Cuisines")
        cuisines = summary["highest_engagement_cuisines"]
        fig_cuis = px.scatter(cuisines.head(15), x="avg_rating", y="avg_votes",
            size="count", color="engagement_score", text="Primary Cuisine",
            color_continuous_scale=["#1E293B", "#7C3AED", "#06B6D4"],
            title="Cuisine Engagement Matrix (size = restaurant count)",
            hover_data=["total_votes"])
        fig_cuis.update_traces(textposition="top center", textfont=dict(size=10, color="#F8FAFC"))
        fig_cuis.update_layout(height=480, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", font_color="#F8FAFC",
            xaxis=dict(title="Average Rating", gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title="Average Votes", gridcolor="rgba(255,255,255,0.05)"))
        st.plotly_chart(fig_cuis, use_container_width=True)

        section_header("🏆 Top Composite Performers")
        tp = summary["top_performers"].head(10)
        st.dataframe(tp[["Restaurant Name", "City", "Primary Cuisine",
            "Aggregate rating", "Votes", "composite_score",
            "Has Online delivery", "Has Table booking"]], use_container_width=True)


    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 5 – SMART AI INSIGHTS
    # ══════════════════════════════════════════════════════════════════════════
    elif section_choice == "💡 Smart AI Insights":

        # FIX: Added z-index:0 + . Removed position:relative
        # from the decorative top-line div and set pointer-events:none + z-index:-1.
        st.markdown("""
        <div style="background:linear-gradient(135deg,rgba(124,58,237,0.10),rgba(6,182,212,0.07));
             border:1px solid rgba(124,58,237,0.22);border-radius:22px;
             padding:28px 32px;margin-bottom:28px;position:relative;
             z-index:0;;overflow:visible;">
          <div style="position:absolute;top:0;left:0;right:0;height:1px;
               background:linear-gradient(90deg,transparent,rgba(124,58,237,0.45),rgba(6,182,212,0.30),transparent);
               pointer-events:none;z-index:-1;"></div>
          <div style="display:inline-flex;align-items:center;gap:8px;
               background:rgba(124,58,237,0.10);border:1px solid rgba(124,58,237,0.25);
               border-radius:999px;padding:5px 14px;font-size:0.67rem;color:#A78BFA;
               letter-spacing:.16em;text-transform:uppercase;font-weight:700;
               margin-bottom:14px;font-family:'DM Sans',sans-serif;">
            ✦ AI Intelligence Layer · Pattern Recognition Engine
          </div>
          <p style="color:#CBD5E1;margin:0;line-height:1.7;font-size:0.90rem;
               font-family:'DM Sans',sans-serif;">
            These insights are automatically derived from statistical analysis of real restaurant data —
            no manual curation. Each finding is backed by quantitative evidence from
            <strong style="color:#F1F5F9;">9,551 restaurants</strong>.
          </p>
        </div>
        """, unsafe_allow_html=True)

        section_header("💡 Smart AI Insights")

        city_delivery, prem_cuisine_city, service_agg, cuisine_diversity = (
            _compute_insights(rated)
        )

        st.markdown("""
        <div style="display:flex;align-items:center;gap:14px;margin-bottom:18px;">
          <div style="flex:1;height:1px;background:linear-gradient(90deg,rgba(124,58,237,0.30),transparent);"></div>
          <span style="font-family:'DM Sans',sans-serif;font-size:0.65rem;font-weight:700;
                color:#64748B;letter-spacing:0.22em;text-transform:uppercase;">Insight 1</span>
          <div style="flex:1;height:1px;background:linear-gradient(90deg,transparent,rgba(6,182,212,0.25));"></div>
        </div>
        <h3 style="font-family:'Sora',sans-serif;font-size:1.1rem;font-weight:700;
             color:#F1F5F9;margin:0 0 12px;letter-spacing:-0.02em;">
          🚚 Online Delivery Boosts Ratings Across All Markets
        </h3>
        <div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:18px;">
          <span class="insight-chip">AI Finding</span>
          <span class="insight-chip">Delivery Impact</span>
          <span class="insight-chip">City Analysis</span>
        </div>
        """, unsafe_allow_html=True)

        fig_del = go.Figure()
        fig_del.add_trace(go.Bar(name="No Delivery", x=city_delivery.index[:12],
            y=city_delivery["No Delivery"][:12], marker_color="rgba(239,68,68,0.65)"))
        fig_del.add_trace(go.Bar(name="With Delivery", x=city_delivery.index[:12],
            y=city_delivery["With Delivery"][:12], marker_color="rgba(6,182,212,0.75)"))
        fig_del.update_layout(barmode="group", title="Delivery vs No-Delivery Rating by City", height=380,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#F8FAFC",
            xaxis=dict(tickangle=-20, gridcolor="rgba(255,255,255,0.04)"),
            yaxis=dict(title="Avg Rating", gridcolor="rgba(255,255,255,0.04)"),
            legend=dict(bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig_del, use_container_width=True)

        best_lift_city = city_delivery["Lift"].idxmax()
        best_lift_val  = city_delivery["Lift"].max()
        st.info(f"💡 **Key Finding:** Restaurants with online delivery consistently outperform "
                f"those without. The biggest effect is in **{best_lift_city}** where delivery "
                f"adds **{best_lift_val:+.2f}★** on average.")

        st.markdown("<div style='margin:28px 0;height:1px;background:rgba(255,255,255,0.06);'></div>", unsafe_allow_html=True)

        st.markdown("""
        <div style="display:flex;align-items:center;gap:14px;margin-bottom:18px;">
          <div style="flex:1;height:1px;background:linear-gradient(90deg,rgba(124,58,237,0.30),transparent);"></div>
          <span style="font-family:'DM Sans',sans-serif;font-size:0.65rem;font-weight:700;
                color:#64748B;letter-spacing:0.22em;text-transform:uppercase;">Insight 2</span>
          <div style="flex:1;height:1px;background:linear-gradient(90deg,transparent,rgba(6,182,212,0.25));"></div>
        </div>
        <h3 style="font-family:'Sora',sans-serif;font-size:1.1rem;font-weight:700;
             color:#F1F5F9;margin:0 0 12px;letter-spacing:-0.02em;">
          💎 Premium Cuisines Dominate Specific Markets
        </h3>
        <div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:18px;">
          <span class="insight-chip">Premium Segment</span>
          <span class="insight-chip">Cuisine Intelligence</span>
        </div>
        """, unsafe_allow_html=True)

        fig_prem = px.bar(prem_cuisine_city, x="Primary Cuisine", y="avg_rating",
            color="avg_cost", color_continuous_scale=["#7C3AED", "#F59E0B", "#EF4444"],
            text="avg_rating", title="Top Premium Cuisines by Avg Rating (Price Range 3–4 only)")
        fig_prem.update_traces(texttemplate="%{text:.2f}★", textposition="outside")
        fig_prem.update_layout(height=380, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", font_color="#F8FAFC",
            xaxis=dict(tickangle=-25, gridcolor="rgba(255,255,255,0.04)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
            coloraxis_colorbar=dict(title="Avg Cost (₹)", tickfont=dict(color="#94A3B8")))
        st.plotly_chart(fig_prem, use_container_width=True)

        st.markdown("<div style='margin:28px 0;height:1px;background:rgba(255,255,255,0.06);'></div>", unsafe_allow_html=True)

        st.markdown("""
        <div style="display:flex;align-items:center;gap:14px;margin-bottom:18px;">
          <div style="flex:1;height:1px;background:linear-gradient(90deg,rgba(124,58,237,0.30),transparent);"></div>
          <span style="font-family:'DM Sans',sans-serif;font-size:0.65rem;font-weight:700;
                color:#64748B;letter-spacing:0.22em;text-transform:uppercase;">Insight 3</span>
          <div style="flex:1;height:1px;background:linear-gradient(90deg,transparent,rgba(6,182,212,0.25));"></div>
        </div>
        <h3 style="font-family:'Sora',sans-serif;font-size:1.1rem;font-weight:700;
             color:#F1F5F9;margin:0 0 12px;letter-spacing:-0.02em;">
          🎯 Full-Service Restaurants Lead in Every Metric
        </h3>
        <div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:18px;">
          <span class="insight-chip">Service Strategy</span>
          <span class="insight-chip">Operational Excellence</span>
        </div>
        """, unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            fig_svc_r = px.bar(service_agg, x="Service Level", y="avg_rating",
                color="avg_rating", color_continuous_scale=["#EF4444", "#F59E0B", "#06B6D4"],
                text="avg_rating", title="Average Rating by Service Level")
            fig_svc_r.update_traces(texttemplate="%{text:.2f}★", textposition="outside")
            fig_svc_r.update_layout(height=320, showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#F8FAFC",
                xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.04)"))
            st.plotly_chart(fig_svc_r, use_container_width=True)

        with col_b:
            fig_svc_v = px.bar(service_agg, x="Service Level", y="avg_votes",
                color="avg_votes", color_continuous_scale=["#1E293B", "#7C3AED", "#06B6D4"],
                text="avg_votes", title="Average Votes by Service Level")
            fig_svc_v.update_traces(texttemplate="%{text:.0f}", textposition="outside")
            fig_svc_v.update_layout(height=320, showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#F8FAFC",
                xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.04)"))
            st.plotly_chart(fig_svc_v, use_container_width=True)

        st.markdown("<div style='margin:28px 0;height:1px;background:rgba(255,255,255,0.06);'></div>", unsafe_allow_html=True)

        st.markdown("""
        <div style="display:flex;align-items:center;gap:14px;margin-bottom:18px;">
          <div style="flex:1;height:1px;background:linear-gradient(90deg,rgba(124,58,237,0.30),transparent);"></div>
          <span style="font-family:'DM Sans',sans-serif;font-size:0.65rem;font-weight:700;
                color:#64748B;letter-spacing:0.22em;text-transform:uppercase;">Insight 4</span>
          <div style="flex:1;height:1px;background:linear-gradient(90deg,transparent,rgba(6,182,212,0.25));"></div>
        </div>
        <h3 style="font-family:'Sora',sans-serif;font-size:1.1rem;font-weight:700;
             color:#F1F5F9;margin:0 0 12px;letter-spacing:-0.02em;">
          🍜 Cuisine Diversity & Its Effect on Ratings
        </h3>
        <div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:18px;">
          <span class="insight-chip">Menu Strategy</span>
          <span class="insight-chip">Diversity Analysis</span>
        </div>
        """, unsafe_allow_html=True)

        fig_div = px.bar(cuisine_diversity, x="Cuisine_Count_Grp", y="avg_rating",
            color="count", color_continuous_scale=["#1E293B", "#8B5CF6", "#06B6D4"],
            text="avg_rating", title="Avg Rating by Number of Cuisines Offered",
            labels={"Cuisine_Count_Grp": "Cuisine Count", "avg_rating": "Avg Rating"})
        fig_div.update_traces(texttemplate="%{text:.3f}★", textposition="outside")
        fig_div.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", font_color="#F8FAFC",
            xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
            coloraxis_colorbar=dict(title="# Restaurants", tickfont=dict(color="#94A3B8")))
        st.plotly_chart(fig_div, use_container_width=True)

        st.markdown("""
        <div style="display:flex;align-items:center;gap:14px;margin:24px 0 18px;">
          <div style="flex:1;height:1px;background:linear-gradient(90deg,rgba(124,58,237,0.30),transparent);"></div>
          <span style="font-family:'DM Sans',sans-serif;font-size:0.65rem;font-weight:700;
                color:#64748B;letter-spacing:0.22em;text-transform:uppercase;">AI Pattern Summary</span>
          <div style="flex:1;height:1px;background:linear-gradient(90deg,transparent,rgba(6,182,212,0.25));"></div>
        </div>
        """, unsafe_allow_html=True)

        fs_row    = service_agg[service_agg["Service Level"] == "Full Service"]["avg_rating"].values
        basic_row = service_agg[service_agg["Service Level"] == "Basic"]["avg_rating"].values
        fs_lift   = float(fs_row[0] - basic_row[0]) if len(fs_row) and len(basic_row) else 0.0

        insights_summary = [
            ("🚚", "Delivery Effect",    f"Online delivery adds +{city_delivery['Lift'].mean():.2f}★ avg across top cities", "#67E8F9"),
            ("💎", "Premium Leader",     f"{prem_cuisine_city.iloc[0]['Primary Cuisine']} tops premium cuisine charts", "#FCD34D"),
            ("🎯", "Full Service Lift",  f"Full-service restaurants score {fs_lift:+.2f}★ vs basic-only", "#A78BFA"),
            ("🍜", "Diversity Sweet Spot","Dual-cuisine restaurants slightly outperform single-cuisine peers", "#6EE7B7"),
        ]
        for icon, title, body, accent in insights_summary:
            st.markdown(f"""
            <div style="background:rgba(9,14,25,0.85);border:1px solid rgba(255,255,255,0.12);
                 border-left:3px solid {accent};border-radius:14px;padding:18px 22px;
                 margin-bottom:12px;display:flex;gap:16px;align-items:center;
                 position:relative;z-index:0;;">
              <span style="font-size:1.6rem;flex-shrink:0;">{icon}</span>
              <div>
                <div style="font-family:'Sora',sans-serif;font-weight:700;color:#F1F5F9;
                     margin-bottom:4px;font-size:0.95rem;">{title}</div>
                <div style="color:#CBD5E1;font-size:0.86rem;font-family:'DM Sans',sans-serif;
                     line-height:1.6;">{body}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:48px;padding:24px 0 8px;border-top:1px solid rgba(255,255,255,0.05);
     display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;">
  <div style="font-family:'DM Sans',sans-serif;color:#334155;font-size:0.70rem;">
    RestaurantIQ · Level 3 Task 3 · Cognifyz Analytics Challenge
  </div>
  <div style="font-family:'JetBrains Mono',monospace;font-size:0.67rem;color:#334155;">
    AI Prediction · Feature Analysis · Executive Summary · Smart Insights
  </div>
  <div style="font-family:'DM Sans',sans-serif;color:#7C3AED;font-size:0.70rem;font-weight:600;">
    Enterprise Analytics Edition
  </div>
</div>
""", unsafe_allow_html=True)
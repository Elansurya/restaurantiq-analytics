import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.components import inject_global_css, section_header
from src.preprocessing import load_and_preprocess
from src.recommendation_system import (
    RestaurantRecommender,
    get_cuisine_recommendations,
    get_pricing_recommendation,
)
from src.success_score import (
    compute_success_scores,
    get_gauge_data,
    get_pillar_averages,
    get_radar_data,
    get_score_distribution,
    get_tier_radar_averages,
    get_top_performers,
    PILLAR_WEIGHTS,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: 8-digit hex → rgba()
# ─────────────────────────────────────────────────────────────────────────────
def hex8_to_rgba(hex8: str, fallback_alpha: float = 0.2) -> str:
    h = hex8.lstrip("#")
    if len(h) == 8:
        r, g, b, a = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), int(h[6:8], 16)
        return f"rgba({r},{g},{b},{round(a / 255, 3)})"
    if len(h) == 6:
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{fallback_alpha})"
    return hex8


inject_global_css()

# ─────────────────────────────────────────────────────────────────────────────
# FIX: Selected-value visibility for selectboxes (Price Range, Country, City)
# Root cause: inject_global_css() applies overflow:hidden to card/column
# containers (used for hero banners & summary cards). The BaseWeb <select>
# renders its selected-value node relative to that clipping ancestor, so the
# chosen value gets clipped/hidden behind the hero header. This override
# restores overflow:visible on Streamlit containers and forces the selected
# value's text color/z-index so it always renders above the hero section.
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# PAGE HERO BANNER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="
    background: linear-gradient(135deg, rgba(108,99,255,0.18) 0%, rgba(0,194,168,0.12) 50%, rgba(236,72,153,0.10) 100%);
    border: 1px solid rgba(108,99,255,0.28);
    border-radius: 20px;
    padding: 36px 40px 30px;
    margin-bottom: 28px;
">
  <div style="display:flex; align-items:flex-start; gap:18px;">
    <div style="font-size:3rem; line-height:1; flex-shrink:0;">🎯</div>
    <div style="flex:1;">
      <div style="
          font-size:2rem; font-weight:900; line-height:1.15;
          background: linear-gradient(90deg, #F8FAFC 30%, #00C2A8 100%);
          -webkit-background-clip: text; -webkit-text-fill-color: transparent;
          background-clip: text; margin-bottom:8px;
      ">Business Success Score &amp; Recommendations</div>
      <div style="color:#94A3B8; font-size:0.92rem; line-height:1.7; max-width:680px;">
        A composite intelligence engine that scores every restaurant across five performance pillars —
        rating quality, customer engagement, market positioning, service excellence, and value delivery.
        Use the gauge deep-dives, tier radar comparisons, and AI-powered recommendation engine to
        identify winners, diagnose laggards, and guide new operators to market-fit decisions.
      </div>
      <div style="display:flex; gap:10px; margin-top:16px; flex-wrap:wrap;">
        <span style="background:rgba(108,99,255,0.15); border:1px solid rgba(108,99,255,0.35);
              color:#A5B4FC; border-radius:999px; padding:4px 14px; font-size:0.75rem; font-weight:600;">
          Composite Scoring
        </span>
        <span style="background:rgba(0,194,168,0.12); border:1px solid rgba(0,194,168,0.3);
              color:#5EEAD4; border-radius:999px; padding:4px 14px; font-size:0.75rem; font-weight:600;">
          Gauge Charts
        </span>
        <span style="background:rgba(236,72,153,0.10); border:1px solid rgba(236,72,153,0.28);
              color:#F9A8D4; border-radius:999px; padding:4px 14px; font-size:0.75rem; font-weight:600;">
          Radar Analysis
        </span>
        <span style="background:rgba(245,158,11,0.10); border:1px solid rgba(245,158,11,0.28);
              color:#FCD34D; border-radius:999px; padding:4px 14px; font-size:0.75rem; font-weight:600;">
          AI Recommendations
        </span>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _load_raw() -> pd.DataFrame:
    return load_and_preprocess()


@st.cache_data(show_spinner=False)
def _load_scored(raw: pd.DataFrame) -> pd.DataFrame:
    return compute_success_scores(raw).copy()


@st.cache_data(show_spinner=False)
def _build_recommender(raw: pd.DataFrame) -> "RestaurantRecommender":
    rec = RestaurantRecommender()
    rec.fit(raw)
    return rec


raw_df      = _load_raw()
scored_df   = _load_scored(raw_df)
recommender = _build_recommender(raw_df)


# ─────────────────────────────────────────────────────────────────────────────
# Session-state initialisation
# ─────────────────────────────────────────────────────────────────────────────
restaurant_names_sorted: list[str] = sorted(
    scored_df["Restaurant Name"].dropna().unique().tolist()
)

if "ss_selected_restaurant" not in st.session_state:
    st.session_state["ss_selected_restaurant"] = restaurant_names_sorted[0]

if st.session_state["ss_selected_restaurant"] not in restaurant_names_sorted:
    st.session_state["ss_selected_restaurant"] = restaurant_names_sorted[0]


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 – Platform Score Overview
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="display:flex; align-items:center; gap:12px; margin: 32px 0 4px;">
  <div style="width:4px; height:32px; border-radius:2px;
       background: linear-gradient(180deg, #6C63FF, #00C2A8);"></div>
  <div>
    <div style="font-size:1.3rem; font-weight:800; color:#F8FAFC; line-height:1.2;">
      📊 Platform Success Score Overview
    </div>
    <div style="font-size:0.8rem; color:#64748B; margin-top:2px;">
      Aggregate performance metrics across all restaurants in the dataset
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

dist    = get_score_distribution(scored_df)
pillars = get_pillar_averages(scored_df)

# Executive KPI summary cards
avg_score  = scored_df['Success_Score'].mean()
top_count  = (scored_df['Success_Tier'] == 'Top Performer').sum()
struggling = (scored_df['Success_Tier'] == 'Struggling').sum()
std_dev    = scored_df['Success_Score'].std()
median_s   = scored_df['Success_Score'].median()

st.markdown(f"""
<div style="display:grid; grid-template-columns: repeat(5, 1fr); gap:12px; margin: 16px 0 24px;">
  <div style="background:rgba(108,99,255,0.10); border:1px solid rgba(108,99,255,0.28);
       border-radius:14px; padding:18px 16px; text-align:center;">
    <div style="font-size:1.6rem; font-weight:900; color:#A5B4FC;">{avg_score:.1f}</div>
    <div style="font-size:0.7rem; color:#64748B; margin-top:4px; text-transform:uppercase; letter-spacing:.06em;">Avg Success Score</div>
    <div style="font-size:0.75rem; color:#6C63FF; margin-top:2px;">out of 100</div>
  </div>
  <div style="background:rgba(0,194,168,0.08); border:1px solid rgba(0,194,168,0.25);
       border-radius:14px; padding:18px 16px; text-align:center;">
    <div style="font-size:1.6rem; font-weight:900; color:#5EEAD4;">{top_count:,}</div>
    <div style="font-size:0.7rem; color:#64748B; margin-top:4px; text-transform:uppercase; letter-spacing:.06em;">Top Performers</div>
    <div style="font-size:0.75rem; color:#00C2A8; margin-top:2px;">score ≥ 80</div>
  </div>
  <div style="background:rgba(239,68,68,0.08); border:1px solid rgba(239,68,68,0.22);
       border-radius:14px; padding:18px 16px; text-align:center;">
    <div style="font-size:1.6rem; font-weight:900; color:#FCA5A5;">{struggling:,}</div>
    <div style="font-size:0.7rem; color:#64748B; margin-top:4px; text-transform:uppercase; letter-spacing:.06em;">Struggling</div>
    <div style="font-size:0.75rem; color:#EF4444; margin-top:2px;">score &lt; 30</div>
  </div>
  <div style="background:rgba(245,158,11,0.08); border:1px solid rgba(245,158,11,0.22);
       border-radius:14px; padding:18px 16px; text-align:center;">
    <div style="font-size:1.6rem; font-weight:900; color:#FCD34D;">{std_dev:.1f}</div>
    <div style="font-size:0.7rem; color:#64748B; margin-top:4px; text-transform:uppercase; letter-spacing:.06em;">Score Std Dev</div>
    <div style="font-size:0.75rem; color:#F59E0B; margin-top:2px;">spread</div>
  </div>
  <div style="background:rgba(16,185,129,0.08); border:1px solid rgba(16,185,129,0.22);
       border-radius:14px; padding:18px 16px; text-align:center;">
    <div style="font-size:1.6rem; font-weight:900; color:#6EE7B7;">{median_s:.1f}</div>
    <div style="font-size:0.7rem; color:#64748B; margin-top:4px; text-transform:uppercase; letter-spacing:.06em;">Median Score</div>
    <div style="font-size:0.75rem; color:#10B981; margin-top:2px;">midpoint</div>
  </div>
</div>
""", unsafe_allow_html=True)

col_dist, col_pil = st.columns(2)

with col_dist:
    tier_colors = {
        "Top Performer":  "#00C2A8",
        "Performing":     "#6C63FF",
        "Average":        "#F59E0B",
        "Below Average":  "#F97316",
        "Struggling":     "#EF4444",
    }
    fig_donut = px.pie(
        dist, names="Tier", values="Count",
        hole=0.55,
        title="Success Tier Distribution",
        color="Tier",
        color_discrete_map=tier_colors,
    )
    fig_donut.update_traces(textposition="outside", textinfo="label+percent")
    fig_donut.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", font_color="#F8FAFC", showlegend=False,
    )
    st.plotly_chart(fig_donut, use_container_width=True, key="ss_tier_donut")

with col_pil:
    cats        = pillars["Pillar"].tolist()
    vals        = pillars["Avg_Score"].tolist()
    cats_closed = cats + [cats[0]]
    vals_closed = vals + [vals[0]]

    fig_radar = go.Figure(go.Scatterpolar(
        r=vals_closed,
        theta=cats_closed,
        fill="toself",
        line_color="#6C63FF",
        fillcolor="rgba(108,99,255,0.25)",
        name="All Restaurants",
    ))
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100],
                            gridcolor="#334155", linecolor="#334155"),
            angularaxis=dict(gridcolor="#334155"),
            bgcolor="rgba(0,0,0,0)",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#F8FAFC",
        title="Avg Pillar Scores (All Restaurants)",
        showlegend=False,
    )
    st.plotly_chart(fig_radar, use_container_width=True, key="ss_platform_radar")

st.markdown("""
<div style="
    background: linear-gradient(90deg, rgba(108,99,255,0.12), rgba(0,194,168,0.08));
    border: 1px solid rgba(108,99,255,0.22);
    border-left: 4px solid #6C63FF;
    border-radius: 12px;
    padding: 14px 20px;
    margin-bottom: 12px;
    display:flex; align-items:center; gap:14px;
">
  <span style="font-size:1.5rem; flex-shrink:0;">💡</span>
  <div style="color:#CBD5E1; font-size:0.87rem; line-height:1.6;">
    <strong style="color:#A5B4FC;">How to read this chart:</strong>
    The distribution reveals the spread of business health across the platform.
    A bell-curve shape indicates a balanced ecosystem; a left skew signals systemic underperformance.
  </div>
</div>
""", unsafe_allow_html=True)

fig_hist = px.histogram(
    scored_df, x="Success_Score", nbins=50,
    color_discrete_sequence=["#6C63FF"],
    title="Business Success Score Distribution",
)
fig_hist.update_layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font_color="#F8FAFC",
)
st.plotly_chart(fig_hist, use_container_width=True, key="ss_score_hist")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 – Individual Restaurant Gauge
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="display:flex; align-items:center; gap:12px; margin: 36px 0 4px;">
  <div style="width:4px; height:32px; border-radius:2px;
       background: linear-gradient(180deg, #00C2A8, #6C63FF);"></div>
  <div>
    <div style="font-size:1.3rem; font-weight:800; color:#F8FAFC; line-height:1.2;">
      🔍 Restaurant Deep Dive
    </div>
    <div style="font-size:0.8rem; color:#64748B; margin-top:2px;">
      Select any restaurant to view its success gauge, pillar breakdown, and weighted contribution scores
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

selected_name: str = st.selectbox(
    "Select a Restaurant",
    options=restaurant_names_sorted,
    key="ss_selected_restaurant",
)

_matches = scored_df[scored_df["Restaurant Name"] == selected_name]
if _matches.empty:
    st.warning(f"No data found for **{selected_name}**. Please select another restaurant.")
    st.stop()

sel_row = _matches.iloc[0].copy()

gauge = get_gauge_data(sel_row)
radar = get_radar_data(sel_row)

_n_cats    = len(radar.get("categories", []))
_n_vals    = len(radar.get("values", []))
_n_weights = len(radar.get("weights", []))

if not (_n_cats == _n_vals == _n_weights) or _n_cats == 0:
    st.error(
        f"Pillar data mismatch for **{selected_name}** "
        f"(categories={_n_cats}, values={_n_vals}, weights={_n_weights}). "
        "Check `get_radar_data()` in success_score.py."
    )
    st.stop()

# Restaurant summary card
tier_badge_colors = {
    "Top Performer":  ("#00C2A8", "rgba(0,194,168,0.12)"),
    "Performing":     ("#6C63FF", "rgba(108,99,255,0.12)"),
    "Average":        ("#F59E0B", "rgba(245,158,11,0.10)"),
    "Below Average":  ("#F97316", "rgba(249,115,22,0.10)"),
    "Struggling":     ("#EF4444", "rgba(239,68,68,0.10)"),
}
tier_label = gauge.get("label", "Unknown")
tc, tbg = tier_badge_colors.get(tier_label, ("#94A3B8", "rgba(148,163,184,0.10)"))

st.markdown(f"""
<div style="
    background: linear-gradient(135deg, rgba(30,41,59,0.9), rgba(15,23,42,0.95));
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 20px 24px;
    margin: 12px 0 20px;
    display:flex; align-items:center; gap:20px;
">
  <div style="flex:1;">
    <div style="font-size:1.15rem; font-weight:800; color:#F8FAFC; margin-bottom:6px;">
      {selected_name}
    </div>
    <div style="display:flex; gap:8px; flex-wrap:wrap; align-items:center;">
      <span style="background:{tbg}; border:1px solid {tc}40; color:{tc};
            border-radius:999px; padding:3px 12px; font-size:0.75rem; font-weight:700;">
        {tier_label}
      </span>
      <span style="color:#64748B; font-size:0.8rem;">
        Score: <strong style="color:#F8FAFC;">{gauge['score']:.1f} / 100</strong>
      </span>
    </div>
  </div>
  <div style="text-align:right; flex-shrink:0;">
    <div style="font-size:2.2rem; font-weight:900; color:{gauge['color']};">{gauge['score']:.0f}</div>
    <div style="font-size:0.7rem; color:#64748B; text-transform:uppercase; letter-spacing:.08em;">Success Score</div>
  </div>
</div>
""", unsafe_allow_html=True)

col_gauge, col_radar_ind = st.columns([1, 1])

with col_gauge:
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=gauge["score"],
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": f"{selected_name}<br><sub>{gauge['label']}</sub>",
               "font": {"color": "#F8FAFC", "size": 14}},
        delta={"reference": 50,
               "increasing": {"color": "#00C2A8"},
               "decreasing": {"color": "#EF4444"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1,
                     "tickcolor": "#94A3B8", "tickfont": {"color": "#94A3B8"}},
            "bar":  {"color": gauge["color"]},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 1,
            "bordercolor": "#334155",
            "steps": [
                {"range": [0,  30],  "color": "rgba(239,68,68,0.15)"},
                {"range": [30, 50],  "color": "rgba(249,115,22,0.15)"},
                {"range": [50, 65],  "color": "rgba(245,158,11,0.15)"},
                {"range": [65, 80],  "color": "rgba(108,99,255,0.15)"},
                {"range": [80, 100], "color": "rgba(0,194,168,0.15)"},
            ],
            "threshold": {
                "line":      {"color": "#F8FAFC", "width": 3},
                "thickness": 0.75,
                "value":     gauge["score"],
            },
        },
        number={"suffix": "/100",
                "font": {"color": gauge["color"], "size": 32}},
    ))
    fig_gauge.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#F8FAFC",
        height=340,
        margin=dict(t=60, b=20, l=20, r=20),
    )
    st.plotly_chart(fig_gauge, use_container_width=True, key="ss_individual_gauge")

with col_radar_ind:
    cats2   = radar["categories"]
    vals2   = radar["values"]
    cats2_c = cats2 + [cats2[0]]
    vals2_c = vals2 + [vals2[0]]

    fig_r2 = go.Figure(go.Scatterpolar(
        r=vals2_c,
        theta=cats2_c,
        fill="toself",
        line_color=gauge["color"],
        fillcolor=hex8_to_rgba(gauge["color"] + "33"),
        name=selected_name,
    ))
    fig_r2.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100],
                            gridcolor="#334155", linecolor="#334155"),
            angularaxis=dict(gridcolor="#334155"),
            bgcolor="rgba(0,0,0,0)",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#F8FAFC",
        title=f"Pillar Breakdown – {selected_name[:30]}",
        showlegend=False,
    )
    st.plotly_chart(fig_r2, use_container_width=True, key="ss_individual_radar")

st.markdown("""
<div style="font-size:0.78rem; font-weight:700; text-transform:uppercase;
     letter-spacing:.1em; color:#6C63FF; margin: 8px 0 8px; padding-left:4px;">
  Weighted Pillar Contribution Table
</div>
""", unsafe_allow_html=True)

pillar_detail = pd.DataFrame({
    "Pillar":       radar["categories"],
    "Score":        [round(v, 2) for v in radar["values"]],
    "Weight":       [f"{w * 100:.0f}%" for w in radar["weights"]],
    "Contribution": [round(v * w, 2)
                     for v, w in zip(radar["values"], radar["weights"])],
})
st.dataframe(pillar_detail, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 – Top Performers
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="display:flex; align-items:center; gap:12px; margin: 36px 0 4px;">
  <div style="width:4px; height:32px; border-radius:2px;
       background: linear-gradient(180deg, #F59E0B, #EC4899);"></div>
  <div>
    <div style="font-size:1.3rem; font-weight:800; color:#F8FAFC; line-height:1.2;">
      🏆 Top 20 Restaurants by Success Score
    </div>
    <div style="font-size:0.8rem; color:#64748B; margin-top:2px;">
      The platform's highest-performing venues ranked by composite success score
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

top_df = get_top_performers(scored_df, n=20)

if not top_df.empty and "Success_Score" in top_df.columns:
    top_avg = top_df["Success_Score"].mean()
    st.markdown(f"""
    <div style="
        background: linear-gradient(90deg, rgba(245,158,11,0.10), rgba(236,72,153,0.08));
        border: 1px solid rgba(245,158,11,0.25);
        border-left: 4px solid #F59E0B;
        border-radius: 12px;
        padding: 14px 20px;
        margin-bottom: 14px;
        display:flex; align-items:center; gap:14px;
    ">
      <span style="font-size:1.4rem; flex-shrink:0;">🏅</span>
      <div style="color:#CBD5E1; font-size:0.87rem; line-height:1.6;">
        The top 20 restaurants average a success score of
        <strong style="color:#FCD34D;">{top_avg:.1f} / 100</strong> —
        significantly above the platform median. Study their pillar profiles for benchmarking best practices.
      </div>
    </div>
    """, unsafe_allow_html=True)

st.dataframe(top_df, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 – Tier Radar Comparison
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="display:flex; align-items:center; gap:12px; margin: 36px 0 4px;">
  <div style="width:4px; height:32px; border-radius:2px;
       background: linear-gradient(180deg, #EC4899, #00C2A8);"></div>
  <div>
    <div style="font-size:1.3rem; font-weight:800; color:#F8FAFC; line-height:1.2;">
      📡 Pillar Comparison Across Success Tiers
    </div>
    <div style="font-size:0.8rem; color:#64748B; margin-top:2px;">
      Radar overlay showing how each tier excels or lags across the five performance pillars
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="display:flex; gap:8px; flex-wrap:wrap; margin-bottom:16px;">
  <div style="background:rgba(0,194,168,0.10); border:1px solid rgba(0,194,168,0.3);
       border-radius:8px; padding:6px 14px; font-size:0.78rem; color:#5EEAD4; font-weight:600;">
    🟢 Top Performer — score ≥ 80
  </div>
  <div style="background:rgba(108,99,255,0.10); border:1px solid rgba(108,99,255,0.3);
       border-radius:8px; padding:6px 14px; font-size:0.78rem; color:#A5B4FC; font-weight:600;">
    🟣 Performing — 65–79
  </div>
  <div style="background:rgba(245,158,11,0.10); border:1px solid rgba(245,158,11,0.3);
       border-radius:8px; padding:6px 14px; font-size:0.78rem; color:#FCD34D; font-weight:600;">
    🟡 Average — 50–64
  </div>
  <div style="background:rgba(249,115,22,0.10); border:1px solid rgba(249,115,22,0.3);
       border-radius:8px; padding:6px 14px; font-size:0.78rem; color:#FDBA74; font-weight:600;">
    🟠 Below Average — 30–49
  </div>
  <div style="background:rgba(239,68,68,0.10); border:1px solid rgba(239,68,68,0.3);
       border-radius:8px; padding:6px 14px; font-size:0.78rem; color:#FCA5A5; font-weight:600;">
    🔴 Struggling — &lt; 30
  </div>
</div>
""", unsafe_allow_html=True)

tier_radar    = get_tier_radar_averages(scored_df)
pillar_labels = list(PILLAR_WEIGHTS.keys())

tier_colors_map = {
    "Top Performer":  "#00C2A8",
    "Performing":     "#6C63FF",
    "Average":        "#F59E0B",
    "Below Average":  "#F97316",
    "Struggling":     "#EF4444",
}

fig_tier_radar = go.Figure()
for tier, vals in tier_radar.items():
    c   = tier_colors_map.get(tier, "#94A3B8")
    v_c = vals + [vals[0]]
    l_c = pillar_labels + [pillar_labels[0]]
    fig_tier_radar.add_trace(go.Scatterpolar(
        r=v_c,
        theta=l_c,
        fill="toself",
        name=tier,
        line_color=c,
        fillcolor=hex8_to_rgba(c + "22"),
    ))

fig_tier_radar.update_layout(
    polar=dict(
        radialaxis=dict(visible=True, range=[0, 100],
                        gridcolor="#334155", linecolor="#334155"),
        angularaxis=dict(gridcolor="#334155"),
        bgcolor="rgba(0,0,0,0)",
    ),
    paper_bgcolor="rgba(0,0,0,0)",
    font_color="#F8FAFC",
    title="Avg Pillar Scores by Success Tier",
    legend=dict(orientation="h", y=-0.15),
    height=520,
)
st.plotly_chart(fig_tier_radar, use_container_width=True, key="ss_tier_radar")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 – Recommendation Engine
# ═══════════════════════════════════════════════════════════════════════════════

# ── Section header (plain HTML — no stacking context, no overflow:hidden) ────
st.markdown("""
<div style="display:flex; align-items:center; gap:12px; margin: 36px 0 0;">
  <div style="width:4px; height:32px; border-radius:2px;
       background: linear-gradient(180deg, #EC4899, #6C63FF);"></div>
  <div>
    <div style="font-size:1.3rem; font-weight:800; color:#F8FAFC; line-height:1.2;">
      🔮 Restaurant Recommendation Engine
    </div>
    <div style="font-size:0.8rem; color:#94A3B8; margin-top:2px;">
      Four intelligent tools to match diners, operators, and analysts with the right
      restaurants and market strategies
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Capability pills row (decorative only — no wrapper div around tabs) ──────
st.markdown("""
<div style="display:flex; gap:10px; flex-wrap:wrap; margin: 14px 0 20px;">
  <span style="background:rgba(108,99,255,0.15); border:1px solid rgba(108,99,255,0.35);
        color:#A5B4FC; border-radius:8px; padding:6px 14px; font-size:0.8rem; font-weight:600;">
    🎯 Filter by preferences
  </span>
  <span style="background:rgba(0,194,168,0.10); border:1px solid rgba(0,194,168,0.3);
        color:#5EEAD4; border-radius:8px; padding:6px 14px; font-size:0.8rem; font-weight:600;">
    🔗 Find similar venues
  </span>
  <span style="background:rgba(236,72,153,0.10); border:1px solid rgba(236,72,153,0.28);
        color:#F9A8D4; border-radius:8px; padding:6px 14px; font-size:0.8rem; font-weight:600;">
    🍜 Best cuisines by city
  </span>
  <span style="background:rgba(245,158,11,0.10); border:1px solid rgba(245,158,11,0.28);
        color:#FCD34D; border-radius:8px; padding:6px 14px; font-size:0.8rem; font-weight:600;">
    💰 Optimal pricing tiers
  </span>
</div>
""", unsafe_allow_html=True)

# ── Tabs rendered at top level — no wrapping HTML div ────────────────────────
tab_pref, tab_similar, tab_cuisine, tab_pricing = st.tabs([
    "🎯 By Preference",
    "🔗 Similar to Restaurant",
    "🍜 Best Cuisines for Location",
    "💰 Pricing for New Operators",
])

# ── Tab 1: By Preference ──────────────────────────────────────────────────────
with tab_pref:
    st.caption("Set your dining preferences below and let the engine surface the best-matched restaurants. All filters are optional — combine freely for precision targeting.")

    st.markdown("**Cuisine & Location**")
    c1, c2 = st.columns(2)
    with c1:
        pref_cuisine = st.text_input(
            "Cuisine (optional)", placeholder="Italian", key="ss_pref_cuisine"
        )
    with c2:
        pref_city = st.text_input(
            "City (optional)", placeholder="New Delhi", key="ss_pref_city"
        )

    st.markdown("**Price & Rating**")
    c3, c4 = st.columns(2)
    with c3:
        pref_price = st.selectbox(
            "Price Range",
            [None, 1, 2, 3, 4],
            format_func=lambda x: "Any" if x is None
            else {1: "Budget", 2: "Affordable", 3: "Premium", 4: "Luxury"}[x],
            key="ss_pref_price",
        )
    with c4:
        pref_rating = st.slider(
            "Min Rating", 0.0, 5.0, 3.5, 0.1, key="ss_pref_rating"
        )

    st.markdown("**Service Options**")
    c5, c6 = st.columns(2)
    with c5:
        pref_delivery = st.checkbox("Needs Online Delivery", key="ss_pref_delivery")
    with c6:
        pref_booking = st.checkbox("Needs Table Booking", key="ss_pref_booking")

    st.write("")  # spacer
    if st.button("🔍 Find Restaurants", key="pref_btn"):
        _cuisine = pref_cuisine.strip() or None
        _city    = pref_city.strip()    or None

        try:
            recs = recommender.recommend_by_preferences(
                cuisine=_cuisine,
                city=_city,
                price_range=pref_price,
                min_rating=float(pref_rating),
                delivery_needed=pref_delivery,
                booking_needed=pref_booking,
            )
            if recs is None or recs.empty:
                st.warning("No restaurants match your criteria. Try relaxing the filters.")
            else:
                st.success(f"✅ Found {len(recs)} restaurants matching your criteria")
                st.dataframe(recs, use_container_width=True)
        except Exception as exc:
            st.error(f"Recommendation failed: {exc}")

# ── Tab 2: Similar to Restaurant ─────────────────────────────────────────────
with tab_similar:
    st.caption("Enter a restaurant name (partial matches supported) to discover venues with a similar cuisine profile, price positioning, and service offering.")

    search_name = st.text_input(
        "Restaurant Name (partial match ok)", "", key="ss_sim_search"
    )
    st.write("")  # spacer
    if st.button("🔗 Find Similar", key="sim_btn"):
        _search = search_name.strip()
        if not _search:
            st.warning("Please enter a restaurant name to search.")
        else:
            try:
                sim_recs = recommender.recommend_by_name(_search)
                if sim_recs is None:
                    st.warning(f"No restaurant found matching '{_search}'")
                else:
                    st.dataframe(sim_recs, use_container_width=True)
            except Exception as exc:
                st.error(f"Similarity search failed: {exc}")

# ── Tab 3: Best Cuisines for Location ────────────────────────────────────────
with tab_cuisine:
    st.caption("Discover which cuisine types perform best in a given market — ranked by a composite score combining average rating, vote volume, and market density signals.")

    cc1, cc2 = st.columns(2)
    with cc1:
        rec_country = st.selectbox(
            "Country",
            [None] + sorted(raw_df["Country"].unique().tolist()),
            format_func=lambda x: "All Countries" if x is None else x,
            key="ss_rec_country",
        )
    with cc2:
        rec_city = st.text_input("City (optional)", key="ss_rec_city")

    st.write("")  # spacer
    if st.button("🍜 Get Cuisine Recommendations", key="ss_cuisine_btn"):
        _rec_city = rec_city.strip() or None

        try:
            cuisine_recs = get_cuisine_recommendations(
                raw_df,
                country=rec_country,
                city=_rec_city,
            )
            if cuisine_recs is None or cuisine_recs.empty:
                st.warning("Not enough data for this location.")
            else:
                fig_cr = px.bar(
                    cuisine_recs.head(10), x="Score", y="Cuisine",
                    orientation="h",
                    color="Avg_Rating",
                    color_continuous_scale=["#F59E0B", "#00C2A8"],
                    title="Recommended Cuisines (composite score)",
                    text="Score",
                )
                fig_cr.update_traces(
                    texttemplate="%{text:.3f}", textposition="outside"
                )
                fig_cr.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#F8FAFC", coloraxis_showscale=False,
                    yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(
                    fig_cr, use_container_width=True, key="ss_cuisine_rec_bar"
                )
                st.dataframe(cuisine_recs, use_container_width=True)
        except Exception as exc:
            st.error(f"Cuisine recommendation failed: {exc}")

# ── Tab 4: Pricing for New Operators ─────────────────────────────────────────
with tab_pricing:
    st.caption("Planning a new restaurant? Enter your cuisine type and target city to receive a data-backed pricing tier recommendation with expected rating benchmarks.")

    p1, p2 = st.columns(2)
    with p1:
        new_cuisine = st.text_input(
            "Cuisine Type", placeholder="North Indian", key="ss_new_cuisine"
        )
    with p2:
        new_city = st.text_input(
            "Target City", placeholder="Mumbai", key="ss_new_city"
        )

    st.write("")  # spacer
    if st.button("💰 Get Pricing Recommendation", key="ss_pricing_btn"):
        _new_cuisine = new_cuisine.strip()
        _new_city    = new_city.strip() or None

        if not _new_cuisine:
            st.warning("Please enter a cuisine type to get a pricing recommendation.")
        else:
            try:
                pr = get_pricing_recommendation(
                    raw_df, cuisine=_new_cuisine, city=_new_city
                )

                st.markdown(f"""
                <div style="background:rgba(0,194,168,0.1);
                     border:1px solid rgba(0,194,168,0.4);
                     border-radius:16px; padding:24px; margin:12px 0;">
                  <div style="font-size:1.4rem; font-weight:700;
                       color:#F8FAFC; margin-bottom:8px;">
                    🎯 Recommended:
                    <span style="color:#00C2A8;">
                      {pr['tier_label']} (Tier {pr['recommended_tier']})
                    </span>
                  </div>
                  <div style="color:#CBD5E1; font-size:0.9rem; line-height:1.65;">
                    {pr['rationale']}
                  </div>
                  {
                    f'<div style="margin-top:12px; font-size:1rem;'
                    f' color:#00C2A8; font-weight:600;">'
                    f'Expected Avg Rating: {pr["expected_rating"]}</div>'
                    if pr.get("expected_rating") else ""
                  }
                </div>
                """, unsafe_allow_html=True)

                if pr.get("tier_stats") is not None and not pr["tier_stats"].empty:
                    fig_pt = px.bar(
                        pr["tier_stats"], x="Tier", y="Avg_Rating",
                        color="Score",
                        color_continuous_scale=["#1E293B", "#6C63FF", "#00C2A8"],
                        title=f"Rating by Price Tier – {_new_cuisine}",
                        text="Avg_Rating",
                    )
                    fig_pt.update_traces(
                        texttemplate="%{text:.3f}", textposition="outside"
                    )
                    fig_pt.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font_color="#F8FAFC",
                        coloraxis_showscale=False,
                    )
                    st.plotly_chart(
                        fig_pt, use_container_width=True,
                        key="ss_pricing_tier_bar"
                    )
            except Exception as exc:
                st.error(f"Pricing recommendation failed: {exc}")
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import inspect
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.components import (
    inject_global_css,
    render_sidebar_filters,
    section_header,
    data_quality_badge,
)
from src.preprocessing import load_and_preprocess, get_summary_kpis, filter_dataframe
from src.visualization import (
    rating_text_donut,
    price_donut,
    delivery_booking_bar,
    cuisine_treemap,
    city_bar,
    country_bar,
    rating_distribution,
)


# ── Gauge chart ────────────────────────────────────────────────────────────────

def gauge_chart(value: float, title: str, color: str = "#6C63FF") -> go.Figure:
    value = max(0.0, min(100.0, float(value)))
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": "%", "font": {"size": 32, "color": color}},
        title={"text": title, "font": {"size": 13, "color": "#94A3B8"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#1E293B"},
            "bar":  {"color": color, "thickness": 0.25},
            "bgcolor": "#1E293B",
            "borderwidth": 0,
            "steps": [{"range": [0, 100], "color": "#0F172A"}],
            "threshold": {
                "line":      {"color": "#F8FAFC", "width": 2},
                "thickness": 0.8,
                "value":     value,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=60, b=20),
        height=220,
        font=dict(color="#F8FAFC", family="Inter"),
    )
    return fig


# ── Filter helper ──────────────────────────────────────────────────────────────

def _call_filter_dataframe(df_full: pd.DataFrame, filters: dict) -> pd.DataFrame:
    sig    = inspect.signature(filter_dataframe)
    params = set(sig.parameters.keys())

    kwargs = {
        "countries":    filters.get("countries", []),
        "cities":       filters.get("cities", []),
        "cuisines":     filters.get("cuisines", []),
        "price_ranges": filters.get("price_ranges", []),
    }

    rating_range = filters.get("rating_range", (0.0, 5.0))
    if isinstance(rating_range, (list, tuple)) and len(rating_range) == 2:
        r_min, r_max = float(rating_range[0]), float(rating_range[1])
    else:
        r_min, r_max = 0.0, 5.0

    if "rating_range" in params:
        kwargs["rating_range"] = (r_min, r_max)
    elif "rating_min" in params and "rating_max" in params:
        kwargs["rating_min"] = r_min
        kwargs["rating_max"] = r_max

    df = filter_dataframe(df_full, **kwargs)

    if "Aggregate rating" in df.columns and (r_min > 0.0 or r_max < 5.0):
        mask = (df["Aggregate rating"] == 0) | (
            (df["Aggregate rating"] >= r_min) & (df["Aggregate rating"] <= r_max)
        )
        df = df.loc[mask]

    return df.reset_index(drop=True)


# ── Page entry point ───────────────────────────────────────────────────────────

def show():
    inject_global_css()

    # ── Load data ──────────────────────────────────────────────────────────────
    try:
        df_full = load_and_preprocess()
    except Exception as exc:
        st.error(f"Failed to load data: {exc}")
        st.stop()

    # ── Sidebar filters ────────────────────────────────────────────────────────
    try:
        filters = render_sidebar_filters(df_full, prefix="dashboard_")
        df      = _call_filter_dataframe(df_full, filters)
    except Exception as exc:
        st.warning(f"Filters could not be applied: {exc}")
        df = df_full.copy()

    df = df.reset_index(drop=True)

    if df.empty:
        st.warning("No data matches the current filters. Please adjust the sidebar.")
        st.stop()

    # ── KPIs ───────────────────────────────────────────────────────────────────
    try:
        kpis = get_summary_kpis(df)
    except Exception as exc:
        st.warning(f"Could not compute KPIs: {exc}")
        kpis = {
            "avg_rating":   "N/A",
            "total_votes":  0,
            "countries":    0,
            "cities":       0,
            "delivery_pct": 0.0,
            "booking_pct":  0.0,
            "top_cuisine":  "N/A",
            "top_city":     "N/A",
            "top_country":  "N/A",
        }

    # ── Page Header ────────────────────────────────────────────────────────────
    is_filtered = len(df) < len(df_full)
    filter_note = (
        f"{len(df):,} of {len(df_full):,} restaurants"
        if is_filtered else f"{len(df):,} restaurants"
    )

    st.title("📊 Executive Dashboard")
    st.caption(
        f"Live Analytics · {filter_note} · "
        f"{kpis.get('countries', 0)} countries · {kpis.get('cities', 0):,} cities"
    )

    data_quality_badge(df)

    st.divider()

    # ── Core KPIs ──────────────────────────────────────────────────────────────
    st.subheader("Core KPIs")

    sc1, sc2, sc3, sc4, sc5 = st.columns(5)
    with sc1:  st.metric("🍽️ Restaurants",  f"{len(df):,}")
    with sc2:  st.metric("⭐ Avg Rating",   str(kpis.get('avg_rating', 'N/A')))
    with sc3:  st.metric("🗳️ Total Votes",  f"{kpis.get('total_votes', 0):,}")
    with sc4:  st.metric("🌍 Countries",    str(kpis.get('countries', 0)))
    with sc5:  st.metric("🏙️ Cities",       f"{kpis.get('cities', 0):,}")

    st.markdown("")

    sc6, sc7, sc8, sc9, sc10 = st.columns(5)
    with sc6:  st.metric("🚴 Delivery %",   f"{kpis.get('delivery_pct', 0.0)}%")
    with sc7:  st.metric("📅 Booking %",    f"{kpis.get('booking_pct', 0.0)}%")
    with sc8:  st.metric("🍛 Top Cuisine",  str(kpis.get('top_cuisine', 'N/A')))
    with sc9:  st.metric("🏆 Top City",     str(kpis.get('top_city', 'N/A')))
    with sc10: st.metric("🌏 Top Country",  str(kpis.get('top_country', 'N/A')))

    st.divider()

    # ── Operational Gauges ─────────────────────────────────────────────────────
    st.subheader("Operational Gauges")

    g1, g2, g3 = st.columns(3)

    with g1:
        try:
            st.plotly_chart(
                gauge_chart(float(kpis.get("delivery_pct", 0.0)), "Online Delivery Rate", "#7C3AED"),
                use_container_width=True, key="dashboard_gauge_delivery",
            )
        except Exception as exc:
            st.warning(f"Delivery gauge unavailable: {exc}")

    with g2:
        try:
            st.plotly_chart(
                gauge_chart(float(kpis.get("booking_pct", 0.0)), "Table Booking Rate", "#06B6D4"),
                use_container_width=True, key="dashboard_gauge_booking",
            )
        except Exception as exc:
            st.warning(f"Booking gauge unavailable: {exc}")

    with g3:
        try:
            if "Is Rated" in df.columns:
                rated_pct = round(float(df["Is Rated"].mean()) * 100, 1)
            elif "Aggregate rating" in df.columns:
                rated_pct = round((df["Aggregate rating"] > 0).mean() * 100, 1)
            else:
                rated_pct = 0.0
            st.plotly_chart(
                gauge_chart(rated_pct, "Rated Restaurants %", "#F59E0B"),
                use_container_width=True, key="dashboard_gauge_rated",
            )
        except Exception as exc:
            st.warning(f"Rated gauge unavailable: {exc}")

    st.divider()

    # ── Distribution Breakdown ─────────────────────────────────────────────────
    st.subheader("Distribution Breakdown")

    d1, d2 = st.columns(2)
    with d1:
        try:
            st.plotly_chart(rating_text_donut(df), use_container_width=True,
                            key="dashboard_donut_rating")
        except Exception as exc:
            st.warning(f"Rating donut unavailable: {exc}")
    with d2:
        try:
            st.plotly_chart(price_donut(df), use_container_width=True,
                            key="dashboard_donut_price")
        except Exception as exc:
            st.warning(f"Price donut unavailable: {exc}")

    # ── Service Features ───────────────────────────────────────────────────────
    st.markdown("")
    try:
        st.plotly_chart(delivery_booking_bar(df), use_container_width=True,
                        key="dashboard_delivery_booking_bar")
    except Exception as exc:
        st.warning(f"Service feature chart unavailable: {exc}")

    st.divider()

    # ── Cuisine Intelligence ───────────────────────────────────────────────────
    st.subheader("Cuisine Intelligence")

    try:
        st.plotly_chart(cuisine_treemap(df, top_n=20), use_container_width=True,
                        key="dashboard_cuisine_treemap")
    except Exception as exc:
        st.warning(f"Cuisine treemap unavailable: {exc}")

    st.divider()

    # ── Geographic Breakdown ───────────────────────────────────────────────────
    st.subheader("Geographic Breakdown")

    gc1, gc2 = st.columns(2)
    with gc1:
        try:
            st.plotly_chart(city_bar(df, top_n=12), use_container_width=True,
                            key="dashboard_city_bar")
        except Exception as exc:
            st.warning(f"City chart unavailable: {exc}")
    with gc2:
        try:
            st.plotly_chart(country_bar(df), use_container_width=True,
                            key="dashboard_country_bar")
        except Exception as exc:
            st.warning(f"Country chart unavailable: {exc}")

    # ── Rating Distribution ────────────────────────────────────────────────────
    st.markdown("")
    section_header("⭐ Rating Histogram")
    try:
        st.plotly_chart(rating_distribution(df), use_container_width=True,
                        key="dashboard_rating_histogram")
    except Exception as exc:
        st.warning(f"Rating histogram unavailable: {exc}")

    st.divider()

    # ── Raw Data Explorer ──────────────────────────────────────────────────────
    st.subheader("Filtered Data Explorer")
    st.caption(f"{len(df):,} restaurants match your filters")

    display_cols = [
        c for c in [
            "Restaurant Name", "City", "Country", "Primary Cuisine",
            "Aggregate rating", "Price Label", "Votes",
            "Has Online delivery", "Has Table booking",
        ]
        if c in df.columns
    ]

    st.dataframe(df[display_cols].reset_index(drop=True), use_container_width=True, hide_index=True)


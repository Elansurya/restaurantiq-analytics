import streamlit as st
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing import load_and_preprocess, get_summary_kpis
from src.components import inject_global_css, render_hero_home


def show():
    inject_global_css()

    df   = load_and_preprocess()
    kpis = get_summary_kpis(df)

    # ── Hero ──────────────────────────────────────────────────────────────────
    # Uses render_hero_home() which is defined in src/components.py.
    # render_page_hero() does NOT exist — removed.
    
    render_hero_home(
        total_restaurants=kpis.get("total_restaurants", 9551),
        countries=kpis.get("countries", 15),
    )

    st.info("🚀 Navigate via the sidebar to explore: Dashboard · EDA · Maps · ML Suite", icon="ℹ️")

    # ── Executive KPI Section ─────────────────────────────────────────────────
    st.markdown('<div class="section-label">Platform at a Glance</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("🍽️ Restaurants",    f"{kpis['total_restaurants']:,}",  "+9,551 records")
    with c2: st.metric("🌍 Countries",       kpis['countries'],                 "Global coverage")
    with c3: st.metric("🏙️ Cities",          f"{kpis['cities']:,}",             "Unique markets")
    with c4: st.metric("⭐ Avg Rating",      kpis['avg_rating'],                "Rated restaurants")

    c5, c6, c7, c8 = st.columns(4)
    with c5: st.metric("🗳️ Total Votes",     f"{kpis['total_votes']:,}",        "Community reviews")
    with c6: st.metric("🚴 Online Delivery", f"{kpis['delivery_pct']}%",        "Delivery enabled")
    with c7: st.metric("📅 Table Booking",   f"{kpis['booking_pct']}%",         "Booking enabled")
    with c8: st.metric("🍛 Top Cuisine",     kpis['top_cuisine'],               "Most common")

    st.divider()

    # ── Analytics Modules ─────────────────────────────────────────────────────
    st.markdown('<div class="section-label">What\'s Inside RestaurantIQ</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-desc">Six integrated analytics modules — navigate via the sidebar to explore</p>',
        unsafe_allow_html=True,
    )

    features = [
        ("📊", "Data Exploration",
         "Histograms, violin plots, treemaps, heatmaps, scatter plots & statistical summaries for deep data exploration."),
        ("📈", "Descriptive Analysis",
         "High-level KPIs, delivery & booking rates, rating distributions and cuisine breakdowns in one powerful view."),
        ("🌍", "Geospatial Analytics",
         "Interactive Plotly Mapbox, Folium cluster maps, PyDeck 3D hexagon layers and city intelligence views."),
        ("🍽️", "Cuisine Intelligence",
         "Treemap of top cuisines, average rating by cuisine, multi-cuisine restaurant analysis and trends."),
        ("🤖", "Predictive Modeling",
         "Model training, cross-validation, SHAP feature importance, and live what-if rating prediction."),
        ("🏆", "Success Scoring",
         "Composite success scoring engine that ranks restaurants on performance, reach, and service quality."),
    ]

    cols = st.columns(3)
    for i, (icon, title, desc) in enumerate(features):
        with cols[i % 3]:
            st.markdown(
                f"""
                <div style="
                    background: rgba(8, 14, 28, 0.96);
                    border: 1px solid rgba(255, 255, 255, 0.16);
                    border-radius: 14px;
                    padding: 18px 20px;
                    margin-bottom: 12px;
                    position: relative;
                    overflow: hidden;
                ">
                    <div style="
                        position: absolute;
                        top: 0; left: 0; right: 0;
                        height: 1px;
                        background: linear-gradient(90deg, transparent, rgba(147, 197, 253, 0.50), transparent);
                    "></div>
                    <div style="font-size: 1.5rem; margin-bottom: 10px; line-height: 1;">{icon}</div>
                    <div style="
                        font-size: 0.92rem;
                        font-weight: 700;
                        color: #FFFFFF;
                        margin-bottom: 6px;
                        letter-spacing: -0.01em;
                    ">{title}</div>
                    <div style="
                        font-size: 0.78rem;
                        color: #CBD5E1;
                        line-height: 1.6;
                    ">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()

    # ── Dataset Preview ───────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Dataset Preview</div>', unsafe_allow_html=True)
    st.markdown(
        f'<p class="section-desc">{len(df):,} records · {len(df.columns)} features · '
        'Showing first 10 rows of the Cognifyz Restaurant Dataset</p>',
        unsafe_allow_html=True,
    )

    preview_cols = [c for c in [
        "Restaurant Name", "City", "Country", "Primary Cuisine",
        "Aggregate rating", "Rating text", "Price Label",
        "Has Online delivery", "Has Table booking", "Votes",
    ] if c in df.columns]

    st.dataframe(
        df[preview_cols].head(10),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    st.markdown(
        '<p class="footer-note">RestaurantIQ · Enterprise Analytics Edition · Phase 1 · Cognifyz Restaurant Dataset</p>',
        unsafe_allow_html=True,
    )
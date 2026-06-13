import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.components import inject_global_css, section_header
from src.feature_engineering import (
    engineer_features,
    encode_features,
    get_feature_stats,
    get_correlation_matrix,
)
from src.insights import delivery_booking_analysis, pricing_analysis
from src.preprocessing import load_and_preprocess


def show():
    inject_global_css()

    # ── Hero title block ──────────────────────────────────────────────────────
    # padding-top clears the Streamlit top-bar / banner so the title is never
    # hidden underneath it.  The explicit color fallback ensures the gradient
    # text is readable even in browsers that don't support
    # -webkit-background-clip:text.
    st.markdown(
        """
        <div style="
            padding-top: 2.5rem;
            padding-bottom: 0.25rem;
            margin-bottom: 0;
        ">
          <h1 style="
              font-size: 2.4rem;
              font-weight: 900;
              line-height: 1.2;
              margin: 0 0 6px 0;
              padding: 0;
              /* Gradient fill with solid-color fallback */
              color: #06B6D4;
              background: linear-gradient(90deg, #06B6D4, #6C63FF);
              -webkit-background-clip: text;
              -webkit-text-fill-color: transparent;
              background-clip: text;
          ">
            Feature Engineering &amp; Pricing Analysis
          </h1>
          <p style="
              color: #94A3B8;
              margin: 0 0 28px 0;
              font-size: 0.95rem;
              line-height: 1.5;
          ">
            Level 2 Task 2 &amp; 3 · Encoded &amp; Engineered Business Features
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df_raw = load_and_preprocess()
    df     = engineer_features(df_raw)

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 1 – Engineered Features Overview
    # ═══════════════════════════════════════════════════════════════════════════
    section_header("🔬 Engineered Features Overview")

    feat_stats = get_feature_stats(df_raw)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        val = df["Cuisine_Count"].mean() if "Cuisine_Count" in df.columns else 0
        st.metric("Avg Cuisine Count", f"{val:.2f}", help="Average cuisines per restaurant")
    with c2:
        val = df["Votes_Per_Rating"].mean() if "Votes_Per_Rating" in df.columns else 0
        st.metric("Avg Votes/Rating", f"{val:.1f}", help="Engagement density")
    with c3:
        val = df["Premium_Restaurant_Flag"].mean() * 100 if "Premium_Restaurant_Flag" in df.columns else 0
        st.metric("Premium Flag Rate", f"{val:.1f}%", help="% restaurants with Price range ≥ 3")
    with c4:
        val = df["Restaurant_Name_Length"].mean() if "Restaurant_Name_Length" in df.columns else 0
        st.metric("Avg Name Length", f"{val:.0f} chars")

    st.markdown("#### Feature Statistics")
    st.dataframe(feat_stats, use_container_width=True, hide_index=True)

    # Correlation heatmap
    section_header("🔗 Feature Correlation Matrix")
    corr = get_correlation_matrix(df_raw)
    fig_corr = px.imshow(
        corr,
        color_continuous_scale="RdBu",
        zmin=-1, zmax=1,
        title="Pearson Correlation – Engineered Features",
        aspect="auto",
        text_auto=".2f",
    )
    fig_corr.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#F8FAFC",
        height=480,
    )
    st.plotly_chart(fig_corr, use_container_width=True, key="features_corr_matrix")

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 2 – Price Range Analysis
    # ═══════════════════════════════════════════════════════════════════════════
    section_header("💰 Price Range Analysis")

    pa       = pricing_analysis(df_raw)
    price_df = pa["price_stats_df"]

    col_left, col_right = st.columns(2)

    with col_left:
        fig_pr = px.bar(
            price_df, x="Price Label", y="Avg_Rating",
            color="Avg_Rating",
            color_continuous_scale=["#EF4444", "#F59E0B", "#6C63FF", "#00C2A8"],
            title="Average Rating by Price Tier",
            text="Avg_Rating",
        )
        fig_pr.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig_pr.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#F8FAFC", coloraxis_showscale=False,
            xaxis_title="Price Tier", yaxis_title="Avg Rating",
        )
        st.plotly_chart(fig_pr, use_container_width=True, key="features_price_rating_bar")

    with col_right:
        fig_pv = px.bar(
            price_df, x="Price Label", y="Avg_Votes",
            color="Avg_Votes",
            color_continuous_scale=["#0F172A", "#6C63FF", "#00C2A8"],
            title="Average Votes by Price Tier",
            text="Avg_Votes",
        )
        fig_pv.update_traces(texttemplate="%{text:.0f}", textposition="outside")
        fig_pv.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#F8FAFC", coloraxis_showscale=False,
        )
        st.plotly_chart(fig_pv, use_container_width=True, key="features_price_votes_bar")

    # Delivery & booking by price tier
    if "Delivery_Pct" in price_df.columns and "Booking_Pct" in price_df.columns:
        fig_svc = go.Figure()
        fig_svc.add_trace(go.Bar(
            name="Delivery %",
            x=price_df["Price Label"].astype(str),
            y=(price_df["Delivery_Pct"] * 100).round(1),
            marker_color="#6C63FF",
        ))
        fig_svc.add_trace(go.Bar(
            name="Booking %",
            x=price_df["Price Label"].astype(str),
            y=(price_df["Booking_Pct"] * 100).round(1),
            marker_color="#00C2A8",
        ))
        fig_svc.update_layout(
            barmode="group",
            title="Delivery & Booking % by Price Tier",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#F8FAFC",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_svc, use_container_width=True, key="features_price_service_bar")

    st.markdown("#### 💡 Pricing Recommendations")
    for rec in pa["recommendations"]:
        st.info(rec)

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 3 – Table Booking & Online Delivery Analysis
    # ═══════════════════════════════════════════════════════════════════════════
    section_header("📅 Booking & Delivery Impact Analysis")

    dba      = delivery_booking_analysis(df_raw)
    booking  = dba["booking"]
    delivery = dba["delivery"]
    cross    = dba["cross"]

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Delivery %",        f"{delivery['yes_pct']}%")
    m2.metric("Delivery Lift",     f"{delivery['rating_lift']:+.3f}",
              help="Avg rating improvement with delivery")
    m3.metric("Delivery Votes",    f"{delivery['yes_avg_votes']:.0f}",
              help="Avg votes when delivery offered")
    m4.metric("Booking %",         f"{booking['yes_pct']}%")
    m5.metric("Booking Lift",      f"{booking['rating_lift']:+.3f}")
    m6.metric("Both Services Avg", f"{cross['both_avg_rating']:.3f}")

    col_b, col_d = st.columns(2)

    with col_b:
        bdf = pd.DataFrame({
            "Segment":    ["With Booking", "Without Booking"],
            "Avg Rating": [booking["yes_avg_rating"], booking["no_avg_rating"]],
            "Avg Votes":  [booking["yes_avg_votes"],  booking["no_avg_votes"]],
        })
        fig_b = px.bar(
            bdf, x="Segment", y="Avg Rating",
            color="Segment",
            color_discrete_map={
                "With Booking": "#00C2A8", "Without Booking": "#6C63FF",
            },
            title="Booking Impact on Avg Rating",
            text="Avg Rating",
        )
        fig_b.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig_b.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#F8FAFC", showlegend=False,
        )
        st.plotly_chart(fig_b, use_container_width=True, key="features_booking_impact_bar")

    with col_d:
        ddf = pd.DataFrame({
            "Segment":    ["With Delivery", "Without Delivery"],
            "Avg Rating": [delivery["yes_avg_rating"], delivery["no_avg_rating"]],
            "Avg Votes":  [delivery["yes_avg_votes"],  delivery["no_avg_votes"]],
        })
        fig_d = px.bar(
            ddf, x="Segment", y="Avg Rating",
            color="Segment",
            color_discrete_map={
                "With Delivery": "#6C63FF", "Without Delivery": "#F59E0B",
            },
            title="Delivery Impact on Avg Rating",
            text="Avg Rating",
        )
        fig_d.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig_d.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#F8FAFC", showlegend=False,
        )
        st.plotly_chart(fig_d, use_container_width=True, key="features_delivery_impact_bar")

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 4 – Encoded Features Preview
    # ═══════════════════════════════════════════════════════════════════════════
    section_header("🔢 Encoded Features Preview")

    df_enc, _ = encode_features(df, fit=True)

    enc_cols  = [c for c in df_enc.columns if c.endswith("_Enc") or c.endswith("_encoded")]
    base_cols = ["Restaurant Name", "Primary Cuisine", "City", "Country"]
    show_cols = [c for c in base_cols if c in df_enc.columns] + enc_cols
    preview   = df_enc[show_cols].head(50)

    st.caption("Sample of label-encoded categorical features (first 50 rows)")
    st.dataframe(preview, use_container_width=True, hide_index=True)

    enc_col_name = next(
        (c for c in ["Primary Cuisine_encoded", "Primary Cuisine_Enc"] if c in df_enc.columns),
        None,
    )
    if enc_col_name:
        fig_enc = px.histogram(
            df_enc, x=enc_col_name, nbins=40,
            title="Distribution of Primary Cuisine (Encoded)",
            color_discrete_sequence=["#6C63FF"],
        )
        fig_enc.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#F8FAFC",
        )
        st.plotly_chart(fig_enc, use_container_width=True, key="features_cuisine_encoded_hist")



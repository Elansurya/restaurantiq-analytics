import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.components import inject_global_css, section_header
from src.insights import generate_all_insights, delivery_booking_analysis, pricing_analysis
from src.preprocessing import load_and_preprocess


def show():
    inject_global_css()

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE HERO BANNER
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, rgba(236,72,153,0.14) 0%, rgba(108,99,255,0.12) 55%, rgba(245,158,11,0.08) 100%);
        border: 1px solid rgba(236,72,153,0.24);
        border-radius: 20px;
        padding: 36px 40px 30px;
        margin-bottom: 28px;
        position: relative;
        overflow: hidden;
    ">
      <div style="
          position:absolute; top:-40px; right:-20px;
          width:200px; height:200px; border-radius:50%;
          background: radial-gradient(circle, rgba(236,72,153,0.16), transparent 70%);
          pointer-events:none;
      "></div>
      <div style="
          position:absolute; bottom:-60px; left:60px;
          width:180px; height:180px; border-radius:50%;
          background: radial-gradient(circle, rgba(245,158,11,0.10), transparent 70%);
          pointer-events:none;
      "></div>

      <div style="display:flex; align-items:flex-start; gap:18px; position:relative;">
        <div style="font-size:3rem; line-height:1; flex-shrink:0;">💡</div>
        <div style="flex:1;">
          <div style="
              font-size:2rem; font-weight:900; line-height:1.15;
              background: linear-gradient(90deg, #F8FAFC 30%, #EC4899 100%);
              -webkit-background-clip: text; -webkit-text-fill-color: transparent;
              background-clip: text; margin-bottom:8px;
          ">Business Insights</div>
          <div style="color:#94A3B8; font-size:0.92rem; line-height:1.7; max-width:680px;">
            Auto-generated intelligence surfaced directly from your restaurant data — no manual curation needed.
            Explore delivery and booking impact, pricing dynamics, city performance rankings, cuisine trends,
            growth opportunities, and the statistical correlates of customer satisfaction.
          </div>
          <div style="display:flex; gap:10px; margin-top:16px; flex-wrap:wrap;">
            <span style="background:rgba(236,72,153,0.12); border:1px solid rgba(236,72,153,0.3);
                  color:#F9A8D4; border-radius:999px; padding:4px 14px; font-size:0.75rem; font-weight:600;">
              Delivery &amp; Booking Impact
            </span>
            <span style="background:rgba(245,158,11,0.10); border:1px solid rgba(245,158,11,0.28);
                  color:#FCD34D; border-radius:999px; padding:4px 14px; font-size:0.75rem; font-weight:600;">
              Pricing Analysis
            </span>
            <span style="background:rgba(108,99,255,0.12); border:1px solid rgba(108,99,255,0.3);
                  color:#A5B4FC; border-radius:999px; padding:4px 14px; font-size:0.75rem; font-weight:600;">
              City Intelligence
            </span>
            <span style="background:rgba(0,194,168,0.10); border:1px solid rgba(0,194,168,0.28);
                  color:#5EEAD4; border-radius:999px; padding:4px 14px; font-size:0.75rem; font-weight:600;">
              Cuisine Trends
            </span>
            <span style="background:rgba(16,185,129,0.08); border:1px solid rgba(16,185,129,0.25);
                  color:#6EE7B7; border-radius:999px; padding:4px 14px; font-size:0.75rem; font-weight:600;">
              Growth Opportunities
            </span>
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    df = load_and_preprocess()
    rated = df.loc[df["Aggregate rating"] > 0].reset_index(drop=True)

    with st.spinner("🧠 Analysing dataset…"):
        insights = generate_all_insights(df)

    kpis    = insights["kpis"]
    dba     = insights["delivery_impact"]
    pricing = insights["pricing"]

    # ═════════════════════════════════════════════════════════════════════════
    # EXECUTIVE KPI STRIP
    # ═════════════════════════════════════════════════════════════════════════
    st.markdown("""
    <div style="display:flex; align-items:center; gap:12px; margin: 8px 0 16px;">
      <div style="width:4px; height:28px; border-radius:2px;
           background: linear-gradient(180deg, #EC4899, #6C63FF);"></div>
      <div>
        <div style="font-size:1.15rem; font-weight:800; color:#F8FAFC;">📊 Platform KPIs</div>
        <div style="font-size:0.78rem; color:#64748B; margin-top:2px;">
          Top-line metrics summarising the entire restaurant dataset
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="display:grid; grid-template-columns: repeat(6, 1fr); gap:10px; margin-bottom:24px;">
      <div style="background:rgba(108,99,255,0.10); border:1px solid rgba(108,99,255,0.25);
           border-radius:12px; padding:14px 10px; text-align:center;">
        <div style="font-size:1.3rem; font-weight:900; color:#A5B4FC;">{kpis['total_restaurants']:,}</div>
        <div style="font-size:0.68rem; color:#64748B; margin-top:4px; text-transform:uppercase; letter-spacing:.06em;">🍽️ Restaurants</div>
      </div>
      <div style="background:rgba(245,158,11,0.08); border:1px solid rgba(245,158,11,0.22);
           border-radius:12px; padding:14px 10px; text-align:center;">
        <div style="font-size:1.3rem; font-weight:900; color:#FCD34D;">{kpis['avg_rating']}★</div>
        <div style="font-size:0.68rem; color:#64748B; margin-top:4px; text-transform:uppercase; letter-spacing:.06em;">⭐ Avg Rating</div>
      </div>
      <div style="background:rgba(0,194,168,0.08); border:1px solid rgba(0,194,168,0.22);
           border-radius:12px; padding:14px 10px; text-align:center;">
        <div style="font-size:1.3rem; font-weight:900; color:#5EEAD4;">{kpis['total_votes']:,}</div>
        <div style="font-size:0.68rem; color:#64748B; margin-top:4px; text-transform:uppercase; letter-spacing:.06em;">🗳️ Total Votes</div>
      </div>
      <div style="background:rgba(236,72,153,0.08); border:1px solid rgba(236,72,153,0.22);
           border-radius:12px; padding:14px 10px; text-align:center;">
        <div style="font-size:1.3rem; font-weight:900; color:#F9A8D4;">{kpis['countries']}</div>
        <div style="font-size:0.68rem; color:#64748B; margin-top:4px; text-transform:uppercase; letter-spacing:.06em;">🌍 Countries</div>
      </div>
      <div style="background:rgba(16,185,129,0.08); border:1px solid rgba(16,185,129,0.22);
           border-radius:12px; padding:14px 10px; text-align:center;">
        <div style="font-size:1.3rem; font-weight:900; color:#6EE7B7;">{kpis['delivery_pct']}%</div>
        <div style="font-size:0.68rem; color:#64748B; margin-top:4px; text-transform:uppercase; letter-spacing:.06em;">🚴 Delivery</div>
      </div>
      <div style="background:rgba(99,102,241,0.08); border:1px solid rgba(99,102,241,0.22);
           border-radius:12px; padding:14px 10px; text-align:center;">
        <div style="font-size:1.3rem; font-weight:900; color:#C7D2FE;">{kpis['booking_pct']}%</div>
        <div style="font-size:0.68rem; color:#64748B; margin-top:4px; text-transform:uppercase; letter-spacing:.06em;">📅 Booking</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🚴 Delivery & Booking",
        "💰 Pricing Analysis",
        "🏙️ City Intelligence",
        "🍽️ Cuisine Intelligence",
        "💡 Growth Opportunities",
    ])

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 1 – Delivery & Booking
    # ─────────────────────────────────────────────────────────────────────────
    with tab1:
        st.markdown("""
        <div style="display:flex; align-items:center; gap:12px; margin: 16px 0 12px;">
          <div style="width:4px; height:28px; border-radius:2px;
               background: linear-gradient(180deg, #00C2A8, #6C63FF);"></div>
          <div>
            <div style="font-size:1.15rem; font-weight:800; color:#F8FAFC;">🚴 Online Delivery &amp; Table Booking Impact</div>
            <div style="font-size:0.78rem; color:#64748B; margin-top:2px;">
              How service capabilities correlate with customer ratings and engagement volume
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        booking  = dba["booking"]
        delivery = dba["delivery"]
        cross    = dba["cross"]

        # Summary insight card
        delivery_lift_color = "#00C2A8" if delivery['rating_lift'] >= 0 else "#EF4444"
        booking_lift_color  = "#00C2A8" if booking['rating_lift'] >= 0 else "#EF4444"
        st.markdown(f"""
        <div style="
            background: linear-gradient(90deg, rgba(0,194,168,0.10), rgba(108,99,255,0.08));
            border: 1px solid rgba(0,194,168,0.22);
            border-radius: 14px;
            padding: 16px 20px;
            margin-bottom: 20px;
            display:grid; grid-template-columns: repeat(6, 1fr); gap:10px;
        ">
          <div style="text-align:center;">
            <div style="font-size:1.2rem; font-weight:800; color:#5EEAD4;">{delivery['yes_pct']}%</div>
            <div style="font-size:0.68rem; color:#64748B; text-transform:uppercase; margin-top:3px;">Delivery %</div>
          </div>
          <div style="text-align:center; border-left:1px solid rgba(255,255,255,0.06);">
            <div style="font-size:1.2rem; font-weight:800; color:{delivery_lift_color};">{delivery['rating_lift']:+.3f}★</div>
            <div style="font-size:0.68rem; color:#64748B; text-transform:uppercase; margin-top:3px;">Delivery Lift</div>
          </div>
          <div style="text-align:center; border-left:1px solid rgba(255,255,255,0.06);">
            <div style="font-size:1.2rem; font-weight:800; color:#A5B4FC;">{delivery['yes_avg_votes']:.0f}</div>
            <div style="font-size:0.68rem; color:#64748B; text-transform:uppercase; margin-top:3px;">Delivery Votes</div>
          </div>
          <div style="text-align:center; border-left:1px solid rgba(255,255,255,0.06);">
            <div style="font-size:1.2rem; font-weight:800; color:#5EEAD4;">{booking['yes_pct']}%</div>
            <div style="font-size:0.68rem; color:#64748B; text-transform:uppercase; margin-top:3px;">Booking %</div>
          </div>
          <div style="text-align:center; border-left:1px solid rgba(255,255,255,0.06);">
            <div style="font-size:1.2rem; font-weight:800; color:{booking_lift_color};">{booking['rating_lift']:+.3f}★</div>
            <div style="font-size:0.68rem; color:#64748B; text-transform:uppercase; margin-top:3px;">Booking Lift</div>
          </div>
          <div style="text-align:center; border-left:1px solid rgba(255,255,255,0.06);">
            <div style="font-size:1.2rem; font-weight:800; color:#FCD34D;">{cross['both_avg_rating']:.3f}★</div>
            <div style="font-size:0.68rem; color:#64748B; text-transform:uppercase; margin-top:3px;">Both Services</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

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
                    "With Booking":    "#00C2A8",
                    "Without Booking": "#6C63FF",
                },
                title="Table Booking Impact on Avg Rating",
                text="Avg Rating",
            )
            fig_b.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            fig_b.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#F8FAFC", showlegend=False,
                yaxis=dict(range=[0, max(booking["yes_avg_rating"],
                                         booking["no_avg_rating"]) * 1.2]),
            )
            st.plotly_chart(fig_b, use_container_width=True,
                            key="insights_booking_impact_bar")

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
                    "With Delivery":    "#6C63FF",
                    "Without Delivery": "#F59E0B",
                },
                title="Online Delivery Impact on Avg Rating",
                text="Avg Rating",
            )
            fig_d.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            fig_d.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#F8FAFC", showlegend=False,
                yaxis=dict(range=[0, max(delivery["yes_avg_rating"],
                                         delivery["no_avg_rating"]) * 1.2]),
            )
            st.plotly_chart(fig_d, use_container_width=True,
                            key="insights_delivery_impact_bar")

        # Service level comparison
        if "Has Online delivery" in rated.columns and "Has Table booking" in rated.columns:
            rated_copy = rated.copy()
            rated_copy["Service Level"] = rated_copy.apply(
                lambda r: "Full Service"  if r["Has Online delivery"] and r["Has Table booking"]
                else ("Delivery Only" if r["Has Online delivery"]
                else ("Booking Only"  if r["Has Table booking"] else "Basic")),
                axis=1,
            )
            svc_agg = (
                rated_copy.groupby("Service Level")
                .agg(
                    avg_rating=("Aggregate rating", "mean"),
                    avg_votes=("Votes", "mean"),
                    count=("Restaurant Name", "count"),
                )
                .reset_index()
                .sort_values("avg_rating", ascending=False)
            )

            st.markdown("""
            <div style="display:flex; align-items:center; gap:12px; margin: 28px 0 10px;">
              <div style="width:4px; height:24px; border-radius:2px;
                   background: linear-gradient(180deg, #F59E0B, #EC4899);"></div>
              <div style="font-size:1.05rem; font-weight:800; color:#F8FAFC;">🎯 Service Level Comparison</div>
            </div>
            """, unsafe_allow_html=True)

            # Insight card
            if not svc_agg.empty:
                top_svc = svc_agg.iloc[0]
                st.markdown(f"""
                <div style="
                    background:rgba(245,158,11,0.08); border:1px solid rgba(245,158,11,0.22);
                    border-left:4px solid #F59E0B; border-radius:12px;
                    padding:12px 18px; margin-bottom:14px;
                    display:flex; align-items:center; gap:12px;
                ">
                  <span style="font-size:1.3rem; flex-shrink:0;">🎯</span>
                  <div style="color:#CBD5E1; font-size:0.85rem; line-height:1.6;">
                    <strong style="color:#FCD34D;">"{top_svc['Service Level']}"</strong> restaurants lead with
                    an average rating of <strong style="color:#F8FAFC;">{top_svc['avg_rating']:.3f}★</strong>.
                    Restaurants offering both delivery and table booking consistently outperform single-service venues.
                  </div>
                </div>
                """, unsafe_allow_html=True)

            col_s1, col_s2 = st.columns(2)
            with col_s1:
                fig_sr = px.bar(
                    svc_agg, x="Service Level", y="avg_rating",
                    color="avg_rating",
                    color_continuous_scale=["#EF4444", "#F59E0B", "#00C2A8"],
                    text="avg_rating",
                    title="Avg Rating by Service Level",
                )
                fig_sr.update_traces(texttemplate="%{text:.3f}★", textposition="outside")
                fig_sr.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#F8FAFC", coloraxis_showscale=False,
                )
                st.plotly_chart(fig_sr, use_container_width=True,
                                key="insights_service_rating_bar")

            with col_s2:
                fig_sv = px.bar(
                    svc_agg, x="Service Level", y="avg_votes",
                    color="avg_votes",
                    color_continuous_scale=["#1E293B", "#6C63FF", "#00C2A8"],
                    text="avg_votes",
                    title="Avg Votes by Service Level",
                )
                fig_sv.update_traces(texttemplate="%{text:.0f}", textposition="outside")
                fig_sv.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#F8FAFC", coloraxis_showscale=False,
                )
                st.plotly_chart(fig_sv, use_container_width=True,
                                key="insights_service_votes_bar")

        # Country cross-analysis
        if "Country" in df.columns:
            st.markdown("""
            <div style="display:flex; align-items:center; gap:12px; margin: 28px 0 10px;">
              <div style="width:4px; height:24px; border-radius:2px;
                   background: linear-gradient(180deg, #6C63FF, #00C2A8);"></div>
              <div style="font-size:1.05rem; font-weight:800; color:#F8FAFC;">🌍 Delivery &amp; Booking by Country</div>
            </div>
            """, unsafe_allow_html=True)

            agg_dict: dict = {"Restaurants": ("Restaurant ID", "count")}
            if "Has Online delivery" in df.columns:
                agg_dict["Online_Delivery"] = (
                    "Has Online delivery", lambda x: f"{x.mean()*100:.1f}%"
                )
            if "Has Table booking" in df.columns:
                agg_dict["Table_Booking"] = (
                    "Has Table booking", lambda x: f"{x.mean()*100:.1f}%"
                )
            country_svc = (
                df.groupby("Country")
                .agg(**agg_dict)
                .sort_values("Restaurants", ascending=False)
                .reset_index()
            )
            st.dataframe(country_svc, use_container_width=True, hide_index=True)

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 2 – Pricing Analysis
    # ─────────────────────────────────────────────────────────────────────────
    with tab2:
        st.markdown("""
        <div style="display:flex; align-items:center; gap:12px; margin: 16px 0 12px;">
          <div style="width:4px; height:28px; border-radius:2px;
               background: linear-gradient(180deg, #F59E0B, #EC4899);"></div>
          <div>
            <div style="font-size:1.15rem; font-weight:800; color:#F8FAFC;">💰 Price Range Analysis</div>
            <div style="font-size:0.78rem; color:#64748B; margin-top:2px;">
              How pricing tiers relate to ratings, engagement, and service adoption
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        price_df = pricing["price_stats_df"]

        col_l, col_r = st.columns(2)

        with col_l:
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
            )
            st.plotly_chart(fig_pr, use_container_width=True,
                            key="insights_price_rating_bar")

        with col_r:
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
            st.plotly_chart(fig_pv, use_container_width=True,
                            key="insights_price_votes_bar")

        if "Delivery_Pct" in price_df.columns:
            fig_svc = go.Figure()
            fig_svc.add_trace(go.Bar(
                name="Delivery %",
                x=price_df["Price Label"].astype(str),
                y=(price_df["Delivery_Pct"] * 100).round(1),
                marker_color="#6C63FF",
            ))
            if "Booking_Pct" in price_df.columns:
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
            st.plotly_chart(fig_svc, use_container_width=True,
                            key="insights_price_service_bar")

        # Pricing recommendations as styled cards
        st.markdown("""
        <div style="display:flex; align-items:center; gap:12px; margin: 28px 0 14px;">
          <div style="width:4px; height:24px; border-radius:2px;
               background: linear-gradient(180deg, #00C2A8, #F59E0B);"></div>
          <div style="font-size:1.05rem; font-weight:800; color:#F8FAFC;">💡 Pricing Recommendations</div>
        </div>
        """, unsafe_allow_html=True)

        for i, rec in enumerate(pricing["recommendations"]):
            accent = ["#00C2A8", "#6C63FF", "#F59E0B", "#EC4899"][i % 4]
            st.markdown(f"""
            <div style="
                background:{accent}0D; border:1px solid {accent}35;
                border-left:4px solid {accent}; border-radius:12px;
                padding:13px 18px; margin-bottom:10px;
                display:flex; align-items:flex-start; gap:12px;
            ">
              <span style="font-size:1.2rem; flex-shrink:0; margin-top:1px;">💡</span>
              <div style="color:#CBD5E1; font-size:0.87rem; line-height:1.65;">{rec}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div style="font-size:0.78rem; font-weight:700; text-transform:uppercase;
             letter-spacing:.1em; color:#6C63FF; margin: 20px 0 8px; padding-left:4px;">
          Price Tier Detail Table
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(price_df, use_container_width=True, hide_index=True)

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 3 – City Intelligence
    # ─────────────────────────────────────────────────────────────────────────
    with tab3:
        st.markdown("""
        <div style="display:flex; align-items:center; gap:12px; margin: 16px 0 12px;">
          <div style="width:4px; height:28px; border-radius:2px;
               background: linear-gradient(180deg, #6C63FF, #00C2A8);"></div>
          <div>
            <div style="font-size:1.15rem; font-weight:800; color:#F8FAFC;">🏙️ City Performance Intelligence</div>
            <div style="font-size:0.78rem; color:#64748B; margin-top:2px;">
              Top cities ranked by average rating, with vote volume as an engagement signal
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        top_cities = insights["top_cities"]
        n = st.slider("Top N cities", 5, 30, 15, key="insights_city_slider")
        city_slice = top_cities.head(n)

        # City insight card
        if not city_slice.empty:
            top_city = city_slice.iloc[0]
            st.markdown(f"""
            <div style="
                background:rgba(108,99,255,0.08); border:1px solid rgba(108,99,255,0.22);
                border-left:4px solid #6C63FF; border-radius:12px;
                padding:12px 18px; margin-bottom:14px;
                display:flex; align-items:center; gap:12px;
            ">
              <span style="font-size:1.3rem; flex-shrink:0;">🏙️</span>
              <div style="color:#CBD5E1; font-size:0.85rem; line-height:1.6;">
                <strong style="color:#A5B4FC;">Top-ranked city:</strong>
                Explore how rating and vote signals diverge — high ratings with low votes
                may indicate niche or underexposed markets ripe for investment.
              </div>
            </div>
            """, unsafe_allow_html=True)

        fig_city = go.Figure()
        fig_city.add_trace(go.Bar(
            name="Avg Rating",
            x=city_slice["City"],
            y=city_slice["avg_rating"],
            marker_color="#6C63FF",
            yaxis="y",
            text=city_slice["avg_rating"].round(2),
            textposition="outside",
        ))
        fig_city.add_trace(go.Scatter(
            name="Total Votes",
            x=city_slice["City"],
            y=city_slice["total_votes"],
            mode="lines+markers",
            line=dict(color="#00C2A8", width=2.5),
            marker=dict(size=7),
            yaxis="y2",
        ))
        fig_city.update_layout(
            title=f"Top {n} Cities — Rating (bars) & Votes (line)",
            height=420,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#F8FAFC",
            xaxis=dict(tickangle=-30, gridcolor="rgba(255,255,255,0.04)"),
            yaxis=dict(title="Avg Rating",   gridcolor="rgba(255,255,255,0.04)"),
            yaxis2=dict(title="Total Votes", overlaying="y", side="right"),
            legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=1.05),
        )
        st.plotly_chart(fig_city, use_container_width=True, key="insights_city_bar_line")

        st.markdown("""
        <div style="font-size:0.78rem; font-weight:700; text-transform:uppercase;
             letter-spacing:.1em; color:#6C63FF; margin: 24px 0 8px; padding-left:4px;">
          🌍 Country Summary
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(insights["country_summary"], use_container_width=True, hide_index=True)

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 4 – Cuisine Intelligence
    # ─────────────────────────────────────────────────────────────────────────
    with tab4:
        st.markdown("""
        <div style="display:flex; align-items:center; gap:12px; margin: 16px 0 12px;">
          <div style="width:4px; height:28px; border-radius:2px;
               background: linear-gradient(180deg, #EC4899, #F59E0B);"></div>
          <div>
            <div style="font-size:1.15rem; font-weight:800; color:#F8FAFC;">🍽️ Top Cuisines by Rating</div>
            <div style="font-size:0.78rem; color:#64748B; margin-top:2px;">
              Cuisine types ranked by average customer rating — minimum 10 restaurants per category
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        top_cuisines = insights["top_cuisines"]
        if top_cuisines.empty:
            st.warning("Primary Cuisine data not available.")
        else:
            n_c = st.slider("Top N cuisines", 5, 30, 20, key="insights_cuisine_slider")
            cslice = top_cuisines.head(n_c)

            # Insight card
            if not cslice.empty:
                top_c = cslice.iloc[0]
                st.markdown(f"""
                <div style="
                    background:rgba(236,72,153,0.08); border:1px solid rgba(236,72,153,0.22);
                    border-left:4px solid #EC4899; border-radius:12px;
                    padding:12px 18px; margin-bottom:14px;
                    display:flex; align-items:center; gap:12px;
                ">
                  <span style="font-size:1.3rem; flex-shrink:0;">🍽️</span>
                  <div style="color:#CBD5E1; font-size:0.85rem; line-height:1.6;">
                    <strong style="color:#F9A8D4;">Highest-rated cuisine:</strong>
                    <strong style="color:#F8FAFC;">{top_c.get('Primary Cuisine', '—')}</strong>
                    with <strong style="color:#EC4899;">{top_c.get('avg_rating', 0):.3f}★</strong>.
                    Use these rankings to identify cuisine niches with high quality ceilings and expansion potential.
                  </div>
                </div>
                """, unsafe_allow_html=True)

            fig_cuis = px.bar(
                cslice.sort_values("avg_rating"),
                x="avg_rating",
                y="Primary Cuisine",
                orientation="h",
                color="avg_rating",
                color_continuous_scale=["#EF4444", "#F59E0B", "#00C2A8"],
                text="avg_rating",
                title=f"Top {n_c} Cuisines by Average Rating (min 10 restaurants)",
            )
            fig_cuis.update_traces(texttemplate="%{text:.3f}★", textposition="outside")
            fig_cuis.update_layout(
                height=max(400, n_c * 28),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#F8FAFC",
                coloraxis_showscale=False,
                xaxis=dict(title="Avg Rating", gridcolor="rgba(255,255,255,0.04)"),
                yaxis=dict(tickfont=dict(size=11)),
                margin=dict(l=160, r=80, t=50, b=40),
            )
            st.plotly_chart(fig_cuis, use_container_width=True,
                            key="insights_cuisine_rating_bar")

            st.markdown("""
            <div style="font-size:0.78rem; font-weight:700; text-transform:uppercase;
                 letter-spacing:.1em; color:#EC4899; margin: 16px 0 8px; padding-left:4px;">
              Cuisine Detail Table
            </div>
            """, unsafe_allow_html=True)
            st.dataframe(top_cuisines, use_container_width=True, hide_index=True)

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 5 – Growth Opportunities
    # ─────────────────────────────────────────────────────────────────────────
    with tab5:
        st.markdown("""
        <div style="display:flex; align-items:center; gap:12px; margin: 16px 0 12px;">
          <div style="width:4px; height:28px; border-radius:2px;
               background: linear-gradient(180deg, #10B981, #6C63FF);"></div>
          <div>
            <div style="font-size:1.15rem; font-weight:800; color:#F8FAFC;">💡 Growth Opportunities</div>
            <div style="font-size:0.78rem; color:#64748B; margin-top:2px;">
              Data-surfaced signals for expansion, differentiation, and market gap exploitation
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        priority_colors = {
            "opportunity": "#00C2A8",
            "niche":       "#F59E0B",
            "warning":     "#EF4444",
        }

        for opp in insights["growth_opps"]:
            st.markdown(f"""
            <div style="
                background:rgba(108,99,255,0.08);
                border:1px solid rgba(108,99,255,0.25);
                border-radius:14px;padding:18px 22px;
                margin-bottom:12px;
                display:flex;gap:16px;align-items:flex-start;
            ">
              <span style="font-size:1.8rem;flex-shrink:0;">{opp['icon']}</span>
              <div>
                <div style="font-weight:700;color:#F8FAFC;font-size:1rem;
                     margin-bottom:6px;">{opp['title']}</div>
                <div style="color:#94A3B8;font-size:0.88rem;line-height:1.65;">{opp['body']}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""
        <div style="display:flex; align-items:center; gap:12px; margin: 8px 0 14px;">
          <div style="width:4px; height:24px; border-radius:2px;
               background: linear-gradient(180deg, #EC4899, #F59E0B);"></div>
          <div style="font-size:1.05rem; font-weight:800; color:#F8FAFC;">🗺️ Market Gap Analysis</div>
        </div>
        """, unsafe_allow_html=True)

        for gap in insights["market_gaps"]:
            c = priority_colors.get(gap["type"], "#6C63FF")
            st.markdown(f"""
            <div style="
                background:{c}11;
                border-left:4px solid {c};
                border-radius:10px;padding:14px 20px;
                margin-bottom:10px;
            ">
              <div style="font-weight:700;color:#F8FAFC;margin-bottom:4px;">
                {gap['icon']} {gap['title']}
              </div>
              <div style="color:#CBD5E1;font-size:0.88rem;line-height:1.6;">
                {gap['body']}
              </div>
            </div>
            """, unsafe_allow_html=True)

        if not insights["market_gaps"] and not insights["growth_opps"]:
            st.info("No specific gaps detected — the dataset looks well-balanced across segments.")

        # ── Correlation insight ────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""
        <div style="display:flex; align-items:center; gap:12px; margin: 8px 0 12px;">
          <div style="width:4px; height:24px; border-radius:2px;
               background: linear-gradient(180deg, #6C63FF, #00C2A8);"></div>
          <div>
            <div style="font-size:1.05rem; font-weight:800; color:#F8FAFC;">🔗 What Drives Ratings? (Correlation)</div>
            <div style="font-size:0.75rem; color:#64748B; margin-top:2px;">
              Pearson correlation of dataset features against aggregate rating
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        num_cols = [
            c for c in ["Aggregate rating", "Votes", "Average Cost for two",
                        "Price range", "Has Online delivery", "Has Table booking",
                        "Cuisine Count", "Is delivering now"]
            if c in df.columns
        ]
        if len(num_cols) > 2:
            corr = df[num_cols].corr()["Aggregate rating"].drop("Aggregate rating").sort_values(
                ascending=False
            )

            # Insight card
            if not corr.empty:
                top_corr_feat = corr.index[0]
                top_corr_val  = corr.iloc[0]
                corr_color    = "#00C2A8" if top_corr_val >= 0 else "#EF4444"
                st.markdown(f"""
                <div style="
                    background:rgba(0,194,168,0.07); border:1px solid rgba(0,194,168,0.20);
                    border-left:4px solid #00C2A8; border-radius:12px;
                    padding:12px 18px; margin-bottom:14px;
                    display:flex; align-items:center; gap:12px;
                ">
                  <span style="font-size:1.3rem; flex-shrink:0;">🔗</span>
                  <div style="color:#CBD5E1; font-size:0.85rem; line-height:1.6;">
                    <strong style="color:#5EEAD4;">Strongest positive correlate:</strong>
                    <strong style="color:#F8FAFC;">{top_corr_feat}</strong>
                    (r = <strong style="color:{corr_color};">{top_corr_val:.3f}</strong>).
                    Features with high positive r are the most actionable levers for improving customer ratings.
                  </div>
                </div>
                """, unsafe_allow_html=True)

            fig_corr = go.Figure(go.Bar(
                x=corr.values,
                y=corr.index,
                orientation="h",
                marker=dict(
                    color=corr.values,
                    colorscale=[[0, "#EF4444"], [0.5, "#1E293B"], [1, "#00C2A8"]],
                    showscale=False,
                ),
                text=[f"{v:.3f}" for v in corr.values],
                textposition="outside",
                textfont=dict(color="#94A3B8"),
            ))
            fig_corr.add_vline(x=0, line_dash="dash", line_color="#475569")
            fig_corr.update_layout(
                title="Pearson Correlation with Aggregate Rating",
                height=380,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#F8FAFC",
                xaxis=dict(title="Correlation Coefficient",
                           gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(tickfont=dict(size=11)),
                margin=dict(l=160, r=80, t=50, b=40),
            )
            st.plotly_chart(fig_corr, use_container_width=True,
                            key="insights_corr_with_rating_bar")



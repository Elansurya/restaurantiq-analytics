from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st

from src.components import (
    inject_global_css,
    section_header,
    data_quality_badge,
    get_filtered_dataframe,
    render_active_filter_badge,
)
from src.preprocessing import load_and_preprocess

# ── Optional heavy deps ───────────────────────────────────────────────────────
try:
    import plotly.express as px
    import plotly.graph_objects as go
    _PLOTLY_OK = True
except ImportError:
    _PLOTLY_OK = False

try:
    import folium
    from folium.plugins import HeatMap, MarkerCluster
    _FOLIUM_OK = True
except ImportError:
    _FOLIUM_OK = False

try:
    import pydeck as pdk
    _PYDECK_OK = True
except ImportError:
    _PYDECK_OK = False

try:
    from src.geospatial_analysis import (
        _ensure_derived_columns,
        _clean_geo,
        _get_rated,
        plotly_scatter_map,
        plotly_density_map,
        plotly_rating_bubble_map,
        city_avg_rating_map,
        folium_cluster_map,
        folium_heatmap,
        pydeck_hexagon_layer,
        pydeck_scatter_layer,
        PALETTE,
        MAPBOX_STYLE,
        RATING_COLOR_MAP,
    )
    _GEO_ENGINE_OK = True
except Exception:
    _GEO_ENGINE_OK = False


def _embed_folium(m) -> None:
    try:
        from streamlit.components.v1 import html as st_html
        html_str = m._repr_html_()
        st_html(html_str, height=560, scrolling=False)
    except Exception as exc:
        st.warning(f"Folium map render error: {exc}")


def _embed_pydeck(deck) -> None:
    try:
        st.pydeck_chart(deck)
    except Exception as exc:
        st.warning(f"PyDeck render error: {exc}")


def _geo_stat_card(icon: str, label: str, value: str, accent: str = "#F59E0B") -> str:
    return f"""
    <div style="background:linear-gradient(135deg,rgba(255,255,255,0.03) 0%,rgba(255,255,255,0.008) 100%),rgba(9,14,25,0.88);
         border:1px solid rgba(255,255,255,0.06);border-radius:16px;padding:20px 22px;
         position:relative;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,0.4),0 4px 16px rgba(0,0,0,0.3);">
      <div style="position:absolute;top:0;left:0;right:0;height:2px;
           background:linear-gradient(90deg,{accent},{accent}55);opacity:0.9;"></div>
      <div style="font-size:1.4rem;margin-bottom:8px;">{icon}</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:1.5rem;font-weight:700;
           color:{accent};letter-spacing:-0.03em;line-height:1;">{value}</div>
      <div style="font-family:'DM Sans',sans-serif;font-size:0.67rem;font-weight:700;
           color:#475569;text-transform:uppercase;letter-spacing:0.16em;margin-top:6px;">{label}</div>
    </div>
    """


def show() -> None:
    inject_global_css()

    # ── Load full dataset ──────────────────────────────────────────────────────
    try:
        df_full = load_and_preprocess()
    except Exception as exc:
        st.error(f"Failed to load data: {exc}")
        st.stop()

    # ── Apply shared filters from session state (set in Advanced EDA) ─────────
    df = get_filtered_dataframe(df_full)

    if _GEO_ENGINE_OK:
        df      = _ensure_derived_columns(df)
        df_full = _ensure_derived_columns(df_full)

    if df.empty:
        st.warning("The active filters return no data. Adjust filters on the Advanced EDA page.")
        st.stop()

    # ── Hero Banner ────────────────────────────────────────────────────────────
    has_geo  = "Latitude" in df.columns and "Longitude" in df.columns
    geo_df   = df.dropna(subset=["Latitude", "Longitude"]) if has_geo else df
    geo_pct  = round(len(geo_df) / len(df) * 100, 1) if len(df) > 0 else 0
    countries_count = df["Country"].nunique() if "Country" in df.columns else 0
    cities_count    = df["City"].nunique()    if "City"    in df.columns else 0

    st.markdown(f"""
    <div style="
        background:linear-gradient(135deg,#04060F 0%,#080D18 35%,#06101C 70%,#040810 100%);
        border:1px solid rgba(245,158,11,0.18);border-radius:24px;
        padding:44px 52px 36px;margin-bottom:28px;position:relative;overflow:hidden;
        box-shadow:0 2px 8px rgba(0,0,0,0.5),0 8px 32px rgba(0,0,0,0.4);
    ">
      <div style="position:absolute;top:-80px;right:-80px;width:340px;height:340px;
           background:radial-gradient(circle,rgba(245,158,11,0.10),transparent 65%);pointer-events:none;"></div>
      <div style="position:absolute;bottom:-60px;left:25%;width:280px;height:280px;
           background:radial-gradient(circle,rgba(239,68,68,0.07),transparent 65%);pointer-events:none;"></div>
      <div style="position:absolute;top:0;left:0;right:0;height:1px;
           background:linear-gradient(90deg,transparent,rgba(245,158,11,0.45) 35%,rgba(239,68,68,0.30) 70%,transparent);"></div>

      <div style="display:inline-flex;align-items:center;gap:8px;
           background:rgba(245,158,11,0.10);border:1px solid rgba(245,158,11,0.28);
           border-radius:999px;padding:5px 16px;font-size:0.67rem;color:#FCD34D;
           letter-spacing:.16em;text-transform:uppercase;font-weight:700;
           margin-bottom:20px;font-family:'DM Sans',sans-serif;">
        &#127758; Geospatial Intelligence
      </div>

      <h1 style="font-family:'Sora',sans-serif;font-size:2.8rem;font-weight:900;
           line-height:1.06;margin:0 0 10px;letter-spacing:-0.04em;">
        <span style="background:linear-gradient(135deg,#FCD34D 0%,#FCA5A5 100%);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
          Geospatial Analysis
        </span>
      </h1>

      <p style="color:#64748B;font-size:0.92rem;margin-bottom:28px;max-width:640px;
           line-height:1.65;font-family:'DM Sans',sans-serif;">
        Multi-layer spatial intelligence &#8212; Plotly Mapbox scatter &amp; density maps,
        Folium interactive cluster maps, and PyDeck 3D hexagon layers.
        Filters from the EDA page are automatically applied.
      </p>

      <div style="display:flex;flex-wrap:wrap;gap:10px;align-items:center;">
        <div style="background:rgba(245,158,11,0.10);border:1px solid rgba(245,158,11,0.22);
             border-radius:999px;padding:5px 16px;font-family:'JetBrains Mono',monospace;
             font-size:0.71rem;color:#FCD34D;font-weight:500;">
          &#128205; {len(geo_df):,} geo-tagged
        </div>
        <div style="background:rgba(6,182,212,0.10);border:1px solid rgba(6,182,212,0.22);
             border-radius:999px;padding:5px 16px;font-family:'JetBrains Mono',monospace;
             font-size:0.71rem;color:#67E8F9;font-weight:500;">
          &#127758; {countries_count} countries
        </div>
        <div style="background:rgba(124,58,237,0.10);border:1px solid rgba(124,58,237,0.22);
             border-radius:999px;padding:5px 16px;font-family:'JetBrains Mono',monospace;
             font-size:0.71rem;color:#A78BFA;font-weight:500;">
          &#127961;&#65039; {cities_count} cities
        </div>
        <div style="background:rgba(16,185,129,0.10);border:1px solid rgba(16,185,129,0.22);
             border-radius:999px;padding:5px 16px;font-family:'JetBrains Mono',monospace;
             font-size:0.71rem;color:#6EE7B7;font-weight:500;">
          {geo_pct}% geo coverage
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    render_active_filter_badge(df_full, df)
    data_quality_badge(df)

    if not _GEO_ENGINE_OK:
        st.error(
            "Geospatial engine (src/geospatial_analysis.py) could not be imported. "
            "Ensure the file exists and all dependencies are installed."
        )
        st.stop()

    if not has_geo:
        st.error(
            "The dataset does not contain Latitude/Longitude columns. "
            "Geospatial visualisations are unavailable."
        )
        st.stop()

    # ── Tabs ───────────────────────────────────────────────────────────────────
    tab_plotly, tab_folium, tab_pydeck, tab_stats = st.tabs([
        "&#128506;&#65039; Plotly Maps",
        "&#128205; Folium Maps",
        "&#127959;&#65039; PyDeck 3-D",
        "&#128202; Geo Stats",
    ])

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 1 – Plotly Mapbox
    # ─────────────────────────────────────────────────────────────────────────
    with tab_plotly:
        if not _PLOTLY_OK:
            st.info("Plotly is not installed.")
        else:
            st.markdown("""
            <div style="background:rgba(245,158,11,0.06);border:1px solid rgba(245,158,11,0.14);
                 border-radius:14px;padding:14px 20px;margin-bottom:20px;">
              <div style="font-family:'DM Sans',sans-serif;font-size:0.80rem;color:#64748B;line-height:1.6;">
                <strong style="color:#FCD34D;">Four Plotly layers</strong> &#8212; scatter (individual restaurants),
                density (concentration hotspots), rating bubbles (size = votes), and city intelligence
                (aggregated city-level view). Pan and zoom freely.
              </div>
            </div>
            """, unsafe_allow_html=True)

            section_header("📍 Restaurant Scatter Map")
            color_by = st.selectbox(
                "Colour markers by",
                options=[c for c in ["Rating text", "Price Label", "Primary Cuisine", "Country"]
                         if c in df.columns],
                key="geo_scatter_color",
            )
            try:
                st.plotly_chart(plotly_scatter_map(df, color_by=color_by),
                                use_container_width=True, key="geo_scatter_map")
            except Exception as exc:
                st.warning(f"Scatter map unavailable: {exc}")

            st.markdown("<br>", unsafe_allow_html=True)
            section_header("🌡️ Density Heatmap")
            try:
                st.plotly_chart(plotly_density_map(df),
                                use_container_width=True, key="geo_density_map")
            except Exception as exc:
                st.warning(f"Density map unavailable: {exc}")

            st.markdown("<br>", unsafe_allow_html=True)
            section_header("🔵 Ratings & Popularity Bubble Map")
            try:
                st.plotly_chart(plotly_rating_bubble_map(df),
                                use_container_width=True, key="geo_bubble_map")
            except Exception as exc:
                st.warning(f"Bubble map unavailable: {exc}")

            st.markdown("<br>", unsafe_allow_html=True)
            section_header("🏙️ City Intelligence Map")
            try:
                st.plotly_chart(city_avg_rating_map(df),
                                use_container_width=True, key="geo_city_map")
            except Exception as exc:
                st.warning(f"City intelligence map unavailable: {exc}")

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 2 – Folium
    # ─────────────────────────────────────────────────────────────────────────
    with tab_folium:
        if not _FOLIUM_OK:
            st.info("Folium is not installed. Run `pip install folium` to enable these maps.")
        else:
            st.markdown("""
            <div style="background:rgba(6,182,212,0.06);border:1px solid rgba(6,182,212,0.14);
                 border-radius:14px;padding:14px 20px;margin-bottom:20px;">
              <div style="font-family:'DM Sans',sans-serif;font-size:0.80rem;color:#64748B;line-height:1.6;">
                <strong style="color:#67E8F9;">Interactive Folium maps</strong> &#8212; fully interactive
                marker clustering with restaurant tooltips, and a weighted heatmap based on votes
                or aggregate rating. Click individual markers for restaurant details.
              </div>
            </div>
            """, unsafe_allow_html=True)

            section_header("📍 Marker Cluster Map")
            max_pts = st.slider(
                "Max markers to render", 200, 3000, 1000, step=100,
                key="geo_folium_cluster_max",
            )
            try:
                m = folium_cluster_map(df, max_points=max_pts)
                _embed_folium(m)
            except Exception as exc:
                st.warning(f"Cluster map unavailable: {exc}")

            st.markdown("<br>", unsafe_allow_html=True)
            section_header("🌡️ Votes / Rating Heatmap")
            weight_col = st.radio(
                "Weight heatmap by",
                options=["Votes", "Aggregate rating"],
                horizontal=True,
                key="geo_folium_heatmap_weight",
            )
            try:
                hm = folium_heatmap(df, weight_col=weight_col)
                _embed_folium(hm)
            except Exception as exc:
                st.warning(f"Heatmap unavailable: {exc}")

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 3 – PyDeck
    # ─────────────────────────────────────────────────────────────────────────
    with tab_pydeck:
        if not _PYDECK_OK:
            st.info("PyDeck is not installed. Run `pip install pydeck` to enable these maps.")
        else:
            st.markdown("""
            <div style="background:rgba(124,58,237,0.06);border:1px solid rgba(124,58,237,0.14);
                 border-radius:14px;padding:14px 20px;margin-bottom:20px;">
              <div style="font-family:'DM Sans',sans-serif;font-size:0.80rem;color:#64748B;line-height:1.6;">
                <strong style="color:#A78BFA;">PyDeck 3D visualisations</strong> &#8212; GPU-accelerated
                WebGL rendering. The hexagon layer elevates columns by restaurant density;
                the scatter layer sizes and colours points by rating.
                Drag to orbit, scroll to zoom, shift+drag to pan.
              </div>
            </div>
            """, unsafe_allow_html=True)

            section_header("🏗️ 3-D Hexagon Density Layer")
            try:
                deck = pydeck_hexagon_layer(df)
                _embed_pydeck(deck)
            except Exception as exc:
                st.warning(f"Hexagon layer unavailable: {exc}")

            st.markdown("<br>", unsafe_allow_html=True)
            section_header("⭕ Rating Scatter Layer")
            try:
                deck2 = pydeck_scatter_layer(df)
                _embed_pydeck(deck2)
            except Exception as exc:
                st.warning(f"Scatter layer unavailable: {exc}")

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 4 – Geo Stats
    # ─────────────────────────────────────────────────────────────────────────
    with tab_stats:
        try:
            geo = _clean_geo(df)
        except Exception:
            geo = df.copy()

        if not geo.empty:
            rated_geo = geo[geo["Aggregate rating"] > 0] if "Aggregate rating" in geo.columns else geo
            gc1, gc2, gc3, gc4 = st.columns(4)
            with gc1:
                st.markdown(_geo_stat_card("📍", "Valid Geo Rows", f"{len(geo):,}", "#FCD34D"), unsafe_allow_html=True)
            with gc2:
                st.markdown(_geo_stat_card("🌍", "Countries", str(geo["Country"].nunique() if "Country" in geo.columns else "—"), "#67E8F9"), unsafe_allow_html=True)
            with gc3:
                st.markdown(_geo_stat_card("🏙️", "Cities", str(geo["City"].nunique() if "City" in geo.columns else "—"), "#A78BFA"), unsafe_allow_html=True)
            with gc4:
                st.markdown(_geo_stat_card("⭐", "Rated", f"{len(rated_geo):,}", "#6EE7B7"), unsafe_allow_html=True)

        st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)

        st.markdown("""
        <div style="display:flex;align-items:center;gap:14px;margin-bottom:18px;">
          <div style="flex:1;height:1px;background:linear-gradient(90deg,rgba(245,158,11,0.25),transparent);"></div>
          <span style="font-family:'DM Sans',sans-serif;font-size:0.65rem;font-weight:700;
                color:#64748B;letter-spacing:0.22em;text-transform:uppercase;">Country Breakdown</span>
          <div style="flex:1;height:1px;background:linear-gradient(90deg,transparent,rgba(239,68,68,0.20));"></div>
        </div>
        """, unsafe_allow_html=True)

        if "Country" in df.columns:
            try:
                country_geo = (
                    df.groupby("Country")
                    .agg(
                        Restaurants=("Restaurant ID", "count"),
                        Cities=("City", "nunique"),
                        Avg_Rating=(
                            "Aggregate rating",
                            lambda x: round(x[x > 0].mean(), 2) if (x > 0).any() else 0.0,
                        ),
                    )
                    .sort_values("Restaurants", ascending=False)
                    .reset_index()
                )
                st.dataframe(country_geo, use_container_width=True, hide_index=True)
            except Exception as exc:
                st.warning(f"Country table unavailable: {exc}")

        st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)

        st.markdown("""
        <div style="display:flex;align-items:center;gap:14px;margin-bottom:18px;">
          <div style="flex:1;height:1px;background:linear-gradient(90deg,rgba(245,158,11,0.25),transparent);"></div>
          <span style="font-family:'DM Sans',sans-serif;font-size:0.65rem;font-weight:700;
                color:#64748B;letter-spacing:0.22em;text-transform:uppercase;">Top Cities by Geo Coverage</span>
          <div style="flex:1;height:1px;background:linear-gradient(90deg,transparent,rgba(239,68,68,0.20));"></div>
        </div>
        """, unsafe_allow_html=True)

        if "City" in df.columns and "Latitude" in df.columns:
            try:
                city_geo = (
                    df.groupby("City")
                    .agg(
                        Restaurants=("Restaurant ID", "count"),
                        Lat=("Latitude", "median"),
                        Lon=("Longitude", "median"),
                        Avg_Rating=(
                            "Aggregate rating",
                            lambda x: round(x[x > 0].mean(), 2) if (x > 0).any() else 0.0,
                        ),
                    )
                    .sort_values("Restaurants", ascending=False)
                    .head(30)
                    .reset_index()
                )
                st.dataframe(city_geo, use_container_width=True, hide_index=True)
            except Exception as exc:
                st.warning(f"City geo table unavailable: {exc}")


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import streamlit as st

from src.components import (
    inject_global_css,
    render_sidebar_filters,
    save_filters_to_state,
    section_header,
    data_quality_badge,
)
from src.preprocessing import load_and_preprocess, filter_dataframe
from src.visualization import (
    rating_distribution,
    votes_histogram,
    rating_vs_cost_scatter,
    violin_rating_by_price,
    heatmap_city_rating,
    top_cuisines_avg_rating,
    cost_by_country,
    cuisine_treemap,
    city_bar,
    country_bar,
    descriptive_stats_table,
    delivery_booking_bar,
)


# ══════════════════════════════════════════════════════════════════════════════
# ── Pandas version compatibility shim ────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _styler_map(styler, func, **kwargs):
    """
    Call Styler.map() on pandas ≥ 2.1, Styler.applymap() on older versions.
    Both accept the same positional function and keyword arguments.
    """
    if hasattr(styler, "map"):          # pandas ≥ 2.1
        return styler.map(func, **kwargs)
    return styler.applymap(func, **kwargs)  # pandas < 2.1


# ══════════════════════════════════════════════════════════════════════════════
# ── Cell-level style functions (used with applymap / map) ────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _white_text_bold(_val) -> str:
    """Force white, bold text on every cell — applied independently of gradient."""
    return "color: #FFFFFF; font-weight: 700;"


def _force_dark_bg(val) -> str:
    """
    For near-zero cells in the RdBu_r correlation heatmap the gradient
    produces a pale grey/white background.  When the numeric value is
    between -0.15 and +0.15, override the background with a visible dark
    charcoal so white text remains legible.
    """
    try:
        v = float(val)
    except (TypeError, ValueError):
        return ""
    if -0.15 <= v <= 0.15:
        return "background-color: #374151;"   # dark charcoal for near-zero
    return ""


def _highlight_max_dark(series: pd.Series) -> list[str]:
    """
    Column-wise highlight_max replacement.
    Returns a list of style strings — max cell gets a dark violet bg,
    all others are transparent.  Written as a Styler.apply() callback
    (axis=0) so it produces an independent style entry, not merged into
    the gradient string.
    """
    is_max = series == series.max()
    return ["background-color: #312e81; color: #FFFFFF; font-weight: 700;"
            if m else "" for m in is_max]


# ══════════════════════════════════════════════════════════════════════════════
# ── Public styling helpers ────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _styled_gradient(
    styler: "pd.io.formats.style.Styler",
    cmap: str = "Blues",
    axis: "int | None" = None,
    vmin: "float | None" = None,
    vmax: "float | None" = None,
    fix_near_zero: bool = False,
) -> "pd.io.formats.style.Styler":
    """
    Apply a background-gradient and guarantee readable white text on
    every cell using the applymap/map approach (v6 fix).

    Key design decisions
    --------------------
    • "Blues"   — used for cross-tabs and descriptive stats.
      Even the lightest Blues cell (#dbeafe) has enough contrast when
      paired with bold white text under Streamlit's dark theme glass.

    • "RdBu_r"  — used for the correlation matrix (replaces RdYlGn).
      Produces dark red (strong negative) → pale/dark blue (strong
      positive) with a mid-grey band near zero.  fix_near_zero=True
      overlays a dark charcoal background for |value| < 0.15 so
      white text stays legible there too.

    The applymap step writes its style dict entry SEPARATELY from
    background_gradient's entry; GlideDataGrid applies both, so the
    white text colour is never overwritten by the gradient's auto-color.
    """
    bg_kwargs: dict = {"cmap": cmap, "axis": axis}
    if vmin is not None:
        bg_kwargs["vmin"] = vmin
    if vmax is not None:
        bg_kwargs["vmax"] = vmax

    styler = styler.background_gradient(**bg_kwargs)

    # Optionally patch near-zero cells before applying white text
    if fix_near_zero:
        styler = _styler_map(styler, _force_dark_bg)

    # Apply white bold text in a SEPARATE applymap call
    styler = _styler_map(styler, _white_text_bold)

    return styler


def _styled_highlight_max(
    styler: "pd.io.formats.style.Styler",
) -> "pd.io.formats.style.Styler":
    """
    Column-wise max-highlight using apply() (axis=0) so the style entry
    is written independently from any background_gradient entry.
    Also guarantees all other cells get white text via applymap.
    """
    styler = styler.apply(_highlight_max_dark, axis=0)
    styler = _styler_map(styler, _white_text_bold)
    return styler


# ══════════════════════════════════════════════════════════════════════════════
# ── Table height helper ───────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _table_height(n_rows: int, row_px: int = 36, header_px: int = 40,
                  min_px: int = 220, max_px: int = 620) -> int:
    """
    Compute a sensible explicit height (px) for st.dataframe() based on
    the number of rows so GlideDataGrid is neither collapsed nor oversized.
    """
    height = header_px + (max(n_rows, 1) * row_px)
    return int(max(min_px, min(max_px, height)))


# ══════════════════════════════════════════════════════════════════════════════
# ── Internal helpers ──────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _call_filter_dataframe(df_full: pd.DataFrame, filters: dict) -> pd.DataFrame:
    rating_range = filters.get("rating_range", (0.0, 5.0))
    if isinstance(rating_range, (list, tuple)) and len(rating_range) == 2:
        r_min, r_max = float(rating_range[0]), float(rating_range[1])
    else:
        r_min = float(filters.get("rating_min", 0.0))
        r_max = float(filters.get("rating_max", 5.0))

    save_filters_to_state(filters)

    df = filter_dataframe(
        df_full,
        countries=filters.get("countries", []),
        cities=filters.get("cities", []),
        cuisines=filters.get("cuisines", []),
        price_ranges=filters.get("price_ranges", []),
        rating_range=(r_min, r_max),
    )
    return df.reset_index(drop=True)


def _ensure_is_rated(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)
    if "Is Rated" not in df.columns:
        if "Aggregate rating" in df.columns:
            df["Is Rated"] = df["Aggregate rating"] > 0
        else:
            df["Is Rated"] = False
    else:
        df["Is Rated"] = df["Is Rated"].astype(bool)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# ── Page entry-point ──────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def show():
    inject_global_css()

    # ── Load & filter ──────────────────────────────────────────────────────────
    try:
        df_full = load_and_preprocess()
    except Exception as exc:
        st.error(f"Failed to load data: {exc}")
        st.stop()

    try:
        filters = render_sidebar_filters(df_full, prefix="eda_")
        df      = _call_filter_dataframe(df_full, filters)
    except Exception as exc:
        st.warning(f"Filters could not be applied: {exc}")
        df = df_full.copy()

    df = _ensure_is_rated(df.reset_index(drop=True))

    if df.empty:
        st.warning("No data matches the current filters. Please adjust the filters above.")
        st.stop()

    # ── Page Header ────────────────────────────────────────────────────────────
    rated_count = int((df["Aggregate rating"] > 0).sum()) if "Aggregate rating" in df.columns else 0
    avg_votes   = int(df["Votes"].mean()) if "Votes" in df.columns else 0

    st.title("🔬 Advanced EDA")
    st.caption(
        f"Deep statistical exploration · {len(df):,} restaurants · "
        f"{rated_count:,} rated · avg {avg_votes:,} votes · "
        "Filters set here propagate to all analytics pages"
    )

    data_quality_badge(df)

    st.divider()

    # ── Tabs ───────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Distributions",
        "🍽️ Cuisine",
        "🏙️ City & Country",
        "🔥 Heatmap",
        "📋 Statistics",
    ])

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 1 – Distributions
    # ─────────────────────────────────────────────────────────────────────────
    with tab1:
        if "Aggregate rating" in df.columns:
            rated_df    = df[df["Aggregate rating"] > 0]
            rated_pct   = round(len(rated_df) / len(df) * 100, 1) if len(df) > 0 else 0
            avg_rating  = round(rated_df["Aggregate rating"].mean(), 2) if not rated_df.empty else 0
            max_votes   = int(df["Votes"].max()) if "Votes" in df.columns else 0
            top_rating  = df["Rating text"].value_counts().index[0] if "Rating text" in df.columns else "N/A"

            ic1, ic2, ic3, ic4 = st.columns(4)
            with ic1: st.metric("⭐ Avg Rating",  str(avg_rating),       "Rated restaurants only")
            with ic2: st.metric("📊 Rated %",     f"{rated_pct}%",       f"{len(rated_df):,} of {len(df):,}")
            with ic3: st.metric("🗳️ Peak Votes",  f"{max_votes:,}",      "Single restaurant")
            with ic4: st.metric("🏅 Top Tier",    str(top_rating),       "Most common rating")

            st.markdown("")

        section_header("⭐ Rating Distributions")
        r1, r2 = st.columns(2)
        with r1:
            try:
                st.plotly_chart(rating_distribution(df), use_container_width=True,
                                key="eda_rating_distribution")
            except Exception as exc:
                st.warning(f"Rating distribution unavailable: {exc}")
        with r2:
            try:
                st.plotly_chart(votes_histogram(df), use_container_width=True,
                                key="eda_votes_histogram")
            except Exception as exc:
                st.warning(f"Votes histogram unavailable: {exc}")

        st.markdown("")
        section_header("🎻 Violin — Rating by Price Tier")
        try:
            violin_df = _ensure_is_rated(df.copy().reset_index(drop=True))
            if "Price Label" not in violin_df.columns and "Price range" in violin_df.columns:
                violin_df["Price Label"] = pd.cut(
                    violin_df["Price range"].clip(1, 4),
                    bins=[0, 1, 2, 3, 4],
                    labels=["Budget", "Affordable", "Premium", "Luxury"],
                ).astype(str)
            st.plotly_chart(violin_rating_by_price(violin_df), use_container_width=True,
                            key="eda_violin_price")
        except Exception as exc:
            st.warning(f"Violin chart unavailable: {exc}")

        st.markdown("")
        section_header("💰 Rating vs. Cost Scatter")
        try:
            scatter_df = _ensure_is_rated(df.copy().reset_index(drop=True))
            st.plotly_chart(rating_vs_cost_scatter(scatter_df), use_container_width=True,
                            key="eda_scatter_cost_rating")
        except Exception as exc:
            st.warning(f"Scatter chart unavailable: {exc}")

        st.markdown("")
        section_header("🛵 Service Feature Adoption")
        try:
            st.plotly_chart(delivery_booking_bar(df), use_container_width=True,
                            key="eda_delivery_booking_bar")
        except Exception as exc:
            st.warning(f"Service chart unavailable: {exc}")

        st.markdown("")
        section_header("🔗 Numeric Correlations")
        corr_cols = [
            c for c in [
                "Aggregate rating", "Votes", "Average Cost for two",
                "Price range", "Cuisine Count",
            ]
            if c in df.columns
        ]
        if len(corr_cols) > 1:
            try:
                num  = df[corr_cols].apply(pd.to_numeric, errors="coerce").dropna()
                corr = num.corr().round(3)

                # v6 FIX: RdBu_r replaces RdYlGn (no light-yellow band).
                # fix_near_zero=True patches the pale mid-grey cells
                # (|r| < 0.15) with a dark charcoal background so white
                # text stays legible across the full -1 → +1 range.
                styled_corr = (
                    corr.style
                   .set_properties(**{
                   "background-color": "#2563eb",
                   "color": "#ffffff",
                   "font-weight": "600"
    })
)
                
                st.dataframe(
                    styled_corr,
                    use_container_width=True,
                    height=_table_height(
                        len(corr), row_px=44, header_px=46,
                        min_px=260, max_px=420,
                    ),
                )
            except Exception as exc:
                st.warning(f"Correlation matrix unavailable: {exc}")
        else:
            st.info("Not enough numeric columns to compute correlations.")

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 2 – Cuisine
    # ─────────────────────────────────────────────────────────────────────────
    with tab2:
        if "Primary Cuisine" in df.columns:
            total_cuisines  = df["Primary Cuisine"].nunique()
            top_cuisine     = df["Primary Cuisine"].value_counts().index[0] if not df.empty else "N/A"
            top_cuisine_cnt = int(df["Primary Cuisine"].value_counts().iloc[0]) if not df.empty else 0
            multi           = int((df["Cuisine Count"] > 1).sum()) if "Cuisine Count" in df.columns else 0

            cc1, cc2, cc3 = st.columns(3)
            with cc1: st.metric("🍽️ Unique Cuisines", str(total_cuisines))
            with cc2: st.metric("🥇 Top Cuisine",      top_cuisine,       f"{top_cuisine_cnt:,} restaurants")
            with cc3: st.metric("🔀 Multi-Cuisine",    f"{multi:,}",      "Serve 2+ cuisines")

            st.markdown("")

        section_header("🗺️ Cuisine Treemap")
        n_cuisines = st.slider("Number of cuisines to show", 10, 40, 25,
                               key="eda_treemap_slider")
        try:
            st.plotly_chart(cuisine_treemap(df, top_n=n_cuisines), use_container_width=True,
                            key="eda_cuisine_treemap")
        except Exception as exc:
            st.warning(f"Cuisine treemap unavailable: {exc}")

        st.markdown("")
        section_header("⭐ Top Cuisines by Avg Rating")
        top_n_rating = st.slider(
            "Top N cuisines (min 20 restaurants)", 5, 25, 15,
            key="eda_cuisine_rating_slider",
        )
        try:
            cuisine_df = _ensure_is_rated(df.copy().reset_index(drop=True))
            st.plotly_chart(
                top_cuisines_avg_rating(cuisine_df, top_n=top_n_rating),
                use_container_width=True,
                key="eda_top_cuisines_rating_bar",
            )
        except Exception as exc:
            st.warning(f"Top cuisines chart unavailable: {exc}")

        st.markdown("")
        section_header("📊 Cuisine Detail Table")

        if "Primary Cuisine" in df.columns:
            try:
                agg_dict: dict = {
                    "Restaurants": ("Restaurant ID", "count"),
                    "Avg_Rating": (
                        "Aggregate rating",
                        lambda x: round(x[x > 0].mean(), 2) if (x > 0).any() else 0.0,
                    ),
                    "Total_Votes": ("Votes", "sum"),
                    "Avg_Cost":    ("Average Cost for two", "median"),
                }
                if "Has Online delivery" in df.columns:
                    agg_dict["Delivery_Pct"] = (
                        "Has Online delivery",
                        lambda x: f"{x.astype(bool).mean() * 100:.1f}%",
                    )
                cuisine_stats = (
                    df.groupby("Primary Cuisine")
                    .agg(**agg_dict)
                    .sort_values("Restaurants", ascending=False)
                    .reset_index()
                )
                st.dataframe(cuisine_stats.head(50), use_container_width=True, hide_index=True)
            except Exception as exc:
                st.warning(f"Cuisine table unavailable: {exc}")
        else:
            st.info("Primary Cuisine column not found.")

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 3 – City & Country
    # ─────────────────────────────────────────────────────────────────────────
    with tab3:
        if "City" in df.columns:
            total_cities    = df["City"].nunique()
            top_city        = df["City"].value_counts().index[0] if not df.empty else "N/A"
            top_city_cnt    = int(df["City"].value_counts().iloc[0]) if not df.empty else 0
            total_countries = df["Country"].nunique() if "Country" in df.columns else 0

            gc1, gc2, gc3 = st.columns(3)
            with gc1: st.metric("🏙️ Unique Cities", f"{total_cities:,}")
            with gc2: st.metric("🥇 Top City",       top_city,        f"{top_city_cnt:,} restaurants")
            with gc3: st.metric("🌍 Countries",      str(total_countries))

            st.markdown("")

        section_header("🏙️ Top Cities by Restaurant Count")
        n_cities = st.slider("Top N cities", 5, 30, 15, key="eda_city_slider")
        try:
            st.plotly_chart(city_bar(df, top_n=n_cities), use_container_width=True,
                            key="eda_city_bar_chart")
        except Exception as exc:
            st.warning(f"City bar chart unavailable: {exc}")

        st.markdown("")
        section_header("🌍 Countries Overview")
        try:
            st.plotly_chart(country_bar(df), use_container_width=True,
                            key="eda_country_bar_chart")
        except Exception as exc:
            st.warning(f"Country bar chart unavailable: {exc}")

        st.markdown("")
        section_header("💰 Median Cost by Country")
        try:
            st.plotly_chart(cost_by_country(df), use_container_width=True,
                            key="eda_cost_by_country")
        except Exception as exc:
            st.warning(f"Cost by country chart unavailable: {exc}")

        st.markdown("")
        section_header("📋 City Intelligence Table")
        if "City" in df.columns:
            try:
                city_agg: dict = {
                    "Restaurants": ("Restaurant ID", "count"),
                    "Avg_Rating": (
                        "Aggregate rating",
                        lambda x: round(x[x > 0].mean(), 2) if (x > 0).any() else 0.0,
                    ),
                    "Total_Votes": ("Votes", "sum"),
                    "Median_Cost": ("Average Cost for two", "median"),
                }
                if "Has Online delivery" in df.columns:
                    city_agg["Delivery_Pct"] = (
                        "Has Online delivery",
                        lambda x: f"{x.astype(bool).mean() * 100:.1f}%",
                    )
                if "Has Table booking" in df.columns:
                    city_agg["Booking_Pct"] = (
                        "Has Table booking",
                        lambda x: f"{x.astype(bool).mean() * 100:.1f}%",
                    )
                group_cols = (
                    ["City", "Country"] if "Country" in df.columns else ["City"]
                )
                city_stats = (
                    df.groupby(group_cols)
                    .agg(**city_agg)
                    .sort_values("Restaurants", ascending=False)
                    .reset_index()
                )
                st.dataframe(city_stats.head(40), use_container_width=True, hide_index=True)
            except Exception as exc:
                st.warning(f"City table unavailable: {exc}")
        else:
            st.info("City column not found.")

        st.markdown("")
        section_header("🌏 Country Intelligence Table")
        if "Country" in df.columns:
            try:
                country_agg: dict = {
                    "Restaurants": ("Restaurant ID", "count"),
                    "Cities":      ("City", "nunique"),
                    "Avg_Rating": (
                        "Aggregate rating",
                        lambda x: round(x[x > 0].mean(), 2) if (x > 0).any() else 0.0,
                    ),
                    "Total_Votes": ("Votes", "sum"),
                }
                if "Has Online delivery" in df.columns:
                    country_agg["Delivery_Pct"] = (
                        "Has Online delivery",
                        lambda x: f"{x.astype(bool).mean() * 100:.1f}%",
                    )
                if "Has Table booking" in df.columns:
                    country_agg["Booking_Pct"] = (
                        "Has Table booking",
                        lambda x: f"{x.astype(bool).mean() * 100:.1f}%",
                    )
                country_stats = (
                    df.groupby("Country")
                    .agg(**country_agg)
                    .sort_values("Restaurants", ascending=False)
                    .reset_index()
                )
                st.dataframe(country_stats, use_container_width=True, hide_index=True)
            except Exception as exc:
                st.warning(f"Country table unavailable: {exc}")
        else:
            st.info("Country column not found.")

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 4 – Heatmap
    # ─────────────────────────────────────────────────────────────────────────
    with tab4:
        st.info(
            "**Heatmap Intelligence** — Cross-dimensional rating analysis reveals which "
            "city-cuisine combinations consistently outperform or underperform. "
            "Darker cells indicate higher average ratings."
        )

        section_header("🔥 Avg Rating Heatmap: City × Cuisine")
        n_top = st.slider("Top N cities to include", 5, 20, 12,
                          key="eda_heatmap_cities_slider")
        try:
            heatmap_df = _ensure_is_rated(df.copy().reset_index(drop=True))
            st.plotly_chart(heatmap_city_rating(heatmap_df, top_n=n_top),
                            use_container_width=True, key="eda_heatmap_city_cuisine")
        except Exception as exc:
            st.warning(f"Heatmap unavailable: {exc}")

        st.markdown("")
        section_header("📊 Price Range × Rating Text Cross-Tab")
        if "Price Label" in df.columns and "Rating text" in df.columns:
            try:
                cross = pd.crosstab(df["Price Label"], df["Rating text"])
                rating_order = [
                    c for c in
                    ["Not rated", "Poor", "Average", "Good", "Very Good", "Excellent"]
                    if c in cross.columns
                ]
                extra = [c for c in cross.columns if c not in rating_order]
                cross = cross.reindex(columns=rating_order + extra, fill_value=0)

                # v6 FIX: Blues cmap (all dark enough), white text via applymap
                styled_cross = _styled_gradient(
                    cross.style,
                    cmap="Blues",
                    axis=None,
                )
                st.dataframe(
                    styled_cross,
                    use_container_width=True,
                    height=_table_height(
                        len(cross), row_px=40, header_px=46,
                        min_px=220, max_px=420,
                    ),
                )
            except Exception as exc:
                st.warning(f"Cross-tab unavailable: {exc}")
        else:
            st.info("Price Label or Rating text columns not found.")

        st.markdown("")
        section_header("📊 Delivery × Booking Cross-Tab by Country")
        if "Country" in df.columns:
            try:
                cross2_agg: dict = {"Restaurants": ("Restaurant ID", "count")}
                if "Has Online delivery" in df.columns:
                    cross2_agg["Online_Delivery"] = (
                        "Has Online delivery",
                        lambda x: f"{x.astype(bool).mean() * 100:.1f}%",
                    )
                if "Has Table booking" in df.columns:
                    cross2_agg["Table_Booking"] = (
                        "Has Table booking",
                        lambda x: f"{x.astype(bool).mean() * 100:.1f}%",
                    )
                if "Is delivering now" in df.columns:
                    cross2_agg["Delivering_Now"] = (
                        "Is delivering now",
                        lambda x: f"{x.astype(bool).mean() * 100:.1f}%",
                    )
                cross2 = (
                    df.groupby("Country")
                    .agg(**cross2_agg)
                    .sort_values("Restaurants", ascending=False)
                    .reset_index()
                )
                st.dataframe(cross2, use_container_width=True, hide_index=True)
            except Exception as exc:
                st.warning(f"Country cross-tab unavailable: {exc}")
        else:
            st.info("Country column not found.")

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 5 – Statistics
    # ─────────────────────────────────────────────────────────────────────────
    with tab5:
        null_cols     = int(df.isnull().any().sum())
        complete_cols = len(df.columns) - null_cols

        sc1, sc2, sc3 = st.columns(3)
        with sc1: st.metric("📐 Total Features",   len(df.columns))
        with sc2: st.metric("✅ Complete Columns",  complete_cols)
        with sc3: st.metric("⚠️ Columns w/ Nulls", null_cols)

        st.markdown("")

        section_header("📋 Descriptive Statistics")
        try:
            stats = descriptive_stats_table(df)

            # v6 FIX: Blues cmap (uniformly dark cells), white text via
            # applymap written as a SEPARATE style entry so GlideDataGrid
            # cannot clobber it with gradient's auto dark-text color.
            styled_stats = (
                      stats.style
                     .set_properties(**{
                     "background-color": "#2563eb",
                     "color": "#ffffff",
                     "font-weight": "600"
    })
)
            st.dataframe(
                styled_stats,
                use_container_width=True,
                height=_table_height(
                    len(stats), row_px=42, header_px=46,
                    min_px=320, max_px=520,
                ),
            )
        except Exception as exc:
            st.warning(f"Descriptive stats unavailable: {exc}")

        st.markdown("")
        section_header("🔍 Missing Value Analysis")
        try:
            missing = df.isnull().sum().reset_index()
            missing.columns = ["Column", "Missing Count"]
            missing["Missing %"] = (missing["Missing Count"] / len(df) * 100).round(2)
            missing["Status"]    = missing["Missing Count"].apply(
                lambda x: "✅ Complete" if x == 0 else "⚠️ Has Nulls"
            )
            st.dataframe(missing, use_container_width=True, hide_index=True)
        except Exception as exc:
            st.warning(f"Missing value analysis unavailable: {exc}")

        st.markdown("")
        section_header("📊 Data Type Overview")
        try:
            dtypes = pd.DataFrame({
                "Column":        df.columns.tolist(),
                "Dtype":         df.dtypes.astype(str).tolist(),
                "Unique Values": [df[c].nunique() for c in df.columns],
                "Sample Value":  [
                    str(df[c].dropna().iloc[0]) if not df[c].dropna().empty else "N/A"
                    for c in df.columns
                ],
            })
            st.dataframe(dtypes, use_container_width=True, hide_index=True)
        except Exception as exc:
            st.warning(f"Data type overview unavailable: {exc}")


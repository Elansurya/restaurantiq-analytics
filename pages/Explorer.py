import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd

from src.components import (
    inject_global_css,
    render_sidebar_filters,
    section_header,
    data_quality_badge,
)
from src.preprocessing import load_and_preprocess, filter_dataframe


def show():
    inject_global_css()

    try:
        df_full = load_and_preprocess()
    except Exception as exc:
        st.error(f"Failed to load data: {exc}")
        st.stop()

    try:
        filters = render_sidebar_filters(df_full, prefix="explorer_")
        df = filter_dataframe(
            df_full,
            countries=filters.get("countries", []),
            cities=filters.get("cities", []),
            cuisines=filters.get("cuisines", []),
            price_ranges=filters.get("price_ranges", []),
            rating_range=filters.get("rating_range", (0.0, 5.0)),
        )
    except Exception as exc:
        st.warning(f"Filters could not be applied: {exc}")
        df = df_full.copy()

    df = df.reset_index(drop=True)

    if df.empty:
        st.warning("No data matches the current filters. Please adjust the sidebar.")
        st.stop()

    # ── Page Header ────────────────────────────────────────────────────────────
    filter_ratio = round(len(df) / len(df_full) * 100, 1)

    st.title("🔎 Data Explorer")
    st.caption(
        f"Browse, search, and export the filtered restaurant dataset · "
        f"{len(df):,} restaurants ({filter_ratio}% of dataset) · "
        f"{len(df.columns)} columns · Refine filters in the sidebar"
    )

    data_quality_badge(df)

    st.divider()

    # ── Filter Summary Metrics ─────────────────────────────────────────────────
    st.subheader("Filter Summary")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🍽️ Restaurants", f"{len(df):,}")
    c2.metric("🌍 Countries",
              df["Country"].nunique() if "Country" in df.columns else "N/A")
    c3.metric("🏙️ Cities",
              f"{df['City'].nunique():,}" if "City" in df.columns else "N/A")
    c4.metric("🍛 Cuisines",
              df["Primary Cuisine"].nunique() if "Primary Cuisine" in df.columns else "N/A")

    st.divider()

    # ── Column Selector ────────────────────────────────────────────────────────
    st.subheader("Configure View")
    st.caption(
        "Choose which fields to display — the table updates instantly. "
        "Use the search box below to filter by restaurant name."
    )

    all_cols     = df.columns.tolist()
    default_cols = [
        c for c in [
            "Restaurant Name", "City", "Country", "Primary Cuisine",
            "Aggregate rating", "Rating text", "Price Label",
            "Votes", "Has Online delivery", "Has Table booking",
        ]
        if c in all_cols
    ]
    selected_cols = st.multiselect(
        "Columns to display",
        options=all_cols,
        default=default_cols,
        key="explorer_column_selector",
    )

    if not selected_cols:
        st.info("Select at least one column to display the table.")
        st.stop()

    # ── Search Box ─────────────────────────────────────────────────────────────
    search_col, _ = st.columns([2, 3])
    with search_col:
        search_term = st.text_input(
            "🔍 Search by restaurant name",
            placeholder="Type to filter…",
            key="explorer_search_input",
        )

    view_df = df[selected_cols].copy().reset_index(drop=True)

    if search_term and "Restaurant Name" in view_df.columns:
        mask = (
            view_df["Restaurant Name"]
            .astype(str)
            .str.contains(search_term, case=False, na=False)
        )
        view_df = view_df.loc[mask].reset_index(drop=True)

    st.divider()

    # ── Data Table ─────────────────────────────────────────────────────────────
    st.subheader("Results")

    result_caption = f"{len(view_df):,} rows · {len(selected_cols)} columns selected"
    if search_term:
        result_caption += f" · Filtered by: \"{search_term}\""
    st.caption(result_caption)

    st.dataframe(view_df, use_container_width=True, hide_index=True)

    st.divider()

    # ── Download ───────────────────────────────────────────────────────────────
    st.subheader("Export Filtered Dataset")
    st.caption("Downloads the current view with selected columns and search filters applied.")

    csv = view_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download filtered data as CSV",
        data=csv,
        file_name="restaurantiq_filtered.csv",
        mime="text/csv",
        key="explorer_csv_download",
    )


"""
RestaurantIQ – Filter Deduplication Patch Guide
================================================

This file documents the minimal changes required in pages/dashboard.py
(Descriptive Analysis) and pages/explorer.py (Data Visualization) to
remove their duplicate filter UIs and instead consume the shared filtered
dataframe from session state.

No analytics logic, chart logic, or business logic changes are needed.

──────────────────────────────────────────────────────────────────────────
STEP 1 – UPDATE IMPORTS  (both files)
──────────────────────────────────────────────────────────────────────────

REMOVE these imports (if present):
    from src.components import render_sidebar_filters

ADD these imports:
    from src.components import (
        get_filtered_dataframe,
        render_active_filter_badge,
    )

──────────────────────────────────────────────────────────────────────────
STEP 2 – REPLACE THE FILTER BLOCK  (both files)
──────────────────────────────────────────────────────────────────────────

Find the section that looks like:

    filters = render_sidebar_filters(df_full, prefix="<page>_")
    df = filter_dataframe(df_full, ...)        # or _call_filter_dataframe(...)

Replace it with:

    df = get_filtered_dataframe(df_full)

That single call reads the shared session-state filters that the Advanced
EDA page writes on every rerun, applies filter_dataframe() internally, and
returns the filtered DataFrame.

──────────────────────────────────────────────────────────────────────────
STEP 3 – ADD THE ACTIVE-FILTER BADGE  (both files, optional but recommended)
──────────────────────────────────────────────────────────────────────────

Directly after the page header (before the first st.metric / chart), add:

    render_active_filter_badge(df_full, df)

This renders a compact, read-only pill bar showing which filters are active
and how many rows are in scope.  No interactive widgets — purely display.

──────────────────────────────────────────────────────────────────────────
EXAMPLE – dashboard.py show() skeleton after the patch
──────────────────────────────────────────────────────────────────────────

def show():
    inject_global_css()

    try:
        df_full = load_and_preprocess()
    except Exception as exc:
        st.error(f"Failed to load data: {exc}")
        st.stop()

    # ── Shared filter (set in Advanced EDA, persisted in session state) ────
    df = get_filtered_dataframe(df_full)

    if df.empty:
        st.warning("No data matches the active filters. Adjust them on the Advanced EDA page.")
        st.stop()

    # ── Page header ────────────────────────────────────────────────────────
    st.markdown("... your existing header HTML ...")

    # ── Active-filter badge (read-only) ────────────────────────────────────
    render_active_filter_badge(df_full, df)

    # ... all existing charts / KPIs / tables using df ...

──────────────────────────────────────────────────────────────────────────
EXAMPLE – explorer.py show() skeleton after the patch
──────────────────────────────────────────────────────────────────────────

def show():
    inject_global_css()

    try:
        df_full = load_and_preprocess()
    except Exception as exc:
        st.error(f"Failed to load data: {exc}")
        st.stop()

    # ── Shared filter (set in Advanced EDA, persisted in session state) ────
    df = get_filtered_dataframe(df_full)

    if df.empty:
        st.warning("No data matches the active filters. Adjust them on the Advanced EDA page.")
        st.stop()

    # ── Page header ────────────────────────────────────────────────────────
    st.markdown("... your existing header HTML ...")

    # ── Active-filter badge (read-only) ────────────────────────────────────
    render_active_filter_badge(df_full, df)

    # ... all existing table / profiler / export logic using df ...

──────────────────────────────────────────────────────────────────────────
NOTHING ELSE CHANGES
──────────────────────────────────────────────────────────────────────────
• All chart functions, KPI calculations, aggregations, and export logic
  remain exactly as they are — they just receive df instead of computing
  their own filtered copy.
• No changes to src/visualization.py, src/preprocessing.py, or app.py.
• No changes to any ML pipeline pages.
"""
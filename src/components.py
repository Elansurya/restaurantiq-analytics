from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd
import streamlit as st


# ══════════════════════════════════════════════════════════════════════════════
# ── Global CSS injection — TRUE NO-OP ─────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def inject_global_css() -> None:
    """TRUE NO-OP — all CSS lives in assets/style.css loaded once by app.py."""
    pass


# ══════════════════════════════════════════════════════════════════════════════
# ── Hero banner — Home page ───────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def render_hero_home(total_restaurants: int = 9054, countries: int = 15, **kwargs) -> None:
    """
    Premium hero card for the Home page.
    All styling lives in assets/style.css (.iq-hero-* classes).
    Icons are rendered as emoji to avoid Streamlit's HTML sanitizer
    stripping external <link> tags (Font Awesome) and breaking the
    surrounding markup.
    """
    hero_html = f"""<div class="iq-hero-card">
<div class="iq-hero-glow2"></div>
<div class="iq-hero-edition">
<span class="iq-hero-edition-dot"></span>
Enterprise Analytics Edition · 2026
</div>
<div class="iq-hero-title">RestaurantIQ</div>
<div class="iq-hero-subtitle">AI-Powered Restaurant Intelligence Platform</div>
<p class="iq-hero-desc">
Unlock the intelligence behind
<strong>{total_restaurants:,} restaurants</strong>
across
<strong>{countries} countries</strong>.
Discover cuisine trends, geospatial hotspots,
and performance drivers through cinematic analytics
and machine-learning insights.
</p>
<div class="iq-hero-divider"></div>
<div class="iq-hero-chips">
<div class="iq-hero-chip">
<div class="iq-hero-chip-icon violet">🍽️</div>
<div class="iq-hero-chip-body">
<span class="iq-hero-chip-value">{total_restaurants:,}</span>
<span class="iq-hero-chip-label">Restaurants</span>
</div>
</div>
<div class="iq-hero-chip">
<div class="iq-hero-chip-icon teal">🌍</div>
<div class="iq-hero-chip-body">
<span class="iq-hero-chip-value">{countries}</span>
<span class="iq-hero-chip-label">Countries</span>
</div>
</div>
<div class="iq-hero-chip">
<div class="iq-hero-chip-icon amber">🏙️</div>
<div class="iq-hero-chip-body">
<span class="iq-hero-chip-value">141</span>
<span class="iq-hero-chip-label">Cities</span>
</div>
</div>
<div class="iq-hero-chip">
<div class="iq-hero-chip-icon sky">⭐</div>
<div class="iq-hero-chip-body">
<span class="iq-hero-chip-value">1.48M</span>
<span class="iq-hero-chip-label">Reviews</span>
</div>
</div>
</div>
</div>"""

    st.markdown(hero_html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ── Hero banner stubs (other pages) — no-ops ──────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def render_hero_descriptive(**kwargs) -> None:
    pass

def render_hero_eda(**kwargs) -> None:
    pass

def render_hero_geospatial(**kwargs) -> None:
    pass

def render_hero_data_viz(**kwargs) -> None:
    pass

def render_hero_ml_pipeline(**kwargs) -> None:
    pass

def render_hero_ai_predictor(**kwargs) -> None:
    pass

def render_hero_recommendations(**kwargs) -> None:
    pass

def render_hero_features(**kwargs) -> None:
    pass

def render_hero_success_score(**kwargs) -> None:
    pass


# ── Section header ─────────────────────────────────────────────────────────────

def section_header(title: str, subtitle: str = "") -> None:
    """Render a premium gradient section header with high-contrast text."""
    st.markdown(
        f"""<div class="iq-section-wrap">
<div class="iq-section-bar"></div>
<div class="iq-section-title">{title}</div>
</div>""",
        unsafe_allow_html=True,
    )
    if subtitle:
        st.caption(subtitle)


# ── Data quality badge ─────────────────────────────────────────────────────────

def data_quality_badge(df: pd.DataFrame) -> None:
    """Render a data quality indicator row using native Streamlit metrics."""
    n_rows       = len(df)
    n_cols       = len(df.columns)
    n_missing    = int(df.isnull().sum().sum())
    missing_pct  = round(n_missing / max(n_rows * n_cols, 1) * 100, 1)
    completeness = round(100 - missing_pct, 1)

    if completeness >= 95:
        quality_label = "✅ Excellent data quality"
    elif completeness >= 80:
        quality_label = "⚠️ Good data quality"
    else:
        quality_label = "❗ Data quality needs attention"

    c1, c2, c3, c4 = st.columns(4, gap="small")
    with c1:
        st.metric("Rows", f"{n_rows:,}")
    with c2:
        st.metric("Columns", str(n_cols))
    with c3:
        st.metric("Completeness", f"{completeness}%")
    with c4:
        st.metric("Nulls", f"{n_missing:,}")

    if completeness >= 95:
        st.success(quality_label)
    elif completeness >= 80:
        st.warning(quality_label)
    else:
        st.error(quality_label)


# ══════════════════════════════════════════════════════════════════════════════
# ── Shared filter state helpers ───────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

_FILTER_STATE_KEY = "iq_shared_filters"


def _default_filters() -> dict:
    return {
        "countries":    [],
        "cities":       [],
        "cuisines":     [],
        "price_ranges": [],
        "rating_range": (0.0, 5.0),
    }


def init_filter_state() -> None:
    if _FILTER_STATE_KEY not in st.session_state:
        st.session_state[_FILTER_STATE_KEY] = _default_filters()


def save_filters_to_state(filters: dict) -> None:
    st.session_state[_FILTER_STATE_KEY] = {
        "countries":    filters.get("countries",    []),
        "cities":       filters.get("cities",       []),
        "cuisines":     filters.get("cuisines",     []),
        "price_ranges": filters.get("price_ranges", []),
        "rating_range": filters.get("rating_range", (0.0, 5.0)),
    }


def get_filters_from_state() -> dict:
    init_filter_state()
    return dict(st.session_state[_FILTER_STATE_KEY])


def get_filtered_dataframe(df_full: pd.DataFrame) -> pd.DataFrame:
    """Apply the shared session-state filters to df_full and return the result."""
    from src.preprocessing import filter_dataframe

    filters = get_filters_from_state()

    rating_range = filters.get("rating_range", (0.0, 5.0))
    if isinstance(rating_range, (list, tuple)) and len(rating_range) == 2:
        r_min, r_max = float(rating_range[0]), float(rating_range[1])
    else:
        r_min, r_max = 0.0, 5.0

    try:
        df = filter_dataframe(
            df_full,
            countries=filters.get("countries",    []),
            cities=filters.get("cities",          []),
            cuisines=filters.get("cuisines",      []),
            price_ranges=filters.get("price_ranges", []),
            rating_range=(r_min, r_max),
        )
        return df.reset_index(drop=True)
    except Exception:
        return df_full.copy().reset_index(drop=True)


def render_active_filter_badge(df_full: pd.DataFrame, df_filtered: pd.DataFrame) -> None:
    """Show a compact banner describing active filters and row counts."""
    filters  = get_filters_from_state()
    total    = len(df_full)
    filtered = len(df_filtered)

    is_active = any([
        filters.get("countries"),
        filters.get("cities"),
        filters.get("cuisines"),
        filters.get("price_ranges"),
        filters.get("rating_range", (0.0, 5.0)) != (0.0, 5.0),
    ])

    if not is_active:
        st.info(
            f"🔍 **No filters active** — showing all {total:,} restaurants.  "
            "Adjust filters on the **Data Exploration** page."
        )
        return

    parts: list[str] = []
    for c in filters.get("countries", []):
        parts.append(f"🌍 {c}")
    for c in filters.get("cities", []):
        parts.append(f"🏙️ {c}")
    for c in filters.get("cuisines", []):
        parts.append(f"🍽️ {c}")
    for p in filters.get("price_ranges", []):
        parts.append(f"💰 {p}")

    r_range = filters.get("rating_range", (0.0, 5.0))
    if isinstance(r_range, (list, tuple)) and len(r_range) == 2:
        r_min, r_max = float(r_range[0]), float(r_range[1])
        if (r_min, r_max) != (0.0, 5.0):
            parts.append(f"⭐ {r_min}–{r_max}")

    pct        = round(filtered / total * 100, 1) if total else 0.0
    filter_str = "  |  ".join(parts) if parts else "Custom range"

    st.success(
        f"🔍 **Active Filters:** {filter_str}  ·  "
        f"**{filtered:,} / {total:,}** restaurants ({pct}%)"
    )


def _filter_pill(label: str, color: str) -> str:
    """Legacy helper — kept for backward compatibility."""
    return f'<span class="insight-chip" style="color:{color};">{label}</span>'


# ── Inline filters (main content area) ────────────────────────────────────────

def render_sidebar_filters(df: pd.DataFrame, prefix: str = "") -> dict:
    """
    Render filter widgets inside the MAIN content area.
    Layout: Row 1 — Countries · Cities · Cuisines · Price Range (4 columns)
            Row 2 — Rating Range (full width)
    """
    init_filter_state()
    saved = get_filters_from_state()

    filters: dict = {}

    st.markdown(
        """<div style="
background: rgba(15,23,42,0.85);
border: 1px solid rgba(108,99,255,0.30);
border-radius: 12px;
padding: 12px 16px 6px;
margin-bottom: 16px;
">
<div style="
font-size: 0.72rem;
font-weight: 700;
letter-spacing: 0.12em;
text-transform: uppercase;
color: #93C5FD;
margin-bottom: 6px;
">Filter Controls</div>
</div>""",
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4, gap="medium")

    with col1:
        if "Country" in df.columns:
            all_countries = sorted(df["Country"].dropna().unique().tolist())
            filters["countries"] = st.multiselect(
                "🌍 Countries",
                options=all_countries,
                default=saved.get("countries", []),
                key=f"{prefix}countries",
                placeholder="All countries…",
            )
        else:
            filters["countries"] = []

    with col2:
        if "City" in df.columns:
            country_filtered = df.copy()
            if filters.get("countries"):
                country_filtered = country_filtered[
                    country_filtered["Country"].isin(filters["countries"])
                ]
            all_cities = sorted(country_filtered["City"].dropna().unique().tolist())
            valid_saved_cities = [c for c in saved.get("cities", []) if c in all_cities]
            filters["cities"] = st.multiselect(
                "🏙️ Cities",
                options=all_cities,
                default=valid_saved_cities,
                key=f"{prefix}cities",
                placeholder="All cities…",
            )
        else:
            filters["cities"] = []

    with col3:
        if "Primary Cuisine" in df.columns:
            all_cuisines = sorted(df["Primary Cuisine"].dropna().unique().tolist())
            valid_saved_cuisines = [c for c in saved.get("cuisines", []) if c in all_cuisines]
            filters["cuisines"] = st.multiselect(
                "🍽️ Cuisines",
                options=all_cuisines,
                default=valid_saved_cuisines,
                key=f"{prefix}cuisines",
                placeholder="All cuisines…",
            )
        else:
            filters["cuisines"] = []

    with col4:
        if "Price Label" in df.columns:
            all_price = sorted(df["Price Label"].dropna().unique().tolist())
            valid_saved_price = [p for p in saved.get("price_ranges", []) if p in all_price]
            filters["price_ranges"] = st.multiselect(
                "💰 Price Range",
                options=all_price,
                default=valid_saved_price,
                key=f"{prefix}price_ranges",
                placeholder="All tiers…",
            )
        else:
            filters["price_ranges"] = []

    if "Aggregate rating" in df.columns:
        min_r = float(df["Aggregate rating"].min())
        max_r = float(df["Aggregate rating"].max())
        saved_range = saved.get("rating_range", (min_r, max_r))
        if isinstance(saved_range, (list, tuple)) and len(saved_range) == 2:
            restored_min = max(min_r, float(saved_range[0]))
            restored_max = min(max_r, float(saved_range[1]))
        else:
            restored_min, restored_max = min_r, max_r

        filters["rating_range"] = st.slider(
            "⭐ Rating Range",
            min_value=min_r,
            max_value=max_r,
            value=(restored_min, restored_max),
            step=0.1,
            key=f"{prefix}rating_range",
        )
    else:
        filters["rating_range"] = (0.0, 5.0)

    save_filters_to_state(filters)
    return filters


# ══════════════════════════════════════════════════════════════════════════════
# ── Premium KPI card helper — high contrast ───────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def render_kpi_cards(
    metrics: list[tuple[str, str, str, str]],
    n_cols: int = 4,
) -> None:
    """
    Render a row of premium KPI cards with strong contrast.
    metrics: list of (icon, label, value, accent_color)
    """
    cols = st.columns(n_cols, gap="small")
    for i, (icon, label, value, accent) in enumerate(metrics):
        with cols[i % n_cols]:
            st.markdown(
                f"""<div class="iq-kpi-card" style="border-top: 3px solid {accent};">
<div class="iq-kpi-card-icon" style="background:{accent}22;border:1px solid {accent}55;">
{icon}
</div>
<div class="iq-kpi-card-label">{label}</div>
<div class="iq-kpi-card-value"
style="background:linear-gradient(135deg,{accent},#67E8F9);
-webkit-background-clip:text;-webkit-text-fill-color:transparent;
background-clip:text;">{value}</div>
</div>""",
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# ── Premium prediction result panel ───────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def render_prediction_panel(
    score: str,
    label: str = "Predicted Rating",
    confidence: str | None = None,
    details: list[tuple[str, str]] | None = None,
    accent: str = "#6C63FF",
) -> None:
    """Render a premium prediction result panel with strong text contrast."""
    confidence_html = ""
    if confidence:
        confidence_html = (
            f'<div style="display:inline-flex;align-items:center;gap:6px;'
            f'background:rgba(16,185,129,0.15);border:1px solid rgba(16,185,129,0.40);'
            f'border-radius:999px;padding:5px 16px;'
            f'font-size:0.72rem;color:#6EE7B7;font-weight:700;margin-top:12px;">'
            f'● {confidence}</div>'
        )

    details_html = ""
    if details:
        rows = "".join(
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'padding:8px 0;border-bottom:1px solid rgba(255,255,255,0.08);">'
            f'<span style="font-size:0.78rem;color:#CBD5E1;">{k}</span>'
            f'<span style="font-size:0.72rem;font-weight:700;color:#F8FAFC;'
            f'background:rgba(108,99,255,0.18);border:1px solid rgba(108,99,255,0.30);'
            f'border-radius:6px;padding:2px 10px;">{v}</span>'
            f'</div>'
            for k, v in details
        )
        details_html = (
            f'<div style="margin-top:18px;border-top:1px solid rgba(108,99,255,0.22);'
            f'padding-top:14px;">{rows}</div>'
        )

    st.markdown(
        f'<div style="'
        f'background:linear-gradient(145deg,{accent}18,rgba(0,194,168,0.08));'
        f'border:1px solid {accent}44;border-radius:16px;padding:28px 28px 24px;">'
        f'<div style="font-size:0.72rem;font-weight:700;letter-spacing:0.13em;'
        f'text-transform:uppercase;color:#CBD5E1;margin-bottom:10px;">{label}</div>'
        f'<div style="font-size:3rem;font-weight:900;letter-spacing:-0.03em;'
        f'background:linear-gradient(135deg,{accent},#00C2A8);'
        f'-webkit-background-clip:text;-webkit-text-fill-color:transparent;'
        f'background-clip:text;">{score}</div>'
        f'{confidence_html}'
        f'{details_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# ── Premium recommendation cards ──────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def render_recommendation_cards(
    items: list[dict],
    n_cols: int = 2,
) -> None:
    """Render a grid of premium recommendation cards."""
    cols = st.columns(n_cols, gap="medium")
    for i, item in enumerate(items):
        badge_html = ""
        if item.get("badge"):
            badge_html = (
                f'<div style="display:inline-block;font-size:0.62rem;font-weight:700;'
                f'letter-spacing:0.1em;text-transform:uppercase;'
                f'background:rgba(108,99,255,0.20);color:#A78BFA;'
                f'border:1px solid rgba(108,99,255,0.40);border-radius:999px;'
                f'padding:3px 12px;margin-bottom:10px;">{item["badge"]}</div>'
            )

        chips_html = ""
        if item.get("chips"):
            chips_html = (
                '<div style="margin-top:12px;display:flex;gap:6px;flex-wrap:wrap;">'
                + "".join(
                    f'<span style="font-size:0.68rem;padding:3px 10px;color:#E2E8F0;'
                    f'background:rgba(255,255,255,0.07);border:1px solid rgba(255,255,255,0.12);'
                    f'border-radius:6px;">{c}</span>'
                    for c in item["chips"]
                )
                + "</div>"
            )

        with cols[i % n_cols]:
            st.markdown(
                f'<div style="'
                f'background:linear-gradient(145deg,rgba(15,23,42,0.95),rgba(30,27,75,0.85));'
                f'border:1px solid rgba(108,99,255,0.30);border-radius:14px;'
                f'padding:20px 22px;margin-bottom:10px;">'
                f'{badge_html}'
                f'<div style="font-size:1.0rem;font-weight:700;color:#F8FAFC;'
                f'margin-bottom:6px;">{item.get("title", "")}</div>'
                f'<div style="font-size:0.82rem;color:#E2E8F0;line-height:1.6;">'
                f'{item.get("desc", "")}</div>'
                f'{chips_html}'
                f'</div>',
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# ── Premium insight chips row ─────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def render_insight_chips(
    chips: list[str],
    title: str = "",
    color: str | None = None,
) -> None:
    """Render a horizontal row of insight-chip tags."""
    if title:
        st.caption(title.upper())

    color_style = f"color:{color};" if color else "color:#E2E8F0;"
    chips_html = "".join(
        f'<span style="display:inline-block;padding:4px 12px;'
        f'background:rgba(108,99,255,0.15);border:1px solid rgba(108,99,255,0.35);'
        f'border-radius:8px;font-size:0.78rem;font-weight:600;{color_style}">{c}</span>'
        for c in chips
    )
    st.markdown(
        f'<div style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:16px;">'
        f'{chips_html}</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# ── Stat badge row ─────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def render_stat_badges(
    stats: list[tuple[str, str]],
    modifier: str = "",
) -> None:
    """Render a horizontal row of stat-badge pills."""
    badges_html = "".join(
        f'<span style="display:inline-block;padding:5px 14px;'
        f'background:rgba(15,23,42,0.85);border:1px solid rgba(108,99,255,0.35);'
        f'border-radius:8px;font-size:0.76rem;font-weight:600;color:#F8FAFC;">'
        f'{label}: <strong style="color:#A78BFA;">{value}</strong></span>'
        for label, value in stats
    )
    st.markdown(
        f'<div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:16px;">'
        f'{badges_html}</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# ── Glass card wrapper ─────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def glass_card_open(extra_style: str = "") -> None:
    """Open a glass-card container div."""
    st.markdown(
        f'<div style="'
        f'background:linear-gradient(145deg,rgba(15,23,42,0.92),rgba(20,10,50,0.88));'
        f'border:1px solid rgba(108,99,255,0.30);border-radius:16px;'
        f'padding:24px 26px;margin-bottom:16px;{extra_style}">',
        unsafe_allow_html=True,
    )


def glass_card_close() -> None:
    """Close a glass-card container div."""
    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ── Chart section header ───────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def chart_section(title: str, subtitle: str = "", accent: str = "#6C63FF") -> None:
    """Render a compact section header directly above a chart."""
    sub_html = (
        f'<span style="font-size:0.74rem;color:#CBD5E1;margin-left:8px;">— {subtitle}</span>'
        if subtitle else ""
    )
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:12px;">'
        f'<span style="display:inline-block;width:8px;height:8px;border-radius:50%;'
        f'background:{accent};box-shadow:0 0 10px {accent}BB;flex-shrink:0;"></span>'
        f'<span style="font-size:1.0rem;font-weight:700;color:#F8FAFC;">{title}</span>'
        f'{sub_html}'
        f'</div>',
        unsafe_allow_html=True,
    )
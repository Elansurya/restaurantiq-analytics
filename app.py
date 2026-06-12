from __future__ import annotations

import sys
import os
import importlib.util
import re
import traceback
from pathlib import Path
from datetime import datetime

# ── 0. Path bootstrap ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
SRC  = ROOT / "src"

for _p in (str(ROOT), str(SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.chdir(str(ROOT))

# ── 1. Streamlit ───────────────────────────────────────────────────────────────
import streamlit as st

st.set_page_config(
    page_title="RestaurantIQ – AI Analytics",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help":     "https://github.com",
        "Report a bug": None,
        "About": (
            "**RestaurantIQ** — AI-Powered Restaurant Intelligence Platform\n\n"
            "Cognifyz Analytics Challenge · Phase 1 + Full ML Suite\n\n"
            "Built with Streamlit · Plotly · Folium · PyDeck · scikit-learn"
        ),
    },
)


# ═══════════════════════════════════════════════════════════════════════════════
# ── SECTION A: CSS ─────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def _load_css(path: Path) -> None:
    """Load style.css if it exists."""
    try:
        if path.exists():
            st.markdown(f"<style>{path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
    except Exception:
        pass


_load_css(ROOT / "assets" / "style.css")

# ── Sidebar redesign CSS ───────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Sidebar shell ───────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: #0D0F1A !important;
        border-right: 1px solid rgba(255,255,255,0.06) !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        padding: 0 !important;
    }
    [data-testid="stSidebarContent"] {
        padding: 0 !important;
        background: transparent !important;
    }

    /* ── Hide default Streamlit button chrome in sidebar ─────────────────── */
    [data-testid="stSidebar"] .stButton > button {
        all: unset !important;
        display: flex !important;
        align-items: center !important;
        gap: 10px !important;
        width: 100% !important;
        padding: 9px 20px 9px 18px !important;
        border-radius: 8px !important;
        cursor: pointer !important;
        font-size: 0.855rem !important;
        font-weight: 500 !important;
        color: #94A3B8 !important;
        background: transparent !important;
        border: none !important;
        transition: background 0.15s ease, color 0.15s ease !important;
        letter-spacing: 0.005em !important;
        line-height: 1.4 !important;
        box-sizing: border-box !important;
        margin: 0 !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(108, 99, 255, 0.10) !important;
        color: #C4BFFF !important;
    }
    [data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background: rgba(108, 99, 255, 0.16) !important;
        color: #E0DDFF !important;
        font-weight: 600 !important;
    }
    [data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
        background: rgba(108, 99, 255, 0.22) !important;
    }
    [data-testid="stSidebar"] .stButton > button p {
        font-size: 0.855rem !important;
        margin: 0 !important;
        line-height: inherit !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }

    /* ── Remove stButton wrapper spacing ─────────────────────────────────── */
    [data-testid="stSidebar"] .stButton {
        margin: 0 !important;
        padding: 0 !important;
    }
    [data-testid="stSidebar"] div[data-testid="stVerticalBlock"] {
        gap: 0 !important;
    }

    /* ── Dividers ─────────────────────────────────────────────────────────── */
    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.07) !important;
        margin: 10px 0 !important;
    }

    /* ── Caption / section labels — hidden ───────────────────────────────── */
    [data-testid="stSidebar"] .stCaption p {
        font-size: 0.62rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.13em !important;
        text-transform: uppercase !important;
        color: #334155 !important;
        padding: 14px 20px 4px !important;
        margin: 0 !important;
    }

    /* ── Metrics in sidebar ───────────────────────────────────────────────── */
    [data-testid="stSidebar"] [data-testid="stMetric"] {
        background: transparent !important;
        padding: 2px 0 !important;
    }
    [data-testid="stSidebar"] [data-testid="stMetricLabel"] p {
        font-size: 0.65rem !important;
        color: #475569 !important;
    }
    [data-testid="stSidebar"] [data-testid="stMetricValue"] {
        font-size: 0.88rem !important;
        color: #94A3B8 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════════
# ── SECTION B: Session state ───────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

_DEFAULTS: dict = {
    "active_page": "Home",
    "data_loaded": False,
    "theme":       "dark",
    "last_visit":  datetime.now().strftime("%H:%M"),
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ═══════════════════════════════════════════════════════════════════════════════
# ── SECTION C: Page registry ───────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

PAGE_REGISTRY: list[dict] = [
    {
        "label":    "Home",
        "icon":     "🏠",
        "stems":    ["home", "1_home"],
        "accent":   "#6C63FF",
        "desc":     "Landing page, KPI overview & platform navigation",
        "category": "Platform",
        "badge":    None,
    },
    {
        "label":    "Data Exploration & Preprocessing",
        "icon":     "📊",
        "stems":    ["eda", "3_eda"],
        "accent":   "#6C63FF",
        "desc":     "Histograms, violin plots, heatmaps, treemaps, statistics",
        "category": "Analytics",
        "badge":    None,
    },
    {
        "label":    "Descriptive Analysis",
        "icon":     "📈",
        "stems":    ["dashboard", "2_dashboard"],
        "accent":   "#00C2A8",
        "desc":     "Executive KPIs, gauges, distributions & service stats",
        "category": "Analytics",
        "badge":    None,
    },
    {
        "label":    "Geospatial Analysis",
        "icon":     "🌍",
        "stems":    ["geospatial", "4_geospatial"],
        "accent":   "#F59E0B",
        "desc":     "Plotly Mapbox · Folium clusters · PyDeck 3D maps",
        "category": "Analytics",
        "badge":    None,
    },
    {
        "label":    "Data Visualization",
        "icon":     "📋",
        "stems":    ["explorer", "5_explorer"],
        "accent":   "#8B5CF6",
        "desc":     "Full dataset browser, column profiler & CSV export",
        "category": "Analytics",
        "badge":    None,
    },
    {
        "label":    "Feature Engineering",
        "icon":     "⚙️",
        "stems":    ["features", "6_features"],
        "accent":   "#06B6D4",
        "desc":     "Feature engineering pipeline & importance analysis",
        "category": "ML Suite",
        "badge":    "ML",
    },
    {
        "label":    "Customer Preference Analysis",
        "icon":     "🍽️",
        "stems":    ["insights", "7_insights"],
        "accent":   "#EC4899",
        "desc":     "Auto-generated business insights & recommendations",
        "category": "ML Suite",
        "badge":    "AI",
    },
    {
        "label":    "Predictive Modeling",
        "icon":     "🤖",
        "stems":    ["ml_pipeline", "8_ml_pipeline"],
        "accent":   "#10B981",
        "desc":     "Model training, cross-validation & performance metrics",
        "category": "ML Suite",
        "badge":    "ML",
    },
    {
        "label":    "Success Score",
        "icon":     "🏆",
        "stems":    ["success_score", "9_success_score"],
        "accent":   "#F59E0B",
        "desc":     "Composite success scoring engine for restaurants",
        "category": "ML Suite",
        "badge":    "AI",
    },
    {
        "label":    "AI Predictor",
        "icon":     "🔮",
        "stems":    ["ai_predictor", "10_ai_predictor"],
        "accent":   "#6C63FF",
        "desc":     "Live rating prediction & what-if scenario explorer",
        "category": "ML Suite",
        "badge":    "LIVE",
    },
]

PAGES_DIR = ROOT / "pages"
PAGES_DIR.mkdir(exist_ok=True)


def _normalize_stem(s: str) -> str:
    s = s.lower()
    s = re.sub(r"^[\d]+[_\-]?", "", s)
    s = re.sub(r"[^\w]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _discover_pages() -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for f in sorted(PAGES_DIR.glob("*.py")):
        if f.name.startswith("_"):
            continue
        key = _normalize_stem(f.stem)
        mapping[key] = f
    return mapping


_DISCOVERED: dict[str, Path] = _discover_pages()


def _resolve_page(entry: dict) -> Path | None:
    for stem in entry["stems"]:
        norm = _normalize_stem(stem)
        if norm in _DISCOVERED:
            return _DISCOVERED[norm]
        for key, path in _DISCOVERED.items():
            if norm in key or key in norm:
                return path
    return None


for _entry in PAGE_REGISTRY:
    _entry["path"] = _resolve_page(_entry)


# ═══════════════════════════════════════════════════════════════════════════════
# ── SECTION D: Data loader ─────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def _safe_load_kpis() -> dict | None:
    try:
        from src.preprocessing import load_and_preprocess, get_summary_kpis
        df   = load_and_preprocess()
        kpis = get_summary_kpis(df)
        return kpis
    except ImportError:
        try:
            from preprocessing import load_and_preprocess, get_summary_kpis  # type: ignore
            df   = load_and_preprocess()
            kpis = get_summary_kpis(df)
            return kpis
        except Exception:
            return None
    except Exception:
        return None


_KPI_DATA: dict | None = _safe_load_kpis()


# ═══════════════════════════════════════════════════════════════════════════════
# ── SECTION E: Page runner ─────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def _run_page_module(path: Path, label: str) -> None:
    module_name = (
        "restaurantiq_page_"
        + re.sub(r"[^\w]", "_", label.lower())
    )
    try:
        sys.modules.pop(module_name, None)
        spec = importlib.util.spec_from_file_location(module_name, str(path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot create module spec for: {path}")
        module = importlib.util.module_from_spec(spec)
        module.__file__  = str(path)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        if hasattr(module, "show") and callable(module.show):
            module.show()
    except SystemExit:
        raise
    except Exception as exc:
        _render_page_error(label, path, exc)


def _render_page_error(label: str, path: Path, exc: Exception) -> None:
    tb = traceback.format_exc()
    st.error(f"**Page \"{label}\" failed to load**\n\n`{path}`\n\n```\n{exc}\n```")
    with st.expander("🔍 Full traceback"):
        st.code(tb, language="python")


# ═══════════════════════════════════════════════════════════════════════════════
# ── SECTION F: Sidebar navigation  (redesigned — no section headers) ──────────
# ═══════════════════════════════════════════════════════════════════════════════

def _render_sidebar() -> str:
    """
    Render a clean, modern SaaS sidebar with NO section header labels.
    All navigation buttons are shown in order with a single separator
    between Platform items and the rest.
    Returns the selected page label.
    """

    with st.sidebar:

        # ── Brand mark ────────────────────────────────────────────────────────
        n_active = sum(1 for e in PAGE_REGISTRY if e["path"] is not None)

        st.markdown(
            f"""
            <div style="padding:24px 20px 16px;">
                <div style="display:flex;align-items:center;gap:11px;margin-bottom:12px;">
                    <div style="
                        width:34px;height:34px;border-radius:9px;flex-shrink:0;
                        display:flex;align-items:center;justify-content:center;
                        background:linear-gradient(135deg,#6C63FF,#8B5CF6);
                        box-shadow:0 0 14px rgba(108,99,255,0.40);
                        font-size:1.05rem;
                    ">🍽️</div>
                    <div>
                        <div style="
                            font-size:1.05rem;font-weight:800;line-height:1.1;
                            background:linear-gradient(90deg,#A78BFA,#67E8F9);
                            -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                            background-clip:text;letter-spacing:-0.02em;
                        ">RestaurantIQ</div>
                        <div style="
                            font-size:0.60rem;color:#475569;
                            text-transform:uppercase;letter-spacing:0.14em;
                            font-weight:600;margin-top:1px;
                        ">AI Analytics Platform</div>
                    </div>
                </div>
                <div style="
                    display:inline-flex;align-items:center;gap:6px;
                    background:rgba(16,185,129,0.08);
                    border:1px solid rgba(16,185,129,0.18);
                    border-radius:999px;padding:3px 10px;
                ">
                    <span style="
                        width:6px;height:6px;border-radius:50%;
                        background:#10B981;display:inline-block;
                        box-shadow:0 0 6px rgba(16,185,129,0.7);
                    "></span>
                    <span style="font-size:0.60rem;color:#6EE7B7;font-weight:600;
                        letter-spacing:0.06em;text-transform:uppercase;">
                        LIVE &nbsp;·&nbsp; {n_active} pages active
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Top separator ──────────────────────────────────────────────────────
        st.markdown(
            '<div style="height:1px;background:rgba(255,255,255,0.06);margin:0 0 6px;"></div>',
            unsafe_allow_html=True,
        )

        # ── Navigation items — ALL categories, NO section labels ──────────────
        current = st.session_state.get("active_page", "Home")
        selected_label = current

        def _badge_pill(badge: str, accent: str) -> str:
            return (
                f'<span style="'
                f'background:{accent}20;border:1px solid {accent}40;'
                f'border-radius:999px;padding:1px 7px;'
                f'font-size:0.55rem;font-weight:700;letter-spacing:0.07em;'
                f'text-transform:uppercase;color:{accent};margin-left:auto;'
                f'flex-shrink:0;">{badge}</span>'
            )

        # Render every page in registry order — no category separators/labels
        for idx, entry in enumerate(PAGE_REGISTRY):
            is_active = (entry["label"] == current)
            exists    = entry["path"] is not None
            accent    = entry["accent"]
            badge     = entry.get("badge") or ""

            safe_key = "nav_" + re.sub(r"[^\w]", "_", entry["label"].lower())

            # Single thin separator after "Home" (Platform → rest)
            if idx == 1:
                st.markdown(
                    '<div style="height:1px;background:rgba(255,255,255,0.05);'
                    'margin:8px 0;"></div>',
                    unsafe_allow_html=True,
                )

            if exists:
                btn_label = f"{entry['icon']}  {entry['label']}"
                if st.button(
                    btn_label,
                    key=safe_key,
                    use_container_width=True,
                    help=entry["desc"],
                    type="primary" if is_active else "secondary",
                ):
                    st.session_state["active_page"] = entry["label"]
                    selected_label = entry["label"]
                    st.rerun()

            else:
                # Disabled / coming-soon item
                st.markdown(
                    f'<div style="'
                    f'display:flex;align-items:center;gap:10px;'
                    f'padding:9px 20px;'
                    f'font-size:0.855rem;color:#334155;'
                    f'cursor:not-allowed;">'
                    f'<span style="opacity:0.4;">{entry["icon"]}</span>'
                    f'<span style="opacity:0.35;">{entry["label"]}</span>'
                    f'<span style="margin-left:auto;font-size:0.58rem;'
                    f'color:#1E293B;text-transform:uppercase;letter-spacing:0.07em;">'
                    f'soon</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # ── Quick stats (when data available) ─────────────────────────────────
        if _KPI_DATA:
            st.markdown(
                '<div style="height:1px;background:rgba(255,255,255,0.05);'
                'margin:12px 0 0;"></div>',
                unsafe_allow_html=True,
            )
            st.caption("Quick Stats")
            qs_col1, qs_col2 = st.columns(2)
            with qs_col1:
                st.metric("🍽️", f"{_KPI_DATA.get('total_restaurants', 0):,}", label_visibility="visible")
                st.metric("⭐", str(_KPI_DATA.get("avg_rating", "—")))
            with qs_col2:
                st.metric("🌍", str(_KPI_DATA.get("countries", 0)))
                st.metric("🚴", f"{_KPI_DATA.get('delivery_pct', 0)}%")

        # ── Footer ────────────────────────────────────────────────────────────
        st.markdown(
            '<div style="height:1px;background:rgba(255,255,255,0.05);margin:14px 0 0;"></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div style="padding:12px 20px 20px;">'
            '<div style="font-size:0.60rem;color:#1E293B;line-height:1.6;">'
            'RestaurantIQ · v1.0<br>'
            'Streamlit · Plotly · scikit-learn'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    return selected_label


# ═══════════════════════════════════════════════════════════════════════════════
# ── SECTION G: Page title bar ──────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def _render_page_title_bar(entry: dict) -> None:
    """Render a slim, single-line title bar above page content."""
    icon       = entry["icon"]
    label      = entry["label"]
    category   = entry["category"]
    accent     = entry["accent"]
    desc       = entry["desc"]
    desc_short = desc[:55] + "…" if len(desc) > 55 else desc
    badge      = entry.get("badge") or ""

    badge_html = (
        f'<span style="'
        f'background:{accent}22;border:1px solid {accent}55;'
        f'border-radius:999px;padding:2px 10px;'
        f'font-size:0.60rem;font-weight:700;letter-spacing:.08em;'
        f'text-transform:uppercase;color:{accent};margin-left:8px;">'
        f'{badge}</span>'
    ) if badge else ""

    st.markdown(
        f'<div style="'
        f'display:flex;align-items:center;gap:12px;'
        f'padding:10px 18px;margin-bottom:8px;'
        f'background:rgba(15,23,42,0.92);'
        f'border:1px solid rgba(255,255,255,0.07);'
        f'border-left:3px solid {accent};border-radius:10px;">'
        f'<div style="'
        f'width:32px;height:32px;border-radius:9px;flex-shrink:0;'
        f'display:flex;align-items:center;justify-content:center;'
        f'background:{accent}22;font-size:1rem;'
        f'border:1px solid {accent}44;">{icon}</div>'
        f'<div style="flex:1;min-width:0;overflow:hidden;">'
        f'<div style="display:flex;align-items:center;gap:4px;flex-wrap:nowrap;">'
        f'<span style="'
        f'font-size:0.97rem;font-weight:700;white-space:nowrap;'
        f'background:linear-gradient(90deg,#A78BFA,#67E8F9);'
        f'-webkit-background-clip:text;-webkit-text-fill-color:transparent;'
        f'background-clip:text;">{label}</span>'
        f'{badge_html}'
        f'</div>'
        f'<div style="font-size:0.66rem;color:#475569;margin-top:1px;'
        f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">'
        f'RestaurantIQ › {category} › {desc_short}'
        f'</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ── SECTION H: Home page (fallback) ───────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def _render_home() -> None:
    """Route to home.py if available, otherwise render an inline dashboard."""
    home_entry = next((e for e in PAGE_REGISTRY if e["label"] == "Home"), None)
    if home_entry and home_entry.get("path"):
        _run_page_module(home_entry["path"], "Home")
        return

    # ── Inline fallback dashboard ──────────────────────────────────────────
    kpis = _KPI_DATA or {}

    st.markdown(
        """
        <div style="
            display:flex;align-items:center;gap:14px;
            padding:14px 20px;margin-bottom:18px;
            background:linear-gradient(135deg,rgba(108,99,255,0.10),rgba(0,194,168,0.06));
            border:1px solid rgba(108,99,255,0.25);border-radius:14px;
        ">
            <div style="
                width:42px;height:42px;border-radius:12px;flex-shrink:0;
                display:flex;align-items:center;justify-content:center;
                background:linear-gradient(135deg,#6C63FF,#8B5CF6);
                box-shadow:0 0 18px rgba(108,99,255,0.50);font-size:1.3rem;
            ">🍽️</div>
            <div>
                <div style="
                    font-size:1.4rem;font-weight:900;line-height:1.2;
                    background:linear-gradient(90deg,#6C63FF,#00C2A8);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                    background-clip:text;letter-spacing:-.02em;
                ">RestaurantIQ</div>
                <div style="font-size:0.68rem;color:#475569;margin-top:2px;
                    text-transform:uppercase;letter-spacing:.14em;font-weight:600;">
                    AI-Powered Restaurant Intelligence Platform
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if kpis:
        st.caption("Live metrics across the full restaurant dataset")
        c1, c2, c3, c4 = st.columns(4, gap="small")
        with c1: st.metric("🍽️ Restaurants", f"{kpis.get('total_restaurants', 0):,}")
        with c2: st.metric("🌍 Countries",   str(kpis.get("countries", 0)))
        with c3: st.metric("🏙️ Cities",      f"{kpis.get('cities', 0):,}")
        with c4: st.metric("⭐ Avg Rating",  str(kpis.get("avg_rating", "—")))

        c5, c6, c7, c8 = st.columns(4, gap="small")
        with c5: st.metric("🗳️ Total Votes", f"{kpis.get('total_votes', 0):,}")
        with c6: st.metric("🚴 Delivery %",  f"{kpis.get('delivery_pct', 0)}%")
        with c7: st.metric("📅 Booking %",   f"{kpis.get('booking_pct', 0)}%")
        with c8: st.metric("🍛 Top Cuisine", str(kpis.get("top_cuisine", "—")))
    else:
        st.info(
            "📂 **Data not loaded yet.**  "
            "Place your CSV file in the `data/` folder and restart the app, "
            "or the app will search for it automatically."
        )

    st.markdown("---")
    st.subheader("Analytics Modules")
    st.caption("Select a module from the sidebar to begin your analysis")

    analytics_pages = [e for e in PAGE_REGISTRY if e["label"] != "Home" and e["path"]]
    cols = st.columns(3, gap="medium")
    for i, entry in enumerate(analytics_pages):
        badge_html = (
            f'<span style="'
            f'background:{entry["accent"]}22;border:1px solid {entry["accent"]}55;'
            f'border-radius:999px;padding:2px 8px;font-size:0.60rem;font-weight:700;'
            f'letter-spacing:.08em;text-transform:uppercase;color:{entry["accent"]};">'
            f'{entry["badge"]}</span>'
            if entry.get("badge") else ""
        )
        with cols[i % 3]:
            st.markdown(
                f'<div style="'
                f'background:rgba(30,41,59,0.6);border:1px solid rgba(255,255,255,0.07);'
                f'border-left:3px solid {entry["accent"]};border-radius:12px;'
                f'padding:14px 16px;margin-bottom:10px;">'
                f'{badge_html}'
                f'<div style="font-size:0.95rem;font-weight:700;margin:6px 0 4px;'
                f'color:#FFFFFF;line-height:1.4;">'
                f'{entry["icon"]} {entry["label"]}</div>'
                f'<div style="font-size:0.72rem;color:#94A3B8;line-height:1.5;">{entry["desc"]}</div>'
                f'<div style="margin-top:8px;font-size:0.65rem;color:#475569;">'
                f'{entry["category"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# ── SECTION I: Footer ──────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def _render_footer() -> None:
    st.divider()
    st.caption(
        "🍽️  **RestaurantIQ** v1.0  ·  Cognifyz Analytics Challenge  ·  "
        "Streamlit · Plotly · scikit-learn"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ── SECTION J: Main ────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    selected_label = _render_sidebar()

    if selected_label and selected_label != st.session_state.get("active_page"):
        st.session_state["active_page"] = selected_label

    active_label = st.session_state.get("active_page", "Home")
    active_entry = next(
        (e for e in PAGE_REGISTRY if e["label"] == active_label),
        PAGE_REGISTRY[0],
    )

    if active_label == "Home":
        _render_home()
    else:
        _render_page_title_bar(active_entry)
        if active_entry.get("path") is not None:
            _run_page_module(active_entry["path"], active_label)
        else:
            st.info(
                f"{active_entry['icon']} **{active_label} — Coming Soon**\n\n"
                f"Add a corresponding `.py` file in `pages/` to activate this module.\n\n"
                f"Category: {active_entry['category']}  |  {active_entry['desc']}"
            )

    _render_footer()


main()
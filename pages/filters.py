from __future__ import annotations
from contextlib import contextmanager
import streamlit as st


# ══════════════════════════════════════════════════════════════════════════════
# ── Design Tokens  ────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

_PRIMARY        = "#7C3AED"
_PRIMARY_LIGHT  = "#A78BFA"
_SECONDARY      = "#06B6D4"
_SECONDARY_LIGHT= "#67E8F9"
_ACCENT         = "#F59E0B"
_SUCCESS        = "#10B981"
_PINK           = "#EC4899"
_DANGER         = "#EF4444"
_TEXT           = "#F8FAFC"
_MUTED          = "#94A3B8"
_MUTED2         = "#64748B"
_CARD           = "rgba(13,20,35,0.85)"
_BORDER         = "rgba(124,58,237,0.22)"
_BORDER2        = "rgba(6,182,212,0.18)"


# ══════════════════════════════════════════════════════════════════════════════
# ── Hero Banner  ──────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def hero_banner(
    title: str,
    subtitle: str = "",
    icon: str = "🍽️",
    accent: str = _PRIMARY,
    accent2: str = _SECONDARY,
    chips: list[str] | None = None,
    stat_left: tuple[str, str] | None = None,
    stat_right: tuple[str, str] | None = None,
) -> None:
    """
    Full-width page hero banner with gradient background, title, subtitle,
    optional chips and two quick-stat pills.
    """
    chips_html = ""
    if chips:
        chips_html = (
            '<div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:14px;">'
            + "".join(
                f'<span class="insight-chip" style="font-size:0.70rem;'
                f'padding:4px 12px;border-color:{accent}44;color:{accent};">'
                f'{c}</span>'
                for c in chips
            )
            + "</div>"
        )

    def _stat_pill(label: str, value: str, color: str) -> str:
        return (
            f'<div style="display:flex;flex-direction:column;align-items:center;'
            f'background:{color}12;border:1px solid {color}30;'
            f'border-radius:12px;padding:10px 20px;min-width:90px;">'
            f'<span style="font-size:1.3rem;font-weight:800;color:{color};line-height:1;">'
            f'{value}</span>'
            f'<span style="font-size:0.60rem;text-transform:uppercase;letter-spacing:.11em;'
            f'color:{_MUTED2};font-weight:700;margin-top:3px;">{label}</span>'
            f'</div>'
        )

    stats_html = ""
    if stat_left or stat_right:
        left_html  = _stat_pill(*stat_left,  accent)  if stat_left  else ""
        right_html = _stat_pill(*stat_right, accent2) if stat_right else ""
        stats_html = (
            f'<div style="display:flex;gap:10px;margin-top:18px;">'
            f'{left_html}{right_html}</div>'
        )

    subtitle_html = (
        f'<p style="margin:8px 0 0;font-size:0.88rem;color:{_MUTED};'
        f'line-height:1.6;max-width:620px;">{subtitle}</p>'
        if subtitle else ""
    )

    st.markdown(
        f"""
        <div style="
            position:relative;overflow:hidden;
            background:linear-gradient(135deg,rgba(10,14,25,0.97) 0%,
                {accent}14 45%,{accent2}0C 100%);
            border:1px solid {accent}28;border-radius:22px;
            padding:36px 40px 30px;margin-bottom:28px;
            box-shadow:0 8px 64px {accent}18,0 2px 16px rgba(0,0,0,0.4);
        ">
            <div style="position:absolute;top:-60px;right:-60px;width:260px;height:260px;
                 border-radius:50%;
                 background:radial-gradient(circle,{accent}1A,transparent 70%);
                 pointer-events:none;"></div>
            <div style="position:absolute;bottom:-40px;left:40%;width:180px;height:180px;
                 border-radius:50%;
                 background:radial-gradient(circle,{accent2}14,transparent 70%);
                 pointer-events:none;"></div>
            <div style="position:absolute;top:0;left:0;right:0;height:2px;
                 background:linear-gradient(90deg,transparent,{accent},{accent2},transparent);
                 border-radius:22px 22px 0 0;"></div>
            <div style="position:relative;display:flex;align-items:flex-start;gap:20px;">
                <div style="flex-shrink:0;width:56px;height:56px;border-radius:16px;
                     display:flex;align-items:center;justify-content:center;font-size:1.7rem;
                     background:linear-gradient(135deg,{accent}22,{accent2}14);
                     border:1px solid {accent}40;box-shadow:0 0 20px {accent}30;">
                    {icon}
                </div>
                <div style="flex:1;min-width:0;">
                    <div style="font-size:0.62rem;font-weight:700;letter-spacing:.18em;
                         text-transform:uppercase;color:{accent};margin-bottom:6px;">
                        RestaurantIQ · Analytics Platform
                    </div>
                    <h1 style="margin:0;padding:0;font-size:2.0rem;font-weight:900;
                         line-height:1.15;letter-spacing:-.025em;
                         background:linear-gradient(135deg,{_TEXT} 30%,{accent} 70%,{accent2});
                         -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                         background-clip:text;">
                        {title}
                    </h1>
                    {subtitle_html}
                    {chips_html}
                    {stats_html}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# ── Executive Page Header  ────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def exec_page_header(
    title: str,
    description: str = "",
    icon: str = "📊",
    accent: str = _PRIMARY,
    badge: str | None = None,
    badge_color: str | None = None,
    meta: list[tuple[str, str]] | None = None,
) -> None:
    """
    Compact executive-style section header with icon, title, description,
    optional badge pill, and small meta key-value tags.
    """
    badge_html = ""
    if badge:
        bc = badge_color or accent
        badge_html = (
            f'<span style="background:{bc}18;color:{bc};border:1px solid {bc}35;'
            f'border-radius:999px;padding:3px 12px;font-size:0.62rem;'
            f'font-weight:800;letter-spacing:.10em;text-transform:uppercase;">'
            f'{badge}</span>'
        )

    meta_html = ""
    if meta:
        meta_html = (
            '<div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:8px;">'
            + "".join(
                f'<span class="stat-badge" style="font-size:0.66rem;">'
                f'<span style="color:{_MUTED2};">{k}</span>&nbsp;·&nbsp;{v}</span>'
                for k, v in meta
            )
            + "</div>"
        )

    desc_html = (
        f'<p style="margin:5px 0 0;font-size:0.80rem;color:{_MUTED};'
        f'line-height:1.55;max-width:680px;">{description}</p>'
        if description else ""
    )

    st.markdown(
        f"""
        <div style="
            display:flex;align-items:flex-start;gap:16px;
            background:linear-gradient(135deg,{accent}08,rgba(6,182,212,0.04));
            border:1px solid {accent}22;border-left:4px solid {accent};
            border-radius:14px;padding:20px 24px;margin-bottom:24px;
            box-shadow:0 2px 20px rgba(0,0,0,0.25);
        ">
            <div style="flex-shrink:0;width:44px;height:44px;border-radius:12px;
                 display:flex;align-items:center;justify-content:center;font-size:1.3rem;
                 background:{accent}18;border:1px solid {accent}30;margin-top:2px;">
                {icon}
            </div>
            <div style="flex:1;min-width:0;">
                <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
                    <span style="font-size:1.15rem;font-weight:800;color:{_TEXT};
                          letter-spacing:-.015em;">{title}</span>
                    {badge_html}
                </div>
                {desc_html}
                {meta_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# ── Insight Card  ─────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def insight_card(
    title: str,
    body: str,
    icon: str = "💡",
    accent: str = _PRIMARY,
    badge: str | None = None,
    chips: list[str] | None = None,
    compact: bool = False,
) -> None:
    """
    Premium insight / callout card.
    """
    pad = "14px 18px" if compact else "20px 24px"

    badge_html = ""
    if badge:
        badge_html = (
            f'<span class="rec-badge" style="background:{accent}14;'
            f'border-color:{accent}30;color:{accent};'
            f'margin-bottom:8px;display:inline-block;">{badge}</span>'
        )

    chips_html = ""
    if chips:
        chips_html = (
            '<div style="display:flex;flex-wrap:wrap;gap:4px;margin-top:10px;">'
            + "".join(
                f'<span class="insight-chip" style="font-size:0.67rem;'
                f'padding:3px 10px;color:{accent};border-color:{accent}30;">'
                f'{c}</span>'
                for c in chips
            )
            + "</div>"
        )

    st.markdown(
        f"""
        <div style="
            position:relative;overflow:hidden;
            background:linear-gradient(135deg,{accent}0D,rgba(6,182,212,0.04));
            border:1px solid {accent}28;border-left:3px solid {accent};
            border-radius:14px;padding:{pad};margin-bottom:14px;
        ">
            <div style="position:absolute;top:-20px;right:-20px;width:80px;height:80px;
                 border-radius:50%;
                 background:radial-gradient(circle,{accent}10,transparent 70%);
                 pointer-events:none;"></div>
            <div style="display:flex;gap:12px;align-items:flex-start;">
                <span style="font-size:{'1.2rem' if compact else '1.5rem'};
                      flex-shrink:0;margin-top:2px;">{icon}</span>
                <div style="flex:1;min-width:0;">
                    {badge_html}
                    <div style="font-size:{'0.82rem' if compact else '0.90rem'};
                         font-weight:700;color:{_TEXT};margin-bottom:5px;">{title}</div>
                    <div style="font-size:{'0.74rem' if compact else '0.78rem'};
                         color:{_MUTED};line-height:1.60;">{body}</div>
                    {chips_html}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# ── Summary Card  ─────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def summary_card(
    rows: list[tuple[str, str]],
    title: str = "",
    accent: str = _SECONDARY,
    n_cols: int = 1,
) -> None:
    """
    Structured summary / stat table card with key-value rows.
    """
    title_html = ""
    if title:
        title_html = (
            f'<div style="font-size:0.68rem;font-weight:700;letter-spacing:.14em;'
            f'text-transform:uppercase;color:{accent};margin-bottom:14px;'
            f'padding-bottom:8px;border-bottom:1px solid {accent}22;">{title}</div>'
        )

    def _row(label: str, value: str) -> str:
        return (
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'padding:6px 0;border-bottom:1px solid rgba(255,255,255,0.04);">'
            f'<span style="font-size:0.74rem;color:{_MUTED2};font-weight:500;">{label}</span>'
            f'<span class="stat-badge" style="background:{accent}12;'
            f'border-color:{accent}28;color:{accent};font-size:0.70rem;">{value}</span>'
            f'</div>'
        )

    if n_cols == 2 and len(rows) > 2:
        mid   = (len(rows) + 1) // 2
        left  = rows[:mid]
        right = rows[mid:]
        left_html  = "".join(_row(l, v) for l, v in left)
        right_html = "".join(_row(l, v) for l, v in right)
        body_html = (
            f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:0 24px;">'
            f'<div>{left_html}</div><div>{right_html}</div></div>'
        )
    else:
        body_html = "".join(_row(l, v) for l, v in rows)

    st.markdown(
        f"""
        <div class="glass-card" style="padding:18px 22px;margin-bottom:16px;
             border-color:{accent}22;box-shadow:0 0 20px {accent}0A;">
            {title_html}
            {body_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# ── Filter Panel  ─────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

@contextmanager
def filter_panel(title: str = "Filter Controls", collapsible: bool = False):
    """
    Context manager. Renders a styled filter panel in the MAIN content area.

    Usage
    -----
        with filter_panel("Filter Data"):
            col1, col2, col3 = st.columns(3)
            with col1:
                country = st.selectbox("Country", options, key="my_country")
    """
    if collapsible:
        with st.expander(f"🔍 {title}", expanded=True):
            yield
    else:
        st.markdown(
            f"""
            <div class="iq-filter-panel">
                <div class="iq-filter-label">🔍 {title}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.container():
            yield
        st.markdown("<div style='margin-bottom:8px;'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ── Page Title  ───────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def page_title(icon: str, title: str, subtitle: str = "") -> None:
    """
    Renders a large gradient page title inside the main content area.
    """
    st.markdown(
        f"""
        <div style="margin-bottom:22px;padding-top:4px;">
            <div style="font-size:1.75rem;font-weight:900;line-height:1.15;
                 letter-spacing:-.02em;
                 background:linear-gradient(90deg,{_PRIMARY_LIGHT},{_SECONDARY_LIGHT});
                 -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                 background-clip:text;">
                {icon}&nbsp;{title}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if subtitle:
        st.caption(subtitle)


# ══════════════════════════════════════════════════════════════════════════════
# ── Section Header  ───────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def section_header(text: str, accent: str = _PRIMARY, subtitle: str = "") -> None:
    """
    Renders a premium sub-section header with accent dot.
    """
    st.markdown(
        f"""
        <div class="iq-section" style="margin-bottom:20px;margin-top:14px;">
            <div style="display:flex;align-items:center;gap:9px;">
                <span style="display:inline-block;width:8px;height:8px;border-radius:50%;
                      background:{accent};box-shadow:0 0 8px {accent}88;flex-shrink:0;"></span>
                <span class="iq-section-header"
                      style="margin:0;padding:0;border-bottom:none;font-size:1.08rem;">
                    {text}
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if subtitle:
        st.caption(subtitle)


# ══════════════════════════════════════════════════════════════════════════════
# ── Chart Container  ──────────────────────────────────────────────════════════
# ══════════════════════════════════════════════════════════════════════════════

@contextmanager
def chart_container(title: str = "", accent: str = _PRIMARY, subtitle: str = ""):
    """
    Wraps a Plotly chart in a titled glass-card container.

    Usage
    -----
        with chart_container("Rating Distribution", accent="#06B6D4"):
            st.plotly_chart(fig, use_container_width=True)
    """
    if title:
        sub_span = (
            f'<span style="font-size:0.70rem;color:{_MUTED2};margin-left:6px;">'
            f'— {subtitle}</span>'
            if subtitle else ""
        )
        st.markdown(
            f"""
            <div style="display:flex;align-items:center;gap:8px;
                 margin-bottom:10px;padding-left:4px;">
                <span style="display:inline-block;width:7px;height:7px;border-radius:50%;
                      background:{accent};box-shadow:0 0 7px {accent}88;flex-shrink:0;"></span>
                <span style="font-size:0.88rem;font-weight:700;color:{_TEXT};">{title}</span>
                {sub_span}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div class="glass-card" style="
            padding:18px 20px 8px;margin-bottom:16px;
            border-color:{accent}20;box-shadow:0 2px 24px rgba(0,0,0,0.28);">
        """,
        unsafe_allow_html=True,
    )

    with st.container():
        yield

    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ── KPI Row  ──────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def kpi_row(items: list[tuple[str, str, str | None]]) -> None:
    """
    Renders a row of native Streamlit metric cards.

    items: list of (label, value, delta_or_None)
    """
    cols = st.columns(len(items), gap="small")
    for col, (label, value, delta) in zip(cols, items):
        with col:
            if delta:
                st.metric(label, value, delta)
            else:
                st.metric(label, value)


# ══════════════════════════════════════════════════════════════════════════════
# ── Status Badge  ─────────────────────────────────════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════

def status_badge(text: str, color: str = _SUCCESS) -> None:
    """
    Renders an animated status pill in the main content area.
    """
    st.markdown(
        f"""
        <style>
        @keyframes iq-pulse{{
            0%,100%{{opacity:1;transform:scale(1);}}
            50%{{opacity:.55;transform:scale(1.25);}}
        }}
        </style>
        <div style="display:inline-flex;align-items:center;gap:8px;
             background:{color}12;border:1px solid {color}35;
             border-radius:999px;padding:5px 16px;
             font-size:0.76rem;font-weight:700;color:{color};
             margin-bottom:14px;">
            <span style="width:7px;height:7px;border-radius:50%;
                  background:{color};display:inline-block;
                  animation:iq-pulse 1.4s infinite;
                  box-shadow:0 0 6px {color}88;"></span>
            {text}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# ── Divider  ──────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def premium_divider(accent: str = _PRIMARY, accent2: str = _SECONDARY) -> None:
    """
    Renders a slim gradient divider line.
    """
    st.markdown(
        f"""
        <div style="height:1px;margin:20px 0;
             background:linear-gradient(90deg,
                 transparent,{accent}60,{accent2}40,transparent);
             border-radius:1px;">
        </div>
        """,
        unsafe_allow_html=True,
    )
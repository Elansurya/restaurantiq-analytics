import re
from pathlib import Path


# ── Pattern 1: Remove sidebar filter blocks ──────────────────────────────────
#
# FIND any block like:
#     with st.sidebar:
#         ... (filter widgets)
#
# REPLACE with a filter_panel() block in main area.
#
# The script below is conservative: it only removes `with st.sidebar:` blocks
# that contain ONLY filter widgets (selectbox, multiselect, slider, radio).
# Manual review is needed if sidebar blocks also contain navigation logic.

SIDEBAR_FILTER_PATTERN = re.compile(
    r"with\s+st\.sidebar\s*:\s*\n((?:\s+(?:st\.selectbox|st\.multiselect|st\.slider"
    r"|st\.radio|st\.checkbox|st\.markdown|st\.write|st\.header|st\.subheader"
    r"|#)[^\n]*\n)*)",
    re.MULTILINE,
)


def _indent_block(code: str, spaces: int = 4) -> str:
    return "\n".join(" " * spaces + line for line in code.splitlines())


def patch_page(path: Path) -> bool:
    """
    Patches a single page file.
    Returns True if the file was modified.
    """
    src = path.read_text(encoding="utf-8")
    original = src

    # 1. Add iq_filters import if not present
    if "from iq_filters import" not in src and "import iq_filters" not in src:
        # Insert after the last sys.path.insert line or at top of imports
        insert_after = "import streamlit as st\n"
        if insert_after in src:
            src = src.replace(
                insert_after,
                insert_after + "from iq_filters import filter_panel, section_header, chart_container\n",
                1,
            )

    # 2. Remove `with st.sidebar:` filter-only blocks
    # (keeps the variable assignments, moves them to top of show())
    def _extract_sidebar_vars(block: str) -> str:
        """Keep only the assignment lines, strip sidebar-specific markup."""
        lines = []
        for line in block.splitlines():
            stripped = line.strip()
            if stripped.startswith("#") or not stripped:
                continue
            if any(stripped.startswith(f"st.{w}") for w in
                   ("markdown", "write", "header", "subheader", "divider")):
                continue
            lines.append(stripped)
        return "\n".join(lines)

    def _replace_sidebar_block(m: re.Match) -> str:
        inner = _extract_sidebar_vars(m.group(1))
        if not inner:
            return ""
        return (
            "    with filter_panel('Filters'):\n"
            "        col1, col2, col3 = st.columns(3)\n"
            "        with col1:\n"
            + _indent_block(inner, 12)
            + "\n"
        )

    src = SIDEBAR_FILTER_PATTERN.sub(_replace_sidebar_block, src)

    if src != original:
        path.write_text(src, encoding="utf-8")
        return True
    return False


# ── Pattern 2: Hide st.spinner / st.progress from sidebar ────────────────────
#
# Pages that call `with st.sidebar: st.spinner(...)` or
# `st.sidebar.progress(...)` should move those calls to the main area.
#
# FIND:
#     st.sidebar.progress(...)
#     with st.sidebar: st.spinner(...)
#
# REPLACE with:
#     st.progress(...)              # in main area
#     with st.spinner(...):         # in main area

SIDEBAR_PROGRESS_PATTERNS = [
    # st.sidebar.progress(n) → st.progress(n)
    (re.compile(r"\bst\.sidebar\.progress\("), "st.progress("),
    # st.sidebar.spinner(msg) → st.spinner(msg)
    (re.compile(r"\bst\.sidebar\.spinner\("), "st.spinner("),
    # st.sidebar.info / warning / success / error → st.*
    (re.compile(r"\bst\.sidebar\.(info|warning|success|error)\("), r"st.\1("),
]


def patch_progress(path: Path) -> bool:
    src = path.read_text(encoding="utf-8")
    original = src
    for pattern, replacement in SIDEBAR_PROGRESS_PATTERNS:
        src = pattern.sub(replacement, src)
    if src != original:
        path.write_text(src, encoding="utf-8")
        return True
    return False


# ── Run ───────────────────────────────────────────────────────────────────────

def main():
    pages_dir = Path(__file__).parent / "pages"
    if not pages_dir.exists():
        print(f"[ERROR] pages/ directory not found at {pages_dir}")
        return

    page_files = [f for f in pages_dir.glob("*.py") if not f.name.startswith("_")]
    print(f"Found {len(page_files)} page files to patch.\n")

    for f in sorted(page_files):
        changed_filters  = patch_page(f)
        changed_progress = patch_progress(f)
        status = []
        if changed_filters:  status.append("sidebar filters moved")
        if changed_progress: status.append("sidebar progress moved")
        if status:
            print(f"  ✓ {f.name}: {', '.join(status)}")
        else:
            print(f"  — {f.name}: no changes needed")

    print("\nDone. Review each modified file before committing.")
    print("Manual review needed for complex sidebar blocks.")


if __name__ == "__main__":
    main()


# ══════════════════════════════════════════════════════════════════════════════
# MANUAL PATTERNS — apply these by hand to each page if auto-patch misses them
# ══════════════════════════════════════════════════════════════════════════════

MANUAL_PATTERNS = """
=== ISSUE 1: Page titles hidden by banner ===

ROOT CAUSE: block-container had padding-top that shifted content under
the Streamlit header bar. Fixed in app.py CSS:
    .block-container { padding-top: 0 !important; }

The _render_page_header() in app.py now renders as the FIRST block
of main content with margin-top:16px — it's a normal div, not a
floating overlay. No additional changes needed in page files.

=== ISSUE 2: Sidebar showing rerun/spinner during ML inference ===

In any page that runs model prediction (ai_predictor, ml_pipeline,
success_score), replace:

    BEFORE:
        with st.sidebar:
            with st.spinner("Running model..."):
                result = model.predict(X)
        st.sidebar.success("Done!")

    AFTER:
        # In main content area
        with st.spinner("Running model..."):
            result = model.predict(X)
        st.success("Prediction complete!")

=== ISSUE 3: Chart titles not visible ===

When creating plotly figures, always set:

    fig.update_layout(
        title=dict(
            text="Your Chart Title",
            font=dict(size=16, color="#F8FAFC", family="Inter"),
            x=0.01,
            xanchor="left",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(22,32,50,0.4)",
        font=dict(color="#94A3B8", family="Inter"),
        margin=dict(t=52, b=40, l=12, r=12),
    )

=== ISSUE 4: Filter controls in sidebar ===

Every page that has `with st.sidebar:` containing selectbox/multiselect:

    BEFORE:
        with st.sidebar:
            st.markdown("### Filters")
            country = st.selectbox("Country", countries, key="g_country")
            cuisine = st.multiselect("Cuisine", cuisines, key="g_cuisine")

    AFTER (at top of show()):
        from iq_filters import filter_panel
        with filter_panel("Filter Data"):
            col1, col2 = st.columns(2)
            with col1:
                country = st.selectbox("Country", countries, key="g_country")
            with col2:
                cuisine = st.multiselect("Cuisine", cuisines, key="g_cuisine")

=== Page-specific notes ===

pages/6_features.py   (Feature Engineering)
  - Move all st.sidebar.selectbox calls to filter_panel()
  - Model execution spinner stays in main area

pages/8_ml_pipeline.py (Predictive Modeling)
  - Move training progress bars to main area
  - Use st.progress() not st.sidebar.progress()
  - Use status_badge() from iq_filters for training status

pages/10_ai_predictor.py (AI Predictor)
  - Prediction status must be in main area, not sidebar
  - Use: with st.spinner("Generating prediction..."): ...
  - Show result with st.success() in main area

pages/9_success_score.py (Success Score)
  - Move score computation spinner to main area
  - Filter panel for score parameters goes at top of show()
"""

if __name__ == "__main__":
    pass  # MANUAL_PATTERNS is documentation only
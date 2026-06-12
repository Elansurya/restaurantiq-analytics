
import streamlit as st


def page_section_header(text: str, accent: str = "#6C63FF") -> None:
    """Visible section sub-header with accent bar."""
    st.markdown(f"""
    <div style="
        font-size:1.05rem;font-weight:700;color:{accent};
        border-left:3px solid {accent};padding-left:10px;
        margin:22px 0 10px;
    ">{text}</div>
    """, unsafe_allow_html=True)


def status_in_main(text: str, color: str = "#10B981") -> None:
    """
    Renders model execution / prediction status INSIDE the main content area.
    Call this instead of putting st.spinner or st.progress in the sidebar.
    """
    st.markdown(f"""
    <div style="display:inline-flex;align-items:center;gap:8px;
         background:{color}15;border:1px solid {color}40;
         border-radius:999px;padding:6px 16px;
         font-size:0.8rem;font-weight:600;color:{color};margin:10px 0 14px;">
      <span style="width:8px;height:8px;border-radius:50%;background:{color};
            display:inline-block;"></span>
      {text}
    </div>
    """, unsafe_allow_html=True)
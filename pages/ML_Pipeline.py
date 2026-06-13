import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.components import inject_global_css, section_header
from src.preprocessing import load_and_preprocess
from src.train_model import (
    LEADERBOARD_PATH,
    BEST_MODEL_PATH,
    load_artefacts,
    train_all_models,
)

inject_global_css()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE HERO BANNER
# FIX: assigned to variable first — prevents Streamlit from mis-parsing the
#      multiline triple-quoted string when the page module is imported/rerun.
# ─────────────────────────────────────────────────────────────────────────────
_HERO_HTML = """
<div style="background:linear-gradient(135deg,rgba(16,185,129,0.14) 0%,rgba(108,99,255,0.12) 55%,rgba(0,194,168,0.10) 100%);border:1px solid rgba(16,185,129,0.25);border-radius:20px;padding:40px 44px 32px;margin-bottom:28px;position:relative;overflow:hidden;">
  <div style="position:absolute;top:-50px;right:-30px;width:280px;height:280px;border-radius:50%;background:radial-gradient(circle,rgba(16,185,129,0.18),transparent 70%);pointer-events:none;"></div>
  <div style="position:absolute;bottom:-60px;left:40px;width:200px;height:200px;border-radius:50%;background:radial-gradient(circle,rgba(108,99,255,0.14),transparent 70%);pointer-events:none;"></div>
  <div style="position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,rgba(16,185,129,0.7) 40%,rgba(108,99,255,0.6) 70%,transparent);"></div>
  <div style="display:flex;align-items:flex-start;gap:22px;position:relative;">
    <div style="width:64px;height:64px;border-radius:18px;flex-shrink:0;display:flex;align-items:center;justify-content:center;background:linear-gradient(135deg,rgba(16,185,129,0.25),rgba(108,99,255,0.20));border:1px solid rgba(16,185,129,0.35);box-shadow:0 0 24px rgba(16,185,129,0.20);font-size:2rem;line-height:1;">&#x1F916;</div>
    <div style="flex:1;">
      <div style="font-size:0.68rem;font-weight:700;letter-spacing:0.20em;text-transform:uppercase;color:#10B981;margin-bottom:10px;display:flex;align-items:center;gap:8px;">
        <span style="display:inline-block;width:20px;height:1px;background:#10B981;opacity:0.6;"></span>
        ML Suite &middot; Predictive Modeling
      </div>
      <div style="font-size:2.6rem;font-weight:900;line-height:1.10;letter-spacing:-0.03em;color:#F8FAFC;margin-bottom:6px;text-shadow:0 2px 20px rgba(16,185,129,0.25);">
        ML Pipeline &amp;
        <span style="background:linear-gradient(90deg,#6EE7B7 0%,#5EEAD4 50%,#A5B4FC 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">Predictive Modeling</span>
      </div>
      <div style="color:#94A3B8;font-size:0.95rem;line-height:1.70;max-width:680px;margin-bottom:20px;">
        End-to-end machine learning pipeline for restaurant rating prediction. Train five regression models,
        evaluate them head-to-head on RMSE, MAE, R&sup2;, and 5-fold cross-validated R&sup2;, then dive deep into
        feature importance and residual diagnostics to understand what truly drives customer satisfaction scores.
      </div>
      <div style="display:flex;gap:8px;flex-wrap:wrap;">
        <span style="background:rgba(16,185,129,0.12);border:1px solid rgba(16,185,129,0.32);color:#6EE7B7;border-radius:999px;padding:5px 14px;font-size:0.75rem;font-weight:600;">5 Regression Models</span>
        <span style="background:rgba(108,99,255,0.12);border:1px solid rgba(108,99,255,0.3);color:#A5B4FC;border-radius:999px;padding:5px 14px;font-size:0.75rem;font-weight:600;">RMSE &middot; MAE &middot; R&sup2;</span>
        <span style="background:rgba(0,194,168,0.10);border:1px solid rgba(0,194,168,0.28);color:#5EEAD4;border-radius:999px;padding:5px 14px;font-size:0.75rem;font-weight:600;">5-Fold Cross Validation</span>
        <span style="background:rgba(245,158,11,0.10);border:1px solid rgba(245,158,11,0.28);color:#FCD34D;border-radius:999px;padding:5px 14px;font-size:0.75rem;font-weight:600;">Feature Importance</span>
        <span style="background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.25);color:#FCA5A5;border-radius:999px;padding:5px 14px;font-size:0.75rem;font-weight:600;">Residual Analysis</span>
      </div>
    </div>
  </div>
</div>
"""
st.markdown(_HERO_HTML, unsafe_allow_html=True)

df = load_and_preprocess()

# ═══════════════════════════════════════════════════════════════════════════════
# Model Controls
# ═══════════════════════════════════════════════════════════════════════════════
_CONTROLS_HEADER_HTML = """
<div style="display:flex;align-items:center;gap:14px;margin:8px 0 16px;">
  <div style="width:4px;height:36px;border-radius:2px;background:linear-gradient(180deg,#10B981,#6C63FF);flex-shrink:0;"></div>
  <div>
    <div style="font-size:1.55rem;font-weight:800;color:#F8FAFC;letter-spacing:-0.02em;line-height:1.2;">&#x1F39B;&#xFE0F; Model Controls</div>
    <div style="font-size:0.80rem;color:#64748B;margin-top:3px;">Trigger the full training pipeline or load results from a previous run</div>
  </div>
</div>
"""
st.markdown(_CONTROLS_HEADER_HTML, unsafe_allow_html=True)

# Training info card
_TRAINING_INFO_HTML = """
<div style="background:rgba(16,185,129,0.07);border:1px solid rgba(16,185,129,0.20);border-radius:12px;padding:14px 20px;margin-bottom:18px;display:grid;grid-template-columns:repeat(3,1fr);gap:16px;">
  <div style="text-align:center;">
    <div style="font-size:1.4rem;font-weight:800;color:#6EE7B7;">5</div>
    <div style="font-size:0.72rem;color:#64748B;text-transform:uppercase;letter-spacing:.07em;">Models Trained</div>
  </div>
  <div style="text-align:center;border-left:1px solid rgba(255,255,255,0.06);border-right:1px solid rgba(255,255,255,0.06);">
    <div style="font-size:1.4rem;font-weight:800;color:#A5B4FC;">5-Fold</div>
    <div style="font-size:0.72rem;color:#64748B;text-transform:uppercase;letter-spacing:.07em;">Cross Validation</div>
  </div>
  <div style="text-align:center;">
    <div style="font-size:1.4rem;font-weight:800;color:#FCD34D;">~60s</div>
    <div style="font-size:0.72rem;color:#64748B;text-transform:uppercase;letter-spacing:.07em;">First-Run Time</div>
  </div>
</div>
"""
st.markdown(_TRAINING_INFO_HTML, unsafe_allow_html=True)

with st.container():
    col_btn, col_status = st.columns([2, 3])
    with col_btn:
        if st.button(
            "🚀 Train All Models",
            type="primary",
            help="Trains 5 models, evaluates with RMSE/MAE/R²/CV",
            key="ml_train_button",
        ):
            st.session_state["_ml_run_training"] = True

    with col_status:
        if BEST_MODEL_PATH.exists():
            _status_html = """
<div style="background:rgba(16,185,129,0.10);border:1px solid rgba(16,185,129,0.28);border-radius:10px;padding:10px 16px;color:#6EE7B7;font-size:0.85rem;font-weight:600;">
  &#x2705; Saved model artefacts found &mdash; results loaded from disk
</div>
"""
        else:
            _status_html = """
<div style="background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.25);border-radius:10px;padding:10px 16px;color:#FCD34D;font-size:0.85rem;font-weight:600;">
  &#x26A0;&#xFE0F; No saved model found &mdash; click Train All Models to begin
</div>
"""
        st.markdown(_status_html, unsafe_allow_html=True)

st.markdown("<div style='margin-bottom:8px;'></div>", unsafe_allow_html=True)
st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# Training trigger
# ═══════════════════════════════════════════════════════════════════════════════
run_training = st.session_state.pop("_ml_run_training", False)

if run_training:
    with st.spinner("Training models — this may take a minute …"):
        progress_bar = st.progress(0, text="Initialising training pipeline …")

        try:
            with st.status("Running training pipeline …", expanded=True) as status:
                st.write("🔄 Loading and preparing features …")
                progress_bar.progress(10, text="Preparing features …")

                st.write("🏋️ Training all models …")
                progress_bar.progress(30, text="Training models …")

                results = train_all_models(df)

                progress_bar.progress(80, text="Evaluating models …")
                st.write("📊 Evaluating performance metrics …")

                progress_bar.progress(95, text="Saving artefacts …")
                st.write("💾 Saving best model artefacts …")

                progress_bar.progress(100, text="Done!")
                status.update(
                    label=f"✅ Training complete! Best model: **{results['best_name']}**",
                    state="complete",
                    expanded=False,
                )

            st.session_state["ml_results"] = results
            st.success(
                f"✅ Training complete! Best model: **{results['best_name']}**",
                icon="🏆",
            )

        except Exception as exc:
            progress_bar.empty()
            st.error(f"Training failed: {exc}")
            st.stop()

results = st.session_state.get("ml_results", None)

# ── Load leaderboard from disk if not in session ───────────────────────────────
if results is None and LEADERBOARD_PATH.exists():
    lb_disk = pd.read_csv(LEADERBOARD_PATH, index_col=0)

    _disk_notice_html = """
<div style="background:rgba(108,99,255,0.08);border:1px solid rgba(108,99,255,0.22);border-left:4px solid #6C63FF;border-radius:12px;padding:12px 18px;color:#CBD5E1;font-size:0.87rem;margin-bottom:16px;">
  &#x1F4C1; Showing results from the last saved training run. Click <strong>Train All Models</strong> to retrain.
</div>
"""
    st.markdown(_disk_notice_html, unsafe_allow_html=True)

    _disk_lb_header_html = """
<div style="display:flex;align-items:center;gap:14px;margin:24px 0 8px;">
  <div style="width:4px;height:32px;border-radius:2px;background:linear-gradient(180deg,#F59E0B,#6C63FF);flex-shrink:0;"></div>
  <div style="font-size:1.45rem;font-weight:800;color:#F8FAFC;letter-spacing:-0.02em;">&#x1F3C6; Model Leaderboard (saved)</div>
</div>
"""
    st.markdown(_disk_lb_header_html, unsafe_allow_html=True)
    st.dataframe(lb_disk, use_container_width=True)

elif results is None:
    _empty_state_html = """
<div style="background:rgba(30,41,59,0.6);border:1px dashed rgba(108,99,255,0.3);border-radius:16px;padding:48px 40px;text-align:center;margin:24px 0;">
  <div style="font-size:3rem;margin-bottom:12px;">&#x1F680;</div>
  <div style="font-size:1.1rem;font-weight:700;color:#F8FAFC;margin-bottom:8px;">Ready to Train</div>
  <div style="color:#64748B;font-size:0.88rem;max-width:400px;margin:0 auto;">
    Click <strong style="color:#A5B4FC;">Train All Models</strong> above to kick off the pipeline.
    Five models will be trained, cross-validated, and ranked automatically.
  </div>
</div>
"""
    st.markdown(_empty_state_html, unsafe_allow_html=True)

else:
    leaderboard = results["leaderboard"]
    best_name   = results["best_name"]
    feat_imp    = results["feature_importance"]
    residuals   = results["residuals"]

    # ── Executive KPI strip ────────────────────────────────────────────────────
    best_row = leaderboard.iloc[0]

    _kpi_strip_html = f"""
<div style="background:linear-gradient(135deg,rgba(16,185,129,0.12),rgba(108,99,255,0.10));border:1px solid rgba(16,185,129,0.22);border-radius:16px;padding:20px 24px;margin:20px 0 24px;">
  <div style="font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:#10B981;margin-bottom:14px;">
    &#x1F3C6; Best Model Performance &mdash; {best_name}
  </div>
  <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:12px;">
    <div style="text-align:center;">
      <div style="font-size:1.2rem;font-weight:900;color:#6EE7B7;">{best_name}</div>
      <div style="font-size:0.7rem;color:#64748B;text-transform:uppercase;margin-top:3px;">Best Model</div>
    </div>
    <div style="text-align:center;border-left:1px solid rgba(255,255,255,0.06);">
      <div style="font-size:1.2rem;font-weight:900;color:#A5B4FC;">{best_row['R\u00b2']:.4f}</div>
      <div style="font-size:0.7rem;color:#64748B;text-transform:uppercase;margin-top:3px;">R&sup2;</div>
    </div>
    <div style="text-align:center;border-left:1px solid rgba(255,255,255,0.06);">
      <div style="font-size:1.2rem;font-weight:900;color:#FCA5A5;">{best_row['RMSE']:.4f}</div>
      <div style="font-size:0.7rem;color:#64748B;text-transform:uppercase;margin-top:3px;">RMSE</div>
    </div>
    <div style="text-align:center;border-left:1px solid rgba(255,255,255,0.06);">
      <div style="font-size:1.2rem;font-weight:900;color:#FCD34D;">{best_row['MAE']:.4f}</div>
      <div style="font-size:0.7rem;color:#64748B;text-transform:uppercase;margin-top:3px;">MAE</div>
    </div>
    <div style="text-align:center;border-left:1px solid rgba(255,255,255,0.06);">
      <div style="font-size:1.2rem;font-weight:900;color:#5EEAD4;">{best_row['CV R\u00b2 Mean']:.4f} &plusmn; {best_row['CV R\u00b2 Std']:.4f}</div>
      <div style="font-size:0.7rem;color:#64748B;text-transform:uppercase;margin-top:3px;">CV R&sup2;</div>
    </div>
  </div>
</div>
"""
    st.markdown(_kpi_strip_html, unsafe_allow_html=True)

    # ── Leaderboard ────────────────────────────────────────────────────────────
    _lb_header_html = """
<div style="display:flex;align-items:center;gap:14px;margin:28px 0 8px;">
  <div style="width:4px;height:36px;border-radius:2px;background:linear-gradient(180deg,#F59E0B,#EC4899);flex-shrink:0;"></div>
  <div>
    <div style="font-size:1.45rem;font-weight:800;color:#F8FAFC;letter-spacing:-0.02em;">&#x1F3C6; Model Leaderboard</div>
    <div style="font-size:0.80rem;color:#64748B;margin-top:3px;">All five models ranked by R&sup2; &mdash; higher is better</div>
  </div>
</div>
"""
    st.markdown(_lb_header_html, unsafe_allow_html=True)

    def colour_r2(val):
        if val >= 0.8:
            return "color: #00C2A8; font-weight: bold"
        if val >= 0.6:
            return "color: #6C63FF"
        return "color: #F59E0B"

    st.dataframe(
        leaderboard.style.applymap(colour_r2, subset=["R²", "CV R² Mean"]),
        use_container_width=True,
    )

    # ── Chart comparison ───────────────────────────────────────────────────────
    _chart_header_html = """
<div style="display:flex;align-items:center;gap:14px;margin:32px 0 8px;">
  <div style="width:4px;height:36px;border-radius:2px;background:linear-gradient(180deg,#6C63FF,#00C2A8);flex-shrink:0;"></div>
  <div>
    <div style="font-size:1.45rem;font-weight:800;color:#F8FAFC;letter-spacing:-0.02em;">&#x1F4CA; Model Comparison Charts</div>
    <div style="font-size:0.80rem;color:#64748B;margin-top:3px;">Visual head-to-head on the three core evaluation metrics</div>
  </div>
</div>
"""
    st.markdown(_chart_header_html, unsafe_allow_html=True)

    _metrics_insight_html = """
<div style="background:rgba(108,99,255,0.08);border:1px solid rgba(108,99,255,0.20);border-left:4px solid #6C63FF;border-radius:12px;padding:12px 18px;margin-bottom:16px;display:flex;align-items:center;gap:12px;">
  <span style="font-size:1.3rem;flex-shrink:0;">&#x1F4D0;</span>
  <div style="color:#CBD5E1;font-size:0.85rem;line-height:1.6;">
    <strong style="color:#A5B4FC;">Interpreting the metrics:</strong>
    R&sup2; closer to 1.0 means better variance explanation.
    Lower RMSE and MAE mean tighter predictions.
    A small CV R&sup2; std deviation signals stable generalisation across folds.
  </div>
</div>
"""
    st.markdown(_metrics_insight_html, unsafe_allow_html=True)

    col_l, col_r = st.columns(2)

    with col_l:
        fig_r2 = px.bar(
            leaderboard,
            x="Model",
            y="R²",
            color="R²",
            color_continuous_scale=["#EF4444", "#F59E0B", "#00C2A8"],
            title="R² Score by Model",
            text="R²",
        )
        fig_r2.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig_r2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#F8FAFC",
            coloraxis_showscale=False,
            xaxis=dict(tickangle=-25),
        )
        st.plotly_chart(fig_r2, use_container_width=True)

    with col_r:
        fig_err = go.Figure()
        fig_err.add_trace(go.Bar(
            name="RMSE",
            x=leaderboard["Model"],
            y=leaderboard["RMSE"],
            marker_color="#EF4444",
        ))
        fig_err.add_trace(go.Bar(
            name="MAE",
            x=leaderboard["Model"],
            y=leaderboard["MAE"],
            marker_color="#F59E0B",
        ))
        fig_err.update_layout(
            barmode="group",
            title="RMSE & MAE by Model",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#F8FAFC",
            xaxis=dict(tickangle=-25),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_err, use_container_width=True)

    _cv_insight_html = """
<div style="background:rgba(0,194,168,0.07);border:1px solid rgba(0,194,168,0.20);border-left:4px solid #00C2A8;border-radius:12px;padding:12px 18px;margin-bottom:12px;display:flex;align-items:center;gap:12px;">
  <span style="font-size:1.3rem;flex-shrink:0;">&#x1F504;</span>
  <div style="color:#CBD5E1;font-size:0.85rem;line-height:1.6;">
    <strong style="color:#5EEAD4;">Cross-validation insight:</strong>
    Error bars represent one standard deviation across 5 folds.
    A model with high mean R&sup2; but large std dev may be overfitting certain data splits.
  </div>
</div>
"""
    st.markdown(_cv_insight_html, unsafe_allow_html=True)

    fig_cv = px.bar(
        leaderboard,
        x="Model",
        y="CV R² Mean",
        error_y="CV R² Std",
        color="CV R² Mean",
        color_continuous_scale=["#F59E0B", "#6C63FF", "#00C2A8"],
        title="5-Fold Cross-Validated R² (Mean ± Std)",
        text="CV R² Mean",
    )
    fig_cv.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig_cv.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#F8FAFC",
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig_cv, use_container_width=True)

    # ── Feature importance ─────────────────────────────────────────────────────
    if feat_imp is not None and len(feat_imp) > 0:
        _fi_header_html = f"""
<div style="display:flex;align-items:center;gap:14px;margin:32px 0 8px;">
  <div style="width:4px;height:36px;border-radius:2px;background:linear-gradient(180deg,#EC4899,#F59E0B);flex-shrink:0;"></div>
  <div>
    <div style="font-size:1.45rem;font-weight:800;color:#F8FAFC;letter-spacing:-0.02em;">&#x1F511; Feature Importance &mdash; {best_name}</div>
    <div style="font-size:0.80rem;color:#64748B;margin-top:3px;">Top 15 predictors ranked by their contribution to model accuracy</div>
  </div>
</div>
"""
        st.markdown(_fi_header_html, unsafe_allow_html=True)

        if not feat_imp.empty:
            top_feat = feat_imp.iloc[0]["Feature"] if "Feature" in feat_imp.columns else "—"
            top_pct  = feat_imp.iloc[0]["Importance %"] if "Importance %" in feat_imp.columns else 0
            _fi_insight_html = f"""
<div style="background:rgba(236,72,153,0.08);border:1px solid rgba(236,72,153,0.22);border-left:4px solid #EC4899;border-radius:12px;padding:12px 18px;margin-bottom:14px;display:flex;align-items:center;gap:12px;">
  <span style="font-size:1.3rem;flex-shrink:0;">&#x1F511;</span>
  <div style="color:#CBD5E1;font-size:0.85rem;line-height:1.6;">
    <strong style="color:#F9A8D4;">Top predictor:</strong>
    <strong style="color:#F8FAFC;">{top_feat}</strong> accounts for
    <strong style="color:#EC4899;">{top_pct:.1f}%</strong> of the model's predictive power.
    Features near the top are the primary levers for improving rating predictions.
  </div>
</div>
"""
            st.markdown(_fi_insight_html, unsafe_allow_html=True)

        top_fi = feat_imp.head(15)
        fig_fi = px.bar(
            top_fi[::-1],
            x="Importance %",
            y="Feature",
            orientation="h",
            color="Importance %",
            color_continuous_scale=["#1E293B", "#6C63FF", "#00C2A8"],
            title=f"Top 15 Features – {best_name}",
            text="Importance %",
        )
        fig_fi.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        fig_fi.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#F8FAFC",
            coloraxis_showscale=False,
            height=520,
        )
        st.plotly_chart(fig_fi, use_container_width=True)
        st.dataframe(feat_imp, use_container_width=True)

    # ── Residual analysis ──────────────────────────────────────────────────────
    _resid_header_html = """
<div style="display:flex;align-items:center;gap:14px;margin:32px 0 8px;">
  <div style="width:4px;height:36px;border-radius:2px;background:linear-gradient(180deg,#EF4444,#F59E0B);flex-shrink:0;"></div>
  <div>
    <div style="font-size:1.45rem;font-weight:800;color:#F8FAFC;letter-spacing:-0.02em;">&#x1F4C9; Residual Analysis</div>
    <div style="font-size:0.80rem;color:#64748B;margin-top:3px;">Diagnose model bias, heteroscedasticity, and prediction accuracy patterns</div>
  </div>
</div>
"""
    st.markdown(_resid_header_html, unsafe_allow_html=True)

    if residuals is not None and not residuals.empty:
        res_mean = residuals["Residual"].mean()
        res_std  = residuals["Residual"].std()
        bias_txt = (
            "slightly over-predicting" if res_mean < 0
            else "slightly under-predicting" if res_mean > 0
            else "well-centered"
        )
        _resid_insight_html = f"""
<div style="background:rgba(239,68,68,0.07);border:1px solid rgba(239,68,68,0.20);border-left:4px solid #EF4444;border-radius:12px;padding:12px 18px;margin-bottom:14px;display:flex;align-items:center;gap:12px;">
  <span style="font-size:1.3rem;flex-shrink:0;">&#x1F4C9;</span>
  <div style="color:#CBD5E1;font-size:0.85rem;line-height:1.6;">
    <strong style="color:#FCA5A5;">Residual summary:</strong>
    Mean residual = <strong style="color:#F8FAFC;">{res_mean:.4f}</strong>
    (model is {bias_txt}), std = <strong style="color:#F8FAFC;">{res_std:.4f}</strong>.
    Points tightly clustered around zero indicate a well-calibrated model with no systematic bias.
  </div>
</div>
"""
        st.markdown(_resid_insight_html, unsafe_allow_html=True)

    col_ra, col_rb = st.columns(2)

    with col_ra:
        fig_res = px.scatter(
            residuals,
            x="Predicted",
            y="Residual",
            opacity=0.4,
            color="Residual",
            color_continuous_scale="RdBu",
            title="Residuals vs Predicted",
        )
        fig_res.add_hline(y=0, line_dash="dash", line_color="#94A3B8")
        fig_res.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#F8FAFC",
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_res, use_container_width=True)

    with col_rb:
        fig_hist = px.histogram(
            residuals,
            x="Residual",
            nbins=50,
            title="Residual Distribution",
            color_discrete_sequence=["#6C63FF"],
        )
        fig_hist.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#F8FAFC",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    _avp_insight_html = """
<div style="background:rgba(108,99,255,0.07);border:1px solid rgba(108,99,255,0.18);border-left:4px solid #6C63FF;border-radius:12px;padding:12px 18px;margin-bottom:12px;display:flex;align-items:center;gap:12px;">
  <span style="font-size:1.3rem;flex-shrink:0;">&#x1F3AF;</span>
  <div style="color:#CBD5E1;font-size:0.85rem;line-height:1.6;">
    <strong style="color:#A5B4FC;">Actual vs Predicted:</strong>
    Points close to the diagonal dashed line indicate accurate predictions.
    Systematic deviations above or below reveal where the model consistently over- or under-estimates.
  </div>
</div>
"""
    st.markdown(_avp_insight_html, unsafe_allow_html=True)

    fig_avp = px.scatter(
        residuals,
        x="Actual",
        y="Predicted",
        opacity=0.35,
        title="Actual vs Predicted Ratings",
        color_discrete_sequence=["#00C2A8"],
    )
    fig_avp.add_shape(
        type="line",
        x0=residuals["Actual"].min(),
        y0=residuals["Actual"].min(),
        x1=residuals["Actual"].max(),
        y1=residuals["Actual"].max(),
        line=dict(color="#6C63FF", dash="dash", width=2),
    )
    fig_avp.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#F8FAFC",
    )
    st.plotly_chart(fig_avp, use_container_width=True)

    _res_stats_label_html = """
<div style="font-size:0.82rem;font-weight:700;text-transform:uppercase;letter-spacing:.12em;color:#A5B4FC;margin:16px 0 8px;padding-left:4px;display:flex;align-items:center;gap:8px;">
  <span style="display:inline-block;width:3px;height:16px;border-radius:2px;background:#6C63FF;"></span>
  Residual Summary Statistics
</div>
"""
    st.markdown(_res_stats_label_html, unsafe_allow_html=True)
    res_stats = residuals["Residual"].describe().to_frame().T.round(4)
    st.dataframe(res_stats, use_container_width=True)
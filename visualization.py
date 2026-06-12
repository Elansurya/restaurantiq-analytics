from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ── Design tokens ─────────────────────────────────────────────────────────────
PRIMARY   = "#6C63FF"
SECONDARY = "#00C2A8"
ACCENT    = "#F59E0B"
DANGER    = "#EF4444"
BG        = "#0F172A"
CARD      = "#1E293B"
TEXT      = "#F8FAFC"
MUTED     = "#94A3B8"

RATING_COLOR_MAP = {
    "Excellent": "#00C2A8",
    "Very Good": "#6C63FF",
    "Good":      "#F59E0B",
    "Average":   "#94A3B8",
    "Poor":      "#EF4444",
    "Not rated": "#334155",
}

PRICE_COLOR_MAP = {
    "Budget":     "#10B981",
    "Affordable": "#6C63FF",
    "Premium":    "#F59E0B",
    "Luxury":     "#EF4444",
}

LAYOUT_DEFAULTS = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=TEXT, family="Inter, sans-serif"),
    margin=dict(l=20, r=20, t=50, b=20),
    legend=dict(
        bgcolor="rgba(30,41,59,0.7)",
        bordercolor=PRIMARY,
        borderwidth=1,
    ),
)


# ── Safe boolean filter helper ────────────────────────────────────────────────

def _safe_filter(df: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    """Filter with .loc and always return a reset-index copy."""
    return df.loc[mask].copy().reset_index(drop=True)


def _rated(df: pd.DataFrame) -> pd.DataFrame:
    """Return only rated rows (Aggregate rating > 0)."""
    if "Is Rated" in df.columns:
        mask = df["Is Rated"].astype(bool)
    elif "Aggregate rating" in df.columns:
        mask = df["Aggregate rating"] > 0
    else:
        mask = pd.Series(True, index=df.index)
    return _safe_filter(df, mask)


# ── 1. Rating distribution histogram ─────────────────────────────────────────

def rating_distribution(df: pd.DataFrame) -> go.Figure:
    rated_df = _rated(df)
    if rated_df.empty or "Aggregate rating" not in rated_df.columns:
        rated_df = df.copy().reset_index(drop=True)

    fig = px.histogram(
        rated_df,
        x="Aggregate rating",
        nbins=30,
        color_discrete_sequence=[PRIMARY],
        title="Rating Distribution (Rated Restaurants)",
        labels={"Aggregate rating": "Rating"},
    )
    fig.update_traces(marker_line_color=SECONDARY, marker_line_width=0.6)
    fig.update_layout(**LAYOUT_DEFAULTS)
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)")
    return fig


# ── 2. Votes histogram ────────────────────────────────────────────────────────

def votes_histogram(df: pd.DataFrame) -> go.Figure:
    col_df = df.copy().reset_index(drop=True)
    if "Votes" not in col_df.columns:
        col_df["Votes"] = 0

    col_df["log_votes"] = np.log1p(col_df["Votes"].clip(lower=0))

    fig = px.histogram(
        col_df,
        x="log_votes",
        nbins=40,
        color_discrete_sequence=[SECONDARY],
        title="Votes Distribution (log scale)",
        labels={"log_votes": "log(1 + Votes)"},
    )
    fig.update_traces(marker_line_color=PRIMARY, marker_line_width=0.6)
    fig.update_layout(**LAYOUT_DEFAULTS)
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)")
    return fig


# ── 3. Rating vs Cost scatter ─────────────────────────────────────────────────

def rating_vs_cost_scatter(df: pd.DataFrame) -> go.Figure:
    rated_df = _rated(df)
    if rated_df.empty:
        rated_df = df.copy().reset_index(drop=True)

    needed = {"Aggregate rating", "Average Cost for two"}
    for col in needed:
        if col not in rated_df.columns:
            rated_df[col] = 0

    plot_df = rated_df.dropna(subset=list(needed)).reset_index(drop=True)
    plot_df = plot_df.loc[plot_df["Average Cost for two"] > 0].reset_index(drop=True)
    plot_df["log_cost"] = np.log1p(plot_df["Average Cost for two"])

    color_col = "Rating text" if "Rating text" in plot_df.columns else None

    fig = px.scatter(
        plot_df.sample(min(3000, len(plot_df)), random_state=1),
        x="log_cost",
        y="Aggregate rating",
        color=color_col,
        color_discrete_map=RATING_COLOR_MAP if color_col else None,
        opacity=0.6,
        title="Rating vs. Cost for Two (log scale)",
        labels={
            "log_cost": "log(Cost for Two)",
            "Aggregate rating": "Rating",
        },
        hover_data={c: True for c in
                    ["Restaurant Name", "City", "Primary Cuisine"]
                    if c in plot_df.columns},
    )
    fig.update_layout(**LAYOUT_DEFAULTS)
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)")
    return fig


# ── 4. Violin – rating by price tier ─────────────────────────────────────────

def violin_rating_by_price(df: pd.DataFrame) -> go.Figure:
    work = df.copy().reset_index(drop=True)

    if "Price Label" not in work.columns and "Price range" in work.columns:
        work["Price Label"] = pd.cut(
            work["Price range"].clip(1, 4),
            bins=[0, 1, 2, 3, 4],
            labels=["Budget", "Affordable", "Premium", "Luxury"],
        ).astype(str)

    if "Is Rated" in work.columns:
        rated_mask = work["Is Rated"].astype(bool)
    elif "Aggregate rating" in work.columns:
        rated_mask = work["Aggregate rating"] > 0
    else:
        rated_mask = pd.Series(True, index=work.index)

    rated_df = work.loc[rated_mask].reset_index(drop=True)

    if rated_df.empty or "Aggregate rating" not in rated_df.columns:
        rated_df = work.reset_index(drop=True)

    if "Price Label" not in rated_df.columns:
        rated_df["Price Label"] = "Unknown"

    order  = ["Budget", "Affordable", "Premium", "Luxury"]
    colors = [PRICE_COLOR_MAP.get(p, PRIMARY) for p in order]

    fig = go.Figure()
    for price, color in zip(order, colors):
        subset = rated_df.loc[
            rated_df["Price Label"].astype(str) == price, "Aggregate rating"
        ].dropna()
        if subset.empty:
            continue
        fig.add_trace(go.Violin(
            y=subset,
            name=price,
            box_visible=True,
            meanline_visible=True,
            fillcolor=color,
            line_color=color,
            opacity=0.7,
        ))

    fig.update_layout(
        title="Rating Distribution by Price Tier",
        yaxis_title="Aggregate Rating",
        **LAYOUT_DEFAULTS,
    )
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)")
    return fig


# ── 5. Heatmap – city × cuisine ──────────────────────────────────────────────

def heatmap_city_rating(df: pd.DataFrame, top_n: int = 12) -> go.Figure:
    work = df.copy().reset_index(drop=True)

    if "Is Rated" in work.columns:
        rated_mask = work["Is Rated"].astype(bool)
    elif "Aggregate rating" in work.columns:
        rated_mask = work["Aggregate rating"] > 0
    else:
        rated_mask = pd.Series(True, index=work.index)

    rated_df = work.loc[rated_mask].reset_index(drop=True)

    if rated_df.empty or "City" not in rated_df.columns or "Primary Cuisine" not in rated_df.columns:
        fig = go.Figure()
        fig.update_layout(title="Not enough data for heatmap", **LAYOUT_DEFAULTS)
        return fig

    top_cities = (
        rated_df["City"].value_counts().head(top_n).index.tolist()
    )
    top_cuisines = (
        rated_df["Primary Cuisine"].value_counts().head(10).index.tolist()
    )

    sub = rated_df.loc[
        rated_df["City"].isin(top_cities) &
        rated_df["Primary Cuisine"].isin(top_cuisines)
    ].reset_index(drop=True)

    if sub.empty:
        fig = go.Figure()
        fig.update_layout(title="Not enough data for heatmap", **LAYOUT_DEFAULTS)
        return fig

    pivot = (
        sub.groupby(["City", "Primary Cuisine"])["Aggregate rating"]
        .mean()
        .round(2)
        .unstack(fill_value=0)
    )

    fig = px.imshow(
        pivot,
        color_continuous_scale=[[0, CARD], [0.5, PRIMARY], [1, SECONDARY]],
        title=f"Avg Rating: Top {top_n} Cities × Top 10 Cuisines",
        labels={"color": "Avg Rating"},
        aspect="auto",
    )
    fig.update_layout(**LAYOUT_DEFAULTS)
    return fig


# ── 6. Top cuisines by avg rating ─────────────────────────────────────────────

def top_cuisines_avg_rating(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    work = df.copy().reset_index(drop=True)

    if "Primary Cuisine" not in work.columns or "Aggregate rating" not in work.columns:
        fig = go.Figure()
        fig.update_layout(title="Primary Cuisine / rating data not available", **LAYOUT_DEFAULTS)
        return fig

    if "Is Rated" in work.columns:
        rated_mask = work["Is Rated"].astype(bool)
    else:
        rated_mask = work["Aggregate rating"] > 0

    rated_df = work.loc[rated_mask].reset_index(drop=True)

    if rated_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No rated restaurants found", **LAYOUT_DEFAULTS)
        return fig

    stats = (
        rated_df.groupby("Primary Cuisine")
        .agg(
            avg_rating=("Aggregate rating", "mean"),
            count=("Aggregate rating", "count"),
        )
        .reset_index()
    )

    stats = stats.loc[stats["count"] >= 20].reset_index(drop=True)

    if stats.empty:
        stats = (
            rated_df.groupby("Primary Cuisine")
            .agg(
                avg_rating=("Aggregate rating", "mean"),
                count=("Aggregate rating", "count"),
            )
            .reset_index()
        )

    top = stats.nlargest(top_n, "avg_rating")

    fig = px.bar(
        top,
        x="avg_rating",
        y="Primary Cuisine",
        orientation="h",
        color="avg_rating",
        color_continuous_scale=[[0, DANGER], [0.5, PRIMARY], [1, SECONDARY]],
        title=f"Top {top_n} Cuisines by Average Rating",
        labels={"avg_rating": "Avg Rating", "Primary Cuisine": "Cuisine"},
        text=top["avg_rating"].round(2),
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(**LAYOUT_DEFAULTS, yaxis=dict(autorange="reversed"))
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)", range=[0, 5.5])
    return fig


# ── 7. Cost by country ────────────────────────────────────────────────────────

def cost_by_country(df: pd.DataFrame) -> go.Figure:
    work = df.copy().reset_index(drop=True)

    if "Country" not in work.columns or "Average Cost for two" not in work.columns:
        fig = go.Figure()
        fig.update_layout(title="Cost / country data not available", **LAYOUT_DEFAULTS)
        return fig

    stats = (
        work.groupby("Country")["Average Cost for two"]
        .median()
        .sort_values(ascending=False)
        .reset_index()
    )
    stats.columns = ["Country", "Median Cost"]

    fig = px.bar(
        stats,
        x="Median Cost",
        y="Country",
        orientation="h",
        color="Median Cost",
        color_continuous_scale=[[0, PRIMARY], [1, ACCENT]],
        title="Median Cost for Two by Country",
        labels={"Median Cost": "Median Cost (local currency)", "Country": ""},
        text=stats["Median Cost"].round(0),
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(**LAYOUT_DEFAULTS, yaxis=dict(autorange="reversed"))
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)")
    return fig


# ── 8. Cuisine treemap ────────────────────────────────────────────────────────

def cuisine_treemap(df: pd.DataFrame, top_n: int = 25) -> go.Figure:
    work = df.copy().reset_index(drop=True)

    if "Primary Cuisine" not in work.columns:
        fig = go.Figure()
        fig.update_layout(title="Primary Cuisine column not found", **LAYOUT_DEFAULTS)
        return fig

    counts = (
        work["Primary Cuisine"].value_counts().head(top_n).reset_index()
    )
    counts.columns = ["Cuisine", "Count"]

    fig = px.treemap(
        counts,
        path=["Cuisine"],
        values="Count",
        color="Count",
        color_continuous_scale=[[0, CARD], [0.5, PRIMARY], [1, SECONDARY]],
        title=f"Top {top_n} Cuisines by Restaurant Count",
    )
    fig.update_layout(**LAYOUT_DEFAULTS)
    return fig


# ── 9. City bar ───────────────────────────────────────────────────────────────

def city_bar(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    work = df.copy().reset_index(drop=True)

    if "City" not in work.columns:
        fig = go.Figure()
        fig.update_layout(title="City column not found", **LAYOUT_DEFAULTS)
        return fig

    counts = work["City"].value_counts().head(top_n).reset_index()
    counts.columns = ["City", "Count"]

    fig = px.bar(
        counts,
        x="Count",
        y="City",
        orientation="h",
        color="Count",
        color_continuous_scale=[[0, PRIMARY], [1, SECONDARY]],
        title=f"Top {top_n} Cities by Restaurant Count",
        text="Count",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(**LAYOUT_DEFAULTS, yaxis=dict(autorange="reversed"))
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)")
    return fig


# ── 10. Country bar ───────────────────────────────────────────────────────────

def country_bar(df: pd.DataFrame) -> go.Figure:
    work = df.copy().reset_index(drop=True)

    if "Country" not in work.columns:
        fig = go.Figure()
        fig.update_layout(title="Country column not found", **LAYOUT_DEFAULTS)
        return fig

    counts = work["Country"].value_counts().reset_index()
    counts.columns = ["Country", "Count"]

    fig = px.bar(
        counts,
        x="Count",
        y="Country",
        orientation="h",
        color="Count",
        color_continuous_scale=[[0, PRIMARY], [1, SECONDARY]],
        title="Restaurant Count by Country",
        text="Count",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(**LAYOUT_DEFAULTS, yaxis=dict(autorange="reversed"))
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)")
    return fig


# ── 11. Descriptive stats table ───────────────────────────────────────────────

def descriptive_stats_table(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = [
        c for c in [
            "Aggregate rating", "Votes", "Average Cost for two",
            "Price range", "Cuisine Count",
        ]
        if c in df.columns
    ]
    if not num_cols:
        return pd.DataFrame({"Info": ["No numeric columns found"]})

    stats = df[num_cols].apply(pd.to_numeric, errors="coerce").describe().T
    stats.index.name = "Column"
    stats = stats.reset_index()
    return stats.round(3)


# ── 12. Delivery & booking bar ────────────────────────────────────────────────

def delivery_booking_bar(df: pd.DataFrame) -> go.Figure:
    work = df.copy().reset_index(drop=True)

    service_cols = {
        "Has Online delivery": "Online Delivery",
        "Has Table booking":   "Table Booking",
        "Is delivering now":   "Delivering Now",
    }

    rows = []
    for col, label in service_cols.items():
        if col in work.columns:
            pct = work[col].astype(bool).mean() * 100
            rows.append({"Service": label, "Percentage": round(pct, 1)})

    if not rows:
        fig = go.Figure()
        fig.update_layout(title="Service columns not found", **LAYOUT_DEFAULTS)
        return fig

    data = pd.DataFrame(rows)

    fig = px.bar(
        data,
        x="Service",
        y="Percentage",
        color="Service",
        color_discrete_sequence=[PRIMARY, SECONDARY, ACCENT],
        title="Service Feature Adoption (%)",
        text=data["Percentage"].apply(lambda x: f"{x:.1f}%"),
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        yaxis=dict(range=[0, 110], gridcolor="rgba(255,255,255,0.06)"),
        showlegend=False,
    )
    return fig


# ── 13. Rating text donut ─────────────────────────────────────────────────────

def rating_text_donut(df: pd.DataFrame) -> go.Figure:
    work = df.copy().reset_index(drop=True)

    if "Rating text" not in work.columns:
        fig = go.Figure()
        fig.update_layout(title="Rating text column not found", **LAYOUT_DEFAULTS)
        return fig

    counts = (
        work["Rating text"]
        .fillna("Not rated")
        .value_counts()
        .reset_index()
    )
    counts.columns = ["Rating", "Count"]

    order = ["Excellent", "Very Good", "Good", "Average", "Poor", "Not rated"]
    counts["Rating"] = pd.Categorical(
        counts["Rating"], categories=order, ordered=True
    )
    counts = counts.sort_values("Rating").reset_index(drop=True)

    colors = [RATING_COLOR_MAP.get(str(r), MUTED) for r in counts["Rating"]]

    fig = go.Figure(go.Pie(
        labels=counts["Rating"],
        values=counts["Count"],
        hole=0.55,
        marker=dict(colors=colors, line=dict(color=BG, width=2)),
        textinfo="label+percent",
        textfont=dict(size=12, color=TEXT),
        hovertemplate=(
            "<b>%{label}</b><br>"
            "Count: %{value:,}<br>"
            "Share: %{percent}<extra></extra>"
        ),
    ))
    fig.update_layout(
        title="Rating Text Distribution",
        **LAYOUT_DEFAULTS,
    )
    return fig


# ── 14. Price tier donut ──────────────────────────────────────────────────────

def price_donut(df: pd.DataFrame) -> go.Figure:
    work = df.copy().reset_index(drop=True)

    if "Price Label" not in work.columns and "Price range" in work.columns:
        work["Price Label"] = pd.cut(
            work["Price range"].clip(1, 4),
            bins=[0, 1, 2, 3, 4],
            labels=["Budget", "Affordable", "Premium", "Luxury"],
        ).astype(str)

    if "Price Label" not in work.columns:
        fig = go.Figure()
        fig.update_layout(title="Price Label column not found", **LAYOUT_DEFAULTS)
        return fig

    counts = (
        work["Price Label"]
        .fillna("Unknown")
        .value_counts()
        .reset_index()
    )
    counts.columns = ["Price", "Count"]

    order = ["Budget", "Affordable", "Premium", "Luxury"]
    counts["Price"] = pd.Categorical(
        counts["Price"], categories=order, ordered=True
    )
    counts = counts.sort_values("Price").reset_index(drop=True)

    colors = [PRICE_COLOR_MAP.get(str(p), MUTED) for p in counts["Price"]]

    fig = go.Figure(go.Pie(
        labels=counts["Price"],
        values=counts["Count"],
        hole=0.55,
        marker=dict(colors=colors, line=dict(color=BG, width=2)),
        textinfo="label+percent",
        textfont=dict(size=12, color=TEXT),
        hovertemplate=(
            "<b>%{label}</b><br>"
            "Count: %{value:,}<br>"
            "Share: %{percent}<extra></extra>"
        ),
    ))
    fig.update_layout(
        title="Price Tier Distribution",
        **LAYOUT_DEFAULTS,
    )
    return fig
from __future__ import annotations
import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Optional heavy deps ───────────────────────────────────────────────────────
try:
    import plotly.express as px
    import plotly.graph_objects as go
    _PLOTLY_OK = True
except ImportError:
    _PLOTLY_OK = False
    print("[WARN] plotly not installed – Plotly maps disabled.")

try:
    import folium
    from folium.plugins import HeatMap, MarkerCluster
    _FOLIUM_OK = True
except ImportError:
    _FOLIUM_OK = False

try:
    import pydeck as pdk
    _PYDECK_OK = True
except ImportError:
    _PYDECK_OK = False

# ── Design tokens ─────────────────────────────────────────────────────────────
PALETTE = {
    "primary":   "#6C63FF",
    "secondary": "#00C2A8",
    "accent":    "#F59E0B",
    "danger":    "#EF4444",
    "bg":        "#0F172A",
    "card":      "#1E293B",
    "text":      "#F8FAFC",
    "muted":     "#94A3B8",
}
MAPBOX_STYLE = "carto-darkmatter"

RATING_COLOR_MAP = {
    "Excellent":  "#00C2A8",
    "Very Good":  "#6C63FF",
    "Good":       "#F59E0B",
    "Average":    "#94A3B8",
    "Poor":       "#EF4444",
    "Not rated":  "#334155",
}
FOLIUM_RATING_COLOR = {
    "Excellent": "green",
    "Very Good": "blue",
    "Good":      "lightblue",
    "Average":   "gray",
    "Poor":      "red",
    "Not rated": "darkgray",
}


# ── Safe boolean filter helpers ───────────────────────────────────────────────

def _safe_filter(df: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    """Filter with .loc and return a reset-index copy — never df[series]."""
    return df.loc[mask.values].copy().reset_index(drop=True)


def _get_rated(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return only rated rows.
    Falls back to full df if nothing is rated, so callers never get empty geo.
    """
    if "Is Rated" in df.columns:
        mask = df["Is Rated"].astype(bool)
    elif "Aggregate rating" in df.columns:
        mask = df["Aggregate rating"] > 0
    else:
        return df.copy().reset_index(drop=True)

    rated = _safe_filter(df, mask)
    return rated if not rated.empty else df.copy().reset_index(drop=True)


# ── Sample data generator ─────────────────────────────────────────────────────

def _make_sample_df(n: int = 800, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    city_anchors = [
        ("Mumbai",       "India",        19.076,  72.877),
        ("Delhi",        "India",        28.704,  77.102),
        ("Bangalore",    "India",        12.972,  77.594),
        ("Chennai",      "India",        13.083,  80.270),
        ("New York",     "USA",          40.712, -74.006),
        ("Los Angeles",  "USA",          34.052,-118.244),
        ("Chicago",      "USA",          41.878, -87.630),
        ("London",       "UK",           51.507,  -0.128),
        ("Manchester",   "UK",           53.483,  -2.244),
        ("Sydney",       "Australia",   -33.869, 151.209),
        ("Melbourne",    "Australia",   -37.814, 144.963),
        ("Toronto",      "Canada",       43.653, -79.383),
        ("Vancouver",    "Canada",       49.283,-123.121),
        ("Dubai",        "UAE",          25.205,  55.270),
        ("Cape Town",    "South Africa",-33.925,  18.424),
        ("Manila",       "Philippines",  14.599, 120.984),
    ]
    cuisines = [
        "North Indian","Chinese","Fast Food","Continental","Italian",
        "Mexican","Japanese","Thai","American","Mediterranean",
        "Middle Eastern","Korean","French","Seafood","Bakery",
    ]
    price_labels = ["Budget","Affordable","Premium","Luxury"]
    anchor_idx   = rng.integers(0, len(city_anchors), n)
    cities_data  = [city_anchors[i] for i in anchor_idx]
    city_col     = [c[0] for c in cities_data]
    country_col  = [c[1] for c in cities_data]
    lat_col      = np.array([c[2] for c in cities_data]) + rng.normal(0, 0.08, n)
    lon_col      = np.array([c[3] for c in cities_data]) + rng.normal(0, 0.08, n)
    cuisine_col  = rng.choice(cuisines, n)
    price_range  = rng.integers(1, 5, n)
    cost_col     = rng.integers(100, 5000, n).astype(float)
    votes_col    = rng.integers(0, 5000, n)
    rating_raw   = np.where(rng.random(n) < 0.12, 0.0,
                            np.round(rng.uniform(1.5, 5.0, n), 1))

    def _rt(r):
        if r == 0:   return "Not rated"
        if r >= 4.5: return "Excellent"
        if r >= 4.0: return "Very Good"
        if r >= 3.5: return "Good"
        if r >= 3.0: return "Average"
        return "Poor"

    n_cuisines   = rng.integers(1, 4, n)
    cuisines_col = [
        ", ".join(rng.choice(cuisines, int(k), replace=False).tolist())
        for k in n_cuisines
    ]
    return pd.DataFrame({
        "Restaurant ID":        range(1, n + 1),
        "Restaurant Name":      [f"Restaurant_{i}" for i in range(1, n + 1)],
        "Country":              country_col,
        "City":                 city_col,
        "Cuisines":             cuisines_col,
        "Primary Cuisine":      cuisine_col,
        "Latitude":             lat_col,
        "Longitude":            lon_col,
        "Average Cost for two": cost_col,
        "Price range":          price_range,
        "Price Label":          [price_labels[p - 1] for p in price_range],
        "Aggregate rating":     rating_raw,
        "Rating text":          [_rt(r) for r in rating_raw],
        "Votes":                votes_col,
        "Is Rated":             rating_raw > 0,
        "Cuisine Count":        n_cuisines,
        "Has Online delivery":  rng.choice([True, False], n, p=[0.4, 0.6]),
        "Has Table booking":    rng.choice([True, False], n, p=[0.3, 0.7]),
        "Is delivering now":    rng.choice([True, False], n, p=[0.2, 0.8]),
    })


# ── Derived-column guard ──────────────────────────────────────────────────────

def _ensure_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Is Rated" not in df.columns:
        df["Is Rated"] = df.get("Aggregate rating", pd.Series(0, index=df.index)) > 0
    if "Primary Cuisine" not in df.columns and "Cuisines" in df.columns:
        df["Primary Cuisine"] = df["Cuisines"].str.split(",").str[0].str.strip()
    if "Price Label" not in df.columns and "Price range" in df.columns:
        df["Price Label"] = pd.cut(
            df["Price range"].clip(1, 4),
            bins=[0, 1, 2, 3, 4],
            labels=["Budget", "Affordable", "Premium", "Luxury"],
        ).astype(str)
    if "Cuisine Count" not in df.columns and "Cuisines" in df.columns:
        df["Cuisine Count"] = df["Cuisines"].str.split(",").str.len()
    if "Rating text" not in df.columns:
        def _rt(r):
            if r == 0:   return "Not rated"
            if r >= 4.5: return "Excellent"
            if r >= 4.0: return "Very Good"
            if r >= 3.5: return "Good"
            if r >= 3.0: return "Average"
            return "Poor"
        df["Rating text"] = df.get(
            "Aggregate rating", pd.Series(0, index=df.index)
        ).apply(_rt)
    for col in ["Has Online delivery", "Has Table booking", "Is delivering now"]:
        if col not in df.columns:
            df[col] = False
    return df.reset_index(drop=True)


# ── Geo cleaner ───────────────────────────────────────────────────────────────

def _clean_geo(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with missing or invalid coordinates."""
    if "Latitude" not in df.columns or "Longitude" not in df.columns:
        raise ValueError("DataFrame must have 'Latitude' and 'Longitude' columns.")
    work = df.copy().reset_index(drop=True)
    mask = (
        work["Latitude"].notna() &
        work["Longitude"].notna() &
        work["Latitude"].between(-90, 90) &
        work["Longitude"].between(-180, 180) &
        (work["Latitude"] != 0) &
        (work["Longitude"] != 0)
    )
    # Use .loc with mask.values to avoid index-as-column-label bug
    return work.loc[mask.values].copy().reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTLY MAPBOX CHARTS
# ═══════════════════════════════════════════════════════════════════════════════

def plotly_scatter_map(df: pd.DataFrame, color_by: str = "Rating text") -> "go.Figure":
    """Scatter map coloured by a chosen field."""
    geo    = _clean_geo(df)
    sample = geo.sample(min(5000, len(geo)), random_state=42).reset_index(drop=True)

    color_map  = RATING_COLOR_MAP if color_by == "Rating text" else None
    hover_cols = {k: True for k in
                  ["City", "Cuisines", "Aggregate rating", "Votes"]
                  if k in sample.columns}
    hover_cols.update({"Latitude": False, "Longitude": False})

    fig = px.scatter_mapbox(
        sample,
        lat="Latitude", lon="Longitude",
        color=color_by if color_by in sample.columns else None,
        color_discrete_map=color_map,
        hover_name="Restaurant Name" if "Restaurant Name" in sample.columns else None,
        hover_data=hover_cols,
        zoom=2, height=560,
        title=f"Restaurant Map — coloured by {color_by}",
        mapbox_style=MAPBOX_STYLE,
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=PALETTE["text"]),
        legend=dict(bgcolor="rgba(30,41,59,0.7)", bordercolor=PALETTE["primary"], borderwidth=1),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def plotly_density_map(df: pd.DataFrame) -> "go.Figure":
    """Density heatmap of restaurant concentration."""
    geo = _clean_geo(df)
    fig = px.density_mapbox(
        geo, lat="Latitude", lon="Longitude",
        radius=12, zoom=2, height=560,
        color_continuous_scale=["#0F172A", "#6C63FF", "#00C2A8", "#F59E0B"],
        mapbox_style=MAPBOX_STYLE,
        title="Restaurant Density Heatmap",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=PALETTE["text"]),
        coloraxis_showscale=False,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def plotly_rating_bubble_map(df: pd.DataFrame) -> "go.Figure":
    """
    Bubble map: size = Votes, colour = Aggregate rating.
    Uses _get_rated() / _clean_geo() — no df[bool_series] indexing.
    """
    # Get rated rows safely
    rated = _get_rated(df)
    geo   = _clean_geo(rated).copy()

    # Fall back to full dataset if geo is empty
    if geo.empty:
        geo = _clean_geo(df).copy()

    if geo.empty:
        fig = go.Figure()
        fig.update_layout(title="No valid geo data for bubble map",
                          paper_bgcolor="rgba(0,0,0,0)")
        return fig

    geo["size_scaled"] = np.log1p(
        pd.to_numeric(geo.get("Votes", 0), errors="coerce").fillna(0)
    ) * 3 + 3

    hover_cols = {k: True for k in
                  ["Aggregate rating", "Votes", "City"]
                  if k in geo.columns}
    hover_cols.update({"size_scaled": False, "Latitude": False, "Longitude": False})

    sample = geo.sample(min(3000, len(geo)), random_state=1).reset_index(drop=True)

    fig = px.scatter_mapbox(
        sample,
        lat="Latitude", lon="Longitude",
        size="size_scaled",
        color="Aggregate rating" if "Aggregate rating" in sample.columns else None,
        color_continuous_scale=["#EF4444", "#F59E0B", "#6C63FF", "#00C2A8"],
        range_color=[1, 5],
        hover_name="Restaurant Name" if "Restaurant Name" in sample.columns else None,
        hover_data=hover_cols,
        zoom=2, height=560,
        mapbox_style=MAPBOX_STYLE,
        title="Ratings & Popularity Bubble Map",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=PALETTE["text"]),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def city_avg_rating_map(df: pd.DataFrame) -> "go.Figure":
    """Bubble map aggregated at city level."""
    rated = _get_rated(df)
    geo   = _clean_geo(rated)
    if geo.empty:
        geo = _clean_geo(df)

    if geo.empty:
        fig = go.Figure()
        fig.update_layout(title="No valid geo data", paper_bgcolor="rgba(0,0,0,0)")
        return fig

    city_stats = (
        geo.groupby("City")
        .agg(
            avg_rating        =("Aggregate rating", "mean"),
            total_restaurants =("Restaurant ID",    "count"),
            lat               =("Latitude",         "median"),
            lon               =("Longitude",        "median"),
            country           =("Country",          "first"),
        )
        .reset_index()
    )
    city_stats["size"] = np.sqrt(city_stats["total_restaurants"]) * 2

    hover_cols = {
        "avg_rating": ":.2f", "total_restaurants": True,
        "country": True, "lat": False, "lon": False, "size": False,
    }

    fig = px.scatter_mapbox(
        city_stats,
        lat="lat", lon="lon",
        size="size",
        color="avg_rating",
        color_continuous_scale=["#EF4444", "#F59E0B", "#6C63FF", "#00C2A8"],
        range_color=[2, 5],
        hover_name="City",
        hover_data=hover_cols,
        zoom=2, height=560,
        mapbox_style=MAPBOX_STYLE,
        title="City Intelligence — Avg Rating & Restaurant Count",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=PALETTE["text"]),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# FOLIUM MAPS
# ═══════════════════════════════════════════════════════════════════════════════

def folium_cluster_map(df: pd.DataFrame, max_points: int = 2000) -> "folium.Map":
    """Folium map with marker clusters and rich popups."""
    geo    = _clean_geo(df)
    sample = geo.sample(min(max_points, len(geo)), random_state=42).reset_index(drop=True)

    center_lat = float(sample["Latitude"].median())
    center_lon = float(sample["Longitude"].median())

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=4,
        tiles="CartoDB dark_matter",
    )
    cluster = MarkerCluster(
        options={"maxClusterRadius": 50, "spiderfyOnMaxZoom": True}
    ).add_to(m)

    for _, row in sample.iterrows():
        icon_color = FOLIUM_RATING_COLOR.get(str(row.get("Rating text", "")), "gray")
        cuisines_str = str(row.get("Cuisines", ""))[:40]
        rating_val   = row.get("Aggregate rating", "N/A")
        rating_text  = row.get("Rating text", "N/A")
        price_label  = row.get("Price Label", "N/A")
        price_range  = row.get("Price range", "N/A")
        votes        = int(row.get("Votes", 0))

        popup_html = f"""
        <div style="font-family:Inter,sans-serif;min-width:200px;">
          <b style="font-size:14px;">{row.get('Restaurant Name','')}</b><br>
          <span style="color:#6C63FF;">📍 {row.get('City','')}, {row.get('Country','')}</span><br>
          <span>🍽️ {cuisines_str}</span><br>
          <span>⭐ {rating_val} — {rating_text}</span><br>
          <span>💰 {price_label} (range {price_range})</span><br>
          <span>🗳️ {votes:,} votes</span>
        </div>"""

        folium.Marker(
            location=[row["Latitude"], row["Longitude"]],
            popup=folium.Popup(popup_html, max_width=280),
            tooltip=str(row.get("Restaurant Name", "")),
            icon=folium.Icon(color=icon_color, icon="cutlery", prefix="fa"),
        ).add_to(cluster)

    return m


def folium_heatmap(df: pd.DataFrame, weight_col: str = "Votes") -> "folium.Map":
    """
    Folium heatmap weighted by Votes or Aggregate rating.
    Uses .loc[] — no df[bool_series] indexing.
    """
    geo = _clean_geo(df).copy()

    if weight_col == "Votes":
        votes_mask = pd.to_numeric(geo.get("Votes", 0), errors="coerce").fillna(0) > 0
        geo = geo.loc[votes_mask.values].copy().reset_index(drop=True)
        geo["weight"] = np.log1p(
            pd.to_numeric(geo["Votes"], errors="coerce").fillna(0)
        )
    else:
        if "Is Rated" in geo.columns:
            rated_mask = geo["Is Rated"].astype(bool)
        elif "Aggregate rating" in geo.columns:
            rated_mask = geo["Aggregate rating"] > 0
        else:
            rated_mask = pd.Series(True, index=geo.index)
        geo = geo.loc[rated_mask.values].copy().reset_index(drop=True)
        geo["weight"] = pd.to_numeric(
            geo.get("Aggregate rating", pd.Series(1, index=geo.index)),
            errors="coerce"
        ).fillna(1)

    if geo.empty:
        geo = _clean_geo(df).copy().reset_index(drop=True)
        geo["weight"] = 1.0

    center_lat = float(geo["Latitude"].median())
    center_lon = float(geo["Longitude"].median())

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=4,
        tiles="CartoDB dark_matter",
    )
    heat_data = geo[["Latitude", "Longitude", "weight"]].values.tolist()
    HeatMap(
        heat_data,
        min_opacity=0.3,
        radius=12,
        blur=15,
        gradient={0.2: "#6C63FF", 0.5: "#00C2A8", 0.8: "#F59E0B", 1.0: "#EF4444"},
    ).add_to(m)
    return m


# ═══════════════════════════════════════════════════════════════════════════════
# PYDECK MAPS  (FIXED)
# ═══════════════════════════════════════════════════════════════════════════════

def pydeck_hexagon_layer(df: pd.DataFrame) -> "pdk.Deck":
    """3-D hexagon density layer."""
    # ── Defensive checks ──────────────────────────────────────────────────────
    if df.empty:
        raise ValueError("DataFrame is empty — no data to render in hexagon layer.")
    for col in ("Latitude", "Longitude"):
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' is missing from DataFrame.")

    geo = _clean_geo(df)
    if geo.empty:
        raise ValueError(
            "No valid coordinates found after cleaning. "
            "Check that Latitude/Longitude columns contain non-zero, in-range values."
        )

    geo = geo[["Latitude", "Longitude"]].copy().reset_index(drop=True)
    geo.columns = ["lat", "lon"]

    layer = pdk.Layer(
        "HexagonLayer",
        data=geo,
        get_position=["lon", "lat"],
        radius=5000,
        elevation_scale=50,
        elevation_range=[0, 3000],
        extruded=True,
        pickable=True,
        coverage=1,
        color_range=[
            [15, 23, 42, 200],
            [108, 99, 255, 220],
            [0, 194, 168, 240],
            [245, 158, 11, 255],
            [239, 68, 68, 255],
        ],
    )
    view_state = pdk.ViewState(
        latitude=float(geo["lat"].median()),
        longitude=float(geo["lon"].median()),
        zoom=3, pitch=45, bearing=0,
    )
    # FIX: use open tile style — "mapbox://…" requires a Mapbox token and renders blank without one
    return pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        map_style="dark",
        tooltip={"text": "Restaurant Cluster\nCount: {elevationValue}"},
    )


def pydeck_scatter_layer(df: pd.DataFrame) -> "pdk.Deck":
    """
    PyDeck scatterplot coloured by rating tier.
    Uses _get_rated() — no df[bool_series] indexing.
    Colors stored as [R,G,B,A] lists; PyDeck serialises object columns of lists correctly.
    """
    # ── Defensive checks ──────────────────────────────────────────────────────
    if df.empty:
        raise ValueError("DataFrame is empty — no data to render in scatter layer.")
    for col in ("Latitude", "Longitude"):
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' is missing from DataFrame.")

    rated = _get_rated(df)
    geo   = _clean_geo(rated).copy()
    if geo.empty:
        geo = _clean_geo(df).copy()
    if geo.empty:
        raise ValueError(
            "No valid coordinates found after cleaning. "
            "Check that Latitude/Longitude columns contain non-zero, in-range values."
        )

    def _rating_to_rgba(r: float) -> list:
        """Return [R, G, B, A] list for a given rating value."""
        if r >= 4.5:   return [0, 194, 168, 220]
        elif r >= 4.0: return [108, 99, 255, 220]
        elif r >= 3.5: return [245, 158, 11, 220]
        elif r >= 3.0: return [148, 163, 184, 200]
        else:          return [239, 68, 68, 200]

    geo = geo.reset_index(drop=True)
    rating_series = pd.to_numeric(
        geo.get("Aggregate rating", pd.Series(0, index=geo.index)),
        errors="coerce"
    ).fillna(0)
    # Build color as Python lists — list comprehension avoids .apply() serialisation quirks
    geo["color"]  = [_rating_to_rgba(r) for r in rating_series]
    geo["radius"] = (
        np.log1p(pd.to_numeric(geo.get("Votes", 0), errors="coerce").fillna(0))
        * 300 + 200
    )

    cols_needed    = ["Latitude", "Longitude", "color", "radius",
                      "Restaurant Name", "Aggregate rating", "City"]
    cols_available = [c for c in cols_needed if c in geo.columns]
    layer_df = geo[cols_available].rename(
        columns={"Latitude": "lat", "Longitude": "lon"}
    )

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=layer_df,
        get_position=["lon", "lat"],
        get_color="color",
        get_radius="radius",
        pickable=True,
        opacity=0.8,
        stroked=True,
        get_line_color=[255, 255, 255, 30],
        line_width_min_pixels=1,
    )
    view_state = pdk.ViewState(
        latitude=float(geo["Latitude"].median()),
        longitude=float(geo["Longitude"].median()),
        zoom=3, pitch=30,
    )
    # FIX: use open tile style — "mapbox://…" requires a Mapbox token and renders blank without one
    return pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        map_style="dark",
        tooltip={"text": "{Restaurant Name}\n⭐ {Aggregate rating}\n📍 {City}"},
    )


# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT / PRINT UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

_GEO_CHART_REGISTRY = [
    ("plotly_scatter", "Scatter Map (by Rating)",         plotly_scatter_map,      "plotly"),
    ("plotly_density", "Density Heatmap",                 plotly_density_map,      "plotly"),
    ("plotly_bubble",  "Ratings & Popularity Bubble Map", plotly_rating_bubble_map,"plotly"),
    ("plotly_city",    "City Intelligence Map",           city_avg_rating_map,     "plotly"),
    ("folium_cluster", "Marker Cluster Map (Folium)",     folium_cluster_map,      "folium"),
    ("folium_heatmap", "Votes Heatmap (Folium)",          folium_heatmap,          "folium"),
    ("pydeck_hexagon", "3-D Hexagon Density (PyDeck)",    pydeck_hexagon_layer,    "pydeck"),
    ("pydeck_scatter", "Rating Scatter Layer (PyDeck)",   pydeck_scatter_layer,    "pydeck"),
]


def print_geo_summary(df: pd.DataFrame) -> None:
    sep = "=" * 62
    print(sep)
    print("  RestaurantIQ – Geospatial Dataset Summary")
    print(sep)
    print(f"  Total rows              : {len(df):,}")
    try:
        geo = _clean_geo(df)
        print(f"  Valid geo rows          : {len(geo):,} ({len(geo)/len(df)*100:.1f}%)")
        print(f"  Countries               : {geo['Country'].nunique()}")
        print(f"  Cities                  : {geo['City'].nunique()}")
        print(f"  Latitude  range         : {geo['Latitude'].min():.3f} → {geo['Latitude'].max():.3f}")
        print(f"  Longitude range         : {geo['Longitude'].min():.3f} → {geo['Longitude'].max():.3f}")
        rated = _get_rated(geo)
        print(f"  Rated restaurants       : {len(rated):,} ({len(rated)/len(geo)*100:.1f}%)")
        if "Aggregate rating" in rated.columns:
            print(f"  Avg rating (rated)      : {rated['Aggregate rating'].mean():.2f}")
        if "Votes" in df.columns:
            print(f"  Avg votes               : {df['Votes'].mean():.0f}")
    except Exception as exc:
        print(f"  [geo summary error] {exc}")
    print(f"  Plotly : {'Yes' if _PLOTLY_OK else 'No'}  "
          f"Folium : {'Yes' if _FOLIUM_OK else 'No'}  "
          f"PyDeck : {'Yes' if _PYDECK_OK else 'No'}")
    print(sep)


def save_all_geo_charts(
    df: pd.DataFrame,
    output_dir: str = "restaurant_iq_geo_charts",
    fmt: str = "html",
    skip_folium: bool = False,
    skip_pydeck: bool = False,
    width: int = 1400,
    height: int = 700,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    total = len(_GEO_CHART_REGISTRY)
    done = skipped = failed = 0
    print(f"\n{'='*62}\n  Saving {total} geo charts → '{output_dir}/' (fmt: {fmt})\n{'='*62}")

    for idx, (key, label, fn, engine) in enumerate(_GEO_CHART_REGISTRY, 1):
        prefix = f"  [{idx:02d}/{total}] {label}"
        if engine == "plotly" and not _PLOTLY_OK:
            print(f"{prefix} ... ⊘ skipped (plotly not installed)"); skipped += 1; continue
        if engine == "folium" and (not _FOLIUM_OK or skip_folium):
            reason = "folium not installed" if not _FOLIUM_OK else "--no-folium"
            print(f"{prefix} ... ⊘ skipped ({reason})"); skipped += 1; continue
        if engine == "pydeck" and (not _PYDECK_OK or skip_pydeck):
            reason = "pydeck not installed" if not _PYDECK_OK else "--no-pydeck"
            print(f"{prefix} ... ⊘ skipped ({reason})"); skipped += 1; continue

        print(f"{prefix} ...", end=" ", flush=True)
        try:
            result = fn(df)
            if engine == "plotly":
                ext      = fmt if fmt in ("html","png","svg","pdf") else "html"
                filepath = os.path.join(output_dir, f"{key}.{ext}")
                if ext == "html":
                    result.write_html(filepath, include_plotlyjs="cdn", full_html=True)
                else:
                    result.write_image(filepath, width=width, height=height)
            elif engine == "folium":
                filepath = os.path.join(output_dir, f"{key}.html")
                result.save(filepath)
            elif engine == "pydeck":
                filepath = os.path.join(output_dir, f"{key}.html")
                result.to_html(filepath)
            print(f"✓  →  {filepath}"); done += 1
        except Exception as exc:
            print(f"✗  ERROR: {exc}"); failed += 1

    print(f"\n{'='*62}\n  ✓ Saved: {done}   ⊘ Skipped: {skipped}   ✗ Failed: {failed}")
    print(f"  Output: {os.path.abspath(output_dir)}\n{'='*62}\n")


def print_all_geo_charts(
    df: pd.DataFrame,
    output_dir: str = "restaurant_iq_geo_charts",
    fmt: str = "html",
    show: bool = False,
    skip_folium: bool = False,
    skip_pydeck: bool = False,
) -> None:
    print_geo_summary(df)
    save_all_geo_charts(df, output_dir=output_dir, fmt=fmt,
                        skip_folium=skip_folium, skip_pydeck=skip_pydeck)
    if show:
        for _, (key, label, fn, engine) in enumerate(_GEO_CHART_REGISTRY):
            try:
                if engine == "plotly" and _PLOTLY_OK:
                    fn(df).show()
            except Exception as exc:
                print(f"  [show error] {label}: {exc}")


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RestaurantIQ Geo Engine – generate all geospatial charts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--csv", "-c", default=None)
    parser.add_argument("--output-dir", "-o", default="restaurant_iq_geo_charts")
    parser.add_argument("--format", "-f", choices=["html","png","svg","pdf"],
                        default="html", dest="fmt")
    parser.add_argument("--show", "-s", action="store_true")
    parser.add_argument("--no-folium", action="store_true")
    parser.add_argument("--no-pydeck", action="store_true")
    parser.add_argument("--sample-size", type=int, default=800)
    args = parser.parse_args()

    if args.csv:
        if not os.path.isfile(args.csv):
            print(f"\n[ERROR] File not found: {args.csv}"); sys.exit(1)
        df = pd.read_csv(args.csv)
    else:
        print(f"\nNo CSV supplied – using built-in sample data ({args.sample_size} rows).")
        df = _make_sample_df(n=args.sample_size)

    df = _ensure_derived_columns(df)
    print_all_geo_charts(df, output_dir=args.output_dir, fmt=args.fmt,
                         show=args.show, skip_folium=args.no_folium,
                         skip_pydeck=args.no_pydeck)


if __name__ == "__main__":
    main()
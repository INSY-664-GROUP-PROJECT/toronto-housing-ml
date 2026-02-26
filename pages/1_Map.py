from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pydeck as pdk
import requests
import streamlit as st
from sklearn.neighbors import NearestNeighbors

from utils import describe_cluster, load_grid_features

st.set_page_config(page_title="Street Cluster Map", layout="wide")
st.title("Street Cluster Map")

ROOT = Path(__file__).resolve().parents[1]
STREETS_GEOJSON = ROOT / "data" / "toronto_streets.geojson"
SUBWAY_CSV = ROOT / "data" / "toronto_subway_stations.csv"
RESTAURANTS_CSV = ROOT / "data" / "trt_rest.csv"
CRIME_CSV = ROOT / "data" / "MCI_2014_to_2019.csv"

SUBWAY_ICON_URL = (
    "data:image/svg+xml;utf8,"
    "<svg xmlns='http://www.w3.org/2000/svg' width='64' height='64' viewBox='0 0 64 64'>"
    "<rect x='10' y='6' width='44' height='42' rx='10' fill='%231f2937'/>"
    "<rect x='16' y='12' width='32' height='18' rx='4' fill='white'/>"
    "<circle cx='22' cy='38' r='4' fill='white'/>"
    "<circle cx='42' cy='38' r='4' fill='white'/>"
    "<rect x='28' y='48' width='8' height='8' fill='%231f2937'/>"
    "</svg>"
)
RESTAURANT_ICON_URL = (
    "data:image/svg+xml;utf8,"
    "<svg xmlns='http://www.w3.org/2000/svg' width='64' height='64' viewBox='0 0 64 64'>"
    "<circle cx='32' cy='32' r='26' fill='white' stroke='%231f2937' stroke-width='4'/>"
    "<path d='M24 18 v18 M28 18 v18 M32 18 v18 M24 30 h8' stroke='%231f2937' stroke-width='3' fill='none' stroke-linecap='round'/>"
    "<path d='M40 18 c4 4 4 10 0 14 v8' stroke='%231f2937' stroke-width='3' fill='none' stroke-linecap='round'/>"
    "</svg>"
)

NOISE_COLOR = [150, 150, 150, 220]
PALETTE = [
    [0, 107, 164, 220],
    [255, 128, 14, 220],
    [171, 171, 171, 220],
    [89, 89, 89, 220],
    [95, 158, 209, 220],
    [200, 82, 0, 220],
    [137, 137, 137, 220],
    [163, 200, 236, 220],
    [255, 188, 121, 220],
    [207, 207, 207, 220],
]
RESTAURANT_Q_COLORS = {
    1: [49, 130, 189, 220],
    2: [158, 202, 225, 220],
    3: [253, 174, 107, 220],
    4: [227, 74, 51, 220],
}


def _zoom_from_bounds(min_lon: float, min_lat: float, max_lon: float, max_lat: float) -> float:
    span = max(abs(max_lon - min_lon), abs(max_lat - min_lat))
    if span < 0.03:
        return 13.0
    if span < 0.08:
        return 12.0
    if span < 0.2:
        return 11.0
    if span < 0.4:
        return 10.0
    return 9.0


def _cluster_color_map(cluster_ids: list[int]) -> dict[int, list[int]]:
    color_map: dict[int, list[int]] = {}
    non_noise = [cid for cid in cluster_ids if cid != -1]
    for i, cid in enumerate(sorted(non_noise)):
        color_map[cid] = PALETTE[i % len(PALETTE)]
    if -1 in cluster_ids:
        color_map[-1] = NOISE_COLOR
    return color_map


@st.cache_data
def _grid_points_from_features() -> gpd.GeoDataFrame:
    df = load_grid_features().copy()
    if "cluster_label" not in df.columns:
        raise ValueError("Missing required column: cluster_label")

    if "latitude" in df.columns and "longitude" in df.columns:
        lat_col, lon_col = "latitude", "longitude"
    elif "centroid_lat" in df.columns and "centroid_lon" in df.columns:
        lat_col, lon_col = "centroid_lat", "centroid_lon"
    else:
        raise ValueError("Missing coordinate columns: need (latitude, longitude) or (centroid_lat, centroid_lon)")

    df[lat_col] = np.asarray(df[lat_col], dtype=float)
    df[lon_col] = np.asarray(df[lon_col], dtype=float)
    df = df.dropna(subset=[lat_col, lon_col, "cluster_label"]).copy()
    if df.empty:
        raise ValueError("No valid grid point rows found in grid_features.parquet")

    points = df[[lat_col, lon_col, "cluster_label"]].copy()
    points.columns = ["latitude", "longitude", "cluster_label"]
    points["cluster_label"] = points["cluster_label"].astype(int)
    return points


@st.cache_resource
def _load_street_edges() -> gpd.GeoDataFrame:
    if not STREETS_GEOJSON.exists():
        raise FileNotFoundError(
            f"Street network file not found: {STREETS_GEOJSON}. "
            "Precompute it with scripts/build_street_network.py."
        )
    edges = gpd.read_file(STREETS_GEOJSON)
    edges = edges.reset_index(drop=True)
    edges = edges.loc[edges.geometry.notna()].copy()
    if edges.empty:
        raise ValueError("No street edges found in GeoJSON.")

    edges["geometry"] = edges.geometry.simplify(0.00003, preserve_topology=True)
    edges = edges.loc[edges.geometry.notna()].copy()
    return edges[["geometry"]]


@st.cache_resource
def _fit_grid_nn() -> tuple[NearestNeighbors, np.ndarray]:
    grid_points = _grid_points_from_features()
    coords = np.radians(grid_points[["latitude", "longitude"]].to_numpy(dtype=float))
    labels = grid_points["cluster_label"].to_numpy(dtype=int)
    nn = NearestNeighbors(n_neighbors=1, metric="haversine", algorithm="ball_tree")
    nn.fit(coords)
    return nn, labels


@st.cache_data
def _clustered_street_records() -> tuple[list[dict[str, Any]], float, float, float, float]:
    edges = _load_street_edges()
    nn, labels = _fit_grid_nn()

    midpoint = edges.geometry.interpolate(0.5, normalized=True)
    edge_lat = midpoint.y.to_numpy(dtype=float)
    edge_lon = midpoint.x.to_numpy(dtype=float)

    _, idx = nn.kneighbors(np.radians(np.column_stack([edge_lat, edge_lon])))
    assigned_labels = labels[idx[:, 0]]

    cluster_ids = sorted(set(int(c) for c in assigned_labels.tolist()))
    color_map = _cluster_color_map(cluster_ids)

    records: list[dict[str, Any]] = []
    for geom, cid, mlat, mlon in zip(edges.geometry, assigned_labels, edge_lat, edge_lon):
        color = color_map.get(int(cid), NOISE_COLOR)
        if geom.geom_type == "LineString":
            coords = [[float(x), float(y)] for x, y in geom.coords]
            if len(coords) >= 2:
                records.append(
                    {
                        "path": coords,
                        "cluster_label": int(cid),
                        "color": color,
                        "midpoint_lat": float(mlat),
                        "midpoint_lon": float(mlon),
                    }
                )
        elif geom.geom_type == "MultiLineString":
            for line in geom.geoms:
                coords = [[float(x), float(y)] for x, y in line.coords]
                if len(coords) >= 2:
                    records.append(
                        {
                            "path": coords,
                            "cluster_label": int(cid),
                            "color": color,
                            "midpoint_lat": float(mlat),
                            "midpoint_lon": float(mlon),
                        }
                    )

    min_lon, min_lat, max_lon, max_lat = edges.total_bounds
    return records, float(min_lon), float(min_lat), float(max_lon), float(max_lat)


@st.cache_data
def _load_subway_points() -> list[dict[str, Any]]:
    if not SUBWAY_CSV.exists():
        return []
    df = pd.read_csv(SUBWAY_CSV)
    if not {"latitude", "longitude"}.issubset(df.columns):
        return []
    df = df.dropna(subset=["latitude", "longitude"]).copy()
    if "name" not in df.columns:
        df["name"] = "Subway"
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"])
    df["icon_data"] = [
        {"url": SUBWAY_ICON_URL, "width": 72, "height": 72, "anchorY": 72}
        for _ in range(len(df))
    ]
    return df[["name", "latitude", "longitude", "icon_data"]].to_dict("records")


@st.cache_data
def _load_restaurant_points(max_points: int = 15000) -> list[dict[str, Any]]:
    if not RESTAURANTS_CSV.exists():
        return []
    df = pd.read_csv(RESTAURANTS_CSV)
    lat_col = "Restaurant Latitude"
    lon_col = "Restaurant Longitude"
    if lat_col not in df.columns or lon_col not in df.columns:
        return []

    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
    df = df.dropna(subset=[lat_col, lon_col]).copy()
    if df.empty:
        return []

    price_col = "Restaurant Price Range"
    if price_col in df.columns:
        score = (
            df[price_col]
            .astype(str)
            .str.replace(r"[^$]", "", regex=True)
            .str.len()
            .replace(0, np.nan)
        )
        if score.notna().sum() >= 4:
            q = pd.qcut(score.rank(method="first"), 4, labels=[1, 2, 3, 4])
            df["quartile"] = pd.to_numeric(q, errors="coerce")
        else:
            df["quartile"] = 2
    else:
        df["quartile"] = 2

    # Ensure NaN quartiles are handled consistently.
    df["quartile"] = pd.to_numeric(df["quartile"], errors="coerce").fillna(2).astype(int)
    df["quartile"] = df["quartile"].clip(lower=1, upper=4)
    df["color"] = df["quartile"].map(RESTAURANT_Q_COLORS).apply(
        lambda x: x if isinstance(x, list) else [140, 140, 140, 220]
    )
    df["name"] = df.get("Restaurant Name", pd.Series(["Restaurant"] * len(df))).astype(str)
    df["icon_data"] = [
        {"url": RESTAURANT_ICON_URL, "width": 72, "height": 72, "anchorY": 72}
        for _ in range(len(df))
    ]
    df = df.rename(columns={lat_col: "latitude", lon_col: "longitude"})
    if len(df) > max_points:
        df = df.sample(max_points, random_state=42)
    return df[["name", "latitude", "longitude", "quartile", "color", "icon_data"]].to_dict("records")


@st.cache_data
def _load_crime_points(max_points: int = 200000) -> list[dict[str, float]]:
    if not CRIME_CSV.exists():
        return []
    df = pd.read_csv(CRIME_CSV, usecols=["Lat", "Long"])
    df["Lat"] = pd.to_numeric(df["Lat"], errors="coerce")
    df["Long"] = pd.to_numeric(df["Long"], errors="coerce")
    df = df.dropna(subset=["Lat", "Long"]).copy()
    if df.empty:
        return []
    if len(df) > max_points:
        df = df.sample(max_points, random_state=42)
    return [{"lat": float(r.Lat), "lon": float(r.Long)} for r in df.itertuples(index=False)]


@st.cache_data(show_spinner=False)
def _geocode_toronto_address(address: str) -> tuple[float, float, str] | None:
    query = f"{address}, Toronto, Ontario, Canada"
    resp = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params={"q": query, "format": "jsonv2", "limit": 1},
        headers={"User-Agent": "toronto-housing-cluster-app/1.0"},
        timeout=20,
    )
    resp.raise_for_status()
    payload = resp.json()
    if not payload:
        return None
    top = payload[0]
    return float(top["lat"]), float(top["lon"]), str(top.get("display_name", query))


def _predict_cluster_at_point(lat: float, lon: float) -> int:
    nn, labels = _fit_grid_nn()
    _, idx = nn.kneighbors(np.radians(np.array([[lat, lon]], dtype=float)))
    return int(labels[idx[0, 0]])


def _extract_picked_object(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    selection = payload.get("selection", payload)
    if not isinstance(selection, dict):
        return None
    objects = selection.get("objects", {})
    if isinstance(objects, dict):
        for _, rows in objects.items():
            if isinstance(rows, list) and rows and isinstance(rows[0], dict):
                return rows[0]
    if isinstance(objects, list) and objects and isinstance(objects[0], dict):
        return objects[0]
    return None


try:
    street_records, min_lon, min_lat, max_lon, max_lat = _clustered_street_records()
except Exception as exc:
    st.error(f"Failed to build street cluster map: {exc}")
    st.stop()

if not street_records:
    st.warning("No street segments available to render.")
    st.stop()

show_subway = st.sidebar.checkbox("Show Subway", value=True)
show_restaurants = st.sidebar.checkbox("Show Restaurants", value=False)
show_crime_hex = st.sidebar.checkbox("Show Crime Hex Intensity", value=False)

lookup_mode = st.sidebar.radio("Lookup Mode", ["Street/Address", "Click on map"])
resolved_lat: float | None = None
resolved_lon: float | None = None
resolved_label = ""
lookup_cluster: int | None = None

if "selected_map_point" not in st.session_state:
    st.session_state["selected_map_point"] = None

if lookup_mode == "Street/Address":
    address = st.sidebar.text_input("Street / Address", value="")
    if st.sidebar.button("Classify Address") and address.strip():
        try:
            geo = _geocode_toronto_address(address.strip())
            if geo is None:
                st.sidebar.warning("Address not found.")
            else:
                resolved_lat, resolved_lon, resolved_label = geo
                lookup_cluster = _predict_cluster_at_point(resolved_lat, resolved_lon)
        except Exception as exc:
            st.sidebar.error(f"Address lookup failed: {exc}")
else:
    st.sidebar.caption("Click a street segment on the map to choose location.")
    map_state = st.session_state.get("street_cluster_map")
    picked_obj = _extract_picked_object(map_state)
    if picked_obj is not None and "midpoint_lat" in picked_obj and "midpoint_lon" in picked_obj:
        st.session_state["selected_map_point"] = {
            "lat": float(picked_obj["midpoint_lat"]),
            "lon": float(picked_obj["midpoint_lon"]),
            "cluster": int(
                picked_obj.get(
                    "cluster_label",
                    _predict_cluster_at_point(float(picked_obj["midpoint_lat"]), float(picked_obj["midpoint_lon"])),
                )
            ),
        }
    selected = st.session_state.get("selected_map_point")
    if selected is not None:
        resolved_lat = float(selected["lat"])
        resolved_lon = float(selected["lon"])
        resolved_label = f"Clicked location ({resolved_lat:.6f}, {resolved_lon:.6f})"
        lookup_cluster = int(selected["cluster"])

center_lat = (min_lat + max_lat) / 2.0
center_lon = (min_lon + max_lon) / 2.0
zoom = _zoom_from_bounds(min_lon, min_lat, max_lon, max_lat)

street_layer = pdk.Layer(
    "PathLayer",
    data=street_records,
    get_path="path",
    get_color="color",
    get_width=2,
    width_min_pixels=1,
    width_max_pixels=4,
    pickable=True,
)

layers: list[Any] = [street_layer]

if show_subway:
    subway_points = _load_subway_points()
    if subway_points:
        layers.append(
            pdk.Layer(
                "IconLayer",
                data=subway_points,
                get_position="[longitude, latitude]",
                get_icon="icon_data",
                get_size=24,
                size_units="meters",
                billboard=True,
                pickable=False,
            )
        )

if show_restaurants:
    restaurant_points = _load_restaurant_points()
    if restaurant_points:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=restaurant_points,
                get_position="[longitude, latitude]",
                get_fill_color="color",
                get_radius=9,
                radius_units="meters",
                radius_min_pixels=0,
                radius_max_pixels=6,
                pickable=False,
            )
        )
        layers.append(
            pdk.Layer(
                "IconLayer",
                data=restaurant_points,
                get_position="[longitude, latitude]",
                get_icon="icon_data",
                get_size=18,
                size_units="meters",
                billboard=True,
                pickable=False,
            )
        )

if show_crime_hex:
    crime_points = _load_crime_points()
    if crime_points:
        layers.append(
            pdk.Layer(
                "HexagonLayer",
                data=crime_points,
                get_position="[lon, lat]",
                radius=180,
                coverage=0.85,
                elevation_scale=0,
                extruded=False,
                color_range=[
                    [255, 255, 204],
                    [255, 237, 160],
                    [254, 217, 118],
                    [254, 178, 76],
                    [253, 141, 60],
                    [240, 59, 32],
                ],
                pickable=False,
            )
        )

if resolved_lat is not None and resolved_lon is not None:
    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=[{"lat": resolved_lat, "lon": resolved_lon}],
            get_position="[lon, lat]",
            get_fill_color=[20, 20, 20, 220],
            get_line_color=[255, 255, 255, 255],
            get_radius=80,
            radius_units="meters",
            stroked=True,
            line_width_min_pixels=2,
            pickable=False,
        )
    )
    center_lat = resolved_lat
    center_lon = resolved_lon
    zoom = max(zoom, 13.0)

view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=zoom, pitch=0, bearing=0)

tooltip = {
    "html": "<b>Cluster ID:</b> {cluster_label}<br/><b>Click to select location</b>",
    "style": {"backgroundColor": "#111827", "color": "white"},
}

st.pydeck_chart(
    pdk.Deck(
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        initial_view_state=view_state,
        layers=layers,
        tooltip=tooltip,
    ),
    use_container_width=True,
    on_select="rerun",
    selection_mode="single-object",
    key="street_cluster_map",
)

st.caption(f"Rendered {len(street_records):,} street segments colored by nearest grid-point cluster")

if show_restaurants:
    st.caption("Restaurant quartiles: Q1 (blue) to Q4 (red)")

if lookup_cluster is not None:
    st.subheader("Lookup Result")
    st.write(f"Location: {resolved_label}")
    st.write(f"Expected Neighborhood Cluster: {lookup_cluster}")
    if lookup_cluster == -1:
        st.write(
            "Cluster characteristics: Noise/outlier area. This location does not strongly belong to a stable cluster profile."
        )
    else:
        st.write(f"Cluster characteristics: {describe_cluster(lookup_cluster)}")

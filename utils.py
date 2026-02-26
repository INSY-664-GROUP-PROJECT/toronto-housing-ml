from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st

try:
    import geopandas as gpd  # type: ignore
except Exception:
    gpd = None

try:
    from hdbscan import approximate_predict  # type: ignore
except Exception:
    approximate_predict = None

FEATURE_COLUMNS = [
    "avg_rent",
    "crime_density",
    "restaurant_density",
    "transit_score",
]

DISPLAY_COLUMNS = FEATURE_COLUMNS + ["cluster_label", "cluster_probability"]


def get_project_root() -> Path:
    return Path(__file__).resolve().parent


def artifact_path(name: str) -> Path:
    return get_project_root() / name


@st.cache_resource
def load_pipeline_artifacts() -> tuple[Any, Any, Any | None]:
    model = joblib.load(artifact_path("hdbscan_model.pkl"))
    scaler = joblib.load(artifact_path("scaler.pkl"))
    pca_path = artifact_path("pca.pkl")
    pca = joblib.load(pca_path) if pca_path.exists() else None
    return model, scaler, pca


@st.cache_data
def load_grid_features() -> pd.DataFrame:
    parquet_path = artifact_path("grid_features.parquet")
    if gpd is not None:
        try:
            gdf = gpd.read_parquet(parquet_path)
            frame = pd.DataFrame(gdf)
        except Exception:
            frame = pd.read_pickle(parquet_path)
    else:
        frame = pd.read_pickle(parquet_path)

    for col in FEATURE_COLUMNS:
        if col not in frame.columns:
            raise ValueError(f"Missing required feature column: {col}")

    if "cluster_label" not in frame.columns:
        raise ValueError("Missing required column: cluster_label")

    if "cluster_probability" not in frame.columns:
        frame["cluster_probability"] = np.nan

    if "geometry" not in frame.columns:
        raise ValueError("Missing required geometry column")

    # Ensure map-centering coordinates exist and are numeric for Streamlit/PyDeck.
    if "centroid_lat" not in frame.columns:
        if "latitude" in frame.columns:
            frame["centroid_lat"] = frame["latitude"]
        else:
            raise ValueError("Missing required column: centroid_lat")
    if "centroid_lon" not in frame.columns:
        if "longitude" in frame.columns:
            frame["centroid_lon"] = frame["longitude"]
        else:
            raise ValueError("Missing required column: centroid_lon")

    frame["centroid_lat"] = pd.to_numeric(frame["centroid_lat"], errors="coerce")
    frame["centroid_lon"] = pd.to_numeric(frame["centroid_lon"], errors="coerce")
    frame = frame.dropna(subset=["centroid_lat", "centroid_lon"]).copy()
    if frame.empty:
        raise ValueError("No valid centroid_lat/centroid_lon rows after numeric coercion.")

    return frame


@st.cache_data
def get_cluster_options() -> list[int]:
    df = load_grid_features()
    clusters = pd.Series(df["cluster_label"]).dropna().unique().tolist()
    return sorted(int(x) for x in clusters)


@st.cache_data
def get_feature_ranges() -> dict[str, tuple[float, float, float]]:
    df = load_grid_features()
    ranges: dict[str, tuple[float, float, float]] = {}
    for col in FEATURE_COLUMNS:
        low = float(np.nanpercentile(df[col], 5))
        high = float(np.nanpercentile(df[col], 95))
        full_low = float(np.nanmin(df[col]))
        full_high = float(np.nanmax(df[col]))
        default = float(np.nanmedian(df[col]))

        if low == high:
            low, high = full_low, full_high
        if low == high:
            high = low + 1.0

        ranges[col] = (low, high, default)
    return ranges


def filter_grid(
    gdf: pd.DataFrame,
    selected_clusters: list[int],
    rent_range: tuple[float, float],
    crime_range: tuple[float, float],
) -> pd.DataFrame:
    mask = gdf["cluster_label"].isin(selected_clusters)
    mask &= gdf["avg_rent"].between(rent_range[0], rent_range[1])
    mask &= gdf["crime_density"].between(crime_range[0], crime_range[1])
    return gdf.loc[mask].copy()


def to_geojson_records(gdf: pd.DataFrame) -> list[dict[str, Any]]:
    frame = gdf[DISPLAY_COLUMNS + ["geometry"]].copy()
    frame["cluster_label"] = frame["cluster_label"].astype("Int64")
    frame = frame.replace({np.nan: None})

    def _json_safe(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, dict):
            return {k: _json_safe(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_json_safe(v) for v in value]
        if pd.isna(value):
            return None
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        return value

    features: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        properties = {k: _json_safe(row[k]) for k in DISPLAY_COLUMNS}
        geom = row["geometry"]
        if hasattr(geom, "__geo_interface__"):
            geom = geom.__geo_interface__
        if isinstance(geom, str):
            geom = json.loads(geom)
        geom = _json_safe(geom)
        features.append({"type": "Feature", "geometry": geom, "properties": properties})
    return features


def map_center(gdf: pd.DataFrame) -> tuple[float, float, float]:
    if gdf.empty:
        return 43.65107, -79.347015, 10.0

    lat = float(gdf["centroid_lat"].mean())
    lon = float(gdf["centroid_lon"].mean())

    width = float(abs(gdf["centroid_lon"].max() - gdf["centroid_lon"].min()))
    height = float(abs(gdf["centroid_lat"].max() - gdf["centroid_lat"].min()))
    span = max(width, height)

    if span < 0.03:
        zoom = 12.5
    elif span < 0.08:
        zoom = 11.5
    elif span < 0.2:
        zoom = 10.5
    else:
        zoom = 9.5

    return lat, lon, zoom


@st.cache_data
def cluster_centroids() -> pd.DataFrame:
    df = load_grid_features()
    centroids = (
        df.groupby("cluster_label", dropna=True)[FEATURE_COLUMNS]
        .mean(numeric_only=True)
        .reset_index()
    )
    centroids["cluster_label"] = centroids["cluster_label"].astype(int)
    return centroids


def describe_cluster(cluster_id: int) -> str:
    centroids = cluster_centroids()
    row = centroids.loc[centroids["cluster_label"] == cluster_id]
    if row.empty:
        return "Cluster profile unavailable for this ID."

    values = row.iloc[0]
    all_centroids = centroids[FEATURE_COLUMNS]

    def level(col: str, higher_is_better: bool) -> str:
        q1 = float(all_centroids[col].quantile(0.33))
        q2 = float(all_centroids[col].quantile(0.66))
        v = float(values[col])
        if v <= q1:
            return "low" if higher_is_better else "high"
        if v >= q2:
            return "high" if higher_is_better else "low"
        return "moderate"

    rent = level("avg_rent", higher_is_better=False)
    crime = level("crime_density", higher_is_better=False)
    restaurants = level("restaurant_density", higher_is_better=True)
    transit = level("transit_score", higher_is_better=True)

    return (
        f"Cluster {cluster_id} appears to be a {rent}-rent, {crime}-crime area "
        f"with {restaurants} restaurant access and {transit} transit coverage."
    )


def _apply_scaler(scaler: Any, x: np.ndarray) -> np.ndarray:
    if hasattr(scaler, "transform"):
        return scaler.transform(x)
    if isinstance(scaler, dict) and scaler.get("scaler_type") == "standard":
        mean = np.asarray(scaler["mean"], dtype=float)
        scale = np.asarray(scaler["scale"], dtype=float)
        return (x - mean) / scale
    raise ValueError("Unsupported scaler artifact format.")


def _build_full_feature_vector(feature_values: dict[str, float], scaler: Any) -> np.ndarray:
    # Default: use 4-feature vector for simple scaler artifacts.
    if not hasattr(scaler, "mean_") and not (
        isinstance(scaler, dict) and scaler.get("scaler_type") == "standard"
    ):
        return np.array([[feature_values[col] for col in FEATURE_COLUMNS]], dtype=float)

    if hasattr(scaler, "mean_"):
        mean = np.asarray(scaler.mean_, dtype=float)
        n_features = int(getattr(scaler, "n_features_in_", len(mean)))
        vec = mean.copy()
        names = list(getattr(scaler, "feature_names_in_", []))
    else:
        mean = np.asarray(scaler["mean"], dtype=float)
        vec = mean.copy()
        names = list(scaler.get("feature_order", []))
        n_features = len(mean)

    if len(vec) != n_features:
        vec = np.resize(vec, n_features)

    # Map Streamlit inputs to likely training feature names.
    alias_map: dict[str, list[str]] = {
        "avg_rent": ["avg_rent", "mean_rent"],
        "crime_density": ["crime_density", "crime_per_km2"],
        "restaurant_density": ["restaurant_density", "restaurant_per_km2"],
        "transit_score": ["transit_score", "subway_per_km2"],
    }

    if names:
        idx_by_name = {str(name): i for i, name in enumerate(names)}
        for ui_feature, aliases in alias_map.items():
            for alias in aliases:
                if alias in idx_by_name:
                    vec[idx_by_name[alias]] = float(feature_values[ui_feature])
                    break
    else:
        # If names are unavailable and scaler expects exactly 4, keep original order.
        if n_features == 4:
            vec = np.array([feature_values[col] for col in FEATURE_COLUMNS], dtype=float)

    return vec.reshape(1, -1)


@st.cache_data
def _load_selected_feature_names() -> list[str]:
    candidates = [
        artifact_path("selected_features.json"),
        artifact_path("outputs/feature_selection/selected_features.json"),
    ]
    for path in candidates:
        if path.exists():
            try:
                payload = json.loads(path.read_text())
                names = payload.get("selected_features", [])
                if isinstance(names, list) and names:
                    return [str(x) for x in names]
            except Exception:
                continue
    return []


def _scaler_feature_names(scaler: Any) -> list[str]:
    if hasattr(scaler, "feature_names_in_"):
        return [str(x) for x in scaler.feature_names_in_]
    if isinstance(scaler, dict):
        names = scaler.get("feature_order", [])
        if isinstance(names, list):
            return [str(x) for x in names]
    return []


def _align_for_pca(transformed: np.ndarray, scaler: Any, pca: Any) -> np.ndarray:
    expected = int(getattr(pca, "n_features_in_", transformed.shape[1]))
    if transformed.shape[1] == expected:
        return transformed

    scaler_names = _scaler_feature_names(scaler)
    selected_names = _load_selected_feature_names()
    if not selected_names:
        raise ValueError(
            "Notebook selected features not found. "
            "Expected selected_features.json from notebook export."
        )
    if not scaler_names:
        raise ValueError(
            "Scaler feature names are unavailable; cannot align by notebook-selected features."
        )

    idx_by_name = {name: i for i, name in enumerate(scaler_names)}
    missing = [name for name in selected_names if name not in idx_by_name]
    if missing:
        sample = ", ".join(missing[:5])
        raise ValueError(
            "Notebook-selected features are missing from scaler feature names: "
            f"{sample}"
        )

    selected_idx = [idx_by_name[name] for name in selected_names]
    if len(selected_idx) != expected:
        raise ValueError(
            "Notebook-selected feature count does not match PCA input width: "
            f"{len(selected_idx)} vs {expected}"
        )
    return transformed[:, selected_idx]


def _apply_model_fallback(model: Any, transformed: np.ndarray) -> tuple[int, float]:
    if isinstance(model, dict) and model.get("model_type") == "quantile_cluster":
        weights = np.asarray(model["weights"], dtype=float)
        edges = np.asarray(model["score_edges"], dtype=float)
        centroids = np.asarray(model["centroids"], dtype=float)
        score = float(transformed[0] @ weights)
        label = int(np.searchsorted(edges, score, side="right") - 1)
        label = int(np.clip(label, 0, len(centroids) - 1))
        dist = float(np.linalg.norm(transformed[0] - centroids[label]))
        prob = float(1.0 / (1.0 + dist))
        return label, prob

    if hasattr(model, "cluster_centers_") and hasattr(model, "predict"):
        labels = model.predict(transformed)
        centers = model.cluster_centers_
        dist = float(np.linalg.norm(transformed[0] - centers[int(labels[0])]))
        strength = 1.0 / (1.0 + dist)
        return int(labels[0]), float(strength)

    if hasattr(model, "predict"):
        labels = model.predict(transformed)
        return int(labels[0]), math.nan

    raise ValueError("Loaded model does not support prediction.")


def _fallback_predict_from_grid(feature_values: dict[str, float]) -> tuple[int, float]:
    centroids = cluster_centroids()
    if centroids.empty:
        raise ValueError("No cluster centroid data available for fallback prediction.")

    x = np.array([feature_values[col] for col in FEATURE_COLUMNS], dtype=float)
    c = centroids[FEATURE_COLUMNS].to_numpy(dtype=float)
    dists = np.linalg.norm(c - x, axis=1)
    idx = int(np.argmin(dists))
    label = int(centroids.iloc[idx]["cluster_label"])
    prob = float(1.0 / (1.0 + dists[idx]))
    return label, prob


def predict_cluster(feature_values: dict[str, float]) -> tuple[int, float]:
    model, scaler, pca = load_pipeline_artifacts()

    vector = _build_full_feature_vector(feature_values, scaler)
    transformed = _apply_scaler(scaler, vector)
    if pca is not None and hasattr(pca, "transform"):
        transformed = _align_for_pca(transformed, scaler, pca)
        transformed = pca.transform(transformed)

    if approximate_predict is not None:
        try:
            labels, strengths = approximate_predict(model, transformed)
            return int(labels[0]), float(strengths[0])
        except Exception:
            pass

    try:
        return _apply_model_fallback(model, transformed)
    except Exception:
        return _fallback_predict_from_grid(feature_values)

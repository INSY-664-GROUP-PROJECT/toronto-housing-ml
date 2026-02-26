from pathlib import Path

import streamlit as st

REQUIRED_ROOT_FILES = [
    "hdbscan_model.pkl",
    "scaler.pkl",
    "grid_features.parquet",
]
OPTIONAL_ROOT_FILES = ["pca.pkl"]


def _artifact_status() -> tuple[list[str], list[str]]:
    root = Path(__file__).resolve().parent
    missing_required = [name for name in REQUIRED_ROOT_FILES if not (root / name).exists()]
    missing_optional = [name for name in OPTIONAL_ROOT_FILES if not (root / name).exists()]
    return missing_required, missing_optional


st.set_page_config(page_title="Toronto Housing Clusters", layout="wide")

st.title("Toronto Housing Clustering")
st.caption("HDBSCAN-powered street-level neighbourhood map")

missing_required_files, missing_optional_files = _artifact_status()

if missing_required_files:
    st.error(
        "Missing required artifact(s) in project root: " + ", ".join(missing_required_files)
    )
    st.stop()

if missing_optional_files:
    st.info("Optional artifact not found: pca.pkl (pipeline will run without PCA)")

st.markdown(
    "Use the left sidebar to open:\n"
    "- **Street Cluster Map** for geospatial cluster exploration and street/address lookup"
)

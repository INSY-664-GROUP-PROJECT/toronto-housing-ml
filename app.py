import streamlit as st

from project_paths import FEATURES_DIR, MODELS_DIR

REQUIRED_MODEL_FILES = ["hdbscan_model.pkl", "scaler.pkl"]
REQUIRED_FEATURE_FILES = ["grid_features.parquet"]
OPTIONAL_MODEL_FILES = ["pca.pkl"]


def _artifact_status() -> tuple[list[str], list[str]]:
    missing_required = [name for name in REQUIRED_MODEL_FILES if not (MODELS_DIR / name).exists()]
    missing_required += [
        name for name in REQUIRED_FEATURE_FILES if not (FEATURES_DIR / name).exists()
    ]
    missing_optional = [name for name in OPTIONAL_MODEL_FILES if not (MODELS_DIR / name).exists()]
    return missing_required, missing_optional


st.set_page_config(page_title="Toronto Housing Clusters", layout="wide")

missing_required_files, missing_optional_files = _artifact_status()

if missing_required_files:
    st.error(
        "Missing required artifact(s) under artifacts/: " + ", ".join(missing_required_files)
    )
    st.stop()

if missing_optional_files:
    st.info("Optional artifact not found in artifacts/models: pca.pkl (pipeline will run without PCA)")

map_page = st.Page("pages/1_Map.py", title="Street Cluster Map")
navigation = st.navigation([map_page], position="hidden")
navigation.run()

# Toronto Housing ML

Data-driven neighborhood intelligence for **Toronto investors** and **individuals searching for rent**.

## Highlights
- Built for decision-making, not just visualization: the project helps users find the **best compromise** between:
  - rental price
  - transit access
  - crime exposure
  - restaurant access
- Uses spatial machine learning to segment Toronto into interpretable neighborhood profiles.
- Adds a causal analysis layer to explain **what most influences rent levels**.
- Includes an interactive map application for exploration and location lookup.

## Live App
The solution is deployed on Streamlit Cloud:

**https://toronto-housing-ml-n3qkcxaijesq5jeuufn4rw.streamlit.app/**

## Table of Contents
- [Project Goal](#project-goal)
- [Methodology](#methodology)
- [Clustering Model Selection (Including SKATER)](#clustering-model-selection-including-skater)
- [Methodology to Explain Rental Prices](#methodology-to-explain-rental-prices)
- [Main Outputs](#main-outputs)
- [Repository Structure](#repository-structure)
- [Setup and Run](#setup-and-run)
- [Data Sources](#data-sources)
- [Limitations](#limitations)
- [License](#license)

## Project Goal
Toronto rental decisions involve trade-offs. A cheap area may have lower transit access, while a transit-rich area may have higher rent or different safety profiles.

This project builds a city-wide analytical framework to support:
- **investors** looking for opportunity areas with balanced fundamentals
- **renters** looking for neighborhoods aligned with budget and lifestyle constraints

The final objective is to identify neighborhood clusters that are coherent, interpretable, and useful in real-world decision contexts.

## Methodology
The end-to-end workflow combines geospatial preprocessing, unsupervised learning, and interpretability:

1. Data integration and cleaning
- Merged rental listings from multiple sources.
- Cleaned geospatial coordinates and filtered to Toronto bounding constraints.
- Applied plausibility filtering (for example rent range limits) and winsorization for outlier control.
- Standardized restaurant price signals and handled missingness.

2. Spatial feature engineering
- Reprojected points to metric coordinates.
- Built an adaptive sparse grid (micro-geographic cells).
- Aggregated local signals per cell (rent, crime, restaurant density/value, subway access).

3. Clustering workflow
- Standardized features.
- Ran dimensionality diagnostics and model comparisons.
- Evaluated KMeans, DBSCAN, and HDBSCAN on cluster quality and robustness.
- Kept HDBSCAN as the final clustering model for production artifacts and app deployment.

4. Cluster interpretation
- Computed cluster-level feature profiles and relative importance.
- Produced targeted spatial diagnostics for transit, safety, and restaurant patterns.

## Clustering Model Selection (Including SKATER)
The final reproducible study keeps **HDBSCAN** as the production clustering method.

Model comparison in the baseline notebook indicates:
- KMeans: fixed number of clusters, no noise handling.
- DBSCAN: viable but more sensitive to parameter tuning and noisier in this context.
- HDBSCAN: best balance of automatic structure discovery and noise handling.

In addition, a **state-of-the-art geographically constrained clustering approach (SKATER)** was tested during exploratory work. It was **not retained** in the final study because results were weaker for this use case (lower practical interpretability and less useful segmentation for investor/renter trade-offs), so it is excluded from the final deployment pipeline.

## Methodology to Explain Rental Prices
Beyond clustering, the project includes a dedicated causal workflow to explain rent drivers:

1. Define treatment scenarios (high vs low exposure), such as:
- transit access
- crime intensity
- food/amenity access

2. Confounder screening
- For each treatment, select confounders using a sensitivity score based on standardized mean difference and correlation with rent.

3. Causal estimation
- Estimate ATE/CATE using:
  - S-learner (Linear Regression)
  - T-learner (XGBoost)
- Use XGBoost-based ATE as primary ranking, with LR as a robustness check.

4. Interpretation
- Separate drivers that **increase rent** from those with the **largest absolute impact** (which can be negative).
- In the notebook’s main conclusions, transit access is the strongest positive rent driver, while high crime shows the largest absolute (negative) effect.

## Main Outputs
- Interactive Streamlit map app (`app.py`, `pages/1_Map.py`)
- Model artifacts in `artifacts/models/` (`hdbscan_model.pkl`, `scaler.pkl`, optional `pca.pkl`)
- Feature artifact in `artifacts/features/grid_features.parquet`
- Visual diagnostics in `reports/figures/`
- Analysis notebooks:
  - `notebooks/toronto_housing_clustering_baseline.ipynb`
  - `notebooks/toronto_housing_causal_inference.ipynb`

## Repository Structure
```text
toronto-housing-ml/
├── app.py
├── pages/                     # Streamlit pages
├── utils.py
├── artifacts/
│   ├── models/
│   └── features/
├── data/
│   ├── README.md
│   └── *.csv
├── notebooks/
├── reports/figures/
├── scripts/
├── outputs/
├── requirements.txt
└── README.md
```

## Setup and Run
```bash
# 1) Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # macOS / Linux

# 2) Install dependencies
pip install -r requirements.txt

# 3) Ensure data files are available under data/
#    (see data/README.md for source details)

# 4) Run the Streamlit app
streamlit run app.py
```

Optional:
```bash
# Generate/update subway stations from OpenStreetMap
python scripts/subway_station_fetcher.py
```

## Data Sources
| File | Description | Source |
|------|-------------|--------|
| `MCI_2014_to_2019.csv` | Toronto Police major crime indicators (2014–2019) | [Kaggle](https://www.kaggle.com/datasets/kapastor/toronto-police-data-crime-rates-by-neighbourhood) |
| `Toronto_apartment_rentals_2018.csv` | Toronto apartment rentals | [Kaggle](https://www.kaggle.com/datasets/rajacsp/toronto-apartment-price) |
| `trt_rest.csv` | Toronto restaurants | [Kaggle](https://www.kaggle.com/datasets/kevinbi/toronto-restaurants) |
| `rentfaster.csv` | RentFaster Toronto listings | Included in project data bundle |
| `toronto_subway_stations.csv` | Toronto subway station locations | Generated by `scripts/subway_station_fetcher.py` |

## Limitations
- Most source data represents historical snapshots (not real-time market feeds).
- Causal estimates are observational and depend on measured confounders.
- Cluster labels summarize patterns, but they should complement domain judgment and local due diligence.

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE).

# Toronto Housing ML: Neighbourhood Clustering Analysis

## Overview

There's a lot more to Toronto housing prices than historical real estate and economic data may suggest. Neighbourhood, transit, crime, and restaurants are huge reasons people choose certain homes.

Our goal is to create an ML clustering model to highlight established as well as up-and-coming neighbourhoods by price and local offerings.

## Project Goals

This project aims to:
- Identify latent structural similarities across Toronto neighbourhoods using unsupervised machine learning
- Create interpretable neighbourhood clusters based on socio-economic, safety, and accessibility features
- Provide insights into established and emerging neighbourhood profiles in Toronto

## Hypotheses

**H1:** Toronto neighbourhoods exhibit latent structural similarities that can be identified using unsupervised clustering.

**H2:** Socio-economic (rent, restaurant prices), safety (crime), and accessibility (subway proximity) variables are sufficient to produce distinct and interpretable neighbourhood clusters.

**H3:** The resulting clusters correspond to meaningful urban profiles.

## Project Structure

```
toronto-housing-ml/
├── data/                  # Raw datasets (see data/README.md for sources)
│   └── README.md          # Download instructions for each dataset
├── notebooks/             # Jupyter notebooks for analysis
│   └── toronto_housing_clustering_baseline.ipynb
├── outputs/               # Generated CSVs and artifacts (not tracked)
│   └── eda/               # EDA-stage intermediate outputs
├── scripts/               # Standalone utility scripts
│   └── subway_station_fetcher.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/<your-org>/toronto-housing-ml.git
cd toronto-housing-ml

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download datasets (see data/README.md for links)
#    Place all CSV files in the data/ directory.

# 5. (Optional) Generate subway station data from OpenStreetMap
python scripts/subway_station_fetcher.py

# 6. Launch JupyterLab
jupyter lab
```

## Data Sources

| Dataset | Description | Source |
|---------|-------------|--------|
| `MCI_2014_to_2019.csv` | Major Crime Indicators (2014–2019) | [Kaggle](https://www.kaggle.com/datasets/kapastor/toronto-police-data-crime-rates-by-neighbourhood) |
| `Toronto_apartment_rentals_2018.csv` | Toronto apartment rentals (2018) | [Kaggle](https://www.kaggle.com/datasets/rajacsp/toronto-apartment-price) |
| `trt_rest.csv` | Toronto restaurant information | [Kaggle](https://www.kaggle.com/datasets/kevinbi/toronto-restaurants) |
| `rentfaster.csv` | RentFaster Toronto listings | *(included in project data bundle)* |
| `toronto_subway_stations.csv` | Subway station locations | Generated via `scripts/subway_station_fetcher.py` |

## Methodology

Using unsupervised clustering techniques, we analyze neighbourhoods across multiple dimensions:
- **Socio-economic factors**: Rental prices and restaurant pricing
- **Safety metrics**: Crime statistics
- **Accessibility**: Proximity to subway stations

These features are combined to create a comprehensive neighbourhood profile that goes beyond traditional real estate metrics.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

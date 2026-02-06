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

## Data Sources

The project utilizes the following datasets:

- **Crime Data**: Major Crime Indicators (MCI) from 2014-2019
  - Source: [Toronto Police Data - Crime Rates by Neighbourhood](https://www.kaggle.com/datasets/kapastor/toronto-police-data-crime-rates-by-neighbourhood)
  - File: `data/MCI_2014_to_2019.csv`

- **Rental Data**: Toronto apartment rentals (2018)
  - Source: [Toronto Apartment Price](https://www.kaggle.com/datasets/rajacsp/toronto-apartment-price)
  - File: `data/Toronto_apartment_rentals_2018.csv`

- **Restaurant Data**: Toronto restaurant information
  - Source: [Toronto Restaurants](https://www.kaggle.com/datasets/kevinbi/toronto-restaurants)
  - File: `data/trt_rest.csv`

- **Transit Data**: Toronto subway station locations
  - Source: Generated using `subway_station_fetcher.py` (fetches data from OpenStreetMap via Overpass API)
  - File: `data/toronto_subway_stations.csv`

## Methodology

Using unsupervised clustering techniques, we analyze neighbourhoods across multiple dimensions:
- **Socio-economic factors**: Rental prices and restaurant pricing
- **Safety metrics**: Crime statistics
- **Accessibility**: Proximity to subway stations

These features are combined to create a comprehensive neighbourhood profile that goes beyond traditional real estate metrics.

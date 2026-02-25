"""Fetch Toronto subway station coordinates from OpenStreetMap (Overpass API)."""

import csv
from pathlib import Path

import requests

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_FILE = PROJECT_ROOT / "data" / "toronto_subway_stations.csv"

QUERY = """
[out:json][timeout:25];
area["name"="Toronto"]["boundary"="administrative"]->.searchArea;
(
  node["railway"="station"]["station"="subway"](area.searchArea);
  way["railway"="station"]["station"="subway"](area.searchArea);
  relation["railway"="station"]["station"="subway"](area.searchArea);
);
out center tags;
"""


def fetch_stations():
    response = requests.post(OVERPASS_URL, data=QUERY)
    response.raise_for_status()
    data = response.json()

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "latitude", "longitude"])

        for el in data["elements"]:
            name = el.get("tags", {}).get("name", "Unknown")

            if "lat" in el and "lon" in el:
                lat, lon = el["lat"], el["lon"]
            else:
                lat, lon = el["center"]["lat"], el["center"]["lon"]

            writer.writerow([name, lat, lon])

    print(f"Saved {len(data['elements'])} stations to {OUTPUT_FILE}")


if __name__ == "__main__":
    fetch_stations()
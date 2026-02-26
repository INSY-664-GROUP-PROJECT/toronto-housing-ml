from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import requests

PLACE_NAME = "Toronto, Ontario, Canada"
OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "toronto_streets.geojson"
USER_AGENT = "toronto-housing-cluster-app/1.0"


def _nominatim_lookup(place_name: str) -> dict[str, Any]:
    resp = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params={"q": place_name, "format": "jsonv2", "limit": 1},
        headers={"User-Agent": USER_AGENT},
        timeout=60,
    )
    resp.raise_for_status()
    payload = resp.json()
    if not payload:
        raise RuntimeError(f"Place not found: {place_name}")
    return payload[0]


def _to_overpass_area_id(osm_type: str, osm_id: int) -> int:
    t = osm_type.lower()
    if t == "relation":
        return 3_600_000_000 + osm_id
    if t == "way":
        return 2_400_000_000 + osm_id
    if t == "node":
        return 3_600_000_000 + osm_id
    raise RuntimeError(f"Unsupported OSM type: {osm_type}")


def _fetch_street_ways(area_id: int) -> list[dict[str, Any]]:
    query = f"""
    [out:json][timeout:300];
    area({area_id})->.searchArea;
    (
      way["highway"](area.searchArea);
    );
    out geom;
    """
    resp = requests.post(
        "https://overpass-api.de/api/interpreter",
        data=query,
        headers={"User-Agent": USER_AGENT},
        timeout=360,
    )
    resp.raise_for_status()
    payload = resp.json()
    elements = payload.get("elements", [])
    return [el for el in elements if el.get("type") == "way" and el.get("geometry")]


def _ways_to_geojson(ways: list[dict[str, Any]]) -> dict[str, Any]:
    features: list[dict[str, Any]] = []
    for way in ways:
        geom = way.get("geometry", [])
        if len(geom) < 2:
            continue
        coords = [[float(pt["lon"]), float(pt["lat"])] for pt in geom]
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coords},
                "properties": {
                    "osm_id": int(way.get("id", -1)),
                    "highway": way.get("tags", {}).get("highway"),
                    "name": way.get("tags", {}).get("name"),
                },
            }
        )

    return {"type": "FeatureCollection", "features": features}


def main() -> None:
    place = _nominatim_lookup(PLACE_NAME)
    area_id = _to_overpass_area_id(str(place["osm_type"]), int(place["osm_id"]))
    ways = _fetch_street_ways(area_id)
    geojson = _ways_to_geojson(ways)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(geojson))
    print(f"Saved {len(geojson['features']):,} street segments to {OUT_PATH}")


if __name__ == "__main__":
    main()

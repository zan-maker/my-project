"""
MineLens AI - Proximity Search Tool
Searches for known mineral deposits near a location using USGS MRDS.
"""

import numpy as np
import json
import os
from typing import Dict, List, Optional


# Sample USGS MRDS data for demo mode
SAMPLE_DEPOSITS = [
    {"name": "Greenbushes Lithium Mine", "lat": -33.85, "lon": 116.05, "minerals": ["lithium", "tantalum"], "country": "Australia", "status": "Active", "type": "Pegmatite"},
    {"name": "Escondida Copper Mine", "lat": -24.27, "lon": -69.07, "minerals": ["copper", "gold", "silver"], "country": "Chile", "status": "Active", "type": "Porphyry"},
    {"name": "Mutanda Cobalt Mine", "lat": -10.96, "lon": 25.82, "minerals": ["cobalt", "copper"], "country": "DRC", "status": "Active", "type": "Sedimentary"},
    {"name": "Bayan Obo REE Deposit", "lat": 41.82, "lon": 110.00, "minerals": ["rare_earth", "iron", "niobium"], "country": "China", "status": "Active", "type": "Carbonatite"},
    {"name": "Voisey's Bay Nickel", "lat": 55.30, "lon": -60.20, "minerals": ["nickel", "copper", "cobalt"], "country": "Canada", "status": "Active", "type": "Magmatic Sulfide"},
    {"name": "Grasberg Copper-Gold", "lat": -4.05, "lon": 137.11, "minerals": ["copper", "gold"], "country": "Indonesia", "status": "Active", "type": "Porphyry"},
    {"name": "Atacama Lithium Brine", "lat": -23.50, "lon": -68.20, "minerals": ["lithium", "potassium"], "country": "Chile", "status": "Active", "type": "Brine"},
    {"name": "Mount Isa Copper-Zinc", "lat": -20.73, "lon": 139.49, "minerals": ["copper", "zinc", "lead", "silver"], "country": "Australia", "status": "Active", "type": "Sedimentary Exhalative"},
    {"name": "Jadar Valley Lithium", "lat": 44.35, "lon": 19.30, "minerals": ["lithium", "boron"], "country": "Serbia", "status": "Development", "type": "Sedimentary"},
    {"name": "Thacker Pass Lithium", "lat": 41.30, "lon": -118.55, "minerals": ["lithium"], "country": "USA", "status": "Development", "type": "Sedimentary Volcanic"},
    {"name": "Kamoto Copper-Cobalt", "lat": -10.70, "lon": 25.40, "minerals": ["copper", "cobalt"], "country": "DRC", "status": "Active", "type": "Sedimentary"},
    {"name": "Olympic Dam", "lat": -30.44, "lon": 136.89, "minerals": ["copper", "uranium", "gold", "silver", "rare_earth"], "country": "Australia", "status": "Active", "type": "IOCG"},
    {"name": "Ram River Cobalt", "lat": 52.60, "lon": -115.40, "minerals": ["cobalt", "copper", "silver"], "country": "Canada", "status": "Exploration", "type": "Vein"},
    {"name": "Bingham Canyon Copper", "lat": 40.53, "lon": -112.15, "minerals": ["copper", "gold", "molybdenum", "silver"], "country": "USA", "status": "Active", "type": "Porphyry"},
    {"name": "Salar de Uyuni Lithium", "lat": -20.22, "lon": -67.62, "minerals": ["lithium", "potassium", "magnesium"], "country": "Bolivia", "status": "Pre-production", "type": "Brine"},
]


def proximity_search(
    latitude: float,
    longitude: float,
    radius_km: float = 50,
    mineral_types: Optional[List[str]] = None,
) -> Dict:
    """
    Search for known mineral deposits, mines, and geological features near a location.
    
    Uses USGS MRDS (Mineral Resources Data System) and other public databases.
    In demo mode, uses a curated sample database.
    """
    
    EARTH_RADIUS_KM = 6371.0
    
    def haversine(lat1, lon1, lat2, lon2):
        """Calculate distance between two points in km."""
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return EARTH_RADIUS_KM * c
    
    # Find deposits within radius
    nearby_deposits = []
    
    # Load from data file if available, otherwise use samples
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "mrds_sample.json")
    deposits = SAMPLE_DEPOSITS
    
    if os.path.exists(data_path):
        with open(data_path) as f:
            deposits = json.load(f)
    
    for deposit in deposits:
        dist = haversine(latitude, longitude, deposit["lat"], deposit["lon"])
        
        if dist <= radius_km:
            # Filter by mineral type if specified
            if mineral_types:
                deposit_minerals = [m.lower() for m in deposit["minerals"]]
                matches = any(
                    any(mt.lower() in dm for dm in deposit_minerals)
                    for mt in mineral_types
                )
                if not matches:
                    continue
            
            nearby_deposits.append({
                **deposit,
                "distance_km": round(dist, 1),
            })
    
    # Sort by distance
    nearby_deposits.sort(key=lambda x: x["distance_km"])
    
    # Analyze patterns
    mineral_counts = {}
    type_counts = {}
    status_counts = {}
    
    for d in nearby_deposits:
        for m in d.get("minerals", []):
            mineral_counts[m] = mineral_counts.get(m, 0) + 1
        t = d.get("type", "Unknown")
        type_counts[t] = type_counts.get(t, 0) + 1
        s = d.get("status", "Unknown")
        status_counts[s] = status_counts.get(s, 0) + 1
    
    return {
        "query": {
            "latitude": latitude,
            "longitude": longitude,
            "radius_km": radius_km,
            "mineral_filter": mineral_types,
        },
        "total_deposits_found": len(nearby_deposits),
        "deposits": nearby_deposits[:20],  # Limit results
        "analysis": {
            "mineral_frequency": dict(sorted(mineral_counts.items(), key=lambda x: -x[1])),
            "deposit_types": type_counts,
            "operational_status": status_counts,
            "most_common_mineral": max(mineral_counts, key=mineral_counts.get) if mineral_counts else None,
        },
        "regional_assessment": _assess_region(nearby_deposits, mineral_types),
    }


def _assess_region(deposits: list, mineral_filter: list) -> Dict:
    """Assess the mineral prospectivity of the queried region."""
    if not deposits:
        return {
            "rating": "Unknown",
            "note": "No known deposits found within search radius. This could indicate an underexplored region (opportunity) or lack of mineral endowment.",
        }
    
    active = [d for d in deposits if d.get("status") == "Active"]
    n = len(deposits)
    
    if n >= 5:
        rating = "Very High"
        note = f"Dense cluster of {n} known deposits including {len(active)} active mines. Region has well-established mineral endowment."
    elif n >= 3:
        rating = "High"
        note = f"Multiple deposits ({n}) found nearby, including {len(active)} active operations. Strong mineral potential."
    elif n >= 1:
        rating = "Moderate"
        note = f"{n} deposit(s) found nearby. Proximity to known mineralization is encouraging."
    else:
        rating = "Low"
        note = "Limited known mineralization in immediate vicinity."
    
    return {
        "rating": rating,
        "note": note,
        "active_mines_nearby": len(active),
        "closest_deposit": f"{deposits[0]['name']} ({deposits[0]['distance_km']} km)" if deposits else None,
    }

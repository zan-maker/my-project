"""
Tool 6: Geological Survey Lookup
Look up geological survey data for a region.
Retrieves rock types, formations, fault lines, mineral occurrences from USGS data.
"""

from typing import Dict, List, Optional

# Curated geological survey data by region
GEOLOGICAL_DATA = {
    # South America
    "chile": {
        "tectonic_setting": "Active continental margin (Andean subduction zone)",
        "dominant_rock_types": ["Andesite", "Dacite", "Granodiorite", "Porphyritic Intrusions"],
        "key_formations": ["Chuquicamata Porphyry Complex", "Atacama Gravels", "La Negra Formation"],
        "fault_lines": ["Atacama Fault System", "Domeyko Fault Zone", "West Fissure Fault System"],
        "mineral_occurrences": {"copper": "abundant", "lithium": "abundant", "gold": "significant", "molybdenum": "significant", "silver": "significant"},
        "geochronology": "Mesozoic-Cenozoic arc magmatism (200-5 Ma)",
    },
    "peru": {
        "tectonic_setting": "Active continental margin (Nazca subduction)",
        "dominant_rock_types": ["Andesite", "Quartz Monzonite", "Diorite", "Volcaniclastics"],
        "key_formations": ["Cerro Verde Porphyry", "Antamina Skarn", "Yanacocha Deposit"],
        "fault_lines": ["Cordillera Blanca Fault", "Huaura Fault System"],
        "mineral_occurrences": {"copper": "abundant", "gold": "abundant", "silver": "abundant", "zinc": "significant", "lead": "significant"},
        "geochronology": "Cenozoic arc magmatism (65-5 Ma)",
    },
    "argentina": {
        "tectonic_setting": "Back-arc basin and foreland",
        "dominant_rock_types": ["Rhyolite", "Basalt", "Pegmatite", "Evaporite"],
        "key_formations": ["Salar de Hombre Muerto", "Salar del Rincon", "Sierra Pintada Pegmatite Belt"],
        "fault_lines": ["Precordilleran Fault System", "Andean Thrust Belt"],
        "mineral_occurrences": {"lithium": "abundant", "copper": "significant", "gold": "moderate", "boron": "significant"},
        "geochronology": "Mesozoic-Cenozoic extension and volcanism",
    },
    # Africa
    "drc": {
        "tectonic_setting": "Central African craton with Proterozoic mobile belts",
        "dominant_rock_types": ["Shale", "Sandstone", "Dolomite", "Katanga Supergroup metasediments"],
        "key_formations": ["Katanga Supergroup", "Kibaran Belt", "Central African Copperbelt"],
        "fault_lines": ["Lufilian Arc", "Kibara Belt shear zones", "Congo Craton boundary"],
        "mineral_occurrences": {"cobalt": "abundant", "copper": "abundant", "tantalum": "significant", "tin": "significant", "diamond": "significant"},
        "geochronology": "Proterozoic (1000-500 Ma)",
    },
    # Australia
    "australia": {
        "tectonic_setting": "Stable continental crust (Archean-Proterozoic cratons)",
        "dominant_rock_types": ["Granite", "Gneiss", "Banded Iron Formation", "Greenstone"],
        "key_formations": ["Yilgarn Craton", "Pilbara Craton", "Mt Isa Inlier", "Olympic Dam Breccia Complex"],
        "fault_lines": ["Darling Fault", "Torrens Hinge Zone", "Isan Orogeny structures"],
        "mineral_occurrences": {"lithium": "abundant", "rare_earth": "abundant", "gold": "abundant", "iron": "abundant", "uranium": "abundant", "copper": "significant", "nickel": "significant"},
        "geochronology": "Archean to Proterozoic (3500-1000 Ma)",
    },
    # North America
    "usa": {
        "tectonic_setting": "Diverse: passive margin, stable craton, active Cordilleran belt",
        "dominant_rock_types": ["Granite", "Basalt", "Limestone", "Shale", "Evaporite"],
        "key_formations": ["Carlin Trend", "Porphyry Belt (AZ, UT)", "Clayton Valley", "Rare Earth deposits (Mountain Pass)"],
        "fault_lines": ["San Andreas Fault", "Basin and Range normal faults", "Cordilleran Thrust Belt"],
        "mineral_occurrences": {"gold": "abundant", "copper": "abundant", "rare_earth": "significant", "lithium": "significant", "uranium": "significant"},
        "geochronology": "Archean to Cenozoic (diverse ages)",
    },
    "canada": {
        "tectonic_setting": "Stable craton with Proterozoic belts and Cordilleran active margin",
        "dominant_rock_types": ["Greenstone", "Granite", "Gabbro", "Anorthosite", "Ultramafic"],
        "key_formations": ["Sudbury Igneous Complex", "Abitibi Greenstone Belt", "Voisey's Bay", "Porphyry Belt (BC)"],
        "fault_lines": ["Great Lakes Tectonic Zone", "Cordilleran Deformation Front", "Trans-Hudson Orogen"],
        "mineral_occurrences": {"nickel": "abundant", "copper": "abundant", "gold": "abundant", "uranium": "significant", "diamond": "significant", "rare_earth": "significant"},
        "geochronology": "Archean to Proterozoic (3000-1000 Ma)",
    },
    # Asia
    "china": {
        "tectonic_setting": "Complex mosaic of cratonic blocks and Phanerozoic orogens",
        "dominant_rock_types": ["Carbonatite", "Granite", "Limestone", "Basalt", "Porphyry"],
        "key_formations": ["Bayan Obo REE Deposit", "Jinchuan Ni-Cu Deposit", "Yangtze Craton", "Tibetan Plateau porphyries"],
        "fault_lines": ["Longmenshan Fault", "Kunlun Fault", "Tan-Lu Fault System"],
        "mineral_occurrences": {"rare_earth": "abundant", "tungsten": "abundant", "tin": "abundant", "antimony": "abundant", "copper": "significant", "nickel": "significant"},
        "geochronology": "Archean to Cenozoic (diverse ages)",
    },
    "indonesia": {
        "tectonic_setting": "Active island arc system (Pacific Ring of Fire)",
        "dominant_rock_types": ["Andesite", "Diorite", "Porphyry", "Volcaniclastics"],
        "key_formations": ["Grasberg-Ertsberg District", "Batu Hijau Porphyry", "Wetar Porphyry Belt"],
        "fault_lines": ["Great Sumatran Fault", "Palu-Koro Fault", "Sorong Fault"],
        "mineral_occurrences": {"copper": "abundant", "gold": "abundant", "nickel": "abundant", "tin": "abundant", "coal": "significant"},
        "geochronology": "Cenozoic arc magmatism (30-1 Ma)",
    },
}


def geological_survey_lookup(latitude: float, longitude: float, data_layers: list = None) -> Dict:
    """
    Look up geological survey data for a region.
    Retrieves rock types, formations, fault lines, mineral occurrences.
    
    Args:
        latitude: Latitude of the point of interest
        longitude: Longitude of the point of interest
        data_layers: Data layers to retrieve. Options:
            ['geology', 'mineral_occurrences', 'fault_lines', 'geochemistry', 'magnetics', 'gravity']
    
    Returns:
        Geological survey data dictionary
    """
    if data_layers is None:
        data_layers = ["geology", "mineral_occurrences"]
    
    # Determine region from coordinates (simplified geographic lookup)
    region = _infer_region(latitude, longitude)
    
    data = GEOLOGICAL_DATA.get(region, {
        "tectonic_setting": "Regional data not available in lookup table",
        "dominant_rock_types": [],
        "key_formations": [],
        "fault_lines": [],
        "mineral_occurrences": {},
        "geochronology": "Unknown",
    })
    
    result = {
        "query": {"latitude": latitude, "longitude": longitude},
        "inferred_region": region,
    }
    
    if "geology" in data_layers:
        result["tectonic_setting"] = data["tectonic_setting"]
        result["dominant_rock_types"] = data["dominant_rock_types"]
        result["key_formations"] = data["key_formations"]
        result["geochronology"] = data["geochronology"]
    
    if "mineral_occurrences" in data_layers:
        result["mineral_occurrences"] = data["mineral_occurrences"]
    
    if "fault_lines" in data_layers:
        result["fault_lines"] = data["fault_lines"]
    
    if "geochemistry" in data_layers:
        # Simulated geochemical baseline data
        result["geochemistry"] = {
            "soil_baseline": {
                "copper_ppm": 35 + (hash(str(latitude)) % 100),
                "nickel_ppm": 25 + (hash(str(longitude)) % 80),
                "cobalt_ppm": 10 + (hash(str(latitude + longitude)) % 40),
                "rare_earth_ppm": 150 + (hash(str(latitude - longitude)) % 200),
            },
            "anomaly_thresholds": {
                "copper_ppm": 100,
                "nickel_ppm": 80,
                "cobalt_ppm": 40,
                "rare_earth_ppm": 300,
            },
        }
    
    if "magnetics" in data_layers:
        result["magnetics"] = {
            "regional_field_nT": 45000 + (hash(str(latitude)) % 10000),
            "anomaly_detection": "Magnetic survey data available from national geological survey",
            "interpretation": "Potential for buried intrusions and mineralized structures",
        }
    
    if "gravity" in data_layers:
        result["gravity"] = {
            "bouguer_anomaly_mGal": -200 + (hash(str(longitude)) % 400),
            "interpretation": "Gravity data useful for delineating basin architecture and intrusion boundaries",
        }
    
    return result


def _infer_region(lat: float, lon: float) -> str:
    """Infer region name from coordinates (simplified)."""
    # South America
    if -56 <= lat <= 13 and -82 <= lon <= -34:
        if lat <= -18 and lon >= -72:
            return "chile"
        if lat <= -1 and lon >= -82:
            return "peru"
        if lat <= -22 and lon <= -62:
            return "argentina"
    
    # Africa
    if -35 <= lat <= 37 and -18 <= lon <= 52:
        if -14 <= lat <= 5 and 12 <= lon <= 32:
            return "drc"
    
    # Australia
    if -45 <= lat <= -10 and 112 <= lon <= 155:
        return "australia"
    
    # North America
    if 24 <= lat <= 72 and -170 <= lon <= -52:
        if lat <= 50:
            return "usa"
        return "canada"
    
    # China
    if 18 <= lat <= 54 and 73 <= lon <= 135:
        return "china"
    
    # Indonesia
    if -11 <= lat <= 6 and 95 <= lon <= 141:
        return "indonesia"
    
    return "unknown"


if __name__ == "__main__":
    # Quick test
    result = geological_survey_lookup(-23.5, -68.18)
    print(json.dumps(result, indent=2))

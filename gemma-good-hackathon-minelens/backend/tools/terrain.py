"""
MineLens AI - Terrain Classification Tool
Classifies terrain types from satellite/DEM imagery.
"""

import numpy as np
from typing import Dict, Optional


def terrain_classifier(
    image_path: str,
    classification_detail: str = "detailed",
) -> Dict:
    """
    Classify terrain types from satellite imagery or elevation data.
    
    Identifies geological formations associated with mineral deposits:
    - Pegmatites (lithium, rare earths)
    - Greenstone belts (gold, copper, nickel)
    - Porphyry intrusions (copper, molybdenum)
    - Lateritic surfaces (nickel, cobalt)
    - Carbonatite complexes (rare earths, niobium)
    """
    
    TERRAIN_TYPES = {
        "pegmatite_intrusion": {
            "description": "Coarse-grained igneous rock bodies, often hosting lithium and rare earth minerals",
            "mineral_association": ["lithium", "tantalum", "rare_earth", "beryllium"],
            "spectral_indicators": ["high_albedo", "quartz_rich", "feldspar_signature"],
            "morphology": ["linear_bodies", "cross_cutting", "associated_with_granite"],
        },
        "greenstone_belt": {
            "description": "Metamorphosed volcanic and sedimentary rocks, key hosts for gold and base metals",
            "mineral_association": ["gold", "copper", "zinc", "nickel"],
            "spectral_indicators": ["iron_oxide_staining", "mafic_minerals", "silica_cap"],
            "morphology": ["linear_belt", "folded_terrain", "shear_zones"],
        },
        "porphyry_system": {
            "description": "Large igneous intrusion with associated hydrothermal alteration, primary copper source",
            "mineral_association": ["copper", "molybdenum", "gold", "silver"],
            "spectral_indicators": ["phyllic_alteration", "argillic_alteration", "potassic_core"],
            "morphology": ["circular_feature", "breccia_pipe", "stockwork_veining"],
        },
        "lateritic_surface": {
            "description": "Weathered ultramafic/mafic rock surfaces, major nickel and cobalt deposits",
            "mineral_association": ["nickel", "cobalt", "manganese", "iron"],
            "spectral_indicators": ["iron_oxide_rich", "red_brown_color", "low_vegetation"],
            "morphology": ["plateau_surface", "duricrust", "iron_cap"],
        },
        "carbonatite_complex": {
            "description": "Igneous carbonate-rich intrusions, primary source of rare earth elements",
            "mineral_association": ["rare_earth", "niobium", "phosphate", "fluorite"],
            "spectral_indicators": ["carbonate_absorption", "circular_pattern", "soil_anomaly"],
            "morphology": ["circular_complex", "ring_dike", "central_intrusion"],
        },
        "alluvial_fan": {
            "description": "Sedimentary fan deposits, can host placer mineral deposits",
            "mineral_association": ["gold", "tin", "rare_earth", "diamond"],
            "spectral_indicators": ["mixed_sediment", "drainage_pattern", "fan_shape"],
            "morphology": ["fan_shaped", "apex_point", "radial_drainage"],
        },
        "sedimentary_basin": {
            "description": "Large sedimentary basin, potential for SEDEX, MVT, and sandstone-hosted deposits",
            "mineral_association": ["lead", "zinc", "copper", "uranium", "vanadium"],
            "spectral_indicators": ["layered_structure", "carbonate_units", "red_beds"],
            "morphology": ["flat_lying", "layered", "large_extent"],
        },
    }
    
    # Simulated classification results
    np.random.seed(hash(image_path) % 2**31)
    
    # Select 2-4 terrain types with confidence scores
    n_types = np.random.randint(2, 5)
    terrain_keys = list(TERRAIN_TYPES.keys())
    selected = np.random.choice(terrain_keys, size=min(n_types, len(terrain_keys)), replace=False)
    
    classifications = []
    for key in selected:
        conf = round(np.random.uniform(0.3, 0.95), 3)
        terrain = TERRAIN_TYPES[key]
        
        classifications.append({
            "terrain_type": key,
            "confidence": conf,
            "description": terrain["description"],
            "mineral_association": terrain["mineral_association"],
            "evidence": {
                "spectral_indicators": terrain["spectral_indicators"][:2],
                "morphological_evidence": terrain["morphology"][:2],
            },
            "mineral_prospectivity": _assess_prospectivity(conf, terrain["mineral_association"]),
        })
    
    # Sort by confidence
    classifications.sort(key=lambda x: x["confidence"], reverse=True)
    
    return {
        "image_path": image_path,
        "classification_detail": classification_detail,
        "terrain_classifications": classifications,
        "dominant_terrain": classifications[0]["terrain_type"] if classifications else None,
        "mineral_prospectivity": classifications[0]["mineral_prospectivity"] if classifications else "Unknown",
        "recommended_investigation_targets": [
            c["terrain_type"] for c in classifications if c["confidence"] >= 0.6
        ],
    }


def _assess_prospectivity(confidence: float, associated_minerals: list) -> str:
    if confidence >= 0.8 and len(associated_minerals) >= 3:
        return "High"
    elif confidence >= 0.6:
        return "Moderate-High"
    elif confidence >= 0.4:
        return "Moderate"
    else:
        return "Low"

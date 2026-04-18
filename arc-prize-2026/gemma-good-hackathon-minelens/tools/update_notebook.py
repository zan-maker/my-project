#!/usr/bin/env python3
"""Update the MineLens AI notebook to include the 6th tool (geological_survey_lookup)."""
import json

NOTEBOOK_PATH = "/home/z/my-project/arc-prize-2026/gemma-good-hackathon-minelens/notebooks/minelens_ai.ipynb"

with open(NOTEBOOK_PATH) as f:
    nb = json.load(f)

print(f"Original cells: {len(nb['cells'])}")

# Read the geological_survey_lookup.py from backend
tool6_lines = [
    "# ============================================================",
    "# TOOL 6: Geological Survey Lookup",
    "# ============================================================",
    "",
    "GEOLOGICAL_SURVEY_DATA = {",
    '    "chile": {',
    '        "tectonic_setting": "Andean Cordillera - Nazca-South American plate subduction zone",',
    '        "dominant_rock_types": ["andesite", "porphyry", "granodiorite", "evaporite"],',
    '        "key_formations": ["Chuquicamata Porphyry Complex", "Atacama Salt Flat"],',
    '        "fault_systems": ["Atacama Fault Zone", "Domeyko Fault System"],',
    '        "mineral_occurrences": {"lithium": "Brine deposits in salars", "copper": "World-class porphyry deposits"},',
    '        "data_layers_available": ["geology", "mineral_occurrences", "fault_lines", "geochemistry", "magnetics", "gravity"]',
    "    },",
    '    "peru": {',
    '        "tectonic_setting": "Andean Cordillera - active continental margin",',
    '        "dominant_rock_types": ["granodiorite", "monzogranite", "andesite", "limestone"],',
    '        "key_formations": ["Antamina Skarn", "Cerro Verde Porphyry"],',
    '        "fault_systems": ["Andahuaylas-Yauri Fault", "Cordillera Blanca Fault"],',
    '        "mineral_occurrences": {"copper": "Major porphyry belt", "gold": "Orogenic and porphyry"},',
    '        "data_layers_available": ["geology", "mineral_occurrences", "fault_lines", "geochemistry"]',
    "    },",
    '    "australia": {',
    '        "tectonic_setting": "Archean-Proterozoic cratons with Phanerozoic cover",',
    '        "dominant_rock_types": ["banded_iron_formation", "granite", "greenstone", "pegmatite"],',
    '        "key_formations": ["Yilgarn Craton Greenstone Belts", "Mount Isa Inlier", "Olympic Dam IOCG"],',
    '        "fault_systems": ["Yilgarn Craton Shear Zones", "Isan Orogeny Structures"],',
    '        "mineral_occurrences": {"lithium": "Pegmatite deposits (Greenbushes)", "rare_earth": "Carbonatite (Mount Weld)"},',
    '        "data_layers_available": ["geology", "mineral_occurrences", "fault_lines", "geochemistry"]',
    "    },",
    '    "drc": {',
    '        "tectonic_setting": "Central African Copperbelt - Proterozoic Lufilian Arc",',
    '        "dominant_rock_types": ["katangan_supergroup", "shale", "dolomite", "conglomerate"],',
    '        "key_formations": ["Central African Copperbelt", "Kamoto Mine Sequence"],',
    '        "fault_systems": ["Lufilian Arc Thrusts", "Kalongwe Fault"],',
    '        "mineral_occurrences": {"cobalt": "Sediment-hosted stratiform", "copper": "Stratiform and vein deposits"},',
    '        "data_layers_available": ["geology", "mineral_occurrences", "fault_lines"]',
    "    },",
    '    "usa": {',
    '        "tectonic_setting": "Diverse: Basin and Range, Cordillera, Appalachian, Stable Craton",',
    '        "dominant_rock_types": ["granite", "basalt", "shale", "sandstone", "evaporite"],',
    '        "key_formations": ["Carlin Trend Gold District", "Bingham Canyon Porphyry"],',
    '        "fault_systems": ["San Andreas Fault", "Basin and Range Normal Faults"],',
    '        "mineral_occurrences": {"lithium": "Clayton Valley brine", "copper": "Porphyry and sedimentary"},',
    '        "data_layers_available": ["geology", "mineral_occurrences", "fault_lines", "geochemistry"]',
    "    },",
    '    "china": {',
    '        "tectonic_setting": "Complex multi-cycle orogen with cratonic blocks",',
    '        "dominant_rock_types": ["granite", "carbonatite", "porphyry", "shale"],',
    '        "key_formations": ["Bayan Obo REE Deposit", "Jinchuan Ni-Cu Sulfide"],',
    '        "fault_systems": ["Tan-Lu Fault Zone", "Kunlun Fault System"],',
    '        "mineral_occurrences": {"rare_earth": "Bayan Obo carbonatite (world largest)", "nickel": "Jinchuan magmatic sulfide"},',
    '        "data_layers_available": ["geology", "mineral_occurrences", "fault_lines"]',
    "    },",
    "}",
    "",
    "",
    "def geological_survey_lookup(region: str, data_layers: list = None) -> dict:",
    '    """Look up regional geological survey data including tectonic setting, rock types, and fault systems."""',
    "    region_key = region.lower().strip()",
    "    survey_data = GEOLOGICAL_SURVEY_DATA.get(region_key)",
    "    if not survey_data:",
    "        for key, data in GEOLOGICAL_SURVEY_DATA.items():",
    "            if key in region_key or region_key in key:",
    "                survey_data = data",
    "                break",
    "    if not survey_data:",
    '        return {"error": f"No geological survey data for region: {region}"}',
    "    result = {",
    '        "region": region,',
    '        "tectonic_setting": survey_data["tectonic_setting"],',
    '        "dominant_rock_types": survey_data["dominant_rock_types"],',
    '        "key_formations": survey_data["key_formations"],',
    '        "fault_systems": survey_data["fault_systems"],',
    '        "mineral_occurrences": survey_data["mineral_occurrences"],',
    "    }",
    "    if data_layers:",
    '        available = survey_data.get("data_layers_available", [])',
    "        matched = [l for l in data_layers if l in available]",
    '        result["requested_layers"] = data_layers',
    '        result["available_layers"] = matched',
    "    else:",
    '        result["available_data_layers"] = survey_data.get("data_layers_available", [])',
    "    return result",
    "",
    'print("Tool 6: Geological Survey Lookup defined \\u2713")',
    'print(f"  Coverage: {list(GEOLOGICAL_SURVEY_DATA.keys())}")',
]

geo_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [line + "\n" for line in tool6_lines],
}
if geo_cell["source"]:
    geo_cell["source"][-1] = geo_cell["source"][-1].rstrip("\n")

nb["cells"].insert(9, geo_cell)

# 2. Update tool registration cell (now index 10)
reg_src = "".join(nb["cells"][10]["source"])

new_schema = [
    "    {",
    '        "type": "function",',
    '        "function": {',
    '            "name": "geological_survey_lookup",',
    '            "description": "Look up regional geological survey data including tectonic setting, rock types, formations, and fault systems.",',
    '            "parameters": {',
    '                "type": "object",',
    '                "properties": {',
    '                    "region": {"type": "string", "description": "Region or country name"},',
    '                    "data_layers": {"type": "array", "items": {"type": "string"}, "description": "Optional: geology, mineral_occurrences, fault_lines, geochemistry, magnetics, gravity"}',
    "                },",
    '                "required": ["region"]',
    "            }",
    "        }",
    "    },",
]

reg_src = reg_src.replace(
    '    {\n        "type": "function",\n        "function": {\n            "name": "generate_report",',
    "\n".join(new_schema) + '\n    {\n        "type": "function",\n        "function": {\n            "name": "generate_report",'
)

reg_src = reg_src.replace(
    '    "generate_report": generate_report,\n}',
    '    "generate_report": generate_report,\n    "geological_survey_lookup": geological_survey_lookup,\n}'
)

nb["cells"][10]["source"] = [line + "\n" for line in reg_src.split("\n")]
if nb["cells"][10]["source"]:
    nb["cells"][10]["source"][-1] = nb["cells"][10]["source"][-1].rstrip("\n")

# 3. Update agentic pipeline (now index 13) to call geological survey
pipe_src = "".join(nb["cells"][13]["source"])

old_text = '        if step_name == "spectral_analysis":'
new_text = (
    '        if step_name == "geological_context":\n'
    '            region_name = location_name.split(",")[-1].strip().lower()\n'
    "            result = geological_survey_lookup(region_name)\n"
    '            analysis_data["geological"] = result\n'
    '            tectonic = result.get("tectonic_setting", "N/A")\n'
    '            print(f"     \\u2192 {tectonic[:70]}...")\n'
    "\n"
    '        elif step_name == "spectral_analysis":'
)

pipe_src = pipe_src.replace(old_text, new_text)

nb["cells"][13]["source"] = [line + "\n" for line in pipe_src.split("\n")]
if nb["cells"][13]["source"]:
    nb["cells"][13]["source"][-1] = nb["cells"][13]["source"][-1].rstrip("\n")

# 4. Update summary markdown (now index 14)
sum_src = "".join(nb["cells"][14]["source"])
sum_src = sum_src.replace("5 Specialized Tools", "6 Specialized Tools")
sum_src = sum_src.replace(
    "spectral analysis, terrain classification, proximity search, risk assessment, and report generation",
    "spectral analysis, terrain classification, proximity search, risk assessment, report generation, and geological survey lookup",
)
sum_src = sum_src.replace("5 tools", "6 tools")
sum_src = sum_src.replace("5+ tools", "6 tools")

nb["cells"][14]["source"] = [line + "\n" for line in sum_src.split("\n")]
if nb["cells"][14]["source"]:
    nb["cells"][14]["source"][-1] = nb["cells"][14]["source"][-1].rstrip("\n")

with open(NOTEBOOK_PATH, "w") as f:
    json.dump(nb, f, indent=1)

print(f"Updated cells: {len(nb['cells'])}")
print("Notebook saved successfully!")

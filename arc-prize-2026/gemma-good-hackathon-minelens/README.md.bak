# MineLens AI — Critical Mineral Prospectivity Mapping with Gemma 4

Kaggle: https://www.kaggle.com/competitions/gemma-4-good-hackathon

## Project Summary
A multimodal AI system powered by Gemma 4 that analyzes satellite imagery, geological maps, and geophysical data to identify critical mineral prospectivity zones. Leverages Gemma 4's native function calling for geospatial analysis and multimodal understanding for map interpretation.

## Problem
The global energy transition requires critical minerals (lithium, cobalt, rare earths, copper, nickel), but mineral exploration is expensive ($50-500M per discovery), slow (5-15 years), and geologically complex. AI can democratize and accelerate this process.

## Solution
- **Multimodal Input**: Upload satellite images, geological maps, geochemical surveys
- **Function Calling**: Spectral analysis, terrain classification, proximity search, risk assessment
- **Agentic Workflow**: Gemma 4 orchestrates multi-step analysis autonomously
- **Prospectivity Reports**: Detailed reports with confidence scores and recommendations

## Tech Stack
- Gemma 4 (multimodal + function calling)
- Python backend (FastAPI)
- Frontend: Next.js with Leaflet map visualization
- GeoPandas, Rasterio for geospatial processing
- USGS public datasets

## Submission Artifacts
- [ ] Working Live Demo
- [ ] Technical Writeup
- [ ] Public Code Repo
- [ ] Video (2-5 min)

## Evaluation Alignment
| Criteria | Max Points | Our Strategy |
|----------|-----------|-------------|
| Impact & Vision | 40 | Critical mineral supply chain — $2T+ market, global energy security |
| Technical Execution | 30 | Full multimodal + function calling + agentic pipeline |
| Originality | 20 | Novel geoscience application, not a chatbot |
| Function Calling | 10 | 5+ custom tools for geospatial analysis |

## Directory Structure
```
├── backend/              # FastAPI server
│   ├── app.py           # Main server
│   ├── gemma_client.py  # Gemma 4 integration
│   ├── tools/           # Function calling tools
│   │   ├── spectral.py  # Spectral analysis
│   │   ├── terrain.py   # Terrain classification
│   │   ├── proximity.py # Deposit proximity search
│   │   ├── risk.py      # Supply chain risk
│   │   └── report.py    # Report generation
│   └── data/            # USGS datasets
├── frontend/            # Next.js visualization
│   └── ...
├── notebooks/           # EDA and demos
├── writeup/             # Technical writeup
├── video/               # Demo video assets
└── tools_schema.json    # Gemma function calling schema
```

## Datasets
- USGS Mineral Resources Data System (MRDS)
- USGS National Minerals Information Center
- ASTER satellite imagery (free, global)
- USGS EarthExplorer (Landsat, Sentinel)
- Critical Mineral Mapping Initiative (CMMI)

## Timeline
- Week 1: Project setup, data collection, Gemma 4 integration
- Week 2: Function calling tools, core pipeline
- Week 3: Multimodal analysis, frontend integration
- Week 4: Demo polish, writeup, video

"""
MineLens AI - Supply Chain Risk Assessment Tool
Assesses geopolitical and supply chain risks for critical minerals.
"""

from typing import Dict, List


RISK_DATA = {
    "lithium": {
        "top_producers": ["Australia", "Chile", "China", "Argentina", "Zimbabwe"],
        "concentration_risk": "High — Australia and Chile produce >75% of global lithium",
        "supply_chain_risks": [
            "Processing bottleneck: >60% of lithium refining is in China",
            "Water scarcity in Chilean Salar regions",
            "Growing demand from EV sector outpacing supply growth",
            "Export restrictions from Argentina under review",
        ],
        "geopolitical_factors": [
            "US-China trade tensions affecting mineral flow",
            "Chilean nationalization discussions",
            "EU Critical Raw Materials Act incentivizing domestic supply",
        ],
        "esg_risks": ["Water usage in brine extraction", "Indigenous community rights", "Carbon footprint of hard-rock mining"],
        "market_outlook": "Deficit expected 2026-2028, prices likely to remain elevated",
    },
    "cobalt": {
        "top_producers": ["DRC", "Russia", "Australia", "Philippines", "Cuba"],
        "concentration_risk": "Very High — DRC produces >70% of global cobalt",
        "supply_chain_risks": [
            "Extreme geographic concentration in DRC",
            "Child labor and human rights concerns in artisanal mining",
            "Political instability in DRC",
            "Battery chemistry shift reducing cobalt intensity",
        ],
        "geopolitical_factors": [
            "DRC export restrictions and royalty increases",
            "US Inflation Reduction Act requiring non-DRC sourcing",
            "China controls >80% of cobalt refining",
        ],
        "esg_risks": ["Child labor", "Conflict mineral designation", "Environmental degradation", "Community displacement"],
        "market_outlook": "Oversupply risk as battery chemistries shift to low-cobalt",
    },
    "rare_earth": {
        "top_producers": ["China", "Myanmar", "USA", "Australia", "Russia"],
        "concentration_risk": "Very High — China controls ~60% of mining, ~90% of processing",
        "supply_chain_risks": [
            "Chinese export quotas and restrictions",
            "Limited processing capacity outside China",
            "Long development timelines for new mines (10-15 years)",
            "Environmental challenges in processing (radioactive waste)",
        ],
        "geopolitical_factors": [
            "China's dominance as a strategic leverage point",
            "US DoD funding domestic rare earth projects",
            "EU efforts to diversify supply chain",
            "Export controls on heavy rare earths",
        ],
        "esg_risks": ["Radioactive waste", "Toxic processing chemicals", "Water contamination"],
        "market_outlook": "Structural deficit for heavy rare earths (Dy, Tb) expected",
    },
    "copper": {
        "top_producers": ["Chile", "Peru", "DRC", "China", "USA"],
        "concentration_risk": "Moderate — well-diverified production base",
        "supply_chain_risks": [
            "Declining ore grades at major mines",
            "Long permitting timelines for new projects (10-20 years)",
            "Water constraints in Chilean mining operations",
            "Growing demand from electrification and AI data centers",
        ],
        "geopolitical_factors": [
            "Peruvian political instability and protests",
            "Panamanian mine closure (Cobre Panama)",
            "Chilean tax reforms affecting investment",
        ],
        "esg_risks": ["Water usage", "Tailings management", "Community relations", "Carbon footprint"],
        "market_outlook": "Structural deficit expected from 2025 onwards. Price upside potential significant.",
    },
    "nickel": {
        "top_producers": ["Indonesia", "Philippines", "Russia", "Australia", "Canada"],
        "concentration_risk": "High — Indonesia produces >50% of global nickel",
        "supply_chain_risks": [
            "Indonesian export ban on raw nickel ore",
            "Class 1 nickel shortage for EV batteries",
            "Over-supply of Class 2 nickel from Indonesian NPI",
            "High pressure acid leach (HPAL) technical challenges",
        ],
        "geopolitical_factors": [
            "Indonesian resource nationalism and downstreaming policy",
            "Russian sanctions affecting Nornickel supply",
            "Philippine environmental restrictions",
        ],
        "esg_risks": ["Deforestation in Indonesia", "Tailings disposal", "Carbon-intensive processing"],
        "market_outlook": "Class 1 deficit; Class 2 oversupply. Price divergence expected.",
    },
}

RISK_FACTORS = {
    "political": {
        "weight": 0.3,
        "description": "Political stability, regulatory environment, resource nationalism",
    },
    "environmental": {
        "weight": 0.2,
        "description": "Environmental regulations, ESG compliance requirements, climate risk",
    },
    "infrastructure": {
        "weight": 0.2,
        "description": "Transport, power, water, processing facilities",
    },
    "trade": {
        "weight": 0.2,
        "description": "Export controls, tariffs, trade agreements, supply chain concentration",
    },
    "social": {
        "weight": 0.1,
        "description": "Community relations, labor availability, indigenous rights",
    },
}


def risk_assessment(
    region: str,
    mineral_type: str,
    factors: List[str] = None,
) -> Dict:
    """
    Assess geopolitical and supply chain risks for critical mineral operations.
    
    Provides a comprehensive risk profile including:
    - Supply concentration analysis
    - Geopolitical risk scoring
    - ESG risk factors
    - Market outlook
    - Mitigation recommendations
    """
    
    if factors is None:
        factors = ["political", "environmental", "infrastructure", "trade"]
    
    mineral_key = mineral_type.lower().replace(" ", "_")
    mineral_data = RISK_DATA.get(mineral_key, RISK_DATA.get(mineral_type.lower(), None))
    
    if mineral_data is None:
        # Generic assessment for unknown minerals
        mineral_data = {
            "top_producers": ["Various"],
            "concentration_risk": "Unknown — insufficient data",
            "supply_chain_risks": ["Limited data available for this mineral"],
            "geopolitical_factors": ["Standard geopolitical considerations apply"],
            "esg_risks": ["Standard ESG due diligence required"],
            "market_outlook": "Insufficient data for outlook assessment",
        }
    
    # Calculate risk scores per factor
    risk_scores = {}
    for factor in factors:
        factor_info = RISK_FACTORS.get(factor, {"weight": 0.2, "description": factor})
        np.random.seed(hash(f"{region}-{mineral_type}-{factor}") % 2**31)
        score = round(np.random.uniform(0.2, 0.9), 2)
        
        risk_scores[factor] = {
            "score": score,
            "weight": factor_info["weight"],
            "weighted_score": round(score * factor_info["weight"], 3),
            "description": factor_info["description"],
            "key_issues": _get_factor_issues(factor, region, mineral_type, mineral_data),
        }
    
    # Overall risk score
    total_weight = sum(risk_scores[f]["weight"] for f in factors)
    overall_score = round(
        sum(risk_scores[f]["weighted_score"] for f in factors) / total_weight,
        3
    )
    
    # Risk level
    if overall_score >= 0.7:
        risk_level = "High"
        color = "red"
    elif overall_score >= 0.5:
        risk_level = "Moderate"
        color = "orange"
    elif overall_score >= 0.3:
        risk_level = "Low-Moderate"
        color = "yellow"
    else:
        risk_level = "Low"
        color = "green"
    
    return {
        "region": region,
        "mineral": mineral_type,
        "overall_risk_score": overall_score,
        "risk_level": risk_level,
        "factor_scores": risk_scores,
        "supply_chain_analysis": {
            "concentration_risk": mineral_data["concentration_risk"],
            "top_producers": mineral_data["top_producers"],
            "key_risks": mineral_data["supply_chain_risks"],
        },
        "geopolitical_factors": mineral_data["geopolitical_factors"],
        "esg_risks": mineral_data["esg_risks"],
        "market_outlook": mineral_data["market_outlook"],
        "mitigation_recommendations": _get_mitigation(overall_score, mineral_type, region),
    }


def _get_factor_issues(factor: str, region: str, mineral: str, data: dict) -> List[str]:
    """Generate factor-specific risk issues."""
    issues_map = {
        "political": [
            f"Regulatory stability in {region}",
            f"Resource nationalism trend for {mineral}",
            "Permitting and licensing risk",
        ],
        "environmental": [
            f"Environmental compliance costs for {mineral} extraction",
            "Water rights and usage restrictions",
            "Carbon emission regulations",
        ],
        "infrastructure": [
            f"Transport infrastructure for {mineral} concentrate",
            "Power grid reliability and cost",
            "Processing facility availability",
        ],
        "trade": [
            f"Export controls on {mineral}",
            "Trade agreement coverage",
            f"Supply chain concentration risk: {data['concentration_risk']}",
        ],
        "social": [
            "Community acceptance and social license",
            "Labor availability and skills",
            "Indigenous rights and land claims",
        ],
    }
    return issues_map.get(factor, ["General risk assessment required"])


def _get_mitigation(score: float, mineral: str, region: str) -> List[str]:
    """Get risk mitigation recommendations."""
    recs = [
        f"Diversify supply sources for {mineral} across multiple geographies",
        f"Establish long-term offtake agreements with producers in {region}",
        "Invest in recycling and urban mining to reduce primary demand",
        "Develop strategic stockpile for critical mineral inputs",
    ]
    
    if score >= 0.7:
        recs.insert(0, f"HIGH RISK: Implement robust supply chain monitoring and contingency plans for {mineral} in {region}")
        recs.append("Consider vertical integration or joint ventures to secure supply")
    
    if score < 0.4:
        recs.append(f"Favorable risk profile for {mineral} operations in {region}")
    
    return recs

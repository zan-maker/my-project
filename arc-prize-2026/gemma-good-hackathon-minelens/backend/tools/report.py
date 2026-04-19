"""
MineLens AI - Report Generation Tool
Generates comprehensive mineral prospectivity reports.
"""

from typing import Dict, List, Any


def generate_report(
    location: str,
    mineral_targets: List[str],
    analysis_results: Dict,
    report_type: str = "detailed",
) -> Dict:
    """
    Generate a comprehensive mineral prospectivity report.
    
    Combines results from spectral analysis, terrain classification,
    proximity search, and risk assessment into a structured report.
    """
    
    report = {
        "title": f"Mineral Prospectivity Report — {location}",
        "location": location,
        "target_minerals": mineral_targets,
        "report_type": report_type,
        "generated_by": "MineLens AI (Gemma 4)",
    }
    
    # Executive Summary
    report["executive_summary"] = _generate_executive_summary(
        location, mineral_targets, analysis_results
    )
    
    # Spectral Findings
    if "spectral" in analysis_results:
        report["spectral_analysis"] = {
            "summary": _summarize_spectral(analysis_results["spectral"]),
            "details": analysis_results["spectral"],
        }
    
    # Terrain Classification
    if "terrain" in analysis_results:
        report["terrain_analysis"] = {
            "summary": _summarize_terrain(analysis_results["terrain"]),
            "details": analysis_results["terrain"],
        }
    
    # Proximity Results
    if "proximity" in analysis_results:
        report["proximity_analysis"] = {
            "summary": _summarize_proximity(analysis_results["proximity"]),
            "details": analysis_results["proximity"],
        }
    
    # Risk Assessment
    if "risk" in analysis_results:
        report["risk_assessment"] = {
            "summary": _summarize_risk(analysis_results["risk"]),
            "details": analysis_results["risk"],
        }
    
    # Geological Context
    report["geological_context"] = _generate_geological_context(
        analysis_results
    )
    
    # Prospecting Recommendations
    report["recommendations"] = _generate_recommendations(
        mineral_targets, analysis_results
    )
    
    # Confidence Assessment
    report["confidence_assessment"] = _assess_overall_confidence(
        analysis_results
    )
    
    # Next Steps
    report["next_steps"] = _generate_next_steps(analysis_results)
    
    return report


def _generate_executive_summary(location, minerals, results):
    """Generate executive summary."""
    lines = [
        f"This report presents a comprehensive mineral prospectivity assessment for {location}, "
        f"targeting {', '.join(minerals)}. The analysis integrates spectral signature detection, "
        f"terrain classification, proximity to known deposits, and supply chain risk evaluation "
        f"using the MineLens AI platform powered by Gemma 4.",
    ]
    
    # Count detections
    detections = 0
    if "spectral" in results:
        spectral = results["spectral"].get("mineral_detections", [])
        detections = sum(1 for d in spectral if d.get("detected", False))
    
    if detections > 0:
        lines.append(
            f"Spectral analysis identified {detections} of {len(minerals)} target minerals "
            f"with confidence above the detection threshold, suggesting promising mineralization "
            f"potential in the analyzed area."
        )
    else:
        lines.append(
            f"Spectral analysis did not identify significant signatures for the target minerals "
            f"at this location. However, the absence of spectral indicators does not preclude "
            f"sub-surface mineralization, and further investigation is recommended."
        )
    
    nearby = 0
    if "proximity" in results:
        nearby = results["proximity"].get("total_deposits_found", 0)
    
    if nearby > 0:
        lines.append(
            f"The proximity search identified {nearby} known mineral deposits within the search "
            f"radius, indicating that the broader region has established mineral endowment."
        )
    
    return " ".join(lines)


def _summarize_spectral(spectral_data):
    """Summarize spectral analysis results."""
    detections = spectral_data.get("mineral_detections", [])
    detected = [d for d in detections if d.get("detected")]
    
    if not detected:
        return "No significant mineral spectral signatures detected above the confidence threshold."
    
    summary_parts = []
    for d in detected:
        summary_parts.append(
            f"{d['mineral']} (confidence: {d['confidence']:.0%}, "
            f"setting: {', '.join(d.get('geological_setting', []))})"
        )
    
    return f"Detected: {', '.join(summary_parts)}. " + \
           f"Overall prospectivity: {spectral_data.get('overall_assessment', {}).get('prospectivity_rating', 'Unknown')}."


def _summarize_terrain(terrain_data):
    """Summarize terrain classification."""
    classifications = terrain_data.get("terrain_classifications", [])
    if not classifications:
        return "No terrain classification available."
    
    top = classifications[0]
    return f"Dominant terrain type: {top['terrain_type']} (confidence: {top['confidence']:.0%}). " + \
           f"Mineral association: {', '.join(top.get('mineral_association', []))}. " + \
           f"Prospectivity: {top.get('mineral_prospectivity', 'Unknown')}."


def _summarize_proximity(proximity_data):
    """Summarize proximity search results."""
    n = proximity_data.get("total_deposits_found", 0)
    assessment = proximity_data.get("regional_assessment", {})
    
    if n == 0:
        return "No known deposits found within search radius. Area may be underexplored."
    
    closest = assessment.get("closest_deposit", "N/A")
    return f"Found {n} known deposit(s) nearby. Regional rating: {assessment.get('rating', 'Unknown')}. " + \
           f"Closest: {closest}."


def _summarize_risk(risk_data):
    """Summarize risk assessment."""
    score = risk_data.get("overall_risk_score", 0)
    level = risk_data.get("risk_level", "Unknown")
    return f"Overall supply chain risk: {level} (score: {score:.2f}). " + \
           f"Concentration: {risk_data.get('supply_chain_analysis', {}).get('concentration_risk', 'N/A')}."


def _generate_geological_context(results):
    """Generate geological context section."""
    context = {
        "terrain_setting": "Not available",
        "structural_controls": "Not available",
        "known_mineralization": "Not available",
    }
    
    if "terrain" in results:
        terrain = results["terrain"]
        if terrain.get("dominant_terrain"):
            context["terrain_setting"] = terrain["dominant_terrain"]
    
    if "proximity" in results:
        proximity = results["proximity"]
        mineral_freq = proximity.get("analysis", {}).get("mineral_frequency", {})
        if mineral_freq:
            top_minerals = list(mineral_freq.keys())[:5]
            context["known_mineralization"] = ", ".join(top_minerals)
    
    return context


def _generate_recommendations(minerals, results):
    """Generate prospecting recommendations."""
    recs = []
    
    # Based on spectral results
    if "spectral" in results:
        detected = [d for d in results["spectral"].get("mineral_detections", []) if d.get("detected")]
        for d in detected:
            if d["confidence"] >= 0.7:
                recs.append(f"PRIORITY: Ground-truth {d['mineral']} anomalies with geochemical soil sampling")
    
    # Based on terrain
    if "terrain" in results:
        targets = results["terrain"].get("recommended_investigation_targets", [])
        if targets:
            recs.append(f"Investigate {', '.join(targets)} terrain features in detail")
    
    # General recommendations
    recs.extend([
        "Conduct regional aeromagnetic/electromagnetic survey",
        "Collect stream sediment samples for multi-element analysis",
        "Review historical exploration data and drill logs for the area",
        "Engage with local geological survey for existing data",
    ])
    
    return recs


def _assess_overall_confidence(results):
    """Assess overall confidence in the prospectivity evaluation."""
    confidences = []
    
    if "spectral" in results:
        detections = results["spectral"].get("mineral_detections", [])
        if detections:
            avg = sum(d["confidence"] for d in detections) / len(detections)
            confidences.append(("spectral", avg))
    
    if "terrain" in results:
        classifications = results["terrain"].get("terrain_classifications", [])
        if classifications:
            avg = sum(c["confidence"] for c in classifications) / len(classifications)
            confidences.append(("terrain", avg))
    
    if not confidences:
        return {"overall": 0.0, "note": "Insufficient data for confidence assessment"}
    
    overall = sum(c[1] for c in confidences) / len(confidences)
    
    return {
        "overall": round(overall, 3),
        "breakdown": {name: round(conf, 3) for name, conf in confidences},
        "reliability": "High" if overall >= 0.7 else "Moderate" if overall >= 0.5 else "Low",
    }


def _generate_next_steps(results):
    """Generate recommended next steps."""
    steps = [
        {"priority": "Immediate", "action": "Compile all available geological data for the target area"},
        {"priority": "Immediate", "action": "Secure land access and exploration permits"},
        {"priority": "Short-term (1-3 months)", "action": "Conduct field reconnaissance and geological mapping"},
        {"priority": "Short-term (1-3 months)", "action": "Collect geochemical soil and rock chip samples"},
        {"priority": "Medium-term (3-6 months)", "action": "Commission geophysical survey (magnetic, EM, radiometric)"},
        {"priority": "Medium-term (3-6 months)", "action": "Integrate all data into 3D geological model"},
        {"priority": "Long-term (6-12 months)", "action": "Design and execute drill program based on targets"},
    ]
    
    # Adjust based on results
    if "spectral" in results:
        high_conf = [d for d in results["spectral"].get("mineral_detections", []) if d.get("confidence", 0) >= 0.7]
        if high_conf:
            steps.insert(0, {
                "priority": "URGENT",
                "action": f"Prioritize field verification for {', '.join(d['mineral'] for d in high_conf)}"
            })
    
    return steps

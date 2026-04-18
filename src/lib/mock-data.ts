// MineLens AI — Comprehensive Mock Data Layer
// Simulates realistic geospatial analysis results

import type {
  Mineral,
  MineralInfo,
  SpectralResult,
  TerrainResult,
  ProximityResult,
  NearbyDeposit,
  RiskResult,
  RiskFactor,
  AnalysisResult,
  ToolCall,
  ChatMessage,
  ToolDefinition,
  ProspectivityReport,
} from "./types";

// ============================================================
// Mineral Metadata
// ============================================================

export const MINERALS: Record<Mineral, MineralInfo> = {
  lithium: {
    id: "lithium",
    name: "Lithium",
    symbol: "Li",
    color: "#10b981",
    bgColor: "rgba(16,185,129,0.15)",
    borderColor: "rgba(16,185,129,0.4)",
    icon: "🔋",
    criticalUse: "EV Batteries, Energy Storage",
  },
  cobalt: {
    id: "cobalt",
    name: "Cobalt",
    symbol: "Co",
    color: "#6366f1",
    bgColor: "rgba(99,102,241,0.15)",
    borderColor: "rgba(99,102,241,0.4)",
    icon: "⚡",
    criticalUse: "Battery Cathodes, Superalloys",
  },
  rare_earth: {
    id: "rare_earth",
    name: "Rare Earth",
    symbol: "RE",
    color: "#f59e0b",
    bgColor: "rgba(245,158,11,0.15)",
    borderColor: "rgba(245,158,11,0.4)",
    icon: "🧲",
    criticalUse: "Magnets, Electronics, Defense",
  },
  copper: {
    id: "copper",
    name: "Copper",
    symbol: "Cu",
    color: "#ef4444",
    bgColor: "rgba(239,68,68,0.15)",
    borderColor: "rgba(239,68,68,0.4)",
    icon: "🔌",
    criticalUse: "Electrical Wiring, EVs, Renewables",
  },
  nickel: {
    id: "nickel",
    name: "Nickel",
    symbol: "Ni",
    color: "#8b5cf6",
    bgColor: "rgba(139,92,246,0.15)",
    borderColor: "rgba(139,92,246,0.4)",
    icon: "⚙️",
    criticalUse: "Stainless Steel, EV Batteries",
  },
};

// ============================================================
// Spectral Analysis Mock Generator
// ============================================================

function generateSpectralResults(
  lat: number,
  lon: number,
  targets: Mineral[]
): SpectralResult[] {
  // Use coordinates as seed for deterministic but varied results
  const seed = (lat * 1000 + lon * 100) % 100;
  return targets.map((mineral, i) => {
    const baseConfidence = 0.4 + ((seed + i * 17) % 45) / 100;
    const confidence = Math.min(0.98, baseConfidence + Math.random() * 0.1);
    const anomalyScore = 0.2 + ((seed + i * 23) % 60) / 100;

    const bandNames = [
      "VNIR-1 (0.45μm)",
      "VNIR-2 (0.55μm)",
      "VNIR-3 (0.65μm)",
      "SWIR-1 (1.6μm)",
      "SWIR-2 (2.2μm)",
      "TIR-1 (8.5μm)",
      "TIR-2 (11.5μm)",
    ];

    return {
      mineral,
      confidence: parseFloat(confidence.toFixed(3)),
      anomaly_score: parseFloat(anomalyScore.toFixed(3)),
      wavelength_bands: bandNames.map((band, j) => ({
        band,
        value: parseFloat(
          (0.1 + ((seed + i * 7 + j * 13) % 80) / 100).toFixed(3)
        ),
      })),
    };
  });
}

// ============================================================
// Terrain Classification Mock Generator
// ============================================================

const TERRAIN_TYPES = [
  {
    terrain_type: "Pegmatite Belt",
    formation: "Granitic Pegmatite Intrusion",
    geological_age: "Proterozoic (1.0–1.8 Ga)",
    description:
      "Highly fractionated granitic pegmatite bodies with extensive lithium-bearing mineralization. Multiple dyke swarms cutting across older metamorphic basement.",
    features: [
      "Coarse-grained feldspar-quartz-mica assemblages",
      "Spodumene and lepidolite occurrences",
      "Tourmaline-rich zones",
      "Albite-cleavelandite replacement",
      "Rare-element enriched pocket zones",
    ],
  },
  {
    terrain_type: "Greenstone Belt",
    formation: "Archean Volcanic-Sedimentary Sequence",
    geological_age: "Archean (2.5–3.5 Ga)",
    description:
      "Classic Archean greenstone terrain with bimodal volcanic sequences and associated Banded Iron Formations. Known for VMS-style copper-nickel and gold mineralization.",
    features: [
      "Komatiitic lava flows",
      "Tholeiitic basalt sequences",
      "Banded Iron Formations",
      "Felsic volcanic centers",
      "Sulphide-bearing shear zones",
    ],
  },
  {
    terrain_type: "Porphyry System",
    formation: "Calc-alkaline Porphyry Intrusion",
    geological_age: "Mesozoic (90–180 Ma)",
    description:
      "Multi-phase porphyry copper system with extensive phyllic and potassic alteration halos. Associated epithermal vein systems present at higher elevations.",
    features: [
      "Quartz-monzonite porphyry stock",
      "Phyllic alteration zone",
      "Potassic core with bornite",
      "Propylitic outer halo",
      "A-type quartz veins",
    ],
  },
  {
    terrain_type: "Lateritic Terrain",
    formation: "Tropical Weathering Profile",
    geological_age: "Cenozoic (< 65 Ma)",
    description:
      "Deep lateritic weathering profile developed over ultramafic parent rock. Significant nickel-cobalt enrichment in saprolite and limonite zones.",
    features: [
      "Ferricrete cap",
      "Limonite zone with Mn oxides",
      "Saprolite zone with garnierite",
      "Bedrock serpentinite",
      "Silica boxworks",
    ],
  },
  {
    terrain_type: "Sedimentary Basin",
    formation: "Intracontinental Rift Basin",
    geological_age: "Paleozoic–Mesozoic (250–450 Ma)",
    description:
      "Thick sedimentary sequence with evaporite horizons and associated rare earth element enrichment in carbonatite-related intrusive complexes.",
    features: [
      "Dolomitic carbonate units",
      "Shale-rich intervals",
      "Carbonatite intrusive bodies",
      "Fenite alteration halos",
      "Magnetite-rich breccia pipes",
    ],
  },
];

function generateTerrainResult(lat: number, lon: number): TerrainResult {
  const idx = Math.abs(Math.round(lat * 3 + lon * 7)) % TERRAIN_TYPES.length;
  const terrain = TERRAIN_TYPES[idx];
  return {
    ...terrain,
    elevation_range: {
      min: Math.abs(Math.round(lat * 50)) % 500 + 200,
      max: Math.abs(Math.round(lon * 50)) % 2000 + 800,
    },
  };
}

// ============================================================
// Proximity Search Mock Generator
// ============================================================

const KNOWN_DEPOSITS: NearbyDeposit[] = [
  { name: "Greenbushes Lithium Mine", mineral_type: "lithium", distance_km: 12, status: "Active", last_activity: "2024" },
  { name: "Mount Marion Lithium Project", mineral_type: "lithium", distance_km: 28, status: "Active", last_activity: "2024" },
  { name: "Salar de Atacama Operations", mineral_type: "lithium", distance_km: 45, status: "Active", last_activity: "2024" },
  { name: "Kamoto Copper-Cobalt Mine", mineral_type: "cobalt", distance_km: 34, status: "Active", last_activity: "2024" },
  { name: "Tenke Fungurume Mine", mineral_type: "cobalt", distance_km: 67, status: "Active", last_activity: "2023" },
  { name: "Mutanda Mining Site", mineral_type: "cobalt", distance_km: 82, status: "Care & Maintenance", last_activity: "2022" },
  { name: "Bayan Obo REE Deposit", mineral_type: "rare_earth", distance_km: 55, status: "Active", last_activity: "2024" },
  { name: "Mountain Pass Mine", mineral_type: "rare_earth", distance_km: 38, status: "Active", last_activity: "2024" },
  { name: "Mount Weld REE Project", mineral_type: "rare_earth", distance_km: 73, status: "Active", last_activity: "2024" },
  { name: "Escondida Copper Mine", mineral_type: "copper", distance_km: 19, status: "Active", last_activity: "2024" },
  { name: "Collahuasi Copper Mine", mineral_type: "copper", distance_km: 41, status: "Active", last_activity: "2024" },
  { name: "Grasberg Copper-Gold Mine", mineral_type: "copper", distance_km: 89, status: "Active", last_activity: "2024" },
  { name: "Sudbury Basin Operations", mineral_type: "nickel", distance_km: 52, status: "Active", last_activity: "2024" },
  { name: "Norilsk Nickel Complex", mineral_type: "nickel", distance_km: 95, status: "Active", last_activity: "2024" },
  { name: "Ravensthorpe Nickel Mine", mineral_type: "nickel", distance_km: 22, status: "Active", last_activity: "2023" },
  { name: "Moa Nickel-Cobalt Operation", mineral_type: "nickel", distance_km: 61, status: "Active", last_activity: "2024" },
  { name: "Thacker Pass Lithium Project", mineral_type: "lithium", distance_km: 76, status: "Development", last_activity: "2024" },
  { name: "Jadar Valley Lithium Deposit", mineral_type: "lithium", distance_km: 48, status: "Exploration", last_activity: "2024" },
];

function generateProximityResult(
  lat: number,
  lon: number,
  targets: Mineral[],
  radiusKm: number = 100
): ProximityResult {
  const relevantDeposits = KNOWN_DEPOSITS.filter(
    (d) =>
      targets.includes(d.mineral_type as Mineral) &&
      d.distance_km <= radiusKm
  ).map((d) => ({
    ...d,
    distance_km: Math.max(
      1,
      d.distance_km + Math.round((lat + lon) * 2) % 20 - 10
    ),
  }));

  // Sort by distance
  relevantDeposits.sort((a, b) => a.distance_km - b.distance_km);

  return {
    center: { lat, lon },
    radius_km: radiusKm,
    deposits_found: relevantDeposits.length,
    deposits: relevantDeposits.slice(0, 8),
  };
}

// ============================================================
// Risk Assessment Mock Generator
// ============================================================

function generateRiskResult(
  lat: number,
  lon: number,
  _minerals: Mineral[]
): RiskResult {
  const seed = Math.abs(Math.round(lat * 13 + lon * 7)) % 100;

  const politicalScore = 20 + (seed % 40);
  const environmentalScore = 25 + ((seed + 15) % 35);
  const infrastructureScore = 15 + ((seed + 30) % 45);
  const tradeScore = 30 + ((seed + 45) % 30);

  const factors: RiskFactor[] = [
    {
      factor: "Political Stability",
      score: politicalScore,
      level: politicalScore > 70 ? "high" : politicalScore > 50 ? "medium" : "low",
      description:
        "Regional governance indicators and regulatory framework assessment. Evaluates permitting requirements, fiscal regime stability, and expropriation risk.",
    },
    {
      factor: "Environmental Regulations",
      score: environmentalScore,
      level:
        environmentalScore > 70
          ? "high"
          : environmentalScore > 50
          ? "medium"
          : "low",
      description:
        "Environmental compliance complexity and ESG requirements. Includes water usage restrictions, tailings management mandates, and biodiversity protections.",
    },
    {
      factor: "Infrastructure Access",
      score: infrastructureScore,
      level:
        infrastructureScore > 70
          ? "high"
          : infrastructureScore > 50
          ? "medium"
          : "low",
      description:
        "Road, rail, port, and power infrastructure availability. Considers proximity to grid connections, water sources, and transport corridors.",
    },
    {
      factor: "Trade & Export Controls",
      score: tradeScore,
      level: tradeScore > 70 ? "high" : tradeScore > 50 ? "medium" : "low",
      description:
        "Export restrictions, tariffs, and critical mineral trade policies. Evaluates supply chain resilience and geopolitical supply concentration.",
    },
  ];

  const overall = Math.round(
    factors.reduce((sum, f) => sum + f.score, 0) / factors.length
  );
  const level =
    overall > 70
      ? ("high" as const)
      : overall > 50
      ? ("medium" as const)
      : ("low" as const);

  return { overall_risk: overall, level, factors };
}

// ============================================================
// Full Analysis Mock Generator
// ============================================================

export function generateMockAnalysis(
  lat: number,
  lon: number,
  targets: Mineral[]
): AnalysisResult {
  const spectral = generateSpectralResults(lat, lon, targets);
  const terrain = generateTerrainResult(lat, lon);
  const proximity = generateProximityResult(lat, lon, targets);
  const risk = generateRiskResult(lat, lon, targets);

  // Calculate overall prospectivity
  const avgConfidence =
    spectral.reduce((sum, s) => sum + s.confidence, 0) / spectral.length;
  const depositBonus = Math.min(proximity.deposits_found * 0.05, 0.2);
  const riskPenalty = risk.overall_risk / 500;
  const prospectivity = Math.min(
    0.98,
    Math.max(0.1, avgConfidence * 0.6 + depositBonus - riskPenalty)
  );

  const recommendations = [
    prospectivity > 0.7
      ? "Strong prospectivity — recommend detailed exploration program with surface sampling and geophysical surveys."
      : prospectivity > 0.4
      ? "Moderate prospectivity — recommend targeted geochemical sampling and structural mapping."
      : "Low prospectivity — recommend regional reconnaissance before committing exploration resources.",
  ];

  return {
    location: { lat, lon },
    mineral_targets: targets,
    spectral,
    terrain,
    proximity,
    risk,
    prospectivity_score: parseFloat(prospectivity.toFixed(3)),
    recommendation: recommendations[0],
  };
}

// ============================================================
// Tool Call Mock Generator (simulates Gemma 4 function calling chain)
// ============================================================

export function generateMockToolCallChain(
  lat: number,
  lon: number,
  targets: Mineral[]
): ToolCall[] {
  const now = Date.now();
  const toolSequence = [
    {
      id: "tc-1",
      tool_name: "geological_survey_lookup",
      status: "completed" as const,
      arguments: {
        latitude: lat,
        longitude: lon,
        data_layers: ["geology", "mineral_occurrences", "fault_lines"],
      },
      duration_ms: 1240 + Math.round(lat) % 500,
      timestamp: now,
    },
    {
      id: "tc-2",
      tool_name: "spectral_analysis",
      status: "completed" as const,
      arguments: {
        image_path: `satellite/${lat.toFixed(2)}_${lon.toFixed(2)}_sentinel2.tif`,
        mineral_targets: targets,
        confidence_threshold: 0.6,
      },
      duration_ms: 3420 + Math.round(lon) % 800,
      timestamp: now + 1500,
    },
    {
      id: "tc-3",
      tool_name: "terrain_classifier",
      status: "completed" as const,
      arguments: {
        image_path: `dem/${lat.toFixed(2)}_${lon.toFixed(2)}_srtm30.tif`,
        classification_detail: "detailed",
      },
      duration_ms: 2180 + Math.round(lat + lon) % 600,
      timestamp: now + 5000,
    },
    {
      id: "tc-4",
      tool_name: "proximity_search",
      status: "completed" as const,
      arguments: {
        latitude: lat,
        longitude: lon,
        radius_km: 100,
        mineral_types: targets,
      },
      duration_ms: 890 + Math.round(lon * 2) % 400,
      timestamp: now + 7200,
    },
    {
      id: "tc-5",
      tool_name: "risk_assessment",
      status: "completed" as const,
      arguments: {
        region: `${lat.toFixed(2)}°${lat >= 0 ? "N" : "S"}, ${lon.toFixed(2)}°${lon >= 0 ? "E" : "W"}`,
        mineral_type: targets[0],
        factors: ["political", "environmental", "infrastructure", "trade"],
      },
      duration_ms: 1560 + Math.round(lat * 3) % 700,
      timestamp: now + 8200,
    },
    {
      id: "tc-6",
      tool_name: "generate_report",
      status: "completed" as const,
      arguments: {
        location: `${lat.toFixed(4)}, ${lon.toFixed(4)}`,
        mineral_targets: targets,
        report_type: "detailed",
      },
      duration_ms: 4210 + Math.round(lon * 5) % 1000,
      timestamp: now + 9800,
    },
  ];

  return toolSequence;
}

// ============================================================
// Mock Chat Responses
// ============================================================

const CHAT_RESPONSES: Record<string, string> = {
  default: `Based on the analysis, this location shows interesting geological features. The terrain classification indicates a potential mineral-bearing formation with favorable structural settings.\n\n**Key observations:**\n- The spectral signatures show anomalous reflectance patterns in the SWIR bands\n- Proximity to known deposits suggests shared geological controls\n- The regional structural framework is consistent with mineralization models\n\nI recommend proceeding with detailed ground-truth sampling to validate these remote sensing indicators.`,

  prospectivity: `## Prospectivity Assessment\n\nThe combined analysis yields a **moderate-to-high** prospectivity rating for this target area.\n\n**Positive indicators:**\n1. Spectral anomalies consistent with target mineral signatures\n2. Favorable geological setting with known mineralization controls\n3. Multiple nearby deposits suggesting district-scale potential\n\n**Risk factors:**\n- Infrastructure access may require significant capital investment\n- Environmental permitting timeline could extend 12-18 months\n\n**Next steps:**\n- Conduct detailed geophysical survey (magnetics + EM)\n- Collect surface geochemical samples along structural trends\n- Stakeholder engagement for access agreements`,

  lithium: `## Lithium Exploration Context\n\nThis location falls within a geological setting that has shown potential for **lithium-bearing pegmatites**.\n\n**Exploration model:**\n- Primary target: Spodumene-bearing pegmatite dykes\n- Secondary target: Lithium-enriched clays in weathered zones\n- Pathfinders: Elevated Li, Rb, Cs, Ta in stream sediments\n\n**Historical context:**\nThe region has documented pegmatite occurrences from geological survey mapping. Several abandoned small-scale operations exist within 50km, suggesting historic lithium recognition.`,

  copper: `## Copper Exploration Assessment\n\nThe geological setting at this location is consistent with **porphyry copper** mineralization models.\n\n**Key indicators:**\n- Calc-alkaline intrusive complex mapped in the area\n- Phyllic alteration assemblage detected in spectral analysis\n- Magnetic low coincident with alteration zone\n\n**Exploration strategy:**\n1. Detailed IP/resistivity survey to identify chargeability anomalies\n2. Soil geochemistry grid (Cu, Mo, Au, Ag, Re)\n3. Diamond drilling to test porphyry center`,

  cobalt: `## Cobalt Potential Analysis\n\nCobalt mineralization at this location is most likely associated with **sediment-hosted** or **lateritic** deposits.\n\n**Depositional model:**\n- Co-enrichment with nickel in lateritic profiles over ultramafic basement\n- Potential for stratiform Co-Cu mineralization in reduced sedimentary units\n\n**Geochemical vectors:**\n- Co/Ni ratios > 1.5 suggest primary cobalt enrichment\n- Mn oxide association indicates supergene concentration\n\n**Recommendation:**\nDetailed laterite profiling with systematic sampling at 1m intervals through the weathering profile.`,
};

export function generateMockChatResponse(
  message: string,
  _lat?: number,
  _lon?: number
): string {
  const lower = message.toLowerCase();
  if (lower.includes("prospect") || lower.includes("potential") || lower.includes("score"))
    return CHAT_RESPONSES.prospectivity;
  if (lower.includes("lithium") || lower.includes("li") || lower.includes("battery"))
    return CHAT_RESPONSES.lithium;
  if (lower.includes("copper") || lower.includes("cu") || lower.includes("porphyry"))
    return CHAT_RESPONSES.copper;
  if (lower.includes("cobalt") || lower.includes("co") || lower.includes("laterite"))
    return CHAT_RESPONSES.cobalt;
  return CHAT_RESPONSES.default;
}

// ============================================================
// Mock Prospectivity Report
// ============================================================

export function generateMockReport(
  lat: number,
  lon: number,
  targets: Mineral[],
  analysis: AnalysisResult
): ProspectivityReport {
  const locationStr = `${lat.toFixed(4)}°${lat >= 0 ? "N" : "S"}, ${Math.abs(lon).toFixed(4)}°${lon >= 0 ? "E" : "W"}`;

  return {
    location: { lat, lon },
    mineral_targets: targets,
    executive_summary: `This report presents a comprehensive mineral prospectivity assessment for the target location at ${locationStr}. The analysis integrates spectral remote sensing, terrain classification, proximity to known deposits, and supply chain risk evaluation for the following target minerals: ${targets.map((t) => MINERALS[t].name).join(", ")}.\n\nThe overall prospectivity score is **${(analysis.prospectivity_score * 100).toFixed(1)}%**, indicating ${analysis.prospectivity_score > 0.6 ? "favorable" : "moderate"} conditions for mineral exploration.`,
    spectral_findings: `Spectral analysis of Sentinel-2 and ASTER imagery identified anomalous reflectance patterns consistent with ${targets.map((t) => MINERALS[t].name).join(", ")} mineralization. Key findings include:\n\n${analysis.spectral.map((s) => `- **${MINERALS[s.mineral as Mineral].name}**: ${(s.confidence * 100).toFixed(1)}% confidence, anomaly score ${s.anomaly_score.toFixed(2)}`).join("\n")}\n\nSWIR absorption features at 2.2μm and 2.35μm are particularly noteworthy, indicating potential clay and carbonate mineral association typical of mineralized systems.`,
    terrain_analysis: `The terrain at this location is classified as **${analysis.terrain.terrain_type}** (${analysis.terrain.formation}), of ${analysis.terrain.geological_age} age.\n\nElevation ranges from ${analysis.terrain.elevation_range.min}m to ${analysis.terrain.elevation_range.max}m above sea level. Key geological features include:\n${analysis.terrain.features.map((f) => `- ${f}`).join("\n")}\n\nThis terrain type is known to host significant mineral deposits globally, and the structural setting is consistent with documented mineralization models.`,
    proximity_assessment: `A proximity search within ${analysis.proximity.radius_km}km radius identified **${analysis.proximity.deposits_found}** known deposits and occurrences. The nearest deposits are:\n\n${analysis.proximity.deposits.slice(0, 5).map((d) => `- **${d.name}** (${d.mineral_type}) — ${d.distance_km}km, ${d.status}`).join("\n")}\n\nThe clustering of known deposits in this region suggests strong geological controls and district-scale mineral potential.`,
    risk_evaluation: `The overall supply chain risk score is **${analysis.risk.overall_risk}/100** (${analysis.risk.level} risk).\n\n${analysis.risk.factors.map((f) => `**${f.factor}**: ${f.score}/100 (${f.level}) — ${f.description}`).join("\n\n")}`,
    recommendations: [
      "Conduct detailed geophysical survey program (magnetics, EM, gravity)",
      "Implement systematic surface geochemical sampling along structural trends",
      "Engage local stakeholders early for access and environmental permitting",
      "Consider phased exploration approach to manage risk and capital allocation",
      "Initiate baseline environmental and social impact assessment",
    ],
    confidence: analysis.prospectivity_score,
    generated_at: new Date().toISOString(),
  };
}

// ============================================================
// Tool Definitions (matching backend schema)
// ============================================================

export const TOOL_DEFINITIONS: ToolDefinition[] = [
  {
    type: "function",
    function: {
      name: "spectral_analysis",
      description:
        "Analyze satellite imagery to identify mineral spectral signatures. Detects anomalies in reflectance patterns that indicate presence of critical minerals.",
      parameters: {
        type: "object",
        properties: {
          image_path: { type: "string", description: "Path to satellite image" },
          mineral_targets: {
            type: "array",
            items: { type: "string" },
            description: "Target minerals to detect",
          },
          confidence_threshold: {
            type: "number",
            description: "Min confidence (0-1)",
            default: 0.6,
          },
        },
        required: ["image_path", "mineral_targets"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "terrain_classifier",
      description:
        "Classify terrain types from satellite or elevation data. Identifies geological formations associated with mineral deposits.",
      parameters: {
        type: "object",
        properties: {
          image_path: { type: "string", description: "Path to DEM image" },
          classification_detail: {
            type: "string",
            enum: ["basic", "detailed", "expert"],
            default: "detailed",
          },
        },
        required: ["image_path"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "proximity_search",
      description:
        "Search for known mineral deposits, mines, and geological features near a given location using USGS MRDS and other public databases.",
      parameters: {
        type: "object",
        properties: {
          latitude: { type: "number", description: "Latitude" },
          longitude: { type: "number", description: "Longitude" },
          radius_km: {
            type: "number",
            description: "Search radius in km",
            default: 50,
          },
          mineral_types: {
            type: "array",
            items: { type: "string" },
            description: "Filter by mineral types",
          },
        },
        required: ["latitude", "longitude"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "risk_assessment",
      description:
        "Assess geopolitical and supply chain risks for critical mineral operations in a region.",
      parameters: {
        type: "object",
        properties: {
          region: { type: "string", description: "Region or country name" },
          mineral_type: { type: "string", description: "Mineral of interest" },
          factors: {
            type: "array",
            items: { type: "string" },
            description: "Risk factors: political, environmental, infrastructure, trade, social",
          },
        },
        required: ["region", "mineral_type"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "generate_report",
      description:
        "Generate a comprehensive mineral prospectivity report based on analysis results.",
      parameters: {
        type: "object",
        properties: {
          location: { type: "string", description: "Location being analyzed" },
          mineral_targets: {
            type: "array",
            items: { type: "string" },
            description: "Target minerals",
          },
          analysis_results: {
            type: "object",
            description: "Combined results from other analysis tools",
          },
          report_type: {
            type: "string",
            enum: ["executive", "detailed", "technical"],
            default: "detailed",
          },
        },
        required: ["location", "mineral_targets", "analysis_results"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "geological_survey_lookup",
      description:
        "Look up geological survey data for a region from USGS and state geological surveys.",
      parameters: {
        type: "object",
        properties: {
          latitude: { type: "number", description: "Latitude" },
          longitude: { type: "number", description: "Longitude" },
          data_layers: {
            type: "array",
            items: { type: "string" },
            description: "Layers: geology, mineral_occurrences, fault_lines, geochemistry, magnetics, gravity",
          },
        },
        required: ["latitude", "longitude"],
      },
    },
  },
];

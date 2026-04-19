// MineLens AI Types

export type Mineral = "lithium" | "cobalt" | "rare_earth" | "copper" | "nickel";

export interface Location {
  lat: number;
  lon: number;
}

export interface SpectralResult {
  mineral: string;
  confidence: number;
  anomaly_score: number;
  wavelength_bands: { band: string; value: number }[];
}

export interface TerrainResult {
  terrain_type: string;
  formation: string;
  elevation_range: { min: number; max: number };
  geological_age: string;
  description: string;
  features: string[];
}

export interface NearbyDeposit {
  name: string;
  mineral_type: string;
  distance_km: number;
  status: string;
  last_activity: string;
}

export interface ProximityResult {
  center: Location;
  radius_km: number;
  deposits_found: number;
  deposits: NearbyDeposit[];
}

export interface RiskFactor {
  factor: string;
  score: number; // 0-100, higher = more risky
  level: "low" | "medium" | "high" | "critical";
  description: string;
}

export interface RiskResult {
  overall_risk: number;
  level: "low" | "medium" | "high" | "critical";
  factors: RiskFactor[];
}

export interface AnalysisResult {
  location: Location;
  mineral_targets: Mineral[];
  spectral: SpectralResult[];
  terrain: TerrainResult;
  proximity: ProximityResult;
  risk: RiskResult;
  prospectivity_score: number;
  recommendation: string;
}

export interface ToolCall {
  id: string;
  tool_name: string;
  status: "pending" | "running" | "completed" | "error";
  arguments: Record<string, unknown>;
  result?: unknown;
  duration_ms?: number;
  timestamp: number;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  tool_calls?: ToolCall[];
  timestamp: number;
}

export interface ToolDefinition {
  type: "function";
  function: {
    name: string;
    description: string;
    parameters: {
      type: string;
      properties: Record<string, unknown>;
      required?: string[];
    };
  };
}

export interface ProspectivityReport {
  location: Location;
  mineral_targets: Mineral[];
  executive_summary: string;
  spectral_findings: string;
  terrain_analysis: string;
  proximity_assessment: string;
  risk_evaluation: string;
  recommendations: string[];
  confidence: number;
  generated_at: string;
}

export interface MineralInfo {
  id: Mineral;
  name: string;
  symbol: string;
  color: string;
  bgColor: string;
  borderColor: string;
  icon: string;
  criticalUse: string;
}

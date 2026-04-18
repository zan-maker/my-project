import { NextResponse } from "next/server";
import { generateMockAnalysis, generateMockToolCallChain, MINERALS } from "@/lib/mock-data";
import type { Mineral } from "@/lib/types";

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { tool_name, arguments: args } = body;

    if (!tool_name) {
      return NextResponse.json({ error: "Tool name is required" }, { status: 400 });
    }

    // Simulate tool execution delay
    await new Promise((resolve) => setTimeout(resolve, 800 + Math.random() * 2000));

    let result: Record<string, unknown>;

    switch (tool_name) {
      case "spectral_analysis":
        result = {
          detections: (args.mineral_targets as Mineral[] || []).map((m: Mineral) => ({
            mineral: m,
            confidence: 0.5 + Math.random() * 0.45,
            anomaly_score: 0.3 + Math.random() * 0.6,
          })),
          image_processed: args.image_path || "synthetic.tif",
          processing_time_ms: 1200 + Math.round(Math.random() * 800),
        };
        break;

      case "terrain_classifier":
        result = {
          terrain_type: "Pegmatite Belt",
          formation: "Granitic Pegmatite Intrusion",
          confidence: 0.85,
          features: ["Coarse-grained feldspar", "Spodumene occurrences", "Tourmaline zones"],
        };
        break;

      case "proximity_search":
        result = {
          deposits_found: 3 + Math.floor(Math.random() * 8),
          radius_km: args.radius_km || 50,
          nearest_deposit: "Greenbushes Lithium Mine",
          nearest_distance_km: 12 + Math.round(Math.random() * 40),
        };
        break;

      case "risk_assessment":
        result = {
          overall_risk: 25 + Math.round(Math.random() * 50),
          level: ["low", "medium", "high"][Math.floor(Math.random() * 3)],
          factors: ["political", "environmental", "infrastructure", "trade"],
        };
        break;

      case "generate_report": {
        const lat = typeof args.location === "string" ? 0 : 0;
        const targets = (args.mineral_targets || []) as Mineral[];
        const analysis = generateMockAnalysis(lat, 0, targets.length > 0 ? targets : ["lithium", "copper"]);
        result = {
          report_id: `RPT-${Date.now()}`,
          status: "generated",
          summary: `Prospectivity report for ${targets.map((t) => MINERALS[t]?.name || t).join(", ")}. Score: ${(analysis.prospectivity_score * 100).toFixed(1)}%`,
          pages: 12,
        };
        break;
      }

      case "geological_survey_lookup":
        result = {
          region: `${args.latitude}, ${args.longitude}`,
          rock_types: ["Granite", "Gneiss", "Schist", "Pegmatite"],
          formations: ["Yilgarn Craton", "Pilbara Supergroup"],
          known_occurrences: 2 + Math.floor(Math.random() * 5),
        };
        break;

      default:
        return NextResponse.json({ error: `Unknown tool: ${tool_name}` }, { status: 400 });
    }

    return NextResponse.json({ tool: tool_name, result, timestamp: Date.now() });
  } catch {
    return NextResponse.json({ error: "Tool execution failed" }, { status: 500 });
  }
}

// Also expose tool call chain simulation
export async function GET() {
  const chain = generateMockToolCallChain(-28.05, 120.35, ["lithium", "cobalt"]);
  return NextResponse.json({ tool_calls: chain });
}

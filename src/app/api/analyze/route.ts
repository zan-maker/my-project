import { NextResponse } from "next/server";
import { generateMockAnalysis } from "@/lib/mock-data";
import type { Mineral } from "@/lib/types";

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { latitude, longitude, mineral_targets } = body;

    const lat = parseFloat(latitude);
    const lon = parseFloat(longitude);
    const targets: Mineral[] = mineral_targets || ["lithium", "cobalt", "rare_earth", "copper", "nickel"];

    if (isNaN(lat) || isNaN(lon) || lat < -90 || lat > 90 || lon < -180 || lon > 180) {
      return NextResponse.json({ error: "Invalid coordinates" }, { status: 400 });
    }

    // Simulate processing delay
    await new Promise((resolve) => setTimeout(resolve, 1500));

    const result = generateMockAnalysis(lat, lon, targets);
    return NextResponse.json(result);
  } catch {
    return NextResponse.json({ error: "Analysis failed" }, { status: 500 });
  }
}

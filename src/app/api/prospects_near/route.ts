import { NextResponse } from "next/server";
import { generateMockAnalysis } from "@/lib/mock-data";
import type { Mineral } from "@/lib/types";

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { latitude, longitude, radius_km = 100, mineral_type } = body;

    const lat = parseFloat(latitude);
    const lon = parseFloat(longitude);

    if (isNaN(lat) || isNaN(lon)) {
      return NextResponse.json({ error: "Invalid coordinates" }, { status: 400 });
    }

    await new Promise((resolve) => setTimeout(resolve, 600));

    const targets: Mineral[] = mineral_type
      ? [mineral_type]
      : ["lithium", "cobalt", "rare_earth", "copper", "nickel"];

    const analysis = generateMockAnalysis(lat, lon, targets);

    return NextResponse.json({
      center: { lat, lon },
      radius_km,
      prospects: analysis.proximity,
    });
  } catch {
    return NextResponse.json({ error: "Search failed" }, { status: 500 });
  }
}

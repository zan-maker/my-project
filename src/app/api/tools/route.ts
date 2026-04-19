import { NextResponse } from "next/server";
import { TOOL_DEFINITIONS } from "@/lib/mock-data";

export async function GET() {
  return NextResponse.json({ tools: TOOL_DEFINITIONS });
}

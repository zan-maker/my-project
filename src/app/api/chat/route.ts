import { NextResponse } from "next/server";
import { generateMockChatResponse, generateMockToolCallChain } from "@/lib/mock-data";

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { message, latitude, longitude } = body;

    if (!message || typeof message !== "string") {
      return NextResponse.json({ error: "Message is required" }, { status: 400 });
    }

    // Simulate processing delay
    await new Promise((resolve) => setTimeout(resolve, 1000 + Math.random() * 1500));

    const response = generateMockChatResponse(message, latitude, longitude);
    const toolCalls = latitude && longitude
      ? generateMockToolCallChain(latitude, longitude, ["lithium", "cobalt", "rare_earth", "copper", "nickel"]).slice(0, 3)
      : [];

    return NextResponse.json({
      response,
      tool_calls: toolCalls,
      timestamp: Date.now(),
    });
  } catch {
    return NextResponse.json({ error: "Chat failed" }, { status: 500 });
  }
}

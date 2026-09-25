import { NextResponse } from "next/server";
import { models } from "@/lib/dashboard-data";

export async function GET() {
  return NextResponse.json({ data: models, total: models.length, generatedAt: new Date().toISOString() });
}

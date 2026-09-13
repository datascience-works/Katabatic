import { NextResponse } from "next/server";
import { datasets } from "@/lib/dashboard-data";

export async function GET() {
  return NextResponse.json({ data: datasets, total: datasets.length, generatedAt: new Date().toISOString() });
}

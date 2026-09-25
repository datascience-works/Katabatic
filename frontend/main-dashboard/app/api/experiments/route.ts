import { NextRequest, NextResponse } from "next/server";
import { experiments } from "@/lib/dashboard-data";

export async function GET(request: NextRequest) {
  const status = request.nextUrl.searchParams.get("status");
  const data = status ? experiments.filter((item) => item.status.toLowerCase() === status.toLowerCase()) : experiments;
  return NextResponse.json({ data, total: data.length, generatedAt: new Date().toISOString() });
}

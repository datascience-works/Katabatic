import { NextResponse } from "next/server";
import { activity } from "@/lib/dashboard-data";

export async function GET() {
  return NextResponse.json({ data: activity, total: activity.length, generatedAt: new Date().toISOString() });
}

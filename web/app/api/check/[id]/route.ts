import { NextRequest, NextResponse } from 'next/server';

export async function GET(_: NextRequest, { params }): Promise<NextResponse> {
  const id = params.id;

  const response = await fetch(`http://results/${id}`);
  const data = await response.json();

  return NextResponse.json(data);
}

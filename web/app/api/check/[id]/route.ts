import { NextRequest, NextResponse } from 'next/server';

export async function GET(_: NextRequest, { params }): Promise<NextResponse> {
  const id = params.id;

  try {
    const response = await fetch(`http://results/${id}`, { cache: 'no-store' });
    const data = await response.json();

    return NextResponse.json(data);
  } catch (e) {
    return NextResponse.json({ success: false });
  }
}

import { NextRequest, NextResponse } from 'next/server';

export async function POST(req: NextRequest): Promise<NextResponse> {
  const formData = await req.formData();

  try {
    const response = await fetch(`http://upload`, {
      method: 'POST',
      body: formData,
      cache: 'no-cache'
    });

    const data = await response.json();

    return NextResponse.json(data);
  } catch (e) {
    return NextResponse.json({ success: false });
  }
}

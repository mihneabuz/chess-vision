import { NextRequest, NextResponse } from 'next/server';

import { publishFile } from 'lib/external';

export async function POST(req: NextRequest): Promise<NextResponse> {
  const formData = await req.formData();
  const image = formData.get('image');

  if (!image || typeof image === 'string') {
    return new NextResponse(null, { status: 400 });
  }

  let id: string | undefined;

  try {
    id = await publishFile(image);
  } catch (e) {
    console.warn(e);
  }

  return NextResponse.json({
    success: id ? true : false,
    id,
  });
}

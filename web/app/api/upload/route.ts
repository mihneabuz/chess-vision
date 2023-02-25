import { NextRequest, NextResponse } from 'next/server';
import { v4 as uuid } from 'uuid';

async function sendToFileServer(image: File, name: string) {
  const formData = new FormData();
  formData.append('file', image);

  const token = process.env.FILE_SERVER_TOKEN;
  const res = await fetch(`http://file-server/files/${name}?token=${token}`, {
    method: 'PUT',
    body: formData,
  });

  return await res.json();
}

export async function POST(req: NextRequest): Promise<NextResponse> {
  const formData = await req.formData();
  const image = formData.get('image');

  if (!image || typeof image === 'string') {
    return new NextResponse(null, { status: 400 });
  }

  const id = uuid();
  const fileServerRes = await sendToFileServer(image, `image_${id}`);
  console.log(fileServerRes);

  return NextResponse.json({
    success: true,
    id,
  });
}

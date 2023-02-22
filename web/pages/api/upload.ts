import { NextApiRequest, NextApiResponse } from 'next';
import { Fields, Files } from 'formidable';

import { readFile } from "node:fs/promises";
import formidable from 'formidable';
import { v4 as uuid } from 'uuid';

type Form = { fields: Fields, files: Files };

async function parseForm(req: NextApiRequest): Promise<Form> {
  return new Promise((resolve, reject) => {
    const form = formidable();

    form.parse(req, (err, fields, files) => {
      if (err) reject({ err });
      resolve({ fields, files });
    });
  });
}

async function sendToFileServer(path: string, name: string) {
  const formData = new FormData();
  formData.append('file', new Blob([await readFile(path)]));

  const token = process.env.FILE_SERVER_TOKEN;
  const res = await fetch(`http://file-server/files/${name}?token=${token}`, {
    method: 'PUT',
    body: formData
  });

  return await res.json();
}

async function handler(req: NextApiRequest, res: NextApiResponse) {
  const { files } = await parseForm(req);

  const imagePath = (files.image as formidable.File).filepath;
  const id = uuid();

  const fileServerRes = await sendToFileServer(imagePath, `image_${id}`);

  console.log(fileServerRes);

  res.status(200).json({
    success: true,
    id
  });
}

export const config = {
  api: {
    bodyParser: false,
  },
};

export default handler;

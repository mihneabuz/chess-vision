import { v4 as uuid } from 'uuid';
import amqplib from 'amqplib';
import { getConfig } from './config';
import md5 from 'md5';

const config = getConfig();

async function sendFile(file: File, name: string) {
  const config = getConfig();
  const formData = new FormData();
  formData.append('file', file);

  const url = new URL(`http://${config.fileServer}/files/${name}`);
  url.searchParams.append('token', config.fileServerToken);

  const res = await fetch(url, {
    method: 'PUT',
    body: formData,
  });

  return await res.json();
}

async function publishId(id: string, hash: string) {
  const connection = await amqplib.connect(`amqp://${config.messageBroker}`);
  const channel = await connection.createChannel();
  await channel.assertQueue(config.messageQueue, { durable: false });

  const message = JSON.stringify({
    id,
    hash,
    data: '',
  });

  console.log(message);
  channel.sendToQueue(config.messageQueue, Buffer.from(message));

  await channel.close();
  await connection.close();
}

export async function publishFile(file: File) {
  const id = uuid();
  const name = `image_${id}`;

  const fileServerResponse = await sendFile(file, name);
  if (!fileServerResponse || !fileServerResponse.ok) {
    const responseString = JSON.stringify(fileServerResponse);
    throw new Error('Could not send file to server: ' + responseString);
  }

  const hash = md5(Buffer.from(await file.arrayBuffer()));
  await publishId(id, hash);

  return id;
}

export const runtime = 'nodejs';

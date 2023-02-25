interface Config {
  fileServer: string;
  fileServerToken: string;
  messageQueue: string;
  messageBroker: string;
}

export function getConfig(): Config {
  const fileServer = process.env.FILE_SERVER || "";
  const fileServerToken = process.env.FILE_SERVER_TOKEN || "";

  const messageBroker = process.env.MESSAGE_BROKER || "";
  const messageQueue = process.env.MESSAGE_QUEUE || "";

  return {
    fileServer, fileServerToken, messageBroker, messageQueue
  };
}

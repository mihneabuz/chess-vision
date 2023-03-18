'use client';

import { MouseEventHandler, ChangeEventHandler, useEffect } from 'react';
import { useRef, useState } from 'react';

interface Upload {
  type: 'upload'
};
interface Waiting {
  type: 'waiting',
  id: string
};
interface Result {
  type: 'result',
  pieces: number[]
};

type State = Upload | Waiting | Result;

export default function Home() {
  const [state, setState] = useState<State>({ type: 'upload' });

  useEffect(() => {
    if (state.type === 'waiting') {
      const interval = setInterval(async () => {
        const req = await fetch(`/api/check/${state.id}`);
        const result = await req.json();

        if (result.success) {
          setState({ type: 'result', pieces: result.pieces });
        }
      }, 500);

      return () => clearInterval(interval);
    }
  }, [state]);

  const handleUpload = (id: string) => {
    setState({ type: 'waiting', id });
  };

  return (
    <div className="w-full flex flex-wrap items-center justify-center">
      <div className="flex-grow bg-red-700 h-fit min-w-fit">
        <Upload onUpload={handleUpload} canUpload={state.type === 'upload'} />
      </div>

      <div className="flex-grow bg-green-800 h-fit min-w-fit">
      {
        state.type === 'upload' &&
          <div className="text-center w-full">
            Take a picture of your game and we'll tell you what to play
          </div>
      }

      {
        state.type === 'waiting' &&
          <Progress message={'Processing'}/>
      }

      {
        state.type === 'result' &&
        <Result pieces={state.type === 'result' ? state.pieces : []} />
      }
      </div>
    </div>
  );
}

function Upload({ onUpload, canUpload }) {
  const [imageFile, setImageFile] = useState<File | null>(null);

  const handleClick: MouseEventHandler<HTMLButtonElement> = async (e) => {
    e.preventDefault();

    if (imageFile) {
      const formData = new FormData();
      formData.append('image', imageFile);

      const req = await fetch('/api/upload', {
        method: 'POST',
        body: formData
      });

      const result = await req.json();

      if (result.success) {
        onUpload(result.id);
      }
    }
  };

  const handleChange: ChangeEventHandler<HTMLInputElement> = (e) => {
    e.preventDefault();

    const file = e.target.files?.item(0);
    setImageFile(file || null);

    if (file && imagePreview.current) {
      imagePreview.current.src = URL.createObjectURL(file);
    }
  };

  const imagePreview = useRef<HTMLImageElement>(null);

  return (
    <form className="flex flex-col items-center">
      <input type="file" accept="image/*" onChange={handleChange} disabled={!canUpload}/>
      <img className="aspect-auto" ref={imagePreview} />
      <button onClick={handleClick} disabled={!canUpload || imageFile === null}>
        Upload image
      </button>
    </form>
  )
}

function Progress({ message }) {
  return (
    <div>
      <span>{message}</span>
    </div>
  );
}

function Result({ pieces }) {
  return (
    <div>
      <span>{pieces}</span>
    </div>
  );
}

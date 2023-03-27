'use client';

import { MouseEventHandler, ChangeEventHandler, useEffect, useMemo } from 'react';
import { useState } from 'react';
import { useEaseIn } from './hooks';

interface Upload {
  type: 'upload';
}
interface Waiting {
  type: 'waiting';
  id: string;
}
interface Result {
  type: 'result';
  pieces: number[];
}

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
    <div className="flex w-full flex-wrap items-center justify-center gap-8 p-8">
      <div className="h-fit max-w-2xl grow bg-red-700">
        <Upload onUpload={handleUpload} canUpload={state.type === 'upload'} />
      </div>

      <div className="grow ">
        {state.type === 'upload' && <Greeting />}

        {state.type === 'waiting' && <Progress message={'Processing'} />}

        {state.type === 'result' && <Result pieces={state.type === 'result' ? state.pieces : []} />}
      </div>
    </div>
  );
}

function Greeting() {
  const text1 = useEaseIn<HTMLSpanElement>(200);
  const text2 = useEaseIn<HTMLSpanElement>(2000);
  const emoji = useEaseIn<HTMLSpanElement>(3000);

  return (
    <div className="flex flex-row items-center justify-center">
      <div className="inline-block text-center">
        <span className="pl-20 transition-opacity duration-600 opacity-0" ref={text1}>
          Take a picture of your game... <br/>
        </span>
        <span className="pl-20 transition-opacity duration-600 opacity-0" ref={text2}>
          ...and we'll tell you what to play!
        </span>
      </div>
      <span className="pl-4 text-5xl transition-opacity duration-600 opacity-0" ref={emoji}>
        😎
      </span>
    </div>
  );
}

function Upload({ onUpload, canUpload }) {
  const [imageFile, setImageFile] = useState<File | null>(null);
  const imagePreview = useMemo(() => {
    if (imageFile === null) return null;

    return URL.createObjectURL(imageFile);
  }, [imageFile]);

  const handleClick: MouseEventHandler<HTMLButtonElement> = async (e) => {
    e.preventDefault();

    if (imageFile) {
      const formData = new FormData();
      formData.append('image', imageFile);

      const req = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
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
  };

  return (
    <form className="flex flex-col items-center">
      <input
        className="rounded bg-kashmir-800 py-1 px-4 hover:bg-kashmir-700"
        type="file"
        accept="image/*"
        onChange={handleChange}
        disabled={!canUpload}
      />

      {imagePreview !== null && <img className="my-8 aspect-auto" src={imagePreview} />}

      {imageFile !== null && (
        <button
          className="rounded bg-kashmir-800 py-1 px-4 hover:bg-kashmir-700"
          onClick={handleClick}
          disabled={!canUpload || imageFile === null}
        >
          Upload image
        </button>
      )}
    </form>
  );
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

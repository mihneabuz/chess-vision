'use client';

import { MouseEventHandler, ChangeEventHandler, useEffect, useMemo } from 'react';
import { useState } from 'react';
import Board from 'app/components/Board';
import { useEaseIn } from 'app/hooks';

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
    <div className="h-full flex w-full flex-wrap items-center justify-around gap-12 p-8">
      <div className="max-w-2xl grow shrink">
        <Upload onUpload={handleUpload} canUpload={state.type !== 'waiting'} />
      </div>

      <div className="max-w-2xl grow shrink">
        {state.type === 'upload' && <Greeting />}

        {state.type === 'waiting' && <Progress message={'Processing'} />}

        {state.type === 'result' && <Board initial={state.type === 'result' ? state.pieces : []} />}
      </div>
    </div>
  );
}

function Greeting() {
  const text1 = useEaseIn<HTMLSpanElement>(100);
  const text2 = useEaseIn<HTMLSpanElement>(2000);
  const emoji = useEaseIn<HTMLSpanElement>(3000);

  return (
    <div className="flex flex-row items-center justify-center text-xl">
      <div className="inline-block text-center">
        <span className="transition-opacity duration-700 opacity-0" ref={text1}>
          Take a picture of your game... <br />
        </span>
        <span className="pl-20 transition-opacity duration-700 opacity-0" ref={text2}>
          ...and we&apos;ll tell you what to play!
        </span>
      </div>
      <span className="pl-4 text-6xl transition-opacity duration-1000 opacity-0" ref={emoji}>
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
      <label
        htmlFor="upload"
        className="rounded bg-kashmir-800 py-1 px-4 hover:bg-kashmir-700 cursor-pointer"
      >
        Select image
      </label>
      <input type="file" name="upload" id="upload" className="hidden" onChange={handleChange} />

      {imagePreview && (
        <img className="my-8 aspect-auto border-4 border-kashmir-800 rounded" src={imagePreview} />
      )}

      {imageFile && (
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
    <div className="flex flex-row justify-center items-center">
      <Spinner />
      <span className="text-2xl">{message}</span>
      <Dots />
    </div>
  );
}

function Spinner() {
  return <div className="w-2 h-12 mx-12 bg-kashmir-100 animate-spin" />;
}

function Dots() {
  const [count, setCount] = useState<number>(0);

  useEffect(() => {
    const timeout = setTimeout(() => {
      setCount((count + 1) % 8);
      return () => clearTimeout(timeout);
    }, 400);
  }, [count]);

  return <span className="w-12 pl-2 text-2xl"> {Array(Math.min(count, 3)).fill('.')} </span>;
}

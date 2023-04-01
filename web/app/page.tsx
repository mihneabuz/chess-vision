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
    <div className="flex min-h-[64rem] flex-wrap items-center justify-around gap-12 p-8">
      <div className="max-w-2xl shrink grow">
        <Upload onUpload={handleUpload} canUpload={state.type !== 'waiting'} />
      </div>

      <div className="max-w-2xl shrink grow">
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
        <span className="opacity-0 transition-opacity duration-700" ref={text1}>
          Take a picture of your game... <br />
        </span>
        <span className="pl-20 opacity-0 transition-opacity duration-700" ref={text2}>
          ...and we&apos;ll tell you what to play!
        </span>
      </div>
      <span className="pl-4 text-6xl opacity-0 transition-opacity duration-1000" ref={emoji}>
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
        className="cursor-pointer rounded bg-kashmir-800 py-1 px-4 text-xl
                   transition-transform hover:scale-95 hover:bg-kashmir-700"
      >
        Select image
      </label>
      <input type="file" name="upload" id="upload" className="hidden" onChange={handleChange} />

      <div className="min-w-[32rem]">
        {imagePreview && (
          <img
            className="my-8 aspect-auto rounded border-4 border-kashmir-800"
            src={imagePreview}
          />
        )}
      </div>

      {imageFile && (
        <button
          className="rounded bg-kashmir-800 py-1 px-4 text-xl
                     transition-transform hover:scale-95 hover:bg-kashmir-700"
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
    <div className="flex min-w-[32rem] flex-row items-center justify-center">
      <Spinner />
      <span className="text-2xl">{message}</span>
      <Dots />
    </div>
  );
}

function Spinner() {
  return <div className="mx-12 h-12 w-2 animate-spin bg-kashmir-100" />;
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

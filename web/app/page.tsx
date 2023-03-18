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
    <div className="grid h-full place-items-center">
      <Upload onUpload={handleUpload} canUpload={state.type === 'upload'}/>
      <Progress message={'Processing'} hidden={state.type !== 'waiting'}/>
      <Result pieces={state.type === 'result' ? state.pieces : []} hidden={state.type !== 'result'}/>
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
      <img className="aspect-auto w-3/4" ref={imagePreview} />
      <button onClick={handleClick} disabled={!canUpload || imageFile === null}>
        Upload image
      </button>
    </form>
  )
}

function Progress({ message, hidden }) {
  return (
    <div hidden={hidden}>
      <span>{message}</span>
    </div>
  );
}

function Result({ pieces, hidden }) {
  return (
    <div hidden={hidden}>
      <span>{pieces}</span>
    </div>
  );
}

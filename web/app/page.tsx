'use client';

import { MouseEventHandler, ChangeEventHandler } from 'react';
import { useRef, useState } from 'react';

export default function Home() {
  const [imageFile, setImageFile] = useState<File | null>(null);
  const imagePreview = useRef<HTMLImageElement>(null);

  const canUpload = imageFile !== null;

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

      console.log(result);
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

  return (
    <div className="grid h-full place-items-center">
      <form className="flex flex-col items-center">
        <input type="file" accept="image/*" onChange={handleChange} />
        <img className="aspect-auto w-3/4" ref={imagePreview} />
        <button onClick={handleClick} disabled={!canUpload}>
          Upload image
        </button>
      </form>
    </div>
  );
}

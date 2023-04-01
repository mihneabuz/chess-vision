'use client';

import { useCallback, useEffect, useState } from 'react';
import { useSounds } from 'app/hooks';
import { pieces, startPosition } from 'app/pieces/util';

export default function Board({ initial }) {
  const [pieces, setPieces] = useState<number[]>(initial || startPosition);
  const [picker, setPicker] = useState<[number, number, number] | null>(null);
  const sounds = useSounds();

  const handleClick = (e: MouseEvent, index: number) => {
    setPicker((old) => (old ? null : [e.clientX, e.clientY + window.scrollY, index]));
  };

  const handlePick = (type: number) => {
    if (!picker) return;
    const square = picker[2];
    sounds.thock1.play();
    setPieces((pieces) => pieces.map((val, idx) => (idx === square ? type : val)));
    setPicker(null);
  };

  const handler = async (black: boolean) => {
    const res = await fetch('/api/generate', {
      method: 'POST',
      body: JSON.stringify({
        pieces,
        black,
      }),
    });

    const data = await res.json();

    console.log(data);
  };

  return (
    <div>
      <div className="h-8"></div>

      <div className="my-8 grid grid-cols-8 rounded border-4 border-kashmir-800">
        {pieces.map((type, index) => (
          <Piece key={`p${index}`} index={index} type={type} onClick={handleClick} />
        ))}
      </div>

      <div className="flex h-8 flex-row items-center justify-center gap-8">
        <button
          className="cursor-pointer rounded bg-kashmir-800 py-1 px-4 text-xl
                     transition-transform hover:scale-95 hover:bg-kashmir-700"
          onClick={() => handler(false)}
        >
          I'm playing white
        </button>
        <button
          onClick={() => handler(true)}
          className="cursor-pointer rounded bg-kashmir-800 py-1 px-4 text-xl
                     transition-transform hover:scale-95 hover:bg-kashmir-700"
        >
          I'm playing black
        </button>
      </div>

      <Picker onClick={handlePick} x={picker && picker[0]} y={picker && picker[1]} />
    </div>
  );
}

function Piece({ index, type, onClick }) {
  const color = (index + Math.floor(index / 8)) % 2 === 0 ? 'bg-kashmir-200' : 'bg-kashmir-500';
  const piece = pieces.get(type);

  return (
    <div
      className={`flex aspect-square min-w-[4rem] items-center justify-center ${color}`}
      onClick={(e) => onClick(e, index)}
    >
      {piece ? (
        <img className="w-full opacity-80" src={piece.src} />
      ) : (
        <div className="border-none" />
      )}
    </div>
  );
}

function Picker({ x, y, onClick }) {
  const [offset, setOffset] = useState<number>(window.scrollY);
  const handleScroll = useCallback(() => setOffset(window.scrollY), []);

  const hidden = !x || !y;

  const style = {
    position: 'fixed',
    left: `${Math.floor(x - 96)}px`,
    top: `${Math.floor(y - offset)}px`,
    opacity: hidden ? '0' : '95',
    display: hidden ? 'none' : undefined,
  };

  useEffect(() => {
    setOffset(window.scrollY);

    if (hidden) return;

    window.addEventListener('scroll', handleScroll, true);
    return () => {
      window.removeEventListener('scroll', handleScroll, true);
    };
  }, [hidden]);

  return (
    <div
      className="grid w-48 grid-cols-4 place-items-center
                 rounded border-2 border-kashmir-800 bg-kashmir-300"
      style={style as any}
    >
      {Array.from(pieces.entries()).map(([key, piece]) => (
        <button
          key={key}
          className="flex w-full justify-center rounded transition-colors hover:bg-kashmir-200"
          onClick={() => onClick(key)}
        >
          <img className="aspect-square w-full" src={piece.src} />
        </button>
      ))}
      <button
        key="clear"
        className="col-span-4 w-full rounded py-2 font-bold text-kashmir-900 transition-colors hover:bg-kashmir-200"
        onClick={() => onClick(0)}
      >
        Clear
      </button>
    </div>
  );
}

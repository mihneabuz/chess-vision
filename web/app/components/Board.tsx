'use client';

import { useState } from 'react';
import { useSounds } from 'app/hooks';
import { pieces, startPosition } from 'app/pieces/util';

export default function Board({ initial }) {
  const [pieces, setPieces] = useState<number[]>(initial || startPosition);
  const [picker, setPicker] = useState<[number, number, number] | null>(null);
  const sounds = useSounds();

  const handleClick = (e: MouseEvent, index: number) => {
    setPicker((old) => (old ? null : [e.clientX, e.clientY, index]));
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
      <div className="grid grid-cols-8 rounded border-4 border-kashmir-800">
        {pieces.map((type, index) => (
          <Piece key={`p${index}`} index={index} type={type} onClick={handleClick} />
        ))}
      </div>

      <div className="flex flex-row items-center justify-center gap-8">
        <button
          className="flex w-full justify-center rounded transition-colors hover:bg-kashmir-200"
          onClick={() => handler(false)}
        >
          white
        </button>
        <button
          className="flex w-full justify-center rounded transition-colors hover:bg-kashmir-200"
          onClick={() => handler(true)}
        >
          black
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
  const hidden = !x || !y;

  const style = {
    position: 'fixed',
    left: `${x || 0}px`,
    top: `${y || 0}px`,
    opacity: hidden ? '0' : '95',
    display: hidden ? 'none' : undefined,
  };

  return (
    <div
      className="grid w-32 grid-cols-2 place-items-center
          rounded border-2 border-kashmir-800 bg-kashmir-300
          transition-opacity"
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
        className="col-span-2 w-full rounded py-2 font-bold text-kashmir-900 transition-colors hover:bg-kashmir-200"
        onClick={() => onClick(0)}
      >
        Clear
      </button>
    </div>
  );
}

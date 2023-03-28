'use client';

import { useState } from 'react';
import { useSounds } from 'app/hooks';
import { pieces, startPosition } from 'app/pieces/util';

export default function Board({ initial }) {
  const [pieces, setPieces] = useState<number[]>(initial || startPosition);
  const sounds = useSounds();

  const handleClick = (index: number) => {
    console.log(index);
  };

  return (
    <div className="grid grid-cols-8 rounded border-4 border-kashmir-800">
      {pieces.map((type, index) => (
        <Piece key={`p${index}`} index={index} type={type} onClick={handleClick} />
      ))}
    </div>
  );
}

function Piece({ index, type, onClick }) {
  const color = (index + Math.floor(index / 8)) % 2 === 0 ? 'bg-kashmir-200' : 'bg-kashmir-500';
  const piece = pieces.get(type);

  return (
    <div
      className={`flex aspect-square min-w-[4rem] items-center justify-center ${color}`}
      onClick={() => onClick(index)}
    >
      {piece
        ? <img className="w-full opacity-80" src={piece.src} />
        : <div className="border-none" />
      }
    </div>
  );
}

'use client';

import { CSSProperties, useCallback, useRef, useState } from 'react';
import { useScroll, useSounds } from 'app/hooks';
import { pieces, startPosition } from 'app/pieces/util';

interface Point {
  x: number;
  y: number;
}

export default function Board({ initial }) {
  const [pieces, setPieces] = useState<number[]>(initial || startPosition);
  const [picker, setPicker] = useState<[number, number, number] | null>(null);
  const [arrow, setArrow] = useState<{ from?: Point; to?: Point }>({});
  const boardRef = useRef<HTMLDivElement>(null);
  const sounds = useSounds();

  const handleClick = useCallback((e: MouseEvent, index: number) => {
    setPicker((old) => (old ? null : [e.clientX, e.clientY, index]));
  }, []);

  const handlePick = useCallback((type: number) => {
    if (!picker) return;
    sounds.thock1.play();
    setPieces((pieces) => pieces.map((val, idx) => (idx === picker[2] ? type : val)));
    setPicker(null);
    setArrow({});
  }, [picker, sounds]);

  const handler = useCallback(async (black: boolean) => {
    setArrow({});

    const res = await fetch('/api/generate', {
      method: 'POST',
      body: JSON.stringify({
        pieces,
        black,
      }),
    });

    const data = await res.json();

    if (!data.success) {
      return;
    }

    let { from, to } = data;

    const board = boardRef.current;
    if (!board) return;

    const fromRect = board.children.item(from)?.getBoundingClientRect();
    const toRect = board.children.item(to)?.getBoundingClientRect();

    if (fromRect && toRect) {
      setArrow({
        from: rectCenter(fromRect),
        to: rectCenter(toRect),
      });
    }

    sounds.thock1.play();
  }, [sounds, pieces]);

  return (
    <div>
      <div className="h-8"></div>

      <div ref={boardRef} className="my-8 grid grid-cols-8 rounded border-4 border-kashmir-800">
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
          I{'\''}m playing white
        </button>
        <button
          onClick={() => handler(true)}
          className="cursor-pointer rounded bg-kashmir-800 py-1 px-4 text-xl
                     transition-transform hover:scale-95 hover:bg-kashmir-700"
        >
          I{'\''}m playing black
        </button>
      </div>

      <Picker onClick={handlePick} x={picker && picker[0]} y={picker && picker[1]} />
      <Arrow from={arrow.from} to={arrow.to} />
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

  const offset = useScroll(!hidden);

  const style = {
    position: 'fixed',
    left: `${Math.floor(x - 96)}px`,
    top: `${Math.floor(y + offset)}px`,
    display: hidden ? 'none' : undefined,
  };

  return (
    <div
      className="z-40 grid w-48 grid-cols-4 place-items-center rounded
                 border-2 border-kashmir-800 bg-kashmir-300 opacity-95"
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

function Arrow({ from, to }) {
  const offset = useScroll(from && to);

  let style: CSSProperties = {
    display: 'none',
  };

  let skew: CSSProperties = {};

  if (from && to) {
    style = {
      left: `${Math.floor(Math.min(from.x, to.x))}px`,
      top: `${Math.floor(Math.min(from.y, to.y)) + offset}px`,
      width: `${Math.floor(Math.max(Math.abs(from.x - to.x), 20))}px`,
      height: `${Math.floor(Math.max(Math.abs(from.y - to.y), 20))}px`,
    };

    const [dist, angle] = computePoints(to, from);

    skew = {
      rotate: `${angle}rad`,
      height: dist - 10,
    };
  }

  return (
    <div className="pointer-events-none fixed flex items-center justify-center" style={style}>
      <div className="relative w-6 rounded bg-kashmir-800 opacity-95" style={skew}>
        <div className="absolute -top-4 -left-4 h-16 w-6 rotate-45 rounded bg-kashmir-800"></div>
        <div className="absolute -top-4 left-4 h-16 w-6 -rotate-45 rounded bg-kashmir-800"></div>
      </div>
    </div>
  );
}

function rectCenter(rect: DOMRect): Point {
  return {
    x: rect.x + rect.width / 2,
    y: rect.y + rect.height / 2,
  };
}

function computePoints(p1: Point, p2: Point): [number, number] {
  const dy = -(p1.y - p2.y);
  const dx = p1.x - p2.x;
  const dist = Math.sqrt(Math.pow(dx, 2) + Math.pow(dy, 2));
  const angle = Math.atan2(dx, dy);
  return [dist, angle];
}

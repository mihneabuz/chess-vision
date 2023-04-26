import { useRef, useEffect, useMemo, useCallback, useState } from 'react';
import { setTimeout } from 'timers';
import { sounds } from './pieces/util';

export function useEaseIn<T extends HTMLElement>(after: number) {
  const ref = useRef<T>(null);
  useEffect(() => {
    const timeout = setTimeout(() => {
      if (ref.current) {
        ref.current.style.opacity = '100%';
      }
    }, after);

    return () => clearTimeout(timeout);
  });

  return ref;
}

export function useScroll(active: boolean) {
  const [initial, setInitial] = useState<number>(window.scrollY);
  const [current, setCurrent] = useState<number>(window.scrollY);
  const handleScroll = useCallback(() => setCurrent(window.scrollY), []);

  useEffect(() => {
    setCurrent(window.scrollY);
    setInitial(window.scrollY);

    if (!active) return;

    window.addEventListener('scroll', handleScroll, true);
    return () => {
      window.removeEventListener('scroll', handleScroll, true);
    };
  }, [active, handleScroll]);

  return initial - current;
}

export function usePosition(): { x: number, y: number } {
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const handleMove = useCallback((e: MouseEvent) => {
    setPosition({ x: e.x, y: e.y });
  }, [setPosition]);

  useEffect(() => {
    window.addEventListener('mousemove', handleMove, true);
    return () => {
    window.removeEventListener('mousemove', handleMove, true);
    };
  })

  return position;
}

export function useSounds() {
  return useMemo(() => sounds(), []);
}

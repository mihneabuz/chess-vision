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

export function useSounds() {
  return useMemo(() => sounds(), []);
}

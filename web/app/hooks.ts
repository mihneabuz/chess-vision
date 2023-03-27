import { useRef, useEffect } from 'react';
import { setTimeout } from 'timers';

export function useEaseIn<T extends HTMLElement>(after: number) {
  const ref = useRef<T>(null);
  useEffect(() => {
    const timeout = setTimeout(() => {
      if (ref.current) {
        ref.current.style.opacity = "100%";
      }
    }, after);

    return () => clearTimeout(timeout);
  });

  return ref;
}

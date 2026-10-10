import { useRef, useCallback, useEffect } from "react";

export interface UseHoverPreviewOptions {
  delayMs?: number;
  onHover?: (id: number) => void;
  onHoverEnd?: () => void;
}

export function useHoverPreview(options: UseHoverPreviewOptions = {}) {
  const { delayMs = 1500, onHover, onHoverEnd } = options;
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const handleMouseEnter = useCallback(
    (id: number) => {
      // Clear any pending timer
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
      
      timeoutRef.current = setTimeout(() => {
        onHover?.(id);
      }, delayMs);
    },
    [delayMs, onHover]
  );

  const handleMouseLeave = useCallback(() => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    }
    onHoverEnd?.();
  }, [onHoverEnd]);

  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, []);

  return { handleMouseEnter, handleMouseLeave };
}

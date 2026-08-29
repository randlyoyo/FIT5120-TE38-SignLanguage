import type { CSSProperties } from "react";

// Deterministic hue per tag name so each category reads as visually
// distinct on the card grid, the same "hash the string, no lookup table"
// approach PlaceholderMedia uses for its per-sign gradient.
function hueForTag(tag: string): number {
  let hash = 0;
  for (let i = 0; i < tag.length; i++) {
    hash = (hash * 31 + tag.charCodeAt(i)) % 360;
  }
  return hash;
}

export function tagChipStyle(tag: string): CSSProperties {
  const hue = hueForTag(tag);
  return {
    background: `hsla(${hue}, 55%, 55%, 0.16)`,
    color: `hsl(${hue}, 70%, 78%)`,
    border: `1px solid hsla(${hue}, 55%, 55%, 0.45)`,
  };
}

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

// Specimen-label style: a neutral paper card with a colored left edge and
// matching label text, rather than a filled pill -- the color still
// encodes the category (helps repeat visitors learn the palette), but the
// shape reads as an index label, not a generic chip.
export function tagChipStyle(tag: string): CSSProperties {
  const hue = hueForTag(tag);
  const shade = `hsl(${hue}, 42%, 32%)`;
  return {
    color: shade,
    borderLeftColor: shade,
  };
}

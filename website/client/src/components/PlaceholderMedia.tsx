import { glyphForIndex } from "./Pagination/handGlyphs";

interface Props {
  seed: number;
  gloss: string;
}

// Deterministic hue per sign so cards look visually distinct without any
// real media -- an honest stand-in, never dressed up to look like footage.
function hueForSeed(seed: number) {
  return (seed * 47) % 360;
}

export function PlaceholderMedia({ seed, gloss }: Props) {
  const hue = hueForSeed(seed);
  const Glyph = glyphForIndex(seed);

  return (
    <div
      className="placeholder-media"
      style={{
        background: `linear-gradient(135deg, hsl(${hue} 28% 28%), hsl(${(hue + 40) % 360} 28% 16%))`,
      }}
      role="img"
      aria-label={`Placeholder illustration -- no verified video available yet for ${gloss}.`}
    >
      <Glyph className="glyph" />
      <span className="placeholder-label">Placeholder</span>
    </div>
  );
}

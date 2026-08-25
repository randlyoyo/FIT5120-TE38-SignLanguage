import type { SVGProps } from "react";

/**
 * Original abstract "palm + fingers" line-art glyphs used for pagination and
 * placeholder media. These are deliberately NOT real Auslan handshapes or
 * fingerspelling letters -- they're purely decorative wayfinding icons, so
 * nothing here could be mistaken for teaching content.
 */

type GlyphProps = SVGProps<SVGSVGElement>;

const base = (props: GlyphProps) => ({
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.5,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
  ...props,
});

export function HandGlyph1(props: GlyphProps) {
  return (
    <svg {...base(props)}>
      <rect x="8" y="12" width="8" height="8" rx="2.5" />
      <path d="M9 12V6M12 12V4M15 12V6M17.5 13V8" />
    </svg>
  );
}

export function HandGlyph2(props: GlyphProps) {
  return (
    <svg {...base(props)}>
      <rect x="8" y="12" width="8" height="8" rx="2.5" />
      <path d="M9 12L6 6M12 12V4M15 12L18 6M17.5 13L20 9" />
    </svg>
  );
}

export function HandGlyph3(props: GlyphProps) {
  return (
    <svg {...base(props)}>
      <rect x="8" y="12" width="8" height="8" rx="2.5" />
      <path d="M9 12V7M12 12V4M15 12V7" />
      <path d="M17.5 13c1.5-1 2-3 .5-4" />
    </svg>
  );
}

export function HandGlyph4(props: GlyphProps) {
  return (
    <svg {...base(props)}>
      <rect x="8" y="12" width="8" height="8" rx="2.5" />
      <path d="M9 12c-1-2-1-4 0-6M12 12V5M15 12c1-2 1-4 0-6" />
    </svg>
  );
}

export function HandGlyph5(props: GlyphProps) {
  return (
    <svg {...base(props)}>
      <rect x="8" y="12" width="8" height="8" rx="2.5" />
      <path d="M9 12V6M12 12V4M15 12V6" />
      <path d="M17.5 13c2-.5 3-2 2-3.5" />
    </svg>
  );
}

export function HandGlyph6(props: GlyphProps) {
  return (
    <svg {...base(props)}>
      <rect x="8" y="12" width="8" height="8" rx="2.5" />
      <path d="M12 12V4" />
      <path d="M9 12c-.5-2 0-4 1.5-5M15 12c.5-2 0-4-1.5-5" />
    </svg>
  );
}

export function HandGlyph7(props: GlyphProps) {
  return (
    <svg {...base(props)}>
      <rect x="8" y="12" width="8" height="8" rx="2.5" />
      <path d="M9 12V8M15 12V8" />
      <path d="M17.5 13c1-1.5.5-3.5-1-4.5" />
    </svg>
  );
}

export function HandGlyph8(props: GlyphProps) {
  return (
    <svg {...base(props)}>
      <rect x="8" y="12" width="8" height="8" rx="2.5" />
      <path d="M9 12V6M12 12V5M15 12V6M17.5 13V9" />
      <path d="M6.5 13c-1.5-.5-2-2-1-3.5" />
    </svg>
  );
}

export const HAND_GLYPHS = [
  HandGlyph1,
  HandGlyph2,
  HandGlyph3,
  HandGlyph4,
  HandGlyph5,
  HandGlyph6,
  HandGlyph7,
  HandGlyph8,
];

export function glyphForIndex(index: number) {
  return HAND_GLYPHS[index % HAND_GLYPHS.length];
}

/**
 * Counting-hand glyphs for pagination: page order is shown by literally
 * counting fingers (1 extended finger = page 1, 2 = page 2, ...), not by an
 * arbitrary decorative shape. Still an original abstract line-art hand, not
 * a real Auslan number handshape.
 */
const countBase = (props: GlyphProps) => ({
  viewBox: "0 0 30 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.6,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
  ...props,
});

// Wide spacing between fingers so each one reads clearly as a distinct
// finger even at a glance, instead of bunching into an unreadable cluster.
const palm = <rect x="7" y="13" width="17" height="8" rx="3" />;
const thumbCurled = <path d="M7 16c-1.2.2-1.8.9-1.8 1.7" />;
const thumbExtended = <path d="M7 16C2 15 0 11.5 1.2 8" />;
const fingerStub = (x: number) => `M${x} 13V11`;
const fingerFull = (x: number, topY: number) => `M${x} 13V${topY}`;

const INDEX_X = 10.5;
const MIDDLE_X = 14.5;
const RING_X = 18.5;
const PINKY_X = 22.5;

export function HandCount1(props: GlyphProps) {
  return (
    <svg {...countBase(props)}>
      {palm}
      <path d={fingerFull(INDEX_X, 5)} />
      <path d={fingerStub(MIDDLE_X)} />
      <path d={fingerStub(RING_X)} />
      <path d={fingerStub(PINKY_X)} />
      {thumbCurled}
    </svg>
  );
}

export function HandCount2(props: GlyphProps) {
  return (
    <svg {...countBase(props)}>
      {palm}
      <path d={fingerFull(INDEX_X, 5)} />
      <path d={fingerFull(MIDDLE_X, 3.5)} />
      <path d={fingerStub(RING_X)} />
      <path d={fingerStub(PINKY_X)} />
      {thumbCurled}
    </svg>
  );
}

export function HandCount3(props: GlyphProps) {
  return (
    <svg {...countBase(props)}>
      {palm}
      <path d={fingerFull(INDEX_X, 5)} />
      <path d={fingerFull(MIDDLE_X, 3.5)} />
      <path d={fingerFull(RING_X, 5)} />
      <path d={fingerStub(PINKY_X)} />
      {thumbCurled}
    </svg>
  );
}

export function HandCount4(props: GlyphProps) {
  return (
    <svg {...countBase(props)}>
      {palm}
      <path d={fingerFull(INDEX_X, 5)} />
      <path d={fingerFull(MIDDLE_X, 3.5)} />
      <path d={fingerFull(RING_X, 5)} />
      <path d={fingerFull(PINKY_X, 7)} />
      {thumbCurled}
    </svg>
  );
}

export function HandCount5(props: GlyphProps) {
  return (
    <svg {...countBase(props)}>
      {palm}
      <path d={fingerFull(INDEX_X, 5)} />
      <path d={fingerFull(MIDDLE_X, 3.5)} />
      <path d={fingerFull(RING_X, 5)} />
      <path d={fingerFull(PINKY_X, 7)} />
      {thumbExtended}
    </svg>
  );
}

const HAND_COUNTS = [HandCount1, HandCount2, HandCount3, HandCount4, HandCount5];

// Two-hand counting (6-10): counting past five naturally moves to a second
// hand, same as counting on your fingers -- one full open hand plus however
// many fingers the remainder needs, rather than an arbitrary tally mark.
// These use a self-contained LOCAL coordinate system (palm spans x 0..17,
// thumb sits at the x=0 side) so the shape can be composed twice via <g
// transform>, the second copy mirrored so its thumb points outward to match
// a real pair of hands.
const localPalm = <rect x="0" y="13" width="17" height="8" rx="3" />;
const localThumbCurled = <path d="M0 16c-1.2.2-1.8.9-1.8 1.7" />;
const localThumbExtended = <path d="M0 16C-5 15 -7 11.5 -5.8 8" />;
const L_INDEX = 3.5;
const L_MIDDLE = 7.5;
const L_RING = 11.5;
const L_PINKY = 15.5;

function handShape(count: number) {
  return (
    <>
      {localPalm}
      <path d={fingerFull(L_INDEX, 5)} />
      {count >= 2 ? <path d={fingerFull(L_MIDDLE, 3.5)} /> : <path d={fingerStub(L_MIDDLE)} />}
      {count >= 3 ? <path d={fingerFull(L_RING, 5)} /> : <path d={fingerStub(L_RING)} />}
      {count >= 4 ? <path d={fingerFull(L_PINKY, 7)} /> : <path d={fingerStub(L_PINKY)} />}
      {count >= 5 ? localThumbExtended : localThumbCurled}
    </>
  );
}

const twoHandBase = (props: GlyphProps) => ({
  viewBox: "0 0 58 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.6,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
  ...props,
});

/** Renders one full open hand (5) plus a second hand showing 1-5 more. */
function TwoHandCount({ second, ...props }: GlyphProps & { second: number }) {
  return (
    <svg {...twoHandBase(props)}>
      <g transform="translate(10,0)">{handShape(5)}</g>
      <g transform="translate(47,0) scale(-1,1)">{handShape(second)}</g>
    </svg>
  );
}

/** Picks the counting-hand icon for a given 1-indexed page number. */
export function PageHandIcon({ page, ...props }: GlyphProps & { page: number }) {
  if (page >= 1 && page <= 5) {
    const Glyph = HAND_COUNTS[page - 1];
    return <Glyph {...props} />;
  }
  if (page <= 10) {
    return <TwoHandCount second={page - 5} {...props} />;
  }
  // Beyond two hands' worth (10) is not a realistic page count for this
  // dataset -- fall back to two open hands rather than invent a third.
  return <TwoHandCount second={5} {...props} />;
}

import { useEffect, useState, type KeyboardEvent, type PointerEvent } from "react";

interface Props {
  tags: string[];
  counts: Record<string, number>;
  /** One color per entry in `tags`, same order -- wedge colors are a
   *  function of position around the pie (lib/pieColors.ts), not of the
   *  tag name, so neighbouring wedges never land on similar hues. */
  colors: string[];
  onChange: (next: Record<string, number>) => void;
}

const SIZE = 260;
const CENTER = SIZE / 2;
const RADIUS = SIZE / 2 - 16;

function pointOnCircle(angleDeg: number, radius: number) {
  const rad = (angleDeg * Math.PI) / 180;
  return { x: CENTER + radius * Math.sin(rad), y: CENTER - radius * Math.cos(rad) };
}

/**
 * Ratio editor for splitting a fixed total across tags. Two ways to adjust,
 * both conserve the total exactly by moving counts between two neighbours
 * rather than touching every wedge:
 *  - Drag a boundary handle to redistribute between the two wedges it sits
 *    between.
 *  - Scroll over a wedge to nudge it by one against its clockwise neighbour.
 *
 * ponytail: the seam between the last and first wedge has no drag handle
 * (dragging across the 0/360 wrap needs extra angle bookkeeping this
 * skips) -- that pair is still reachable by scrolling either wedge, so
 * every tag stays adjustable, just not by every method.
 */
export function PieRatioChart({ tags, counts, colors, onChange }: Props) {
  const [svgEl, setSvgEl] = useState<SVGSVGElement | null>(null);
  const [dragBoundary, setDragBoundary] = useState<number | null>(null);

  const total = tags.reduce((sum, t) => sum + (counts[t] ?? 0), 0) || 1;
  const boundaries: number[] = [0];
  for (const tag of tags) {
    boundaries.push(boundaries[boundaries.length - 1] + ((counts[tag] ?? 0) / total) * 360);
  }

  function angleFromPointer(clientX: number, clientY: number): number {
    if (!svgEl) return 0;
    const rect = svgEl.getBoundingClientRect();
    const x = ((clientX - rect.left) / rect.width) * SIZE - CENTER;
    const y = ((clientY - rect.top) / rect.height) * SIZE - CENTER;
    let deg = (Math.atan2(x, -y) * 180) / Math.PI;
    if (deg < 0) deg += 360;
    return deg;
  }

  // Moves `delta` from tagA to tagB (negative delta runs the other way),
  // never letting either drop below 1 -- clamped by whichever side is
  // actually giving count away, not the side receiving it.
  function redistribute(indexA: number, indexB: number, delta: number) {
    const tagA = tags[indexA];
    const tagB = tags[indexB];
    const countA = counts[tagA] ?? 0;
    const countB = counts[tagB] ?? 0;
    const clamped = Math.max(-(countB - 1), Math.min(delta, countA - 1));
    if (clamped === 0) return;
    onChange({ ...counts, [tagA]: countA - clamped, [tagB]: countB + clamped });
  }

  function handlePointerDown(boundaryIndex: number, e: PointerEvent) {
    e.currentTarget.setPointerCapture(e.pointerId);
    setDragBoundary(boundaryIndex);
  }

  // Keyboard equivalent of dragging: the arrow keys move a boundary the
  // same way a drag would, one count at a time -- the same two-wedge
  // redistribution, just discrete instead of following the pointer.
  function handleBoundaryKeyDown(boundaryIndex: number, e: KeyboardEvent) {
    const before = boundaryIndex - 1;
    const after = boundaryIndex;
    if (e.key === "ArrowRight" || e.key === "ArrowUp") {
      e.preventDefault();
      redistribute(after, before, 1);
    } else if (e.key === "ArrowLeft" || e.key === "ArrowDown") {
      e.preventDefault();
      redistribute(after, before, -1);
    }
  }

  function handlePointerMove(e: PointerEvent) {
    if (dragBoundary === null) return;
    const before = dragBoundary - 1; // wedge ending at this boundary
    const after = dragBoundary; // wedge starting at this boundary
    const spanStart = boundaries[before];
    const spanEnd = boundaries[after + 1];
    const combined = (counts[tags[before]] ?? 0) + (counts[tags[after]] ?? 0);
    const minSliver = (1 / total) * 360;

    const angle = angleFromPointer(e.clientX, e.clientY);
    const clampedAngle = Math.min(Math.max(angle, spanStart + minSliver), spanEnd - minSliver);
    const fraction = (clampedAngle - spanStart) / (spanEnd - spanStart);
    let newBefore = Math.round(fraction * combined);
    newBefore = Math.max(1, Math.min(combined - 1, newBefore));

    onChange({ ...counts, [tags[before]]: newBefore, [tags[after]]: combined - newBefore });
  }

  // Bound natively (not via React's onWheel) with { passive: false } --
  // React 17+ registers wheel/touch listeners at the root as passive for
  // scroll performance, so calling preventDefault() from a JSX onWheel
  // handler doesn't actually stop the page behind the modal from
  // scrolling. One listener on the chart itself, not per wedge, since
  // which wedge is "under the cursor" is just the angle lookup already
  // used for dragging.
  useEffect(() => {
    if (!svgEl) return;
    function onWheel(e: globalThis.WheelEvent) {
      e.preventDefault();
      if (tags.length < 2) return; // nothing to redistribute against
      const angle = angleFromPointer(e.clientX, e.clientY);
      let index = tags.findIndex((_, i) => angle >= boundaries[i] && angle < boundaries[i + 1]);
      if (index === -1) index = tags.length - 1;
      const next = (index + 1) % tags.length;
      redistribute(next, index, e.deltaY < 0 ? 1 : -1);
    }
    svgEl.addEventListener("wheel", onWheel, { passive: false });
    return () => svgEl.removeEventListener("wheel", onWheel);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [svgEl, tags, counts]);

  return (
    <svg
      ref={setSvgEl}
      className="pie-ratio-chart"
      viewBox={`0 0 ${SIZE} ${SIZE}`}
      onPointerMove={handlePointerMove}
      onPointerUp={() => setDragBoundary(null)}
      onPointerLeave={() => setDragBoundary(null)}
      role="group"
      aria-label="Word count ratio across the chosen topics -- drag a boundary, scroll over a wedge, or tab to a boundary handle and use the arrow keys"
    >
      {tags.length === 1 ? (
        // A single wedge spans the full 360 degrees, whose start and end
        // points are the same coordinate -- an SVG arc command from a
        // point back to itself draws nothing, so a full circle needs its
        // own element rather than the general arc-path case below.
        <circle cx={CENTER} cy={CENTER} r={RADIUS} fill={colors[0]} className="pie-ratio-wedge" />
      ) : (
        tags.map((tag, i) => {
          const start = boundaries[i];
          const end = boundaries[i + 1];
          const large = end - start > 180 ? 1 : 0;
          const p1 = pointOnCircle(start, RADIUS);
          const p2 = pointOnCircle(end, RADIUS);
          return (
            <path
              key={tag}
              d={`M${CENTER},${CENTER} L${p1.x},${p1.y} A${RADIUS},${RADIUS} 0 ${large} 1 ${p2.x},${p2.y} Z`}
              fill={colors[i]}
              className="pie-ratio-wedge"
            />
          );
        })
      )}
      {boundaries.slice(1, -1).map((angle, i) => {
        const boundaryIndex = i + 1;
        const before = tags[boundaryIndex - 1];
        const after = tags[boundaryIndex];
        const countBefore = counts[before] ?? 0;
        const countAfter = counts[after] ?? 0;
        const p = pointOnCircle(angle, RADIUS);
        return (
          <circle
            key={i}
            cx={p.x}
            cy={p.y}
            r={7}
            className="pie-ratio-handle"
            tabIndex={0}
            role="slider"
            aria-label={`Split between ${before} and ${after}`}
            aria-valuemin={1}
            aria-valuemax={countBefore + countAfter - 1}
            aria-valuenow={countBefore}
            aria-valuetext={`${before}: ${countBefore}, ${after}: ${countAfter}`}
            onPointerDown={(e) => handlePointerDown(boundaryIndex, e)}
            onKeyDown={(e) => handleBoundaryKeyDown(boundaryIndex, e)}
          />
        );
      })}
    </svg>
  );
}

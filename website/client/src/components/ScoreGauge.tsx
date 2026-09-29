import { useEffect, useRef, useState, type CSSProperties } from "react";

interface Props {
  /** Final score (0-100) to animate up to. */
  score: number;
  durationMs?: number;
}

const SIZE = 220;
const CENTER_X = SIZE / 2;
const CENTER_Y = SIZE / 2 + 6;
const RADIUS = 82;
const STROKE_WIDTH = 20;
// A car dashboard's sweep, not a half-circle speedometer -- 0 sits down at
// the lower left, 100 down at the lower right, the needle crossing the top.
const START_ANGLE = -120;
const END_ANGLE = 120;
// Matches the "excellent" tier boundary the server already scores against
// (recognize.js tierFor) -- reaching it is what triggers the celebration.
const CELEBRATE_THRESHOLD = 80;

function angleForValue(value: number): number {
  const clamped = Math.min(100, Math.max(0, value));
  return START_ANGLE + (clamped / 100) * (END_ANGLE - START_ANGLE);
}

function pointOnArc(angleDeg: number, radius: number) {
  const rad = (angleDeg * Math.PI) / 180;
  return { x: CENTER_X + radius * Math.sin(rad), y: CENTER_Y - radius * Math.cos(rad) };
}

function arcPath(startAngle: number, endAngle: number, radius: number): string {
  const p1 = pointOnArc(startAngle, radius);
  const p2 = pointOnArc(endAngle, radius);
  const large = endAngle - startAngle > 180 ? 1 : 0;
  return `M${p1.x},${p1.y} A${radius},${radius} 0 ${large} 1 ${p2.x},${p2.y}`;
}

// Settles into place instead of stopping dead, the same reason a real
// speedometer needle has a little damping.
function easeOutCubic(t: number): number {
  return 1 - Math.pow(1 - t, 3);
}

const SPARK_COUNT = 8;

/**
 * Speedometer-style reveal for a verify score: the needle climbs from 0 up
 * to the real score over `durationMs`, the way a speed-test result or a car
 * dashboard settles on a reading, instead of the number just appearing. No
 * pre-drawn tier bands -- the only color cue is the track filling in blue
 * behind the needle as it sweeps, plus a one-time celebration (green number,
 * a burst of sparks) the moment the climb reaches the top tier.
 */
export function ScoreGauge({ score, durationMs = 1400 }: Props) {
  const [displayed, setDisplayed] = useState(0);
  const [celebrating, setCelebrating] = useState(false);
  const frameRef = useRef<number | null>(null);
  const hasCelebratedRef = useRef(false);

  useEffect(() => {
    hasCelebratedRef.current = false;
    setCelebrating(false);
    const start = performance.now();
    function tick(now: number) {
      const t = Math.min(1, (now - start) / durationMs);
      const next = easeOutCubic(t) * score;
      setDisplayed(next);
      if (!hasCelebratedRef.current && next >= CELEBRATE_THRESHOLD) {
        hasCelebratedRef.current = true;
        setCelebrating(true);
      }
      if (t < 1) frameRef.current = requestAnimationFrame(tick);
    }
    frameRef.current = requestAnimationFrame(tick);
    return () => {
      if (frameRef.current !== null) cancelAnimationFrame(frameRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [score, durationMs]);

  const needleTip = pointOnArc(angleForValue(displayed), RADIUS - STROKE_WIDTH / 2 - 12);

  return (
    <div className="score-gauge">
      <svg
        viewBox={`0 0 ${SIZE} ${SIZE * 0.8}`}
        className="score-gauge-svg"
        role="img"
        aria-label={`Score ${Math.round(displayed)} out of 100`}
      >
        <path
          d={arcPath(START_ANGLE, END_ANGLE, RADIUS)}
          className="score-gauge-track-bg"
          fill="none"
          strokeWidth={STROKE_WIDTH}
        />
        {displayed > 0 && (
          <path
            d={arcPath(START_ANGLE, angleForValue(displayed), RADIUS)}
            className="score-gauge-track-fill"
            fill="none"
            strokeWidth={STROKE_WIDTH}
          />
        )}
        <line
          x1={CENTER_X}
          y1={CENTER_Y}
          x2={needleTip.x}
          y2={needleTip.y}
          className="score-gauge-needle"
        />
        <circle cx={CENTER_X} cy={CENTER_Y} r="7" className="score-gauge-hub" />
      </svg>

      <div className="score-gauge-readout">
        <div className={`score-gauge-value ${celebrating ? "celebrate" : ""}`}>
          {Math.round(displayed)}
        </div>
        {celebrating && (
          <div className="score-gauge-sparks" aria-hidden="true">
            {Array.from({ length: SPARK_COUNT }).map((_, i) => (
              <span
                key={i}
                className={`score-gauge-spark ${i % 2 === 0 ? "spark-accent" : "spark-success"}`}
                style={{ "--spark-angle": `${(360 / SPARK_COUNT) * i}deg` } as CSSProperties}
              />
            ))}
          </div>
        )}
      </div>
      <p className="score-gauge-caption">Accuracy</p>
    </div>
  );
}

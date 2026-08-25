import { useEffect, useState } from "react";
import { glyphForIndex } from "./Pagination/handGlyphs";

const SPEEDS = [0.5, 1, 2] as const;
const BASE_STEP_MS = 1800;

interface Props {
  gloss: string;
  steps: string[];
}

/**
 * Honest stand-in for a real demonstration video: no verified Auslan
 * footage exists yet, so instead of a video player this auto-advances
 * through the sign's text steps with a matching decorative icon per step.
 * Rewind/fast-forward/speed still make sense as playback controls over
 * this step sequence, without ever pretending real footage exists.
 */
export function SignDemonstration({ gloss, steps }: Props) {
  const [stepIndex, setStepIndex] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState<(typeof SPEEDS)[number]>(1);

  const lastIndex = Math.max(steps.length - 1, 0);

  useEffect(() => {
    if (!playing || steps.length <= 1) return;
    const timer = window.setInterval(() => {
      setStepIndex((prev) => {
        if (prev >= lastIndex) {
          setPlaying(false);
          return prev;
        }
        return prev + 1;
      });
    }, BASE_STEP_MS / speed);
    return () => window.clearInterval(timer);
  }, [playing, speed, lastIndex, steps.length]);

  if (steps.length === 0) {
    return <p className="demo-empty">No demonstration steps available yet.</p>;
  }

  const Glyph = glyphForIndex(stepIndex);

  return (
    <div className="sign-demo">
      <div className="sign-demo-stage" aria-live="polite">
        <Glyph className="sign-demo-glyph" width={72} height={72} />
        <p className="sign-demo-step-text">{steps[stepIndex]}</p>
        <p className="sign-demo-progress">
          Step {stepIndex + 1} of {steps.length}
        </p>
      </div>

      <div className="playback-controls" role="group" aria-label="Demonstration playback controls">
        <button type="button" onClick={() => setStepIndex(0)} aria-label="Rewind to first step">
          ⏮
        </button>
        <button
          type="button"
          onClick={() => setPlaying((p) => !p)}
          aria-label={playing ? "Pause" : "Play"}
        >
          {playing ? "⏸" : "▶"}
        </button>
        <button
          type="button"
          onClick={() => setStepIndex(lastIndex)}
          aria-label="Fast-forward to last step"
        >
          ⏭
        </button>
        <label>
          Speed
          <select
            value={speed}
            onChange={(e) => setSpeed(Number(e.target.value) as (typeof SPEEDS)[number])}
          >
            {SPEEDS.map((s) => (
              <option key={s} value={s}>
                {s}×
              </option>
            ))}
          </select>
        </label>
        <input
          type="range"
          min={0}
          max={lastIndex}
          value={stepIndex}
          onChange={(e) => {
            setPlaying(false);
            setStepIndex(Number(e.target.value));
          }}
          aria-label={`Scrub steps for ${gloss}`}
        />
      </div>
    </div>
  );
}

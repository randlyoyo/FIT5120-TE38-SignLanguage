import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { HandSkeletonCanvas } from "../components/HandSkeletonCanvas";
import { useSign, useSigns } from "../context/SignsContext";
import { getSignInfo } from "../lib/signs";

const BASE_FPS = 30;
const SPEEDS = [0.5, 0.75, 1] as const;

export function Learn() {
  const { id } = useParams<{ id: string }>();
  const sign = useSign(id);
  const { loading } = useSigns();
  const info = id ? getSignInfo(id) : undefined;

  const [frameIndex, setFrameIndex] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState<(typeof SPEEDS)[number]>(1);

  const frameCount = sign?.frames.length ?? 0;

  useEffect(() => {
    setFrameIndex(0);
    setPlaying(true);
  }, [id]);

  useEffect(() => {
    if (!playing || frameCount === 0) return;
    const intervalMs = 1000 / (BASE_FPS * speed);
    const timer = window.setInterval(() => {
      setFrameIndex((prev) => {
        const next = prev + 1;
        if (next >= frameCount) {
          setPlaying(false);
          return frameCount - 1;
        }
        return next;
      });
    }, intervalMs);
    return () => window.clearInterval(timer);
  }, [playing, speed, frameCount]);

  if (loading) return <p>Loading…</p>;
  if (!sign) {
    return (
      <section>
        <p>Sign not found.</p>
        <Link to="/">Back to library</Link>
      </section>
    );
  }

  const replay = () => {
    setFrameIndex(0);
    setPlaying(true);
  };

  return (
    <section>
      <p>
        <Link to="/">← Sign library</Link>
      </p>
      <h1>{sign.label}</h1>
      {info && <p>{info.description}</p>}

      <HandSkeletonCanvas frame={sign.frames[frameIndex] ?? null} width={360} height={280} />

      <div className="playback-controls" role="group" aria-label="Playback controls">
        <button type="button" onClick={() => setPlaying((p) => !p)}>
          {playing ? "Pause" : "Play"}
        </button>
        <button type="button" onClick={replay}>
          Replay
        </button>
        <label>
          Speed
          <select value={speed} onChange={(e) => setSpeed(Number(e.target.value) as (typeof SPEEDS)[number])}>
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
          max={Math.max(frameCount - 1, 0)}
          value={frameIndex}
          onChange={(e) => {
            setPlaying(false);
            setFrameIndex(Number(e.target.value));
          }}
          aria-label="Scrub frame"
        />
      </div>

      {info && (
        <>
          <h2>Steps</h2>
          <ol className="step-list">
            {info.steps.map((step) => (
              <li key={step}>{step}</li>
            ))}
          </ol>
        </>
      )}

      <p>
        <Link to={`/practice/${sign.id}`}>Practise this sign with your webcam →</Link>
      </p>
    </section>
  );
}

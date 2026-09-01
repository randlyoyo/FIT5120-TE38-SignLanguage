import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { HandSkeletonCanvas } from "../components/HandSkeletonCanvas";
import { useSigns } from "../context/SignsContext";
import { recordAttempt } from "../lib/storage";
import type { ReferenceSign } from "../lib/referenceData";

const FPS = 30;
const OPTION_COUNT = 4;

function shuffle<T>(items: T[]): T[] {
  const copy = [...items];
  for (let i = copy.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy;
}

function pickQuestion(signs: ReferenceSign[]) {
  const target = signs[Math.floor(Math.random() * signs.length)];
  const distractors = shuffle(signs.filter((s) => s.id !== target.id)).slice(0, OPTION_COUNT - 1);
  const options = shuffle([target, ...distractors]);
  return { target, options };
}

export function Quiz() {
  const { signs, loading } = useSigns();
  const [question, setQuestion] = useState<{ target: ReferenceSign; options: ReferenceSign[] } | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [score, setScore] = useState({ correct: 0, total: 0 });
  const [frameIndex, setFrameIndex] = useState(0);
  const frameIndexRef = useRef(0);

  const nextQuestion = useCallback(() => {
    if (signs.length < 2) return;
    setQuestion(pickQuestion(signs));
    setSelectedId(null);
    setFrameIndex(0);
    frameIndexRef.current = 0;
  }, [signs]);

  useEffect(() => {
    if (!loading && signs.length >= 2 && !question) nextQuestion();
  }, [loading, signs, question, nextQuestion]);

  const frameCount = question?.target.frames.length ?? 0;

  useEffect(() => {
    if (!question || selectedId) return; // pause the demo once answered
    const timer = window.setInterval(() => {
      frameIndexRef.current = (frameIndexRef.current + 1) % Math.max(frameCount, 1);
      setFrameIndex(frameIndexRef.current);
    }, 1000 / FPS);
    return () => window.clearInterval(timer);
  }, [question, selectedId, frameCount]);

  const currentFrame = useMemo(() => question?.target.frames[frameIndex] ?? null, [question, frameIndex]);

  if (loading) return <p>Loading quiz…</p>;
  if (signs.length < 2) return <p>Need at least two signs loaded to run a quiz.</p>;
  if (!question) return null;

  const answer = (optionId: string) => {
    if (selectedId) return;
    setSelectedId(optionId);
    const correct = optionId === question.target.id;
    setScore((s) => ({ correct: s.correct + (correct ? 1 : 0), total: s.total + 1 }));
    recordAttempt(question.target.id, correct, correct ? 1 : 0);
  };

  return (
    <section>
      <h1>Quiz — which sign is this?</h1>
      <p>
        Score: {score.correct} / {score.total}
      </p>

      <HandSkeletonCanvas frame={currentFrame} width={320} height={240} />

      <ul className="quiz-options">
        {question.options.map((option) => {
          const isSelected = selectedId === option.id;
          const isTarget = option.id === question.target.id;
          const showResult = selectedId !== null;
          let className = "quiz-option";
          if (showResult && isTarget) className += " correct";
          else if (showResult && isSelected) className += " incorrect";

          return (
            <li key={option.id}>
              <button type="button" className={className} onClick={() => answer(option.id)} disabled={showResult}>
                {option.label}
              </button>
            </li>
          );
        })}
      </ul>

      {selectedId && (
        <p aria-live="polite">
          {selectedId === question.target.id ? "Correct! " : "Not quite — "}
          That was <strong>{question.target.label}</strong>.
        </p>
      )}

      <button type="button" onClick={nextQuestion} disabled={!selectedId}>
        Next question
      </button>
    </section>
  );
}

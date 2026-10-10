import { useEffect, useState } from "react";
import type { Scenario } from "../lib/scenarioStories";
import { WORD_TO_SIGN_ID } from "../lib/wordToSignId";
import { isLearned } from "../lib/learnedSigns";

export interface ScenarioProgress {
  /** word -> resolved sign id (or null if the dataset has no exact match). */
  wordIds: Record<string, number | null>;
  /** Per-step: every one of its words is marked "Learned" (lib/learnedSigns.ts). */
  stepDone: boolean[];
  learnedCount: number;
  totalSteps: number;
}

/** A step's real learned-progress -- shared by the scenario list (one badge
 *  per card) and the scenario detail page (per-step checkmarks), so both
 *  read the same "Learned" flag the rest of the site uses instead of each
 *  tracking its own notion of "done". */
export function useScenarioProgress(scenario: Scenario): ScenarioProgress {
  const [wordIds, setWordIds] = useState<Record<string, number | null>>({});
  const [tick, setTick] = useState(0);

  useEffect(() => {
    // Use the pre-computed WORD_TO_SIGN_ID map instead of API calls
    const words = [...new Set(scenario.steps.flatMap((s) => s.words))];
    const pairs = words.map((w) => [w, WORD_TO_SIGN_ID.get(w) ?? null] as const);
    setWordIds(Object.fromEntries(pairs));
  }, [scenario]);

  useEffect(() => {
    const refresh = () => setTick((t) => t + 1);
    window.addEventListener("focus", refresh);
    return () => window.removeEventListener("focus", refresh);
  }, []);

  // Recomputed on every render, including the ones `tick` forces -- isLearned
  // just reads localStorage, there's nothing here worth memoizing.
  void tick;
  const stepDone = scenario.steps.map(
    (step) => step.words.length > 0 && step.words.every((w) => wordIds[w] != null && isLearned(wordIds[w]!))
  );

  return { wordIds, stepDone, learnedCount: stepDone.filter(Boolean).length, totalSteps: scenario.steps.length };
}

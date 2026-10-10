import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { resolveScenarioWord } from "../lib/resolveScenarioWord";

interface Props {
  word: string;
  /** Pre-resolved sign id, when the parent already looked it up (e.g. to
   *  check learned-status for progress tracking) -- skips a redundant
   *  fetch on click. Falls back to resolving it itself if omitted. */
  signId?: number | null;
  /** Shows this word as already learned (lib/learnedSigns.ts), for the
   *  scenario page's per-step progress. */
  learned?: boolean;
}

/** A scenario step's vocab chip -- resolves to the real sign's detail page
 *  (exact gloss match in the library) at click time, rather than a fixed
 *  sign id that would break if the dataset changes. scenarioStories.ts
 *  words are verified exact matches (scripts/checkScenarioWords.mjs), so
 *  the library-search fallback below is only a safety net for a future
 *  dataset change, never silently landing on the wrong sign. */
export function ScenarioWordLink({ word, signId, learned }: Props) {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);

  async function go() {
    if (signId !== undefined) {
      navigate(signId ? `/signs/${signId}` : `/library?query=${encodeURIComponent(word)}`);
      return;
    }
    setLoading(true);
    try {
      const id = await resolveScenarioWord(word);
      navigate(id ? `/signs/${id}` : `/library?query=${encodeURIComponent(word)}`);
    } catch {
      navigate(`/library?query=${encodeURIComponent(word)}`);
    } finally {
      setLoading(false);
    }
  }

  return (
    <button
      type="button"
      className={`scenario-story-word ${learned ? "scenario-story-word-learned" : ""}`}
      onClick={go}
      disabled={loading}
    >
      {learned && <span aria-hidden="true">✓ </span>}
      {word}
    </button>
  );
}

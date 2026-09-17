import { useEffect, useState } from "react";
import { fetchRandomSignsByTag, fetchTags } from "../api/signs";
import type { TagCount } from "../api/types";
import { getLearnedIds } from "../lib/learnedSigns";
import { addAllToLearn, getToLearnIds } from "../lib/toLearnSigns";

interface Props {
  /** Called after a session is built and added, so the caller can refresh
   *  its own list of ids. */
  onBuilt: () => void;
}

/**
 * Epic 5 (Personalised Learning Recommendation): the learner picks how many
 * signs to pull from each category (US5.1) up to a target session size
 * (US5.2), and the picks skip anything already learned or already queued
 * (US5.3) -- built signs are added straight to the To Learn list rather
 * than a separate one-off session, so this is the one place that list gets
 * populated in bulk instead of one sign at a time.
 */
export function PersonalizeSessionPanel({ onBuilt }: Props) {
  const [open, setOpen] = useState(false);
  const [tags, setTags] = useState<TagCount[]>([]);
  const [counts, setCounts] = useState<Record<string, number>>({});
  const [targetSize, setTargetSize] = useState(10);
  const [building, setBuilding] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<string | null>(null);

  useEffect(() => {
    if (!open || tags.length > 0) return;
    const controller = new AbortController();
    // Counts must reflect what's actually left to pick (US5.3), not the
    // category's raw size -- otherwise "42 available" invites a learner to
    // ask for more than remains once already-learned/queued signs are
    // excluded.
    const excludeIds = [...getLearnedIds(), ...getToLearnIds()];
    fetchTags(excludeIds, controller.signal)
      .then(setTags)
      .catch((err) => {
        if (err.name !== "AbortError") console.error("Failed to load tags:", err);
      });
    return () => controller.abort();
  }, [open, tags.length]);

  const selectedTotal = Object.values(counts).reduce((sum, n) => sum + n, 0);
  const maxPossible = tags.reduce((sum, t) => sum + t.count, 0);
  // "Exactly n" (US5.2) is the goal, but a target bigger than every
  // category combined can never be hit -- once the learner has claimed
  // everything there is, that's as close as physically possible, so treat
  // it as satisfied rather than locking the button forever.
  const matchesTarget = selectedTotal === targetSize || (selectedTotal === maxPossible && maxPossible < targetSize);

  function setCount(tag: string, value: number, available: number) {
    const clamped = Math.max(0, Math.min(value, available));
    setCounts((prev) => ({ ...prev, [tag]: clamped }));
    setError(null);
  }

  // US5.2: distribute the target size across categories automatically
  // instead of requiring the learner to hand-tune every row to make the
  // numbers add up. Spreads across whichever tags already have a count
  // (the learner's expressed interest), or every tag if none do yet, one
  // sign at a time round-robin so the split stays even and never exceeds a
  // category's remaining pool.
  function autoDistribute() {
    const chosen = tags.filter((t) => (counts[t.tag] ?? 0) > 0);
    const pool = chosen.length > 0 ? chosen : tags;
    const next: Record<string, number> = {};
    let remaining = targetSize;
    let addedThisRound = true;
    while (remaining > 0 && addedThisRound) {
      addedThisRound = false;
      for (const { tag, count: cap } of pool) {
        if (remaining <= 0) break;
        const cur = next[tag] ?? 0;
        if (cur < cap) {
          next[tag] = cur + 1;
          remaining--;
          addedThisRound = true;
        }
      }
    }
    setCounts(next);
    setError(null);
  }

  async function build() {
    setError(null);
    setResult(null);
    if (selectedTotal === 0) {
      setError("Pick at least one sign from a category first.");
      return;
    }
    if (!matchesTarget) {
      setError(`Selections add up to ${selectedTotal}, not your target of ${targetSize} -- adjust a category or use Auto-fill.`);
      return;
    }
    setBuilding(true);
    try {
      // Accumulated as we go, not just read once -- a sign filed under two
      // categories must not be drawn twice for two different quotas. Also
      // catches a category running short mid-build (e.g. two overlapping
      // tags competing for the same signs), not just a stale "available"
      // count from before the panel was opened.
      const excludeIds = [...getLearnedIds(), ...getToLearnIds()];
      const picked: number[] = [];
      const shortfalls: string[] = [];
      for (const [tag, count] of Object.entries(counts)) {
        if (count <= 0) continue;
        const signs = await fetchRandomSignsByTag(tag, count, excludeIds);
        for (const sign of signs) {
          picked.push(sign.id);
          excludeIds.push(sign.id);
        }
        if (signs.length < count) {
          shortfalls.push(`${tag} (wanted ${count}, only ${signs.length} left to learn)`);
        }
      }
      addAllToLearn(picked);
      const addedLine = `Added ${picked.length} sign${picked.length === 1 ? "" : "s"} to your To Learn list.`;
      setResult(
        shortfalls.length > 0
          ? `${addedLine} Ran short in: ${shortfalls.join("; ")}.`
          : addedLine
      );
      setCounts({});
      onBuilt();
    } catch (err) {
      console.error("Failed to build session:", err);
      setError("Something went wrong building that session. Please try again.");
    } finally {
      setBuilding(false);
    }
  }

  return (
    <div className="personalize-panel">
      <button
        type="button"
        className="personalize-panel-toggle"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
      >
        {open ? "▾" : "▸"} Personalize your learning
      </button>

      {open && (
        <div className="personalize-panel-body">
          <div className="personalize-target-row">
            <label className="personalize-target">
              Target session size
              <input
                type="number"
                min={1}
                max={50}
                value={targetSize}
                onChange={(e) => setTargetSize(Math.max(1, Math.min(50, Number(e.target.value) || 1)))}
              />
            </label>
            <button
              type="button"
              className="personalize-autofill"
              onClick={autoDistribute}
              disabled={tags.length === 0}
              title="Spread the target size evenly across categories"
            >
              Auto-fill
            </button>
          </div>

          <ul className="personalize-tag-list">
            {tags.map(({ tag, count: available }) => (
              <li key={tag} className="personalize-tag-row">
                <span className="personalize-tag-name">{tag}</span>
                <span className="personalize-tag-available">{available} available</span>
                <input
                  type="number"
                  min={0}
                  max={available}
                  value={counts[tag] ?? 0}
                  onChange={(e) => setCount(tag, Number(e.target.value) || 0, available)}
                />
              </li>
            ))}
          </ul>

          <div className="personalize-panel-footer">
            <p className={`personalize-total ${matchesTarget ? "on-target" : ""}`}>
              <span className="personalize-total-dot" aria-hidden="true" />
              {selectedTotal} of {targetSize} selected
            </p>
            <button
              type="button"
              className="practice-start-button"
              onClick={build}
              disabled={building || selectedTotal === 0}
            >
              {building ? "Building…" : "Build session"}
            </button>
          </div>

          {error && (
            <p role="alert" className="practice-error">
              {error}
            </p>
          )}
          {result && <p className="personalize-result">{result}</p>}
        </div>
      )}
    </div>
  );
}

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
    fetchTags(controller.signal)
      .then(setTags)
      .catch((err) => {
        if (err.name !== "AbortError") console.error("Failed to load tags:", err);
      });
    return () => controller.abort();
  }, [open, tags.length]);

  const selectedTotal = Object.values(counts).reduce((sum, n) => sum + n, 0);

  function setCount(tag: string, value: number, available: number) {
    const clamped = Math.max(0, Math.min(value, available));
    setCounts((prev) => ({ ...prev, [tag]: clamped }));
  }

  async function build() {
    setBuilding(true);
    setError(null);
    setResult(null);
    try {
      // Accumulated as we go, not just read once -- a sign filed under two
      // categories must not be drawn twice for two different quotas.
      const excludeIds = [...getLearnedIds(), ...getToLearnIds()];
      const picked: number[] = [];
      for (const [tag, count] of Object.entries(counts)) {
        if (count <= 0) continue;
        const signs = await fetchRandomSignsByTag(tag, count, excludeIds);
        for (const sign of signs) {
          picked.push(sign.id);
          excludeIds.push(sign.id);
        }
      }
      if (picked.length === 0) {
        setError("Pick at least one sign from a category first.");
        return;
      }
      addAllToLearn(picked);
      setResult(`Added ${picked.length} sign${picked.length === 1 ? "" : "s"} to your To Learn list.`);
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
            <p className="personalize-total">
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

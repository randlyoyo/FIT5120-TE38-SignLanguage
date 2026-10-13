import { useEffect, useRef, useState } from "react";
import { fetchRandomSignsByTag, fetchTags } from "../api/signs";
import type { TagCount } from "../api/types";
import { getLearnedIds } from "../lib/learnedSigns";
import { pieColors } from "../lib/pieColors";
import { evenSplit, rescaleToTotal } from "../lib/proportional";
import { tagChipStyle } from "../lib/tagColors";
import { addAllToLearn, getToLearnIds } from "../lib/toLearnSigns";
import { PieRatioChart } from "./PieRatioChart";

interface Props {
  /** Called after a session is built and added, so the caller can refresh
   *  its own list of ids. */
  onBuilt: () => void;
}

type Step = "tags" | "ratio";

const MAX_TOTAL = 50;

/**
 * Epic 5 (Personalised Learning Recommendation): a two-step popup wizard.
 * Step 1 picks which topics to pull from (US5.1). Step 2 sets a total word
 * count -- minimum one per chosen topic, capped at 50 (US5.2) -- and its
 * split across topics via a pie chart the learner can drag or scroll to
 * adjust, rather than typing a number into every row. Picks skip anything
 * already learned or queued (US5.3) and land straight in the To Learn list.
 */
export function PersonalizeSessionPanel({ onBuilt }: Props) {
  const [modalOpen, setModalOpen] = useState(false);
  const [step, setStep] = useState<Step>("tags");
  const [allTags, setAllTags] = useState<TagCount[]>([]);
  const [selectedTags, setSelectedTags] = useState<string[]>([]);
  const [total, setTotal] = useState(0);
  const [counts, setCounts] = useState<Record<string, number>>({});
  const [building, setBuilding] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<string | null>(null);
  const modalRef = useRef<HTMLDivElement | null>(null);
  const openerRef = useRef<HTMLButtonElement | null>(null);

  // Standard dialog keyboard behaviour (WAI-ARIA APG): Escape dismisses it,
  // Tab/Shift+Tab cycle only through the dialog's own focusable elements
  // instead of leaking out to the page underneath, and focus starts inside
  // it and returns to the button that opened it on close.
  useEffect(() => {
    if (!modalOpen) return;
    const modal = modalRef.current;
    const focusable = modal?.querySelectorAll<HTMLElement>(
      'button:not(:disabled), input:not(:disabled), [tabindex]:not([tabindex="-1"])'
    );
    focusable?.[0]?.focus();

    function onKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") {
        e.stopPropagation();
        closeWizard();
        return;
      }
      if (e.key !== "Tab" || !focusable || focusable.length === 0) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault();
        first.focus();
      }
    }

    document.addEventListener("keydown", onKeyDown, true);
    return () => {
      document.removeEventListener("keydown", onKeyDown, true);
      openerRef.current?.focus();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [modalOpen, step, result]);

  useEffect(() => {
    if (!modalOpen || allTags.length > 0) return;
    const controller = new AbortController();
    // Counts must reflect what's actually left to pick (US5.3), not the
    // category's raw size -- otherwise a topic offers more than remains
    // once already-learned/queued signs are excluded.
    const excludeIds = [...getLearnedIds(), ...getToLearnIds()];
    fetchTags(excludeIds, controller.signal)
      .then(setAllTags)
      .catch((err) => {
        if (err.name !== "AbortError") console.error("Failed to load tags:", err);
      });
    return () => controller.abort();
  }, [modalOpen, allTags.length]);

  function openWizard() {
    setModalOpen(true);
    setStep("tags");
    setError(null);
    setResult(null);
  }

  function closeWizard() {
    setModalOpen(false);
    setSelectedTags([]);
    setCounts({});
    setResult(null);
    setError(null);
  }

  function toggleTag(tag: string) {
    setSelectedTags((prev) => (prev.includes(tag) ? prev.filter((t) => t !== tag) : [...prev, tag]));
  }

  function goToRatioStep() {
    if (selectedTags.length === 0) return;
    const min = selectedTags.length;
    const initialTotal = Math.max(min, Math.min(MAX_TOTAL, min * 5));
    setTotal(initialTotal);
    setCounts(evenSplit(selectedTags, initialTotal));
    setStep("ratio");
  }

  function changeTotal(next: number) {
    const min = selectedTags.length;
    const clamped = Math.max(min, Math.min(MAX_TOTAL, next || min));
    setTotal(clamped);
    setCounts((prev) => rescaleToTotal(prev, selectedTags, clamped));
  }

  const availableByTag = new Map(allTags.map((t) => [t.tag, t.count]));
  const wedgeColors = pieColors(selectedTags.length);

  // Typing a count directly sets that one tag and lets the total move to
  // match, rather than the pie's drag/scroll behaviour of conserving the
  // total by taking from a neighbour -- typing a specific number is the
  // learner stating an exact target for that topic, not nudging a split.
  function setTagCount(tag: string, value: number) {
    const otherTotal = selectedTags.reduce((sum, t) => (t === tag ? sum : sum + (counts[t] ?? 0)), 0);
    const available = availableByTag.get(tag);
    const maxForTag = Math.min(MAX_TOTAL - otherTotal, available ?? MAX_TOTAL);
    const clamped = Math.max(1, Math.min(maxForTag, value || 1));
    setCounts((prev) => ({ ...prev, [tag]: clamped }));
    setTotal(otherTotal + clamped);
  }

  async function generate() {
    setError(null);
    setBuilding(true);
    try {
      // Accumulated as we go, not just read once -- a sign filed under two
      // chosen topics must not be drawn twice for two different quotas.
      const excludeIds = [...getLearnedIds(), ...getToLearnIds()];
      const picked: number[] = [];
      const shortfalls: string[] = [];
      for (const tag of selectedTags) {
        const count = counts[tag] ?? 0;
        if (count <= 0) continue;
        const signs = await fetchRandomSignsByTag(tag, count, excludeIds);
        for (const sign of signs) {
          picked.push(sign.id);
          excludeIds.push(sign.id);
        }
        if (signs.length < count) {
          shortfalls.push(`${tag} (wanted ${count}, only ${signs.length} left)`);
        }
      }
      addAllToLearn(picked);
      const addedLine = `Added ${picked.length} sign${picked.length === 1 ? "" : "s"} to your To Learn list.`;
      setResult(shortfalls.length > 0 ? `${addedLine} Ran short in: ${shortfalls.join("; ")}.` : addedLine);
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
      <button type="button" ref={openerRef} className="personalize-cta" onClick={openWizard}>
        <span className="personalize-cta-icon" aria-hidden="true">
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
            <line x1="12" y1="5" x2="12" y2="19" />
            <line x1="5" y1="12" x2="19" y2="12" />
          </svg>
        </span>
        <span className="personalize-cta-text">
          <span className="personalize-cta-title">Personalize your learning</span>
          <span className="personalize-cta-sub">
            Pick the topics you care about and how many words you want &mdash; we&rsquo;ll fill your list
            with signs you haven&rsquo;t learned yet.
          </span>
        </span>
        <span className="personalize-cta-go" aria-hidden="true">
          Start &rarr;
        </span>
      </button>

      {modalOpen && (
        <div className="wizard-overlay" onClick={closeWizard}>
          <div
            ref={modalRef}
            className="wizard-modal"
            role="dialog"
            aria-modal="true"
            aria-label="Personalize your learning session"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="wizard-header">
              <div>
                <ol className="wizard-stepper" aria-label="Progress">
                  {["Topics", "Word mix"].map((label, i) => {
                    const current = result ? 3 : step === "tags" ? 1 : 2;
                    const state = i + 1 < current ? "done" : i + 1 === current ? "current" : "todo";
                    return (
                      <li key={label} className={`wizard-stepper-step is-${state}`} aria-current={state === "current" ? "step" : undefined}>
                        <span className="wizard-stepper-num" aria-hidden="true">
                          {state === "done" ? "✓" : i + 1}
                        </span>
                        {label}
                      </li>
                    );
                  })}
                </ol>
                <h2 className="wizard-title">
                  {result ? "All set" : step === "tags" ? "Choose your topics" : "Set word count & ratio"}
                </h2>
              </div>
              <button type="button" className="wizard-close" onClick={closeWizard} aria-label="Close">
                &times;
              </button>
            </div>

            {result ? (
              <div className="wizard-body wizard-result">
                <span className="wizard-result-icon" aria-hidden="true">
                  &#10003;
                </span>
                <p>{result}</p>
              </div>
            ) : step === "tags" ? (
              <div className="wizard-body">
                <p className="wizard-hint">Pick every topic you&rsquo;d like signs pulled from.</p>
                <ul className="wizard-tag-grid">
                  {allTags.map(({ tag, count }) => {
                    const active = selectedTags.includes(tag);
                    return (
                      <li key={tag}>
                        <button
                          type="button"
                          className={`wizard-tag-chip ${active ? "active" : ""}`}
                          style={tagChipStyle(tag)}
                          onClick={() => toggleTag(tag)}
                          aria-pressed={active}
                        >
                          <span className="wizard-tag-check" aria-hidden="true">
                            {active ? "✓" : ""}
                          </span>
                          <span className="wizard-tag-label">{tag}</span>
                          <span className="wizard-tag-chip-count">{count} new</span>
                        </button>
                      </li>
                    );
                  })}
                </ul>
              </div>
            ) : (
              <div className="wizard-body wizard-ratio-body">
                <label className="wizard-total-input">
                  Total words
                  <input
                    type="number"
                    min={selectedTags.length}
                    max={MAX_TOTAL}
                    value={total}
                    onChange={(e) => changeTotal(Number(e.target.value))}
                  />
                </label>

                <div className="wizard-ratio-layout">
                  <PieRatioChart
                    tags={selectedTags}
                    counts={counts}
                    colors={wedgeColors}
                    onChange={setCounts}
                  />
                  <ul className="wizard-ratio-legend">
                    {selectedTags.map((tag, i) => (
                      <li key={tag} className="wizard-ratio-legend-row">
                        <span
                          className="wizard-ratio-swatch"
                          style={{ background: wedgeColors[i] }}
                          aria-hidden="true"
                        />
                        <span className="wizard-ratio-legend-name">{tag}</span>
                        <span className="wizard-ratio-legend-count">
                          <input
                            type="number"
                            className="wizard-ratio-legend-input"
                            min={1}
                            max={availableByTag.get(tag) ?? MAX_TOTAL}
                            value={counts[tag] ?? 0}
                            onChange={(e) => setTagCount(tag, Number(e.target.value))}
                            aria-label={`Word count for ${tag}`}
                          />
                          <span className="wizard-ratio-legend-available">/ {availableByTag.get(tag) ?? "?"}</span>
                        </span>
                      </li>
                    ))}
                  </ul>
                </div>
                <p className="wizard-ratio-hint">
                  Drag a boundary, scroll over a wedge, or type a count directly. A boundary handle can also be
                  tabbed to and adjusted with the arrow keys.
                </p>
              </div>
            )}

            {error && (
              <p role="alert" className="practice-error wizard-error">
                {error}
              </p>
            )}

            <div className="wizard-footer">
              <span className="wizard-footer-summary">
                {result
                  ? ""
                  : step === "tags"
                    ? selectedTags.length === 0
                      ? "Choose at least one topic"
                      : `${selectedTags.length} topic${selectedTags.length === 1 ? "" : "s"} selected`
                    : `${total} words from ${selectedTags.length} topic${selectedTags.length === 1 ? "" : "s"}`}
              </span>
              {result ? (
                <button type="button" className="practice-start-button" onClick={closeWizard}>
                  Done
                </button>
              ) : step === "tags" ? (
                <button
                  type="button"
                  className="practice-start-button"
                  onClick={goToRatioStep}
                  disabled={selectedTags.length === 0}
                >
                  Next
                </button>
              ) : (
                <>
                  <button type="button" className="wizard-back-button" onClick={() => setStep("tags")}>
                    Back
                  </button>
                  <button type="button" className="practice-start-button" onClick={generate} disabled={building}>
                    {building ? "Generating…" : "Generate"}
                  </button>
                </>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

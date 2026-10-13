import { useState, useMemo, useEffect } from "react";
import { Link, useParams, useLocation } from "react-router-dom";
import { getScenarioById } from "../lib/scenarioCategories";
import { WORD_TO_SIGN_ID } from "../lib/wordToSignId";
import { isLearned } from "../lib/learnedSigns";
import { addAllToLearn, getToLearnIds } from "../lib/toLearnSigns";
import { FINISH_PICTOGRAM, scenarioPictogram } from "../lib/pictograms";
import { useHoverPreview } from "../hooks/useHoverPreview";
import { WordPreviewModal } from "../components/WordPreviewModal";

const CATEGORY_NAMES = {
  A: "Daily Life",
  B: "Professional",
  C: "Language Basics",
};

export function ScenarioCategoryPage() {
  const { id } = useParams<{ id: string }>();
  const location = useLocation();
  const scenario = id ? getScenarioById(id) : undefined;
  const [expandedSubId, setExpandedSubId] = useState<string | null>(null);
  // Bumped after "add to To Learn" so the button re-reads storage.
  const [, setQueuedVersion] = useState(0);
  const [previewSignId, setPreviewSignId] = useState<number | null>(null);
  const [previewWord, setPreviewWord] = useState<string>("");
  const [isPreviewLoaded, setIsPreviewLoaded] = useState(false);

  // Auto-expand subcategory when returning from detail page
  useEffect(() => {
    const selectedSubId = (location.state as { selectedSubId?: string } | null)?.selectedSubId;
    if (selectedSubId) {
      setExpandedSubId(selectedSubId);
    }
  }, [location.state]);

  const { handleMouseEnter, handleMouseLeave } = useHoverPreview({
    delayMs: 1000,
    onHover: (signId: number) => {
      setPreviewSignId(signId);
      setIsPreviewLoaded(false);
    },
    onHoverEnd: () => {
      // Only close if preview has finished loading
      if (isPreviewLoaded) {
        setPreviewSignId(null);
      }
    },
  });

  const subScenarioStats = useMemo(() => {
    if (!scenario) return new Map<string, { learned: number; total: number }>();
    
    const stats = new Map<string, { learned: number; total: number }>();
    for (const sub of scenario.subScenarios) {
      let learned = 0;
      for (const word of sub.words) {
        const signId = WORD_TO_SIGN_ID.get(word);
        if (signId !== undefined && isLearned(signId)) {
          learned++;
        }
      }
      stats.set(sub.id, { learned, total: sub.wordCount });
    }
    return stats;
  }, [scenario]);

  if (!scenario) {
    return (
      <div className="page-container">
        <Link to="/scenarios" className="back-link">
          ↤ Back to scenarios
        </Link>
        <p role="alert">Couldn't find that scenario category.</p>
      </div>
    );
  }

  return (
    <div className="page-container scenario-category-page">
      <Link to="/scenarios" className="back-link">
        ↤ Back to scenarios
      </Link>

      {(() => {
        const learnedTotal = [...subScenarioStats.values()].reduce((sum, st) => sum + st.learned, 0);
        const isDone = (subId: string) => {
          const st = subScenarioStats.get(subId);
          return st !== undefined && st.learned === st.total;
        };
        const nextIndex = scenario.subScenarios.findIndex((s) => !isDone(s.id));
        const activeIndex =
          expandedSubId !== null
            ? Math.max(0, scenario.subScenarios.findIndex((s) => s.id === expandedSubId))
            : Math.max(0, nextIndex);
        const selectedSub = scenario.subScenarios[activeIndex];
        const nextSub = scenario.subScenarios[activeIndex + 1];
        const subStats = subScenarioStats.get(selectedSub.id) ?? { learned: 0, total: selectedSub.wordCount };
        const queued = new Set(getToLearnIds());
        const toQueue = selectedSub.words
          .map((w) => WORD_TO_SIGN_ID.get(w))
          .filter((sid): sid is number => sid !== undefined && !isLearned(sid) && !queued.has(sid));
        const groupDone = subStats.learned === subStats.total;

        return (
          <>
            <header className="scenario-path-header journey-head">
              <img className="journey-head-icon" src={scenarioPictogram(scenario.id)} alt="" />
              <div>
                <p className="scenario-path-eyebrow">Learning journey · {CATEGORY_NAMES[scenario.category]}</p>
                <h1 className="category-title">{scenario.title}</h1>
                <div className="progress-bar" aria-hidden="true">
                  <div className="progress-bar-fill" style={{ width: `${(learnedTotal / scenario.wordCount) * 100}%` }} />
                </div>
                <p className="scenario-path-summary">
                  {learnedTotal} of {scenario.wordCount} signs learned
                </p>
              </div>
            </header>

            <div className="journey-layout">
              {/* The route: stops top to bottom, joined by downward arrows that
                  fill in as stops are cleared, ending at the finish. */}
              <ol className="journey-route" aria-label="Journey stops">
                {scenario.subScenarios.map((sub, i) => {
                  const st = subScenarioStats.get(sub.id) ?? { learned: 0, total: sub.wordCount };
                  const done = st.learned === st.total;
                  const active = i === activeIndex;
                  const state = done ? "done" : active ? "current" : i === nextIndex ? "next" : "todo";
                  return (
                    <li key={sub.id} className={`journey-stop is-${state} ${active ? "is-active" : ""}`}>
                      <button type="button" aria-current={active ? "step" : undefined} onClick={() => setExpandedSubId(sub.id)}>
                        <span className="journey-stop-node" aria-hidden="true">
                          {done ? "✓" : i + 1}
                        </span>
                        <span className="journey-stop-body">
                          <span className="journey-stop-name">{sub.name}</span>
                          <span className="journey-stop-bar" aria-hidden="true">
                            <span style={{ width: `${(st.learned / st.total) * 100}%` }} />
                          </span>
                          <span className="journey-stop-count">
                            {st.learned}/{st.total}
                          </span>
                        </span>
                        {active && <span className="journey-stop-here">You are here</span>}
                      </button>
                    </li>
                  );
                })}
                <li className={`journey-stop journey-finish ${nextIndex === -1 ? "is-done" : ""}`} aria-label="Finish">
                  <span className="journey-finish-node">
                    <img src={FINISH_PICTOGRAM} alt="" />
                  </span>
                  <span className="journey-stop-name">Finish</span>
                </li>
              </ol>

            <section className="scenario-path-panel journey-panel" aria-label={selectedSub.name}>
              <div className="scenario-path-panel-head">
                <div>
                  <p className="scenario-path-eyebrow">Stop {activeIndex + 1}</p>
                  <h2 className="detail-panel-title">{selectedSub.name}</h2>
                </div>
                <button
                  type="button"
                  className="scenario-path-queue"
                  disabled={toQueue.length === 0}
                  onClick={() => {
                    addAllToLearn(toQueue);
                    setQueuedVersion((v) => v + 1);
                  }}
                >
                  {toQueue.length === 0 ? "✓ All in To Learn" : `+ Add ${toQueue.length} to To Learn`}
                </button>
              </div>

              <div className="detail-words-grid scenario-path-words">
                {selectedSub.words.map((word, idx) => {
                  const signId = WORD_TO_SIGN_ID.get(word);
                  const wordLearned = signId !== undefined && isLearned(signId);
                  const wordQueued = signId !== undefined && queued.has(signId);

                  return (
                    <div
                      key={idx}
                      className="detail-word-chip-wrapper"
                      onMouseEnter={() => {
                        if (signId !== undefined && signId !== null) {
                          setPreviewWord(word);
                          handleMouseEnter(signId);
                        }
                      }}
                      onMouseLeave={() => {
                        handleMouseLeave();
                      }}
                      style={{ cursor: "pointer" }}
                    >
                      <Link
                        to={signId !== undefined ? `/signs/${signId}` : "#"}
                        className={`detail-word-chip scenario-word-card ${wordLearned ? "learned" : ""}`}
                        state={{
                          returnTo: `/scenarios/category/${id}`,
                          selectedSubId: selectedSub.id,
                        }}
                        onClick={(e) => {
                          if (signId === undefined) {
                            e.preventDefault();
                          }
                        }}
                      >
                        <span className="scenario-word-card-gloss">{word}</span>
                        <span className="scenario-word-card-state">
                          {wordLearned ? "✓ Learned" : wordQueued ? "In To Learn" : "Not started"}
                        </span>
                      </Link>
                    </div>
                  );
                })}
              </div>
              {nextSub ? (
                <button
                  type="button"
                  className={`journey-next ${groupDone ? "is-ready" : ""}`}
                  onClick={() => setExpandedSubId(nextSub.id)}
                >
                  <span className="journey-next-arrow" aria-hidden="true">↓</span>
                  <span>
                    <span className="journey-next-label">{groupDone ? "Stop cleared — next stop" : "Next stop"}</span>
                    <span className="journey-next-name">{nextSub.name}</span>
                  </span>
                </button>
              ) : (
                groupDone && (
                  <Link to="/scenarios" className="journey-next is-ready">
                    <img className="journey-next-trophy" src={FINISH_PICTOGRAM} alt="" />
                    <span>
                      <span className="journey-next-label">Journey complete</span>
                      <span className="journey-next-name">Choose another journey</span>
                    </span>
                  </Link>
                )
              )}
            </section>
            </div>
          </>
        );
      })()}

      <WordPreviewModal
        signId={previewSignId}
        word={previewWord}
        onClose={() => {
          setPreviewSignId(null);
          setIsPreviewLoaded(false);
        }}
        onLoadingStateChange={(isLoaded) => {
          setIsPreviewLoaded(isLoaded);
        }}
      />
    </div>
  );
}

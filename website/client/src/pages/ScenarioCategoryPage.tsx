import { useState, useMemo, useEffect } from "react";
import { Link, useParams, useLocation } from "react-router-dom";
import { getScenarioById } from "../lib/scenarioCategories";
import { WORD_TO_SIGN_ID } from "../lib/wordToSignId";
import { isLearned } from "../lib/learnedSigns";
import { useHoverPreview } from "../hooks/useHoverPreview";
import { WordPreviewModal } from "../components/WordPreviewModal";

const CATEGORY_NAMES = {
  A: "Daily Life",
  B: "Professional",
  C: "Language Basics",
};

const CATEGORY_COLORS = {
  A: "#FF6B6B",
  B: "#4ECDC4",
  C: "#45B7D1",
};

export function ScenarioCategoryPage() {
  const { id } = useParams<{ id: string }>();
  const location = useLocation();
  const scenario = id ? getScenarioById(id) : undefined;
  const [expandedSubId, setExpandedSubId] = useState<string | null>(null);
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

      <header className="scenario-category-header" style={{ borderColor: CATEGORY_COLORS[scenario.category] }}>
        <div className="category-header-content">
          <div className="breadcrumb">
            <span className="breadcrumb-item">{CATEGORY_NAMES[scenario.category]}</span>
            <span className="breadcrumb-sep">/</span>
            <span className="breadcrumb-item current">{scenario.id}</span>
          </div>
          <h1 className="category-title">{scenario.title}</h1>
          <p className="category-description">
            Explore {scenario.subScenarios.length} sub-categories with {scenario.wordCount} signs related to this topic.
          </p>
        </div>

        <div className="category-header-stats">
          <div className="stat-block">
            <span className="stat-number">{scenario.wordCount}</span>
            <span className="stat-label">Signs</span>
          </div>
          <div className="stat-block">
            <span className="stat-number">{scenario.subScenarios.length}</span>
            <span className="stat-label">Sub-categories</span>
          </div>
        </div>
      </header>

      <div className="scenario-category-content">
        <div className="scenario-sidebar-layout">
          <div className="scenario-sidebar">
            <div className="scenario-sidebar-list">
              {scenario.subScenarios.map((sub) => (
                <button
                  key={sub.id}
                  className={`scenario-sidebar-item ${expandedSubId === sub.id ? "active" : ""}`}
                  onClick={() => setExpandedSubId(sub.id)}
                >
                  <div className="sidebar-item-header">
                    <h3 className="sidebar-item-title">{sub.name}</h3>
                    <span className="sidebar-item-id">{sub.id}</span>
                  </div>
                  <div className="sidebar-item-meta">
                    <span className="sidebar-item-stats">{sub.wordCount} signs</span>
                    {subScenarioStats.has(sub.id) && (
                      <span className="sidebar-item-learned">
                        {subScenarioStats.get(sub.id)!.learned}/{subScenarioStats.get(sub.id)!.total}
                      </span>
                    )}
                  </div>
                </button>
              ))}
            </div>
          </div>

          <div className="scenario-main-content">
            {expandedSubId ? (
              (() => {
                const selectedSub = scenario.subScenarios.find((s) => s.id === expandedSubId);
                return selectedSub ? (
                  <div className="scenario-detail-panel">
                    <div className="detail-panel-header">
                      <h2 className="detail-panel-title">{selectedSub.name}</h2>
                      <span className="detail-panel-id">{selectedSub.id}</span>
                    </div>
                    <div className="detail-panel-info">
                      <span className="detail-info-item">{selectedSub.wordCount} signs</span>
                    </div>
                    <div className="detail-panel-words">
                      <h3 className="detail-words-title">Words</h3>
                      <div className="detail-words-grid">
                        {selectedSub.words.map((word, idx) => {
                          const signId = WORD_TO_SIGN_ID.get(word);
                          const wordLearned = signId !== undefined && isLearned(signId);
                          
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
                                className={`detail-word-chip ${wordLearned ? "learned" : ""}`}
                                state={{ 
                                  returnTo: `/scenarios/category/${id}`,
                                  selectedSubId: selectedSub.id
                                }}
                                onClick={(e) => {
                                  if (signId === undefined) {
                                    e.preventDefault();
                                  }
                                }}
                              >
                                {word}
                              </Link>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  </div>
                ) : null;
              })()
            ) : (
              <div className="scenario-empty-state">
                <p>Select a sub-category from the left to view details.</p>
              </div>
            )}
          </div>
        </div>
      </div>

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

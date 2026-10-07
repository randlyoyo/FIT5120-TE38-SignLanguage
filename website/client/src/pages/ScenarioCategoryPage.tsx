import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import { getScenarioById } from "../lib/scenarioCategories";
import { WORD_TO_SIGN_ID } from "../lib/wordToSignId";

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
  const scenario = id ? getScenarioById(id) : undefined;
  const [expandedSubId, setExpandedSubId] = useState<string | null>(null);

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
        <div className="subcategory-grid">
          {scenario.subScenarios.map((sub) => (
            <div
              key={sub.id}
              className={`subcategory-card ${expandedSubId === sub.id ? "expanded" : ""}`}
              onClick={() => setExpandedSubId(expandedSubId === sub.id ? null : sub.id)}
              role="button"
              tabIndex={0}
            >
              <div className="subcategory-header">
                <h3 className="subcategory-title">{sub.name}</h3>
                <span className="subcategory-id">{sub.id}</span>
              </div>
              <div className="subcategory-stats">
                <span className="word-badge">{sub.wordCount} signs</span>
              </div>

              {expandedSubId === sub.id && (
                <div className="subcategory-words">
                  <div className="words-grid">
                    {sub.words.map((word, idx) => {
                      const signId = WORD_TO_SIGN_ID.get(word);
                      return (
                        <Link
                          key={idx}
                          to={`/signs/${signId}`}
                          className="word-chip"
                          state={{ returnTo: `/scenarios/category/${id}` }}
                        >
                          {word}
                        </Link>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

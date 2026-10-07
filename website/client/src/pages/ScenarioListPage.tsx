import { Link } from "react-router-dom";
import { SCENARIO_CATEGORIES, SCENARIO_STATS, type ScenarioCategory } from "../lib/scenarioCategories";

const CATEGORY_EMOJIS: Record<ScenarioCategory, string> = {
  A: "🏠",
  B: "💼",
  C: "📝",
};

const CATEGORY_DESCRIPTIONS: Record<ScenarioCategory, string> = {
  A: "Real-life everyday situations",
  B: "Professional and specialized topics",
  C: "Grammar and abstract concepts",
};

function CategoryCard({ group }: { group: typeof SCENARIO_CATEGORIES[0] }) {
  return (
    <Link to={`/scenarios/category/${group.id}`} className="category-card">
      <div className="category-card-icon">{group.icon}</div>
      <h3 className="category-card-title">{group.title}</h3>
      <div className="category-card-meta">
        <span className="category-badge">{group.id}</span>
        <span className="word-count">{group.wordCount} words</span>
      </div>
      <div className="category-card-subs">
        {group.subScenarios.map((sub) => (
          <span key={sub.id} className="sub-badge">
            {sub.name}
          </span>
        ))}
      </div>
    </Link>
  );
}

function CategorySection({ category, title, description }: { category: ScenarioCategory; title: string; description: string }) {
  const scenarios = SCENARIO_CATEGORIES.filter((s) => s.category === category);
  const scenarioCount = scenarios.length;
  const wordCount = scenarios.reduce((sum, s) => sum + s.wordCount, 0);

  return (
    <section className="scenario-category-section">
      <div className="category-section-header">
        <div>
          <span className="category-emoji" aria-hidden="true">
            {CATEGORY_EMOJIS[category]}
          </span>
          <h2 className="category-section-title">{title}</h2>
          <p className="category-section-description">{description}</p>
        </div>
        <div className="category-stats">
          <div className="stat">
            <span className="stat-value">{scenarioCount}</span>
            <span className="stat-label">Scenarios</span>
          </div>
          <div className="stat">
            <span className="stat-value">{wordCount}</span>
            <span className="stat-label">Words</span>
          </div>
        </div>
      </div>

      <div className="category-grid">
        {scenarios.map((group) => (
          <CategoryCard key={group.id} group={group} />
        ))}
      </div>
    </section>
  );
}

/** Scenario Learning index -- browse the scenario library organized by topic.
 *  Explore everyday situations, professional topics, and grammar concepts. */
export function ScenarioListPage() {
  return (
    <>
      <header className="library-hero-band home-hero-band">
        <div className="library-hero-inner">
          <div>
            <p className="library-eyebrow-light">Index</p>
            <h1 className="page-title">Scenario Learning</h1>
          </div>
          <p className="results-count library-hero-count">
            {SCENARIO_STATS.totalCategories} scenarios
          </p>
        </div>
      </header>

      <div className="page-container scenario-list-page">
        <p className="scenario-list-intro">
          Browse the complete scenario library organized by topic. Explore everyday situations, professional topics,
          and grammar concepts.
        </p>
        <div className="scenario-categories-container">
          <CategorySection
            category="A"
            title="Daily Life"
            description={CATEGORY_DESCRIPTIONS.A}
          />
          <CategorySection
            category="B"
            title="Professional"
            description={CATEGORY_DESCRIPTIONS.B}
          />
          <CategorySection
            category="C"
            title="Language Basics"
            description={CATEGORY_DESCRIPTIONS.C}
          />
        </div>
      </div>
    </>
  );
}

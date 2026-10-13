import { Link } from "react-router-dom";
import { SCENARIO_CATEGORIES, SCENARIO_STATS, type ScenarioCategory, type ScenarioGroup } from "../lib/scenarioCategories";
import { WORD_TO_SIGN_ID } from "../lib/wordToSignId";
import { isLearned } from "../lib/learnedSigns";
import { SECTION_PICTOGRAMS, scenarioPictogram } from "../lib/pictograms";

const CATEGORY_DESCRIPTIONS: Record<ScenarioCategory, string> = {
  A: "Real-life everyday situations",
  B: "Professional and specialized topics",
  C: "Grammar and abstract concepts",
};

/** Friendly first journeys for someone who hasn't started any. */
const STARTER_IDS = ["A15", "A14", "A01"];

function learnedIn(group: ScenarioGroup): number {
  let count = 0;
  for (const sub of group.subScenarios) {
    for (const word of sub.words) {
      const signId = WORD_TO_SIGN_ID.get(word);
      if (signId !== undefined && isLearned(signId)) count++;
    }
  }
  return count;
}

function JourneyCard({ group, learned }: { group: ScenarioGroup; learned: number }) {
  const pct = (learned / group.wordCount) * 100;
  const status = learned === 0 ? "Not started" : learned >= group.wordCount ? "Complete" : "In progress";

  return (
    <Link to={`/scenarios/category/${group.id}`} className="category-card journey-card">
      <img className="category-card-icon" src={scenarioPictogram(group.id)} alt="" />
      <h3 className="category-card-title">{group.title}</h3>
      <div className="journey-card-progress">
        <span className="journey-card-bar" aria-hidden="true">
          <span style={{ width: `${pct}%` }} />
        </span>
        <span className={`journey-card-status is-${status.replace(" ", "-").toLowerCase()}`}>
          {status} · {learned}/{group.wordCount}
        </span>
      </div>
    </Link>
  );
}

function CategorySection({ category, title, learnedById }: { category: ScenarioCategory; title: string; learnedById: Map<string, number> }) {
  const scenarios = SCENARIO_CATEGORIES.filter((s) => s.category === category);
  const wordCount = scenarios.reduce((sum, s) => sum + s.wordCount, 0);

  return (
    <section className="scenario-category-section">
      <div className="category-section-header">
        <div>
          <img className="category-emoji" src={SECTION_PICTOGRAMS[category]} alt="" />
          <h2 className="category-section-title">{title}</h2>
          <p className="category-section-description">{CATEGORY_DESCRIPTIONS[category]}</p>
        </div>
        <div className="category-stats">
          <div className="stat">
            <span className="stat-value">{scenarios.length}</span>
            <span className="stat-label">Journeys</span>
          </div>
          <div className="stat">
            <span className="stat-value">{wordCount}</span>
            <span className="stat-label">Signs</span>
          </div>
        </div>
      </div>

      <div className="category-grid">
        {scenarios.map((group) => (
          <JourneyCard key={group.id} group={group} learned={learnedById.get(group.id) ?? 0} />
        ))}
      </div>
    </section>
  );
}

/** Scenario Learning index, framed as learning journeys: what a journey is
 *  and how to do one, where to pick up (or start), then every journey with
 *  its own progress. */
export function ScenarioListPage() {
  const learnedById = new Map(SCENARIO_CATEGORIES.map((g) => [g.id, learnedIn(g)]));
  // Pick up the journey that's furthest along but not finished.
  const inProgress = SCENARIO_CATEGORIES.filter((g) => {
    const n = learnedById.get(g.id) ?? 0;
    return n > 0 && n < g.wordCount;
  }).sort((a, b) => (learnedById.get(b.id)! / b.wordCount) - (learnedById.get(a.id)! / a.wordCount));
  const current = inProgress[0];
  const starters = STARTER_IDS.map((id) => SCENARIO_CATEGORIES.find((g) => g.id === id)).filter(
    (g): g is ScenarioGroup => g !== undefined
  );

  return (
    <>
      <header className="library-hero-band home-hero-band">
        <div className="library-hero-inner">
          <div>
            <p className="library-eyebrow-light">Learning journeys</p>
            <h1 className="page-title">Scenario Learning</h1>
          </div>
          <p className="results-count library-hero-count">{SCENARIO_STATS.totalCategories} journeys</p>
        </div>
      </header>

      <div className="page-container scenario-list-page">
        <section className="journey-intro" aria-label="How a journey works">
          <p className="journey-intro-lead">
            Learn the signs you need for real situations — ordering a coffee, seeing a doctor, meeting family —
            one small step at a time.
          </p>
          <ol className="journey-steps">
            <li>
              <span className="journey-step-num">1</span>
              <strong>Pick a situation</strong>
              <span>Choose a journey below that matches something you want to sign.</span>
            </li>
            <li>
              <span className="journey-step-num">2</span>
              <strong>Go stop by stop</strong>
              <span>Each journey is split into short stops of related signs.</span>
            </li>
            <li>
              <span className="journey-step-num">3</span>
              <strong>Watch, practise, mark learned</strong>
              <span>Open each sign, try it, and mark it learned — your progress fills in as you go.</span>
            </li>
          </ol>
        </section>

        {current ? (
          <Link to={`/scenarios/category/${current.id}`} className="journey-continue">
            <img className="journey-continue-icon" src={scenarioPictogram(current.id)} alt="" />
            <span className="journey-continue-text">
              <span className="journey-continue-label">Continue your journey</span>
              <span className="journey-continue-title">{current.title}</span>
              <span className="journey-continue-meta">
                {learnedById.get(current.id)}/{current.wordCount} signs learned
              </span>
            </span>
            <span className="journey-continue-go" aria-hidden="true">
              Continue →
            </span>
          </Link>
        ) : (
          <section className="journey-starters" aria-label="Good first journeys">
            <p className="journey-starters-label">Not sure where to begin? Start with one of these:</p>
            <div className="journey-starters-list">
              {starters.map((g) => (
                <Link key={g.id} to={`/scenarios/category/${g.id}`} className="journey-starter">
                  <img src={scenarioPictogram(g.id)} alt="" /> {g.title}
                  <span className="journey-starter-go" aria-hidden="true">
                    →
                  </span>
                </Link>
              ))}
            </div>
          </section>
        )}

        <div className="scenario-categories-container">
          <CategorySection category="A" title="Daily Life" learnedById={learnedById} />
          <CategorySection category="B" title="Professional" learnedById={learnedById} />
          <CategorySection category="C" title="Language Basics" learnedById={learnedById} />
        </div>
        <p className="pictogram-credit">
          Pictograms: <a href="https://pictogram2.com/" target="_blank" rel="noreferrer">Human Pictogram 2.0</a>{" "}
          by TopeconHeroes
        </p>
      </div>
    </>
  );
}

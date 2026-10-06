import { Link } from "react-router-dom";
import { SCENARIOS, type Scenario } from "../lib/scenarioStories";
import { useScenarioProgress } from "../hooks/useScenarioProgress";

const LEVEL_LABELS = {
  beginner: "Beginner",
  intermediate: "Intermediate",
  advanced: "Advanced",
};

function ScenarioCard({ scenario }: { scenario: Scenario }) {
  const { learnedCount, totalSteps } = useScenarioProgress(scenario);

  return (
    <Link to={`/scenarios/${scenario.id}`} className="scenario-card">
      <span className="scenario-card-emoji" aria-hidden="true">
        {scenario.emoji}
      </span>
      <h2 className="scenario-card-title">{scenario.title}</h2>
      <p className="scenario-card-situation">{scenario.situation}</p>
      <div className="scenario-card-footer">
        <span className={`level-badge level-badge-${scenario.level}`}>{LEVEL_LABELS[scenario.level]}</span>
        <span className={`scenario-card-progress ${learnedCount === totalSteps ? "scenario-card-progress-done" : ""}`}>
          {learnedCount}/{totalSteps} learned
        </span>
      </div>
    </Link>
  );
}

/** Scenario Learning index -- a scenario is a short story walked through
 *  step by step (ScenarioLearningPage), rather than a flat vocabulary list.
 *  Pick one here to open its teaching page. Each card's progress is real
 *  (useScenarioProgress reads lib/learnedSigns.ts), not decorative. */
export function ScenarioListPage() {
  return (
    <>
      <header className="library-hero-band home-hero-band">
        <div className="library-hero-inner">
          <div>
            <p className="library-eyebrow-light">Index</p>
            <h1 className="page-title">Scenario Learning</h1>
          </div>
          <p className="results-count library-hero-count">{SCENARIOS.length} scenarios</p>
        </div>
      </header>

      <div className="page-container scenario-list-page">
        <p className="scenario-list-intro">
          Walk through a real-life situation step by step, learning the signs you'd actually need for it.
        </p>

        <div className="scenario-card-grid">
          {SCENARIOS.map((scenario) => (
            <ScenarioCard key={scenario.id} scenario={scenario} />
          ))}
        </div>
      </div>
    </>
  );
}

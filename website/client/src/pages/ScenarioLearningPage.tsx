import { useNavigate, useParams } from "react-router-dom";
import { SCENARIOS } from "../lib/scenarioStories";
import { useScenarioProgress } from "../hooks/useScenarioProgress";
import { isLearned } from "../lib/learnedSigns";
import { ScenarioWordLink } from "../components/ScenarioWordLink";

const LEVEL_LABELS = {
  beginner: "Beginner",
  intermediate: "Intermediate",
  advanced: "Advanced",
};

function ScenarioStory({ scenario }: { scenario: (typeof SCENARIOS)[number] }) {
  const { wordIds, stepDone, learnedCount, totalSteps } = useScenarioProgress(scenario);
  const allDone = learnedCount === totalSteps;

  return (
    <>
      <div className="scenario-story-header">
        <h1 className="page-title">
          <span aria-hidden="true">{scenario.emoji}</span> {scenario.title}
        </h1>
        <span className={`level-badge level-badge-${scenario.level}`}>{LEVEL_LABELS[scenario.level]}</span>
      </div>

      <blockquote className="scenario-story-situation">{scenario.situation}</blockquote>

      <div className="scenario-story-progress">
        <p className="results-count">
          {learnedCount} of {totalSteps} steps learned
        </p>
        <div
          className="progress-bar"
          role="progressbar"
          aria-valuenow={Math.round((learnedCount / totalSteps) * 100)}
          aria-valuemin={0}
          aria-valuemax={100}
        >
          <div className="progress-bar-fill" style={{ width: `${(learnedCount / totalSteps) * 100}%` }} />
        </div>
      </div>

      <ol className="scenario-story-steps">
        {scenario.steps.map((step, i) => (
          <li key={step.title} className="scenario-story-step">
            <span
              className={`scenario-story-step-number ${stepDone[i] ? "scenario-story-step-number-done" : ""}`}
              aria-hidden="true"
            >
              {stepDone[i] ? "✓" : i + 1}
            </span>
            <div className="scenario-story-step-body">
              <h2 className="scenario-story-step-title">{step.title}</h2>
              <p className="scenario-step-phrase">“{step.phrase}”</p>
              <div className="scenario-story-words">
                {step.words.map((word) => (
                  <ScenarioWordLink
                    key={word}
                    word={word}
                    signId={wordIds[word]}
                    learned={wordIds[word] != null && isLearned(wordIds[word]!)}
                  />
                ))}
              </div>
              {!stepDone[i] && (
                <p className="scenario-step-hint">
                  Open a word above and mark it "Learned" on its Practice tab to check off this step.
                </p>
              )}
            </div>
          </li>
        ))}
      </ol>

      {allDone && (
        <p className="scenario-story-complete">
          <span aria-hidden="true">✓</span> Scenario complete
        </p>
      )}
    </>
  );
}

/**
 * Scenario Learning: a scenario is a short story walked through step by
 * step (team's own scenario-design notes), each step teaching a handful of
 * signs, rather than a flat vocabulary list behind a difficulty filter.
 * Each step's words link straight to that sign's own detail page
 * (ScenarioWordLink), where the real Practice tab (camera + recognition
 * model) is what actually checks a signer got it right -- this page never
 * re-implements that, it only reads its result. A step counts as learned
 * once every one of its words has been marked "Learned" there
 * (useScenarioProgress / lib/learnedSigns.ts, the same flag the Learned
 * page itself reads), so progress here is real and shared with the rest of
 * the site, not a separate tracker that can drift from it.
 */
export function ScenarioLearningPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const scenario = SCENARIOS.find((s) => s.id === id);

  return (
    <div className="page-container scenario-story-page">
      <button type="button" className="back-link" onClick={() => navigate("/scenarios")}>
        ↤ Back to scenarios
      </button>
      {scenario ? (
        <ScenarioStory scenario={scenario} />
      ) : (
        <p role="alert">Couldn't find that scenario.</p>
      )}
    </div>
  );
}

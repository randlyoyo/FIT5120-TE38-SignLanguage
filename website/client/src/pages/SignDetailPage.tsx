import { useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { fetchRecognitionVocabulary } from "../api/recognize";
import { fetchSignById, fetchSigns } from "../api/signs";
import { PracticeVerify } from "../components/PracticeVerify";
import { ResultCard } from "../components/ResultCard";
import { SignDemonstration } from "../components/SignDemonstration";
import { isLearned, toggleLearned } from "../lib/learnedSigns";
import { tagChipStyle } from "../lib/tagColors";
import { isToLearn, removeFromToLearn, toggleToLearn } from "../lib/toLearnSigns";
import type { Sign } from "../api/types";

export function SignDetailPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const signId = Number(id);

  const [sign, setSign] = useState<Sign | null>(null);
  const [isError, setIsError] = useState(false);
  const [learned, setLearned] = useState(false);
  const [queuedToLearn, setQueuedToLearn] = useState(false);
  // Hidden rather than shown-with-an-error for a sign outside the
  // recognizer's closed vocabulary (API.md §9), and equally hidden if the
  // recognizer isn't deployed at all yet -- fetchRecognitionVocabulary()
  // resolves to an empty set rather than rejecting in that case, so both
  // "not supported" and "not ready" read the same way here: no button.
  const [canPractice, setCanPractice] = useState(false);
  // Learn is the entry point for every sign; Practice only exists once
  // canPractice is known, so it never opens on a sign this resets back to
  // Learn for below.
  const [mode, setMode] = useState<"learn" | "practice">("learn");
  const [related, setRelated] = useState<Sign[]>([]);

  useEffect(() => {
    if (!Number.isInteger(signId)) return;
    const controller = new AbortController();
    setSign(null);
    setIsError(false);
    setMode("learn");
    fetchSignById(signId, controller.signal)
      .then((s) => {
        setSign(s);
        setLearned(isLearned(s.id));
        setQueuedToLearn(isToLearn(s.id));
      })
      .catch((err) => {
        if (err.name !== "AbortError") setIsError(true);
      });
    return () => controller.abort();
  }, [signId]);

  // "Related signs" (Auslan Signbank calls this "Semantically related
  // signs") -- other entries sharing the sign's primary category tag, the
  // same grouping already visible to the user as tag chips.
  useEffect(() => {
    setRelated([]);
    const firstTag = sign?.tags[0];
    if (!firstTag) return;
    const controller = new AbortController();
    fetchSigns({ tag: firstTag, pageSize: 7, signal: controller.signal })
      .then((res) => setRelated(res.results.filter((s) => s.id !== sign!.id).slice(0, 6)))
      .catch((err) => {
        if (err.name !== "AbortError") console.error("Failed to load related signs:", err);
      });
    return () => controller.abort();
  }, [sign]);

  useEffect(() => {
    let active = true;
    fetchRecognitionVocabulary().then((vocabulary) => {
      if (active) {
        const supported = vocabulary.has(sign?.gloss ?? "");
        setCanPractice(supported);
        if (!supported) setMode("learn");
      }
    });
    return () => {
      active = false;
    };
  }, [sign?.gloss]);

  if (isError) {
    return (
      <div className="page-container">
        <p role="alert">Couldn't load this sign.</p>
        <button type="button" className="back-link" onClick={() => navigate(-1)}>
          &larr; Back to library
        </button>
      </div>
    );
  }

  if (!sign) {
    return (
      <div className="page-container">
        <p>Loading…</p>
      </div>
    );
  }

  return (
    <div className="page-container">
      {/* Fixed to the viewport edges rather than the layout, so they stay
          reachable at a glance regardless of scroll position -- a gallery-
          style "next/previous entry" control, not part of the card itself.
          Previous clamps at the first catalogue entry; next has no known
          upper bound here, so it relies on the existing not-found handling
          if it ever runs past the last one. */}
      <button
        type="button"
        className="detail-nav-arrow prev"
        onClick={() => navigate(`/signs/${sign.id - 1}`)}
        disabled={sign.id <= 1}
        aria-label="Previous sign"
      >
        <svg width="120" height="120" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
          <polyline points="10.5,3 7,12 10.5,21" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </button>
      <button
        type="button"
        className="detail-nav-arrow next"
        onClick={() => navigate(`/signs/${sign.id + 1}`)}
        aria-label="Next sign"
      >
        <svg width="120" height="120" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
          <polyline points="13.5,3 17,12 13.5,21" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </button>

      <div className="detail-layout">
        <div className="detail-back">
          <button type="button" className="back-link" onClick={() => navigate(-1)}>
            &larr; Back to library
          </button>
          <div className="detail-back-right">
            {canPractice && (
              <div className="detail-mode-toggle" role="tablist" aria-label="Learn or practice this sign">
                <button
                  type="button"
                  role="tab"
                  aria-selected={mode === "learn"}
                  className={`detail-mode-tab ${mode === "learn" ? "active" : ""}`}
                  onClick={() => setMode("learn")}
                >
                  Learn
                </button>
                <button
                  type="button"
                  role="tab"
                  aria-selected={mode === "practice"}
                  className={`detail-mode-tab ${mode === "practice" ? "active" : ""}`}
                  onClick={() => setMode("practice")}
                >
                  Practice
                </button>
              </div>
            )}
            <span className="catalog-number-inline">No. {sign.id}</span>
          </div>
        </div>

        <div className="detail-media">
          <SignDemonstration gloss={sign.gloss} videos={sign.videos ?? []} />
        </div>

        {/* Keyed by mode so switching Learn <-> Practice remounts this panel
            and replays its entrance animation. Practice mode replaces this
            slot with nothing but the capture stage -- no title, tags, or
            "mark as learned" -- so its top edge lines up with the demo
            video's, instead of trailing behind it under a heading. */}
        <div className="detail-title">
          <div key={mode} className="detail-mode-panel">
            {mode === "learn" ? (
              <>
                <h1 className="page-title detail-gloss-title">{sign.gloss}</h1>
                <p className="result-card-meta">{sign.source ?? "Unknown source"}</p>
                {sign.tags.length > 0 && (
                  <div className="tag-chips">
                    {sign.tags.map((tag) => (
                      <Link
                        key={tag}
                        to={`/library?tag=${encodeURIComponent(tag)}`}
                        className="tag-chip"
                        style={tagChipStyle(tag)}
                      >
                        #{tag}
                      </Link>
                    ))}
                  </div>
                )}

                <div className="detail-list-actions">
                  <button
                    type="button"
                    className={`learned-toggle to-learn-toggle ${queuedToLearn ? "active" : ""}`}
                    onClick={() => setQueuedToLearn(toggleToLearn(sign.id))}
                  >
                    {queuedToLearn ? "✓ In to-learn list" : "Add to to-learn list"}
                  </button>
                  <button
                    type="button"
                    className={`learned-toggle ${learned ? "learned" : ""}`}
                    onClick={() => {
                      const next = toggleLearned(sign.id);
                      setLearned(next);
                      // A sign just marked learned no longer needs to be
                      // queued to learn -- the two lists are meant to be
                      // mutually exclusive, not two independent flags.
                      if (next && queuedToLearn) {
                        removeFromToLearn(sign.id);
                        setQueuedToLearn(false);
                      }
                    }}
                  >
                    {learned ? "✓ Learned" : "Mark as learned"}
                  </button>
                </div>
              </>
            ) : (
              <PracticeVerify gloss={sign.gloss} />
            )}
          </div>
        </div>

        {mode === "learn" && (
          <div className="detail-definitions-area">
            <div className="detail-mode-panel">
              <h2 className="sign-detail-heading">Sign Definition</h2>
              <div className="definitions">
                {sign.definitions.map((group) => (
                  <div key={group.partOfSpeech} className="definition-group">
                    <h3 className="definition-pos">{group.partOfSpeech}</h3>
                    <ol className="definition-senses">
                      {group.senses.map((sense, i) => (
                        <li key={i}>{sense}</li>
                      ))}
                    </ol>
                  </div>
                ))}
              </div>

              {/* The movement description (same text across a sign's video
                  sources, per recognition/API.md's video import) is the real,
                  per-sign answer to "how is this formed" -- usageNotes below
                  is the generic fallback for signs the description hasn't
                  reached yet. */}
              {sign.videos?.[0]?.movementDescription ? (
                <>
                  <h2 className="sign-detail-heading">How It's Formed</h2>
                  <div className="movement-description">
                    <p>{sign.videos[0].movementDescription}</p>
                  </div>
                </>
              ) : (
                sign.usageNotes.length > 0 && (
                  <>
                    <h2 className="sign-detail-heading">How It's Formed</h2>
                    <ol className="usage-steps">
                      {sign.usageNotes.map((note, i) => (
                        <li key={i}>{note}</li>
                      ))}
                    </ol>
                  </>
                )
              )}
            </div>
          </div>
        )}
      </div>

      {related.length > 0 && (
        <div className="detail-related">
          <h2 className="sign-detail-heading">Related Signs</h2>
          <ul className="result-list">
            {related.map((s) => (
              <ResultCard key={s.id} sign={s} />
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

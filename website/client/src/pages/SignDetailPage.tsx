import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { fetchSignById } from "../api/signs";
import { SignDemonstration } from "../components/SignDemonstration";
import { isLearned, toggleLearned } from "../lib/learnedSigns";
import type { Sign } from "../api/types";

export function SignDetailPage() {
  const { id } = useParams<{ id: string }>();
  const signId = Number(id);

  const [sign, setSign] = useState<Sign | null>(null);
  const [isError, setIsError] = useState(false);
  const [learned, setLearned] = useState(false);

  useEffect(() => {
    if (!Number.isInteger(signId)) return;
    const controller = new AbortController();
    setSign(null);
    setIsError(false);
    fetchSignById(signId, controller.signal)
      .then((s) => {
        setSign(s);
        setLearned(isLearned(s.id));
      })
      .catch((err) => {
        if (err.name !== "AbortError") setIsError(true);
      });
    return () => controller.abort();
  }, [signId]);

  if (isError) {
    return (
      <div className="page-container">
        <p role="alert">Couldn't load this sign.</p>
        <Link to="/">&larr; Back to library</Link>
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
      <p>
        <Link to="/">&larr; Back to library</Link>
      </p>

      <h1 className="page-title" style={{ fontSize: "2.25rem" }}>
        {sign.gloss}
      </h1>
      <p className="result-card-meta" style={{ textAlign: "center" }}>
        {sign.source ?? "Unknown source"}
      </p>
      {sign.tags.length > 0 && (
        <div className="tag-chips" style={{ justifyContent: "center" }}>
          {sign.tags.map((tag) => (
            <span key={tag} className="tag-chip">
              #{tag}
            </span>
          ))}
        </div>
      )}

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

      <SignDemonstration
       gloss={sign.gloss}
       steps={sign.usageNotes}
       videos={sign.videos ?? []}
      />

      <h2 className="sign-detail-heading">How to sign it</h2>
      <ol className="usage-notes sign-detail-steps">
        {sign.usageNotes.map((note, i) => (
          <li key={i}>{note}</li>
        ))}
      </ol>

      <button
        type="button"
        className={`learned-toggle ${learned ? "learned" : ""}`}
        onClick={() => setLearned(toggleLearned(sign.id))}
      >
        {learned ? "✓ Learned" : "Mark as learned"}
      </button>
    </div>
  );
}

import { useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { fetchSignById } from "../api/signs";
import { SignDemonstration } from "../components/SignDemonstration";
import { isLearned, toggleLearned } from "../lib/learnedSigns";
import { tagChipStyle } from "../lib/tagColors";
import type { Sign } from "../api/types";

export function SignDetailPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
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
      <div className="detail-layout">
        <p className="detail-back">
          <button type="button" className="back-link" onClick={() => navigate(-1)}>
            &larr; Back to library
          </button>
          <span className="catalog-number-inline">No. {sign.id}</span>
        </p>

        <div className="detail-media">
          <SignDemonstration gloss={sign.gloss} videos={sign.videos ?? []} />
        </div>

        <div className="detail-title">
          <h1 className="page-title" style={{ fontSize: "2.1rem" }}>
            {sign.gloss}
          </h1>
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

          <button
            type="button"
            className={`learned-toggle ${learned ? "learned" : ""}`}
            onClick={() => setLearned(toggleLearned(sign.id))}
          >
            {learned ? "✓ Learned" : "Mark as learned"}
          </button>
        </div>

        <div className="detail-definitions-area">
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
        </div>
      </div>
    </div>
  );
}

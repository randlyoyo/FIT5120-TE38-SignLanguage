import { Link } from "react-router-dom";
import type { Sign } from "../api/types";
import { isLearned } from "../lib/learnedSigns";
import { PlaceholderMedia } from "./PlaceholderMedia";

interface Props {
  sign: Sign;
}

/** First sense of the first definition group, for a compact card preview. */
function primarySense(sign: Sign): string | null {
  return sign.definitions[0]?.senses[0] ?? null;
}

export function ResultCard({ sign }: Props) {
  const preview = primarySense(sign);

  return (
    <li>
      <Link to={`/signs/${sign.id}`} className="result-card">
        <div className="result-card-media">
          <PlaceholderMedia seed={sign.id} gloss={sign.gloss} />
        </div>
        <div className="result-card-body">
          <h3 className="result-card-title">{sign.gloss}</h3>
          <p className="result-card-meta">
            {sign.source ?? "Unknown source"} &middot; #{sign.id}
          </p>
          {sign.tags.length > 0 && (
            <div className="tag-chips">
              {sign.tags.map((tag) => (
                <span key={tag} className="tag-chip">
                  #{tag}
                </span>
              ))}
            </div>
          )}
          {preview && <p className="result-card-preview">{preview}</p>}
          <ol className="usage-notes">
            {sign.usageNotes.slice(0, 2).map((note, i) => (
              <li key={i}>{note}</li>
            ))}
          </ol>
          {isLearned(sign.id) && (
            <div className="result-card-badges">
              <span className="learned-badge">&#10003; Learned</span>
            </div>
          )}
        </div>
      </Link>
    </li>
  );
}

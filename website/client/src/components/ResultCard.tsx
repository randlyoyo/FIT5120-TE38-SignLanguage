import { Link } from "react-router-dom";
import type { Sign } from "../api/types";
import { isLearned } from "../lib/learnedSigns";
import { tagChipStyle } from "../lib/tagColors";
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
    <li className="result-card">
      <Link to={`/signs/${sign.id}`} className="result-card-link">
        <div className="result-card-media">
          <PlaceholderMedia seed={sign.id} gloss={sign.gloss} />
        </div>
        <div className="result-card-body">
          <div className="result-card-top-row">
            <h3 className="result-card-title">{sign.gloss}</h3>
            {isLearned(sign.id) && (
              <span className="learned-badge">&#10003; Learned</span>
            )}
          </div>
          <p className="result-card-meta">
            {sign.source ?? "Unknown source"} &middot; #{sign.id}
          </p>
          {preview && <p className="result-card-preview">{preview}</p>}
        </div>
      </Link>
      {sign.tags.length > 0 && (
        <div className="tag-chips">
          {sign.tags.map((tag) => (
            <Link
              key={tag}
              to={`/?tag=${encodeURIComponent(tag)}`}
              className="tag-chip"
              style={tagChipStyle(tag)}
              onClick={(e) => e.stopPropagation()}
            >
              #{tag}
            </Link>
          ))}
        </div>
      )}
    </li>
  );
}

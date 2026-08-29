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
  const previewVideoUrl = sign.previewVideo?.videoUrl;

  return (
    <li className="result-card">
      <Link to={`/signs/${sign.id}`} className="result-card-link">
        <div className="result-card-media">
          {previewVideoUrl ? (
            <video
              className="result-card-video"
              src={previewVideoUrl}
              autoPlay
              muted
              loop
              playsInline
              preload="auto"
              aria-label={`${sign.gloss} Auslan demonstration preview`}
            />
          ) : (
            <PlaceholderMedia seed={sign.id} gloss={sign.gloss} />
          )}
          <span className="catalog-number">No. {sign.id}</span>
          {isLearned(sign.id) && (
            <span className="learned-badge">&#10003; Learned</span>
          )}
        </div>
        <div className="result-card-body">
          <h3 className="result-card-title">{sign.gloss}</h3>
          <p className="result-card-meta">{sign.source ?? "Unknown source"}</p>
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

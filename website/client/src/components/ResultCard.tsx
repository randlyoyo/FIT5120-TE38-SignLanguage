import { Link } from "react-router-dom";
import type { Sign } from "../api/types";
import { isLearned } from "../lib/learnedSigns";
import { tagChipStyle } from "../lib/tagColors";
import { PlaceholderMedia } from "./PlaceholderMedia";

interface Props {
  sign: Sign;
  /** The ordered ids of the list this card is part of (search results,
   *  a tag page, related signs, ...), so the detail page's prev/next
   *  arrows can step through what the learner actually browsed instead
   *  of the raw database id sequence. Omit where there's no meaningful
   *  order to hand off. */
  siblingIds?: number[];
  /** This list's own URL (path + query string), so "Back to library"
   *  still lands there after the learner has stepped through several
   *  signs with prev/next -- otherwise it's one history entry per click,
   *  and "back" just undoes the last arrow instead of leaving the list. */
  returnTo?: string;
}

/** First sense of the first definition group, for a compact card preview. */
function primarySense(sign: Sign): string | null {
  return sign.definitions[0]?.senses[0] ?? null;
}

export function ResultCard({ sign, siblingIds, returnTo }: Props) {
  const preview = primarySense(sign);
  // The list endpoint sets `previewVideo`; the single-sign endpoint (used by
  // e.g. the Learned page, which fetches signs by id) sets `videos` instead --
  // fall back to its first entry so both shapes render a preview here.
  const previewVideoUrl = sign.previewVideo?.videoUrl ?? sign.videos?.[0]?.videoUrl;

  return (
    <li className="result-card">
      <Link
        to={`/signs/${sign.id}`}
        className="result-card-link"
        state={siblingIds || returnTo ? { siblingIds, returnTo } : undefined}
      >
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
              to={`/library?tag=${encodeURIComponent(tag)}`}
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

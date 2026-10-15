import { useState, type MouseEvent } from "react";
import { Link } from "react-router-dom";
import type { Sign } from "../api/types";
import type { Selection } from "../hooks/useSelection";
import { isLearned, removeLearned } from "../lib/learnedSigns";
import { addToLearn, markLearnt } from "../lib/signLists";
import { tagChipStyle } from "../lib/tagColors";
import { isToLearn, removeFromToLearn } from "../lib/toLearnSigns";
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
  /** While a bulk selection is active, clicking the card selects it instead of opening it. */
  selection?: Selection;
  /** Called after the card's own To Learn / Learnt buttons change a list. */
  onListsChange?: () => void;
}

/** First sense of the first definition group, for a compact card preview. */
function primarySense(sign: Sign): string | null {
  return sign.definitions[0]?.senses[0] ?? null;
}

export function ResultCard({ sign, siblingIds, returnTo, selection, onListsChange }: Props) {
  const [, rerender] = useState(0);
  const preview = primarySense(sign);
  // The list endpoint sets `previewVideo`; the single-sign endpoint (used by
  // e.g. the Learnt page, which fetches signs by id) sets `videos` instead --
  // fall back to its first entry so both shapes render a preview here.
  const previewVideoUrl = sign.previewVideo?.videoUrl ?? sign.videos?.[0]?.videoUrl;
  const learned = isLearned(sign.id);
  const queued = isToLearn(sign.id);
  const selecting = selection?.active ?? false;
  const selected = selection?.selected.has(sign.id) ?? false;
  const linkState = siblingIds || returnTo ? { siblingIds, returnTo } : undefined;

  function onCardClick(e: MouseEvent) {
    if (!selecting) return;
    e.preventDefault();
    selection!.toggle(sign.id);
  }

  function quick(change: () => void) {
    change();
    rerender((n) => n + 1);
    onListsChange?.();
  }

  return (
    <li className={`result-card ${selecting ? "is-selecting" : ""} ${selected ? "is-selected" : ""}`}>
      <div className="result-card-media">
        <Link to={`/signs/${sign.id}`} state={linkState} onClick={onCardClick} tabIndex={-1} aria-hidden="true">
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
        </Link>
        {sign.level && <span className={`level-badge level-badge-${sign.level} level-badge-on-media`}>{sign.level}</span>}
        <span className="catalog-number">No. {sign.id}</span>
        {selecting ? (
          <span className={`result-card-check ${selected ? "checked" : ""}`} aria-hidden="true">
            {selected ? "✓" : ""}
          </span>
        ) : learned ? (
          <span className="learned-badge">&#10003; Learnt</span>
        ) : (
          queued && <span className="learned-badge to-learn-badge">To Learn</span>
        )}
      </div>
      <Link
        to={`/signs/${sign.id}`}
        className="result-card-link"
        state={linkState}
        onClick={onCardClick}
        aria-pressed={selecting ? selected : undefined}
      >
        <div className="result-card-body">
          <h3 className="result-card-title">{sign.gloss}</h3>
          {preview && <p className="result-card-preview">{preview}</p>}
        </div>
      </Link>
      <div className="result-card-footer">
        {sign.tags.length > 0 ? (
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
        ) : (
          <span />
        )}
        {!selecting && (
          <div className="result-card-actions">
            <button
              type="button"
              className={`card-action ${queued ? "on" : ""}`}
              aria-pressed={queued}
              aria-label={queued ? `Remove ${sign.gloss} from To Learn` : `Add ${sign.gloss} to To Learn`}
              title={queued ? "In To Learn — click to remove" : "Add to To Learn"}
              onClick={() => quick(() => (queued ? removeFromToLearn(sign.id) : addToLearn([sign.id])))}
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.6" strokeLinecap="round" aria-hidden="true">
                <path d="M12 5v14M5 12h14" />
              </svg>
            </button>
            <button
              type="button"
              className={`card-action card-action-learnt ${learned ? "on" : ""}`}
              aria-pressed={learned}
              aria-label={learned ? `Unmark ${sign.gloss} as learnt` : `Mark ${sign.gloss} as learnt`}
              title={learned ? "Learnt — click to unmark" : "Mark as learnt"}
              onClick={() => quick(() => (learned ? removeLearned(sign.id) : markLearnt([sign.id])))}
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <path d="M5 12.5l4.5 4.5L19 7.5" />
              </svg>
            </button>
          </div>
        )}
      </div>
    </li>
  );
}

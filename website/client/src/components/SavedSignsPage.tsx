import { useEffect, useState, type ReactNode } from "react";
import { useNavigate } from "react-router-dom";
import { fetchSignById } from "../api/signs";
import type { Sign } from "../api/types";
import { ResultCard } from "./ResultCard";
import { ResultCardSkeleton } from "./ResultCardSkeleton";

interface Props {
  eyebrow: string;
  title: string;
  getIds: () => number[];
  emptyIcon: ReactNode;
  emptyTitle: string;
  emptyBody: string;
  /** Extra header content next to the title, e.g. Learned's progress bar --
   *  optional since not every saved-signs list has something to show here. */
  headerExtra?: (count: number) => ReactNode;
  /** Extra content between the back-link and the list, e.g. the
   *  personalized session builder -- a full panel, not header-sized. */
  beforeList?: ReactNode;
  /** Bump this to force a re-fetch (e.g. after adding signs from outside the
   *  normal per-sign toggle, such as the personalised session builder). */
  refreshKey?: number;
}

/** Shared shell for a page listing signs the user has saved locally (learned,
 *  or queued to learn) -- same fetch-by-id, sort, loading/empty/error
 *  states, different id source and copy. */
export function SavedSignsPage({
  eyebrow,
  title,
  getIds,
  emptyIcon,
  emptyTitle,
  emptyBody,
  headerExtra,
  beforeList,
  refreshKey,
}: Props) {
  const navigate = useNavigate();
  const [signs, setSigns] = useState<Sign[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isError, setIsError] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    let active = true;

    async function load() {
      setIsLoading(true);
      setIsError(false);

      try {
        const ids = getIds();
        if (ids.length === 0) {
          if (active) setSigns([]);
          return;
        }

        // Fetch each saved sign directly by id rather than scanning every
        // page of the catalogue -- saved lists are small, the catalogue is not.
        const settled = await Promise.allSettled(
          ids.map((id) => fetchSignById(id, controller.signal))
        );
        const loaded = settled
          .filter((r): r is PromiseFulfilledResult<Sign> => r.status === "fulfilled")
          .map((r) => r.value);
        loaded.sort((a, b) => {
          const glossCompare = a.gloss.localeCompare(b.gloss, undefined, { sensitivity: "base" });
          return glossCompare !== 0 ? glossCompare : a.id - b.id;
        });
        if (active) setSigns(loaded);
      } catch (err) {
        if (active && (err as Error).name !== "AbortError") setIsError(true);
      } finally {
        if (active) setIsLoading(false);
      }
    }

    load();
    return () => {
      active = false;
      controller.abort();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [getIds, refreshKey]);

  return (
    <>
      <header className="library-hero-band home-hero-band">
        <div className="library-hero-inner">
          <div>
            <p className="library-eyebrow-light">{eyebrow}</p>
            <h1 className="page-title">{title}</h1>
          </div>
          {headerExtra?.(signs.length)}
        </div>
      </header>

      <div className="page-container">
        <div className="detail-back learned-back-header">
          <button type="button" className="back-link" onClick={() => navigate(-1)}>
            &larr; Back to library
          </button>
        </div>

        {beforeList}

        {isError && <p role="alert">Couldn't load your signs.</p>}

        {isLoading ? (
          <ul className="result-list">
            {Array.from({ length: 6 }).map((_, i) => (
              <ResultCardSkeleton key={i} />
            ))}
          </ul>
        ) : signs.length === 0 ? (
          <div className="empty-state learned-empty-state">
            {emptyIcon}
            <h2>{emptyTitle}</h2>
            <p>{emptyBody}</p>
            <button type="button" onClick={() => navigate("/library")}>
              Browse the library
            </button>
          </div>
        ) : (
          <ul className="result-list">
            {signs.map((sign) => (
              <ResultCard key={sign.id} sign={sign} />
            ))}
          </ul>
        )}
      </div>
    </>
  );
}

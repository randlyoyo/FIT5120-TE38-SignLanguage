import { useEffect, useState, type ReactNode } from "react";
import { useNavigate } from "react-router-dom";
import { fetchSignById } from "../api/signs";
import type { IdEntry } from "../lib/idListStore";
import { useSelection } from "../hooks/useSelection";
import { removeLearned } from "../lib/learnedSigns";
import { addToLearn, markLearnt } from "../lib/signLists";
import { removeFromToLearn } from "../lib/toLearnSigns";
import { BulkActions, type BulkAction } from "./BulkActions";
import { ResultCardSkeleton } from "./ResultCardSkeleton";
import { TimeGroupedList, type TimedSign } from "./TimeGroupedList";

/** Bulk actions offered on each list: move to the other list, or drop from this one. */
const LIST_ACTIONS: Record<"toLearn" | "learnt", BulkAction[]> = {
  toLearn: [
    { label: "Mark as learnt", run: markLearnt },
    { label: "Remove from To Learn", run: (ids) => ids.forEach(removeFromToLearn) },
  ],
  learnt: [
    { label: "Move to To Learn", run: addToLearn },
    { label: "Remove from Learnt", run: (ids) => ids.forEach(removeLearned) },
  ],
};

interface Props {
  /** Which saved list this page shows -- picks its bulk actions. */
  list: "toLearn" | "learnt";
  eyebrow: string;
  title: string;
  getEntries: () => IdEntry[];
  emptyIcon?: ReactNode;
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
  list,
  eyebrow,
  title,
  getEntries,
  emptyIcon,
  emptyTitle,
  emptyBody,
  headerExtra,
  beforeList,
  refreshKey,
}: Props) {
  const navigate = useNavigate();
  const [timedSigns, setTimedSigns] = useState<TimedSign[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isError, setIsError] = useState(false);
  const selection = useSelection();
  // Bumped after a card's quick buttons or a bulk action change the lists;
  // signs no longer on this list drop out without a re-fetch.
  const [, setListVersion] = useState(0);
  const refreshLists = () => setListVersion((v) => v + 1);
  const onList = new Set(getEntries().map((e) => e.id));
  const visible = timedSigns.filter((t) => onList.has(t.sign.id));

  useEffect(() => {
    const controller = new AbortController();
    let active = true;

    async function load() {
      setIsLoading(true);
      setIsError(false);

      try {
        const entries = getEntries();
        if (entries.length === 0) {
          if (active) setTimedSigns([]);
          return;
        }

        // Fetch each saved sign directly by id rather than scanning every
        // page of the catalogue -- saved lists are small, the catalogue is not.
        const settled = await Promise.allSettled(
          entries.map((e) => fetchSignById(e.id, controller.signal))
        );
        const loaded: TimedSign[] = [];
        settled.forEach((r, i) => {
          if (r.status === "fulfilled") loaded.push({ sign: r.value, addedAt: entries[i].addedAt });
        });
        if (active) setTimedSigns(loaded);
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
  }, [getEntries, refreshKey]);

  return (
    <>
      <header className="library-hero-band home-hero-band">
        <div className="library-hero-inner">
          <div>
            <p className="library-eyebrow-light">{eyebrow}</p>
            <h1 className="page-title">{title}</h1>
          </div>
          {headerExtra?.(visible.length)}
        </div>
      </header>

      <div className="page-container">
        <div className="detail-back learned-back-header">
          <button type="button" className="back-link" onClick={() => navigate("/library")}>
            &larr; Back to library
          </button>
        </div>

        {/* Always above the list, full width -- not tucked into the list's toolbar. */}
        {beforeList && <div className="personalize-panel-standalone">{beforeList}</div>}

        {isError && <p role="alert">Couldn't load your signs.</p>}

        {isLoading ? (
          <>
            <ul className="result-list">
              {Array.from({ length: 6 }).map((_, i) => (
                <ResultCardSkeleton key={i} />
              ))}
            </ul>
          </>
        ) : visible.length === 0 ? (
          <>
            <div className="empty-state learned-empty-state">
              {emptyIcon}
              <h2>{emptyTitle}</h2>
              <p>{emptyBody}</p>
              <button type="button" onClick={() => navigate("/library")}>
                Browse the library
              </button>
            </div>
          </>
        ) : (
          <TimeGroupedList
            entries={visible}
            returnTo={window.location.pathname}
            selection={selection}
            onListsChange={refreshLists}
            headerExtra={
              <BulkActions
                selection={selection}
                visibleIds={visible.map((t) => t.sign.id)}
                actions={LIST_ACTIONS[list]}
                onDone={refreshLists}
              />
            }
          />
        )}
      </div>
    </>
  );
}

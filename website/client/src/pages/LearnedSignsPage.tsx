import { useEffect, useState } from "react";
import { fetchSigns } from "../api/signs";
import { SavedSignsPage } from "../components/SavedSignsPage";
import { getLearnedIds } from "../lib/learnedSigns";

export function LearnedSignsPage() {
  const [totalEntries, setTotalEntries] = useState<number | null>(null);

  // A busy parent squeezing in five minutes wants to see progress at a
  // glance, not just a list -- so "learned" is framed as a fraction of the
  // whole catalogue, not just a bare count.
  useEffect(() => {
    const controller = new AbortController();
    fetchSigns({ pageSize: 1, signal: controller.signal })
      .then((data) => setTotalEntries(data.pagination.totalResults))
      .catch((err) => {
        if (err.name !== "AbortError") console.error(err);
      });
    return () => controller.abort();
  }, []);

  return (
    <SavedSignsPage
      eyebrow="Your list"
      title="Learned Words"
      getIds={getLearnedIds}
      emptyIcon={
        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
          <circle cx="10" cy="10" r="6.5" />
          <path d="m19 19-4-4" strokeLinecap="round" />
          <path d="M8 10h4" strokeLinecap="round" />
        </svg>
      }
      emptyTitle="No learned words yet"
      emptyBody="Mark a sign as learned from its detail page to see it here."
      headerExtra={(count) =>
        totalEntries !== null && (
          <div className="library-header-tools learned-progress">
            <p className="results-count library-hero-count">
              {count} of {totalEntries} signs learned
            </p>
            <div
              className="progress-bar"
              role="progressbar"
              aria-valuenow={Math.round((count / totalEntries) * 100)}
              aria-valuemin={0}
              aria-valuemax={100}
            >
              <div
                className="progress-bar-fill"
                style={{ width: `${Math.min(100, (count / totalEntries) * 100)}%` }}
              />
            </div>
          </div>
        )
      }
    />
  );
}

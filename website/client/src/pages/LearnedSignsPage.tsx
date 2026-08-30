import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { fetchSigns } from "../api/signs";
import type { Sign } from "../api/types";
import { ResultCard } from "../components/ResultCard";
import { ResultCardSkeleton } from "../components/ResultCardSkeleton";
import { getLearnedIds } from "../lib/learnedSigns";

export function LearnedSignsPage() {
  const navigate = useNavigate();
  const [learnedSigns, setLearnedSigns] = useState<Sign[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isError, setIsError] = useState(false);

  useEffect(() => {
    let active = true;

    async function loadLearnedSigns() {
      setIsLoading(true);
      setIsError(false);

      try {
        const learnedIds = getLearnedIds();
        if (learnedIds.length === 0) {
          if (active) {
            setLearnedSigns([]);
            setIsLoading(false);
          }
          return;
        }

        const learnedSet = new Set(learnedIds);
        const combined: Sign[] = [];
        let page = 1;
        let totalPages = 1;

        do {
          const data = await fetchSigns({ page, pageSize: 50 });
          const newSigns = data.results.filter((sign) => learnedSet.has(sign.id));
          combined.push(...newSigns);
          totalPages = data.pagination.totalPages;
          page += 1;

          // Update UI progressively as each page loads
          if (active && newSigns.length > 0) {
            const sorted = combined.sort((a, b) => {
              const glossaryCompare = a.gloss.localeCompare(b.gloss, undefined, {
                sensitivity: "base",
              });
              return glossaryCompare !== 0 ? glossaryCompare : a.id - b.id;
            });
            setLearnedSigns(sorted);
            setIsLoading(false);
          }
        } while (page <= totalPages);

        // Final sorted result
        if (active) {
          const sorted = combined.sort((a, b) => {
            const glossaryCompare = a.gloss.localeCompare(b.gloss, undefined, {
              sensitivity: "base",
            });
            return glossaryCompare !== 0 ? glossaryCompare : a.id - b.id;
          });
          setLearnedSigns(sorted);
        }
      } catch (error) {
        console.error(error);
        if (active) setIsError(true);
      } finally {
        if (active) setIsLoading(false);
      }
    }

    loadLearnedSigns();
    return () => {
      active = false;
    };
  }, []);

  return (
    <div className="page-container">
      <div className="detail-back learned-back-header">
        <button type="button" className="back-link" onClick={() => navigate(-1)}>
          &larr; Back to library
        </button>
      </div>

      <header className="library-header learned-header">
        <div>
          <p className="library-eyebrow">Your list</p>
          <h1 className="page-title">Learned words</h1>
        </div>
      </header>

      {isError && <p role="alert">Couldn't load your learned signs.</p>}

      {isLoading ? (
        <ul className="result-list">
          {Array.from({ length: 6 }).map((_, i) => (
            <ResultCardSkeleton key={i} />
          ))}
        </ul>
      ) : learnedSigns.length === 0 ? (
        <div className="empty-state learned-empty-state">
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <circle cx="10" cy="10" r="6.5" />
            <path d="m19 19-4-4" strokeLinecap="round" />
            <path d="M8 10h4" strokeLinecap="round" />
          </svg>
          <h2>No learned words yet</h2>
          <p>Mark a sign as learned from the detail page to see it here.</p>
          <button type="button" onClick={() => navigate("/")}>
            Back to library
          </button>
        </div>
      ) : (
        <ul className="result-list">
          {learnedSigns.map((sign) => (
            <ResultCard key={sign.id} sign={sign} />
          ))}
        </ul>
      )}
    </div>
  );
}

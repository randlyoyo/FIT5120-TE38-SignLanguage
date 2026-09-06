import { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { fetchRecognitionVocabulary, type IdentifyCandidate } from "../api/recognize";
import { fetchSigns } from "../api/signs";
import { CategoryRail } from "../components/CategoryRail";
import { EmptyState } from "../components/EmptyState";
import { HandGlyphPagination } from "../components/Pagination/HandGlyphPagination";
import { ResultCard } from "../components/ResultCard";
import { ResultCardSkeleton } from "../components/ResultCardSkeleton";
import { SearchBar } from "../components/SearchBar";
import { useDebouncedValue } from "../hooks/useDebouncedValue";
import { useSignSearch } from "../hooks/useSignSearch";
import { useTags } from "../hooks/useTags";
import { tagChipStyle } from "../lib/tagColors";
import type { Sign } from "../api/types";

export function SignLibraryPage() {
  const [searchParams, setSearchParams] = useSearchParams();

  const page = Number(searchParams.get("page")) || 1;
  const tag = searchParams.get("tag") ?? "";

  // The search input keeps its own local state so every keystroke feels
  // instant. Routing every keystroke through useSearchParams (which is
  // backed by browser history) can race when characters arrive faster than
  // a navigation commits, silently dropping characters. Instead, only the
  // debounced value gets synced into the URL, and only that debounced value
  // drives the actual API fetch.
  const [queryInput, setQueryInput] = useState(searchParams.get("query") ?? "");
  const debouncedQuery = useDebouncedValue(queryInput, 300);

  // Hidden entirely (not shown-then-erroring) when the recognizer isn't
  // deployed -- fetchRecognitionVocabulary() resolves to an empty set rather
  // than rejecting in that case (same convention SignDetailPage uses).
  const [canGestureSearch, setCanGestureSearch] = useState(false);
  useEffect(() => {
    let active = true;
    fetchRecognitionVocabulary().then((vocabulary) => {
      if (active) setCanGestureSearch(vocabulary.size > 0);
    });
    return () => {
      active = false;
    };
  }, []);

  // A gesture search jumps straight to a small results view of exactly the
  // model's top-4 candidates, bypassing the normal query/tag/pagination flow
  // entirely -- `null` means "not in that view", so it also doubles as the
  // flag for which UI (this vs. the regular grid) is on screen.
  const [gestureResults, setGestureResults] = useState<Sign[] | null>(null);
  const [gestureResultsLoading, setGestureResultsLoading] = useState(false);

  async function handleGestureResults(candidates: IdentifyCandidate[]) {
    setGestureResultsLoading(true);
    try {
      // The recognizer's vocabulary is sourced from the same `gloss` column
      // this searches, and the query endpoint already ranks an exact gloss
      // match first (routes/signs.js), so a plain query-by-word reliably
      // finds the same sign without a dedicated lookup-by-gloss endpoint.
      const signs = await Promise.all(
        candidates.map(async (candidate) => {
          const response = await fetchSigns({ query: candidate.word, pageSize: 1 });
          return response.results[0] ?? null;
        })
      );
      setGestureResults(signs.filter((sign): sign is Sign => sign !== null));
    } finally {
      setGestureResultsLoading(false);
    }
  }

  useEffect(() => {
    const currentQuery = searchParams.get("query") ?? "";
    if (debouncedQuery === currentQuery) return;
    setSearchParams((prev) => {
      const params = new URLSearchParams(prev);
      if (debouncedQuery) params.set("query", debouncedQuery);
      else params.delete("query");
      params.delete("page");
      return params;
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [debouncedQuery]);

  const { data, isLoading, isError } = useSignSearch({ query: debouncedQuery, tag, page });
  const tags = useTags();

  function updateParams(next: { page?: number }) {
    setSearchParams((prev) => {
      const params = new URLSearchParams(prev);
      if (next.page !== undefined) {
        if (next.page > 1) params.set("page", String(next.page));
        else params.delete("page");
      }
      return params;
    });
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function clearTag() {
    setSearchParams((prev) => {
      const params = new URLSearchParams(prev);
      params.delete("tag");
      params.delete("page");
      return params;
    });
  }

  function clearFilters() {
    setQueryInput("");
    setSearchParams({});
  }

  // Typing a real keyword search is how a signer leaves the gesture-results
  // view -- there's no separate "exit" step to remember.
  function handleQueryInputChange(next: string) {
    setGestureResults(null);
    setQueryInput(next);
  }

  const results = data?.results ?? [];
  const totalPages = data?.pagination.totalPages ?? 1;
  const totalResults = data?.pagination.totalResults ?? 0;

  return (
    <>
      <header className="library-hero-band home-hero-band">
        <div className="library-hero-inner">
          <div>
            <p className="library-eyebrow-light">Auslan</p>
            <h1 className="page-title">Sign Library</h1>
          </div>
          <div className="library-header-tools">
            <SearchBar
              value={queryInput}
              onChange={handleQueryInputChange}
              canGestureSearch={canGestureSearch}
              onGestureResults={handleGestureResults}
            />
            {!isLoading && !isError && (
              <p className="results-count library-hero-count">{totalResults} entries indexed</p>
            )}
          </div>
        </div>
      </header>

      <div className="page-container">
        {gestureResultsLoading || gestureResults ? (
          <div className="gesture-results-panel">
            <div className="gesture-results-header">
              <h2 className="sign-detail-heading">Matches for your sign</h2>
              <button
                type="button"
                className="back-link"
                onClick={() => setGestureResults(null)}
              >
                &larr; Back to browsing
              </button>
            </div>

            {gestureResultsLoading ? (
              <ul className="result-list">
                {Array.from({ length: 4 }).map((_, i) => (
                  <ResultCardSkeleton key={i} />
                ))}
              </ul>
            ) : gestureResults!.length === 0 ? (
              <p role="alert">Couldn't find a match for that sign. Try signing again?</p>
            ) : (
              <ul className="result-list">
                {gestureResults!.map((sign) => (
                  <ResultCard key={sign.id} sign={sign} />
                ))}
              </ul>
            )}
          </div>
        ) : (
          <>
            <div className="library-layout">
              <CategoryRail tags={tags} activeTag={tag} />

              <div className="library-main">
                {tag && (
                  <div className="active-tag-filter">
                    <span style={tagChipStyle(tag)} className="tag-chip">
                      #{tag}
                    </span>
                    <button type="button" onClick={clearTag} aria-label={`Clear tag filter ${tag}`}>
                      &times; Clear tag
                    </button>
                  </div>
                )}

                {isError && <p role="alert">Couldn't load the sign library. Is the server running?</p>}

                {isLoading ? (
                  <ul className="result-list">
                    {Array.from({ length: 6 }).map((_, i) => (
                      <ResultCardSkeleton key={i} />
                    ))}
                  </ul>
                ) : results.length === 0 && !isError ? (
                  <EmptyState onClear={clearFilters} />
                ) : (
                  <ul className="result-list">
                    {results.map((sign) => (
                      <ResultCard key={sign.id} sign={sign} />
                    ))}
                  </ul>
                )}
              </div>
            </div>

            {/* Rendered outside .library-layout, not inside .library-main --
                .library-main is only the narrower right-hand grid column (after
                the category rail), so centering the nav there put it visibly
                off-center from the page as a whole. This centers it against the
                full page-container width instead. */}
            <HandGlyphPagination
              page={page}
              totalPages={totalPages}
              onPageChange={(p) => updateParams({ page: p })}
            />
          </>
        )}
      </div>
    </>
  );
}

import { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { fetchRecognitionVocabulary, type IdentifyCandidate } from "../api/recognize";
import { fetchSigns, type SignSort } from "../api/signs";
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
import type { Sign, SignLevel } from "../api/types";

const LEVEL_FILTERS: { id: SignLevel; label: string }[] = [
  { id: "beginner", label: "Beginner" },
  { id: "intermediate", label: "Intermediate" },
  { id: "advanced", label: "Advanced" },
];

type SortType = "gloss" | "level" | "popularity";
type SortOrder = "asc" | "desc";

const SORT_TYPES: { id: SortType; label: string }[] = [
  { id: "gloss", label: "A–Z" },
  { id: "level", label: "Difficulty" },
  { id: "popularity", label: "Popularity" },
];

const SORT_ORDERS: { id: SortOrder; label: string }[] = [
  { id: "asc", label: "Ascending" },
  { id: "desc", label: "Descending" },
];

const DEFAULT_SORT: { type: SortType; order: SortOrder } = { type: "gloss", order: "asc" };

// The URL's `sort` param is empty for the default (relevance / A-Z) rather
// than the literal string "gloss_asc", so a plain library link doesn't
// carry a redundant ?sort= -- everything else round-trips as `${type}_${order}`.
function parseSort(sort: SignSort | ""): { type: SortType; order: SortOrder } {
  if (!sort) return DEFAULT_SORT;
  const [type, order] = sort.split("_") as [SortType, SortOrder];
  return { type, order };
}

function buildSort(type: SortType, order: SortOrder): SignSort | "" {
  if (type === DEFAULT_SORT.type && order === DEFAULT_SORT.order) return "";
  return `${type}_${order}` as SignSort;
}

export function SignLibraryPage() {
  const [searchParams, setSearchParams] = useSearchParams();

  const page = Number(searchParams.get("page")) || 1;
  const tag = searchParams.get("tag") ?? "";
  const level = (searchParams.get("level") ?? "") as SignLevel | "";
  const sort = (searchParams.get("sort") ?? "") as SignSort | "";

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

  const { data, isLoading, isError } = useSignSearch({ query: debouncedQuery, tag, level, sort, page });
  const tags = useTags();

  function setLevel(next: SignLevel | "") {
    setSearchParams((prev) => {
      const params = new URLSearchParams(prev);
      if (next) params.set("level", next);
      else params.delete("level");
      params.delete("page");
      return params;
    });
  }

  function setSort(next: SignSort | "") {
    setSearchParams((prev) => {
      const params = new URLSearchParams(prev);
      if (next) params.set("sort", next);
      else params.delete("sort");
      params.delete("page");
      return params;
    });
  }

  const { type: sortType, order: sortOrder } = parseSort(sort);
  function setSortType(next: SortType) {
    setSort(buildSort(next, sortOrder));
  }
  function setSortOrder(next: SortOrder) {
    setSort(buildSort(sortType, next));
  }

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
                  <ResultCard
                    key={sign.id}
                    sign={sign}
                    siblingIds={gestureResults!.map((s) => s.id)}
                    returnTo="/library"
                  />
                ))}
              </ul>
            )}
          </div>
        ) : (
          <>
            <div className="library-layout">
              <CategoryRail tags={tags} activeTag={tag} />

              <div className="library-main">
                <div className="library-filter-bar">
                  <div className="detail-mode-toggle library-level-toggle" role="tablist" aria-label="Filter by estimated difficulty">
                    <button
                      type="button"
                      role="tab"
                      aria-selected={level === ""}
                      className={`detail-mode-tab ${level === "" ? "active" : ""}`}
                      onClick={() => setLevel("")}
                    >
                      All levels
                    </button>
                    {LEVEL_FILTERS.map(({ id, label }) => (
                      <button
                        key={id}
                        type="button"
                        role="tab"
                        aria-selected={level === id}
                        className={`detail-mode-tab ${level === id ? "active" : ""}`}
                        onClick={() => setLevel(id)}
                      >
                        {label}
                      </button>
                    ))}
                  </div>

                  <div
                    className="library-sort-controls"
                    title={debouncedQuery ? "Sort is ignored while searching -- results are ranked by relevance instead" : undefined}
                  >
                    <label className="library-sort-select">
                      Sort by
                      <select
                        value={sortType}
                        disabled={Boolean(debouncedQuery)}
                        onChange={(e) => setSortType(e.target.value as SortType)}
                      >
                        {SORT_TYPES.map(({ id, label }) => (
                          <option key={id} value={id}>
                            {label}
                          </option>
                        ))}
                      </select>
                    </label>
                    <label className="library-sort-select">
                      Order
                      <select
                        value={sortOrder}
                        disabled={Boolean(debouncedQuery)}
                        onChange={(e) => setSortOrder(e.target.value as SortOrder)}
                      >
                        {SORT_ORDERS.map(({ id, label }) => (
                          <option key={id} value={id}>
                            {label}
                          </option>
                        ))}
                      </select>
                    </label>
                  </div>
                </div>

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
                      <ResultCard
                        key={sign.id}
                        sign={sign}
                        siblingIds={results.map((s) => s.id)}
                        returnTo={`${window.location.pathname}${window.location.search}`}
                      />
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

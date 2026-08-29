import { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";
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

  const results = data?.results ?? [];
  const totalPages = data?.pagination.totalPages ?? 1;
  const totalResults = data?.pagination.totalResults ?? 0;

  return (
    <div className="page-container">
      <header className="library-header">
        <div>
          <p className="library-eyebrow">Auslan</p>
          <h1 className="page-title">Sign Library</h1>
        </div>
        <div className="library-header-tools">
          <SearchBar value={queryInput} onChange={setQueryInput} />
          {!isLoading && !isError && (
            <p className="results-count">{totalResults} entries indexed</p>
          )}
        </div>
      </header>

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

          <HandGlyphPagination
            page={page}
            totalPages={totalPages}
            onPageChange={(p) => updateParams({ page: p })}
          />
        </div>
      </div>
    </div>
  );
}

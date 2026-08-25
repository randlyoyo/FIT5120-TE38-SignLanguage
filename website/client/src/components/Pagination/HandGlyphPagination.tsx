import { PageHandIcon } from "./handGlyphs";

interface Props {
  page: number;
  totalPages: number;
  onPageChange: (page: number) => void;
}

function getVisiblePages(page: number, totalPages: number): (number | "ellipsis")[] {
  if (totalPages <= 8) {
    return Array.from({ length: totalPages }, (_, i) => i + 1);
  }

  const pages = new Set<number>([1, totalPages, page, page - 1, page + 1]);
  const sorted = [...pages].filter((p) => p >= 1 && p <= totalPages).sort((a, b) => a - b);

  const result: (number | "ellipsis")[] = [];
  let prev = 0;
  for (const p of sorted) {
    if (prev && p - prev > 1) result.push("ellipsis");
    result.push(p);
    prev = p;
  }
  return result;
}

export function HandGlyphPagination({ page, totalPages, onPageChange }: Props) {
  if (totalPages <= 1) return null;

  return (
    <nav className="pagination" aria-label="Search results pages">
      {getVisiblePages(page, totalPages).map((entry, i) =>
        entry === "ellipsis" ? (
          <span key={`ellipsis-${i}`} className="pagination-ellipsis" aria-hidden="true">
            &hellip;
          </span>
        ) : (
          <PageButton
            key={entry}
            pageNumber={entry}
            isCurrent={entry === page}
            onClick={() => onPageChange(entry)}
          />
        )
      )}
    </nav>
  );
}

function PageButton({
  pageNumber,
  isCurrent,
  onClick,
}: {
  pageNumber: number;
  isCurrent: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      className={isCurrent ? "current" : ""}
      onClick={onClick}
      aria-label={`Page ${pageNumber}`}
      aria-current={isCurrent ? "page" : undefined}
    >
      <PageHandIcon page={pageNumber} />
    </button>
  );
}

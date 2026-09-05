// Mirrors ResultCard's exact DOM structure and classes (media, body, tag
// row) rather than an approximated shape -- otherwise the skeleton renders
// shorter than a real card (which almost always has a tag-chips row), and
// the grid visibly reflows the instant real results replace it.
export function ResultCardSkeleton() {
  return (
    <li className="result-card">
      <div className="result-card-link">
        <div className="result-card-media">
          <div className="skeleton-block result-card-video" />
        </div>
        <div className="result-card-body">
          <div className="skeleton-block" style={{ width: "70%", height: "1.15rem", marginBottom: "0.3rem" }} />
          {/* Every sign shares the same long attribution string, which
              wraps to 2 lines at this card width -- so, unlike the preview
              text below, this isn't a guess. */}
          <div className="skeleton-block" style={{ width: "95%", height: "0.85rem", marginBottom: "0.25rem" }} />
          <div className="skeleton-block" style={{ width: "60%", height: "0.85rem", marginBottom: "0.5rem" }} />
          <div className="skeleton-block" style={{ width: "100%", height: "0.92rem", marginBottom: "0.3rem" }} />
          <div className="skeleton-block" style={{ width: "75%", height: "0.92rem" }} />
        </div>
      </div>
      <div className="tag-chips">
        <div className="skeleton-block" style={{ width: 64, height: 22 }} />
        <div className="skeleton-block" style={{ width: 84, height: 22 }} />
      </div>
    </li>
  );
}

export function ResultCardSkeleton() {
  return (
    <li className="result-card-skeleton">
      <div className="skeleton-block" style={{ width: "100%", aspectRatio: "4 / 3" }} />
      <div>
        <div className="skeleton-block" style={{ width: "60%", height: 20, marginBottom: 10 }} />
        <div className="skeleton-block" style={{ width: "40%", height: 12, marginBottom: 10 }} />
        <div className="skeleton-block" style={{ width: "90%", height: 12, marginBottom: 6 }} />
        <div className="skeleton-block" style={{ width: "80%", height: 12 }} />
      </div>
    </li>
  );
}

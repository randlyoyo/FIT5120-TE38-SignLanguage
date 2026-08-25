interface Props {
  onClear: () => void;
}

export function EmptyState({ onClear }: Props) {
  return (
    <div className="empty-state">
      <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
        <circle cx="10" cy="10" r="6.5" />
        <path d="m19 19-4-4" strokeLinecap="round" />
        <path d="M8 10h4" strokeLinecap="round" />
      </svg>
      <h2>No signs found</h2>
      <p>Try a different keyword.</p>
      <button type="button" onClick={onClear}>
        Clear filters
      </button>
    </div>
  );
}

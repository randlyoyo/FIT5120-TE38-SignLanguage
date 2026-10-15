import type { Selection } from "../hooks/useSelection";

export interface BulkAction {
  label: string;
  run: (ids: number[]) => void;
}

interface Props {
  selection: Selection;
  /** Every id currently shown, for "Select all". */
  visibleIds: number[];
  actions: BulkAction[];
  /** Called after an action ran, so the page can re-read its lists. */
  onDone: () => void;
}

/** "Select" button that turns a card grid into a multi-select, plus the bar
 *  pinned to the bottom of the screen with the bulk actions. */
export function BulkActions({ selection, visibleIds, actions, onDone }: Props) {
  if (!selection.active) {
    return (
      <button type="button" className="bulk-start" onClick={selection.start} disabled={visibleIds.length === 0}>
        Select signs
      </button>
    );
  }

  const ids = [...selection.selected];
  const allSelected = visibleIds.length > 0 && visibleIds.every((id) => selection.selected.has(id));

  return (
    <div className="bulk-bar" role="region" aria-label="Bulk actions">
      <span className="bulk-count" aria-live="polite">
        {ids.length} selected
      </span>
      <button
        type="button"
        className="bulk-link"
        onClick={() => (allSelected ? selection.clear() : selection.setAll(visibleIds))}
      >
        {allSelected ? "Clear selection" : `Select all (${visibleIds.length})`}
      </button>
      <span className="bulk-actions">
        {actions.map((action) => (
          <button
            key={action.label}
            type="button"
            className="bulk-action"
            disabled={ids.length === 0}
            onClick={() => {
              action.run(ids);
              selection.clear();
              onDone();
            }}
          >
            {action.label}
          </button>
        ))}
      </span>
      <button type="button" className="bulk-done" onClick={selection.stop}>
        Done
      </button>
    </div>
  );
}

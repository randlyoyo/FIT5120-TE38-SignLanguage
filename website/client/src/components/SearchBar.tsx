import { useState } from "react";
import { GestureSearch } from "./GestureSearch";

interface Props {
  value: string;
  onChange: (value: string) => void;
  /** Hidden entirely when the recognizer isn't deployed (or its vocabulary
   *  hasn't loaded yet) -- same "not ready" == "not supported" treatment
   *  SignDetailPage gives fetchRecognitionVocabulary()'s empty-set case. */
  canGestureSearch?: boolean;
}

export function SearchBar({ value, onChange, canGestureSearch = false }: Props) {
  const [gestureSearchOpen, setGestureSearchOpen] = useState(false);

  return (
    <div className="search-bar-wrap">
      <div className="search-bar">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <circle cx="11" cy="11" r="7" />
          <path d="m20 20-3.5-3.5" strokeLinecap="round" />
        </svg>
        <input
          type="search"
          placeholder="Search signs by keyword…"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          aria-label="Search signs"
        />
        {canGestureSearch && (
          <button
            type="button"
            className="search-bar-gesture-button"
            onClick={() => setGestureSearchOpen(true)}
            aria-label="Search by signing at your camera"
            title="Search by signing"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6">
              <rect x="3" y="6" width="14" height="11" rx="2" />
              <path d="m17 10 4-2.5v9L17 14" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </button>
        )}
      </div>

      {gestureSearchOpen && (
        <div className="gesture-search-popover">
          <GestureSearch
            onPick={(word) => {
              onChange(word);
              setGestureSearchOpen(false);
            }}
            onClose={() => setGestureSearchOpen(false)}
          />
        </div>
      )}
    </div>
  );
}

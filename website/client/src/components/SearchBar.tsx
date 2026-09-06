import { useState } from "react";
import type { IdentifyCandidate } from "../api/recognize";
import { GestureSearch } from "./GestureSearch";

interface Props {
  value: string;
  onChange: (value: string) => void;
  /** Hidden entirely when the recognizer isn't deployed (or its vocabulary
   *  hasn't loaded yet) -- same "not ready" == "not supported" treatment
   *  SignDetailPage gives fetchRecognitionVocabulary()'s empty-set case. */
  canGestureSearch?: boolean;
  /** Called with the model's top-4 candidates the moment a capture resolves.
   *  Omitted (gesture search hidden) unless the caller is ready to show a
   *  results view for them. */
  onGestureResults?: (candidates: IdentifyCandidate[]) => void;
}

export function SearchBar({ value, onChange, canGestureSearch = false, onGestureResults }: Props) {
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
            {/* A friendly, rounded photo-camera glyph (viewfinder bump, big
                "eye" lens, a glint and a flash dot) rather than a plain
                outline icon -- this button is a kid-facing "point your
                camera and sign" affordance, styled closer to the playful
                camera marks search apps use than the rest of the site's
                thin utility icons. */}
            <svg width="21" height="21" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
              <rect x="2.5" y="7.5" width="19" height="12" rx="4" strokeLinejoin="round" />
              <rect x="8.5" y="4.5" width="7" height="3.6" rx="1.8" strokeLinejoin="round" />
              <circle cx="12" cy="13.5" r="4.3" fill="currentColor" stroke="none" />
              <circle cx="10.3" cy="11.9" r="1" fill="#fff" stroke="none" />
              <circle cx="18.2" cy="10.3" r="1" fill="currentColor" stroke="none" />
            </svg>
          </button>
        )}
      </div>

      {gestureSearchOpen && (
        <div className="gesture-search-popover">
          <GestureSearch
            onResults={(candidates) => {
              onGestureResults?.(candidates);
              setGestureSearchOpen(false);
            }}
            onClose={() => setGestureSearchOpen(false)}
          />
        </div>
      )}
    </div>
  );
}

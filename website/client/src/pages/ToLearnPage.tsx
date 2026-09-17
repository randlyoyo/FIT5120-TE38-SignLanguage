import { useState } from "react";
import { PersonalizeSessionPanel } from "../components/PersonalizeSessionPanel";
import { SavedSignsPage } from "../components/SavedSignsPage";
import { getToLearnIds } from "../lib/toLearnSigns";

export function ToLearnPage() {
  const [refreshKey, setRefreshKey] = useState(0);

  return (
    <SavedSignsPage
      eyebrow="Your list"
      title="To Learn"
      getIds={getToLearnIds}
      refreshKey={refreshKey}
      emptyIcon={
        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
          <path d="M12 5v14M5 12h14" strokeLinecap="round" />
        </svg>
      }
      emptyTitle="No words queued yet"
      emptyBody="Add a sign to this list from its detail page, or personalize a session below, to build your practice queue."
      beforeList={<PersonalizeSessionPanel onBuilt={() => setRefreshKey((k) => k + 1)} />}
    />
  );
}

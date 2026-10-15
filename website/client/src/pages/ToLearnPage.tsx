import { useState } from "react";
import { PersonalizeSessionPanel } from "../components/PersonalizeSessionPanel";
import { SavedSignsPage } from "../components/SavedSignsPage";
import { getToLearnEntries } from "../lib/toLearnSigns";

export function ToLearnPage() {
  const [refreshKey, setRefreshKey] = useState(0);

  return (
    <SavedSignsPage
      list="toLearn"
      eyebrow="Your list"
      title="To Learn"
      getEntries={getToLearnEntries}
      refreshKey={refreshKey}
      emptyTitle="No words queued yet"
      emptyBody="Add a sign to this list from its detail page, or personalize a session below, to build your practice queue."
      beforeList={<PersonalizeSessionPanel onBuilt={() => setRefreshKey((k) => k + 1)} />}
    />
  );
}

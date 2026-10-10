export interface IdEntry {
  id: number;
  addedAt: number;
}

/** localStorage-backed set of sign ids, shared by the "Learned" and "To
 *  Learn" lists -- same persistence shape, different keys. Each id carries
 *  an `addedAt` timestamp so the saved lists can be grouped by when a sign
 *  was added (day/week/month/year), phone-album style. */
export function createIdListStore(key: string) {
  function readEntries(): IdEntry[] {
    try {
      const raw = window.localStorage.getItem(key);
      if (!raw) return [];
      const parsed = JSON.parse(raw);
      if (!Array.isArray(parsed)) return [];

      // Legacy format: a plain array of ids, no timestamp. There's no real
      // history to recover, so every legacy id is stamped "now" once, on
      // first read, and the migrated shape is written straight back so
      // this branch only ever runs a single time per browser.
      if (parsed.length > 0 && typeof parsed[0] === "number") {
        const migrated = parsed
          .filter((n): n is number => typeof n === "number")
          .map((id) => ({ id, addedAt: Date.now() }));
        writeEntries(migrated);
        return migrated;
      }

      return parsed.filter(
        (e): e is IdEntry => e && typeof e.id === "number" && typeof e.addedAt === "number"
      );
    } catch {
      return [];
    }
  }

  function writeEntries(entries: IdEntry[]) {
    window.localStorage.setItem(key, JSON.stringify(entries));
  }

  return {
    getIds: () => readEntries().map((e) => e.id),
    getEntries: readEntries,
    has: (id: number) => readEntries().some((e) => e.id === id),
    /** Adds/removes `id`; returns whether it's now in the list. */
    toggle: (id: number): boolean => {
      const entries = readEntries();
      const index = entries.findIndex((e) => e.id === id);
      if (index === -1) {
        entries.push({ id, addedAt: Date.now() });
        writeEntries(entries);
        return true;
      }
      entries.splice(index, 1);
      writeEntries(entries);
      return false;
    },
    /** Removes `id` if present; a no-op otherwise. */
    remove: (id: number) => {
      const entries = readEntries();
      const index = entries.findIndex((e) => e.id === id);
      if (index === -1) return;
      entries.splice(index, 1);
      writeEntries(entries);
    },
    /** Adds every id not already present, all stamped with the same "now"
     *  (a single build's worth of adds is one moment for grouping purposes)
     *  -- unlike `toggle`, never removes. */
    addAll: (idsToAdd: number[]) => {
      const entries = readEntries();
      const existing = new Set(entries.map((e) => e.id));
      const now = Date.now();
      for (const id of idsToAdd) {
        if (!existing.has(id)) {
          entries.push({ id, addedAt: now });
          existing.add(id);
        }
      }
      writeEntries(entries);
    },
  };
}

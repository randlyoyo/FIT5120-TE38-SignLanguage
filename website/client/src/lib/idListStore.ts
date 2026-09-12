/** localStorage-backed set of sign ids, shared by the "Learned" and "To
 *  Learn" lists -- same persistence shape, different keys. */
export function createIdListStore(key: string) {
  function readIds(): number[] {
    try {
      const raw = window.localStorage.getItem(key);
      if (!raw) return [];
      const parsed = JSON.parse(raw);
      return Array.isArray(parsed) ? parsed.filter((n) => typeof n === "number") : [];
    } catch {
      return [];
    }
  }

  function writeIds(ids: number[]) {
    window.localStorage.setItem(key, JSON.stringify(ids));
  }

  return {
    getIds: readIds,
    has: (id: number) => readIds().includes(id),
    /** Adds/removes `id`; returns whether it's now in the list. */
    toggle: (id: number): boolean => {
      const ids = readIds();
      const index = ids.indexOf(id);
      if (index === -1) {
        ids.push(id);
        writeIds(ids);
        return true;
      }
      ids.splice(index, 1);
      writeIds(ids);
      return false;
    },
    /** Removes `id` if present; a no-op otherwise. */
    remove: (id: number) => {
      const ids = readIds();
      const index = ids.indexOf(id);
      if (index === -1) return;
      ids.splice(index, 1);
      writeIds(ids);
    },
    /** Adds every id not already present; a no-op for ones that are (unlike
     *  `toggle`, never removes). */
    addAll: (idsToAdd: number[]) => {
      const ids = readIds();
      for (const id of idsToAdd) if (!ids.includes(id)) ids.push(id);
      writeIds(ids);
    },
  };
}

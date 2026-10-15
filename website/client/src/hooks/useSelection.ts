import { useState } from "react";

/** Multi-select state for a list of sign cards (bulk To Learn / Learnt actions). */
export function useSelection() {
  const [active, setActive] = useState(false);
  const [selected, setSelected] = useState<Set<number>>(new Set());

  return {
    active,
    selected,
    start: () => setActive(true),
    stop: () => {
      setActive(false);
      setSelected(new Set());
    },
    toggle: (id: number) =>
      setSelected((prev) => {
        const next = new Set(prev);
        if (next.has(id)) next.delete(id);
        else next.add(id);
        return next;
      }),
    setAll: (ids: number[]) => setSelected(new Set(ids)),
    clear: () => setSelected(new Set()),
  };
}

export type Selection = ReturnType<typeof useSelection>;

import { useState, type ReactNode } from "react";
import type { Sign } from "../api/types";
import { groupByPeriod, type Granularity } from "../lib/groupByPeriod";
import { ResultCard } from "./ResultCard";

export interface TimedSign {
  sign: Sign;
  addedAt: number;
}

interface Props {
  entries: TimedSign[];
  returnTo: string;
  /** Rendered on the same row as the granularity switch, pushed to the
   *  right -- e.g. the "Personalize your learning" trigger, which only
   *  makes sense once there's an actual grouped list to sit beside. */
  headerExtra?: ReactNode;
}

const GRANULARITIES: { id: Granularity; label: string }[] = [
  { id: "day", label: "Day" },
  { id: "week", label: "Week" },
  { id: "month", label: "Month" },
  { id: "year", label: "Year" },
];

/** Phone-album-style grouping for a saved-signs list: switch how signs are
 *  bucketed by when they were added (day/week/month/year), each bucket a
 *  collapsible section behind a divider and a triangle disclosure arrow,
 *  instead of one long flat grid. */
export function TimeGroupedList({ entries, returnTo, headerExtra }: Props) {
  const [granularity, setGranularity] = useState<Granularity>("month");
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set());

  const groups = groupByPeriod(entries, (e) => e.addedAt, granularity);
  const allIds = entries.map((e) => e.sign.id);

  function toggleGroup(key: string) {
    setCollapsed((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }

  return (
    <div className="time-grouped-list">
      <div className="time-group-toolbar">
        <div className="detail-mode-toggle time-group-switch" role="tablist" aria-label="Group by">
          {GRANULARITIES.map(({ id, label }) => (
            <button
              key={id}
              type="button"
              role="tab"
              aria-selected={granularity === id}
              className={`detail-mode-tab ${granularity === id ? "active" : ""}`}
              onClick={() => setGranularity(id)}
            >
              {label}
            </button>
          ))}
        </div>
        {headerExtra}
      </div>

      {groups.map((group) => {
        const isCollapsed = collapsed.has(group.key);
        return (
          <section key={group.key} className="time-group">
            <button
              type="button"
              className="time-group-header"
              onClick={() => toggleGroup(group.key)}
              aria-expanded={!isCollapsed}
            >
              <svg
                className={`time-group-arrow ${isCollapsed ? "" : "open"}`}
                width="10"
                height="10"
                viewBox="0 0 10 10"
                aria-hidden="true"
              >
                <path
                  d="M1 1l4 4-4 4"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.6"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
              <span className="time-group-label">{group.label}</span>
              <span className="time-group-count">{group.items.length}</span>
              <span className="time-group-rule" aria-hidden="true" />
            </button>

            {!isCollapsed && (
              <ul className="result-list">
                {group.items.map(({ sign }) => (
                  <ResultCard key={sign.id} sign={sign} siblingIds={allIds} returnTo={returnTo} />
                ))}
              </ul>
            )}
          </section>
        );
      })}
    </div>
  );
}

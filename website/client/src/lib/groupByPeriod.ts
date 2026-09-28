export type Granularity = "day" | "week" | "month" | "year";

export interface TimeGroup<T> {
  key: string;
  label: string;
  items: T[];
}

function startOfDay(d: Date): Date {
  return new Date(d.getFullYear(), d.getMonth(), d.getDate());
}

// Monday-based week, matching most calendar apps outside the US.
function startOfWeek(d: Date): Date {
  const day = d.getDay(); // 0 = Sunday
  const diff = (day === 0 ? -6 : 1) - day;
  const start = startOfDay(d);
  start.setDate(start.getDate() + diff);
  return start;
}

function startOfMonth(d: Date): Date {
  return new Date(d.getFullYear(), d.getMonth(), 1);
}

function startOfYear(d: Date): Date {
  return new Date(d.getFullYear(), 0, 1);
}

const START_OF: Record<Granularity, (d: Date) => Date> = {
  day: startOfDay,
  week: startOfWeek,
  month: startOfMonth,
  year: startOfYear,
};

const DAY_FMT = new Intl.DateTimeFormat(undefined, { day: "numeric", month: "short", year: "numeric" });
const MONTH_FMT = new Intl.DateTimeFormat(undefined, { month: "long", year: "numeric" });

function labelFor(start: Date, granularity: Granularity): string {
  switch (granularity) {
    case "day": {
      const diffDays = Math.round((startOfDay(new Date()).getTime() - start.getTime()) / 86400000);
      if (diffDays === 0) return "Today";
      if (diffDays === 1) return "Yesterday";
      return DAY_FMT.format(start);
    }
    case "week":
      return `Week of ${DAY_FMT.format(start)}`;
    case "month":
      return MONTH_FMT.format(start);
    case "year":
      return String(start.getFullYear());
  }
}

/** Buckets `items` by when each was added, newest group (and newest item
 *  within each group) first -- the same recency-first order a phone photo
 *  album uses when grouping by day/week/month/year. */
export function groupByPeriod<T>(
  items: T[],
  getAddedAt: (item: T) => number,
  granularity: Granularity
): TimeGroup<T>[] {
  const startOf = START_OF[granularity];
  const buckets = new Map<string, { start: Date; items: T[] }>();

  for (const item of items) {
    const start = startOf(new Date(getAddedAt(item)));
    const key = start.toISOString();
    let bucket = buckets.get(key);
    if (!bucket) {
      bucket = { start, items: [] };
      buckets.set(key, bucket);
    }
    bucket.items.push(item);
  }

  return [...buckets.values()]
    .sort((a, b) => b.start.getTime() - a.start.getTime())
    .map(({ start, items: bucketItems }) => ({
      key: start.toISOString(),
      label: labelFor(start, granularity),
      items: [...bucketItems].sort((a, b) => getAddedAt(b) - getAddedAt(a)),
    }));
}

import type { Sign, SignsResponse, TagCount } from "./types";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "/api";

export interface FetchSignsParams {
  query?: string;
  tag?: string;
  page?: number;
  pageSize?: number;
  signal?: AbortSignal;
}

export async function fetchSigns({
  query,
  tag,
  page,
  pageSize,
  signal,
}: FetchSignsParams): Promise<SignsResponse> {
  const params = new URLSearchParams();
  if (query) params.set("query", query);
  if (tag) params.set("tag", tag);
  if (page) params.set("page", String(page));
  if (pageSize) params.set("pageSize", String(pageSize));

  const res = await fetch(`${API_BASE}/signs?${params.toString()}`, { signal });
  if (!res.ok) throw new Error(`Failed to fetch signs (${res.status})`);
  return res.json();
}

export async function fetchSignById(id: number, signal?: AbortSignal): Promise<Sign> {
  const res = await fetch(`${API_BASE}/signs/${id}`, { signal });
  if (!res.ok) throw new Error(`Failed to fetch sign ${id} (${res.status})`);
  return res.json();
}

/** `count` random signs from `tag`, excluding `excludeIds` (personalised
 *  session builder: a category quota that skips signs already learned or
 *  already queued). */
export async function fetchRandomSignsByTag(
  tag: string,
  count: number,
  excludeIds: number[],
  signal?: AbortSignal
): Promise<Sign[]> {
  const params = new URLSearchParams({ tag, count: String(count) });
  if (excludeIds.length) params.set("exclude", excludeIds.join(","));

  const res = await fetch(`${API_BASE}/signs/sample?${params.toString()}`, { signal });
  if (!res.ok) throw new Error(`Failed to fetch sample signs for ${tag} (${res.status})`);
  const data = await res.json();
  return data.results;
}

export async function fetchTags(signal?: AbortSignal): Promise<TagCount[]> {
  const res = await fetch(`${API_BASE}/signs/tags`, { signal });
  if (!res.ok) throw new Error(`Failed to fetch tags (${res.status})`);
  const data = await res.json();
  return data.tags;
}

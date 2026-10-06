import type { Sign, SignLevel, SignsResponse, TagCount } from "./types";
import type { SmplxPoseJson } from "../lib/smplx";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "/api";

export type SignSort =
  | "gloss_asc"
  | "gloss_desc"
  | "level_asc"
  | "level_desc"
  | "popularity_asc"
  | "popularity_desc";

export interface FetchSignsParams {
  query?: string;
  tag?: string;
  level?: SignLevel;
  sort?: SignSort;
  page?: number;
  pageSize?: number;
  signal?: AbortSignal;
}

export async function fetchSigns({
  query,
  tag,
  level,
  sort,
  page,
  pageSize,
  signal,
}: FetchSignsParams): Promise<SignsResponse> {
  const params = new URLSearchParams();
  if (query) params.set("query", query);
  if (tag) params.set("tag", tag);
  if (level) params.set("level", level);
  if (sort) params.set("sort", sort);
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

/** SMPL-X clip for the skeleton preview; null when this sign has none yet. */
export async function fetchSignPose(id: number, signal?: AbortSignal): Promise<SmplxPoseJson | null> {
  const res = await fetch(`${API_BASE}/signs/${id}/pose`, { signal });
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`Failed to fetch pose for sign ${id} (${res.status})`);
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

/** Tag categories with counts. `excludeIds` (personalised session builder:
 *  already-learned/queued signs) makes each count reflect what's actually
 *  still available to pick, not the category's raw size. */
export async function fetchTags(excludeIds?: number[], signal?: AbortSignal): Promise<TagCount[]> {
  const params = new URLSearchParams();
  if (excludeIds?.length) params.set("exclude", excludeIds.join(","));
  const res = await fetch(`${API_BASE}/signs/tags?${params.toString()}`, { signal });
  if (!res.ok) throw new Error(`Failed to fetch tags (${res.status})`);
  const data = await res.json();
  return data.tags;
}

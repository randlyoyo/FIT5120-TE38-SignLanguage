import type { Sign, SignsResponse } from "./types";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "/api";

export interface FetchSignsParams {
  query?: string;
  page?: number;
  pageSize?: number;
  signal?: AbortSignal;
}

export async function fetchSigns({
  query,
  page,
  pageSize,
  signal,
}: FetchSignsParams): Promise<SignsResponse> {
  const params = new URLSearchParams();
  if (query) params.set("query", query);
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

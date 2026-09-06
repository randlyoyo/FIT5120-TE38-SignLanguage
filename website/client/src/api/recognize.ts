// Client for the sign-recognition backend. Contract: recognition/API.md.
// The feature pipeline (resampling, centring, scaling) runs server-side --
// this file only ships the raw per-frame landmark arrays (§6) and reads the
// server's verdict back.

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "/api";

/** A `[frame][landmark][x, y]` capture, exactly the shape recognition/API.md
 *  §6 expects. A missing landmark is `[-999, -999]`, never `[0, 0]` --
 *  `(0, 0)` is a legal on-frame position, not "absent". */
export interface Capture {
  width: number;
  height: number;
  fps: number;
  pose: number[][][];
  left_hand: number[][][];
  right_hand: number[][][];
}

export interface VerifyResult {
  word: string;
  distance: number;
  threshold: number;
  matched: boolean;
  frames: number;
}

export interface IdentifyCandidate {
  word: string;
  distance: number;
}

export interface IdentifyResult {
  candidates: IdentifyCandidate[];
  margin: number;
  confident: boolean;
  marginThreshold: number;
  frames: number;
}

export type RecognizeErrorCode = "unusable_capture" | "unknown_word" | "model_unavailable" | "unknown_error";

/** Carries the server's error code (API.md §6 "Errors") so the UI can tell a
 *  bad capture (ask for a retake) apart from the model not being deployed
 *  yet (recognition/ASSETS.md) apart from an unrecognised word. */
export class RecognizeApiError extends Error {
  status: number;
  code: RecognizeErrorCode;

  constructor(status: number, code: RecognizeErrorCode, message: string) {
    super(message);
    this.status = status;
    this.code = code;
  }
}

async function postJson<T>(path: string, body: unknown, signal?: AbortSignal): Promise<T> {
  const res = await fetch(`${API_BASE}/recognize/${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  const data = await res.json().catch(() => null);
  if (!res.ok) {
    const code: RecognizeErrorCode = data?.error ?? "unknown_error";
    const message = data?.detail ?? data?.word ?? `Recognition request failed (${res.status})`;
    throw new RecognizeApiError(res.status, code, message);
  }
  return data as T;
}

export function verifySign(word: string, capture: Capture, signal?: AbortSignal): Promise<VerifyResult> {
  return postJson<VerifyResult>("verify", { word, capture }, signal);
}

export function identifySign(capture: Capture, signal?: AbortSignal): Promise<IdentifyResult> {
  return postJson<IdentifyResult>("identify", { capture }, signal);
}

// The vocabulary is fixed for the lifetime of the deployed model, so cache
// it module-wide -- every sign detail page would otherwise re-fetch the same
// 3215-word list on every visit.
let vocabularyPromise: Promise<Set<string>> | null = null;

/** Which glosses the recognizer knows, so the UI can hide the practice
 *  feature entirely for a sign outside the closed vocabulary (API.md §9)
 *  instead of letting a learner hit `unknown_word`. Resolves to an empty
 *  set (never rejects) if the recognizer isn't deployed yet, so callers can
 *  treat "not ready" and "not supported" the same way: no button shown. */
export function fetchRecognitionVocabulary(): Promise<Set<string>> {
  if (!vocabularyPromise) {
    vocabularyPromise = fetch(`${API_BASE}/recognize/vocabulary`)
      .then((res) => (res.ok ? res.json() : { words: [] }))
      .then((data: { words?: string[] }) => new Set(data.words ?? []))
      .catch(() => new Set<string>());
  }
  return vocabularyPromise;
}

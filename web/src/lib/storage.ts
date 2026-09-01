/**
 * All persistence for this app is localStorage-only: no account, no server,
 * matching the proposal's privacy stance (webcam frames and any derived
 * practice/quiz stats never leave the device).
 */

export interface ProgressEntry {
  attempts: number;
  correct: number;
  lastConfidence: number;
  lastPracticedAt: string;
}

export type ProgressMap = Record<string, ProgressEntry>;

const PROGRESS_KEY = "auslan-demo.progress.v1";
const SETTINGS_KEY = "auslan-demo.settings.v1";

export function loadProgress(): ProgressMap {
  try {
    return JSON.parse(localStorage.getItem(PROGRESS_KEY) ?? "{}") as ProgressMap;
  } catch {
    return {};
  }
}

export function recordAttempt(signId: string, correct: boolean, confidence: number): ProgressMap {
  const progress = loadProgress();
  const entry = progress[signId] ?? { attempts: 0, correct: 0, lastConfidence: 0, lastPracticedAt: "" };
  entry.attempts += 1;
  if (correct) entry.correct += 1;
  entry.lastConfidence = confidence;
  entry.lastPracticedAt = new Date().toISOString();
  progress[signId] = entry;
  localStorage.setItem(PROGRESS_KEY, JSON.stringify(progress));
  return progress;
}

export interface AccessibilitySettings {
  textScale: number;
  highContrast: boolean;
  reduceMotion: boolean;
}

export const DEFAULT_SETTINGS: AccessibilitySettings = {
  textScale: 1,
  highContrast: false,
  reduceMotion: false,
};

export function loadSettings(): AccessibilitySettings {
  try {
    return { ...DEFAULT_SETTINGS, ...(JSON.parse(localStorage.getItem(SETTINGS_KEY) ?? "{}") as Partial<AccessibilitySettings>) };
  } catch {
    return DEFAULT_SETTINGS;
  }
}

export function saveSettings(settings: AccessibilitySettings): void {
  localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
}

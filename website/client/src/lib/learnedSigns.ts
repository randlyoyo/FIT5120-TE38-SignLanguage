const KEY = "auslan-website.learnedSigns.v1";

function readIds(): number[] {
  try {
    const raw = window.localStorage.getItem(KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed.filter((n) => typeof n === "number") : [];
  } catch {
    return [];
  }
}

function writeIds(ids: number[]) {
  window.localStorage.setItem(KEY, JSON.stringify(ids));
}

export function isLearned(id: number): boolean {
  return readIds().includes(id);
}

export function toggleLearned(id: number): boolean {
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
}

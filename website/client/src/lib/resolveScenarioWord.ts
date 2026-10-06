import { fetchSigns } from "../api/signs";

/** Resolves a scenario word to its real sign id (exact gloss match in the
 *  library), or null if the dataset no longer has one -- shared by
 *  ScenarioWordLink (navigates on click) and ScenarioLearningPage (checks
 *  learned-status for every word up front, to show real progress). */
export async function resolveScenarioWord(word: string): Promise<number | null> {
  const res = await fetchSigns({ query: word, pageSize: 10 });
  const match = res.results.find((s) => s.gloss.toUpperCase() === word.toUpperCase());
  return match?.id ?? null;
}

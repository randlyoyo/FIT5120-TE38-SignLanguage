// Verifies every scenario step's words resolve to an exact gloss in the
// live library -- ScenarioWordLink falls back to a search result rather
// than erroring on a miss, so a typo here would silently teach the wrong
// sign instead of failing loudly. Run with the server's API up:
//   node scripts/checkScenarioWords.mjs [API_BASE=http://localhost:4000/api]
import { readFileSync } from "node:fs";

const API_BASE = process.argv[2] ?? "http://localhost:4000/api";
const src = readFileSync(new URL("../src/lib/scenarioStories.ts", import.meta.url), "utf8");
const words = [...src.matchAll(/words:\s*\[([^\]]*)\]/g)].flatMap((m) =>
  [...m[1].matchAll(/"([^"]+)"/g)].map((w) => w[1])
);
const unique = [...new Set(words)];

let failed = 0;
for (const word of unique) {
  const res = await fetch(`${API_BASE}/signs?query=${encodeURIComponent(word)}&pageSize=10`);
  const data = await res.json();
  const exact = data.results?.some((s) => s.gloss.toUpperCase() === word.toUpperCase());
  if (!exact) {
    failed++;
    console.error(`MISSING exact match: ${word}`);
  }
}

console.log(`${unique.length - failed}/${unique.length} words have an exact gloss match.`);
process.exit(failed > 0 ? 1 : 0);

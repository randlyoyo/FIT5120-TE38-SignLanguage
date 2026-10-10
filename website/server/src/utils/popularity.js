/**
 * "How common is this word" isn't in the dataset either, so unlike
 * difficulty.js's self-contained heuristic, this ranks each gloss against a
 * real English word-frequency list (popular-english-words: ~225k words
 * ranked by occurrence count across English Wikipedia) instead of inventing
 * a proxy from the sign data itself.
 *
 * A multi-word gloss is looked up word by word and takes the *worst*
 * (least common) rank among them -- a phrase reads only as common as its
 * rarest component, e.g. "HOW ARE YOU" is bounded by "how"/"are"/"you"
 * being ordinary, while "BALL-UP (AUSTRALIAN RULES FOOTBALL)" is bounded by
 * an obscure "ball"/"up" cousin only once its parenthetical qualifier is
 * stripped away (it names a sense, not part of the word). A gloss where
 * every word is missing from the frequency list (proper nouns, technical
 * terms) has no rank at all, rather than being scored as maximally rare.
 *
 * ESM-only package inside this CommonJS server -- loaded once via dynamic
 * import and cached, the same lazy-load-once pattern as recognition's
 * encoder in recognition/model.js.
 */
let rankMapPromise = null;

function getPopularityRankMap() {
  if (!rankMapPromise) {
    // `words` is the package's own Words instance, not a plain array --
    // getAll() hands back its internal list so this can index it once into
    // a Map instead of the O(n) linear scan its own getWordRank() does.
    rankMapPromise = import("popular-english-words").then(({ words }) => {
      const list = words.getAll();
      const map = new Map();
      for (let i = 0; i < list.length; i++) {
        // Keep the first (most common) occurrence if a word repeats.
        if (!map.has(list[i])) map.set(list[i], i);
      }
      return map;
    });
  }
  return rankMapPromise;
}

function wordsInGloss(gloss) {
  return gloss
    .replace(/\([^)]*\)/g, " ")
    .toLowerCase()
    .split(/[^a-z]+/)
    .filter(Boolean);
}

/** Lower rank number = more common. `null` when no word in the gloss is in
 *  the frequency list at all. */
function popularityRankForGloss(gloss, rankMap) {
  const ranks = wordsInGloss(gloss)
    .map((w) => rankMap.get(w))
    .filter((r) => r !== undefined);
  return ranks.length ? Math.max(...ranks) : null;
}

module.exports = { getPopularityRankMap, popularityRankForGloss };

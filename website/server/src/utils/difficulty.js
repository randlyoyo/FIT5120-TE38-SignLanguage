/**
 * Difficulty isn't in the dataset, so it's estimated from how much a gloss
 * asks of a learner: longer words, multi-word phrases, parenthetical
 * qualifiers (regional/topic variants, e.g. "DECLARE (CRICKET)"), and words
 * with more dictionary senses or keywords all read as more demanding than a
 * short, single-sense, single-word gloss.
 *
 * One SQL expression is the single source of truth (used in SELECT, WHERE
 * and ORDER BY) so filtering and sorting can never disagree with each
 * other. Thresholds are the real 33rd/66th percentile of this score across
 * the actual 3215-word vocabulary (measured directly against the database,
 * not guessed), so the three levels land roughly even instead of skewed.
 */
const DIFFICULTY_SQL = `(
  CHAR_LENGTH(gloss)
  + IF(INSTR(gloss, ' ') > 0, 8, 0)
  + IF(INSTR(gloss, '(') > 0, 6, 0)
  + COALESCE(JSON_LENGTH(definitions), 0) * 2
  + COALESCE(JSON_LENGTH(keywords), 0)
)`;

const LEVEL_CONDITION_SQL = {
  beginner: `${DIFFICULTY_SQL} <= 9`,
  intermediate: `${DIFFICULTY_SQL} BETWEEN 10 AND 12`,
  advanced: `${DIFFICULTY_SQL} >= 13`,
};

function levelForScore(score) {
  if (score <= 9) return "beginner";
  if (score <= 12) return "intermediate";
  return "advanced";
}

module.exports = { DIFFICULTY_SQL, LEVEL_CONDITION_SQL, levelForScore };

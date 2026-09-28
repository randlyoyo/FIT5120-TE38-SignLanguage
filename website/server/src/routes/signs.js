const express = require("express");
const pool = require("../db");
const { parsePagination, buildPaginationMeta } = require("../utils/pagination");
const { formatSign } = require("../utils/formatSign");
const { formatVideo } = require("../utils/video");
const { DIFFICULTY_SQL, LEVEL_CONDITION_SQL, levelForScore } = require("../utils/difficulty");
const { getPopularityRankMap, popularityRankForGloss } = require("../utils/popularity");

const router = express.Router();

// Additional sort orders on top of the default gloss order -- "level_*"
// reuses DIFFICULTY_SQL's SELECT alias so it never has to repeat the
// scoring expression.
const SORTS = {
  gloss_asc: "gloss ASC",
  gloss_desc: "gloss DESC",
  id_asc: "id ASC",
  id_desc: "id DESC",
  level_asc: "difficulty_score ASC, gloss ASC",
  level_desc: "difficulty_score DESC, gloss ASC",
};

// GET /api/signs?query=&tag=&level=&sort=&page=&pageSize=
// Keyword search (US1.1) across gloss, keywords and definitions, an
// optional exact `tag` filter (US1.x tag browsing), an optional estimated
// `level` filter (beginner/intermediate/advanced, see utils/difficulty.js),
// a `sort` order, plus pagination. When `query` is given, results are
// ranked keyword matches first (gloss/keywords), tag matches second,
// definition matches last -- `keywords` and `tags` are distinct fields in
// the schema (search-only synonyms vs. visible classification), and search
// order should reflect that distinction, so `sort` is ignored whenever a
// free-text query is present.
router.get("/", async (req, res, next) => {
  try {
    const { query = "", tag = "", level = "", sort = "" } = req.query;
    const { page, pageSize, offset } = parsePagination(req.query);
    const trimmedQuery = query.trim();
    const trimmedTag = tag.trim();
    const trimmedLevel = level.trim().toLowerCase();

    const conditions = [];
    const params = [];

    if (trimmedQuery) {
      const like = `%${trimmedQuery}%`;
      conditions.push(
        "(gloss LIKE ? OR JSON_SEARCH(keywords, 'one', ?) IS NOT NULL OR JSON_SEARCH(tags, 'one', ?) IS NOT NULL OR JSON_SEARCH(definitions, 'one', ?) IS NOT NULL)"
      );
      params.push(like, like, like, like);
    }

    if (trimmedTag) {
      conditions.push("JSON_CONTAINS(tags, JSON_QUOTE(?))");
      params.push(trimmedTag);
    }

    if (LEVEL_CONDITION_SQL[trimmedLevel]) {
      conditions.push(LEVEL_CONDITION_SQL[trimmedLevel]);
    }

    const whereClause = conditions.length ? `WHERE ${conditions.join(" AND ")}` : "";

    const [countRows] = await pool.query(
      `SELECT COUNT(*) AS total FROM signs ${whereClause}`,
      params
    );
    const totalResults = countRows[0].total;

    let orderClause = "ORDER BY gloss ASC";
    let orderParams = [];
    if (trimmedQuery) {
      const like = `%${trimmedQuery}%`;
      const prefixLike = `${trimmedQuery}%`;
      // Rank an exact gloss match first, then a gloss that starts with the
      // query, before falling back to "contains" -- otherwise "love" and
      // "glove" tie (both match gloss LIKE '%love%') and the alphabetical
      // tiebreak puts "glove" ahead of the exact "love" match.
      orderClause = `ORDER BY
        CASE
          WHEN gloss = ? THEN 0
          WHEN gloss LIKE ? THEN 1
          WHEN gloss LIKE ? THEN 2
          WHEN JSON_SEARCH(keywords, 'one', ?) IS NOT NULL THEN 3
          WHEN JSON_SEARCH(tags, 'one', ?) IS NOT NULL THEN 4
          ELSE 5
        END ASC,
        gloss ASC`;
      orderParams = [trimmedQuery, prefixLike, like, like, like];
    } else if (SORTS[sort]) {
      orderClause = `ORDER BY ${SORTS[sort]}`;
    }

    // Popularity isn't a SQL column -- it's ranked against an external
    // word-frequency list in JS (utils/popularity.js) -- so this sort can't
    // join the ORDER BY above. Instead it fetches every row the existing
    // WHERE already matches (same conditions/params, so `totalResults`
    // above still applies), ranks and sorts in JS, then slices the page.
    // 3215 signs total, so a full fetch here is a non-issue at this scale.
    let rows;
    if (!trimmedQuery && (sort === "popularity_asc" || sort === "popularity_desc")) {
      const [allRows] = await pool.query(
        `SELECT *, ${DIFFICULTY_SQL} AS difficulty_score FROM signs ${whereClause} ORDER BY gloss ASC`,
        params
      );
      const rankMap = await getPopularityRankMap();
      const ranked = allRows.map((row) => ({ row, rank: popularityRankForGloss(row.gloss, rankMap) }));
      ranked.sort((a, b) => {
        // Unranked (no word of the gloss is in the frequency list at all --
        // proper nouns, technical terms) sort after every ranked word,
        // alphabetically among themselves rather than at a fake rank.
        if (a.rank === null && b.rank === null) return a.row.gloss.localeCompare(b.row.gloss);
        if (a.rank === null) return 1;
        if (b.rank === null) return -1;
        return sort === "popularity_asc" ? a.rank - b.rank : b.rank - a.rank;
      });
      rows = ranked.slice(offset, offset + pageSize).map((r) => r.row);
    } else {
      [rows] = await pool.query(
        `SELECT *, ${DIFFICULTY_SQL} AS difficulty_score FROM signs ${whereClause} ${orderClause} LIMIT ? OFFSET ?`,
        [...params, ...orderParams, pageSize, offset]
      );
    }

    // One short (~2s) preview clip per card, so the library grid can play
    // the actual sign instead of a static thumbnail -- cheap enough at the
    // default page size of 4 to fetch eagerly rather than lazy-load.
    const signIds = rows.map((row) => row.id);
    const previewBySignId = new Map();
    if (signIds.length) {
      const [videoRows] = await pool.query(
        `SELECT sign_id, source_id, file_name, video_url, movement_description
         FROM sign_videos
         WHERE sign_id IN (?)
         ORDER BY sign_id ASC, source_id ASC`,
        [signIds]
      );
      for (const video of videoRows) {
        if (!previewBySignId.has(video.sign_id)) {
          previewBySignId.set(video.sign_id, formatVideo(video));
        }
      }
    }

    res.json({
      results: rows.map((row) => ({
        ...formatSign(row),
        previewVideo: previewBySignId.get(row.id) ?? null,
        level: levelForScore(row.difficulty_score),
      })),
      pagination: buildPaginationMeta(page, pageSize, totalResults),
      query: { query: trimmedQuery || null, tag: trimmedTag || null },
    });
  } catch (err) {
    next(err);
  }
});

// GET /api/signs/tags?exclude= -- distinct tag categories with counts, for
// the library page's category rail (browse-by-topic navigation) and the
// personalised session builder (US5.3: counts must reflect what's actually
// left to pick, so a learner isn't told "12 available" for a category
// where 12 are already learned). `exclude` is optional and unused by the
// category rail. Registered before /:id so "tags" doesn't get swallowed as
// an id param.
router.get("/tags", async (req, res, next) => {
  try {
    const excludeIds = new Set(
      String(req.query.exclude ?? "")
        .split(",")
        .map((s) => Number(s.trim()))
        .filter((n) => Number.isInteger(n) && n > 0)
    );
    const [rows] = await pool.query("SELECT id, tags FROM signs");
    const counts = new Map();
    for (const row of rows) {
      if (excludeIds.has(row.id)) continue;
      for (const tag of row.tags ?? []) {
        counts.set(tag, (counts.get(tag) ?? 0) + 1);
      }
    }
    const tags = [...counts.entries()]
      .map(([tag, count]) => ({ tag, count }))
      .sort((a, b) => a.tag.localeCompare(b.tag));
    res.json({ tags });
  } catch (err) {
    next(err);
  }
});

// GET /api/signs/sample?tag=&count=&exclude= -- `count` random signs from
// `tag` (US5.1/5.2: personalised session builder picks a count per
// category), optionally excluding a comma-separated id list (US5.3: already-
// learned signs, tracked client-side in localStorage, so the exclusion list
// has to come from the caller rather than a server-side "learned" table).
// Registered before /:id for the same reason as /tags above.
router.get("/sample", async (req, res, next) => {
  try {
    const tag = String(req.query.tag ?? "").trim();
    const count = Math.min(Math.max(Number(req.query.count) || 0, 0), 50);
    if (!tag || count === 0) {
      return res.json({ results: [] });
    }
    const excludeIds = String(req.query.exclude ?? "")
      .split(",")
      .map((s) => Number(s.trim()))
      .filter((n) => Number.isInteger(n) && n > 0);

    const conditions = ["JSON_CONTAINS(tags, JSON_QUOTE(?))"];
    const params = [tag];
    if (excludeIds.length) {
      conditions.push("id NOT IN (?)");
      params.push(excludeIds);
    }

    const [rows] = await pool.query(
      `SELECT * FROM signs WHERE ${conditions.join(" AND ")} ORDER BY RAND() LIMIT ?`,
      [...params, count]
    );

    res.json({ results: rows.map(formatSign) });
  } catch (err) {
    next(err);
  }
});

// GET /api/signs/:id -- single sign detail (US1.3, clicking a result).
router.get("/:id", async (req, res, next) => {
  try {
    const id = Number(req.params.id);
    if (!Number.isInteger(id) || id <= 0) {
      return res.status(400).json({ error: "Invalid sign id" });
    }

    const [rows] = await pool.query(
  "SELECT * FROM signs WHERE id = ?",
  [id]
);

if (rows.length === 0) {
  return res.status(404).json({ error: "Sign not found", id });
}

const [videoRows] = await pool.query(
  `SELECT source_id, file_name, video_url, movement_description
   FROM sign_videos
   WHERE sign_id = ?
   ORDER BY source_id ASC`,
  [id]
);

const sign = formatSign(rows[0]);

sign.videos = videoRows.map(formatVideo);

res.json(sign);
  } catch (err) {
    next(err);
  }
});

module.exports = router;

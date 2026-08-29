const express = require("express");
const pool = require("../db");
const { parsePagination, buildPaginationMeta } = require("../utils/pagination");
const { formatSign } = require("../utils/formatSign");
const { formatVideo } = require("../utils/video");

const router = express.Router();

// GET /api/signs?query=&tag=&page=&pageSize=
// Keyword search (US1.1) across gloss, keywords and definitions, an
// optional exact `tag` filter (US1.x tag browsing), plus pagination.
// When `query` is given, results are ranked keyword matches first
// (gloss/keywords), tag matches second, definition matches last --
// `keywords` and `tags` are distinct fields in the schema (search-only
// synonyms vs. visible classification), and search order should reflect
// that distinction.
router.get("/", async (req, res, next) => {
  try {
    const { query = "", tag = "" } = req.query;
    const { page, pageSize, offset } = parsePagination(req.query);
    const trimmedQuery = query.trim();
    const trimmedTag = tag.trim();

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
      orderClause = `ORDER BY
        CASE
          WHEN gloss LIKE ? THEN 0
          WHEN JSON_SEARCH(keywords, 'one', ?) IS NOT NULL THEN 1
          WHEN JSON_SEARCH(tags, 'one', ?) IS NOT NULL THEN 2
          ELSE 3
        END ASC,
        gloss ASC`;
      orderParams = [like, like, like];
    }

    const [rows] = await pool.query(
      `SELECT * FROM signs ${whereClause} ${orderClause} LIMIT ? OFFSET ?`,
      [...params, ...orderParams, pageSize, offset]
    );

    // One short (~2s) preview clip per card, so the library grid can play
    // the actual sign instead of a static thumbnail -- cheap enough at the
    // default page size of 4 to fetch eagerly rather than lazy-load.
    const signIds = rows.map((row) => row.id);
    const previewBySignId = new Map();
    if (signIds.length) {
      const [videoRows] = await pool.query(
        `SELECT sign_id, source_id, file_name, video_url
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
      })),
      pagination: buildPaginationMeta(page, pageSize, totalResults),
      query: { query: trimmedQuery || null, tag: trimmedTag || null },
    });
  } catch (err) {
    next(err);
  }
});

// GET /api/signs/tags -- distinct tag categories with counts, for the
// library page's category rail (browse-by-topic navigation). Registered
// before /:id so "tags" doesn't get swallowed as an id param.
router.get("/tags", async (req, res, next) => {
  try {
    const [rows] = await pool.query("SELECT tags FROM signs");
    const counts = new Map();
    for (const row of rows) {
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
  `SELECT source_id, file_name, video_url
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

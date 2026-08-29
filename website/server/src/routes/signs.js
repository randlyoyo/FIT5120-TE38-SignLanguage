const express = require("express");
const pool = require("../db");
const { parsePagination, buildPaginationMeta } = require("../utils/pagination");
const { formatSign } = require("../utils/formatSign");

const router = express.Router();

const VIDEO_BASE_URL = (process.env.VIDEO_BASE_URL || "").replace(/\/+$/, "");

// GET /api/signs?query=&page=&pageSize=
// Keyword search (US1.1) across gloss, tags, keywords and definitions,
// plus pagination. There is no separate category filter -- classification
// now lives in `tags`, which is just another field this search matches.
router.get("/", async (req, res, next) => {
  try {
    const { query = "" } = req.query;
    const { page, pageSize, offset } = parsePagination(req.query);

    const conditions = [];
    const params = [];

    if (query.trim()) {
      const like = `%${query.trim()}%`;
      conditions.push(
        "(gloss LIKE ? OR JSON_SEARCH(tags, 'one', ?) IS NOT NULL OR JSON_SEARCH(keywords, 'one', ?) IS NOT NULL OR JSON_SEARCH(definitions, 'one', ?) IS NOT NULL)"
      );
      params.push(like, like, like, like);
    }

    const whereClause = conditions.length ? `WHERE ${conditions.join(" AND ")}` : "";

    const [countRows] = await pool.query(
      `SELECT COUNT(*) AS total FROM signs ${whereClause}`,
      params
    );
    const totalResults = countRows[0].total;

    const [rows] = await pool.query(
      `SELECT * FROM signs ${whereClause} ORDER BY gloss ASC LIMIT ? OFFSET ?`,
      [...params, pageSize, offset]
    );

    res.json({
      results: rows.map(formatSign),
      pagination: buildPaginationMeta(page, pageSize, totalResults),
      query: { query: query.trim() || null },
    });
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

sign.videos = videoRows.map((video) => ({
  sourceId: video.source_id,
  fileName: video.file_name,
  videoUrl:
    video.video_url ||
    (VIDEO_BASE_URL
      ? `${VIDEO_BASE_URL}/${encodeURIComponent(video.file_name)}`
      : null),
}));

res.json(sign);
  } catch (err) {
    next(err);
  }
});

module.exports = router;

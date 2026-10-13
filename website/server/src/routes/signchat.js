const crypto = require("crypto");
const express = require("express");
const pool = require("../db");

const router = express.Router();

// The Sign Chat models run on Colab behind a quick tunnel whose URL changes on
// every restart. The notebook reports the new URL here (PUT, with a shared
// token) and the client reads it (GET) when Sign Chat opens -- so a restart
// needs no client redeploy. One row in a small key/value table, so the URL
// survives an API restart.
const KEY = "signchat_url";
const TABLE_SQL = `CREATE TABLE IF NOT EXISTS app_settings (
  name VARCHAR(64) NOT NULL PRIMARY KEY,
  value TEXT NOT NULL,
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB`;

// GET /api/signchat-url -> { url, updatedAt }; 404 until the notebook has reported one.
router.get("/", async (req, res, next) => {
  try {
    const [rows] = await pool.query("SELECT value, updated_at FROM app_settings WHERE name = ?", [KEY]);
    if (!rows.length) return res.status(404).json({ error: "No Sign Chat URL reported yet" });
    res.set("Cache-Control", "no-store");
    res.json({ url: rows[0].value, updatedAt: rows[0].updated_at });
  } catch (err) {
    if (err.code === "ER_NO_SUCH_TABLE") return res.status(404).json({ error: "No Sign Chat URL reported yet" });
    next(err);
  }
});

function tokenMatches(given, expected) {
  const a = Buffer.from(given);
  const b = Buffer.from(expected);
  return a.length === b.length && crypto.timingSafeEqual(a, b);
}

// PUT /api/signchat-url  Authorization: Bearer <SIGNCHAT_UPDATE_TOKEN>  { url }
router.put("/", async (req, res, next) => {
  const expected = process.env.SIGNCHAT_UPDATE_TOKEN;
  if (!expected) return res.status(503).json({ error: "SIGNCHAT_UPDATE_TOKEN is not set on the server" });
  const given = (req.get("authorization") || "").replace(/^Bearer\s+/i, "");
  if (!tokenMatches(given, expected)) return res.status(401).json({ error: "Bad token" });

  let url;
  try {
    url = new URL(String(req.body?.url ?? ""));
  } catch {
    return res.status(400).json({ error: "url must be an absolute URL" });
  }
  if (url.protocol !== "https:") return res.status(400).json({ error: "url must use https" });
  const value = url.origin;

  try {
    await pool.query(TABLE_SQL);
    await pool.query(
      "INSERT INTO app_settings (name, value) VALUES (?, ?) ON DUPLICATE KEY UPDATE value = VALUES(value)",
      [KEY, value]
    );
    res.json({ url: value });
  } catch (err) {
    next(err);
  }
});

module.exports = router;

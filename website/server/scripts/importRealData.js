require("dotenv").config();
const fs = require("fs");
const path = require("path");
const pool = require("../src/db");

const signs = JSON.parse(
  fs.readFileSync(path.join(__dirname, "../data/signs.json"), "utf8")
);

const videos = JSON.parse(
  fs.readFileSync(path.join(__dirname, "../data/sign_videos.json"), "utf8")
);

async function importRealData() {
  const conn = await pool.getConnection();

  try {
    await conn.beginTransaction();

    for (const sign of signs) {
  await conn.query(
    `INSERT INTO signs
      (gloss, definitions, usage_notes, tags, keywords, source)
     VALUES (?, CAST(? AS JSON), CAST(? AS JSON), CAST(? AS JSON), CAST(? AS JSON), ?)
     ON DUPLICATE KEY UPDATE
       definitions = VALUES(definitions),
       usage_notes = VALUES(usage_notes),
       tags = VALUES(tags),
       keywords = VALUES(keywords),
       source = VALUES(source)`,
    [
      sign.gloss,
      JSON.stringify(sign.definitions),
      JSON.stringify(sign.usage_notes),
      JSON.stringify(sign.tags),
      JSON.stringify(sign.keywords),
      sign.source,
    ]
  );
}

    const [signRows] = await conn.query("SELECT id, gloss FROM signs");
    const signIdByGloss = new Map(signRows.map((row) => [row.gloss, row.id]));

    for (const video of videos) {
      const signId = signIdByGloss.get(video.gloss);

      if (!signId) {
        throw new Error(`No sign row found for gloss: ${video.gloss}`);
      }

      await conn.query(
        `INSERT INTO sign_videos
          (sign_id, source_id, file_name, video_url)
         VALUES (?, ?, ?, ?)
         ON DUPLICATE KEY UPDATE
           sign_id = VALUES(sign_id),
           file_name = VALUES(file_name),
           video_url = VALUES(video_url)`,
        [signId, video.source_id, video.file_name, video.video_url]
      );
    }

    await conn.commit();

    const [[signCount]] = await conn.query(
      "SELECT COUNT(*) AS count FROM signs"
    );
    const [[videoCount]] = await conn.query(
      "SELECT COUNT(*) AS count FROM sign_videos"
    );

    console.log(`Imported/verified ${signCount.count} signs.`);
    console.log(`Imported/verified ${videoCount.count} sign videos.`);
  } catch (err) {
    await conn.rollback();
    throw err;
  } finally {
    conn.release();
    await pool.end();
  }
}

importRealData().catch((err) => {
  console.error(err);
  process.exit(1);
});

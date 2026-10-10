// Loads <GLOSS>.json.gz files from scripts/convert_poses.py into sign_poses,
// matched to signs by exact gloss (the npz file name).
// Usage: node scripts/importPoses.js <dir_of_json_gz>
const fs = require("fs");
const path = require("path");
const pool = require("../src/db");

async function main() {
  const dir = process.argv[2];
  if (!dir) throw new Error("Usage: node scripts/importPoses.js <dir_of_json_gz>");

  // No foreign key: the deployed `signs` table has no primary/unique key on
  // id to reference (unlike schema.sql), so MySQL rejects the constraint.
  await pool.query(`CREATE TABLE IF NOT EXISTS sign_poses (
    sign_id INT UNSIGNED NOT NULL PRIMARY KEY,
    pose_gz MEDIUMBLOB NOT NULL,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
  ) ENGINE=InnoDB`);

  const files = fs.readdirSync(dir).filter((f) => f.endsWith(".json.gz"));
  const unmatched = [];
  let imported = 0;

  for (const file of files) {
    const gloss = file.slice(0, -".json.gz".length);
    const [rows] = await pool.query("SELECT id FROM signs WHERE gloss = ?", [gloss]);
    if (rows.length === 0) {
      unmatched.push(gloss);
      continue;
    }
    await pool.query(
      `INSERT INTO sign_poses (sign_id, pose_gz) VALUES (?, ?)
       ON DUPLICATE KEY UPDATE pose_gz = VALUES(pose_gz)`,
      [rows[0].id, fs.readFileSync(path.join(dir, file))]
    );
    imported++;
  }

  console.log(`Imported ${imported}/${files.length} pose clip(s).`);
  if (unmatched.length) console.log(`No sign with gloss: ${unmatched.join(", ")}`);
}

main()
  .catch((err) => {
    console.error(err);
    process.exitCode = 1;
  })
  .finally(() => pool.end());

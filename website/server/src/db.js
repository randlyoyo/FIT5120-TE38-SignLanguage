require("dotenv").config();
const mysql = require("mysql2/promise");

const pool = mysql.createPool({
  host: process.env.DB_HOST || "localhost",
  port: Number(process.env.DB_PORT) || 3306,
  user: process.env.DB_USER || "root",
  password: process.env.DB_PASSWORD || "",
  database: process.env.DB_NAME || "auslan_learning",
  waitForConnections: true,
  connectionLimit: 10,
  // Railway drops idle connections; without keepalive the first query after
  // a quiet spell hits a dead socket and fails with ECONNRESET.
  enableKeepAlive: true,
  keepAliveInitialDelay: 10000,
});

module.exports = pool;

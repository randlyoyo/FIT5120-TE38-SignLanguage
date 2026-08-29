const VIDEO_BASE_URL = (process.env.VIDEO_BASE_URL || "").replace(/\/+$/, "");

/** Resolves a sign_videos row to the API's camelCase video shape. */
function formatVideo(row) {
  return {
    sourceId: row.source_id,
    fileName: row.file_name,
    videoUrl:
      row.video_url ||
      (VIDEO_BASE_URL
        ? `${VIDEO_BASE_URL}/${encodeURIComponent(row.file_name)}`
        : null),
  };
}

module.exports = { formatVideo };

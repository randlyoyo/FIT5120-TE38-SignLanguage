/** Maps a raw MySQL `signs` row (snake_case) to the API's camelCase JSON shape. */
function formatSign(row) {
  return {
    id: row.id,
    gloss: row.gloss,
    definitions: row.definitions,
    usageNotes: row.usage_notes,
    source: row.source,
    tags: row.tags ?? [],
    keywords: row.keywords ?? [],
  };
}

module.exports = { formatSign };

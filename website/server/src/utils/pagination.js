const DEFAULT_PAGE_SIZE = 6;
const MAX_PAGE_SIZE = 50;

/** Parses page/pageSize query params into safe, clamped integers. */
function parsePagination(query) {
  const page = Math.max(1, parseInt(query.page, 10) || 1);
  const pageSize = Math.min(
    MAX_PAGE_SIZE,
    Math.max(1, parseInt(query.pageSize, 10) || DEFAULT_PAGE_SIZE)
  );
  const offset = (page - 1) * pageSize;
  return { page, pageSize, offset };
}

function buildPaginationMeta(page, pageSize, totalResults) {
  return {
    page,
    pageSize,
    totalResults,
    totalPages: Math.max(1, Math.ceil(totalResults / pageSize)),
  };
}

module.exports = { parsePagination, buildPaginationMeta };

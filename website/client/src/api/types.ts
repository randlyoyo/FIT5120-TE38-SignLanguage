export interface DefinitionGroup {
  partOfSpeech: string;
  senses: string[];
}

export interface SignVideo {
  sourceId: string;
  fileName: string;
  videoUrl: string | null;
  movementDescription: string | null;
}

export type SignLevel = "beginner" | "intermediate" | "advanced";

export interface Sign {
  id: number;
  gloss: string;
  definitions: DefinitionGroup[];
  usageNotes: string[];
  source: string | null;
  tags: string[];
  keywords: string[];
  videos?: SignVideo[];
  previewVideo?: SignVideo | null;
  /** Estimated difficulty (server/src/utils/difficulty.js) -- there's no
   *  real pedagogical rating in the dataset, so this is a heuristic based
   *  on gloss length/shape and how many senses and keywords it carries. */
  level?: SignLevel;
}

export interface PaginationMeta {
  page: number;
  pageSize: number;
  totalResults: number;
  totalPages: number;
}

export interface SignsResponse {
  results: Sign[];
  pagination: PaginationMeta;
  query: { query: string | null; tag: string | null };
}

export interface TagCount {
  tag: string;
  count: number;
}

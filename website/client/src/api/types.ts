export interface DefinitionGroup {
  partOfSpeech: string;
  senses: string[];
}

export interface SignVideo {
  sourceId: string;
  fileName: string;
  videoUrl: string | null;
}

export interface Sign {
  id: number;
  gloss: string;
  definitions: DefinitionGroup[];
  usageNotes: string[];
  source: string | null;
  tags: string[];
  keywords: string[];
  videos?: SignVideo[];
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
  query: { query: string | null };
}

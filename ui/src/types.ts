export type Lang = "en" | "fr";

export interface Chunk {
  chunk_id: string;
  unit_id: string;
  regulation: string;
  lang: Lang;
  text: string;
  header: string;
  paragraph: string | null;
  title: string;
}

export interface SearchHit {
  chunk: Chunk;
  score: number;
  rank: number;
  source: string;
}

export interface Citation {
  marker: number;
  chunk_id: string;
  unit_id: string;
  label: string;
  quote: string;
}

export interface Answer {
  question: string;
  language: Lang;
  answer: string;
  citations: Citation[];
  abstained: boolean;
  abstain_reason: string | null;
  provider: string;
  latency_ms?: number;
}

export interface Paragraph {
  number: string | null;
  lead: string;
  points: string[];
}

export interface LegalUnit {
  unit_id: string;
  regulation: string;
  kind: "article" | "recital" | "annex";
  number: string;
  title: string;
  lang: Lang;
  chapter: string;
  section: string;
  paragraphs: Paragraph[];
  source_url: string;
}

export interface Health {
  status: string;
  version: string;
  units: number;
  chunks: number;
  retriever: string;
  reranker: string;
  llm: string;
  disclaimer: string;
}

export interface Turn {
  id: string;
  question: string;
  streamed: string;
  hits: SearchHit[];
  answer?: Answer;
  error?: string;
  pending: boolean;
}

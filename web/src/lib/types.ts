export type TaskMode = "auto" | "summarize" | "extract" | "compare";

export interface SourceRef {
  source_number: number;
  chunk_id: string;
  source_name: string;
  page_label: string;
  file_path: string;
  source_label?: string;
  snippet?: string;
}

export interface ChatAnswer {
  mode: string;
  text: string;
  error?: string | null;
  assistant_note?: string | null;
  web_snippets?: Array<Record<string, string>> | null;
  sources?: SourceRef[] | null;
  validation_warning?: string | null;
  retrieval_chunk_count?: number | null;
  diagnostics?: Record<string, unknown> | null;
}

export interface MessageOut {
  role: string;
  content: string;
  extra?: Record<string, unknown>;
}

export interface ChatSessionRow {
  id: string;
  title: string;
  updated_at: number;
}

export interface DocumentRow {
  filename: string;
  health: string;
  note: string | null;
  updated_at: string | null;
  extraction_quality?: string;
  extraction_hint?: string | null;
}

export interface DocumentsListResponse {
  documents: DocumentRow[];
  count: number;
  library_needs_sync?: boolean;
}

export interface UiAssistantMeta {
  mode?: string;
  sources?: SourceRef[];
  web_sources?: Array<Record<string, string>>;
  status_note?: string;
  validation_warning?: string;
  diagnostics?: Record<string, unknown>;
}

export type UiMessage =
  | { id: string; role: "user"; content: string }
  | {
      id: string;
      role: "assistant";
      content: string;
      streaming?: boolean;
      meta?: UiAssistantMeta;
    };

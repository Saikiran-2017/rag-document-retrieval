/** Mirrors `app/services/message_service.py` product copy. */

export const APP_NAME = "Knowledge Assistant";
export const EMPTY_STATE_VALUE_PROP =
  "Chat normally. When your library applies, answers cite your files.";
export const HERO_BEST_FOR =
  "Best for a focused set of files. Works without uploads. Complements your other AI tools.";
export const SIDEBAR_CAPTION = "Upload, sync, ask.";

export const STARTER_QUESTIONS = [
  "Summarize the key points I should remember.",
  "What themes appear in my documents?",
  "Give me three concise writing tips.",
] as const;

export const TASK_MODE_OPTIONS = [
  { value: "auto", label: "Auto" },
  { value: "summarize", label: "Summarize" },
  { value: "extract", label: "Extract" },
  { value: "compare", label: "Compare" },
] as const;

export const HEALTH_LABEL: Record<string, string> = {
  ready: "Ready",
  ready_limited: "Ready · limited",
  processing: "Indexing…",
  failed: "Needs attention",
  uploaded: "Pending sync",
};

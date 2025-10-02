import type { MessageOut, UiAssistantMeta, UiMessage } from "./types";

function randomId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

export function messageOutToUi(m: MessageOut): UiMessage {
  if (m.role === "user") {
    return { id: randomId(), role: "user", content: m.content };
  }
  const x = (m.extra ?? {}) as Record<string, unknown>;
  const meta: UiAssistantMeta = {
    mode: typeof x.mode === "string" ? x.mode : undefined,
    sources: Array.isArray(x.sources) ? (x.sources as UiAssistantMeta["sources"]) : undefined,
    web_sources:
      (Array.isArray(x.web_sources) ? x.web_sources : null) ||
      (Array.isArray(x.web_snippets) ? x.web_snippets : undefined),
    status_note:
      typeof x.status_note === "string"
        ? x.status_note
        : typeof x.assistant_note === "string"
          ? x.assistant_note
          : undefined,
    validation_warning:
      typeof x.validation_warning === "string" ? x.validation_warning : undefined,
  };
  return {
    id: randomId(),
    role: "assistant",
    content: m.content,
    meta,
  };
}

export function mapMessages(rows: MessageOut[]): UiMessage[] {
  return rows.map(messageOutToUi);
}

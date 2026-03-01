import type {
  ChatAnswer,
  ChatSessionRow,
  DocumentRow,
  MessageOut,
  TaskMode,
} from "./types";

export function getApiBase(): string {
  const b = process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "");
  if (!b) {
    return "http://127.0.0.1:8000";
  }
  return b;
}

function apiUrl(path: string): string {
  return `${getApiBase()}${path.startsWith("/") ? path : `/${path}`}`;
}

/** Human-readable message from FastAPI / JSON error bodies. */
export function formatApiErrorBody(data: unknown): string {
  if (data == null || typeof data !== "object") {
    return "Request failed";
  }
  const d = data as Record<string, unknown>;
  if (typeof d.message === "string" && d.message.trim()) {
    return d.message;
  }
  const detail = d.detail;
  if (typeof detail === "string" && detail.trim()) {
    return detail;
  }
  if (Array.isArray(detail)) {
    const parts = detail.map((x) => {
      if (x && typeof x === "object" && "msg" in x) {
        return String((x as { msg?: string }).msg ?? x);
      }
      return String(x);
    });
    return parts.join("; ") || "Request failed";
  }
  return "Request failed";
}

async function readErrorBody(r: Response): Promise<string> {
  try {
    const data: unknown = await r.json();
    return formatApiErrorBody(data);
  } catch {
    return `HTTP ${r.status}`;
  }
}

/** Lightweight connectivity check (no auth). */
export async function checkApiHealth(): Promise<boolean> {
  try {
    const r = await fetch(apiUrl("/health"), { cache: "no-store" });
    return r.ok;
  } catch {
    return false;
  }
}

export async function listChats(limit = 40): Promise<ChatSessionRow[]> {
  const r = await fetch(apiUrl(`/api/v1/chats?limit=${limit}`), {
    cache: "no-store",
  });
  if (!r.ok) throw new Error(await readErrorBody(r));
  return r.json();
}

export async function createChat(title = "New chat"): Promise<{ id: string; title: string }> {
  const r = await fetch(
    apiUrl(`/api/v1/chats?title=${encodeURIComponent(title)}`),
    { method: "POST" },
  );
  if (!r.ok) throw new Error(await readErrorBody(r));
  return r.json();
}

export async function deleteChat(sessionId: string): Promise<void> {
  const r = await fetch(apiUrl(`/api/v1/chats/${sessionId}`), {
    method: "DELETE",
  });
  if (!r.ok) throw new Error(await readErrorBody(r));
}

/** Normalize API message: supports nested ``extra`` or flat legacy shape. */
export function normalizeMessageOut(raw: Record<string, unknown>): MessageOut {
  const role = String(raw.role ?? "");
  const content = String(raw.content ?? "");
  const nested = raw.extra;
  if (nested && typeof nested === "object" && !Array.isArray(nested)) {
    return {
      role,
      content,
      extra: nested as Record<string, unknown>,
    };
  }
  const skip = new Set(["role", "content", "extra"]);
  const flat: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(raw)) {
    if (!skip.has(k)) flat[k] = v;
  }
  return { role, content, extra: flat };
}

export async function getMessages(sessionId: string): Promise<MessageOut[]> {
  const r = await fetch(apiUrl(`/api/v1/chats/${sessionId}/messages`), {
    cache: "no-store",
  });
  if (!r.ok) throw new Error(await readErrorBody(r));
  const rows: unknown = await r.json();
  if (!Array.isArray(rows)) return [];
  return rows.map((row) =>
    normalizeMessageOut(row && typeof row === "object" ? (row as Record<string, unknown>) : {}),
  );
}

export async function appendMessage(
  sessionId: string,
  role: string,
  content: string,
  extra: Record<string, unknown> = {},
): Promise<void> {
  const r = await fetch(apiUrl(`/api/v1/chats/${sessionId}/messages`), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ role, content, extra }),
  });
  if (!r.ok) throw new Error(await readErrorBody(r));
}

export async function setChatTitle(sessionId: string, title: string): Promise<void> {
  const r = await fetch(apiUrl(`/api/v1/chats/${sessionId}`), {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title }),
  });
  if (!r.ok) throw new Error(await readErrorBody(r));
}

export async function listDocuments(): Promise<{ documents: DocumentRow[]; count: number }> {
  const r = await fetch(apiUrl("/api/v1/documents"), { cache: "no-store" });
  if (!r.ok) throw new Error(await readErrorBody(r));
  return r.json();
}

export interface SyncLibraryResult {
  ok: boolean;
  status: string;
  message: string;
  sync_action: string;
  vector_count: number;
}

export async function syncLibrary(): Promise<SyncLibraryResult> {
  const r = await fetch(apiUrl("/api/v1/sync"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({}),
  });
  const data: unknown = await r.json();
  if (!r.ok) {
    throw new Error(formatApiErrorBody(data));
  }
  return data as SyncLibraryResult;
}

export async function uploadFiles(files: FileList | File[]): Promise<unknown> {
  const fd = new FormData();
  const arr = Array.from(files);
  for (const f of arr) {
    fd.append("files", f);
  }
  const r = await fetch(apiUrl("/api/v1/upload"), {
    method: "POST",
    body: fd,
  });
  const data: unknown = await r.json();
  if (!r.ok) {
    throw new Error(formatApiErrorBody(data));
  }
  return data;
}

export function buildAssistantExtra(answer: ChatAnswer): Record<string, unknown> {
  const ex: Record<string, unknown> = { mode: answer.mode };
  if (answer.sources?.length) ex.sources = answer.sources;
  if (answer.web_snippets?.length) ex.web_sources = answer.web_snippets;
  if (answer.assistant_note) ex.status_note = answer.assistant_note;
  if (answer.validation_warning) ex.validation_warning = answer.validation_warning;
  if (answer.mode === "web") ex.web_only = true;
  if (answer.mode === "blended") ex.blended = true;
  return ex;
}

export type StreamOutcome = "complete" | "aborted" | "incomplete";

export async function postChatStream(
  message: string,
  taskMode: TaskMode,
  signal: AbortSignal,
  handlers: {
    onToken: (delta: string) => void;
    onDone: (answer: ChatAnswer) => void;
    onError: (detail: string) => void;
  },
): Promise<StreamOutcome> {
  let sawTerminal = false;
  const markDone = (answer: ChatAnswer) => {
    sawTerminal = true;
    handlers.onDone(answer);
  };
  const markError = (detail: string) => {
    sawTerminal = true;
    handlers.onError(detail);
  };

  let r: Response;
  try {
    r = await fetch(apiUrl("/api/v1/chat/stream"), {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify({ message, task_mode: taskMode, summarize_scope: "all" }),
      signal,
    });
  } catch (e) {
    if (signal.aborted) return "aborted";
    markError(e instanceof Error ? e.message : "Network error");
    return "complete";
  }

  if (signal.aborted) return "aborted";

  if (!r.ok) {
    markError(await readErrorBody(r));
    return "complete";
  }

  const reader = r.body?.getReader();
  if (!reader) {
    markError("No response body");
    return "complete";
  }

  const dec = new TextDecoder();
  let buf = "";

  const flushBlock = (block: string) => {
    let ev: string | undefined;
    const dataLines: string[] = [];
    for (const line of block.split("\n")) {
      if (line.startsWith("event:")) ev = line.slice(6).trim();
      else if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
    }
    const raw = dataLines.join("\n");
    if (!raw) return;
    let payload: Record<string, unknown>;
    try {
      payload = JSON.parse(raw) as Record<string, unknown>;
    } catch {
      return;
    }
    const typ = payload.type as string;
    if (ev === "token" || typ === "token") {
      const d = payload.delta as string | undefined;
      if (d) handlers.onToken(d);
      return;
    }
    if (ev === "done" || typ === "done") {
      const ans = payload.answer as ChatAnswer | undefined;
      if (ans) markDone(ans);
      return;
    }
    if (ev === "error" || typ === "error") {
      markError(String(payload.detail ?? "Unknown error"));
    }
  };

  try {
    for (;;) {
      if (signal.aborted) return "aborted";
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const parts = buf.split("\n\n");
      buf = parts.pop() ?? "";
      for (const block of parts) {
        if (block.trim()) flushBlock(block);
      }
    }
    if (buf.trim()) flushBlock(buf);
  } finally {
    reader.releaseLock();
  }

  if (signal.aborted) return "aborted";
  if (!sawTerminal) return "incomplete";
  return "complete";
}

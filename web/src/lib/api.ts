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

export async function listChats(limit = 40): Promise<ChatSessionRow[]> {
  const r = await fetch(apiUrl(`/api/v1/chats?limit=${limit}`), {
    cache: "no-store",
  });
  if (!r.ok) throw new Error(`List chats failed: ${r.status}`);
  return r.json();
}

export async function createChat(title = "New chat"): Promise<{ id: string; title: string }> {
  const r = await fetch(
    apiUrl(`/api/v1/chats?title=${encodeURIComponent(title)}`),
    { method: "POST" },
  );
  if (!r.ok) throw new Error(`Create chat failed: ${r.status}`);
  return r.json();
}

export async function deleteChat(sessionId: string): Promise<void> {
  const r = await fetch(apiUrl(`/api/v1/chats/${sessionId}`), {
    method: "DELETE",
  });
  if (!r.ok) throw new Error(`Delete chat failed: ${r.status}`);
}

export async function getMessages(sessionId: string): Promise<MessageOut[]> {
  const r = await fetch(apiUrl(`/api/v1/chats/${sessionId}/messages`), {
    cache: "no-store",
  });
  if (!r.ok) throw new Error(`Load messages failed: ${r.status}`);
  return r.json();
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
  if (!r.ok) throw new Error(`Save message failed: ${r.status}`);
}

export async function setChatTitle(sessionId: string, title: string): Promise<void> {
  const r = await fetch(apiUrl(`/api/v1/chats/${sessionId}`), {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title }),
  });
  if (!r.ok) throw new Error(`Rename chat failed: ${r.status}`);
}

export async function listDocuments(): Promise<{ documents: DocumentRow[]; count: number }> {
  const r = await fetch(apiUrl("/api/v1/documents"), { cache: "no-store" });
  if (!r.ok) throw new Error(`List documents failed: ${r.status}`);
  return r.json();
}

export async function syncLibrary(): Promise<{
  ok: boolean;
  status: string;
  message: string;
  sync_action: string;
  vector_count: number;
}> {
  const r = await fetch(apiUrl("/api/v1/sync"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({}),
  });
  const data = await r.json();
  if (!r.ok) {
    throw new Error(data?.message || data?.detail || `Sync failed: ${r.status}`);
  }
  return data;
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
  const data = await r.json();
  if (!r.ok) {
    throw new Error(data?.message || data?.detail || `Upload failed: ${r.status}`);
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

export async function postChatStream(
  message: string,
  taskMode: TaskMode,
  signal: AbortSignal,
  handlers: {
    onToken: (delta: string) => void;
    onDone: (answer: ChatAnswer) => void;
    onError: (detail: string) => void;
  },
): Promise<void> {
  const r = await fetch(apiUrl("/api/v1/chat/stream"), {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify({ message, task_mode: taskMode, summarize_scope: "all" }),
    signal,
  });

  if (!r.ok) {
    let t = `HTTP ${r.status}`;
    try {
      const j = await r.json();
      t = j.detail || j.message || t;
    } catch {
      /* ignore */
    }
    handlers.onError(t);
    return;
  }

  const reader = r.body?.getReader();
  if (!reader) {
    handlers.onError("No response body");
    return;
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
      if (ans) handlers.onDone(ans);
      return;
    }
    if (ev === "error" || typ === "error") {
      handlers.onError(String(payload.detail ?? "Unknown error"));
    }
  };

  try {
    for (;;) {
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
}

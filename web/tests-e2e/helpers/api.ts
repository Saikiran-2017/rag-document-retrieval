import type { APIRequestContext } from "@playwright/test";

const API_BASE = "http://127.0.0.1:8000";

export type HealthPayload = {
  status?: string;
  openai_key_configured?: boolean;
  openai_key_placeholder?: boolean;
  openai_key_source?: string;
  web_search_enabled?: boolean;
};

export async function fetchHealth(request: APIRequestContext): Promise<HealthPayload | null> {
  try {
    const r = await request.get(`${API_BASE}/health`, { timeout: 10_000 });
    if (!r.ok()) return null;
    return (await r.json()) as HealthPayload;
  } catch {
    return null;
  }
}

/** Remove every file from the E2E raw library (uses same API as the web app). */
export async function clearE2ELibrary(request: APIRequestContext): Promise<void> {
  const r = await request.get(`${API_BASE}/api/v1/documents`);
  if (!r.ok()) return;
  const data = (await r.json()) as { documents?: { filename: string }[] };
  const docs = data.documents ?? [];
  for (const d of docs) {
    await request.delete(
      `${API_BASE}/api/v1/documents?filename=${encodeURIComponent(d.filename)}`,
    );
  }
}

export function shouldRunLlmSuite(health: HealthPayload | null): boolean {
  if (!health || health.status !== "ok") return false;
  if (!health.openai_key_configured) return false;
  if (health.openai_key_placeholder) return false;
  return true;
}

/** Non-streaming chat (same routing as UI SSE). Reliable for E2E assertions vs dev streaming quirks. */
export async function postChatJson(
  request: APIRequestContext,
  message: string,
  taskMode: "auto" | "summarize" | "extract" | "compare" = "auto",
): Promise<{ text: string; mode: string; sources?: unknown[] }> {
  const r = await request.post(`${API_BASE}/api/v1/chat`, {
    headers: { "Content-Type": "application/json" },
    data: JSON.stringify({
      message,
      task_mode: taskMode,
      summarize_scope: "all",
    }),
  });
  if (!r.ok()) {
    throw new Error(`POST /chat failed: ${r.status()} ${await r.text()}`);
  }
  return r.json() as { text: string; mode: string; sources?: unknown[] };
}

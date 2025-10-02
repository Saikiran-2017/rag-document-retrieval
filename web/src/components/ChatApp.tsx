"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  appendMessage,
  buildAssistantExtra,
  createChat,
  deleteChat,
  getMessages,
  getApiBase,
  listChats,
  listDocuments,
  postChatStream,
  setChatTitle,
  syncLibrary,
  uploadFiles,
} from "@/lib/api";
import { mapMessages } from "@/lib/messages";
import type { ChatAnswer, ChatSessionRow, DocumentRow, TaskMode, UiMessage } from "@/lib/types";
import { ChatThread } from "@/components/ChatThread";
import { Composer } from "@/components/Composer";
import { EmptyHero } from "@/components/EmptyHero";
import { Sidebar } from "@/components/Sidebar";

const STORAGE_KEY = "ka_web_active_chat";

function newId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

function answerToMeta(ans: ChatAnswer) {
  return {
    mode: ans.mode,
    sources: ans.sources ?? undefined,
    web_sources: ans.web_snippets ?? undefined,
    status_note: ans.assistant_note ?? undefined,
    validation_warning: ans.validation_warning ?? undefined,
  };
}

export function ChatApp() {
  const [sessions, setSessions] = useState<ChatSessionRow[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<UiMessage[]>([]);
  const [documents, setDocuments] = useState<DocumentRow[]>([]);
  const [taskMode, setTaskMode] = useState<TaskMode>("auto");
  const [busy, setBusy] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [banner, setBanner] = useState<string | null>(null);
  const [uploadHint, setUploadHint] = useState<string | null>(null);
  const [syncHint, setSyncHint] = useState<string | null>(null);
  const [ready, setReady] = useState(false);

  const abortRef = useRef<AbortController | null>(null);

  const refreshDocuments = useCallback(async () => {
    try {
      const d = await listDocuments();
      setDocuments(d.documents);
    } catch {
      /* sidebar optional */
    }
  }, []);

  const refreshSessions = useCallback(async () => {
    const list = await listChats();
    setSessions(list);
    return list;
  }, []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        let list = await listChats();
        if (cancelled) return;
        if (list.length === 0) {
          await createChat();
          list = await listChats();
        }
        if (cancelled) return;
        setSessions(list);
        const stored = localStorage.getItem(STORAGE_KEY);
        const pick =
          stored && list.some((s) => s.id === stored) ? stored : list[0]?.id ?? null;
        setActiveSessionId(pick);
        if (pick) localStorage.setItem(STORAGE_KEY, pick);
      } catch (e) {
        setBanner(
          e instanceof Error
            ? e.message
            : "Could not reach the API. Start the backend and set NEXT_PUBLIC_API_URL.",
        );
      } finally {
        if (!cancelled) setReady(true);
      }
    })();
    refreshDocuments();
    return () => {
      cancelled = true;
    };
  }, [refreshDocuments]);

  useEffect(() => {
    if (!activeSessionId || !ready) return;
    let cancelled = false;
    abortRef.current?.abort();
    abortRef.current = null;
    (async () => {
      try {
        const rows = await getMessages(activeSessionId);
        if (cancelled) return;
        setMessages(mapMessages(rows));
      } catch {
        if (!cancelled) setMessages([]);
      }
    })();
    localStorage.setItem(STORAGE_KEY, activeSessionId);
    return () => {
      cancelled = true;
    };
  }, [activeSessionId, ready]);

  const selectChat = (id: string) => {
    abortRef.current?.abort();
    abortRef.current = null;
    setStreaming(false);
    setActiveSessionId(id);
  };

  const handleNewChat = async () => {
    setBanner(null);
    try {
      const c = await createChat();
      const list = await listChats();
      setSessions(list);
      setActiveSessionId(c.id);
      setMessages([]);
      localStorage.setItem(STORAGE_KEY, c.id);
    } catch (e) {
      setBanner(e instanceof Error ? e.message : "Could not create chat");
    }
  };

  const handleDeleteChat = async (id: string) => {
    try {
      await deleteChat(id);
      const list = await listChats();
      setSessions(list);
      if (activeSessionId === id) {
        const next = list[0]?.id ?? null;
        setActiveSessionId(next);
        if (!next) {
          const c = await createChat();
          const again = await listChats();
          setSessions(again);
          setActiveSessionId(c.id);
        }
      }
    } catch (e) {
      setBanner(e instanceof Error ? e.message : "Delete failed");
    }
  };

  const handleUpload = async (files: FileList | null) => {
    if (!files?.length) return;
    setUploadHint(null);
    setBanner(null);
    try {
      const data = (await uploadFiles(files)) as {
        message?: string;
        status?: string;
        saved_count?: number;
      };
      setUploadHint(data.message ?? "Upload finished.");
      await refreshDocuments();
    } catch (e) {
      setBanner(e instanceof Error ? e.message : "Upload failed");
    }
  };

  const handleSync = async () => {
    setSyncHint(null);
    setBanner(null);
    setBusy(true);
    try {
      const res = await syncLibrary();
      setSyncHint(res.message);
      await refreshDocuments();
    } catch (e) {
      setBanner(e instanceof Error ? e.message : "Sync failed");
    } finally {
      setBusy(false);
    }
  };

  const sendMessage = async (q: string) => {
    if (!activeSessionId || streaming) return;
    setBanner(null);
    const sid = activeSessionId;
    const wasEmpty = messages.length === 0;

    const userMsg: UiMessage = { id: newId(), role: "user", content: q };
    setMessages((prev) => [...prev, userMsg]);

    try {
      await appendMessage(sid, "user", q, {});
    } catch (e) {
      setBanner(e instanceof Error ? e.message : "Could not save message");
      return;
    }

    if (wasEmpty) {
      const title = q.trim().slice(0, 80);
      try {
        await setChatTitle(sid, title);
        setSessions((prev) =>
          prev.map((s) => (s.id === sid ? { ...s, title } : s)),
        );
      } catch {
        /* non-fatal */
      }
    }

    const assistId = newId();
    const assistantPlaceholder: UiMessage = {
      id: assistId,
      role: "assistant",
      content: "",
      streaming: true,
      meta: {},
    };
    setMessages((prev) => [...prev, assistantPlaceholder]);
    setStreaming(true);

    const ac = new AbortController();
    abortRef.current = ac;

    await postChatStream(q, taskMode, ac.signal, {
      onToken: (delta) => {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistId && m.role === "assistant"
              ? { ...m, content: m.content + delta }
              : m,
          ),
        );
      },
      onDone: async (answer: ChatAnswer) => {
        const text = answer.text || answer.error || "";
        const meta = answerToMeta(answer);
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistId && m.role === "assistant"
              ? { ...m, content: text, streaming: false, meta }
              : m,
          ),
        );
        try {
          await appendMessage(sid, "assistant", text, buildAssistantExtra(answer));
        } catch {
          /* persisted in UI */
        }
        setStreaming(false);
        abortRef.current = null;
        await refreshSessions();
      },
      onError: async (detail) => {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistId && m.role === "assistant"
              ? {
                  ...m,
                  content: detail,
                  streaming: false,
                  meta: { mode: "error" },
                }
              : m,
          ),
        );
        try {
          await appendMessage(sid, "assistant", detail, { mode: "error" });
        } catch {
          /* ignore */
        }
        setStreaming(false);
        abortRef.current = null;
      },
    });
  };

  if (!ready) {
    return (
      <div className="flex h-screen items-center justify-center bg-[var(--ka-bg)] text-sm text-stone-500">
        Loading…
      </div>
    );
  }

  return (
    <div className="flex h-screen overflow-hidden bg-[var(--ka-bg)] text-stone-900">
      <Sidebar
        sessions={sessions}
        activeId={activeSessionId}
        documents={documents}
        busy={busy || streaming}
        onSelectChat={selectChat}
        onNewChat={handleNewChat}
        onDeleteChat={handleDeleteChat}
        onUpload={handleUpload}
        onSync={handleSync}
        uploadHint={uploadHint}
        syncHint={syncHint}
      />
      <main className="flex min-w-0 flex-1 flex-col">
        {banner ? (
          <div className="border-b border-amber-200 bg-amber-50 px-4 py-2 text-center text-sm text-amber-950">
            {banner}
            <span className="ml-2 text-xs text-stone-500">({getApiBase()})</span>
          </div>
        ) : null}
        <div className="flex min-h-0 flex-1 flex-col">
          {messages.length === 0 ? (
            <EmptyHero onPickPrompt={sendMessage} disabled={streaming || !activeSessionId} />
          ) : (
            <ChatThread messages={messages} />
          )}
        </div>
        <Composer
          onSend={sendMessage}
          disabled={streaming || busy || !activeSessionId}
          taskMode={taskMode}
          onTaskModeChange={setTaskMode}
        />
      </main>
    </div>
  );
}

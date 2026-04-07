"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  appendMessage,
  buildAssistantExtra,
  checkApiHealth,
  createChat,
  deleteChat,
  getApiBase,
  getMessages,
  listChats,
  deleteDocument,
  listDocuments,
  postChatStream,
  setChatTitle,
  syncLibrary,
  uploadFiles,
} from "@/lib/api";
import { loadInitialChatSessions } from "@/lib/bootstrap";
import { mapMessages } from "@/lib/messages";
import type { ChatAnswer, ChatSessionRow, DocumentRow, TaskMode, UiMessage } from "@/lib/types";
import { AppLoadingShell } from "@/components/AppLoadingShell";
import { ChatSkeleton } from "@/components/ChatSkeleton";
import { ChatThread } from "@/components/ChatThread";
import { Composer } from "@/components/Composer";
import { EmptyHero } from "@/components/EmptyHero";
import { MobileNavBar, Sidebar } from "@/components/Sidebar";

const STORAGE_KEY = "ka_web_active_chat";
const SKELETON_DELAY_MS = 220;

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
  const [libraryNeedsSync, setLibraryNeedsSync] = useState(false);
  const [ready, setReady] = useState(false);
  const [threadLoading, setThreadLoading] = useState(false);
  const [showThreadSkeleton, setShowThreadSkeleton] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);

  const abortRef = useRef<AbortController | null>(null);

  const refreshDocuments = useCallback(async () => {
    try {
      const d = await listDocuments();
      setDocuments(d.documents);
      setLibraryNeedsSync(Boolean(d.library_needs_sync));
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
        const reachable = await checkApiHealth();
        if (cancelled) return;
        if (!reachable) {
          setBanner(
            `API not reachable at ${getApiBase()}. Start the backend (uvicorn) and set NEXT_PUBLIC_API_URL if needed.`,
          );
          return;
        }
        const list = await loadInitialChatSessions();
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
            : "Could not load chats. Check the backend and NEXT_PUBLIC_API_URL.",
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
    setMessages([]);
    setThreadLoading(true);

    (async () => {
      try {
        const rows = await getMessages(activeSessionId);
        if (cancelled) return;
        setMessages(mapMessages(rows));
      } catch {
        if (!cancelled) setMessages([]);
      } finally {
        if (!cancelled) setThreadLoading(false);
      }
    })();
    localStorage.setItem(STORAGE_KEY, activeSessionId);
    return () => {
      cancelled = true;
    };
  }, [activeSessionId, ready]);

  useEffect(() => {
    if (!threadLoading) {
      setShowThreadSkeleton(false);
      return;
    }
    const t = window.setTimeout(() => setShowThreadSkeleton(true), SKELETON_DELAY_MS);
    return () => window.clearTimeout(t);
  }, [threadLoading]);

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
      if (!window.confirm("Delete this chat? This does not delete your uploaded documents.")) {
        return;
      }
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

  const handleClearConversation = async () => {
    if (!activeSessionId) return;
    setBanner(null);
    try {
      if (
        !window.confirm(
          "Clear this conversation? This starts a new chat and keeps your uploaded documents.",
        )
      ) {
        return;
      }
      const old = activeSessionId;
      await deleteChat(old);
      const c = await createChat();
      const list = await listChats();
      setSessions(list);
      setActiveSessionId(c.id);
      setMessages([]);
      localStorage.setItem(STORAGE_KEY, c.id);
    } catch (e) {
      setBanner(e instanceof Error ? e.message : "Could not clear conversation");
    }
  };

  const handleStop = () => {
    abortRef.current?.abort();
    abortRef.current = null;
    setStreaming(false);
    setMessages((prev) =>
      prev.map((m) => {
        if (m.role !== "assistant" || !m.streaming) return m;
        const c = m.content.trim();
        return {
          ...m,
          streaming: false,
          content: c || "Stopped.",
          meta: { ...(m.meta ?? {}), status_note: "Stopped by you." },
        };
      }),
    );
  };

  const handleUpload = async (files: FileList | null) => {
    if (!files?.length) return;
    setUploadHint(null);
    setBanner(null);
    setBusy(true);
    try {
      const t0 = performance.now();
      const data = await uploadFiles(files);
      const dt = performance.now() - t0;
      const extra =
        data?.diagnostics && typeof data.diagnostics === "object" && "timing_ms_total" in data.diagnostics
          ? ` (server ${String((data.diagnostics as any).timing_ms_total)} ms)`
          : ` (client ${Math.round(dt)} ms)`;
      setUploadHint((data.message ?? "Upload finished. Syncing your library…") + extra);
      await refreshDocuments();
      const t1 = performance.now();
      const res = await syncLibrary();
      const dt2 = performance.now() - t1;
      const extra2 =
        res?.diagnostics && typeof res.diagnostics === "object" && "timing_ms_total" in res.diagnostics
          ? ` (server ${String((res.diagnostics as any).timing_ms_total)} ms)`
          : ` (client ${Math.round(dt2)} ms)`;
      setSyncHint(res.message + extra2);
      await refreshDocuments();
    } catch (e) {
      setBanner(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setBusy(false);
    }
  };

  const handleSync = async () => {
    setSyncHint(null);
    setBanner(null);
    setBusy(true);
    try {
      const t0 = performance.now();
      const res = await syncLibrary();
      const dt = performance.now() - t0;
      const extra =
        res?.diagnostics && typeof res.diagnostics === "object" && "timing_ms_total" in res.diagnostics
          ? ` (server ${String((res.diagnostics as any).timing_ms_total)} ms)`
          : ` (client ${Math.round(dt)} ms)`;
      setSyncHint(res.message + extra);
      await refreshDocuments();
    } catch (e) {
      setBanner(e instanceof Error ? e.message : "Sync failed");
    } finally {
      setBusy(false);
    }
  };

  const handleDeleteDocument = async (filename: string) => {
    setBanner(null);
    setBusy(true);
    try {
      if (!window.confirm(`Remove "${filename}" from your library?`)) {
        return;
      }
      const res = await deleteDocument(filename);
      setSyncHint(res.message ?? "Removed from library.");
      await refreshDocuments();
    } catch (e) {
      setBanner(e instanceof Error ? e.message : "Could not remove file");
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
      setMessages((prev) => prev.filter((m) => m.id !== userMsg.id));
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

    const outcome = await postChatStream(q, taskMode, ac.signal, {
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

    if (outcome === "incomplete") {
      setStreaming(false);
      abortRef.current = null;
      setMessages((prev) =>
        prev.map((m) => {
          if (m.id !== assistId || m.role !== "assistant") return m;
          const c = m.content.trim();
          return {
            ...m,
            streaming: false,
            content: c || "The reply did not finish. Try sending again.",
          };
        }),
      );
      setBanner("The reply did not finish. Check your connection and try again.");
    }
  };

  if (!ready) {
    return <AppLoadingShell />;
  }

  const mainBody = threadLoading ? (
    showThreadSkeleton ? (
      <ChatSkeleton />
    ) : (
      <div
        className="flex flex-1 flex-col items-center justify-center gap-3 py-24"
        role="status"
        aria-live="polite"
      >
        <div className="ka-shimmer h-1.5 w-24 rounded-full opacity-70" />
        <p className="text-xs text-stone-400">Loading conversation…</p>
      </div>
    )
  ) : messages.length === 0 ? (
    <EmptyHero onPickPrompt={sendMessage} disabled={streaming || !activeSessionId} />
  ) : (
    <ChatThread messages={messages} />
  );

  return (
    <>
      {menuOpen ? (
        <button
          type="button"
          className="fixed inset-0 z-40 bg-stone-900/20 backdrop-blur-[1px] lg:hidden"
          aria-label="Close menu"
          onClick={() => setMenuOpen(false)}
        />
      ) : null}
      <MobileNavBar menuOpen={menuOpen} onToggleMenu={() => setMenuOpen((o) => !o)} />
      <div className="flex h-[100dvh] overflow-hidden bg-[var(--ka-bg)] text-stone-900">
        <Sidebar
          sessions={sessions}
          activeId={activeSessionId}
          documents={documents}
          busy={busy || streaming}
          mobileOpen={menuOpen}
          onMobileClose={() => setMenuOpen(false)}
          onSelectChat={selectChat}
          onNewChat={handleNewChat}
          onDeleteChat={handleDeleteChat}
          onUpload={handleUpload}
          onSync={handleSync}
          onDeleteDocument={handleDeleteDocument}
          uploadHint={uploadHint}
          syncHint={syncHint}
        />
        <main className="flex min-w-0 flex-1 flex-col pt-12 lg:pt-0">
          {libraryNeedsSync && documents.length > 0 ? (
            <div
              className="border-b border-sky-200/90 bg-sky-50/95 px-4 py-2.5 text-center text-sm leading-snug text-sky-950 backdrop-blur-sm"
              role="status"
            >
              Your library is not indexed for search yet, or it changed since the last sync. Use{" "}
              <strong>Sync documents</strong> in the sidebar before relying on answers from your files.
            </div>
          ) : null}
          {banner ? (
            <div className="border-b border-amber-200/90 bg-amber-50/95 px-4 py-2.5 text-center text-sm leading-snug text-amber-950 backdrop-blur-sm">
              <span>{banner}</span>
              <span className="mt-0.5 block text-xs font-normal text-stone-500 lg:inline lg:mt-0 lg:ml-2">
                ({getApiBase()})
              </span>
            </div>
          ) : null}
          <div className="flex min-h-0 flex-1 flex-col">{mainBody}</div>
          <Composer
            onSend={sendMessage}
            onStop={handleStop}
            onNewChat={handleNewChat}
            onClearConversation={handleClearConversation}
            disabled={streaming || busy || !activeSessionId}
            isStreaming={streaming}
            taskMode={taskMode}
            onTaskModeChange={setTaskMode}
            readinessHint={
              libraryNeedsSync && documents.length > 0
                ? "Until you sync, the assistant may answer generally and note that your file index is stale."
                : null
            }
          />
        </main>
      </div>
    </>
  );
}

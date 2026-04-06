"use client";

import { useRef } from "react";
import {
  APP_NAME,
  HEALTH_LABEL,
  SIDEBAR_CAPTION,
} from "@/lib/constants";
import type { ChatSessionRow, DocumentRow } from "@/lib/types";

function MenuIcon({ open }: { open: boolean }) {
  return (
    <svg
      className="h-5 w-5"
      fill="none"
      viewBox="0 0 24 24"
      stroke="currentColor"
      strokeWidth={1.75}
      aria-hidden
    >
      {open ? (
        <>
          <path strokeLinecap="round" d="M6 6l12 12" />
          <path strokeLinecap="round" d="M18 6L6 18" />
        </>
      ) : (
        <>
          <path strokeLinecap="round" d="M5 7h14M5 12h14M5 17h14" />
        </>
      )}
    </svg>
  );
}

function healthPillClass(health: string): string {
  if (health === "ready") return "bg-emerald-50 text-emerald-700 ring-emerald-200/80";
  if (health === "ready_limited") return "bg-amber-50 text-amber-800 ring-amber-200/80";
  if (health === "processing") return "bg-stone-100 text-stone-600 ring-stone-200/80";
  if (health === "failed") return "bg-rose-50 text-rose-700 ring-rose-200/80";
  return "bg-stone-100 text-stone-600 ring-stone-200/80";
}

export function MobileNavBar({
  menuOpen,
  onToggleMenu,
}: {
  menuOpen: boolean;
  onToggleMenu: () => void;
}) {
  return (
    <header className="fixed left-0 right-0 top-0 z-30 flex h-12 items-center gap-2 border-b border-stone-200/80 bg-[var(--ka-surface)]/90 px-3 backdrop-blur-md supports-[backdrop-filter]:bg-[var(--ka-surface)]/80 lg:hidden">
      <button
        type="button"
        onClick={onToggleMenu}
        className="rounded-lg p-2 text-stone-600 transition hover:bg-stone-100 hover:text-stone-900 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-stone-400"
        aria-controls="app-sidebar"
      >
        <span className="sr-only">{menuOpen ? "Close menu" : "Open menu"}</span>
        <MenuIcon open={menuOpen} />
      </button>
      <span className="min-w-0 truncate text-sm font-semibold tracking-tight text-stone-900">
        {APP_NAME}
      </span>
    </header>
  );
}

export function Sidebar({
  sessions,
  activeId,
  documents,
  busy,
  mobileOpen,
  onMobileClose,
  onSelectChat,
  onNewChat,
  onDeleteChat,
  onUpload,
  onSync,
  onDeleteDocument,
  uploadHint,
  syncHint,
}: {
  sessions: ChatSessionRow[];
  activeId: string | null;
  documents: DocumentRow[];
  busy: boolean;
  mobileOpen: boolean;
  onMobileClose: () => void;
  onSelectChat: (id: string) => void;
  onNewChat: () => void;
  onDeleteChat: (id: string) => void;
  onUpload: (files: FileList | null) => void;
  onSync: () => void;
  onDeleteDocument: (filename: string) => void;
  uploadHint?: string | null;
  syncHint?: string | null;
}) {
  const fileRef = useRef<HTMLInputElement>(null);

  const closeIfMobile = () => {
    onMobileClose();
  };

  return (
    <aside
      id="app-sidebar"
      className={`flex h-full w-[min(280px,100vw)] shrink-0 flex-col border-stone-200 bg-[var(--ka-elevated)] max-lg:fixed max-lg:left-0 max-lg:top-0 max-lg:z-50 max-lg:border-r max-lg:shadow-lg max-lg:transition-transform max-lg:duration-200 max-lg:ease-out lg:border-r ${
        mobileOpen
          ? "max-lg:translate-x-0 max-lg:pointer-events-auto"
          : "max-lg:pointer-events-none max-lg:-translate-x-full"
      } lg:pointer-events-auto lg:translate-x-0`}
    >
      <div className="border-b border-stone-200/80 px-4 py-4 lg:block">
        <h1 className="text-sm font-semibold tracking-tight text-stone-900">{APP_NAME}</h1>
        <p className="mt-1 text-xs leading-relaxed text-stone-500">{SIDEBAR_CAPTION}</p>
      </div>

      <div className="flex flex-1 flex-col overflow-y-auto overscroll-contain px-3 py-4">
        <button
          type="button"
          onClick={() => {
            onNewChat();
            closeIfMobile();
          }}
          disabled={busy}
          className="mb-5 w-full rounded-xl border border-stone-300/90 bg-white py-2.5 text-sm font-medium text-stone-800 shadow-sm transition hover:border-stone-400 hover:bg-stone-50 disabled:opacity-50"
        >
          New chat
        </button>

        <p className="mb-2 px-1 text-[10px] font-semibold uppercase tracking-[0.14em] text-stone-400">
          Chats
        </p>
        <ul className="space-y-0.5">
          {sessions.map((s) => (
            <li key={s.id} className="group flex items-center gap-0.5">
              <button
                type="button"
                onClick={() => {
                  onSelectChat(s.id);
                  closeIfMobile();
                }}
                disabled={busy}
                className={`min-w-0 flex-1 truncate rounded-lg px-2.5 py-2 text-left text-sm transition ${
                  s.id === activeId
                    ? "bg-white font-medium text-stone-900 shadow-sm ring-1 ring-stone-200/90"
                    : "text-stone-600 hover:bg-stone-100/90"
                }`}
              >
                {s.title || "Chat"}
              </button>
              <button
                type="button"
                title="Delete chat"
                onClick={() => onDeleteChat(s.id)}
                disabled={busy}
                className="rounded-md p-1.5 text-stone-400 opacity-0 transition hover:bg-stone-200/80 hover:text-stone-700 group-hover:opacity-100 max-lg:opacity-100"
              >
                <span className="sr-only">Delete</span>×
              </button>
            </li>
          ))}
        </ul>

        <div className="my-6 border-t border-stone-200/80" />

        <p className="mb-1 px-1 text-[10px] font-semibold uppercase tracking-[0.14em] text-stone-400">
          Library
        </p>
        <p className="mb-3 px-1 text-[11px] leading-snug text-stone-500">
          PDF, Word, or text files. Upload, then sync to index.
        </p>
        <input
          ref={fileRef}
          type="file"
          multiple
          accept=".pdf,.docx,.txt,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document,text/plain"
          className="hidden"
          aria-label="Choose documents to upload"
          onChange={(e) => {
            onUpload(e.target.files);
            e.target.value = "";
          }}
        />
        <button
          type="button"
          disabled={busy}
          onClick={() => fileRef.current?.click()}
          className="mb-2 w-full rounded-xl border border-stone-200 bg-white py-2.5 text-sm text-stone-700 shadow-sm transition hover:border-stone-300 hover:bg-stone-50 disabled:opacity-50"
        >
          Upload documents
        </button>
        {uploadHint ? (
          <p className="mb-3 px-1 text-xs leading-relaxed text-stone-500">{uploadHint}</p>
        ) : null}

        <button
          type="button"
          disabled={busy}
          onClick={onSync}
          className="mb-2 w-full rounded-xl bg-stone-800 py-2.5 text-sm font-medium text-white shadow-sm transition hover:bg-stone-900 disabled:opacity-50"
        >
          {busy ? "Working…" : "Sync documents"}
        </button>
        {syncHint ? (
          <p className="mb-5 px-1 text-xs leading-relaxed text-stone-500">{syncHint}</p>
        ) : null}

        <p className="mb-2 px-1 text-[10px] font-semibold uppercase tracking-[0.14em] text-stone-400">
          Files in library
        </p>
        {documents.length === 0 ? (
          <p className="px-1 text-xs leading-relaxed text-stone-500">
            No files yet. Upload, then Sync.
          </p>
        ) : (
          <ul className="space-y-2">
            {documents.map((d) => (
              <li
                key={d.filename}
                className="rounded-lg border border-stone-200/80 bg-white px-2.5 py-2 text-xs shadow-sm shadow-stone-900/[0.02]"
              >
                <div className="flex items-start gap-1">
                  <div className="min-w-0 flex-1 truncate font-medium text-stone-800" title={d.filename}>
                    {d.filename}
                  </div>
                  <span
                    className={`shrink-0 rounded-full px-2 py-0.5 text-[10px] font-semibold ring-1 ${healthPillClass(
                      d.health,
                    )}`}
                    title={HEALTH_LABEL[d.health] ?? d.health}
                  >
                    {HEALTH_LABEL[d.health] ?? d.health}
                  </span>
                  <button
                    type="button"
                    title="Remove from library"
                    disabled={busy}
                    onClick={() => onDeleteDocument(d.filename)}
                    className="shrink-0 rounded px-1 text-stone-400 transition hover:bg-stone-100 hover:text-stone-700 disabled:opacity-40"
                  >
                    <span className="sr-only">Remove {d.filename}</span>×
                  </button>
                </div>
                {d.note ? (
                  <div className="mt-1.5 text-[11px] leading-snug text-amber-900/90">{d.note}</div>
                ) : null}
              </li>
            ))}
          </ul>
        )}
      </div>
    </aside>
  );
}

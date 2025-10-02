"use client";

import { useRef } from "react";
import {
  APP_NAME,
  HEALTH_LABEL,
  SIDEBAR_CAPTION,
} from "@/lib/constants";
import type { ChatSessionRow, DocumentRow } from "@/lib/types";

export function Sidebar({
  sessions,
  activeId,
  documents,
  busy,
  onSelectChat,
  onNewChat,
  onDeleteChat,
  onUpload,
  onSync,
  uploadHint,
  syncHint,
}: {
  sessions: ChatSessionRow[];
  activeId: string | null;
  documents: DocumentRow[];
  busy: boolean;
  onSelectChat: (id: string) => void;
  onNewChat: () => void;
  onDeleteChat: (id: string) => void;
  onUpload: (files: FileList | null) => void;
  onSync: () => void;
  uploadHint?: string | null;
  syncHint?: string | null;
}) {
  const fileRef = useRef<HTMLInputElement>(null);

  return (
    <aside className="flex h-full w-[280px] shrink-0 flex-col border-r border-stone-200 bg-stone-50/90">
      <div className="border-b border-stone-200/80 px-4 py-4">
        <h1 className="text-sm font-semibold tracking-tight text-stone-900">{APP_NAME}</h1>
        <p className="mt-1 text-xs text-stone-500">{SIDEBAR_CAPTION}</p>
      </div>

      <div className="flex-1 overflow-y-auto px-3 py-3">
        <button
          type="button"
          onClick={onNewChat}
          disabled={busy}
          className="mb-4 w-full rounded-lg border border-stone-300 bg-white py-2.5 text-sm font-medium text-stone-800 shadow-sm transition hover:bg-stone-50 disabled:opacity-50"
        >
          New chat
        </button>

        <p className="mb-2 px-1 text-[11px] font-semibold uppercase tracking-wider text-stone-400">
          Chats
        </p>
        <ul className="space-y-0.5">
          {sessions.map((s) => (
            <li key={s.id} className="group flex items-center gap-1">
              <button
                type="button"
                onClick={() => onSelectChat(s.id)}
                disabled={busy}
                className={`min-w-0 flex-1 truncate rounded-md px-2 py-2 text-left text-sm transition ${
                  s.id === activeId
                    ? "bg-white font-medium text-stone-900 shadow-sm ring-1 ring-stone-200"
                    : "text-stone-600 hover:bg-stone-100/80"
                }`}
              >
                {s.title || "Chat"}
              </button>
              <button
                type="button"
                title="Delete chat"
                onClick={() => onDeleteChat(s.id)}
                disabled={busy}
                className="rounded p-1 text-stone-400 opacity-0 transition hover:bg-stone-200 hover:text-stone-700 group-hover:opacity-100"
              >
                ×
              </button>
            </li>
          ))}
        </ul>

        <div className="my-5 border-t border-stone-200/80" />

        <p className="mb-2 px-1 text-[11px] font-semibold uppercase tracking-wider text-stone-400">
          Library
        </p>
        <input
          ref={fileRef}
          type="file"
          multiple
          accept=".pdf,.docx,.txt,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document,text/plain"
          className="hidden"
          onChange={(e) => {
            onUpload(e.target.files);
            e.target.value = "";
          }}
        />
        <button
          type="button"
          disabled={busy}
          onClick={() => fileRef.current?.click()}
          className="mb-2 w-full rounded-lg border border-stone-200 bg-white py-2 text-sm text-stone-700 hover:bg-stone-50 disabled:opacity-50"
        >
          Upload documents
        </button>
        {uploadHint ? (
          <p className="mb-2 px-1 text-xs text-stone-500">{uploadHint}</p>
        ) : null}

        <button
          type="button"
          disabled={busy}
          onClick={onSync}
          className="mb-4 w-full rounded-lg bg-stone-800 py-2 text-sm font-medium text-white hover:bg-stone-900 disabled:opacity-50"
        >
          Sync documents
        </button>
        {syncHint ? <p className="mb-3 px-1 text-xs text-stone-500">{syncHint}</p> : null}

        <p className="mb-2 px-1 text-[11px] font-semibold uppercase tracking-wider text-stone-400">
          Files
        </p>
        {documents.length === 0 ? (
          <p className="px-1 text-xs text-stone-500">No files yet. Upload, then Sync.</p>
        ) : (
          <ul className="space-y-1.5">
            {documents.map((d) => (
              <li
                key={d.filename}
                className="rounded-md border border-stone-100 bg-white/80 px-2 py-1.5 text-xs"
              >
                <div className="truncate font-medium text-stone-800" title={d.filename}>
                  {d.filename}
                </div>
                <div className="mt-0.5 text-stone-500">
                  {HEALTH_LABEL[d.health] ?? d.health}
                </div>
                {d.note ? (
                  <div className="mt-1 text-[11px] leading-snug text-amber-800/90">{d.note}</div>
                ) : null}
              </li>
            ))}
          </ul>
        )}
      </div>
    </aside>
  );
}

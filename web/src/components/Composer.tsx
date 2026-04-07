"use client";

import { useState } from "react";
import { TaskModeSelect } from "@/components/ModeBadge";
import type { TaskMode } from "@/lib/types";

export function Composer({
  onSend,
  onStop,
  onNewChat,
  onClearConversation,
  disabled,
  isStreaming,
  taskMode,
  onTaskModeChange,
  readinessHint,
}: {
  onSend: (text: string) => void;
  onStop?: () => void;
  onNewChat?: () => void;
  onClearConversation?: () => void;
  disabled?: boolean;
  isStreaming?: boolean;
  taskMode: TaskMode;
  onTaskModeChange: (m: TaskMode) => void;
  readinessHint?: string | null;
}) {
  const [text, setText] = useState("");

  const submit = () => {
    const q = text.trim();
    if (!q || disabled) return;
    setText("");
    onSend(q);
  };

  return (
    <div className="border-t border-stone-200/90 bg-[var(--ka-surface)]/95 px-4 py-3 backdrop-blur-sm md:px-8 md:py-4 pb-[max(0.75rem,env(safe-area-inset-bottom))]">
      <div className="mx-auto flex max-w-3xl flex-col gap-2">
        <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-stone-500">
          <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
          <span className="font-medium text-stone-600">Mode</span>
          <TaskModeSelect value={taskMode} onChange={onTaskModeChange} disabled={disabled} />
          <span className="hidden text-stone-500 sm:inline md:max-w-[28rem] md:leading-snug">
            Auto: chat and Q&amp;A. Other modes target your synced library.
          </span>
          </div>
          <div className="flex items-center gap-1.5">
            {onNewChat ? (
              <button
                type="button"
                onClick={onNewChat}
                disabled={disabled}
                className="rounded-lg border border-stone-200 bg-white px-2.5 py-1.5 text-[11px] font-medium text-stone-700 shadow-sm transition hover:bg-stone-50 disabled:opacity-50"
              >
                New chat
              </button>
            ) : null}
            {onClearConversation ? (
              <button
                type="button"
                onClick={onClearConversation}
                disabled={disabled}
                className="rounded-lg border border-stone-200 bg-white px-2.5 py-1.5 text-[11px] font-medium text-stone-700 shadow-sm transition hover:bg-stone-50 disabled:opacity-50"
              >
                Clear
              </button>
            ) : null}
          </div>
        </div>
        {readinessHint ? (
          <p className="text-xs leading-snug text-sky-900/90">{readinessHint}</p>
        ) : null}
        {disabled && isStreaming ? (
          <p className="text-xs text-stone-500">Waiting for the reply…</p>
        ) : null}
        <div className="flex gap-2 rounded-2xl border border-stone-200/90 bg-white p-2 shadow-sm shadow-stone-900/[0.04] transition focus-within:border-stone-300 focus-within:shadow-md focus-within:shadow-stone-900/[0.06] focus-within:ring-1 focus-within:ring-stone-200">
          <textarea
            rows={1}
            value={text}
            disabled={disabled}
            placeholder="Message…"
            className="max-h-40 min-h-[48px] flex-1 resize-none bg-transparent px-3 py-2.5 text-[15px] leading-normal text-stone-800 outline-none placeholder:text-stone-400 disabled:opacity-60"
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                submit();
              }
            }}
          />
          <button
            type="button"
            disabled={disabled ? true : isStreaming ? false : !text.trim()}
            onClick={() => {
              if (isStreaming && onStop) onStop();
              else submit();
            }}
            className="self-end rounded-xl bg-stone-800 px-4 py-2.5 text-sm font-medium text-white shadow-sm transition hover:bg-stone-900 disabled:pointer-events-none disabled:opacity-35"
          >
            {isStreaming ? "Stop" : "Send"}
          </button>
        </div>
      </div>
    </div>
  );
}

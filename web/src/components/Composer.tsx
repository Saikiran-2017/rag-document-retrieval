"use client";

import { useState } from "react";
import { TaskModeSelect } from "@/components/ModeBadge";
import type { TaskMode } from "@/lib/types";

export function Composer({
  onSend,
  disabled,
  taskMode,
  onTaskModeChange,
}: {
  onSend: (text: string) => void;
  disabled?: boolean;
  taskMode: TaskMode;
  onTaskModeChange: (m: TaskMode) => void;
}) {
  const [text, setText] = useState("");

  const submit = () => {
    const q = text.trim();
    if (!q || disabled) return;
    setText("");
    onSend(q);
  };

  return (
    <div className="border-t border-stone-200 bg-[var(--ka-surface)] px-4 py-3 md:px-8">
      <div className="mx-auto flex max-w-3xl flex-col gap-2">
        <div className="flex items-center gap-2 text-xs text-stone-500">
          <span>Mode</span>
          <TaskModeSelect value={taskMode} onChange={onTaskModeChange} disabled={disabled} />
          <span className="hidden sm:inline">
            Auto: normal chat and Q&A. Other modes use a focused prompt on your synced library.
          </span>
        </div>
        <div className="flex gap-2 rounded-xl border border-stone-200 bg-white p-2 shadow-sm focus-within:ring-1 focus-within:ring-stone-300">
          <textarea
            rows={1}
            value={text}
            disabled={disabled}
            placeholder="Message…"
            className="max-h-40 min-h-[44px] flex-1 resize-none bg-transparent px-2 py-2 text-[15px] text-stone-800 outline-none placeholder:text-stone-400"
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
            disabled={disabled || !text.trim()}
            onClick={submit}
            className="self-end rounded-lg bg-stone-800 px-4 py-2 text-sm font-medium text-white hover:bg-stone-900 disabled:opacity-40"
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
}

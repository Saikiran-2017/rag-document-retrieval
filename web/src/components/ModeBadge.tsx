"use client";

import type { TaskMode } from "@/lib/types";

export function modeChipLabel(mode: string | undefined): string | null {
  if (mode === "web") return "Web";
  if (mode === "blended") return "Documents + web";
  if (mode === "grounded") return "Documents";
  return null;
}

export function ModeBadge({ mode }: { mode: string | undefined }) {
  const label = modeChipLabel(mode);
  if (!label) return null;
  return (
    <span className="inline-flex items-center rounded-full border border-stone-200/90 bg-white px-2.5 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-stone-500 shadow-sm shadow-stone-900/[0.03]">
      {label}
    </span>
  );
}

export function TaskModeSelect({
  value,
  onChange,
  disabled,
}: {
  value: TaskMode;
  onChange: (v: TaskMode) => void;
  disabled?: boolean;
}) {
  return (
    <select
      value={value}
      disabled={disabled}
      onChange={(e) => onChange(e.target.value as TaskMode)}
      className="rounded-lg border border-stone-200 bg-white px-2.5 py-1.5 text-xs text-stone-700 shadow-sm outline-none transition hover:border-stone-300 focus-visible:ring-2 focus-visible:ring-stone-400/40 disabled:opacity-50"
      aria-label="Task mode"
    >
      <option value="auto">Auto</option>
      <option value="summarize">Summarize</option>
      <option value="extract">Extract</option>
      <option value="compare">Compare</option>
    </select>
  );
}

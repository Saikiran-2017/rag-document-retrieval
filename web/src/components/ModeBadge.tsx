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
    <span className="inline-flex items-center rounded-md border border-stone-200 bg-stone-50 px-2 py-0.5 text-[11px] font-medium tracking-wide text-stone-600">
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
      className="rounded-md border border-stone-200 bg-white px-2 py-1 text-xs text-stone-700 outline-none focus:ring-1 focus:ring-stone-400 disabled:opacity-50"
      aria-label="Task mode"
    >
      <option value="auto">Auto</option>
      <option value="summarize">Summarize</option>
      <option value="extract">Extract</option>
      <option value="compare">Compare</option>
    </select>
  );
}

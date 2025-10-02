"use client";

import type { SourceRef } from "@/lib/types";

export function SourcesPanel({ sources }: { sources: SourceRef[] }) {
  if (!sources.length) return null;
  return (
    <div className="mt-4 border-t border-stone-200/80 pt-4">
      <h3 className="mb-2 text-xs font-semibold uppercase tracking-wider text-stone-500">
        Sources
      </h3>
      <ul className="space-y-2 text-sm text-stone-700">
        {sources.map((s, i) => (
          <li
            key={`${s.chunk_id}-${i}`}
            className="rounded-lg border border-stone-100 bg-stone-50/80 px-3 py-2"
          >
            <span className="font-medium text-stone-800">
              [{s.source_number}] {s.source_name}
            </span>
            {s.page_label ? (
              <span className="ml-2 text-stone-500">· {s.page_label}</span>
            ) : null}
          </li>
        ))}
      </ul>
    </div>
  );
}

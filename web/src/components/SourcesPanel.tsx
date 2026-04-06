"use client";

import type { SourceRef } from "@/lib/types";

function truncatePath(path: string, max = 42): string {
  if (path.length <= max) return path;
  return `…${path.slice(-(max - 1))}`;
}

export function SourcesPanel({ sources }: { sources: SourceRef[] }) {
  if (!sources.length) return null;
  return (
    <div className="mt-6 border-t border-stone-200/90 pt-5">
      <h3 className="mb-3 text-[11px] font-semibold uppercase tracking-[0.12em] text-stone-400">
        Sources referenced
      </h3>
      <ol className="list-none space-y-2.5">
        {sources.map((s, i) => (
          <li
            key={`${s.chunk_id}-${i}`}
            className="rounded-xl border border-stone-200/80 bg-white px-3.5 py-3 shadow-sm shadow-stone-900/[0.02]"
          >
            <div className="flex flex-wrap items-baseline gap-x-2 gap-y-0.5">
              <span className="inline-flex h-5 min-w-[1.25rem] items-center justify-center rounded bg-stone-100 px-1 text-[11px] font-semibold tabular-nums text-stone-600">
                {s.source_number}
              </span>
              <span className="text-sm font-medium text-stone-900">
                {s.source_label || s.source_name}
              </span>
            </div>
            {s.snippet ? (
              <p className="mt-1.5 line-clamp-3 text-sm leading-relaxed text-stone-600">
                {s.snippet}
              </p>
            ) : null}
            {s.file_path ? (
              <p
                className="mt-1.5 font-mono text-[11px] leading-tight text-stone-400"
                title={s.file_path}
              >
                {truncatePath(s.file_path)}
              </p>
            ) : null}
            {s.chunk_id ? (
              <p className="mt-1 text-[10px] uppercase tracking-wide text-stone-400/90">
                Chunk <span className="font-mono normal-case tracking-normal">{s.chunk_id}</span>
              </p>
            ) : null}
          </li>
        ))}
      </ol>
    </div>
  );
}

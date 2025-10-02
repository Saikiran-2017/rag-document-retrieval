"use client";

import { useEffect, useRef } from "react";
import { ModeBadge } from "@/components/ModeBadge";
import { SourcesPanel } from "@/components/SourcesPanel";
import { WebSourcesPanel } from "@/components/WebSourcesPanel";
import type { UiMessage } from "@/lib/types";

function AssistantBody({ text, streaming }: { text: string; streaming?: boolean }) {
  return (
    <div className="whitespace-pre-wrap text-[15px] leading-relaxed text-stone-800">
      {text || (streaming ? "…" : "")}
    </div>
  );
}

export function ChatThread({ messages }: { messages: UiMessage[] }) {
  const endRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <div className="flex flex-1 flex-col gap-6 overflow-y-auto px-4 py-6 md:px-8">
      {messages.map((m) =>
        m.role === "user" ? (
          <div key={m.id} className="flex justify-end">
            <div className="max-w-[min(100%,42rem)] rounded-2xl bg-stone-800 px-4 py-3 text-[15px] leading-relaxed text-stone-50">
              {m.content}
            </div>
          </div>
        ) : (
          <div key={m.id} className="flex justify-start">
            <div className="max-w-[min(100%,48rem)]">
              <div className="mb-2 flex items-center gap-2">
                <ModeBadge mode={m.meta?.mode} />
              </div>
              <AssistantBody text={m.content} streaming={m.streaming} />
              {m.meta?.status_note ? (
                <p className="mt-3 text-sm text-stone-500">{m.meta.status_note}</p>
              ) : null}
              {m.meta?.validation_warning ? (
                <p className="mt-2 rounded-md border border-amber-200/80 bg-amber-50/90 px-3 py-2 text-sm text-amber-900">
                  {m.meta.validation_warning}
                </p>
              ) : null}
              {m.meta?.sources?.length ? (
                <SourcesPanel sources={m.meta.sources} />
              ) : null}
              {m.meta?.web_sources?.length ? (
                <WebSourcesPanel items={m.meta.web_sources} />
              ) : null}
            </div>
          </div>
        ),
      )}
      <div ref={endRef} />
    </div>
  );
}

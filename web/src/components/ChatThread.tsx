"use client";

import { useEffect, useRef } from "react";
import { ModeBadge } from "@/components/ModeBadge";
import { SourcesPanel } from "@/components/SourcesPanel";
import { WebSourcesPanel } from "@/components/WebSourcesPanel";
import type { UiMessage } from "@/lib/types";

function StreamingIndicator() {
  return (
    <span className="ka-stream-dots inline-flex gap-1 text-stone-400" aria-hidden>
      <span className="inline-block h-1 w-1 rounded-full bg-current" />
      <span className="inline-block h-1 w-1 rounded-full bg-current" />
      <span className="inline-block h-1 w-1 rounded-full bg-current" />
    </span>
  );
}

function AssistantBody({ text, streaming }: { text: string; streaming?: boolean }) {
  const showThinking = streaming && !text.trim();
  return (
    <div className="text-[15px] leading-[1.65] text-stone-800">
      {showThinking ? (
        <p className="flex items-center gap-2 text-sm text-stone-500">
          <StreamingIndicator />
          <span>Thinking</span>
        </p>
      ) : (
        <div className="whitespace-pre-wrap">{text}</div>
      )}
    </div>
  );
}

export function ChatThread({ messages }: { messages: UiMessage[] }) {
  const endRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  return (
    <div className="flex flex-1 flex-col gap-10 overflow-y-auto px-4 py-8 md:px-10 md:py-10">
      {messages.map((m) =>
        m.role === "user" ? (
          <div key={m.id} className="flex justify-end scroll-mt-24">
            <div className="max-w-[min(100%,34rem)] rounded-2xl rounded-br-md bg-stone-800 px-4 py-3.5 text-[15px] leading-[1.6] text-stone-50 shadow-sm">
              {m.content}
            </div>
          </div>
        ) : (
          <div key={m.id} className="flex justify-start scroll-mt-24">
            <div className="relative max-w-[min(100%,40rem)] pl-0 md:pl-1">
              <div
                className="absolute -left-0.5 top-1 hidden h-[calc(100%-0.25rem)] w-px bg-stone-200 md:block"
                aria-hidden
              />
              <div className="mb-2.5 flex min-h-[1.25rem] items-center gap-2">
                <ModeBadge mode={m.meta?.mode} />
              </div>
              <AssistantBody text={m.content} streaming={m.streaming} />
              {m.meta?.status_note ? (
                <p className="mt-4 border-l-2 border-stone-200 pl-3 text-sm leading-relaxed text-stone-500">
                  {m.meta.status_note}
                </p>
              ) : null}
              {m.meta?.validation_warning ? (
                <p className="mt-3 rounded-lg border border-amber-200/90 bg-amber-50 px-3.5 py-2.5 text-sm leading-snug text-amber-950">
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
      <div ref={endRef} className="h-px shrink-0" aria-hidden />
    </div>
  );
}

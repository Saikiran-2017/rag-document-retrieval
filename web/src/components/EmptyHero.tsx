"use client";

import {
  APP_NAME,
  EMPTY_STATE_VALUE_PROP,
  HERO_BEST_FOR,
  STARTER_QUESTIONS,
} from "@/lib/constants";

export function EmptyHero({
  onPickPrompt,
  disabled,
}: {
  onPickPrompt: (q: string) => void;
  disabled?: boolean;
}) {
  return (
    <div className="relative flex flex-1 flex-col items-center justify-center overflow-hidden px-5 py-12 md:py-16">
      <div
        className="pointer-events-none absolute inset-0 bg-[radial-gradient(ellipse_80%_50%_at_50%_-20%,rgb(231_229_228/0.45),transparent)]"
        aria-hidden
      />
      <div className="relative mx-auto flex w-full max-w-lg flex-col items-center text-center">
        <p className="text-[11px] font-semibold uppercase tracking-[0.2em] text-stone-400">
          Welcome
        </p>
        <h2 className="ka-text-balance mt-3 text-2xl font-semibold tracking-tight text-stone-900 md:text-[1.75rem] md:leading-snug">
          {APP_NAME}
        </h2>
        <p className="ka-text-balance mt-5 max-w-md text-[15px] leading-relaxed text-stone-600">
          {EMPTY_STATE_VALUE_PROP}
        </p>
        <p className="ka-text-balance mt-3 max-w-md text-sm leading-relaxed text-stone-500">
          {HERO_BEST_FOR}
        </p>
        <p className="mt-12 text-[10px] font-semibold uppercase tracking-[0.14em] text-stone-400">
          Try asking
        </p>
        <div className="mt-4 flex w-full max-w-md flex-col gap-2.5">
          {STARTER_QUESTIONS.map((q) => (
            <button
              key={q}
              type="button"
              disabled={disabled}
              onClick={() => onPickPrompt(q)}
              className="rounded-xl border border-stone-200/90 bg-white px-4 py-3.5 text-left text-sm leading-snug text-stone-700 shadow-sm shadow-stone-900/[0.04] transition hover:border-stone-300 hover:bg-stone-50 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-stone-400 disabled:pointer-events-none disabled:opacity-45"
            >
              {q}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

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
    <div className="mx-auto flex max-w-xl flex-col items-center px-6 py-16 text-center">
      <h2 className="text-2xl font-semibold tracking-tight text-stone-900">{APP_NAME}</h2>
      <p className="mt-4 text-[15px] leading-relaxed text-stone-600">{EMPTY_STATE_VALUE_PROP}</p>
      <p className="mt-3 text-sm leading-relaxed text-stone-500">{HERO_BEST_FOR}</p>
      <p className="mt-10 text-xs font-semibold uppercase tracking-wider text-stone-400">
        Suggested
      </p>
      <div className="mt-3 flex w-full max-w-md flex-col gap-2">
        {STARTER_QUESTIONS.map((q) => (
          <button
            key={q}
            type="button"
            disabled={disabled}
            onClick={() => onPickPrompt(q)}
            className="rounded-lg border border-stone-200 bg-white px-4 py-3 text-left text-sm text-stone-700 shadow-sm transition hover:border-stone-300 hover:bg-stone-50 disabled:opacity-50"
          >
            {q}
          </button>
        ))}
      </div>
    </div>
  );
}

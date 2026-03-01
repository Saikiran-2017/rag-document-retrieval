"use client";

import { APP_NAME } from "@/lib/constants";

export function AppLoadingShell() {
  return (
    <div
      className="flex min-h-screen flex-col items-center justify-center px-6"
      role="status"
      aria-live="polite"
      aria-label="Loading application"
    >
      <div className="w-full max-w-xs space-y-4 text-center">
        <p className="text-sm font-semibold tracking-tight text-stone-800">{APP_NAME}</p>
        <div className="space-y-2">
          <div className="ka-shimmer h-2.5 w-full rounded-full opacity-80" />
          <div className="ka-shimmer mx-auto h-2.5 w-4/5 rounded-full opacity-60" />
          <div className="ka-shimmer mx-auto h-2.5 w-3/5 rounded-full opacity-40" />
        </div>
        <p className="text-xs text-stone-500">Connecting…</p>
      </div>
    </div>
  );
}

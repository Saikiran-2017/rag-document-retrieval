"use client";

export function ChatSkeleton() {
  return (
    <div
      className="flex flex-1 flex-col gap-10 overflow-hidden px-4 py-8 md:px-10"
      role="status"
      aria-live="polite"
      aria-label="Loading messages"
    >
      <div className="ml-auto w-[min(100%,24rem)] space-y-2">
        <div className="ka-shimmer ml-auto h-10 w-3/4 rounded-2xl rounded-br-md" />
      </div>
      <div className="mr-auto w-[min(100%,28rem)] space-y-3">
        <div className="ka-shimmer h-3 w-20 rounded-full" />
        <div className="space-y-2">
          <div className="ka-shimmer h-3.5 w-full rounded-md" />
          <div className="ka-shimmer h-3.5 w-[92%] rounded-md" />
          <div className="ka-shimmer h-3.5 w-[70%] rounded-md" />
        </div>
      </div>
    </div>
  );
}

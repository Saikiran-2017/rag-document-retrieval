"use client";

function ExternalIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      width="12"
      height="12"
      viewBox="0 0 12 12"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden
    >
      <path
        d="M4.5 7.5H2A.5.5 0 0 1 1.5 7V2A.5.5 0 0 1 2 1.5h5a.5.5 0 0 1 .5.5V4M6 6l4-4m0 0v2.5M10 2H7.5"
        stroke="currentColor"
        strokeWidth="1.1"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export function WebSourcesPanel({
  items,
}: {
  items: Array<Record<string, string>>;
}) {
  if (!items.length) return null;
  return (
    <div className="mt-6 border-t border-stone-200/90 pt-5">
      <h3 className="mb-3 text-[11px] font-semibold uppercase tracking-[0.12em] text-stone-400">
        Web sources
      </h3>
      <ul className="space-y-3">
        {items.map((w, i) => {
          const title = w.title || w.snippet?.slice(0, 96)?.trim() || "Web result";
          const url = w.url || w.link || "";
          return (
            <li
              key={`${url}-${i}`}
              className="rounded-xl border border-stone-200/80 bg-white px-3.5 py-3 shadow-sm shadow-stone-900/[0.02]"
            >
              {url ? (
                <a
                  href={url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="group inline-flex items-start gap-1.5 text-sm font-medium text-stone-800 decoration-transparent underline-offset-4 hover:text-sky-900 hover:underline"
                >
                  <span className="min-w-0 break-words">{title}</span>
                  <ExternalIcon className="mt-0.5 shrink-0 text-stone-400 group-hover:text-sky-800" />
                </a>
              ) : (
                <span className="text-sm font-medium text-stone-800">{title}</span>
              )}
              {w.snippet ? (
                <p className="mt-2 text-xs leading-relaxed text-stone-500">{w.snippet}</p>
              ) : null}
            </li>
          );
        })}
      </ul>
    </div>
  );
}

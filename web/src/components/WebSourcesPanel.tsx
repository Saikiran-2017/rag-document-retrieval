"use client";

export function WebSourcesPanel({
  items,
}: {
  items: Array<Record<string, string>>;
}) {
  if (!items.length) return null;
  return (
    <div className="mt-4 border-t border-stone-200/80 pt-4">
      <h3 className="mb-2 text-xs font-semibold uppercase tracking-wider text-stone-500">
        Web sources
      </h3>
      <ul className="space-y-2 text-sm">
        {items.map((w, i) => {
          const title = w.title || w.snippet?.slice(0, 80) || "Result";
          const url = w.url || w.link || "";
          return (
            <li key={`${url}-${i}`}>
              {url ? (
                <a
                  href={url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-sky-800 underline decoration-stone-300 underline-offset-2 hover:decoration-sky-600"
                >
                  {title}
                </a>
              ) : (
                <span className="text-stone-700">{title}</span>
              )}
              {w.snippet ? (
                <p className="mt-1 text-xs leading-relaxed text-stone-500">{w.snippet}</p>
              ) : null}
            </li>
          );
        })}
      </ul>
    </div>
  );
}

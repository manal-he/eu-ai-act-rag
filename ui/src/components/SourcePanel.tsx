import { useEffect, useState } from "react";
import { getUnit } from "../api";
import { t } from "../i18n";
import type { Lang, LegalUnit } from "../types";

export interface Selection {
  unitId: string;
  lang: Lang;
  label: string;
  chunkText: string;
  paragraph: string | null;
}

interface Props {
  selection: Selection | null;
  uiLang: Lang;
  onClose: () => void;
}

function normalize(s: string) {
  return s.replace(/\s+/g, " ").trim();
}

export function SourcePanel({ selection, uiLang, onClose }: Props) {
  const [unit, setUnit] = useState<LegalUnit | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!selection) return;
    let cancelled = false;
    setUnit(null);
    setError(null);
    getUnit(selection.unitId, selection.lang)
      .then((u) => !cancelled && setUnit(u))
      .catch((e: Error) => !cancelled && setError(e.message));
    return () => {
      cancelled = true;
    };
  }, [selection]);

  useEffect(() => {
    if (!unit || !selection) return;
    document
      .getElementById("cited-paragraph")
      ?.scrollIntoView({ behavior: "smooth", block: "center" });
  }, [unit, selection]);

  const s = t[uiLang];
  if (!selection) {
    return (
      <div className="flex h-full items-center justify-center p-8 text-center text-sm text-slate-500 dark:text-slate-400">
        <p className="max-w-[16rem]">{s.panelEmpty}</p>
      </div>
    );
  }

  const chunk = normalize(selection.chunkText);
  return (
    <div className="flex h-full flex-col">
      <div className="flex items-start justify-between gap-3 border-b border-slate-200 px-5 py-4 dark:border-slate-800">
        <div className="min-w-0">
          <p className="text-[11px] font-semibold uppercase tracking-wider text-eu-600 dark:text-eu-100">
            {selection.label}
          </p>
          <h2 className="mt-1 font-serif text-lg font-semibold leading-snug">
            {unit?.title || (unit ? `${unit.kind} ${unit.number}` : "…")}
          </h2>
          {unit?.chapter && (
            <p className="mt-1 truncate text-xs text-slate-500 dark:text-slate-400">{unit.chapter}</p>
          )}
        </div>
        <button
          type="button"
          onClick={onClose}
          aria-label="Close"
          className="rounded-md p-1 text-slate-400 hover:bg-slate-100 hover:text-slate-700 dark:hover:bg-slate-800"
        >
          <svg viewBox="0 0 20 20" className="h-5 w-5" fill="currentColor">
            <path d="M6.28 5.22a.75.75 0 0 0-1.06 1.06L8.94 10l-3.72 3.72a.75.75 0 1 0 1.06 1.06L10 11.06l3.72 3.72a.75.75 0 1 0 1.06-1.06L11.06 10l3.72-3.72a.75.75 0 0 0-1.06-1.06L10 8.94 6.28 5.22Z" />
          </svg>
        </button>
      </div>
      <div className="flex-1 space-y-3 overflow-y-auto px-5 py-4 font-serif text-[14.5px] leading-relaxed">
        {error && <p className="text-sm text-red-600">{error}</p>}
        {!unit && !error && (
          <div className="space-y-2">
            {[0, 1, 2].map((i) => (
              <div key={i} className="h-4 animate-pulse rounded bg-slate-200 dark:bg-slate-800" />
            ))}
          </div>
        )}
        {unit?.paragraphs.map((p, i) => {
          const blocks = [p.lead, ...p.points].filter(Boolean);
          const text = normalize(blocks.join(" "));
          const cited =
            (selection.paragraph !== null && p.number === selection.paragraph) ||
            (selection.paragraph === null && text.includes(chunk.slice(0, 120)));
          return (
            <div
              key={i}
              id={cited ? "cited-paragraph" : undefined}
              className={`rounded-lg px-3 py-2 ${
                cited ? "bg-star/15 ring-1 ring-star/60" : ""
              }`}
            >
              {blocks.map((b, j) => (
                <p key={j} className={j > 0 ? "mt-1.5 pl-4" : ""}>
                  {b}
                </p>
              ))}
            </div>
          );
        })}
      </div>
      {unit?.source_url && (
        <a
          href={unit.source_url}
          target="_blank"
          rel="noreferrer"
          className="border-t border-slate-200 px-5 py-3 text-xs font-medium text-eu-600 hover:underline dark:border-slate-800 dark:text-eu-100"
        >
          {s.official} ↗
        </a>
      )}
    </div>
  );
}

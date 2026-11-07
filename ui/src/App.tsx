import { useEffect, useRef, useState } from "react";
import { askStream, getHealth } from "./api";
import { AnswerText } from "./components/AnswerText";
import { SourcePanel, type Selection } from "./components/SourcePanel";
import { t } from "./i18n";
import type { Citation, Health, Lang, SearchHit, Turn } from "./types";

type LangChoice = Lang | "auto";

function guessLang(q: string): Lang {
  return /[éèêàçùôî]|\b(le|la|les|des|est|quelle?s?|quand|une)\b/i.test(q) ? "fr" : "en";
}

export default function App() {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [langChoice, setLangChoice] = useState<LangChoice>("auto");
  const [uiLang, setUiLang] = useState<Lang>("en");
  const [health, setHealth] = useState<Health | null>(null);
  const [selection, setSelection] = useState<Selection | null>(null);
  const [activeChunk, setActiveChunk] = useState<string | null>(null);
  const bottom = useRef<HTMLDivElement>(null);
  const abort = useRef<AbortController | null>(null);
  const s = t[uiLang];
  const busy = turns.some((x) => x.pending);

  useEffect(() => {
    getHealth().then(setHealth).catch(() => setHealth(null));
  }, []);

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns]);

  const update = (id: string, patch: Partial<Turn> | ((t: Turn) => Partial<Turn>)) =>
    setTurns((all) =>
      all.map((x) => (x.id === id ? { ...x, ...(typeof patch === "function" ? patch(x) : patch) } : x)),
    );

  async function ask(question: string) {
    const q = question.trim();
    if (!q || busy) return;
    const lang = langChoice === "auto" ? null : langChoice;
    setUiLang(lang ?? guessLang(q));
    const id = crypto.randomUUID();
    setTurns((all) => [...all, { id, question: q, streamed: "", hits: [], pending: true }]);
    setInput("");
    abort.current = new AbortController();
    try {
      await askStream(
        q,
        lang,
        {
          onRetrieval: (hits) => update(id, { hits }),
          onToken: (tok) => update(id, (x) => ({ streamed: x.streamed + tok })),
          onFinal: (answer) => {
            update(id, { answer, pending: false });
            setUiLang(answer.language);
          },
          onError: (error) => update(id, { error, pending: false }),
        },
        abort.current.signal,
      );
    } catch (e) {
      update(id, { error: (e as Error).message, pending: false });
    }
  }

  function openCitation(turn: Turn, c: Citation) {
    const hit = turn.hits.find((h) => h.chunk.chunk_id === c.chunk_id);
    setActiveChunk(c.chunk_id);
    setSelection({
      unitId: c.unit_id,
      lang: hit?.chunk.lang ?? turn.answer?.language ?? "en",
      label: c.label,
      chunkText: hit?.chunk.text ?? c.quote,
      paragraph: hit?.chunk.paragraph ?? null,
    });
  }

  function openHit(h: SearchHit, label: string) {
    setActiveChunk(h.chunk.chunk_id);
    setSelection({
      unitId: h.chunk.unit_id,
      lang: h.chunk.lang,
      label,
      chunkText: h.chunk.text,
      paragraph: h.chunk.paragraph,
    });
  }

  return (
    <div className="flex h-screen flex-col">
      <header className="border-b border-slate-200 bg-white/80 backdrop-blur dark:border-slate-800 dark:bg-slate-900/80">
        <div className="mx-auto flex max-w-[1400px] items-center justify-between gap-4 px-4 py-3 sm:px-6">
          <div className="flex min-w-0 items-center gap-3">
            <img src="/favicon.svg" alt="" className="h-8 w-8 shrink-0" />
            <div className="min-w-0">
              <h1 className="truncate text-[15px] font-semibold">{s.title}</h1>
              <p className="hidden truncate text-xs text-slate-500 sm:block dark:text-slate-400">
                {s.subtitle}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            {health && (
              <span
                className="hidden items-center gap-1.5 rounded-full border border-slate-200 px-2.5 py-1 text-[11px] text-slate-600 md:inline-flex dark:border-slate-700 dark:text-slate-300"
                title={`${health.retriever} · reranker: ${health.reranker}`}
              >
                <span className="h-1.5 w-1.5 rounded-full bg-emerald-500" />
                {health.llm} · {health.units} units
              </span>
            )}
            <div className="flex rounded-lg border border-slate-200 p-0.5 text-xs dark:border-slate-700">
              {(["auto", "en", "fr"] as LangChoice[]).map((l) => (
                <button
                  key={l}
                  type="button"
                  onClick={() => {
                    setLangChoice(l);
                    if (l !== "auto") setUiLang(l);
                  }}
                  className={`rounded-md px-2.5 py-1 font-medium uppercase ${
                    langChoice === l
                      ? "bg-eu-700 text-white"
                      : "text-slate-500 hover:text-slate-900 dark:hover:text-white"
                  }`}
                >
                  {l}
                </button>
              ))}
            </div>
          </div>
        </div>
      </header>

      <div className="mx-auto flex w-full max-w-[1400px] flex-1 overflow-hidden">
        <main className="flex min-w-0 flex-1 flex-col">
          <div className="flex-1 overflow-y-auto px-4 py-6 sm:px-6">
            {turns.length === 0 && (
              <div className="mx-auto mt-10 max-w-xl text-center">
                <p className="font-serif text-2xl font-semibold">{s.title}</p>
                <p className="mt-2 text-sm text-slate-500 dark:text-slate-400">{s.subtitle}</p>
                <div className="mt-8 grid gap-2">
                  {[...t.en.examples, ...t.fr.examples].map((ex) => (
                    <button
                      key={ex}
                      type="button"
                      onClick={() => ask(ex)}
                      className="rounded-xl border border-slate-200 bg-white px-4 py-3 text-left text-sm text-slate-700 shadow-sm transition hover:border-eu-500 hover:shadow dark:border-slate-800 dark:bg-slate-900 dark:text-slate-200"
                    >
                      {ex}
                    </button>
                  ))}
                </div>
              </div>
            )}

            <div className="mx-auto max-w-3xl space-y-8">
              {turns.map((turn) => (
                <section key={turn.id} className="space-y-3">
                  <div className="flex justify-end">
                    <p className="max-w-[85%] rounded-2xl rounded-br-md bg-eu-700 px-4 py-2.5 text-sm text-white">
                      {turn.question}
                    </p>
                  </div>
                  <article className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm dark:border-slate-800 dark:bg-slate-900">
                    {turn.error && <p className="text-sm text-red-600">{turn.error}</p>}
                    {turn.pending && !turn.streamed && (
                      <p className="animate-pulse text-sm text-slate-500">{t[uiLang].thinking}</p>
                    )}
                    {turn.answer?.abstained ? (
                      <div className="flex gap-3 rounded-xl bg-amber-50 p-3 text-sm text-amber-900 dark:bg-amber-950/40 dark:text-amber-200">
                        <span aria-hidden>⚠</span>
                        <div>
                          <p className="font-semibold">{t[turn.answer.language].abstained}</p>
                          <p className="mt-0.5">{turn.answer.answer}</p>
                        </div>
                      </div>
                    ) : (
                      (turn.answer || turn.streamed) && (
                        <AnswerText
                          text={turn.answer?.answer ?? turn.streamed}
                          citations={turn.answer?.citations ?? []}
                          activeChunk={activeChunk}
                          onCite={(c) => openCitation(turn, c)}
                        />
                      )
                    )}
                    {turn.answer && turn.answer.citations.length > 0 && (
                      <div className="mt-4 flex flex-wrap gap-2 border-t border-slate-100 pt-3 dark:border-slate-800">
                        <span className="mr-1 self-center text-[11px] font-semibold uppercase tracking-wider text-slate-400">
                          {t[turn.answer.language].sources}
                        </span>
                        {turn.answer.citations.map((c) => (
                          <button
                            key={c.marker}
                            type="button"
                            onClick={() => openCitation(turn, c)}
                            className={`rounded-full border px-2.5 py-1 text-xs transition ${
                              activeChunk === c.chunk_id
                                ? "border-eu-600 bg-eu-600 text-white"
                                : "border-slate-200 text-slate-700 hover:border-eu-500 dark:border-slate-700 dark:text-slate-200"
                            }`}
                          >
                            <span className="font-semibold">[{c.marker}]</span> {c.label}
                          </button>
                        ))}
                      </div>
                    )}
                    {turn.hits.length > 0 && (
                      <details className="mt-3 text-xs text-slate-500">
                        <summary className="cursor-pointer select-none hover:text-slate-800 dark:hover:text-slate-200">
                          {t[turn.answer?.language ?? uiLang].retrieved} ({turn.hits.length})
                          {turn.answer?.latency_ms !== undefined &&
                            ` · ${Math.round(turn.answer.latency_ms)} ms · ${turn.answer.provider}`}
                        </summary>
                        <ol className="mt-2 space-y-1.5">
                          {turn.hits.map((h, i) => {
                            const label = `${h.chunk.unit_id}${h.chunk.paragraph ? `(${h.chunk.paragraph})` : ""}`;
                            return (
                              <li key={h.chunk.chunk_id}>
                                <button
                                  type="button"
                                  onClick={() => openHit(h, label)}
                                  className="w-full rounded-lg px-2 py-1.5 text-left hover:bg-slate-50 dark:hover:bg-slate-800"
                                >
                                  <span className="font-mono text-[11px] text-eu-600 dark:text-eu-100">
                                    {i + 1}. {label}
                                  </span>{" "}
                                  <span className="line-clamp-1">{h.chunk.text}</span>
                                </button>
                              </li>
                            );
                          })}
                        </ol>
                      </details>
                    )}
                  </article>
                </section>
              ))}
              <div ref={bottom} />
            </div>
          </div>

          <form
            onSubmit={(e) => {
              e.preventDefault();
              ask(input);
            }}
            className="border-t border-slate-200 bg-white px-4 py-3 sm:px-6 dark:border-slate-800 dark:bg-slate-900"
          >
            <div className="mx-auto flex max-w-3xl gap-2">
              <input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder={s.placeholder}
                maxLength={2000}
                className="min-w-0 flex-1 rounded-xl border border-slate-300 bg-white px-4 py-2.5 text-sm outline-none ring-eu-500 placeholder:text-slate-400 focus:ring-2 dark:border-slate-700 dark:bg-slate-950"
              />
              <button
                type="submit"
                disabled={busy || !input.trim()}
                className="rounded-xl bg-eu-700 px-5 py-2.5 text-sm font-semibold text-white transition hover:bg-eu-600 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {s.send}
              </button>
            </div>
            <p className="mx-auto mt-2 max-w-3xl text-center text-[11px] text-slate-400">
              {s.disclaimer}
            </p>
          </form>
        </main>

        <aside
          className={`${
            selection ? "fixed inset-0 z-20 lg:static" : "hidden lg:block"
          } w-full border-l border-slate-200 bg-white lg:w-[440px] lg:shrink-0 dark:border-slate-800 dark:bg-slate-900`}
        >
          <SourcePanel
            selection={selection}
            uiLang={uiLang}
            onClose={() => {
              setSelection(null);
              setActiveChunk(null);
            }}
          />
        </aside>
      </div>
    </div>
  );
}

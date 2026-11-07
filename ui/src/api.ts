import type { Answer, Health, Lang, LegalUnit, SearchHit } from "./types";

const BASE = import.meta.env.VITE_API_BASE ?? "/api";

export async function getHealth(): Promise<Health> {
  const r = await fetch(`${BASE}/health`);
  if (!r.ok) throw new Error(`health ${r.status}`);
  return r.json();
}

export async function getUnit(unitId: string, lang: Lang): Promise<LegalUnit> {
  const r = await fetch(`${BASE}/units/${encodeURIComponent(unitId)}?lang=${lang}`);
  if (!r.ok) throw new Error(`unit ${r.status}`);
  return r.json();
}

export interface StreamHandlers {
  onRetrieval: (hits: SearchHit[]) => void;
  onToken: (text: string) => void;
  onFinal: (answer: Answer) => void;
  onError: (message: string) => void;
}

/** POST /ask/stream and parse the Server-Sent Events by hand (EventSource is GET-only). */
export async function askStream(
  question: string,
  lang: Lang | null,
  h: StreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  const r = await fetch(`${BASE}/ask/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, lang }),
    signal,
  });
  if (!r.ok || !r.body) {
    h.onError(`Request failed (${r.status})`);
    return;
  }
  const reader = r.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let sep: number;
    while ((sep = buffer.indexOf("\n\n")) !== -1) {
      const block = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);
      const event = /^event: (.*)$/m.exec(block)?.[1];
      const data = /^data: (.*)$/m.exec(block)?.[1];
      if (!event || data === undefined) continue;
      const payload = JSON.parse(data);
      if (event === "retrieval") h.onRetrieval(payload);
      else if (event === "token") h.onToken(payload.text);
      else if (event === "final") h.onFinal(payload);
      else if (event === "error") h.onError(payload.detail ?? "error");
    }
  }
}

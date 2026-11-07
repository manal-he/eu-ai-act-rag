import type { Citation } from "../types";

interface Props {
  text: string;
  citations: Citation[];
  activeChunk: string | null;
  onCite: (c: Citation) => void;
}

/** Renders answer text, turning [n] markers into clickable citation chips. */
export function AnswerText({ text, citations, activeChunk, onCite }: Props) {
  const byMarker = new Map(citations.map((c) => [c.marker, c]));
  const lines = text.split("\n");
  return (
    <div className="space-y-1.5 font-serif text-[15.5px] leading-relaxed">
      {lines.map((line, i) => (
        <p key={i} className={line.startsWith("- ") ? "pl-4 -indent-3" : ""}>
          {line
            .replace(/^- /, "• ")
            .split(/(\[\d{1,2}\])/g)
            .map((part, j) => {
              const m = /^\[(\d{1,2})\]$/.exec(part);
              const cite = m ? byMarker.get(Number(m[1])) : undefined;
              if (!m) return <span key={j}>{part}</span>;
              if (!cite) return null;
              const active = activeChunk === cite.chunk_id;
              return (
                <button
                  key={j}
                  type="button"
                  onClick={() => onCite(cite)}
                  title={cite.label}
                  className={`mx-0.5 inline-flex -translate-y-px items-center rounded-md px-1.5 py-px align-baseline font-sans text-[11px] font-semibold transition ${
                    active
                      ? "bg-eu-600 text-white"
                      : "bg-eu-50 text-eu-700 hover:bg-eu-100 dark:bg-eu-900 dark:text-eu-100"
                  }`}
                >
                  {cite.marker}
                </button>
              );
            })}
        </p>
      ))}
    </div>
  );
}

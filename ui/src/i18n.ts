import type { Lang } from "./types";

export const t = {
  en: {
    title: "AI Act & GDPR Assistant",
    subtitle: "Answers grounded in Regulation (EU) 2024/1689 and Regulation (EU) 2016/679",
    placeholder: "Ask about the AI Act or the GDPR, in English or French…",
    send: "Ask",
    sources: "Sources",
    retrieved: "Retrieved passages",
    abstained: "No grounded answer",
    panelEmpty: "Click a citation to read the source provision.",
    official: "Official text on EUR-Lex",
    disclaimer:
      "Informational tool, not legal advice. Answers may be incomplete; always check the official text.",
    examples: [
      "Is emotion recognition allowed in the workplace?",
      "When must a personal data breach be notified?",
      "What obligations apply to providers of general-purpose AI models?",
    ],
    thinking: "Searching the regulations…",
  },
  fr: {
    title: "Assistant Règlement IA & RGPD",
    subtitle: "Réponses fondées sur le règlement (UE) 2024/1689 et le règlement (UE) 2016/679",
    placeholder: "Posez une question sur le règlement IA ou le RGPD, en français ou en anglais…",
    send: "Demander",
    sources: "Sources",
    retrieved: "Passages retrouvés",
    abstained: "Pas de réponse fondée",
    panelEmpty: "Cliquez sur une citation pour lire la disposition source.",
    official: "Texte officiel sur EUR-Lex",
    disclaimer:
      "Outil d'information, pas un conseil juridique. Les réponses peuvent être incomplètes ; vérifiez toujours le texte officiel.",
    examples: [
      "La notation sociale est-elle interdite ?",
      "Quand une AIPD est-elle obligatoire ?",
      "Quelles obligations pèsent sur les déployeurs de systèmes à haut risque ?",
    ],
    thinking: "Recherche dans les règlements…",
  },
} satisfies Record<Lang, Record<string, string | string[]>>;

import { createChat, listChats } from "./api";
import type { ChatSessionRow } from "./types";

/**
 * Deduplicate concurrent bootstraps (e.g. React Strict Mode double mount) so we
 * only run the "create chat if empty" path once per overlapping load.
 */
let inFlight: Promise<ChatSessionRow[]> | null = null;

export function loadInitialChatSessions(): Promise<ChatSessionRow[]> {
  if (inFlight) return inFlight;
  const p = (async () => {
    let list = await listChats();
    if (list.length === 0) {
      await createChat();
      list = await listChats();
    }
    return list;
  })();
  inFlight = p;
  p.finally(() => {
    if (inFlight === p) inFlight = null;
  });
  return p;
}

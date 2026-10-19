// Durable resume cursor. Jetstream v2 `seq` values are specific to one
// instance, so the file records the service too. When the service changes,
// resume by time instead (values >= 1e15 are read as unix microseconds).

import { mkdir, rename, writeFile, readFile } from "node:fs/promises";
import { dirname } from "node:path";

export interface CursorState {
  service: string;
  seq: number;
  timeUs: number;
}

export async function loadCursor(path: string): Promise<CursorState | null> {
  let text: string;
  try {
    text = await readFile(path, "utf8");
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return null;
    throw error;
  }
  const parsed = JSON.parse(text) as CursorState;
  if (typeof parsed.service !== "string" || !Number.isSafeInteger(parsed.seq) || !Number.isSafeInteger(parsed.timeUs)) {
    throw new Error(`invalid cursor file ${path}; refusing to guess a resume point`);
  }
  return parsed;
}

export async function saveCursor(path: string, state: CursorState): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  const tmp = `${path}.tmp`;
  await writeFile(tmp, JSON.stringify(state));
  await rename(tmp, path);
}

/**
 * Pick the live() cursor for a service. Same service: its seq. Different
 * service: last event time minus a 30 s overlap, as a microsecond timestamp.
 * Nothing saved: undefined (start at the live tip).
 */
export function resumeCursor(saved: CursorState | null, service: string): number | undefined {
  if (!saved) return undefined;
  if (saved.service === service && saved.seq > 0) return saved.seq;
  if (saved.timeUs > 0) return Math.max(1e15, saved.timeUs - 30_000_000);
  return undefined;
}

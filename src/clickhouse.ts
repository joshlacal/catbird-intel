// Batched ClickHouse writer over the HTTP interface. Rows accumulate per table
// and are swapped out atomically before flushing so the WebSocket consumer
// never blocks on HTTP I/O (avoiding Bun's unimplemented ws.pause() buffer
// overflow during high-rate catch-up). The caller persists its cursor only
// after a batch write succeeds, so a crash replays events rather than dropping
// them. Replays are harmless because every record table is keyed and versioned.

import type { Row, TableName } from "./extract.ts";

export interface WriterOptions {
  url: string;
  database: string;
  user?: string;
  password?: string;
  maxRows: number;
  timeoutMs?: number;
}

export interface FlushBatch {
  batches: [TableName, Row[]][];
  count: number;
}

export class ClickHouseWriter {
  private buffers = new Map<TableName, Row[]>();
  private pending = 0;

  constructor(private readonly opts: WriterOptions) {}

  get bufferedRows(): number {
    return this.pending;
  }

  add(table: TableName, row: Row): void {
    let buf = this.buffers.get(table);
    if (!buf) {
      buf = [];
      this.buffers.set(table, buf);
    }
    buf.push(row);
    this.pending++;
  }

  shouldFlush(): boolean {
    return this.pending >= this.opts.maxRows;
  }

  /** Atomically swap out all buffered rows so new events can accumulate while HTTP I/O runs. */
  takeBatch(): FlushBatch {
    if (this.pending === 0) return { batches: [], count: 0 };
    const batches = [...this.buffers.entries()].filter(([, rows]) => rows.length > 0);
    const count = this.pending;
    this.buffers = new Map<TableName, Row[]>();
    this.pending = 0;
    return { batches, count };
  }

  /** Write a previously taken batch. Safe to retry on transient ClickHouse errors. */
  async writeBatch(batch: FlushBatch): Promise<number> {
    if (batch.count === 0 || batch.batches.length === 0) return 0;
    await Promise.all(batch.batches.map(([table, rows]) => this.insert(table, rows)));
    return batch.count;
  }

  /** Insert every buffered table. Returns the number of rows written. */
  async flush(): Promise<number> {
    const batch = this.takeBatch();
    return this.writeBatch(batch);
  }

  async insert(table: string, rows: Row[]): Promise<void> {
    const query = `INSERT INTO ${this.opts.database}.${table} FORMAT JSONEachRow`;
    const body = rows.map((r) => JSON.stringify(r)).join("\n");
    await this.request(query, body);
  }

  async query(sql: string): Promise<string> {
    return this.request(sql, undefined);
  }

  private async request(query: string, body: string | undefined): Promise<string> {
    const u = new URL(this.opts.url);
    u.searchParams.set("query", query);
    u.searchParams.set("input_format_skip_unknown_fields", "1");
    u.searchParams.set("date_time_input_format", "best_effort");
    u.searchParams.set("input_format_json_throw_on_bad_escape_sequence", "0");
    // Monthly partitions bound this to ~50; the guard is headroom, not a fix.
    u.searchParams.set("max_partitions_per_insert_block", "500");
    const headers: Record<string, string> = {};
    if (this.opts.user) headers["X-ClickHouse-User"] = this.opts.user;
    if (this.opts.password) headers["X-ClickHouse-Key"] = this.opts.password;
    const res = await fetch(u, {
      method: "POST",
      headers,
      body: body ?? "",
      signal: AbortSignal.timeout(this.opts.timeoutMs ?? 60_000),
    });
    const text = await res.text();
    if (!res.ok) throw new Error(`ClickHouse ${res.status}: ${text.slice(0, 500)}`);
    return text;
  }
}

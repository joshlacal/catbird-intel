// Batched ClickHouse writer over the HTTP interface. Rows accumulate per table
// and are flushed together; the caller persists its cursor only after a flush
// succeeds, so a crash replays events rather than dropping them. Replays are
// harmless because every record table is keyed and versioned.

import type { Row, TableName } from "./extract.ts";

export interface WriterOptions {
  url: string;
  database: string;
  user?: string;
  password?: string;
  maxRows: number;
  timeoutMs?: number;
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

  /** Insert every buffered table. Returns the number of rows written. */
  async flush(): Promise<number> {
    if (this.pending === 0) return 0;
    const batches = [...this.buffers.entries()].filter(([, rows]) => rows.length > 0);
    // Insert all tables before clearing anything, so a failure retries the
    // whole batch (duplicates are collapsed by the table engines).
    await Promise.all(batches.map(([table, rows]) => this.insert(table, rows)));
    const written = this.pending;
    this.buffers.clear();
    this.pending = 0;
    return written;
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

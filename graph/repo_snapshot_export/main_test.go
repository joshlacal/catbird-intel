package main

import (
	"bytes"
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cockroachdb/pebble"
)

func TestExportReadsOnlyRepoRows(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "meta.pebble")
	db, err := pebble.Open(dir, &pebble.Options{})
	if err != nil {
		t.Fatal(err)
	}
	put := func(k, v string) {
		if err := db.Set([]byte(k), []byte(v), pebble.Sync); err != nil {
			t.Fatal(err)
		}
	}
	put("repo/did:plc:a", `{"backfill":{"status":"complete","rev":"3mya2222222b2","completed_at":"2026-10-21T12:00:00Z"},"host":"morel.us-east.host.bsky.network","rev":"3mya2222222c2","active":true}`)
	put("repo/did:plc:b", `{"backfill":{"status":"failed","rev":"","last_error":"xrpc 429"},"host":"amanita.us-east.host.bsky.network","active":true}`)
	put("repo/did:plc:c", `{"backfill":{"status":"unavailable"},"active":false}`)
	put("host/morel.us-east.host.bsky.network", `{}`)
	put("repo0", `not a repo row`)
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}

	var buf bytes.Buffer
	n, err := Export(dir, &buf)
	if err != nil {
		t.Fatal(err)
	}
	if n != 3 {
		t.Fatalf("rows = %d, want 3", n)
	}
	var rows []Row
	for _, line := range strings.Split(strings.TrimSpace(buf.String()), "\n") {
		var r Row
		if err := json.Unmarshal([]byte(line), &r); err != nil {
			t.Fatal(err)
		}
		rows = append(rows, r)
	}
	a, b, c := rows[0], rows[1], rows[2]
	if a.DID != "did:plc:a" || a.SnapshotRev != "3mya2222222b2" || a.LatestRev != "3mya2222222c2" ||
		a.CompletedAt != "2026-10-21T12:00:00Z" || a.Active != 1 {
		t.Fatalf("complete row wrong: %+v", a)
	}
	if b.Status != "failed" || b.SnapshotRev != "" || b.Host != "amanita.us-east.host.bsky.network" {
		t.Fatalf("failed row must not carry a snapshot rev: %+v", b)
	}
	if c.Status != "unavailable" || c.Active != 0 {
		t.Fatalf("unavailable row wrong: %+v", c)
	}
}

func TestExportRefusesLockedStore(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "meta.pebble")
	db, err := pebble.Open(dir, &pebble.Options{})
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if _, err := Export(dir, &bytes.Buffer{}); err == nil {
		t.Fatal("expected the live store's lock to refuse a second open")
	}
}

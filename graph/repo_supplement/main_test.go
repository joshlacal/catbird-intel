package main

import (
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"testing"
	"time"
)

func TestReadTargetsSkipsCompleteAndInactiveRows(t *testing.T) {
	path := filepath.Join(t.TempDir(), "snap.jsonl")
	rows := `{"did":"did:plc:a","host":"h1","status":"failed","active":1}
{"did":"did:plc:b","host":"h1","status":"failed","active":0}
{"did":"did:plc:c","host":"h1","status":"complete","active":1}
{"did":"did:plc:d","host":"h2","status":"pending","active":1}
{"did":"did:plc:e","host":"","status":"failed","active":1}
`
	if err := os.WriteFile(path, []byte(rows), 0o644); err != nil {
		t.Fatal(err)
	}
	got, err := readTargets(path, "")
	if err != nil {
		t.Fatal(err)
	}
	if len(got["h1"]) != 1 || got["h1"][0] != "did:plc:a" || len(got["h2"]) != 1 {
		t.Fatalf("targets = %v", got)
	}
	only, _ := readTargets(path, "h2")
	if len(only) != 1 || len(only["h2"]) != 1 {
		t.Fatalf("host filter = %v", only)
	}
}

func TestPacerPausesWhenWindowNearlySpent(t *testing.T) {
	p := &pacer{interval: time.Millisecond, reserve: 0.25}
	reset := time.Now().Add(90 * time.Second)
	h := http.Header{}
	h.Set("RateLimit-Limit", "6000")
	h.Set("RateLimit-Remaining", "1000")
	h.Set("RateLimit-Reset", strconv.FormatInt(reset.Unix(), 10))
	p.observe(h)
	if p.next.Before(reset) {
		t.Fatalf("expected pause until %v, next=%v", reset, p.next)
	}

	q := &pacer{interval: time.Millisecond, reserve: 0.25}
	h.Set("RateLimit-Remaining", "5000")
	q.observe(h)
	if !q.next.IsZero() {
		t.Fatalf("plenty of budget should not pause, next=%v", q.next)
	}
}

func TestSlimRecordKeepsOnlyExtractorFields(t *testing.T) {
	got := slimRecord(map[string]any{
		"$type":     "app.bsky.feed.like",
		"subject":   map[string]any{"uri": "at://did:plc:x/app.bsky.feed.post/1", "cid": "bafy"},
		"createdAt": "2026-01-01T00:00:00Z",
		"feeds":     []any{map[string]any{"uri": "at://f/1", "cid": "bafy"}, "junk"},
		"text":      "dropped",
	})
	if _, ok := got["text"]; ok {
		t.Fatal("unexpected field kept")
	}
	if got["subject"].(map[string]any)["uri"] != "at://did:plc:x/app.bsky.feed.post/1" {
		t.Fatalf("subject = %v", got["subject"])
	}
	if len(got["feeds"].([]any)) != 1 {
		t.Fatalf("feeds = %v", got["feeds"])
	}
}

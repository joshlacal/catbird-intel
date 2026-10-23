// Command repo_supplement downloads whole repos for DIDs the collector could not
// fetch and writes them in the shapes graph/fold_follows.sql already reads.
//
// Input is repo_snapshot_export JSONL (did, host, status ...); only active rows
// whose status is failed or pending are fetched, grouped by PDS host. Each host gets
// its own pacer: requests are spaced to -rate per second, and the host's
// RateLimit-Remaining/RateLimit-Reset headers pause the host before the
// published window is spent. A 429 pauses that host until its reset.
//
// Output, per host, in -out:
//
//	events-<host>.jsonl     decoder-shaped commit events, one per kept record
//	snapshots-<host>.jsonl  repo_snapshot_export rows with the download's rev
//	failures-<host>.jsonl   DIDs that could not be fetched, with the error
//
// A rerun skips DIDs already present in snapshots or failures.
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/bluesky-social/indigo/atproto/atdata"
	cid "github.com/ipfs/go-cid"
	"github.com/bluesky-social/indigo/atproto/repo"
)

var defaultCollections = []string{
	"app.bsky.graph.follow", "app.bsky.graph.block", "app.bsky.graph.list",
	"app.bsky.graph.listitem", "app.bsky.graph.starterpack", "app.bsky.feed.generator",
	"app.bsky.actor.profile", "app.bsky.actor.contentVisibilityDeclaration",
}

// Record fields archive_graph_extract.row_from_event reads; everything else is dropped.
var keptFields = []string{"subject", "createdAt", "list", "name", "displayName",
	"description", "purpose", "feeds", "did", "hideFromAlgorithmicRecommendations"}

type target struct {
	DID    string `json:"did"`
	Host   string `json:"host"`
	Status string `json:"status"`
	Active int    `json:"active"`
}

type config struct {
	out         string
	rate        float64
	workers     int
	reserve     float64
	userAgent   string
	collections map[string]bool
	client      *http.Client
}

func main() {
	in := flag.String("input", "", "repo_snapshot_export JSONL")
	out := flag.String("out", "", "output directory")
	rate := flag.Float64("rate", 12, "max getRepo requests per second per host")
	workers := flag.Int("workers", 6, "concurrent downloads per host")
	reserve := flag.Float64("reserve", 0.25, "pause a host when RateLimit-Remaining falls below this share of the limit")
	hosts := flag.String("hosts", "", "comma-separated hosts to fetch (default: all)")
	limit := flag.Int("limit", 0, "fetch at most this many DIDs per host (0 = all)")
	cols := flag.String("collections", strings.Join(defaultCollections, ","), "collections to keep")
	ua := flag.String("user-agent", "catbird-intel-repo-supplement/1 (+https://catbird.blue)", "User-Agent")
	flag.Parse()
	if *in == "" || *out == "" {
		fmt.Fprintln(os.Stderr, "usage: repo_supplement -input repo_snapshots.jsonl -out DIR [-hosts h1,h2] [-rate 12]")
		os.Exit(2)
	}
	cfg := config{out: *out, rate: *rate, workers: *workers, reserve: *reserve, userAgent: *ua,
		collections: map[string]bool{}, client: &http.Client{Timeout: 5 * time.Minute}}
	for _, c := range strings.Split(*cols, ",") {
		if c != "" {
			cfg.collections[c] = true
		}
	}
	if err := os.MkdirAll(*out, 0o755); err != nil {
		fatal(err)
	}
	byHost, err := readTargets(*in, *hosts)
	if err != nil {
		fatal(err)
	}
	var wg sync.WaitGroup
	for host, dids := range byHost {
		done, err := doneDIDs(*out, host)
		if err != nil {
			fatal(err)
		}
		var todo []string
		for _, d := range dids {
			if !done[d] {
				todo = append(todo, d)
			}
		}
		if *limit > 0 && len(todo) > *limit {
			todo = todo[:*limit]
		}
		logf("host=%s queued=%d already_done=%d", host, len(todo), len(done))
		wg.Add(1)
		go func(host string, todo []string) {
			defer wg.Done()
			if err := runHost(context.Background(), cfg, host, todo); err != nil {
				logf("host=%s stopped: %v", host, err)
			}
		}(host, todo)
	}
	wg.Wait()
}

func readTargets(path, hostFilter string) (map[string][]string, error) {
	allow := map[string]bool{}
	for _, h := range strings.Split(hostFilter, ",") {
		if h != "" {
			allow[h] = true
		}
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	out := map[string][]string{}
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		var t target
		if err := json.Unmarshal(sc.Bytes(), &t); err != nil {
			return nil, err
		}
		// active=0 means the host listed the account inactive; getRepo would refuse it.
		if (t.Status != "failed" && t.Status != "pending") || t.Active != 1 || t.Host == "" || !strings.HasPrefix(t.DID, "did:") {
			continue
		}
		if len(allow) > 0 && !allow[t.Host] {
			continue
		}
		out[t.Host] = append(out[t.Host], t.DID)
	}
	return out, sc.Err()
}

func doneDIDs(dir, host string) (map[string]bool, error) {
	done := map[string]bool{}
	for _, kind := range []string{"snapshots", "failures"} {
		f, err := os.Open(filepath.Join(dir, kind+"-"+host+".jsonl"))
		if errors.Is(err, os.ErrNotExist) {
			continue
		}
		if err != nil {
			return nil, err
		}
		sc := bufio.NewScanner(f)
		for sc.Scan() {
			var row struct {
				DID string `json:"did"`
			}
			if json.Unmarshal(sc.Bytes(), &row) == nil {
				done[row.DID] = true
			}
		}
		f.Close()
	}
	return done, nil
}

// pacer spaces requests to one host and honours its rate-limit headers.
type pacer struct {
	mu       sync.Mutex
	interval time.Duration
	next     time.Time
	reserve  float64
}

func (p *pacer) wait(ctx context.Context) error {
	p.mu.Lock()
	now := time.Now()
	at := p.next
	if at.Before(now) {
		at = now
	}
	p.next = at.Add(p.interval)
	p.mu.Unlock()
	select {
	case <-time.After(time.Until(at)):
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// pauseUntil holds every later request to this host until t.
func (p *pacer) pauseUntil(t time.Time) {
	p.mu.Lock()
	if t.After(p.next) {
		p.next = t
	}
	p.mu.Unlock()
}

// observe pauses the host when the published window is nearly spent.
func (p *pacer) observe(h http.Header) {
	limit, err1 := strconv.Atoi(h.Get("RateLimit-Limit"))
	remaining, err2 := strconv.Atoi(h.Get("RateLimit-Remaining"))
	if err1 != nil || err2 != nil || limit <= 0 {
		return
	}
	if float64(remaining) < p.reserve*float64(limit) {
		p.pauseUntil(resetTime(h, time.Minute))
	}
}

// resetTime reads RateLimit-Reset (epoch seconds) or Retry-After (seconds).
func resetTime(h http.Header, fallback time.Duration) time.Time {
	if v, err := strconv.ParseInt(h.Get("RateLimit-Reset"), 10, 64); err == nil && v > 0 {
		t := time.Unix(v, 0)
		if v < 1e9 { // a relative number of seconds
			t = time.Now().Add(time.Duration(v) * time.Second)
		}
		return t.Add(time.Second)
	}
	if v, err := strconv.Atoi(h.Get("Retry-After")); err == nil && v > 0 {
		return time.Now().Add(time.Duration(v)*time.Second + time.Second)
	}
	return time.Now().Add(fallback)
}

type writers struct {
	mu                    sync.Mutex
	events, snaps, failed *os.File
}

func openWriters(dir, host string) (*writers, error) {
	open := func(kind string) (*os.File, error) {
		return os.OpenFile(filepath.Join(dir, kind+"-"+host+".jsonl"), os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	}
	w := &writers{}
	var err error
	if w.events, err = open("events"); err != nil {
		return nil, err
	}
	if w.snaps, err = open("snapshots"); err != nil {
		return nil, err
	}
	if w.failed, err = open("failures"); err != nil {
		return nil, err
	}
	return w, nil
}

// commit appends one repo's events, then its snapshot row, then syncs, so a
// snapshot row never exists without its events.
func (w *writers) commit(events [][]byte, snapshot []byte) error {
	w.mu.Lock()
	defer w.mu.Unlock()
	for _, e := range events {
		if _, err := w.events.Write(append(e, '\n')); err != nil {
			return err
		}
	}
	if err := w.events.Sync(); err != nil {
		return err
	}
	if _, err := w.snaps.Write(append(snapshot, '\n')); err != nil {
		return err
	}
	return w.snaps.Sync()
}

func (w *writers) fail(did, msg string) error {
	w.mu.Lock()
	defer w.mu.Unlock()
	b, _ := json.Marshal(map[string]string{"did": did, "error": msg, "at": time.Now().UTC().Format(time.RFC3339)})
	_, err := w.failed.Write(append(b, '\n'))
	return err
}

func runHost(ctx context.Context, cfg config, host string, dids []string) error {
	w, err := openWriters(cfg.out, host)
	if err != nil {
		return err
	}
	p := &pacer{interval: time.Duration(float64(time.Second) / cfg.rate), reserve: cfg.reserve}
	jobs := make(chan string)
	var wg sync.WaitGroup
	var mu sync.Mutex
	ok, failed := 0, 0
	start := time.Now()
	for i := 0; i < cfg.workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for did := range jobs {
				events, snap, err := fetchWithRetry(ctx, cfg, p, host, did)
				if err == nil {
					err = w.commit(events, snap)
				} else {
					err = w.fail(did, err.Error())
					mu.Lock()
					failed++
					mu.Unlock()
					if err != nil {
						logf("host=%s write failure: %v", host, err)
					}
					continue
				}
				if err != nil {
					logf("host=%s write: %v", host, err)
					continue
				}
				mu.Lock()
				ok++
				if (ok+failed)%1000 == 0 {
					logf("host=%s done=%d failed=%d rate=%.1f/s", host, ok, failed,
						float64(ok+failed)/time.Since(start).Seconds())
				}
				mu.Unlock()
			}
		}()
	}
	for _, d := range dids {
		jobs <- d
	}
	close(jobs)
	wg.Wait()
	logf("host=%s finished ok=%d failed=%d in %s", host, ok, failed, time.Since(start).Round(time.Second))
	return nil
}

var errRateLimited = errors.New("rate limited")

func fetchWithRetry(ctx context.Context, cfg config, p *pacer, host, did string) ([][]byte, []byte, error) {
	var last error
	for attempt := 0; attempt < 6; attempt++ {
		if err := p.wait(ctx); err != nil {
			return nil, nil, err
		}
		events, snap, err := fetch(ctx, cfg, p, host, did)
		if err == nil {
			return events, snap, nil
		}
		last = err
		var perm permanentError
		if errors.As(err, &perm) {
			return nil, nil, err
		}
		if !errors.Is(err, errRateLimited) {
			time.Sleep(time.Duration(1<<attempt) * time.Second)
		}
	}
	return nil, nil, last
}

type permanentError struct{ msg string }

func (e permanentError) Error() string { return e.msg }

func fetch(ctx context.Context, cfg config, p *pacer, host, did string) ([][]byte, []byte, error) {
	u := "https://" + host + "/xrpc/com.atproto.sync.getRepo?did=" + url.QueryEscape(did)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return nil, nil, err
	}
	req.Header.Set("User-Agent", cfg.userAgent)
	req.Header.Set("Accept", "application/vnd.ipld.car")
	resp, err := cfg.client.Do(req)
	if err != nil {
		return nil, nil, err
	}
	defer resp.Body.Close()
	p.observe(resp.Header)
	switch {
	case resp.StatusCode == http.StatusTooManyRequests:
		p.pauseUntil(resetTime(resp.Header, 5*time.Minute))
		io.Copy(io.Discard, resp.Body)
		return nil, nil, errRateLimited
	case resp.StatusCode == http.StatusBadRequest || resp.StatusCode == http.StatusNotFound:
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return nil, nil, permanentError{fmt.Sprintf("http %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))}
	case resp.StatusCode != http.StatusOK:
		io.Copy(io.Discard, resp.Body)
		return nil, nil, fmt.Errorf("http %d", resp.StatusCode)
	}
	commit, r, err := repo.LoadRepoFromCAR(ctx, resp.Body)
	if err != nil {
		return nil, nil, permanentError{"car: " + err.Error()}
	}
	if commit.DID != did {
		return nil, nil, permanentError{"repo commit is for " + commit.DID}
	}
	now := time.Now().UTC().Format(time.RFC3339Nano)
	events, err := collectEvents(ctx, cfg, r, did, commit.Rev, now)
	if err != nil {
		return nil, nil, permanentError{"records: " + err.Error()}
	}
	snap, _ := json.Marshal(map[string]any{
		"did": did, "status": "complete", "snapshot_rev": commit.Rev, "completed_at": now,
		"latest_rev": commit.Rev, "host": host, "active": 1,
	})
	return events, snap, nil
}

func collectEvents(ctx context.Context, cfg config, r *repo.Repo, did, rev, now string) ([][]byte, error) {
	type entry struct {
		path string
		cid  cid.Cid
	}
	var entries []entry
	if err := r.MST.Walk(func(key []byte, val cid.Cid) error {
		k := string(key)
		if i := strings.IndexByte(k, '/'); i > 0 && cfg.collections[k[:i]] {
			entries = append(entries, entry{k, val})
		}
		return nil
	}); err != nil {
		return nil, err
	}
	sort.Slice(entries, func(a, b int) bool { return entries[a].path < entries[b].path })
	events := make([][]byte, 0, len(entries))
	for _, e := range entries {
		path := e.path
		i := strings.IndexByte(path, '/')
		collection, rkey := path[:i], path[i+1:]
		blk, err := r.RecordStore.Get(ctx, e.cid)
		if err != nil {
			return nil, fmt.Errorf("%s: missing record block", path)
		}
		obj, err := atdata.UnmarshalCBOR(blk.RawData())
		if err != nil {
			continue // undecodable record: skip, as the decoder does
		}
		ev, err := json.Marshal(map[string]any{
			"kind": "commit", "did": did, "time": now,
			"commit": map[string]any{"rev": rev, "operation": "create", "collection": collection,
				"rkey": rkey, "record": slimRecord(obj)},
		})
		if err != nil {
			return nil, err
		}
		events = append(events, ev)
	}
	return events, nil
}

// slimRecord keeps the fields the extractor reads, as plain JSON values.
func slimRecord(obj map[string]any) map[string]any {
	out := map[string]any{}
	for _, k := range keptFields {
		v, ok := obj[k]
		if !ok {
			continue
		}
		switch t := v.(type) {
		case string, bool:
			out[k] = t
		case map[string]any: // strongRef subject
			if uri, ok := t["uri"].(string); ok {
				out[k] = map[string]any{"uri": uri}
			}
		case []any: // starter pack feeds
			var list []any
			for _, item := range t {
				if m, ok := item.(map[string]any); ok {
					if uri, ok := m["uri"].(string); ok {
						list = append(list, map[string]any{"uri": uri})
					}
				}
			}
			out[k] = list
		}
	}
	return out
}

func logf(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "%s "+format+"\n", append([]any{time.Now().UTC().Format(time.RFC3339)}, args...)...)
}

func fatal(err error) {
	logf("fatal: %v", err)
	os.Exit(1)
}

// Command repo_snapshot_export dumps the collector's per-repo backfill state as
// JSON lines, one row per DID, for graph/fold_follows.sql.
//
// The self-hosted Jetstream (v0.4.4) stores a JSON RepoStatus at key repo/<did>
// in <data-dir>/meta.pebble. backfill.rev is the commit rev of the downloaded
// repo snapshot: every record of that download carries the same rev, so it is
// the exact floor below which older records for the DID are superseded.
//
// Run it only against a checkpoint COPY of meta.pebble. Pebble's directory
// lock makes opening the live store fail while the collector holds it.
//
//	repo_snapshot_export -pebble /data/checkpoints/<copy>/meta.pebble -out repo_snapshots.jsonl
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"time"

	"github.com/cockroachdb/pebble"
)

const repoKeyPrefix = "repo/"

// repoStatus mirrors the fields of jetstream's backfill.RepoStatus we need.
type repoStatus struct {
	Backfill struct {
		Status      string    `json:"status"`
		Rev         string    `json:"rev"`
		CompletedAt time.Time `json:"completed_at"`
	} `json:"backfill"`
	Host   string `json:"host"`
	Rev    string `json:"rev"`
	Active bool   `json:"active"`
}

// Row is one output line. snapshot_rev is empty unless status is complete.
type Row struct {
	DID         string `json:"did"`
	Status      string `json:"status"`
	SnapshotRev string `json:"snapshot_rev"`
	CompletedAt string `json:"completed_at"`
	LatestRev   string `json:"latest_rev"`
	Host        string `json:"host"`
	Active      uint8  `json:"active"`
}

func main() {
	dir := flag.String("pebble", "", "path to a checkpoint copy of meta.pebble")
	out := flag.String("out", "-", "output JSONL path, or - for stdout")
	flag.Parse()
	if *dir == "" {
		fmt.Fprintln(os.Stderr, "usage: repo_snapshot_export -pebble <meta.pebble copy> [-out file]")
		os.Exit(2)
	}
	w := io.Writer(os.Stdout)
	if *out != "-" {
		f, err := os.OpenFile(*out, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
		if err != nil {
			fatal(err)
		}
		defer f.Close()
		w = f
	}
	n, err := Export(*dir, w)
	if err != nil {
		fatal(err)
	}
	fmt.Fprintf(os.Stderr, "exported %d repo rows\n", n)
}

// Export writes one JSON line per repo/<did> key and returns the row count.
func Export(dir string, w io.Writer) (int, error) {
	db, err := pebble.Open(dir, &pebble.Options{ReadOnly: true})
	if err != nil {
		return 0, fmt.Errorf("open %s read-only: %w", dir, err)
	}
	defer db.Close()
	it, err := db.NewIter(&pebble.IterOptions{
		LowerBound: []byte(repoKeyPrefix),
		UpperBound: []byte("repo0"), // '0' is the byte after '/'
	})
	if err != nil {
		return 0, err
	}
	defer it.Close()

	bw := bufio.NewWriterSize(w, 1<<20)
	enc := json.NewEncoder(bw)
	n := 0
	for it.First(); it.Valid(); it.Next() {
		var rs repoStatus
		if err := json.Unmarshal(it.Value(), &rs); err != nil {
			return n, fmt.Errorf("decode %s: %w", it.Key(), err)
		}
		row := Row{
			DID:       string(it.Key()[len(repoKeyPrefix):]),
			Status:    rs.Backfill.Status,
			LatestRev: rs.Rev,
			Host:      rs.Host,
		}
		if rs.Active {
			row.Active = 1
		}
		if rs.Backfill.Status == "complete" {
			row.SnapshotRev = rs.Backfill.Rev
			if !rs.Backfill.CompletedAt.IsZero() {
				row.CompletedAt = rs.Backfill.CompletedAt.UTC().Format(time.RFC3339Nano)
			}
		}
		if err := enc.Encode(row); err != nil {
			return n, err
		}
		n++
	}
	if err := it.Error(); err != nil {
		return n, err
	}
	return n, bw.Flush()
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, "repo_snapshot_export:", err)
	os.Exit(1)
}

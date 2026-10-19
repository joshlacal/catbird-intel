#!/bin/bash
# Full-network Jetstream bootstrap probe: run unlimited for RUN1 seconds, stop it
# gracefully mid-crawl, restart, and run RUN2 more seconds. Samples metrics every
# 30s so we can measure repos/s, bytes/s, and whether a restart resumes or starts over.
set -u
RUN1=${RUN1:-600}
RUN2=${RUN2:-300}
IMG=ghcr.io/bluesky-social/jetstream:v0.4.4
OUT=/data/probe
mkdir -p "$OUT"
docker rm -f js >/dev/null 2>&1
rm -rf /data/jetstream && mkdir -p /data/jetstream && chown -R 65532:65532 /data/jetstream

start() {
  docker run -d --name js --restart=no --network=host -v /data/jetstream:/data "$IMG" serve \
    --addr 127.0.0.1:8080 --debug-addr 127.0.0.1:6060 --data-dir /data --log-format text >/dev/null
}

sample() {
  local phase=$1
  local m
  m=$(curl -s 127.0.0.1:6060/metrics)
  local completed failed discovered enumerated
  completed=$(echo "$m" | awk '/^jetstream_backfill_completed_total /{print $2}')
  failed=$(echo "$m" | awk '/^jetstream_backfill_failed_total /{print $2}')
  discovered=$(echo "$m" | awk '/^jetstream_backfill_discovered_total /{print $2}')
  enumerated=$(echo "$m" | awk '/^jetstream_backfill_host_enumerated_repos_total /{print $2}')
  local mush
  mush=$(echo "$m" | grep -E '^jetstream_backfill_hosts_total\{host_class="mushroom"' | sed -E 's/.*state="([a-z_]+)"\} ([0-9]+)/\1=\2/' | tr '\n' ' ')
  local disk rx
  disk=$(du -sb /data/jetstream 2>/dev/null | cut -f1)
  rx=$(awk '/ens4:|eth0:/{print $2}' /proc/net/dev | head -1)
  local cpu
  cpu=$(docker stats --no-stream --format '{{.CPUPerc}} {{.MemUsage}}' js 2>/dev/null)
  echo "$(date -u +%FT%TZ) phase=$phase completed=${completed:-0} failed=${failed:-0} discovered=${discovered:-0} enumerated=${enumerated:-0} disk_bytes=${disk:-0} net_rx_bytes=${rx:-0} cpu_mem=\"$cpu\" mushroom: $mush" >> "$OUT/samples.log"
}

: > "$OUT/samples.log"
start
end=$((SECONDS + RUN1))
while [ $SECONDS -lt $end ]; do sample run1; sleep 30; done
sample before_stop
docker stop -t 30 js >/dev/null
docker logs js > "$OUT/run1.log" 2>&1
docker rm js >/dev/null
sample stopped_offline || true
start
end=$((SECONDS + RUN2))
while [ $SECONDS -lt $end ]; do sample run2; sleep 30; done
docker logs js > "$OUT/run2.log" 2>&1
docker stop -t 30 js >/dev/null
echo "$(date -u +%FT%TZ) PROBE_DONE" >> "$OUT/samples.log"

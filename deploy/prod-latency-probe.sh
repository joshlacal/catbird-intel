#!/usr/bin/env bash
# Production latency probe for the Phase 0 "no change to production" check.
# Read-only: issues GET requests to local health endpoints and a trivial
# ClickHouse query, then prints p50/p99 in milliseconds.
set -euo pipefail
N="${N:-200}"
probe() {
  local name="$1" url="$2"
  for _ in $(seq "$N"); do curl -s -o /dev/null -m 2 -w '%{time_total}\n' "$url"; sleep 0.05; done \
    | sort -n | awk -v n="$name" '{a[NR]=$1} END {printf "%s p50=%.1fms p99=%.1fms n=%d\n", n, a[int(NR*0.5)]*1000, a[int(NR*0.99)]*1000, NR}'
}
echo "ts=$(date -u +%FT%TZ) load=$(cut -d' ' -f1-3 /proc/loadavg) mem_avail=$(awk '/MemAvailable/{printf "%.1fGiB", $2/1048576}' /proc/meminfo) disk_free=$(df -h / | awk 'NR==2{print $4}')"
probe mls-ds:3001/health http://127.0.0.1:3001/health
probe svc:3006/health http://127.0.0.1:3006/health
probe clickhouse:select1 "http://127.0.0.1:8123/?query=SELECT%201"

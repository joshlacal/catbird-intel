#!/usr/bin/env bash
# Deploy catbird-intel to netcup. Additive only: creates/updates the `catbird`
# ClickHouse database and the catbird-intel-* units. Never touches other
# services or the prototype `feed` database.
set -euo pipefail

HOST="${HOST:-netcup}"
DEST="/home/josh/catbird-intel"
HERE="$(cd "$(dirname "$0")/.." && pwd)"

rsync -az --delete \
  --exclude node_modules --exclude .jj --exclude .git --exclude '*.cursor' \
  "$HERE/" "$HOST:$DEST/"

ssh "$HOST" bash -s <<EOF
set -euo pipefail
cd "$DEST"
~/.bun/bin/bun install --production --frozen-lockfile
for f in sql/*.sql; do clickhouse-client --multiquery < "\$f"; done
for unit in deploy/*.service deploy/*.timer; do
  [ -e "\$unit" ] || continue
  sudo -n install -m 0644 "\$unit" /etc/systemd/system/
done
sudo -n systemctl daemon-reload
EOF

echo "Deployed. Start/restart explicitly, e.g.:"
echo "  ssh $HOST sudo systemctl enable --now catbird-intel-ingest"

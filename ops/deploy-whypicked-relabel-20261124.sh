#!/usr/bin/env bash
# Switch the intel API to the relabelled catalog and pool (whyPicked text only), then restart it.
# Run on netcup as josh. The relabelled files were built and checked on 2026-11-24; see
# docs/DECISIONS.md (workspace) for the numbers. Restarting drops in-progress onboarding sessions.
#
# Rollback: sudo rm /etc/systemd/system/catbird-intel-api.service.d/60-whyfix-20261124.conf
#           sudo systemctl daemon-reload && sudo systemctl restart catbird-intel-api.service
set -euo pipefail

D=/var/lib/catbird-intel/onboarding-20261122
A=/etc/systemd/system/catbird-intel-api.service.d
CONF=$A/60-whyfix-20261124.conf

for f in intel_serving_v2.whyfix-20261124.json intel_onboarding_pool_v1.whyfix-20261124.json; do
  [ -s "$D/$f" ] || { echo "missing $D/$f" >&2; exit 1; }
done
sudo chmod 644 "$D"/intel_serving_v2.whyfix-20261124.json "$D"/intel_onboarding_pool_v1.whyfix-20261124.json

sudo tee "$CONF" >/dev/null <<EOF
[Service]
# 2026-11-24: whyPicked relabel (catbird-intel 10af5de8, text only). Rollback: delete this file, daemon-reload, restart.
Environment=INTEL_SERVING_BUNDLE_PATH=$D/intel_serving_v2.whyfix-20261124.json
Environment=INTEL_ONBOARDING_POOL_PATH=$D/intel_onboarding_pool_v1.whyfix-20261124.json
EOF

sudo systemctl daemon-reload
sudo systemctl restart catbird-intel-api.service
for _ in $(seq 60); do curl -sf http://127.0.0.1:3210/health >/dev/null && break; sleep 1; done
systemctl is-active catbird-intel-api.service
curl -s http://127.0.0.1:3210/health | python3 -c '
import json, sys
d = json.load(sys.stdin)
keys = ["s17_catalog_version", "onboarding_engine", "onboarding_pool_version", "onboarding_pool_skipped",
        "onboarding_signals_accounts", "onboarding_signals_error"]
print({k: d.get(k) for k in keys})
'

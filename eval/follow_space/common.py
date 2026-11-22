"""Shared paths and helpers for the follow-space evaluation (read-only on the lake)."""
import os

os.environ.setdefault("OMP_NUM_THREADS", "10")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "10")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "10")

import resource
import sys
import time

HOME = os.path.expanduser("~")
LAKE = os.path.join(HOME, "catbird-lake-local")
S2A = os.path.join(LAKE, "s2-prep-20261028/out/s2a")
EDGES_BIN = os.path.join(S2A, "train_T/edges.bin")  # RowBinary (uid u32, pid u32), ORDER BY did == uid order
PRODUCERS = os.path.join(S2A, "train_T/producers.tsv")  # pid \t did \t followers_at_T
FOLLOWER_WEIGHTS = os.path.join(S2A, "train_T/follower_weights.tsv")  # uid \t did \t is_stale \t weight
TEST_FOLLOWS = os.path.join(S2A, "test/test_follows.parquet")
POOL = os.path.join(LAKE, "s17-picker-v1-20261030/intel_onboarding_pool_v1.candidate-20261122.json")
SERVING = os.path.join(LAKE, "intel-serving-v4")
SCRATCH = os.path.join(LAKE, "ml-follow-space-20261122")
DATA = os.path.join(SCRATCH, "data")
SEED = 20261122
DAY_US = 86_400_000_000
N_THREADS = 10

_T0 = time.time()


def log(*args):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30  # bytes on macOS
    print(f"[{time.time() - _T0:7.1f}s peak={rss:5.1f}G]", *args, flush=True)
    sys.stdout.flush()


def peak_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30

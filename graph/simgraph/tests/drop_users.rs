//! Runs the built binary on tiny files: `--drop-users` with an empty file leaves
//! sim.bin byte-identical, and dropping users (by uid or DID) equals a run on a
//! filtered edges.bin with recounted producers.tsv.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// A fresh directory per call; tests never delete files (workspace rule).
fn temp_dir(name: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let d = std::env::temp_dir().join(format!("simgraph-{name}-{}-{nanos}", std::process::id()));
    fs::create_dir_all(&d).unwrap();
    d
}

/// Users 0..30 follow producers 0..8 by a fixed pattern; user u has DID did:test:u{u}.
fn graph() -> Vec<(u32, u32)> {
    let mut pairs = Vec::new();
    for u in 0..30u32 {
        for p in 0..8u32 {
            if (u * 5 + p * 3) % 4 < 2 {
                pairs.push((u, p));
            }
        }
    }
    pairs
}

fn write_inputs(dir: &Path, pairs: &[(u32, u32)]) {
    let mut deg = [0u32; 8];
    let mut bytes = Vec::new();
    for &(u, p) in pairs {
        deg[p as usize] += 1;
        bytes.extend_from_slice(&u.to_le_bytes());
        bytes.extend_from_slice(&p.to_le_bytes());
    }
    fs::write(dir.join("edges.bin"), bytes).unwrap();
    let producers: String = (0..8)
        .map(|p| format!("{p}\tdid:test:p{p}\t{}\n", deg[p]))
        .collect();
    fs::write(dir.join("producers.tsv"), producers).unwrap();
    let users: String = (0..30)
        .map(|u| format!("{u}\tdid:test:u{u}\t0\t1.0\n"))
        .collect();
    fs::write(dir.join("follower_weights.tsv"), users).unwrap();
}

fn simgraph(dir: &Path, extra: &[&str]) -> (Vec<u8>, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_simgraph"))
        .args([
            "--dir",
            dir.to_str().unwrap(),
            "--cap",
            "3",
            "--min-cos",
            "0.0",
            "--top-k",
            "5",
        ])
        .args(extra)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    (
        fs::read(dir.join("sim.bin")).unwrap(),
        String::from_utf8(out.stdout).unwrap(),
    )
}

#[test]
fn empty_drop_file_is_byte_identical() {
    let d = temp_dir("empty");
    write_inputs(&d, &graph());
    let (base, receipt) = simgraph(&d, &[]);
    assert!(!base.is_empty());
    assert!(!receipt.contains("drop_users"));
    fs::write(d.join("drop.txt"), "# no B1 owners\n\n").unwrap();
    let (same, receipt) = simgraph(&d, &["--drop-users", d.join("drop.txt").to_str().unwrap()]);
    assert_eq!(base, same);
    assert!(receipt.contains("\"drop_users_requested\":0"));
    assert!(receipt.contains("\"dropped_edges\":0"));
}

#[test]
fn dropping_by_uid_or_did_equals_a_filtered_copy() {
    let pairs = graph();
    let drop = [2u32, 9, 10, 29];
    let kept: Vec<_> = pairs
        .iter()
        .copied()
        .filter(|(u, _)| !drop.contains(u))
        .collect();
    let fd = temp_dir("filtered");
    write_inputs(&fd, &kept);
    let (filtered, _) = simgraph(&fd, &[]);

    let d = temp_dir("drop");
    write_inputs(&d, &pairs);
    let (full, _) = simgraph(&d, &[]);
    assert_ne!(full, filtered);
    // Mixed uid and DID lines, a duplicate, and a DID that follows no producer.
    fs::write(
        d.join("drop.txt"),
        "2\ndid:test:u9\ndid:test:u10\n29\n2\ndid:test:nobody\n",
    )
    .unwrap();
    let (dropped, receipt) = simgraph(&d, &["--drop-users", d.join("drop.txt").to_str().unwrap()]);
    assert_eq!(dropped, filtered);
    let n_dropped = pairs.len() - kept.len();
    assert!(receipt.contains("\"drop_users_requested\":6"), "{receipt}");
    assert!(
        receipt.contains("\"drop_users_unresolved_dids\":1"),
        "{receipt}"
    );
    assert!(receipt.contains("\"drop_users_uids\":4"), "{receipt}");
    assert!(
        receipt.contains(&format!("\"dropped_edges\":{n_dropped}")),
        "{receipt}"
    );
}

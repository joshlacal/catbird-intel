//! Producer–producer cosine similarity of follower sets.
//!
//! Inputs (from graph/fold_follows.sql):
//!   producers.tsv  pid \t did \t followers
//!   edges.bin      RowBinary (uid u32, pid u32), sorted by uid
//!
//! For every producer p we sample up to S of its followers and count, for each
//! other producer q, how many sampled followers also follow q. That count
//! scaled by |F_p| / S estimates |F_p ∩ F_q|, and
//!   cosine(p, q) = |F_p ∩ F_q| / sqrt(|F_p| * |F_q|).
//! Users who follow more than C producers contribute a random C of them, and
//! the estimate is reweighted by len/C so heavy followers are not undercounted.
//! We keep, per producer, the top-K neighbours with cosine >= MIN_COS and at
//! least MIN_SUPPORT co-followers in the sample (noise floor).
//!
//! Output: sim.bin RowBinary (p u32, q u32, w f32), directed top-K lists.
//!
//! `--drop-users FILE` removes users' follow edges at load time, as if edges.bin had
//! been filtered and producers.tsv recounted: a dropped user is in no producer's
//! follower sample, adds nothing to any co-follow count, and is subtracted from every
//! producer degree (the cosine denominator and the `--producer-deg-beta` damping).
//! Producer ids stay as in producers.tsv. FILE holds one uid or DID per line (blank
//! lines and `#` comments skipped); DIDs resolve through `<dir>/follower_weights.tsv`
//! (`uid \t did \t ...`), and a DID absent from it follows no producer, so there is
//! nothing to drop. With an empty FILE, sim.bin is byte-identical to a run without the
//! flag. Whole-network plan S3 uses it for the follow-kNN that excludes B1 list
//! owners (plan §5.4 "Bias controls").

use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

struct Args {
    dir: String,
    sample: usize,
    cap: usize,
    min_cos: f32,
    top_k: usize,
    min_support: u32,
    seed: u64,
    follower_weights: Option<String>,
    follower_idf_alpha: f32,
    producer_deg_beta: f32,
    drop_users: Option<String>,
}

fn parse_args() -> Args {
    let mut a = Args {
        dir: ".".into(),
        sample: 1000,
        cap: 500,
        min_cos: 0.02,
        top_k: 400,
        min_support: 2,
        seed: 42,
        follower_weights: None,
        follower_idf_alpha: 0.0,
        producer_deg_beta: 0.0,
        drop_users: None,
    };
    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1;
    let mut use_default_weights = false;
    while i < argv.len() {
        if argv[i].as_str() == "--use-follower-weights" {
            use_default_weights = true;
            i += 1;
            continue;
        }
        let v = argv.get(i + 1).cloned().unwrap_or_default();
        match argv[i].as_str() {
            "--dir" => a.dir = v,
            "--sample" => a.sample = v.parse().expect("--sample"),
            "--cap" => a.cap = v.parse().expect("--cap"),
            "--min-cos" => a.min_cos = v.parse().expect("--min-cos"),
            "--top-k" => a.top_k = v.parse().expect("--top-k"),
            "--min-support" => a.min_support = v.parse().expect("--min-support"),
            "--seed" => a.seed = v.parse().expect("--seed"),
            "--follower-weights" => a.follower_weights = Some(v),
            "--follower-idf-alpha" => {
                a.follower_idf_alpha = v.parse().expect("--follower-idf-alpha")
            }
            "--producer-deg-beta" => a.producer_deg_beta = v.parse().expect("--producer-deg-beta"),
            "--drop-users" => a.drop_users = Some(v),
            other => panic!("unknown argument {other}"),
        }
        i += 2;
    }
    if use_default_weights && a.follower_weights.is_none() {
        a.follower_weights = Some(format!("{}/follower_weights.tsv", a.dir));
    }
    a
}

fn read_producers(path: &str) -> Vec<u32> {
    let f = BufReader::new(File::open(path).expect("producers.tsv"));
    let mut deg = Vec::new();
    for (i, line) in f.lines().enumerate() {
        let line = line.unwrap();
        let mut parts = line.split('\t');
        let pid: usize = parts.next().unwrap().parse().unwrap();
        assert_eq!(pid, i, "producers.tsv must be dense and ordered by pid");
        let _did = parts.next();
        let followers: u32 = parts.next().unwrap().parse().unwrap();
        deg.push(followers);
    }
    deg
}

/// Read optional per-follower coverage weights (`uid \t did \t is_stale \t weight` or `uid \t weight`).
fn read_follower_weights(path: &str, n_users: usize) -> Vec<f32> {
    let f = BufReader::new(File::open(path).expect("follower_weights.tsv"));
    let mut weights = vec![1.0f32; n_users];
    for line in f.lines() {
        let line = line.unwrap();
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with("uid\t") {
            continue;
        }
        let cols: Vec<&str> = trimmed.split('\t').collect();
        assert!(
            cols.len() >= 2,
            "follower_weights.tsv row needs uid and weight"
        );
        let uid: usize = cols[0].parse().expect("follower_weights uid");
        let w: f32 = cols
            .last()
            .unwrap()
            .parse()
            .expect("follower_weights weight");
        if uid < n_users {
            weights[uid] = w.max(0.0);
        }
    }
    weights
}

/// Users to drop: sorted, deduplicated uids, plus how many lines were given and how
/// many DIDs were not in follower_weights.tsv (those follow no producer).
struct DropUsers {
    uids: Vec<u32>,
    requested: usize,
    unresolved_dids: usize,
}

fn read_drop_users(path: &str, users_map: &str) -> DropUsers {
    let f = BufReader::new(File::open(path).expect("--drop-users file"));
    let mut uids: Vec<u32> = Vec::new();
    let mut dids: HashSet<String> = HashSet::new();
    let mut requested = 0usize;
    for line in f.lines() {
        let line = line.unwrap();
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') {
            continue;
        }
        requested += 1;
        if t.bytes().all(|b| b.is_ascii_digit()) {
            uids.push(t.parse().expect("--drop-users uid"));
        } else if t.starts_with("did:") {
            dids.insert(t.to_string());
        } else {
            panic!("--drop-users line is neither a uid nor a DID: {t}");
        }
    }
    let n_dids = dids.len();
    let mut resolved = 0usize;
    if n_dids > 0 {
        let m = BufReader::with_capacity(
            1 << 22,
            File::open(users_map).expect("follower_weights.tsv"),
        );
        for line in m.lines() {
            let line = line.unwrap();
            let mut cols = line.split('\t');
            let (Some(uid), Some(did)) = (cols.next(), cols.next()) else {
                continue;
            };
            if dids.contains(did) {
                uids.push(uid.parse().expect("follower_weights uid"));
                resolved += 1;
            }
        }
    }
    uids.sort_unstable();
    uids.dedup();
    DropUsers {
        uids,
        requested,
        unresolved_dids: n_dids - resolved,
    }
}

/// Read (uid, pid) pairs sorted by uid into a CSR of each user's producers.
/// Rows of users in `drop` (sorted) are skipped and counted per producer in `dropped_in`.
fn read_edges<R: Read>(r: R, drop: &[u32], dropped_in: &mut [u32]) -> (Vec<u64>, Vec<u32>) {
    let mut r = BufReader::with_capacity(1 << 24, r);
    let mut offsets: Vec<u64> = vec![0];
    let mut targets: Vec<u32> = Vec::new();
    let mut buf = [0u8; 8];
    let mut current: i64 = -1;
    let mut skip = false;
    loop {
        match r.read_exact(&mut buf) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => panic!("{e}"),
        }
        let uid = u32::from_le_bytes(buf[0..4].try_into().unwrap()) as i64;
        let pid = u32::from_le_bytes(buf[4..8].try_into().unwrap());
        assert!(uid >= current, "edges.bin must be sorted by uid");
        if uid > current {
            while current < uid {
                if current >= 0 {
                    offsets.push(targets.len() as u64);
                }
                current += 1;
            }
            skip = !drop.is_empty() && drop.binary_search(&(uid as u32)).is_ok();
        }
        if skip {
            dropped_in[pid as usize] += 1;
        } else {
            targets.push(pid);
        }
    }
    offsets.push(targets.len() as u64);
    (offsets, targets)
}

/// Subtract dropped users from producer degrees; producers.tsv counts edges.bin rows.
fn subtract_dropped(deg: &mut [u32], dropped_in: &[u32]) {
    for (p, (d, &x)) in deg.iter_mut().zip(dropped_in).enumerate() {
        *d = d.checked_sub(x).unwrap_or_else(|| {
            panic!("pid {p}: {x} dropped follows exceed producers.tsv followers {d}")
        });
    }
}

/// Write sim.bin rows (p, q, w) and return (edges, producers with neighbours).
fn write_sim<W: Write>(w: W, results: &[Vec<(u32, f32)>]) -> (u64, u64) {
    let mut w = BufWriter::with_capacity(1 << 24, w);
    let mut n_edges = 0u64;
    let mut with_nbrs = 0u64;
    for (p, nbrs) in results.iter().enumerate() {
        if !nbrs.is_empty() {
            with_nbrs += 1;
        }
        for &(q, cos) in nbrs {
            w.write_all(&(p as u32).to_le_bytes()).unwrap();
            w.write_all(&q.to_le_bytes()).unwrap();
            w.write_all(&cos.to_le_bytes()).unwrap();
            n_edges += 1;
        }
    }
    w.flush().unwrap();
    (n_edges, with_nbrs)
}

fn compute_similarities(
    a: &Args,
    deg: &[u32],
    u_off: &[u64],
    u_tgt: &[u32],
    cov_weights: Option<&[f32]>,
    t0: Instant,
) -> Vec<Vec<(u32, f32)>> {
    let n_users = u_off.len() - 1;
    let n_prod = deg.len();

    // Per-user capped follow lists and reweighting factor (len / cap) * coverage_weight * follower_idf.
    let mut rng = SmallRng::seed_from_u64(a.seed);
    let mut c_off: Vec<u64> = Vec::with_capacity(n_users + 1);
    let mut c_tgt: Vec<u32> = Vec::new();
    let mut weight: Vec<f32> = Vec::with_capacity(n_users);
    let mut user_base_w: Vec<f32> = Vec::with_capacity(n_users);
    c_off.push(0);
    for u in 0..n_users {
        let s = &u_tgt[u_off[u] as usize..u_off[u + 1] as usize];
        let cov_w = cov_weights.map_or(1.0, |cw| cw[u]);
        let idf_w = if a.follower_idf_alpha > 0.0 {
            (50.0f32 / (50.0f32 + s.len() as f32)).powf(a.follower_idf_alpha)
        } else {
            1.0f32
        };
        let uw = cov_w * idf_w;
        user_base_w.push(uw);
        if s.len() > a.cap {
            let mut v = s.to_vec();
            v.shuffle(&mut rng);
            v.truncate(a.cap);
            c_tgt.extend_from_slice(&v);
            weight.push((s.len() as f32 / a.cap as f32) * uw);
        } else {
            c_tgt.extend_from_slice(s);
            weight.push(uw);
        }
        c_off.push(c_tgt.len() as u64);
    }

    // Producer -> followers CSR (full lists, from uncapped edges) and effective producer degrees.
    let mut p_count = vec![0u64; n_prod + 1];
    for &p in u_tgt {
        p_count[p as usize + 1] += 1;
    }
    for i in 0..n_prod {
        p_count[i + 1] += p_count[i];
    }
    let p_off = p_count.clone();
    let mut fill = p_count;
    let mut p_fol = vec![0u32; u_tgt.len()];
    let mut eff_deg: Vec<f32> = deg.iter().map(|&d| d as f32).collect();
    if cov_weights.is_some() || a.follower_idf_alpha > 0.0 {
        eff_deg.fill(0.0);
        for u in 0..n_users {
            let uw = user_base_w[u];
            for &p in &u_tgt[u_off[u] as usize..u_off[u + 1] as usize] {
                eff_deg[p as usize] += uw;
            }
        }
    }
    for u in 0..n_users {
        for &p in &u_tgt[u_off[u] as usize..u_off[u + 1] as usize] {
            let slot = &mut fill[p as usize];
            p_fol[*slot as usize] = u as u32;
            *slot += 1;
        }
    }
    eprintln!("indexes built in {:.1}s", t0.elapsed().as_secs_f32());

    let completed = AtomicUsize::new(0);
    (0..n_prod)
        .into_par_iter()
        .map_init(
            || (vec![0f32; n_prod], vec![0u32; n_prod], Vec::<u32>::new()),
            |(acc, support, touched), p| {
                // A producer's sample must be invariant to Rayon scheduling.
                let mut rng = SmallRng::seed_from_u64(
                    a.seed
                        .wrapping_add((p as u64).wrapping_mul(0x9E3779B97F4A7C15)),
                );
                let fol = &p_fol[p_off[p] as usize..p_off[p + 1] as usize];
                if fol.is_empty() {
                    return Vec::new();
                }
                let s = fol.len().min(a.sample);
                let scale = fol.len() as f32 / s as f32;
                // Partial Fisher-Yates over an index window to sample s followers.
                let mut idx: Vec<u32> = if s == fol.len() {
                    fol.to_vec()
                } else {
                    let mut v = fol.to_vec();
                    for i in 0..s {
                        let j = rng.gen_range(i..v.len());
                        v.swap(i, j);
                    }
                    v.truncate(s);
                    v
                };
                idx.sort_unstable();
                for &u in &idx {
                    let u = u as usize;
                    let w = weight[u];
                    for &q in &c_tgt[c_off[u] as usize..c_off[u + 1] as usize] {
                        let qi = q as usize;
                        if qi == p {
                            continue;
                        }
                        if support[qi] == 0 {
                            touched.push(q);
                        }
                        support[qi] += 1;
                        acc[qi] += w;
                    }
                }
                let dp = eff_deg[p];
                let fp = if a.producer_deg_beta > 0.0 {
                    25000.0f32 / (25000.0f32 + (deg[p] as f32 - 25000.0f32).max(0.0))
                } else {
                    1.0f32
                };
                let mut out: Vec<(u32, f32)> = Vec::new();
                for &q in touched.iter() {
                    let qi = q as usize;
                    let dq = eff_deg[qi];
                    if support[qi] >= a.min_support && dp > 0.0 && dq > 0.0 {
                        let inter = (acc[qi] * scale).min(dp.min(dq));
                        let damp = if a.producer_deg_beta > 0.0 {
                            let fq =
                                25000.0f32 / (25000.0f32 + (deg[qi] as f32 - 25000.0f32).max(0.0));
                            (fp * fq).powf(0.5 * a.producer_deg_beta)
                        } else {
                            1.0f32
                        };
                        let cos = (inter / (dp * dq).sqrt()) * damp;
                        if cos >= a.min_cos {
                            out.push((q, cos));
                        }
                    }
                    acc[qi] = 0.0;
                    support[qi] = 0;
                }
                touched.clear();
                if out.len() > a.top_k {
                    out.select_nth_unstable_by(a.top_k, |x, y| {
                        y.1.total_cmp(&x.1).then_with(|| x.0.cmp(&y.0))
                    });
                    out.truncate(a.top_k);
                }
                out.sort_unstable_by(|x, y| y.1.total_cmp(&x.1).then_with(|| x.0.cmp(&y.0)));
                // truncate keeps the pre-cut capacity; over ~2M producers that held ~180 GB.
                out.shrink_to_fit();
                let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
                if done.is_multiple_of(10_000) {
                    eprintln!(
                        "similarities {}/{} producers in {:.1}s",
                        done,
                        n_prod,
                        t0.elapsed().as_secs_f32()
                    );
                }
                out
            },
        )
        .collect()
}

fn main() {
    let a = parse_args();
    let t0 = Instant::now();
    let mut deg = read_producers(&format!("{}/producers.tsv", a.dir));
    let drop = a
        .drop_users
        .as_deref()
        .map(|p| read_drop_users(p, &format!("{}/follower_weights.tsv", a.dir)));
    let mut dropped_in = vec![0u32; deg.len()];
    let (u_off, u_tgt) = read_edges(
        File::open(format!("{}/edges.bin", a.dir)).expect("edges.bin"),
        drop.as_ref().map_or(&[][..], |d| &d.uids[..]),
        &mut dropped_in,
    );
    subtract_dropped(&mut deg, &dropped_in);
    let n_users = u_off.len() - 1;
    let n_prod = deg.len();
    let cov_weights = a
        .follower_weights
        .as_deref()
        .map(|p| read_follower_weights(p, n_users));
    eprintln!(
        "loaded {} users, {} producers, {} edges (weighted={}, idf_alpha={}, deg_beta={}) in {:.1}s",
        n_users,
        n_prod,
        u_tgt.len(),
        cov_weights.is_some(),
        a.follower_idf_alpha,
        a.producer_deg_beta,
        t0.elapsed().as_secs_f32()
    );

    let results = compute_similarities(&a, &deg, &u_off, &u_tgt, cov_weights.as_deref(), t0);

    let (n_edges, with_nbrs) = write_sim(
        File::create(format!("{}/sim.bin", a.dir)).unwrap(),
        &results,
    );
    // The receipt gains the drop fields only when the flag is given.
    let drop_json = drop.as_ref().map_or(String::new(), |d| {
        format!(
            ",\"drop_users_requested\":{},\"drop_users_unresolved_dids\":{},\"drop_users_uids\":{},\"dropped_edges\":{}",
            d.requested,
            d.unresolved_dids,
            d.uids.len(),
            dropped_in.iter().map(|&x| x as u64).sum::<u64>()
        )
    });
    println!(
        "{{\"producers\":{},\"producers_with_neighbors\":{},\"sim_edges\":{},\"users\":{},\"weighted\":{},\"follower_idf_alpha\":{},\"producer_deg_beta\":{}{},\"seconds\":{:.1}}}",
        n_prod,
        with_nbrs,
        n_edges,
        n_users,
        cov_weights.is_some(),
        a.follower_idf_alpha,
        a.producer_deg_beta,
        drop_json,
        t0.elapsed().as_secs_f32()
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unweighted_and_weighted_cosine_similarity() {
        // 3 users:
        // u0 follows p0, p1
        // u1 follows p0, p1
        // u2 follows p0, p2
        let deg = vec![3u32, 2u32, 1u32];
        let u_off = vec![0u64, 2, 4, 6];
        let u_tgt = vec![0u32, 1, 0, 1, 0, 2];
        let args = Args {
            dir: ".".into(),
            sample: 1000,
            cap: 500,
            min_cos: 0.01,
            top_k: 10,
            min_support: 2,
            seed: 42,
            follower_weights: None,
            follower_idf_alpha: 0.0,
            producer_deg_beta: 0.0,
            drop_users: None,
        };
        let unweighted = compute_similarities(&args, &deg, &u_off, &u_tgt, None, Instant::now());
        // p0 and p1 share u0 and u1 (2 co-followers); cos = 2 / sqrt(3 * 2) = 0.8164966
        assert_eq!(unweighted[0].len(), 1);
        assert_eq!(unweighted[0][0].0, 1);
        assert!((unweighted[0][0].1 - (2.0f32 / 6.0f32.sqrt())).abs() < 1e-5);

        // Downweight u2 (stale fallback weight 0.25) while keeping u0, u1 at 1.0:
        // Effective degree of p0 becomes 2.25, p1 stays 2.0, intersection stays 2.0 -> cos increases!
        let cw = vec![1.0f32, 1.0, 0.25];
        let weighted = compute_similarities(&args, &deg, &u_off, &u_tgt, Some(&cw), Instant::now());
        assert_eq!(weighted[0][0].0, 1);
        let expected_weighted = 2.0f32 / (2.25f32 * 2.0f32).sqrt();
        assert!((weighted[0][0].1 - expected_weighted).abs() < 1e-5);
        assert!(weighted[0][0].1 > unweighted[0][0].1);

        // Follower IDF downweights indiscriminate mass followers while keeping numerator and
        // denominator on the exact same scale; producer_deg_beta damps >25k mega-hub pairs.
        let deg_hub = vec![100_000u32, 100_000u32, 1u32];
        let args_pop = Args {
            dir: ".".into(),
            sample: 1000,
            cap: 500,
            min_cos: 0.01,
            top_k: 10,
            min_support: 2,
            seed: 42,
            follower_weights: None,
            follower_idf_alpha: 0.5,
            producer_deg_beta: 0.5,
            drop_users: None,
        };
        let pop_res =
            compute_similarities(&args_pop, &deg_hub, &u_off, &u_tgt, None, Instant::now());
        assert_eq!(pop_res[0][0].0, 1);
        assert!(pop_res[0][0].1 < unweighted[0][0].1);
    }

    fn args(cap: usize) -> Args {
        Args {
            dir: ".".into(),
            sample: 1000,
            cap,
            min_cos: 0.01,
            top_k: 10,
            min_support: 2,
            seed: 42,
            follower_weights: None,
            follower_idf_alpha: 0.0,
            producer_deg_beta: 0.0,
            drop_users: None,
        }
    }

    fn edge_bytes(pairs: &[(u32, u32)]) -> Vec<u8> {
        pairs
            .iter()
            .flat_map(|&(u, p)| u.to_le_bytes().into_iter().chain(p.to_le_bytes()))
            .collect()
    }

    /// Full in-memory pipeline: edges -> drop -> degrees -> similarities -> sim.bin bytes.
    fn run(
        pairs: &[(u32, u32)],
        deg: &[u32],
        drop: &[u32],
        a: &Args,
        cw: Option<&[f32]>,
    ) -> Vec<u8> {
        let mut deg = deg.to_vec();
        let mut dropped_in = vec![0u32; deg.len()];
        let (off, tgt) = read_edges(&edge_bytes(pairs)[..], drop, &mut dropped_in);
        subtract_dropped(&mut deg, &dropped_in);
        let res = compute_similarities(a, &deg, &off, &tgt, cw, Instant::now());
        let mut out = Vec::new();
        write_sim(&mut out, &res);
        out
    }

    fn rows(bytes: &[u8]) -> Vec<(u32, u32, f32)> {
        bytes
            .chunks(12)
            .map(|c| {
                (
                    u32::from_le_bytes(c[0..4].try_into().unwrap()),
                    u32::from_le_bytes(c[4..8].try_into().unwrap()),
                    f32::from_le_bytes(c[8..12].try_into().unwrap()),
                )
            })
            .collect()
    }

    // u0: p0 p1 | u1: p0 p1 | u2: p0 p2 | u3: p0 p2. Degrees p0=4, p1=2, p2=2.
    const PAIRS: [(u32, u32); 8] = [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 2),
        (3, 0),
        (3, 2),
    ];
    const DEG: [u32; 3] = [4, 2, 2];

    #[test]
    fn dropped_rows_are_skipped_and_counted_per_producer() {
        let mut dropped_in = vec![0u32; 3];
        let (off, tgt) = read_edges(&edge_bytes(&PAIRS)[..], &[1, 3], &mut dropped_in);
        assert_eq!(off, vec![0, 2, 2, 4, 4]);
        assert_eq!(tgt, vec![0, 1, 0, 2]);
        assert_eq!(dropped_in, vec![2, 1, 1]);
        let mut deg = DEG.to_vec();
        subtract_dropped(&mut deg, &dropped_in);
        assert_eq!(deg, vec![2, 1, 1]);
    }

    #[test]
    fn dropped_user_contributes_nothing_hand_computed() {
        let a = args(500);
        // No drop: p0-p1 and p0-p2 each have 2 co-followers; cos = 2 / sqrt(4 * 2).
        let full = rows(&run(&PAIRS, &DEG, &[], &a, None));
        let c = 2.0f32 / 8.0f32.sqrt();
        assert_eq!(full.len(), 4);
        for &(p, q, w) in &full {
            assert!(matches!((p, q), (0, 1) | (0, 2) | (1, 0) | (2, 0)));
            assert!((w - c).abs() < 1e-6);
        }
        // Drop u3: p0's degree is 3 and p2's is 1; p0-p2 keeps 1 co-follower, below
        // min-support 2, so it disappears; p0-p1 = 2 / sqrt(3 * 2).
        let dropped = rows(&run(&PAIRS, &DEG, &[3], &a, None));
        let c01 = 2.0f32 / 6.0f32.sqrt();
        assert_eq!(dropped.len(), 2);
        assert_eq!((dropped[0].0, dropped[0].1), (0, 1));
        assert_eq!((dropped[1].0, dropped[1].1), (1, 0));
        assert!((dropped[0].2 - c01).abs() < 1e-6 && (dropped[1].2 - c01).abs() < 1e-6);
        // With follower weights the dropped user's weight never enters a degree either.
        let cw = [1.0f32, 1.0, 1.0, 7.0];
        let weighted = rows(&run(&PAIRS, &DEG, &[3], &a, Some(&cw)));
        assert_eq!(rows(&run(&PAIRS, &DEG, &[3], &a, None)), weighted);
    }

    #[test]
    fn empty_drop_list_is_byte_identical() {
        for cap in [2usize, 500] {
            let a = args(cap);
            let mut nodrop_in = vec![0u32; 3];
            let (off, tgt) = read_edges(&edge_bytes(&PAIRS)[..], &[], &mut nodrop_in);
            let res = compute_similarities(&a, &DEG, &off, &tgt, None, Instant::now());
            let mut base = Vec::new();
            write_sim(&mut base, &res);
            assert!(!base.is_empty());
            assert_eq!(run(&PAIRS, &DEG, &[], &a, None), base);
        }
    }

    #[test]
    fn drop_equals_a_filtered_copy_with_recounted_degrees() {
        // A larger graph with heavy followers so the cap shuffle consumes the RNG.
        let mut pairs = Vec::new();
        for u in 0..40u32 {
            for p in 0..12u32 {
                if (u * 7 + p * 3) % 5 < 3 {
                    pairs.push((u, p));
                }
            }
        }
        let drop = [3u32, 4, 17, 39];
        let mut deg = vec![0u32; 12];
        let mut kept = Vec::new();
        let mut kept_deg = vec![0u32; 12];
        for &(u, p) in &pairs {
            deg[p as usize] += 1;
            if !drop.contains(&u) {
                kept.push((u, p));
                kept_deg[p as usize] += 1;
            }
        }
        for cap in [2usize, 500] {
            let a = args(cap);
            let filtered = run(&kept, &kept_deg, &[], &a, None);
            assert!(!filtered.is_empty());
            assert_eq!(run(&pairs, &deg, &drop, &a, None), filtered);
            assert_ne!(run(&pairs, &deg, &[], &a, None), filtered);
        }
    }
}

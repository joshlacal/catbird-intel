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

use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
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

/// Read (uid, pid) pairs sorted by uid into a CSR of each user's producers.
fn read_edges(path: &str) -> (Vec<u64>, Vec<u32>) {
    let mut r = BufReader::with_capacity(1 << 24, File::open(path).expect("edges.bin"));
    let mut offsets: Vec<u64> = vec![0];
    let mut targets: Vec<u32> = Vec::new();
    let mut buf = [0u8; 8];
    let mut current: i64 = -1;
    loop {
        match r.read_exact(&mut buf) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => panic!("{e}"),
        }
        let uid = u32::from_le_bytes(buf[0..4].try_into().unwrap()) as i64;
        let pid = u32::from_le_bytes(buf[4..8].try_into().unwrap());
        assert!(uid >= current, "edges.bin must be sorted by uid");
        while current < uid {
            if current >= 0 {
                offsets.push(targets.len() as u64);
            }
            current += 1;
        }
        targets.push(pid);
    }
    offsets.push(targets.len() as u64);
    (offsets, targets)
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
    let deg = read_producers(&format!("{}/producers.tsv", a.dir));
    let (u_off, u_tgt) = read_edges(&format!("{}/edges.bin", a.dir));
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

    let mut w =
        BufWriter::with_capacity(1 << 24, File::create(format!("{}/sim.bin", a.dir)).unwrap());
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
    println!(
        "{{\"producers\":{},\"producers_with_neighbors\":{},\"sim_edges\":{},\"users\":{},\"weighted\":{},\"follower_idf_alpha\":{},\"producer_deg_beta\":{},\"seconds\":{:.1}}}",
        n_prod,
        with_nbrs,
        n_edges,
        n_users,
        cov_weights.is_some(),
        a.follower_idf_alpha,
        a.producer_deg_beta,
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
        };
        let pop_res =
            compute_similarities(&args_pop, &deg_hub, &u_off, &u_tgt, None, Instant::now());
        assert_eq!(pop_res[0][0].0, 1);
        assert!(pop_res[0][0].1 < unweighted[0][0].1);
    }
}

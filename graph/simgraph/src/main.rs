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
    };
    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < argv.len() {
        let v = argv.get(i + 1).cloned().unwrap_or_default();
        match argv[i].as_str() {
            "--dir" => a.dir = v,
            "--sample" => a.sample = v.parse().expect("--sample"),
            "--cap" => a.cap = v.parse().expect("--cap"),
            "--min-cos" => a.min_cos = v.parse().expect("--min-cos"),
            "--top-k" => a.top_k = v.parse().expect("--top-k"),
            "--min-support" => a.min_support = v.parse().expect("--min-support"),
            "--seed" => a.seed = v.parse().expect("--seed"),
            other => panic!("unknown argument {other}"),
        }
        i += 2;
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

fn main() {
    let a = parse_args();
    let t0 = Instant::now();
    let deg = read_producers(&format!("{}/producers.tsv", a.dir));
    let (u_off, u_tgt) = read_edges(&format!("{}/edges.bin", a.dir));
    let n_users = u_off.len() - 1;
    let n_prod = deg.len();
    eprintln!(
        "loaded {} users, {} producers, {} edges in {:.1}s",
        n_users,
        n_prod,
        u_tgt.len(),
        t0.elapsed().as_secs_f32()
    );

    // Per-user capped follow lists and reweighting factor len / cap.
    let mut rng = SmallRng::seed_from_u64(a.seed);
    let mut c_off: Vec<u64> = Vec::with_capacity(n_users + 1);
    let mut c_tgt: Vec<u32> = Vec::new();
    let mut weight: Vec<f32> = Vec::with_capacity(n_users);
    c_off.push(0);
    for u in 0..n_users {
        let s = &u_tgt[u_off[u] as usize..u_off[u + 1] as usize];
        if s.len() > a.cap {
            let mut v = s.to_vec();
            v.shuffle(&mut rng);
            v.truncate(a.cap);
            c_tgt.extend_from_slice(&v);
            weight.push(s.len() as f32 / a.cap as f32);
        } else {
            c_tgt.extend_from_slice(s);
            weight.push(1.0);
        }
        c_off.push(c_tgt.len() as u64);
    }

    // Producer -> followers CSR (full lists, from uncapped edges).
    let mut p_count = vec![0u64; n_prod + 1];
    for &p in &u_tgt {
        p_count[p as usize + 1] += 1;
    }
    for i in 0..n_prod {
        p_count[i + 1] += p_count[i];
    }
    let p_off = p_count.clone();
    let mut fill = p_count;
    let mut p_fol = vec![0u32; u_tgt.len()];
    for u in 0..n_users {
        for &p in &u_tgt[u_off[u] as usize..u_off[u + 1] as usize] {
            let slot = &mut fill[p as usize];
            p_fol[*slot as usize] = u as u32;
            *slot += 1;
        }
    }
    drop(u_tgt);
    drop(u_off);
    eprintln!("indexes built in {:.1}s", t0.elapsed().as_secs_f32());

    let completed = AtomicUsize::new(0);
    let results: Vec<Vec<(u32, f32)>> = (0..n_prod)
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
                let dp = deg[p] as f32;
                let mut out: Vec<(u32, f32)> = Vec::new();
                for &q in touched.iter() {
                    let qi = q as usize;
                    if support[qi] >= a.min_support {
                        let inter = (acc[qi] * scale).min(dp.min(deg[qi] as f32));
                        let cos = inter / (dp * deg[qi] as f32).sqrt();
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
        .collect();

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
        "{{\"producers\":{},\"producers_with_neighbors\":{},\"sim_edges\":{},\"users\":{},\"seconds\":{:.1}}}",
        n_prod, with_nbrs, n_edges, n_users, t0.elapsed().as_secs_f32()
    );
}

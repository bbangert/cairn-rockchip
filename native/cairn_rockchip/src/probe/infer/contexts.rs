//! `--contexts N` above 1: whether N contexts of one model give more passes
//! than one. The first sample's slot is read once. On the calling thread,
//! every context runs one untimed pass of it, whose outputs it hashes, and
//! then context 0 alone runs `--samples` timed passes: the baseline, measured
//! as the threads are. Then one thread per context runs `--samples` passes,
//! all let go together behind a barrier. A pass is `rknn_inputs_set`,
//! `rknn_run`, `rknn_outputs_get` and the release. `ctx_total`'s `out` is
//! context 0's hash, and a context's `agree` whether its own matches it.
//!
//! `ctx` per thread and `ctx_solo` for the baseline: `n` passes, `pps` them
//! over its own span, `p50`/`p95` of `rknn_run` in ms. In `ctx_total`:
//! `wall_s` from the first thread's start to the last one's end, `pps` all
//! passes over it, and `gain` that over `ctx_solo`'s `pps` — about 1 if N
//! contexts give no more passes than one, up to N if they scale fully. A
//! thread's own span cannot tell those apart: one blocked in `rknn_run`
//! while another context holds the NPU is as busy as one running. `ctx_rss`
//! is `VmRSS` before the contexts opened, after, and after the run.
//!
//! The lines are printed as soon as the run ends, before any context is
//! destroyed: a destroy order the runtime does not survive must not take the
//! measurement with it. Each is under 100 bytes, which the board's ssh
//! endpoint needs.

use std::sync::Barrier;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context as _, Result};
use cairn_detect::infer::{InputSize, InputTensor, TensorLayout, TensorValues};
use serde_json::{json, Value};

use super::super::scale::{fnv1a, hex, FNV_OFFSET};
use crate::decoder::ms;
use crate::rknn::{ContextMode, DestroyOrder, RknnBackend};

pub(super) const MAX_CONTEXTS: usize = 8;

pub(super) const MODES: [(&str, bool); 2] = [("load", false), ("dup", true)];
pub(super) const DESTROY_ORDERS: [(&str, DestroyOrder); 2] = [
    ("dup-first", DestroyOrder::DupFirst),
    ("parent-first", DestroyOrder::ParentFirst),
];

/// `--context-mode` and `--destroy-order` as printed.
pub(super) fn mode_names(mode: ContextMode) -> (&'static str, Option<&'static str>) {
    match mode {
        ContextMode::Load => ("load", None),
        ContextMode::Dup(order) => ("dup", Some(super::name(&DESTROY_ORDERS, order))),
    }
}

/// `VmRSS` in kB, if the kernel reports it.
pub(super) fn rss_kb() -> Option<u64> {
    let text = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = text.lines().find_map(|line| line.strip_prefix("VmRSS:"))?;
    line.split_whitespace().next()?.parse().ok()
}

/// One context's timed passes.
struct Worker {
    started: Instant,
    ended: Instant,
    run_ms: Vec<f64>,
    out: u64,
}

/// One pass of `tensor`, handed back for the next; the outputs' hash when
/// `hash` names them.
fn pass(
    backend: &mut RknnBackend,
    tensor: InputTensor,
    hash: Option<&[String]>,
) -> Result<(InputTensor, Duration, u64)> {
    backend.set_input(tensor)?;
    let handed = Instant::now();
    let tensor = backend.infer()?.context("rknn_run kept no input")?;
    let ran = Instant::now() - handed;
    let outputs = backend.outputs()?;
    let mut out = FNV_OFFSET;
    for name in hash.unwrap_or_default() {
        for v in outputs.get(name)?.values {
            out = fnv1a(out, &v.to_le_bytes());
        }
    }
    Ok((tensor, ran, out))
}

/// `samples` passes; `out` is the warm pass's hash.
fn timed(
    backend: &mut RknnBackend,
    mut tensor: InputTensor,
    samples: u64,
    out: u64,
) -> Result<(InputTensor, Worker)> {
    let started = Instant::now();
    let mut run_ms = Vec::new();
    for _ in 0..samples {
        let ran;
        (tensor, ran, _) = pass(backend, tensor, None)?;
        run_ms.push(ms(ran));
    }
    let worker = Worker {
        started,
        ended: Instant::now(),
        run_ms,
        out,
    };
    Ok((tensor, worker))
}

/// The sample at `p` of `sorted`.
fn at(sorted: &[f64], p: f64) -> f64 {
    sorted[((sorted.len() - 1) as f64 * p).round() as usize]
}

fn round(x: f64, places: i32) -> f64 {
    let scale = 10f64.powi(places);
    (x * scale).round() / scale
}

pub(super) struct Run<'a> {
    pub(super) mode: ContextMode,
    pub(super) samples: u64,
    pub(super) outputs: &'a [String],
    /// `VmRSS` before the contexts opened and after.
    pub(super) rss: [Option<u64>; 2],
}

impl Run<'_> {
    /// Runs every context in `backends` on its own thread over `bytes`, and
    /// prints the lines.
    pub(super) fn run(
        &self,
        backends: &mut [RknnBackend],
        size: InputSize,
        bytes: &[u8],
    ) -> Result<()> {
        // The warm passes and the baseline before any thread exists: a
        // failure or panic in them cannot leave a thread on the barrier.
        let mut warm = Vec::with_capacity(backends.len());
        for (i, backend) in backends.iter_mut().enumerate() {
            let tensor =
                InputTensor::new(size, TensorLayout::Nhwc, TensorValues::U8(bytes.to_vec()));
            let (tensor, _, out) = pass(backend, tensor, Some(self.outputs))
                .with_context(|| format!("context {i}'s warm pass"))?;
            warm.push((tensor, out));
        }
        let first = backends.first_mut().context("no contexts")?;
        let (tensor, out) = warm.remove(0);
        let (tensor, solo) = timed(first, tensor, self.samples, out).context("context 0 alone")?;
        warm.insert(0, (tensor, out));
        let start = Barrier::new(backends.len());
        let results: Vec<_> = thread::scope(|scope| {
            let handles: Vec<_> = backends
                .iter_mut()
                .zip(warm)
                .map(|(backend, (tensor, out))| {
                    let start = &start;
                    // Each worker holds the only `&mut` to its context for
                    // the run, which needs `RknnBackend: Send` and nothing
                    // more — `cairn_detect`'s `Backend: Send` holds the type
                    // to it, nothing is shared, and no `unsafe` is added
                    // here. Whether librknnrt runs a context on a thread
                    // other than the one that made it is what this finds out.
                    scope.spawn(move || {
                        start.wait();
                        Ok(timed(backend, tensor, self.samples, out)?.1)
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| {
                    h.join()
                        .unwrap_or_else(|_| anyhow::bail!("a context's thread panicked"))
                })
                .collect()
        });
        let workers = results
            .into_iter()
            .enumerate()
            .map(|(i, r)| r.with_context(|| format!("context {i}")))
            .collect::<Result<Vec<_>>>()?;
        for line in self.lines(solo, workers, rss_kb()) {
            println!("{line}");
        }
        Ok(())
    }

    fn lines(
        &self,
        mut solo: Worker,
        mut workers: Vec<Worker>,
        rss_end: Option<u64>,
    ) -> Vec<Value> {
        let first = workers.iter().map(|w| w.started).min();
        let last = workers.iter().map(|w| w.ended).max();
        let wall = first.zip(last).map_or(Duration::ZERO, |(a, b)| b - a);
        let passes = self.samples * workers.len() as u64;
        let per_s = |n: u64, span: Duration| n as f64 / span.as_secs_f64().max(1e-9);
        let (mode, order) = mode_names(self.mode);
        let mut lines = vec![json!({
            "event": "ctx_run", "mode": mode, "order": order, "contexts": workers.len(),
            "passes": passes,
        })];
        let timing = |event: &str, i: Option<usize>, w: &mut Worker| {
            w.run_ms.sort_by(f64::total_cmp);
            let mut line = json!({
                "event": event, "n": w.run_ms.len(),
                "pps": round(per_s(w.run_ms.len() as u64, w.ended - w.started), 2),
                "p50": at(&w.run_ms, 0.5), "p95": at(&w.run_ms, 0.95),
            });
            if let Some(i) = i {
                line["i"] = i.into();
            }
            line
        };
        let solo_line = timing("ctx_solo", None, &mut solo);
        let out = workers.first().map(|w| w.out);
        for (i, w) in workers.iter_mut().enumerate() {
            let mut line = timing("ctx", Some(i), w);
            line["agree"] = (out == Some(w.out)).into();
            lines.push(line);
        }
        lines.push(solo_line);
        let total = per_s(passes, wall);
        let alone = per_s(solo.run_ms.len() as u64, solo.ended - solo.started);
        lines.push(json!({
            "event": "ctx_total", "wall_s": round(wall.as_secs_f64(), 3),
            "pps": round(total, 2), "gain": round(total / alone, 3), "out": hex(out),
        }));
        let [before, open] = self.rss;
        lines.push(json!({
            "event": "ctx_rss", "start_kb": before, "open_kb": open, "end_kb": rss_end,
        }));
        lines
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every value wider than a board run's, so the 100-byte bound holds.
    #[test]
    fn every_line_is_under_100_bytes() {
        let now = Instant::now();
        let worker = |span_ms| Worker {
            started: now,
            ended: now + Duration::from_millis(span_ms),
            run_ms: vec![123_456.789; 1_000_000],
            out: u64::MAX,
        };
        let run = Run {
            mode: ContextMode::Dup(DestroyOrder::ParentFirst),
            samples: 1_000_000,
            outputs: &[],
            rss: [Some(99_999_999), Some(99_999_999)],
        };
        // One fast (100 000 passes/s), the rest slow and long; the solo run
        // slower still, for the widest gain.
        let workers = (0..MAX_CONTEXTS as u64)
            .map(|i| worker(if i == 0 { 10_001 } else { 99_999_999 }))
            .collect();
        for line in run.lines(worker(999_999_999_999), workers, Some(99_999_999)) {
            let text = line.to_string();
            assert!(text.len() < 100, "{} bytes: {text}", text.len());
        }
    }

    #[test]
    fn gain_is_the_total_rate_over_one_context_alone() {
        let now = Instant::now();
        let worker = |from: u64, to: u64| Worker {
            started: now + Duration::from_secs(from),
            ended: now + Duration::from_secs(to),
            run_ms: vec![1.0, 2.0, 3.0, 4.0],
            out: 1,
        };
        let run = Run {
            mode: ContextMode::Load,
            samples: 4,
            outputs: &[],
            rss: [None, None],
        };
        let mut other = worker(1, 4);
        other.out = 2;
        let lines = run.lines(worker(0, 3), vec![worker(0, 2), other], None);
        let events: Vec<_> = lines.iter().map(|l| l["event"].clone()).collect();
        assert_eq!(
            events,
            ["ctx_run", "ctx", "ctx", "ctx_solo", "ctx_total", "ctx_rss"]
        );
        assert_eq!(lines[0]["order"], Value::Null);
        assert_eq!(lines[1]["p50"], 3.0);
        assert_eq!(
            (&lines[1]["agree"], &lines[2]["agree"]),
            (&true.into(), &false.into())
        );
        assert_eq!(lines[1]["pps"], 2.0);
        let solo = &lines[3];
        assert_eq!(
            (solo["n"].as_u64(), solo["pps"].as_f64()),
            (Some(4), Some(1.33))
        );
        assert_eq!(solo.get("i"), None);
        let total = &lines[4];
        assert_eq!(
            (total["wall_s"].as_f64(), total["pps"].as_f64()),
            (Some(4.0), Some(2.0))
        );
        assert_eq!(total["gain"], 1.5);
    }

    /// Over the stub runtime: the warm passes and context 0's solo run on
    /// the calling thread, then each context's timed passes on a thread of
    /// its own, and none waits for ever.
    #[test]
    fn every_context_runs_its_passes_on_its_own_thread() {
        use crate::rknn::tests::{open_contexts, run_threads, stubs, FIRST_DUP, PARENT};

        let _guard = stubs();
        let mode = ContextMode::Dup(DestroyOrder::DupFirst);
        let mut contexts = open_contexts("probe", 3, mode).unwrap();
        let samples = 5;
        let run = Run {
            mode,
            samples,
            outputs: &["output".to_string()],
            rss: [None, None],
        };
        let size = InputSize { w: 6, h: 4 };
        run.run(contexts.all(), size, &[0; 72]).unwrap();
        let runs = run_threads();
        let here = thread::current().id();
        let mut workers = Vec::new();
        for (ctx, on_caller) in [(PARENT, 1 + samples), (FIRST_DUP, 1), (FIRST_DUP + 1, 1)] {
            let threads: Vec<_> = runs.iter().filter(|r| r.0 == ctx).map(|r| r.1).collect();
            let (before, timed) = threads.split_at(on_caller as usize);
            assert!(before.iter().all(|&t| t == here), "context {ctx}: {runs:?}");
            assert_eq!(timed.len() as u64, samples, "context {ctx}: {runs:?}");
            assert!(
                timed.iter().all(|&t| t == timed[0] && t != here),
                "{runs:?}"
            );
            workers.push(timed[0]);
        }
        let distinct: std::collections::HashSet<_> = workers.iter().collect();
        assert_eq!(distinct.len(), 3, "{runs:?}");
    }
}

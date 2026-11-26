//! `--contexts N` above 1: whether N contexts of one model run passes at the
//! same time. The first sample's slot is read once; then one thread per
//! context runs `--samples` passes of it, each `rknn_inputs_set`,
//! `rknn_run`, `rknn_outputs_get` and the release, all threads let go
//! together behind a barrier. Before the barrier each runs one untimed pass,
//! whose outputs it hashes; `ctx_total`'s `out` is the first context's, and
//! a thread's `agree` whether its own matches it.
//!
//! Per thread: `n` passes, `pps` them over its own span, `p50`/`p95` of
//! `rknn_run` in ms. In total: `wall_s` from the first thread's start to the
//! last one's end, `pps` all passes over it, and `overlap` every thread's
//! time inside its passes over `wall_s` — about 1 if the driver ran one
//! context at a time, about N if they ran fully overlapped. `ctx_rss` is
//! `VmRSS` before the contexts opened, after, and after the run.
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

struct Worker {
    started: Instant,
    ended: Instant,
    /// Inside the passes, from `rknn_inputs_set` to the outputs' release.
    busy: Duration,
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

fn work(
    backend: &mut RknnBackend,
    tensor: InputTensor,
    outputs: &[String],
    samples: u64,
    start: &Barrier,
) -> Result<Worker> {
    let warm = pass(backend, tensor, Some(outputs));
    // Reached failed or not, or the others would wait on it for ever.
    start.wait();
    let (mut tensor, _, out) = warm?;
    let started = Instant::now();
    let (mut busy, mut run_ms) = (Duration::ZERO, Vec::new());
    for _ in 0..samples {
        let began = Instant::now();
        let ran;
        (tensor, ran, _) = pass(backend, tensor, None)?;
        busy += began.elapsed();
        run_ms.push(ms(ran));
    }
    Ok(Worker {
        started,
        ended: Instant::now(),
        busy,
        run_ms,
        out,
    })
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
        let start = Barrier::new(backends.len());
        let results: Vec<_> = thread::scope(|scope| {
            let handles: Vec<_> = backends
                .iter_mut()
                .map(|backend| {
                    let tensor = InputTensor::new(
                        size,
                        TensorLayout::Nhwc,
                        TensorValues::U8(bytes.to_vec()),
                    );
                    let start = &start;
                    // Each worker holds the only `&mut` to its context for
                    // the run, which needs `RknnBackend: Send` and nothing
                    // more — `cairn_detect`'s `Backend: Send` holds the type
                    // to it, nothing is shared, and no `unsafe` is added
                    // here. Whether librknnrt runs a context on a thread
                    // other than the one that made it is what this finds out.
                    scope.spawn(move || work(backend, tensor, self.outputs, self.samples, start))
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
        for line in self.lines(workers, rss_kb()) {
            println!("{line}");
        }
        Ok(())
    }

    fn lines(&self, mut workers: Vec<Worker>, rss_end: Option<u64>) -> Vec<Value> {
        let first = workers.iter().map(|w| w.started).min();
        let last = workers.iter().map(|w| w.ended).max();
        let wall = first.zip(last).map_or(Duration::ZERO, |(a, b)| b - a);
        let busy: Duration = workers.iter().map(|w| w.busy).sum();
        let passes = self.samples * workers.len() as u64;
        let per_s = |n: u64, span: Duration| round(n as f64 / span.as_secs_f64().max(1e-9), 2);
        let (mode, order) = mode_names(self.mode);
        let mut lines = vec![json!({
            "event": "ctx_run", "mode": mode, "order": order, "contexts": workers.len(),
            "passes": passes,
        })];
        let out = workers.first().map(|w| w.out);
        for (i, w) in workers.iter_mut().enumerate() {
            w.run_ms.sort_by(f64::total_cmp);
            lines.push(json!({
                "event": "ctx", "i": i, "n": w.run_ms.len(),
                "pps": per_s(w.run_ms.len() as u64, w.ended - w.started),
                "p50": at(&w.run_ms, 0.5), "p95": at(&w.run_ms, 0.95),
                "agree": out == Some(w.out),
            }));
        }
        lines.push(json!({
            "event": "ctx_total", "wall_s": round(wall.as_secs_f64(), 3),
            "pps": per_s(passes, wall),
            "overlap": round(busy.as_secs_f64() / wall.as_secs_f64().max(1e-9), 3),
            "out": hex(out),
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
            busy: Duration::from_millis(span_ms),
            run_ms: vec![123_456.789; 1_000_000],
            out: u64::MAX,
        };
        let run = Run {
            mode: ContextMode::Dup(DestroyOrder::ParentFirst),
            samples: 1_000_000,
            outputs: &[],
            rss: [Some(99_999_999), Some(99_999_999)],
        };
        // One fast (100 000 passes/s), the rest slow and long.
        let workers = (0..MAX_CONTEXTS as u64)
            .map(|i| worker(if i == 0 { 10_001 } else { 99_999_999 }))
            .collect();
        for line in run.lines(workers, Some(99_999_999)) {
            let text = line.to_string();
            assert!(text.len() < 100, "{} bytes: {text}", text.len());
        }
    }

    #[test]
    fn overlap_is_busy_time_over_the_wall() {
        let now = Instant::now();
        let worker = |from: u64, to: u64| Worker {
            started: now + Duration::from_secs(from),
            ended: now + Duration::from_secs(to),
            busy: Duration::from_secs(to - from),
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
        let lines = run.lines(vec![worker(0, 2), other], None);
        assert_eq!(lines[0]["order"], Value::Null);
        assert_eq!(lines[1]["p50"], 3.0);
        assert_eq!(
            (&lines[1]["agree"], &lines[2]["agree"]),
            (&true.into(), &false.into())
        );
        assert_eq!(lines[1]["pps"], 2.0);
        let total = &lines[3];
        assert_eq!(
            (total["wall_s"].as_f64(), total["pps"].as_f64()),
            (Some(4.0), Some(2.0))
        );
        assert_eq!(total["overlap"], 1.25);
    }

    /// Over the stub runtime: every context runs its warm-up pass and its
    /// own `samples` on its own thread, and none waits for ever.
    #[test]
    fn every_context_runs_its_passes_on_its_own_thread() {
        use crate::rknn::tests::{called_on, open_contexts, stubs, FIRST_DUP, PARENT};

        let _guard = stubs();
        let mode = ContextMode::Dup(DestroyOrder::DupFirst);
        let mut contexts = open_contexts("probe", 3, mode).unwrap();
        let run = Run {
            mode,
            samples: 5,
            outputs: &["output".to_string()],
            rss: [None, None],
        };
        let size = InputSize { w: 6, h: 4 };
        run.run(contexts.all(), size, &[0; 72]).unwrap();
        let runs = called_on("rknn_run");
        for ctx in [PARENT, FIRST_DUP, FIRST_DUP + 1] {
            let n = runs.iter().filter(|&&c| c == ctx).count();
            assert_eq!(n, 6, "context {ctx}: {runs:?}");
        }
    }
}

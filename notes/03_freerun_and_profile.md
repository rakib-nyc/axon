# AXON — F8: free-run diagnostic + real per-kernel profile

Config: nand=4, target=64, k=24, dthr=1, mode=1, inbr=4, stimcols=64, T=20, cue=10 steps.
Cue at 75%, released at step 10, averaged over 12 stimuli. Trained = 40 epochs.

## F8.1 — the blocking check: k/k+1 margin

| | trained | untrained |
|---|---|---|
| mean margin (cue phase) | 1.5 – 2.9 | 0.5 – 0.9 |
| mean margin (free phase) | 1.8 – 3.9 | 0.8 – 0.9 |
| **% columns with margin == 0** | **30 – 38%** | **38 – 45%** |
| % columns with k-th drive == 0 | ~0 (falls to 0.0 in free phase) | 0.0 |
| mean k-th drive (free phase) | 50 → 83 | 42 |

**Not the "margin ≈ 0" catastrophe, but not clean either.** There is real recurrent
drive in the free phase — the k-th winner always has nonzero drive, and mean margin
is 2–4 against a drive of 50–83. But **a third of all columns resolve the k/(k+1)
boundary by exact tie, i.e. by lane index.** Training improves this (margin 0.8 → 2–4,
ties 41% → 33%) but does not remove it.

Consequence for contrastive learning: ~1/3 of free-phase winner selections are
arbitrary, and a naive free-phase decrement would carve those into the weights.
**Fix is free on this hardware:** the rank loop in `kwta_ballot_margin` already
computes the k-th and (k+1)-th drive (2 `simd_max`), so plasticity can be gated on
`margin > 0` at zero cost. This is a better fix than a decaying cue because it is
targeted — it suppresses exactly the arbitrary decisions rather than softening all of them.

## F8.2 — free-run trajectory (STEP 2 metric)

Overlap of the free-running state with the trained assembly, per step after release:

| step | phase | trained ov(assembly) | untrained ov(assembly) |
|---|---|---|---|
| 9 | last cue | 0.185 | 0.025 |
| 10 | FREE | 0.166 | 0.017 |
| 12 | FREE | 0.302 | 0.016 |
| 14 | FREE | 0.471 | 0.022 |
| 16 | FREE | 0.593 | 0.014 |
| 18 | FREE | **0.657** | 0.021 |
| 19 | FREE | 0.634 | 0.026 |

**The trajectory rises, and the control is flat.** The untrained network has no
attractor at all: released state wanders, overlap with the cue-end state collapses
to 0.021, overlap with any reference stays at chance. Training builds a genuine
attractor that the free dynamics climb into over ~8 steps.

## F8.3 — the negative result that matters

| | trained | untrained |
|---|---|---|
| cross-talk between assemblies (T=20, with release) | **0.624** | 0.016 |
| pattern completion, 75% cue: self | 0.651 | 0.013 |
| pattern completion, 75% cue: **best-other** | **0.731** | 0.048 |
| margin (self − best-other) | **−0.081** | −0.035 |

**Training builds an attractor. It builds exactly one.** All 12 stimuli converge to
substantially the same free-run state; best-other exceeds self at every partial cue.
The rise in F8.2 is the network falling into the single dominant attractor, which the
reference assembly (itself a free-run endpoint) also occupies.

Note this is invisible under the old protocol: with the cue clamped for all T steps,
cross-talk is 0.003. The attractor collapse only appears once the input is released.

This is the failure mode the review predicted for free-running Hebb — and it is
already present in weights trained under full clamp, before any free-phase rule was
added. It is also precisely what contrastive divergence is built to fix: the free
state is stimulus-*independent*, so a free-phase decrement would specifically punish
the shared attractor. The diagnostic supports proceeding, with margin gating.

## F8.4 — per-kernel GPU profile

M1 Pro exposes exactly **one counter set: `timestamp`, one counter (`GPUTimestamp`)**.
`MTLCounterSamplingPointAtDispatchBoundary` is **not supported**; stage boundary is.
So per-kernel timing requires one encoder per kernel. There are no ALU/occupancy/
memory-stall counters available through the public API on this device.

Two methods, one discarded:

- **Marginal/slope (discarded, invalid):** dispatching a kernel R extra times gives
  negative costs for `k_emit` and `k_bias` and marginals summing to 171 µs against a
  569 µs step. Repeated back-to-back dispatches run hot in cache, so the slope
  underestimates badly. Reported here only so the method is not retried.
- **Cumulative ablation in the production encoder (used):**

| kernel added | µs/step | delta |
|---|---|---|
| k_drive only | 228.5 | +228.5 |
| + k_topk | 294.7 | **+66.2** |
| + k_emit | 303.0 | +8.3 |
| + k_bias | 331.5 | +28.5 |
| + k_learn (full) | 503.4 | **+171.9** |

Per-encoder timestamps agree on ordering: drive 272.4 µs (50.9%), learn 126.7 (23.7%),
topk 92.9 (17.3%), emit 32.1 (6.0%), bias 11.5 (2.2%).

**Answer to "is the plasticity pass the bottleneck": no, but it is not small.**
`k_learn` is **34–37% of the step** by ablation (24% by timestamp). `k_drive` is the
bottleneck at 45–51%. The expectation was directionally right; "not the bottleneck"
understates it at roughly a third of the step.

**Unasked-for finding: `k_topk` costs 66–93 µs, 13–17% of the engine.** An exact
top-64 selection over 4096 numbers costs more than emit and bias combined, because it
runs on ONE threadgroup (256 threads — 1/8 of the reliably co-resident capacity from
F3) doing three barrier-separated passes with a serial 256-bin scan in thread 0.
Parallelising it across 16 threadgroups with a merged histogram is a
straightforward ~13% engine-wide win and touches no model semantics.

Throughput at this config: **0.3795 ms/step = 2,635 steps/s (2.6x biological real time)**
in production; 0.503 ms/step in the ablation harness (learning always on, no skipping).

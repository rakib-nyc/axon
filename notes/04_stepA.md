# AXON — STEP A: the collapse mechanism, identified

Config: nand=4 target=64 k=24 dthr=1 mode=1 inbr=4 stimcols=64 T=20 cue=10, 40 epochs.
Throughput: **2,252 steps/s trained, 2,263 untrained** (0.442 ms/step).

## A1.1 — hub domination is FALSIFIED

Functional in-degree = popcount(MSB plane) over the 28 recurrent branches, 114,688 neurons.

| | trained | untrained |
|---|---|---|
| mean | 672.4 | 448.9 |
| sd / **cv** | 100.9 / **0.150** | 22.7 / **0.051** |
| p50 | 688 | 448 |
| p99 | 862 | 509 |
| max | 1036 | 700 |
| **p99/p50** | **1.25** | 1.14 |

**There is no heavy tail.** The distribution is near-Gaussian with cv 0.15. No hubs exist.

## A1.2 — the discriminating test, and it points the other way

Winner identity at the free-run endpoint, per group, trained:

| group | n | in-degree | **intrinsic bias** |
|---|---|---|---|
| wins in ≥11/12 stimuli | 467 | 499.2 | **555.2** |
| wins in any stimulus | 4,347 | 460.2 | **129.0** |
| never wins | 110,341 | **680.8** | **3,357.8** |
| (clamped-assembly neurons) | 18,440 | 697.9 | **3,439.3** |

**HUB RATIO = 0.733 — inverted.** The neurons that win across all stimuli have *lower*
in-degree than those that never win. Collapse is not hub domination.

**The separation is in the intrinsic bias: 129 vs 3,358, a 26x gap.**

## The mechanism

Bias enters as `acc -= bias >> biasShift`, biasShift=4. So:

- clamped-assembly neuron: penalty 3439/16 = **215**
- free-run winner: penalty 129/16 = **8**
- free-phase drive at the k-th winner: **50 – 83**

**The neurons holding the memory are suppressed 3–4x below their own signal.** They
cannot fire once the cue is removed. What wins instead is whatever has near-zero bias —
i.e. neurons that never fired during training — and "lowest bias" is stimulus-independent,
which is exactly why all 12 stimuli land in the same state.

Root cause is that `k_bias` is a **non-leaky integrator**: `bias += 73 if fired else -1`.
Its equilibrium (biasUp/biasDown = (1-p)/p, p = 1.35%) is only correct for Poisson firing
at the target rate. Assembly neurons fire in **bursts** — all 20 steps of their own
stimulus, 1 presentation in 12 — gaining +730 per presentation and decaying only ~220
across the other eleven. Bias diverges monotonically for exactly the neurons that
encode something.

Untrained control confirms: free-run Jaccard = **1.000** (all 12 stimuli give the
identical state, complete collapse, hub ratio 1.173 — the *untrained* net does pick
best-connected neurons). Training moves it 1.000 → 0.401. Learning helps; recall is broken.

## A2 — the reference was degenerate, and correcting it INVERTS the previous result

| measure | CLAMPED ref (correct) | FREE ref (previously used) |
|---|---|---|
| cross-talk, trained | **0.000** | 0.624 |
| cross-talk, untrained | 0.062 | 1.000 |
| assembly pairwise Jaccard, trained | **0.000** (max 0.005) | 0.401 |
| 100% cue: self / other / margin | 0.013 / 0.047 / **−0.034** | 1.000 / 0.695 / +0.305 |
| 75% cue | 0.014 / 0.047 / −0.033 | 0.651 / 0.731 / −0.081 |
| 50% cue | 0.014 / 0.046 / −0.031 | 0.647 / 0.751 / −0.104 |

Free-run trajectory against the CLAMPED reference **declines**: 0.139 (release) → 0.079 →
0.043 → 0.026 → 0.020 → 0.014 → 0.016 by step 19. Against the FREE reference it "rises"
0.166 → 0.657.

### Retraction

**The "rising trajectory / genuine attractor" result in the previous report is withdrawn.**
It was an artifact of defining the reference as a free-run endpoint. Measured against the
clamped state, the released network retains **1.3–1.4% overlap with the trained assembly at
every cue level, including a 100% cue**, and moves monotonically away from it. There is no
attractor. The rise was convergence onto the shared low-bias population.

## What this does to STEP C

The stated prediction was: "if A1 shows a heavy in-degree tail, C2 does more work than C1."
**A1 shows no tail (cv 0.15, p99/p50 = 1.25) and an inverted hub ratio.** C2's
heterosynaptic in-degree normalisation targets a failure mode that is not present.

More importantly, both C1 and C2 would run on a substrate where memory neurons are drive-
suppressed 3–4x below signal. Any contrastive result would be measuring the bias pathology,
not the learning rule.

**Prerequisite fix:** make the bias a leaky integrator (multiplicative decay, `bias -= bias/tau`)
so equilibrium is `tau * biasUp * rate` — bounded and on the scale of drive — instead of
diverging. Note the bias cannot simply be removed: ablating it earlier gave cross-talk
0.45–0.78. It is what creates the orthogonality (clamped Jaccard 0.000). It needs bounding,
not deletion.

STEP B (parallelise k_topk) is independent of all this and still worth taking.

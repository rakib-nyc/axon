# AXON — STEP A3 / A4

Throughput: **2,448 steps/s trained, 2,198 untrained** (0.408 / 0.455 ms/step).

## A3 — the orthogonality is allocation, not learning

Pool = 114,688 recurrent neurons (3584 columns x 32), 12 assemblies of m = 1538.

    E[|A n B|] = m^2/n = 20.63 neurons
    E[Jaccard] = E/(2m - E) = 0.00675          <-- CHANCE

| | trained | untrained |
|---|---|---|
| measured mean pairwise Jaccard | **0.00008** (max 0.00524) | 0.03462 (max 0.610) |
| ratio to chance | **0.012 (85x BELOW)** | 5.12 (above) |
| total pairwise overlap | **16** neurons vs 1361 at chance | 5806 vs 1365 |
| sum of sizes / union / excess | 18456 / 18440 / **16** | 18480 / 14606 / 3874 |
| verdict | **NEAR-EXACT PARTITION** | overlapping |

Your estimate of ~0.006 for chance was right (0.00675). Measured is **85x below** chance,
not at it. The assemblies are not independently drawn and not merely orthogonal — they
are a partition, with 16 shared neurons out of 18,456 assigned.

**This is an allocator.** The mechanism is the bias veto exactly as described: a neuron
fires, its bias saturates, it is permanently vetoed, and the next stimulus is forced onto
untouched neurons. Zero cross-talk was bookkeeping, not associative learning. The
untrained control confirms the contrast — without training the same protocol gives
5.1x *more* overlap than chance and a union far smaller than the sum.

## A4 — zero bias at readout: RECALL APPEARS

Trained weights unchanged; bias ignored at test only (`P.useBias = 0`).

    clamped assembly preserved when bias zeroed : overlap 0.217
    cross-talk of clamped assemblies, bias OFF  : 0.000  (bias ON: 0.000)

### Release trajectory, 75% cue — same weights, only the readout veto differs

| step | phase | ov(clampRef) bias **ON** | ov(clampRef) bias **OFF** |
|---|---|---|---|
| 9 | last cue | 0.103 | 0.235 |
| 10 | FREE | 0.129 | 0.224 |
| 12 | FREE | 0.049 | 0.234 |
| 14 | FREE | 0.016 | 0.233 |
| 16 | FREE | 0.016 | 0.230 |
| 19 | FREE | **0.014** | **0.224** |

Bias ON decays monotonically to nothing. Bias OFF is **flat across all ten free steps**
(0.224 at release, 0.224 at step 19). That is the attractor signature, produced with no
retraining whatsoever.

### Pattern completion, bias OFF

| cue | trained self | trained other | **trained margin** | untrained self | untrained other | **untrained margin** |
|---|---|---|---|---|---|---|
| 100% | 0.208 | 0.000 | **+0.208** | 0.233 | 0.980 | **−0.748** |
| 75% | 0.215 | 0.000 | **+0.215** | 0.233 | 0.980 | −0.748 |
| 50% | 0.219 | 0.003 | +0.216 | 0.233 | 0.980 | −0.748 |
| 35% | 0.243 | 0.001 | **+0.242** | 0.233 | 0.980 | −0.748 |
| 25% | 0.210 | 0.000 | +0.210 | 0.233 | 0.980 | −0.748 |
| 15% | 0.226 | 0.000 | **+0.226** | 0.233 | 0.980 | −0.748 |
| **0% (no cue)** | **0.000** | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 |

Against the no-bias clamped reference the trained margins are larger still: +0.306, +0.296,
+0.333, +0.318, +0.319, +0.314.

## Verdict on A4's decision

**"Recall appears" — the first branch.** Connectivity contains a recallable memory and the
bias was a pure readout veto. The leaky fix (A5) is confirmed on target.

**Success criterion met for the first time:** self > best-other against the CLAMPED
reference, +0.208 to +0.242, with the untrained control at −0.748 — opposite sign, not
merely smaller.

### Honest limits

1. **Recall is partial: 0.21–0.24** (0.30–0.33 against the no-bias reference). Roughly a
   quarter to a third of each assembly is recovered and held. Stable, not complete.
2. **No basin edge down to a 15% cue.** Completion is flat from 100% to 15% (~10 of 64
   stimulus columns) and exactly 0.000 with no cue. The cue selects the basin; the recalled
   core is cue-independent once selected. Finding the knee needs finer resolution below 15%.
3. The untrained `self = 0.233` is meaningless — it is overlap with a collapsed state shared
   by all stimuli (other = 0.980). Only the margin is interpretable there.
4. A3 stands: the *code* being recalled was allocated by the veto, not learned as orthogonal.
   What A4 shows is that the recurrent weights genuinely store and retrieve that allocated
   code. Allocation and association are separate results here.

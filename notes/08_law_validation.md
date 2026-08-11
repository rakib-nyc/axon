# AXON — capacity law validation, T sweep, plateau probe

## Spikes measured directly (the unmeasured assumption)

Firing steps per presentation, restricted to **final-state assembly members** (the
neurons that actually get allocated; transient one-shot firers must not be averaged in —
16,582 neurons fire at least once against an assembly of 1,536):

| T (clamp+free) | members | mean firing steps | clamped | free | p50 | fraction of T |
|---|---|---|---|---|---|---|
| 20 (10+10) | 1536 | 1.08 | **0.00** | 1.08 | 1 | 0.054 |
| 10 (5+5) | 1536 | 1.30 | 0.12 | 1.17 | 1 | 0.130 |
| 6 (3+3) | 1536 | 1.24 | 0.12 | 1.12 | 1 | 0.207 |

**Constant across a 3.3x change in T ⇒ `biasUp/T` confirmed as the organising variable.**

Two corrections this forces:
1. My original `spikes ≈ 10` was wrong by ~10x. The wall formula's *prefactor* is therefore
   empirical, not derived — the scaling `∝ biasUp/T` holds, the absolute constant does not
   follow from a single-presentation balance argument (which would predict a wall at ~5).
2. **At T=20 the clamped contribution is exactly 0.00.** The trained assembly is a
   **free-phase object** — its members are not the neurons the cue drives. This is
   consistent with A1 (free-run winners had bias 129 / in-degree 460; clamped-assembly
   neurons had bias 3439 / in-degree 698) and explains why the readout gate works and why
   contrastive, which acts in the free phase, mattered so much.

## T sweep (stim=12, 40 epochs)

| T+clamp | steps/s | wdens | Jaccard | r75 self | r75 other | margin | LTD/LTP | freeOv(end) |
|---|---|---|---|---|---|---|---|---|
| 20+10 | 2495 | 7.04 | 0.543 | 0.591 | 0.010 | +0.581 | 0.631 | 0.604 |
| **10+5** | 2217 | 7.52 | **0.580** | **0.746** | 0.042 | **+0.704** | 0.995 | 0.747 |
| 8+4 | 2076 | 7.65 | 0.499 | 0.668 | 0.092 | +0.576 | 1.016 | 0.681 |
| 6+3 | 2334 | 6.85 | 0.182 | 0.310 | 0.193 | +0.117 | **1.420** | **0.257** |
| 4+2 | 3132 | 6.29 | 0.069 | 0.309 | 0.147 | +0.162 | 1.118 | 0.315 |
| untrained (T=10) | 4539 | 6.26 | 0.019 | 0.052 | 0.081 | −0.029 | — | — |

**Both settling signatures fire together at T=6**: LTD/LTP jumps 1.016 → 1.420 and
free/clamped overlap collapses 0.681 → 0.257. **The floor is between T=8 and T=6** —
higher than estimated; T=6 is already below it rather than near it.

**T=10 is the optimum.** Compute per presentation: T=20 → 8.0 ms, T=10 → 4.5 ms (1.8x less),
T=8 → 3.9 ms.

## Plateau probe: it does NOT hold

Default C (73/1/T=10), 40 epochs:

| stim | steps/s | wdens | Jaccard | r75 self | **r75 other** | margin | union | % pool |
|---|---|---|---|---|---|---|---|---|
| 90 | 2156 | 9.34 | 0.169 | 0.421 | 0.182 | +0.239 | 72,714 | 63.4 |
| 140 | 2166 | 11.68 | 0.100 | 0.347 | 0.178 | **+0.169** | 94,480 | 82.4 |
| 200 | 2035 | 13.02 | 0.062 | 0.237 | 0.186 | +0.051 | 100,208 | 87.4 |
| 300 | 2525 | 14.16 | 0.037 | 0.137 | 0.179 | **−0.041** | 103,547 | **90.3** |
| untrained 90 | 4457 | — | 0.013 | 0.027 | 0.096 | −0.069 | | |
| untrained 300 | 4689 | — | 0.013 | 0.026 | 0.106 | −0.079 | | |

Zero-crossing at **~230 stimuli**. Usable capacity **~140** (margin +0.169, recall 0.347);
~200 is marginal (+0.051). Against baseline's ~40, that is a **3.5–5x increase**.

### The failure mode is not interference

**best-other is pinned at 0.178–0.186 across 90 → 300 while self decays 0.421 → 0.137.**
Capacity fails by *progressive loss of the stored trace*, not by growing confusion between
stimuli. Willshaw/Hopfield fail the opposite way — cross-talk rises. This is a third
distinct signature, matching the structured-reuse picture rather than interference.

That flatness also argues against an input-space confound: with 300 stimuli of 64 columns
drawn from 512, expected pairwise input overlap is 8/64 = 12.5%, but if collisions drove the
failure best-other would *rise* with NSTIM. It does not.

### Exhaustion regime confirmed

Union saturates at **103,547 = 90.3% of the 114,688 pool** (72,714 → 94,480 → 100,208 →
103,547), with reuse reaching 357,325 of 460,872 assigned (77.5%). At strong veto the
mechanism is genuinely exhaustion-limited, as distinct from the veto-timescale limit at
weak veto.

## Preserved

Two qualitatively different capacity mechanisms in one architecture, selected by one
parameter: **veto-timescale-limited at weak veto, exhaustion-limited at strong veto.**
Roll-off is structured **reuse**, not Willshaw interference — assemblies stay exactly 1536,
allocation never fails, overlap reaches 1.78x chance. Anchor: 12 stimuli, 18,432 assigned,
18,432 distinct, pairwise Jaccard 0.00000 across all 66 pairs — exact partition.

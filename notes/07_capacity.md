# AXON — capacity (default config: C1b 0.5x LTD, margin gate OFF)

## Resolution of the ltdrep anomaly (prerequisite)

The ep-40 near-identity (0.857 vs 0.867) was a **sampling artifact**. LTD/LTP is not a
stable characteristic — it decays through training in every config:

| epoch | 0.5x | 1x | 2x |
|---|---|---|---|
| 1 | **0.693** | **0.924** | **1.527** |
| 10 | 0.785 | 1.067 | 1.553 |
| 25 | 0.785 | 0.750 | 1.027 |
| 40 | 0.631 | 0.537 | 0.680 |

At epoch 1 the three are cleanly separated and monotonic in LTD volume, as designed. The
ratio decays because the free-phase co-active population shrinks as the free state converges
toward the clamped one — so a late-epoch sample compresses the configs onto each other.

Equilibrium densities differ genuinely (120 epochs): 0.5x → 6.50%, 1x → 5.35%, 2x → ~3.5%.
Neither 0.5x nor 1x is uniquely "converged" (both still drift ~0.003–0.006%/epoch at ep 120);
the difference is equilibrium density, not update volume. **0.5x stands as a real optimum.**

## Capacity curve

| stim | steps/s | wdens | Jaccard | core | r75 self | r75 other | **margin** | sum \| union \| excess | reuse |
|---|---|---|---|---|---|---|---|---|---|
| 12 | 3015 | 7.04 | 0.541 | 27.3% | 0.599 | 0.010 | **+0.589** | 18432 \| 18432 \| **0** | **0.0%** |
| 24 | 2843 | 6.17 | 0.488 | 31.0% | 0.696 | 0.113 | **+0.583** | 36864 \| 36258 \| 606 | 1.6% |
| 30 | 2582 | 5.98 | 0.317 | 17.1% | 0.490 | 0.210 | +0.280 | 46104 \| 44876 \| 1228 | 2.7% |
| 36 | 2518 | 5.84 | 0.258 | 11.3% | 0.408 | 0.242 | +0.165 | 55296 \| 51274 \| 4022 | 7.3% |
| 42 | 2551 | 5.88 | 0.142 | 3.5% | 0.213 | 0.292 | **−0.079** | 64512 \| 53795 \| 10717 | 16.6% |
| 48 | 2654 | 5.89 | 0.140 | 2.9% | 0.163 | 0.281 | −0.117 | 73752 \| 52818 \| 20934 | 28.4% |
| 60 | 2909 | 6.25 | 0.075 | 0.9% | 0.150 | 0.253 | −0.103 | 92160 \| 58773 \| 33387 | 36.2% |
| 74 | 2973 | 6.55 | 0.071 | 1.4% | 0.145 | 0.212 | −0.067 | 113688 \| 62061 \| 51627 | 45.4% |
| 90 | 2833 | 6.65 | 0.052 | 0.4% | 0.083 | 0.194 | −0.111 | 138240 \| 66103 \| 72137 | 52.2% |

Untrained controls (12 / 48 / 90): Jaccard 0.008 / 0.007 / 0.008, recall 0.009 / 0.015 /
0.015, best-other 0.043 / 0.067 / 0.075, assembly Jaccard 1.13x / 1.06x / 1.07x chance,
4966–5018 steps/s. **Flat in stimulus count** — the curve above is a property of learning.

## Shape: plateau, then a fast roll-off — not a cliff, not graceful decline

Margin is **flat to 24** (+0.589 → +0.583), then rolls off steeply but continuously over a
narrow band (24 → 42), crossing zero at **~40 stimuli**, then floors beyond 48. This is
neither the predicted cliff at pool exhaustion nor Willshaw-style graceful degradation.

**Useful capacity ≈ 24 stimuli (36,864 neurons, 32% of pool); zero-margin at ~40.**

Assembly orthogonality crosses chance in the same band: measured pairwise assembly Jaccard
0.00000 (12) → 0.00074 (24) → 0.00937 (48) against chance 0.00674, i.e. ratio
**0.000 → 0.110 → 1.389 → 1.776 (90)**. Allocation goes from *perfectly disjoint* to
*worse than random*.

**At 12 stimuli the partition is exact: 18,432 assigned, 18,432 distinct, excess 0, pairwise
Jaccard 0.00000 across all 66 pairs.**

## Mechanism at the wall: reuse, not failure to allocate

- Assembly size is **1536 at every point** (sum/nstim = 1536 exactly, 12 through 90).
  Allocation never fails to fill an assembly.
- Union grows sublinearly and **saturates near 66k = 58% of the 114,688 pool** while sum
  grows to 138k. **The pool is never exhausted.**
- Reuse rises 0% → 1.6% → 7.3% → 28.4% → 52.2%.

So the allocator **reuses neurons**, and the reuse is *structured* (the same low-bias
neurons get picked repeatedly), which is why overlap ends up above chance rather than at it.

## Why ~40 and not ~74

The ~74 extrapolation assumed the veto is permanent. It is not — bias decays at
`biasDown = 1` per step, so a neuron not firing loses `(NSTIM−1)·T` per epoch while its own
stimulus restores `biasUp · (firing steps)`. The veto holds while

    biasUp · spikes  >=  biasDown · (NSTIM − 1) · T

With biasUp=73, T=20, and ~10 firing steps (the clamped phase only — the assembly partially
collapses during the free phase), this gives **NSTIM ≈ 37.5**. Measured zero-crossing is
**~40**, within 7%.

**Capacity is set by the veto timescale, not by the pool size.** That predicts capacity
scales linearly with biasUp/biasDown, which is directly testable and would be the cheapest
way to move the wall.

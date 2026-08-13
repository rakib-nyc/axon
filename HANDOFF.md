# HANDOFF — read this first

State as of 2026-08-12. Written so a new session can resume without re-deriving anything.

---

## 1. Status in one paragraph

The measurement phase is **closed**. The preprint is written, compiled, deposited
(DOI `10.5281/zenodo.21881612`), and this repository is published at
<https://github.com/rakib-nyc/axon>, tagged `v1.0`. The code has been verified from a
fresh clone: it compiles unmodified and reproduces published numbers exactly. **Nothing
is half-finished.** If you are picking this up, you are starting new work, not
completing old work.

## 2. Where everything is

| Path | What |
|---|---|
| `/Users/rakib/Documents/axon-repo` | **The repository.** Work here. Git remote `origin` → github.com/rakib-nyc/axon |
| `/Users/rakib/Documents/axon` | Old working directory. **Do not use.** Contents of uncertain provenance after a destructive merge (see §8) |
| `/Users/rakib/Documents/axon-premerge-backup.tar.gz` | Read-only (mode 444) snapshot of that directory, 306,872 bytes |
| `/Users/rakib/Downloads/Preprint.pdf` | User's compiled PDF, md5 `807c3eadd528c1efccfb2807696df2aa` — same file as `paper/Preprint.pdf` |
| `/Users/rakib/Downloads/axon_preprint.tex` | User's LaTeX source, same as `paper/axon_preprint.tex` |

## 3. The finding

Sparse binary assembly network: 131,072 neurons (4096 columns × 32), 32 dendritic
branches × 256 presynaptic each → 8192 fan-in, 1.07 G binary synapses. Synapses are
bounded 4-bit counters stored as 4 bit-planes; **the functional weight is the MSB only**,
so propagation is `popcount(W3 & spikes)`. Readout is exact *k*-of-32 winners-take-all
within a column plus a top-64 column gate, so the active set is pinned at 1536 neurons
by construction.

Across 12 → 60 stored patterns (5 seeds/load) the sustained-recall ratio falls 3.7×
while **within-assembly functional density stays at 80.7–82.2%** — and stays flat
separately for exclusive members (82.7→81.2%) and shared members (69.3→82.1%, rising)
while shared fraction goes 4.8%→64.4%. Retrieval *initiation* is load-invariant (release
transient 0.694–0.728 SD≤0.022; initial drive 116–120). At mid load loop gain sits at
unity (0.977–1.016) while overlap still decays 0.720→0.520. The failure is **identity
drift** into a structured neighbourhood: visited-but-non-member neurons carry
within-assembly input at 59.68% vs an 8.62% population baseline (6.9× enrichment,
untrained 1.00×), and load shifts that population from unallocated assembly-adjacent
neurons (multiplicity 0.12 vs 0.16 population) to neurons in competing assemblies
(1.53 vs 0.80). Capacity is a pair: **sustained ≈36** (soft, overlapping per-seed
distributions), **transient ≈200**.

## 4. Configuration C — used for every table in the paper

```
nand=4 target=64 k=24 dthr=1 mode=1 inbr=4 stimcols=64 contrast=1 margingate=0 \
ltdrep=1 ltdevery=2 biasup=73 biasdown=1 T=10 clampsteps=5
```

**The binary's built-in defaults are NOT config C** (they are `target=256 k=6 dthr=2
inbr=8 stimcols=128 biasup=0 contrast=0 margingate=1 epochs=30`). Always pass C
explicitly. Every `bench/*.sh` script defines it at the top.

### Full CLI reference (`key=value`, last occurrence wins)

**Geometry** `ncol=4096` columns · `incols=512` clamped input columns · `inbr=8` afferent
branches of 32 · `nand=4` initial functional density = 2^-nand · `stimcols=128` active
input columns per stimulus · `stim=12` number of stored patterns

**Dynamics** `target=256` active columns (top-K) · `k=6` winners per column ·
`dthr=2` dendritic branch threshold · `boost=4` supralinear NMDA bonus ·
`mode=1` 0=point neuron, 1=supralinear, 2=strict two-layer · `T=10` steps per
presentation · `clampsteps=5` cue steps before release · `ktest=0` readout-only k
override · `facil=0` readout-only facilitation bonus

**Learning** `epochs=30` · `contrast=0` enable contrastive phases · `ltdrep=1` LTD
repetitions per free step · `ltdevery=1` apply LTD every Nth free step (2 → "0.5×") ·
`bandlo=5 bandhi=11` symmetric counter clip band · `margingate=1` skip columns with zero
k/(k+1) margin · `hetero=0` heterosynaptic LTD · `biasup=0` intrinsic excitability
increment (**`biasup=1` means auto-compute**, use 73 for C) · `biasdown=1` ·
`biasdownevery=1` fractional decay · `biasshift=4` bias resolution shift

**Diagnostics** `seed=0` varies weights+connectivity+stimuli jointly · `deep=1`
multiplicity, heavy tail, drive trace · `split=1` shared vs exclusive members ·
`unionchar=1` visited-population characterisation · `wave=1 wavet=35` long free run ·
`diag=1` counter histogram · `profile=1` per-kernel GPU timing · `quiet=1` one-line
output · `topkpar=1` parallel top-K · `vtopk=1` verify parallel==serial ·
`vbs=1` verify bit-serial vs scalar reference

## 5. Build and reproduce

```sh
cd /Users/rakib/Documents/axon-repo
./build.sh                       # single clang++ call, seconds
```

Toolchain: macOS 15.1, Xcode **Command Line Tools 14.3** (clang 14.0.3, SDK 13.3).
Full Xcode is **not** required — Metal shaders compile at runtime from strings embedded
in `axon.mm`, which also makes geometry specialisation by string substitution free.
Tectonic is installed (`brew install tectonic`) if LaTeX is ever needed again.

**Fastest verification that a build is good (~1 min):**

```sh
./build/axon nand=4 target=64 k=24 dthr=1 mode=1 inbr=4 stimcols=64 contrast=1 \
  margingate=0 ltdrep=1 ltdevery=2 biasup=73 biasdown=1 T=10 clampsteps=5 \
  stim=12 epochs=40 deep=1 wave=1 wavet=35
```

Must give: density **82.0905%**, recall **0.746**, best-other **0.042**, sustained
**0.661**, drive t6 **118**. Verified from a clean GitHub clone on 2026-08-11.

Table scripts: `bench/cap10.sh` (T1) · `bench/seeds.sh` (T2) ·
`bench/table3_subgroups.sh` (T3) · `bench/table4_gaintraj.sh` (T4) ·
`bench/table5_union.sh` (T5). `bench/capacity.sh` is the **superseded T=20 sweep** —
retracted, kept for the record, do not use for Table 1.

## 6. What is settled — do not re-litigate

**Ruled out** (each with a falsifying measurement): hub capture (in-degree CV 0.150,
hub ratio 0.733 *inverted*) · LTD erosion at the boundary (density flat) · allocation
interference (0.79–1.04× the sharing-adjusted null at 4.45× multiplicity) · input-space
collisions (4× width reduction with pool and afferent drive matched: no effect) · pool
size (flat over a 6× range at fixed sparsity) · averaging artifact (both subgroups flat)
· drive magnitude at mid load (unity gain while overlap decays).

**Retired hypotheses** (all died to replication): a bifurcation in loop gain (5 seeds →
smooth monotone decline; the 0.652→0.284 step was noise) · a predictive `gain^14.5`
model (**the exponent turns 1% gain error into 14.5% prediction error, so n=1 agreement
was near-random by construction**) · a two-cycle explanation of the visited population
(each parity stream alone visits 1.90×, parity overlap 0.892) · readout-side rescue via
lower *k* or facilitation (facilitation makes the *untrained* net sustain at 1.000).

**Novelty claims, narrowed after literature checks:** the dynamics-vs-storage framing is
**not novel** (kernel-Hopfield precedent) · sustained/transient pair has precedent
(Clark, arXiv:2506.05303) → narrowed to empirical instantiation · sub-chance allocation
overlap is close to expected from k-WTA/competitive learning → narrowed to the
forced-multiplicity regime · initiation/maintenance was *strengthened* after checking
(the human precedent is component-specific at N=29 and reports load effects during
encoding). Surviving: the direct weight measurement, contrasted as
same-synapse-model/different-storage-regime with Fusi & Abbott.

**Validity precondition for any sustain ratio** (three conditions, none redundant):
(a) free-phase drive > 2× the untrained floor of 29; (b) recall > best-other; (c) the
matched untrained control does not itself sustain. Each catches a distinct failure —
collapse, spurious-attractor convergence, and mechanisms that freeze any state.

## 7. Engine internals worth knowing

- Weights are 4 planes `W0..W3`; **only `W3` is read by propagation**, all four are
  read-modify-written by learning. Bit-serial ripple-carry increment/decrement/clip,
  verified exact against a scalar reference over 2,097,152 values per operation.
- `k_drive` → `k_topk` → `k_emit` → `k_bias` → `k_learn`, one dispatch each per timestep,
  batched into one command buffer. **No persistent megakernel** — measured and rejected.
- `driveLog` is **4 slots per step** (shared sum, shared count, exclusive sum, exclusive
  count). Any new reader must sum both halves. See §8.
- Intrinsic bias is a **learning-time mechanism only**; it is zeroed at readout
  (`P.useBias = learn ? 1 : 0`). This "readout gate" was the single most decisive fix in
  the project — before it, recall was ~0 because assembly members were vetoed by their
  own accumulated bias (3439 vs a drive of ~65).

### Hardware facts (M1 Pro, measured, in `notes/00`–`02`)

Sequential read 170 GB/s · scalar random gather 5.4–7.9 GB/s (3–5% of peak) ·
**simdgroup-cooperative gather 102–113 GB/s (13.6× the scalar case at identical block
size and randomness)** · columnar vs random sparsity 14,585 vs 1,285 steps/s at identical
spike count (11.4×) · dense binary propagation 1.36 T synapse-ops/s · only 8 threadgroups
reliably co-resident · **ordinary device writes are not visible across threadgroups
mid-dispatch**; spin reads must be RMW, not `atomic_load` · device exposes only a
`timestamp` counter set, no dispatch-boundary sampling · host round trip 165 µs vs a
~0.4 ms timestep, which is why no CPU/AMX/ANE offload is viable.

## 8. Gotchas that cost real time — do not repeat

1. **`zsh` does not word-split unquoted `$VAR`.** Passing `$ARGS` to a binary sends the
   whole string as ONE argv entry, silently reverting every tunable to its default. This
   invalidated a complete ablation table. Always use arrays: `A=(k=1 v=2); prog "${A[@]}"`.
2. **`argu()` originally returned the FIRST match**, so a key already in a base array
   could not be overridden. Fixed to last-match-wins. Detected only because two sweeps
   agreed to four decimal places.
3. **The macOS filesystem is case-insensitive.** `/Documents/AXON` and `/Documents/axon`
   are the same directory. A `mv` to a case-variant path merged directories and a
   following `rm` deleted a user file. **Before any `mv`/`rm` near an existing path, run
   `ls -di <src> <dst>` and confirm the inodes differ.**
4. **`driveLog` stride.** It was widened 2→4 slots per step; an older reader still used
   stride 2 and produced a misaligned trace (values present at index `2s+1`). Fixed and
   commented in `axon.mm`. The simulation was never affected — it was a print bug.
5. **`grep -A2 ... | tail -1`** grabs the separator line, not the data line. Use `-A1`.
6. **Sustain ratio is a noisy metric** (CV 7.7%→62.4% with load); **margin is stable**
   (SD 0.007–0.033). Prefer margin. Always report sustain with error bars.
7. Timing from multi-run sweep tables varied up to 3× and was never reproducible in a
   controlled test. **Never compare steps/s across rows of a sweep.** Clean single runs
   are ±3%: 2554–2956 trained, 3795–4930 untrained.

## 9. Provenance warnings

- **`notes/00`–`08` are an unedited running log**, not results. They contain retired
  hypotheses and numbers later corrected. A number in `notes/` is *not* a finding — the
  paper's corrections section is authoritative.
- **File timestamps in this repo are copy times, not authorship times.** The evidence
  that the code is the code is the exact numeric reproduction in §5, on five independent
  quantities.

## 10. Open threads, if new work is wanted

None are required; the paper stands without them.

- **Replicate the loop-gain trajectory more widely.** Gain values are 5-seed but the
  *trajectory* table (T4) is the thinnest evidence in the paper.
- **Sequential storage.** Everything here is interleaved-to-equilibrium. The Fusi &
  Abbott contrast would be much stronger under a genuine palimpsest protocol — this is
  the objection a reviewer is most likely to press, and it is answered in §5.1 of the
  paper by argument rather than by experiment.
- **Does the visited-population bound explain capacity?** It co-varies monotonically
  with load, so a correlation is guaranteed and proves nothing. Any test must be able to
  fail. (This caution is exactly what the retired gain model did not get.)
- **Structural plasticity.** ~50% of recurrent synapses were never touched by
  plasticity; rewiring `conn[]` on idle CPU cores was scoped but never built.
- **Generality.** One architecture, one machine, one task family. Untested elsewhere.

## 11. Working style that produced this

Every substantive claim went through: propose → measure → **replicate across seeds** →
check the untrained control → search the literature before claiming novelty. Five
hypotheses, four of them ours, died to that loop; three novelty claims were dropped or
narrowed. Single-run results were repeatedly wrong in ways that looked convincing.
**Replicate before believing anything, including your own reasoning.**

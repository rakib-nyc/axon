# AXON

Code, sweep scripts and measurement logs for the preprint **"Memory capacity as a retrieval limit: direct synaptic measurement across a capacity range in a sparse binary assembly network."**

In associative memory theory, capacity is normally a property of storage: patterns superpose in a synaptic matrix until interference makes them unrecoverable. This work reports a case where that account does not apply, established by measuring the synapses directly rather than inferring their state from behaviour. In a sparse binary assembly network (131,072 neurons, 1.07 G binary synapses, bounded 4-bit synaptic counters, contrastive learning, exact *k*-winners-take-all readout), a five-fold increase in stored patterns (12 → 60, five seeds per load) reduces the sustained-recall ratio by 3.7× while the functional density of recurrent synapses *within* stored assemblies stays constant at 80.7–82.2% — separately for neurons exclusive to one assembly and for neurons shared across several, so it is not an averaging artifact. Retrieval *initiation* is load-invariant, and at intermediate load recurrent drive is maintained at unity gain while the retrieved population nonetheless drifts away from the stored assembly. The paper reports capacity as a pair (sustained ≈ 36 patterns, transient ≈ 200) and documents at length what was ruled out, what was retired, and what was corrected along the way.

## Paper

- **PDF:** [`paper/Preprint.pdf`](paper/Preprint.pdf)
- **LaTeX source:** [`paper/axon_preprint.tex`](paper/axon_preprint.tex)
- **DOI:** [10.5281/zenodo.21881612](https://doi.org/10.5281/zenodo.21881612)

## Build

Requirements, as used for every number in the paper:

| | |
|---|---|
| Machine | Apple M1 Pro (8 performance + 2 efficiency cores, 16-core GPU), 16 GB unified memory |
| OS | macOS 15.1 (Darwin 24.1.0) |
| Toolchain | Xcode **Command Line Tools 14.3** (Apple clang 14.0.3, macOS SDK 13.3) |
| Xcode | **Not required.** Metal shaders are compiled at runtime from source embedded in `src/axon.mm`, so no offline `metal` compiler is needed |
| Frameworks | Metal, Foundation |

```sh
./build.sh          # -> build/axon
```

That is the whole build: a single `clang++` invocation, a few seconds. The binary allocates roughly 512 MB of synaptic weight planes plus working buffers; ~2 GB free memory is comfortable.

A quick sanity run (a few seconds, untrained):

```sh
./build/axon stim=12 epochs=0 T=10 clampsteps=5
```

## Reproduction table

Configuration **C** — used for every table in the paper — is defined at the top of each script:

```
nand=4 target=64 k=24 dthr=1 mode=1 inbr=4 stimcols=64 contrast=1 margingate=0 \
ltdrep=1 ltdevery=2 biasup=73 biasdown=1 T=10 clampsteps=5
```

| Paper table | Command | Seed(s) | Approx. runtime |
|---|---|---|---|
| Table 1 — capacity sweep (`tab:capacity`) | `bench/cap10.sh` | 0 | ~20 min |
| Table 2 — five-seed replication (`tab:seeds`) | `bench/seeds.sh` | 0–4 | ~15 min |
| Table 3 — shared vs exclusive members (`tab:subgroups`) | `bench/table3_subgroups.sh` | 0 | ~5 min |
| Table 4 — loop-gain trajectory (`tab:gaintraj`) | `bench/table4_gaintraj.sh` | 0–4 | ~15 min |
| Table 5 — visited-population characterisation (`tab:union`) | `bench/table5_union.sh` | 0–4 | ~20 min |

Single-row spot check, the fastest way to confirm a working build reproduces published numbers (~1 min):

```sh
./build/axon nand=4 target=64 k=24 dthr=1 mode=1 inbr=4 stimcols=64 contrast=1 \
  margingate=0 ltdrep=1 ltdevery=2 biasup=73 biasdown=1 T=10 clampsteps=5 \
  stim=12 epochs=40 deep=1 wave=1 wavet=35
```

Expected (Table 1, 12 patterns, seed 0): within-assembly density **82.0905%**, recall **0.746**, best-other **0.042**, sustained ratio **0.661**, first-free-step drive **118**. These were re-verified from a clean checkout of this repository and matched exactly.

### Honest notes on the scripts

- `bench/cap10.sh` and `bench/seeds.sh` are the scripts that originally produced Tables 1 and 2.
- `bench/table3_subgroups.sh`, `bench/table4_gaintraj.sh` and `bench/table5_union.sh` were written **after the fact** to reproduce results originally obtained from equivalent inline command lines. They issue the same binary invocations; they are not the literal shell history.
- `bench/capacity.sh` is the **superseded** capacity sweep, run at `T=20`. Its output was reported and then retracted when a configuration mismatch was found; see the corrections section of the paper. It is kept for the record. **Do not use it to reproduce Table 1** — use `cap10.sh`.
- Other scripts (`c1.sh`, `c1b.sh`, `ltd.sh`, `law.sh`, `gain.sh`, `traj.sh`, `drift.sh`, `audit.sh`) produced in-text numbers rather than numbered tables.
- `bench/probe0*.mm` are standalone hardware-characterisation programs (memory roofline, threadgroup co-residency, cross-threadgroup coherence, GPU counters). Compile individually, e.g. `clang++ -std=c++17 -fobjc-arc -O2 bench/probe04_roofline.mm -framework Metal -framework Foundation -o build/probe04`.

## Hardware note

Every result was measured on one machine: an **Apple M1 Pro with 16 GB of unified memory**. Throughput will differ on other hardware, and the engineering measurements in the paper's Methods (memory bandwidth, threadgroup co-residency, barrier cost, counter availability) are specific to this GPU and may not hold on other Apple Silicon parts, let alone other vendors.

**No result in the paper depends on a timing measurement.** All overlap, density, Jaccard, margin, turnover and gain figures are deterministic functions of the simulation given a configuration and seed. The paper additionally documents that throughput figures taken from multi-run sweep tables proved unreliable (up to threefold variation that could not be reproduced in a controlled test) and are therefore never compared across rows.

## What's in `notes/`

`notes/00`–`08` are a **running measurement log kept during the work, left unedited**. They are not a summary of the findings and should not be read as one.

They contain hypotheses that were subsequently **retired** (a bifurcation in loop gain, a predictive `gain^14.5` model, a two-cycle explanation of the visited population, a readout-side rescue), numbers that were later **corrected** (a capacity sweep run at `T=20` and reported as `T=10`, and an "exact partition" claim that did not survive that correction), and intermediate values produced before analysis bugs were found and fixed.

Their unedited state is part of the evidence — it is what allows the corrections section of the paper to be checked — but **a number appearing in `notes/` is not a result.** For the final state of every claim, including which were withdrawn and why, see the paper's *Corrections and controls* and *What was retired* sections.

## Repository layout

```
src/axon.mm                 engine: host code + all Metal kernels, single file
build.sh                    one-command build
bench/*.sh                  sweep scripts (see reproduction table)
bench/probe0*.mm            standalone hardware characterisation programs
notes/00-08                 running measurement log, unedited (see caveat above)
paper/Preprint.pdf          the preprint
paper/axon_preprint.tex     its LaTeX source
```

## Licence

- **Code** (`src/`, `bench/`, `build.sh`): MIT — see [`LICENSE`](LICENSE)
- **Paper** (`paper/`): CC BY 4.0 — see [`LICENSE-PAPER`](LICENSE-PAPER)

## Citation

```bibtex
@article{islam2026axon,
  title  = {Memory capacity as a retrieval limit: direct synaptic measurement
            across a capacity range in a sparse binary assembly network},
  author = {Islam, Muhammad Rakibul},
  year   = {2026},
  doi    = {10.5281/zenodo.21881612},
  url    = {https://doi.org/10.5281/zenodo.21881612}
}
```

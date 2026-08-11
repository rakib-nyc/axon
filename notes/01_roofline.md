# AXON — F6: the memory roofline (probe04)

512 MB private buffer, best-of-5, GPU pre-warmed 2 s.

## Sequential read = the ceiling

| grid | GB/s |
|---|---|
| 262144 | 162.1 |
| 1048576 | 166.4 |
| 4194304 | **170.0** |

170 GB/s = 85% of M1 Pro's 200 GB/s spec. That is the ceiling for everything below.

## Random gather — thread-per-block (each lane chases its own pointer)

| block bytes | GB/s | blocks/s | % of sequential |
|---|---|---|---|
| 16 | 5.4 | 334.6 M | 3.3% |
| 32 | 6.0 | 188.4 M | 3.7% |
| 64 | 7.7 | 120.5 M | 4.7% |
| 128 (cache line) | 7.7 | 60.1 M | 4.7% |
| 256 | 7.4 | 28.7 M | 4.5% |
| 512 | 7.5 | 14.7 M | 4.6% |
| 1024 | 7.9 | 7.8 M | 4.8% |
| 2048 | 67.8 | 33.1 M | 41.2% |

## Random gather — simdgroup-cooperative (32 lanes share one block)

| block bytes | GB/s | % of sequential |
|---|---|---|
| 512 | **102.0** | 61.9% |
| 1024 | 96.5 | 58.6% |
| 2048 | 107.6 | 65.4% |
| 4096 | 112.7 | 68.5% |

## Read+write round-trip (neuron state, per timestep)

| state | ms | GB/s (r+w) |
|---|---|---|
| 1 MB | 0.187 | 11.2 |
| 4 MB | 0.208 | 40.4 |
| 16 MB | 0.275 | 121.8 |
| 64 MB | 0.826 | 162.5 |
| 256 MB | 3.148 | 170.6 |

The small sizes are floored at ~0.19 ms because each is its own command buffer
with `waitUntilCompleted` — that is the same 165 µs host round-trip from F5,
not a memory effect. Batched, 16 MB of state costs ~0.11 ms of real GPU work.

---

# DECISION 2: sparsity must be simdgroup-collective, not per-neuron.

**The original design assumed random sparse synaptic gather would reach "10–20%
of peak." It reaches 3–5%.** Event-driven per-neuron synaptic gather — the
approach essentially every neuromorphic simulator uses — runs at **5–8 GB/s on
this machine, 4% of the hardware**. That approach is dead here.

**Making blocks contiguous does NOT fix it.** 16 B → 1 KB moves 5.4 → 7.9 GB/s.
The cache line is a red herring. What fixes it is *who issues the load*: the
same random blocks read cooperatively by 32 lanes go **7.5 → 102 GB/s, a 13.6×
jump at identical block size and identical randomness.**

So the unit of structure is neither the 128 B cache line nor the AMX 32×32 tile.
It is **32 lanes × 16 B = 512 B, the simdgroup coalescing width.**

## The co-design claim this produces

Unstructured sparsity is unaffordable. But cortex is not unstructured — it is
columnar. So: **k-winners-take-all among columns, dense within a winning column.**
Columnar sparsity is both more biologically faithful than the random sparsity
usual in SNN models *and* the only form of sparsity this silicon can execute at
speed. That coincidence is the project.

## Tile geometry that falls out

One simdgroup load = 32 lanes × `uint4` = 512 B = **4096 binary synapses**:

- lane *l* holds postsynaptic neuron *l* of a 32-neuron **column**
- its `uint4` = that neuron's weights for **128 presynaptic neurons**
- so one coalesced load = **32 postsyn × 128 presyn binary weight tile**
- the 128 presyn spikes are 4 `uint32` — broadcast, register-resident
- propagation = 4 × `popcount(w & spikes)` per lane

Skip rule: the tile is skipped iff all 4 presyn spike words are zero, and that
test is **simdgroup-uniform**, so skipping costs no divergence. With *random* 1%
activity a 128-neuron presyn group is silent only 27% of the time (0.99^128) —
nearly useless. With *columnar* activity, a few active groups mean ~99% of tiles
are skipped. **The skip rate is precisely why the code must be columnar.**

## Budget

At 1-bit weights and 102 GB/s coalesced, the ceiling is ~820 G binary
synapse-reads/s. 131,072 neurons × 8192 fan-in = 1.07 G synapses = **134 MB at
1 bit/synapse**, comfortably resident. At 5% of presyn groups active that is
~6.7 MB of tile traffic per timestep → ~60 µs → **~16,000 timesteps/s**.
Verify against probe05 before believing it.

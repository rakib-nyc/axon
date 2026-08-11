# AXON — measured facts about this specific machine

All numbers measured on **Apple M1 Pro (T6000), 8P+2E, 16 GB, macOS 15.1 (24B83),
Command Line Tools 14.3 / SDK 13.3, no full Xcode.**
Nothing here is from documentation. Every line has a probe that produced it.

---

## F1. Toolchain

| Fact | Value |
|---|---|
| `xcrun metal` (offline shader compiler) | **ABSENT** — no full Xcode |
| `newLibraryWithSource:` at runtime | **WORKS** |

**Consequence:** all shaders are compiled at runtime from embedded source strings.
No `.metallib` build step exists in this project, and none can. Upside: shader
specialization by string substitution is free, so network geometry can be baked
into the shader as literal constants instead of being read from a uniform buffer.

## F2. GPU capabilities (probe00)

| Property | Value |
|---|---|
| Family | Apple7 + Metal3 |
| `threadExecutionWidth` | **32** |
| max threads / threadgroup | 1024 |
| threadgroup memory | 32 KB |
| `recommendedMaxWorkingSetSize` | 10.67 GB |
| `maxBufferLength` | 8.00 GB |
| unified memory | yes |

`simd_ballot` verified lane-exact: input pattern `0xA5A5A5A5` returned
`0xA5A5A5A5`. **Bit *i* of the ballot is lane *i*.** This is the primitive the
whole design rests on and it behaves exactly as hoped.

## F3. Threadgroup co-residency is NOT a stable hardware property (probe01/02)

Single-trial probing gave wildly non-monotonic answers (10, 13, 11, 12, 16, 33
TGs as tgsize grew; different again after warmup). Requiring **5/5 identical
trials** collapses it to a stable floor:

| tgsize | reliably co-resident TGs | threads |
|---|---|---|
| 128 | 8 | 1024 |
| 256 | 8 | 2048 |
| 512 | 8 | 4096 |
| 1024 | 8 | 8192 |

**8 threadgroups is the only number you can count on**, regardless of size.
Anything above that is the scheduler being generous, not a guarantee. A
persistent megakernel may therefore only assume ~8 resident threadgroups —
a small fraction of a 16-core GPU.

## F4. Cross-threadgroup memory coherence inside one dispatch (probe03)

This is the big one. Same handshake, three memory disciplines, 2000 rounds:

| payload write / read | spin read | result |
|---|---|---|
| plain store / plain load | plain atomic load | DEADLOCK, corrupt |
| atomic store / atomic load | plain atomic load | DEADLOCK |
| atomic store / RMW read | plain atomic load | DEADLOCK |
| **plain store / plain load** | **RMW (`fetch_or 0`)** | **INCOHERENT** — corrupt 3997 / 23993 / 111971 at N=2/4/8 |
| **atomic store / atomic load** | **RMW** | **OK** — 2000/2000 rounds, 0 corrupt |
| **atomic store / RMW read** | **RMW** | **OK** — 2000/2000 rounds, 0 corrupt |

Two independent facts, and they are separable:

1. **A spin loop must read via an RMW** (`atomic_fetch_or(p,0)`). A plain
   `atomic_load_explicit` in a tight loop never refreshes and deadlocks —
   the same trap `volatile` solves in CUDA. Every deadlock above is a spin=load row.
2. **Ordinary `device` stores are not visible to other threadgroups mid-dispatch**,
   even with `threadgroup_barrier(mem_flags::mem_device)`. Corruption scales with
   N (≈ every read wrong). Only *atomic* accesses reach the coherent point.

### Working device-wide barrier cost (RMW spin, correct)

| threadgroups | ns / barrier |
|---|---|
| 2 | 1134 |
| 4 | 1305 |
| 8 | 1679 |

## F5. The alternative: just dispatch per timestep (probe02)

| dispatches per command buffer | ns / step |
|---|---|
| 1 | **165,000** |
| 100 | 3,974 |
| 1000 | **2,411 – 2,646** |

Flat in threadgroup count (8 / 16 / 32 all the same).

---

# DECISION 1: no persistent megakernel.

The original design made the persistent megakernel + atomic grid barrier the
centerpiece, and named "is the atomic device barrier stable under Apple's
scheduler?" as risk #1. Measured answer: **it is stable, and it is still the
wrong choice.**

| | megakernel + grid barrier | batched dispatches |
|---|---|---|
| sync cost | 1.7 µs @ 8 TGs | 2.5 µs |
| usable parallelism | **8 threadgroups** (F3) | unlimited |
| cross-TG data traffic | **must be atomics** (F4) | plain stores, free |
| inside Metal's memory model | no | yes |
| deadlock risk | real | none |

The barrier saves ~0.8 µs per step and costs a 16-core GPU shrunk to 8
threadgroups plus every byte of shared spike traffic promoted to an atomic.
That trade is bad by a wide margin.

**Architecture:** one *fused* kernel dispatch per timestep, ~1000 dispatches
batched per command buffer, host submits triple-buffered. Budget **2.5 µs of
sync overhead per timestep** and spend the design effort on making the timestep
kernel itself good. Fusing matters: 3 kernels per timestep = 7.5 µs of pure
overhead, so per-timestep work belongs in one dispatch wherever possible.

The one thing per-timestep dispatch costs us: state cannot live in registers
across timesteps, so neuron state round-trips to memory every step. At 32 B of
state per neuron and 250 k neurons that is 8 MB/step — which is why **F6
(bandwidth) is now the load-bearing measurement**, and the next probe.

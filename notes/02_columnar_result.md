# AXON — F7: columnar sparsity is worth 11x (probe06)

131,072 neurons (4096 columns x 32), fan-in 8192, 1.07 G binary synapses, W = 128 MB.
Real spike buffer (not a hash). Fan-in held constant across all rows.
**Both conditions have exactly the same number of spikes.**

## Slot granularity x activity structure, 1% activity

| config | random ms | random steps/s | clustered ms | clustered steps/s | speedup |
|---|---|---|---|---|---|
| 64 slots x 128 presyn | 0.5738 | 1,743 | 0.0840 | 11,905 | 6.83x |
| **32 slots x 256 presyn** | 0.7785 | 1,285 | **0.0686** | **14,585** | **11.35x** |
| 16 slots x 512 presyn | 0.8187 | 1,221 | 0.0801 | 12,483 | 10.22x |
| 8 slots x 1024 presyn | 0.8022 | 1,247 | 0.2218 | 4,508 | 3.62x |

## Activity sweep (64 x 128 config)

| activity | random ms | clustered ms | speedup |
|---|---|---|---|
| 0.25% | 0.2217 | 0.0803 | 2.76x |
| 0.50% | 0.3847 | 0.0814 | 4.73x |
| 1.00% | 0.5722 | 0.0834 | 6.86x |
| 2.00% | 0.7336 | 0.0921 | 7.97x |
| 5.00% | 0.7829 | 0.1322 | 5.92x |
| 10.00% | 0.7847 | 0.2840 | 2.76x |

# DECISION 3: the sparse code must be columnar, and this is a scientific claim, not a tuning knob.

Random sparsity is nearly worthless on this hardware: at 10% random activity
(0.785 ms) the network runs at the **same speed as fully dense** (0.787 ms,
probe05). The reason is arithmetic — with 1% *random* activity a 128-neuron
presynaptic group is silent only 0.99^128 = 27% of the time, so almost nothing
is skippable. Sparsity that is not aligned to the coalescing granularity buys
nothing.

Cluster the same spikes into columns and 92% of groups fall silent. That is the
entire 11x.

There is an optimum at 256 presyn/branch. Coarser (1024) reads fewer, larger
blocks but a 1024-neuron group spans 32 columns, so P(silent) collapses and the
speedup falls to 3.6x. **Skip granularity must match assembly granularity.**

## The neuroscience this forces, and why it is a feature

To go fast, the model *must* have:
- activity clustered into columns (not random sparse firing)
- an inhibitory mechanism that picks winning columns, not winning neurons
- assemblies whose spatial extent is ~256 neurons

That list is not a compromise. It is minicolumnar organization, lateral
inhibition, and cortical assembly size. **The hardware's fastest configuration
is also the more biologically faithful one.** That coincidence is the thesis of
this project, and F7 is its quantitative statement: 1,285 -> 14,585 steps/s,
same spikes, purely from geometry.

## Consequence: the slot IS a dendritic branch

The inner loop already computes, per lane, per slot, in registers:

    popcount(w & sp)     over one 128-bit presynaptic tile

Summing that across slots gives a point neuron. **Thresholding it per slot
instead gives an NMDA-dependent dendritic branch spike** (Poirazi & Mel's
two-layer pyramidal neuron, Losonczy/Polsky branch-local nonlinearity):

    acc += pc                 ->  point neuron   (one add)
    acc += (pc >= dthresh)    ->  2-layer neuron (one compare)

Same memory traffic, same registers, one instruction either way. The natural
hardware unit — one coalesced tile per lane per slot — is exactly a dendritic
branch with its own local threshold. Verify the cost is really zero (probe07),
then build on it.

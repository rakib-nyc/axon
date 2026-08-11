// ============================================================================
// AXON v0 — columnar assembly engine for Apple M1 Pro
//
// Every geometric choice here is forced by a measurement in notes/00..02:
//   F5  batched dispatches (2.5 us) beat a persistent megakernel  -> no grid barrier
//   F6  simdgroup-collective 512 B tiles are 13.6x random gather  -> 1 column = 1 simdgroup
//   F7  columnar sparsity is worth 11x over random sparsity       -> k-WTA + column gate
//
// Neuron:  32 dendritic branches x 256 presynaptic binary synapses = 8192 fan-in
//          branch computes popcount(w & spikes) in registers; NMDA-like local threshold
// Synapse: 4-bit counter stored as 4 BITPLANES. Effective weight = MSB plane.
//          Propagation reads 1 plane; learning read-modify-writes 4.
//          Updates are bit-serial ripple-carry: 128 synapses per ~20 bitwise ops.
// Code:    exact k-of-32 winners-take-all per column (simd_shuffle rank + ballot)
//          + homeostatic global column threshold  -> column-granular activity
// ============================================================================

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <string>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <numeric>

// ---------------- geometry ----------------
static uint32_t NCOL     = 4096;   // runtime: recurrent pool = (NCOL-IN_COLS)*32          // columns
static const uint32_t NPC      = 32;            // neurons per column (= SIMD width)
static uint32_t N        = NCOL * NPC;    // 131072 neurons
static const uint32_t NSLOT    = 32;            // dendritic branches per neuron
static const uint32_t SLOT_U4  = 2;             // uint4 per branch per lane -> 256 presyn
static const uint32_t PRESYN   = SLOT_U4 * 128; // 256 presynaptic neurons per branch
static const uint32_t FANIN    = NSLOT * PRESYN;// 8192
static const uint32_t COLS_PER_GRP = PRESYN / NPC;      // 8 columns per presyn group
static uint32_t NGRP     = NCOL / COLS_PER_GRP;   // 512 groups
static size_t   PLANE_U4 = (size_t)NCOL * NSLOT * SLOT_U4 * 32; // uint4 per plane
static size_t   PLANE_B  = PLANE_U4 * 16;

static uint32_t IN_COLS  = 512;   // runtime-settable: widens the input space           // columns 0..511 are the clamped input area

static const char* kSrc = R"MSL(
#include <metal_stdlib>
using namespace metal;

#define NPC       32u
#define NSLOT     32u
#define SLOT_U4   2u
#define NCOLD     4096u

inline uint hash32(uint x){ x^=x>>16; x*=0x7feb352du; x^=x>>15; x*=0x846ca68bu; x^=x>>16; return x; }

struct Params {
    uint  k;          // winners per column
    uint  theta;      // column drive threshold (homeostatic)
    uint  dthr;       // dendritic branch threshold (coincident inputs)
    uint  boost;      // supralinear NMDA boost added when a branch crosses dthr
    uint  mode;       // 0 = point neuron, 1 = supralinear dendrite, 2 = strict 2-layer
    uint  inCols;     // clamped input columns
    uint  step;
    uint  learn;      // 0/1
    uint  depress;    // 0/1 apply heterosynaptic depression this step
    uint  targetCols; // homeostatic target for active column count
    uint  clamp;      // 1 = drive input area from stim, 0 = let it run free
    uint  freeze;     // 1 = hold theta fixed (test time)
    uint  biasUp;     // intrinsic homeostasis: threshold rise per spike
    uint  biasDown;   // ... and decay per silent step
    uint  biasShift;  // bias is kept at 2^shift resolution so it moves slowly
    uint  histOn;     // record per-step spike snapshot
    uint  useBias;    // 0 = ignore the intrinsic bias entirely (A4 readout ablation)
    uint  phase;      // C1: 0 = legacy rule, 1 = clamped (increment), 2 = free (decrement)
    uint  bandLo;     // symmetric clip band around threshold 8
    uint  bandHi;
    uint  marginGate; // 1 = skip columns whose k/(k+1) margin is zero
    uint  hetero;     // 1 = also apply heterosynaptic LTD on presyn-silent synapses
    uint  ltdRep;     // C1b: times bs_dec is applied per free-phase update (volume scaling)
    uint  biasDownEvery; // apply the bias decay only every Nth step => fractional biasDown
    uint  facil;      // readout-time facilitation: bonus drive for neurons active last step
    uint  _pad6;
};

// ---------------------------------------------------------------- init
kernel void k_init_w(device uint4* W0 [[buffer(0)]], device uint4* W1 [[buffer(1)]],
                     device uint4* W2 [[buffer(2)]], device uint4* W3 [[buffer(3)]],
                     constant uint& nAnd [[buffer(4)]], constant uint& seed [[buffer(5)]],
                     uint gg [[thread_position_in_grid]])
{
    uint g = gg ^ (seed * 0x9E3779B9u);        // decorrelate weight init across seeds
    // AND of nAnd independent random words -> functional density 2^-nAnd
    uint4 m = uint4(0xFFFFFFFFu);
    for (uint i = 0; i < nAnd; ++i)
        m &= uint4(hash32(g*32u+i*4u+0),hash32(g*32u+i*4u+1),
                   hash32(g*32u+i*4u+2),hash32(g*32u+i*4u+3));
    uint4 c = uint4(hash32(g*8u+11),hash32(g*8u+13),hash32(g*8u+17),hash32(g*8u+19));
    W3[gg] = m;                         // MSB = functional
    // sub-MSB bits start just below threshold for non-functional synapses, so a
    // few coincidences can recruit them; functional ones start mid-range.
    W2[gg] = (~m) | c;
    W1[gg] = c;
    W0[gg] = ~c;
}
// Dedicated afferent pathway: the first `inBranches` dendritic branches of every
// column sample ONLY the input area; the rest sample only the recurrent area.
// Diffuse sampling gives a column ~0.5 active afferent branches, which is too
// weak for the input to determine the assembly at all.
kernel void k_init_conn(device uint* conn [[buffer(0)]], constant uint4& cfg [[buffer(1)]],
                        constant uint& seed [[buffer(2)]],
                        uint g [[thread_position_in_grid]])
{
    uint nGrp = cfg.x, nInGrp = cfg.y, nSlot = cfg.z, inBr = cfg.w;
    uint s = g % nSlot;
    uint h = hash32(g*2246822519u + 7u + seed*2654435761u);
    conn[g] = (s < inBr) ? (h % nInGrp) : (nInGrp + h % (nGrp - nInGrp));
}

// ---------------------------------------------------------------- k-WTA
// exact k winners among 32 lanes; ties broken by lane index; result is a ballot
inline uint kwta_ballot(uint v, uint lane, uint k){
    uint rank = 0;
    for (uint j = 0; j < 32u; ++j) {
        uint ov = simd_shuffle(v, j);
        rank += ((ov > v) || (ov == v && j < lane)) ? 1u : 0u;
    }
    return (uint)((simd_vote::vote_t)simd_ballot(rank < k));
}

// Same, but also reports the k-th and (k+1)-th largest drive. Their difference is
// the margin that decides whether the winner set is genuinely resolved or is being
// settled by lane index. The rank loop already has the information; two simd_max
// extract it, so this is free.
inline uint kwta_ballot_margin(uint v, uint lane, uint k,
                               thread uint& vk, thread uint& vk1){
    uint rank = 0;
    for (uint j = 0; j < 32u; ++j) {
        uint ov = simd_shuffle(v, j);
        rank += ((ov > v) || (ov == v && j < lane)) ? 1u : 0u;
    }
    vk  = simd_max((rank == (k - 1u)) ? v : 0u);
    vk1 = simd_max((rank == k)        ? v : 0u);
    return (uint)((simd_vote::vote_t)simd_ballot(rank < k));
}

// ---------------------------------------------------------------- timestep
// Split into drive -> select -> emit so the k-cap is EXACT and same-step.
// (Measured: applying the threshold one step late makes global inhibition
//  oscillate between near-silence and saturation. See notes/03.)
kernel void k_drive(device const uint4* W3     [[buffer(0)]],
                    device const uint*  conn   [[buffer(1)]],
                    device const uint4* spkIn  [[buffer(2)]],
                    device uint*        drive  [[buffer(3)]],
                    device uint*        bbits  [[buffer(4)]],
                    device uint*        accBuf [[buffer(5)]],
                    device const uint*  bias   [[buffer(7)]],
                    device const uint*  prevCol[[buffer(8)]],
                    constant Params&    P      [[buffer(6)]],
                    uint gid  [[thread_position_in_grid]],
                    uint lane [[thread_index_in_simdgroup]])
{
    uint col = gid >> 5;
    uint acc = 0, bb = 0, base = col * NSLOT;
    for (uint s = 0; s < NSLOT; ++s) {
        uint  g  = conn[base + s];
        uint4 s0 = spkIn[g * SLOT_U4 + 0];
        uint4 s1 = spkIn[g * SLOT_U4 + 1];
        uint any = s0.x|s0.y|s0.z|s0.w|s1.x|s1.y|s1.z|s1.w;
        if (any != 0u) {                                    // simdgroup-uniform skip
            uint  wb = (base + s) * SLOT_U4 * 32u + lane;
            uint4 p0 = popcount(W3[wb]       & s0);
            uint4 p1 = popcount(W3[wb + 32u] & s1);
            uint  c  = p0.x+p0.y+p0.z+p0.w + p1.x+p1.y+p1.z+p1.w;
            bool  nmda = (c >= P.dthr);
            if (nmda) bb |= (1u << s);
            if      (P.mode == 0u) acc += c;
            else if (P.mode == 1u) acc += nmda ? (c + P.boost) : c;
            else                   acc += nmda ? 1u : 0u;
        }
    }
    // readout-time facilitation: structurally the bias term with the sign flipped —
    // bias suppresses recently-active neurons, this facilitates currently-active ones.
    if (P.facil != 0u && ((prevCol[col] >> lane) & 1u) != 0u) acc += P.facil;
    uint bi = (P.useBias != 0u) ? (bias[gid] >> P.biasShift) : 0u;
    acc = (acc > bi) ? (acc - bi) : 0u;      // intrinsic excitability homeostasis
    accBuf[gid] = acc;
    bbits[gid]  = bb;
    uint drv = simd_sum(acc);
    if (lane == 0) drive[col] = drv;
}

kernel void k_emit(device const uint*  accBuf [[buffer(0)]],
                   device const uint*  drive  [[buffer(1)]],
                   device const uint*  theta  [[buffer(2)]],
                   device const uint*  stim   [[buffer(3)]],
                   device const uint*  bbits  [[buffer(4)]],
                   device uint*        spkOut [[buffer(5)]],
                   constant Params&    P      [[buffer(6)]],
                   device atomic_uint* stats  [[buffer(7)]],
                   device uint*        hist   [[buffer(8)]],
                   device uint*        margOK   [[buffer(9)]],
                   device const uint*  refBall  [[buffer(10)]],
                   device atomic_uint* driveLog [[buffer(11)]],
                   device const uint*  multB    [[buffer(12)]],
                   uint gid  [[thread_position_in_grid]],
                   uint lane [[thread_index_in_simdgroup]])
{
    uint col = gid >> 5;
    uint vk = 0u, vk1 = 0u;
    uint b   = kwta_ballot_margin(accBuf[gid], lane, P.k, vk, vk1);
    if (drive[col] < max(theta[0], 1u)) b = 0u;              // exact top-K column gate
    if (P.clamp != 0u && col < P.inCols) b = stim[col];      // clamp input area
    if (((b >> lane) & 1u) != 0u && col >= P.inCols)
        atomic_fetch_add_explicit(&stats[2], (uint)popcount(bbits[gid]), memory_order_relaxed);
    // drive arriving at reference-assembly members, whether or not they win
    if (col >= P.inCols && ((refBall[col] >> lane) & 1u) != 0u) {
        uint o = (multB[gid] > 1u) ? 0u : 2u;      // shared vs exclusive member
        atomic_fetch_add_explicit(&driveLog[P.step*4u+o+0u], accBuf[gid], memory_order_relaxed);
        atomic_fetch_add_explicit(&driveLog[P.step*4u+o+1u], 1u,          memory_order_relaxed);
    }
    if (lane == 0) {
        spkOut[col] = b;
        margOK[col] = (vk > vk1) ? 1u : 0u;
        if (P.histOn != 0u) hist[P.step * NCOLD + col] = b;
        if (b != 0u && col >= P.inCols) {
            uint sb = 64u + P.step * 8u;
            atomic_fetch_add_explicit(&stats[0], 1u, memory_order_relaxed);
            atomic_fetch_add_explicit(&stats[1], (uint)popcount(b), memory_order_relaxed);
            atomic_fetch_add_explicit(&stats[sb+0u], vk - vk1, memory_order_relaxed);  // margin
            atomic_fetch_add_explicit(&stats[sb+1u], vk,       memory_order_relaxed);  // k-th drive
            atomic_fetch_add_explicit(&stats[sb+2u], 1u,       memory_order_relaxed);  // n cols
            if (vk == vk1) atomic_fetch_add_explicit(&stats[sb+3u], 1u, memory_order_relaxed);
            if (vk == 0u)  atomic_fetch_add_explicit(&stats[sb+4u], 1u, memory_order_relaxed);
        }
    }
}

// per-neuron intrinsic homeostasis. Equilibrium firing rate p satisfies
// p*biasUp = (1-p)*biasDown, so biasUp/biasDown = (1-p)/p sets the target rate.
kernel void k_bias(device const uint* spk  [[buffer(0)]],
                   device uint*       bias [[buffer(1)]],
                   constant Params&   P    [[buffer(2)]],
                   uint gid  [[thread_position_in_grid]],
                   uint lane [[thread_index_in_simdgroup]])
{
    uint col = gid >> 5;
    if (col < P.inCols) return;
    bool fired = ((spk[col] >> lane) & 1u) != 0u;
    // biasDownEvery > 1 gives a fractional effective decay of biasDown/biasDownEvery
    int  dec = ((P.step % max(P.biasDownEvery,1u)) == 0u) ? (int)P.biasDown : 0;
    int  b = (int)bias[gid] + (fired ? (int)P.biasUp : -dec);
    bias[gid] = (uint)max(0, b);
}

// ---------------------------------------------------------------- inhibition
// Exact top-K columns by drive, via a 3-pass byte histogram in ONE threadgroup.
// Returns the K-th largest drive as the threshold. Applied with one step of
// delay, which is what feedback inhibition does anyway.
kernel void k_topk_serial(device const uint* drive [[buffer(0)]],
                   device uint*       theta [[buffer(1)]],
                   constant Params&   P     [[buffer(2)]],
                   constant uint&     nCol  [[buffer(3)]],
                   uint tid [[thread_position_in_threadgroup]])
{
    threadgroup atomic_uint hist[256];
    threadgroup uint selBin, need, prefix;
    if (tid == 0) { need = P.targetCols; prefix = 0u; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int by = 2; by >= 0; --by) {
        for (uint i = tid; i < 256u; i += 256u) atomic_store_explicit(&hist[i], 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint shHi = (uint)(by + 1) * 8u, shLo = (uint)by * 8u;
        for (uint c = P.inCols + tid; c < nCol; c += 256u) {
            uint d = drive[c];
            if ((d >> shHi) == (prefix >> shHi))
                atomic_fetch_add_explicit(&hist[(d >> shLo) & 255u], 1u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            uint cum = 0, b = 0, above = 0;
            for (int i = 255; i >= 0; --i) {
                uint h = atomic_load_explicit(&hist[i], memory_order_relaxed);
                if (cum + h >= need) { b = (uint)i; above = cum; break; }
                cum += h;
            }
            selBin = b; need = need - above; prefix |= (b << shLo);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) { theta[0] = max(prefix, 1u); theta[1] = need; }
}


// STEP B: same selection, parallelised. Two changes:
//   1. 1024 threads instead of 256 -> 4x the histogram accumulation width
//   2. the 256-bin scan, previously a serial loop in thread 0 (768 dependent
//      threadgroup loads across 3 passes), becomes a Hillis-Steele suffix sum
//      in log2(256) = 8 barrier steps.
// Selection semantics are unchanged: theta = the targetCols-th largest drive.
kernel void k_topk_par(device const uint* drive [[buffer(0)]],
                       device uint*       theta [[buffer(1)]],
                       constant Params&   P     [[buffer(2)]],
                       constant uint&     nCol  [[buffer(3)]],
                       uint tid [[thread_position_in_threadgroup]],
                       uint nth [[threads_per_threadgroup]])
{
    threadgroup atomic_uint hist[256];
    threadgroup uint scan[256];
    threadgroup uint selBin, need, prefix;
    if (tid == 0) { need = P.targetCols; prefix = 0u; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int by = 2; by >= 0; --by) {
        for (uint i = tid; i < 256u; i += nth) atomic_store_explicit(&hist[i], 0u, memory_order_relaxed);
        if (tid == 0) selBin = 0u;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint shHi = (uint)(by + 1) * 8u, shLo = (uint)by * 8u;
        for (uint c = P.inCols + tid; c < nCol; c += nth) {
            uint d = drive[c];
            if ((d >> shHi) == (prefix >> shHi))
                atomic_fetch_add_explicit(&hist[(d >> shLo) & 255u], 1u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // inclusive suffix sum: scan[i] = sum of hist[j] for j >= i
        if (tid < 256u) scan[tid] = atomic_load_explicit(&hist[tid], memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint off = 1u; off < 256u; off <<= 1) {
            uint v = 0u;
            if (tid < 256u && tid + off < 256u) v = scan[tid + off];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid < 256u) scan[tid] += v;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        // unique crossing bin: scan[b] >= need and scan[b+1] < need
        if (tid < 256u) {
            uint above = (tid + 1u < 256u) ? scan[tid + 1u] : 0u;
            if (scan[tid] >= need && above < need) selBin = tid;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            uint b = selBin;
            uint above = (b + 1u < 256u) ? scan[b + 1u] : 0u;
            need = need - above;
            prefix |= (b << shLo);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) { theta[0] = max(prefix, 1u); theta[1] = need; }
}

// count how many columns are at/above threshold, and branch-coincidence stats
kernel void k_census(device const uint* drive [[buffer(0)]],
                     device const uint* spk   [[buffer(1)]],
                     device const uint* theta [[buffer(2)]],
                     device atomic_uint* out  [[buffer(3)]],
                     constant Params&   P     [[buffer(4)]],
                     uint c [[thread_position_in_grid]])
{
    if (c < P.inCols) return;
    if (spk[c] != 0u) {
        atomic_fetch_add_explicit(&out[0], 1u, memory_order_relaxed);
        atomic_fetch_add_explicit(&out[1], (uint)popcount(spk[c]), memory_order_relaxed);
    }
}

// ---------------------------------------------------------------- plasticity
// bit-serial saturating +1 / -1 on a 4-bitplane counter, 128 synapses per uint4.
inline void bs_inc(thread uint4& a0, thread uint4& a1, thread uint4& a2, thread uint4& a3, uint4 m){
    uint4 c = m, t;
    t = a0 ^ c; c = a0 & c; a0 = t;
    t = a1 ^ c; c = a1 & c; a1 = t;
    t = a2 ^ c; c = a2 & c; a2 = t;
    t = a3 ^ c; c = a3 & c; a3 = t;
    a0 |= c; a1 |= c; a2 |= c; a3 |= c;                      // saturate at 15
}
inline void bs_dec(thread uint4& a0, thread uint4& a1, thread uint4& a2, thread uint4& a3, uint4 m){
    uint4 b = m, t;
    t = a0 ^ b; b = (~a0) & b; a0 = t;
    t = a1 ^ b; b = (~a1) & b; a1 = t;
    t = a2 ^ b; b = (~a2) & b; a2 = t;
    t = a3 ^ b; b = (~a3) & b; a3 = t;
    uint4 keep = ~b; a0 &= keep; a1 &= keep; a2 &= keep; a3 &= keep;   // saturate at 0
}

// ---- bit-serial comparison against a constant, MSB first ----
inline uint4 bs_lt(uint4 a3,uint4 a2,uint4 a1,uint4 a0, uint K){
    uint4 lt=uint4(0u), eq=uint4(0xFFFFFFFFu); uint kb;
    kb=(K>>3)&1u; if(kb) lt|=eq&(~a3); eq &= kb? a3:(~a3);
    kb=(K>>2)&1u; if(kb) lt|=eq&(~a2); eq &= kb? a2:(~a2);
    kb=(K>>1)&1u; if(kb) lt|=eq&(~a1); eq &= kb? a1:(~a1);
    kb=(K>>0)&1u; if(kb) lt|=eq&(~a0);
    return lt;
}
inline uint4 bs_gt(uint4 a3,uint4 a2,uint4 a1,uint4 a0, uint K){
    uint4 gt=uint4(0u), eq=uint4(0xFFFFFFFFu); uint kb;
    kb=(K>>3)&1u; if(!kb) gt|=eq&a3; eq &= kb? a3:(~a3);
    kb=(K>>2)&1u; if(!kb) gt|=eq&a2; eq &= kb? a2:(~a2);
    kb=(K>>1)&1u; if(!kb) gt|=eq&a1; eq &= kb? a1:(~a1);
    kb=(K>>0)&1u; if(!kb) gt|=eq&a0;
    return gt;
}
inline void bs_set(thread uint4& a0, thread uint4& a1, thread uint4& a2, thread uint4& a3,
                   uint4 m, uint C){
    uint4 keep=~m, z=uint4(0u);
    a0=(a0&keep)|(((C>>0)&1u)? m:z);
    a1=(a1&keep)|(((C>>1)&1u)? m:z);
    a2=(a2&keep)|(((C>>2)&1u)? m:z);
    a3=(a3&keep)|(((C>>3)&1u)? m:z);
}
// symmetric clip into [lo,hi]. Both masks are computed before either is applied.
inline void bs_clip(thread uint4& a0, thread uint4& a1, thread uint4& a2, thread uint4& a3,
                    uint lo, uint hi){
    uint4 l = bs_lt(a3,a2,a1,a0,lo);
    uint4 g = bs_gt(a3,a2,a1,a0,hi);
    bs_set(a0,a1,a2,a3,l,lo);
    bs_set(a0,a1,a2,a3,g,hi);
}

// verification target: run the real inc/dec/clip on device data so the host can
// compare against a scalar reference. op 0=inc 1=dec 2=clip
kernel void k_bstest(device uint4* A0 [[buffer(0)]], device uint4* A1 [[buffer(1)]],
                     device uint4* A2 [[buffer(2)]], device uint4* A3 [[buffer(3)]],
                     device const uint4* M [[buffer(4)]], constant uint4& cfg [[buffer(5)]],
                     uint g [[thread_position_in_grid]])
{
    uint4 a0=A0[g],a1=A1[g],a2=A2[g],a3=A3[g],m=M[g];
    if      (cfg.x==0u) bs_inc(a0,a1,a2,a3,m);
    else if (cfg.x==1u) bs_dec(a0,a1,a2,a3,m);
    else                bs_clip(a0,a1,a2,a3,cfg.y,cfg.z);
    A0[g]=a0;A1[g]=a1;A2[g]=a2;A3[g]=a3;
}

kernel void k_learn(device uint4* W0 [[buffer(0)]], device uint4* W1 [[buffer(1)]],
                    device uint4* W2 [[buffer(2)]], device uint4* W3 [[buffer(3)]],
                    device const uint*  conn   [[buffer(4)]],
                    device const uint4* spkIn  [[buffer(5)]],
                    device const uint*  spkOut [[buffer(6)]],
                    device const uint*  bbits  [[buffer(7)]],
                    constant Params&    P      [[buffer(8)]],
                    device const uint*  margOK [[buffer(9)]],
                    device atomic_uint* stats  [[buffer(10)]],
                    uint gid  [[thread_position_in_grid]],
                    uint lane [[thread_index_in_simdgroup]])
{
    if (P.phase == 3u) return;                               // free step with LTD suppressed
    uint nLTP = 0, nLTD = 0;                                 // synapse-update counters
    uint col = gid >> 5;
    uint b   = spkOut[col];
    if (b == 0u) return;                                     // silent column: whole simdgroup exits
    // margin gating: if the k-th and (k+1)-th drives tie, the winner set was
    // settled by lane index. Do not write that into the weights.
    if (P.marginGate != 0u && margOK[col] == 0u) return;
    bool post = ((b >> lane) & 1u) != 0u;
    uint bb   = bbits[gid];
    uint base = col * NSLOT;

    for (uint s = 0; s < NSLOT; ++s) {
        bool want = post && (((bb >> s) & 1u) != 0u);        // branch-local (clustered) plasticity
        uint vote = (uint)((simd_vote::vote_t)simd_ballot(want));
        if (vote == 0u) continue;                            // uniform: no lane learns this branch
        uint  g  = conn[base + s];
        uint  wb = (base + s) * SLOT_U4 * 32u + lane;
        for (uint j = 0; j < SLOT_U4; ++j) {
            uint4 sp = spkIn[g * SLOT_U4 + j];
            uint4 z  = uint4(0u);
            uint  idx = wb + j * 32u;
            uint4 a0 = W0[idx], a1 = W1[idx], a2 = W2[idx], a3 = W3[idx];
            uint4 co = want ? sp : z;                        // co-active pairs
            uint4 pcv = popcount(co);
            uint  nco = pcv.x + pcv.y + pcv.z + pcv.w;
            if (P.phase == 1u) { bs_inc(a0,a1,a2,a3, co); nLTP += nco; }
            else if (P.phase == 2u) {
                // C1b: LTD volume is scaled by repetition, because equal step
                // counts do NOT give equal update volume between the phases.
                for (uint rr = 0; rr < P.ltdRep; ++rr) bs_dec(a0,a1,a2,a3, co);
                nLTD += nco * P.ltdRep;
            }
            else {                                            // legacy rule
                bs_inc(a0,a1,a2,a3, co);
                bs_dec(a0,a1,a2,a3, (want && P.depress!=0u) ? (~sp) : z);
            }
            if (P.hetero != 0u && P.phase == 1u)
                bs_dec(a0,a1,a2,a3, want ? (~sp) : z);
            if (P.bandHi != 0u) bs_clip(a0,a1,a2,a3, P.bandLo, P.bandHi);
            W0[idx]=a0; W1[idx]=a1; W2[idx]=a2; W3[idx]=a3;
        }
    }
    if (nLTP) atomic_fetch_add_explicit(&stats[8],  nLTP, memory_order_relaxed);
    if (nLTD) atomic_fetch_add_explicit(&stats[9],  nLTD, memory_order_relaxed);
    if (nLTP) atomic_fetch_add_explicit(&stats[10], 1u,   memory_order_relaxed);
    if (nLTD) atomic_fetch_add_explicit(&stats[11], 1u,   memory_order_relaxed);
}

kernel void k_clear(device uint* p [[buffer(0)]], uint g [[thread_position_in_grid]]) { p[g] = 0u; }

// A4b: per-neuron count of recurrent synapses still at the 5/6 initializer peaks,
// i.e. never touched by plasticity. value 5 = 0101, value 6 = 0110 in (W3..W0).
kernel void k_untouched(device const uint4* W0 [[buffer(0)]], device const uint4* W1 [[buffer(1)]],
                        device const uint4* W2 [[buffer(2)]], device const uint4* W3 [[buffer(3)]],
                        device uint* out [[buffer(4)]], constant uint& inBr [[buffer(5)]],
                        uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]])
{
    uint col = gid >> 5, base = col * NSLOT, u = 0;
    for (uint s = inBr; s < NSLOT; ++s) {
        uint wb = (base + s) * SLOT_U4 * 32u + lane;
        for (uint j = 0; j < SLOT_U4; ++j) {
            uint idx = wb + j * 32u;
            uint4 a0=W0[idx], a1=W1[idx], a2=W2[idx], a3=W3[idx];
            uint4 v5 = (~a3) & a2 & (~a1) & a0;
            uint4 v6 = (~a3) & a2 & a1 & (~a0);
            uint4 pc = popcount(v5 | v6);
            u += pc.x + pc.y + pc.z + pc.w;
        }
    }
    out[gid] = u;
}

// Functional recurrent density split by whether the PRESYNAPTIC partner belongs to
// the same assembly as the postsynaptic neuron. Decides whether drift is LTD erosion
// of the within-assembly scaffold, or a purely dynamical failure.
kernel void k_within(device const uint4* W3   [[buffer(0)]],
                     device const uint*  conn [[buffer(1)]],
                     device const uint*  ball [[buffer(2)]],   // NCOL ballots of ONE assembly
                     device const uint4* ballV[[buffer(3)]],   // same bits, uint4 view
                     constant uint2&     cfg  [[buffer(4)]],   // (inBr, inCols)
                     device atomic_uint* out  [[buffer(5)]],
                     device const uint*  mult [[buffer(6)]],   // assemblies containing each neuron
                     device const uint*  postMask [[buffer(7)]], // 0 = use ballot, else select postsyn
                     constant uint&      useMask  [[buffer(8)]],
                     uint gid  [[thread_position_in_grid]],
                     uint lane [[thread_index_in_simdgroup]])
{
    uint col = gid >> 5;
    if (col < cfg.y) return;
    if (useMask != 0u) { if (postMask[gid] == 0u) return; }
    else if (((ball[col] >> lane) & 1u) == 0u) return;      // postsyn must be a member
    uint base = col * NSLOT, fIn=0, pIn=0, fOut=0, pOut=0;
    for (uint s = cfg.x; s < NSLOT; ++s) {
        uint g  = conn[base + s];
        uint wb = (base + s) * SLOT_U4 * 32u + lane;
        for (uint j = 0; j < SLOT_U4; ++j) {
            uint4 ap = ballV[g * SLOT_U4 + j];
            uint4 w  = W3[wb + j * 32u];
            uint4 a = popcount(w & ap);    fIn  += a.x+a.y+a.z+a.w;
            uint4 b = popcount(ap);        pIn  += b.x+b.y+b.z+b.w;
            uint4 c = popcount(w & (~ap)); fOut += c.x+c.y+c.z+c.w;
            uint4 d = popcount(~ap);       pOut += d.x+d.y+d.z+d.w;
        }
    }
    atomic_fetch_add_explicit(&out[0], fIn,  memory_order_relaxed);
    atomic_fetch_add_explicit(&out[1], pIn,  memory_order_relaxed);
    atomic_fetch_add_explicit(&out[2], fOut, memory_order_relaxed);
    atomic_fetch_add_explicit(&out[3], pOut, memory_order_relaxed);
    // split by whether this postsynaptic member is SHARED with another assembly
    uint b = (mult[gid] > 1u) ? 4u : 6u;
    atomic_fetch_add_explicit(&out[b+0u], fIn, memory_order_relaxed);
    atomic_fetch_add_explicit(&out[b+1u], pIn, memory_order_relaxed);
    atomic_fetch_add_explicit(&out[(mult[gid]>1u)?8u:9u], 1u, memory_order_relaxed);
}

// STEP A1: functional in-degree of every neuron over its RECURRENT branches only.
// popcount of the MSB plane is exactly the number of synapses the forward pass sees.
kernel void k_indeg(device const uint4* W3 [[buffer(0)]],
                    device uint*        out [[buffer(1)]],
                    constant uint&      inBr [[buffer(2)]],
                    uint gid  [[thread_position_in_grid]],
                    uint lane [[thread_index_in_simdgroup]])
{
    uint col = gid >> 5, base = col * NSLOT, d = 0;
    for (uint s = inBr; s < NSLOT; ++s) {
        uint wb = (base + s) * SLOT_U4 * 32u + lane;
        for (uint j = 0; j < SLOT_U4; ++j) {
            uint4 pc = popcount(W3[wb + j * 32u]);
            d += pc.x + pc.y + pc.z + pc.w;
        }
    }
    out[gid] = d;
}

// ---------------------------------------------------------------- STEP 0 diagnostic
// Histogram of the 4-bit synaptic counters, split by whether the pre/post pair
// belongs to the same trained assembly. No per-bit iteration: the set of
// synapses holding value v is a 4-plane bitwise equality mask, so 16 popcounts
// give the whole distribution of a 128-synapse word.
//   cat 0 = within-assembly  (pre and post share at least one assembly)
//   cat 1 = between-assembly (pre is in some assembly, but not one post is in)
//   cat 2 = all synapses of this branch class
kernel void k_wdiag(device const uint4* W0 [[buffer(0)]], device const uint4* W1 [[buffer(1)]],
                    device const uint4* W2 [[buffer(2)]], device const uint4* W3 [[buffer(3)]],
                    device const uint*  conn    [[buffer(4)]],
                    device const uint*  memb    [[buffer(5)]],   // per-neuron assembly bitmask
                    device const uint4* asmSpk  [[buffer(6)]],   // [NSTIM][NCOL/4] ballots
                    constant uint4&     cfg     [[buffer(7)]],   // (nstim, inBr, branchSel, ncolDiv4)
                    device atomic_uint* hist    [[buffer(8)]],   // 3*16
                    uint gid  [[thread_position_in_grid]],
                    uint lane [[thread_index_in_simdgroup]])
{
    uint mypost = memb[gid];
    if (mypost == 0u) return;                  // only postsynaptic neurons that joined an assembly
    uint nstim = cfg.x, inBr = cfg.y, sel = cfg.z, ncol4 = cfg.w;
    uint col = gid >> 5, base = col * NSLOT;
    uint acc[48];
    for (uint i = 0; i < 48u; ++i) acc[i] = 0u;

    for (uint s = 0; s < NSLOT; ++s) {
        bool isAff = (s < inBr);
        if (sel == 0u && isAff)  continue;     // recurrent branches only
        if (sel == 1u && !isAff) continue;     // afferent branches only
        uint g  = conn[base + s];
        uint wb = (base + s) * SLOT_U4 * 32u + lane;
        for (uint j = 0; j < SLOT_U4; ++j) {
            uint idx = wb + j * 32u;
            uint4 a0=W0[idx], a1=W1[idx], a2=W2[idx], a3=W3[idx];
            uint4 within = uint4(0u), anyp = uint4(0u);
            for (uint q = 0; q < nstim; ++q) {
                uint4 ap = asmSpk[q * ncol4 + 2u * g + j];
                anyp |= ap;
                if (((mypost >> q) & 1u) != 0u) within |= ap;
            }
            uint4 between = anyp & (~within);
            for (uint v = 0; v < 16u; ++v) {
                uint4 eq = ((v&1u)? a0 : ~a0) & ((v&2u)? a1 : ~a1)
                         & ((v&4u)? a2 : ~a2) & ((v&8u)? a3 : ~a3);
                uint4 pw = popcount(eq & within);
                uint4 pb = popcount(eq & between);
                uint4 pa = popcount(eq);
                acc[ 0u+v] += pw.x+pw.y+pw.z+pw.w;
                acc[16u+v] += pb.x+pb.y+pb.z+pb.w;
                acc[32u+v] += pa.x+pa.y+pa.z+pa.w;
            }
        }
    }
    for (uint i = 0; i < 48u; ++i)
        if (acc[i]) atomic_fetch_add_explicit(&hist[i], acc[i], memory_order_relaxed);
}

// count functional synapses (MSB set) for diagnostics
kernel void k_wstat(device const uint4* W3 [[buffer(0)]], device atomic_uint* out [[buffer(1)]],
                    uint g [[thread_position_in_grid]])
{
    uint4 pc = popcount(W3[g]);
    atomic_fetch_add_explicit(out, pc.x+pc.y+pc.z+pc.w, memory_order_relaxed);
}
)MSL";

// ============================================================================
struct Params {
    uint32_t k, theta, dthr, boost, mode, inCols, step, learn, depress, targetCols, clamp, freeze, biasUp, biasDown, biasShift, histOn, useBias, phase, bandLo, bandHi, marginGate, hetero, ltdRep, biasDownEvery, facil, _pad6;
};

static id<MTLDevice> D; static id<MTLCommandQueue> Q; static id<MTLLibrary> L;
static std::string __gsrc;
static id<MTLComputePipelineState> P_(NSString* n){NSError*e=nil;
    id<MTLComputePipelineState> p=[D newComputePipelineStateWithFunction:[L newFunctionWithName:n] error:&e];
    if(!p){printf("pipeline %s: %s\n",n.UTF8String,e.localizedDescription.UTF8String);exit(3);} return p;}

static uint32_t argu(int argc, char** argv, const char* key, uint32_t dflt){
    size_t kl = strlen(key);
    uint32_t v = dflt;
    // LAST occurrence wins, so a later argument overrides an earlier one
    for (int i=1;i<argc;++i)
        if (!strncmp(argv[i],key,kl) && argv[i][kl]=='=') v = (uint32_t)atoi(argv[i]+kl+1);
    return v;
}

int main(int argc, char** argv) {
@autoreleasepool {
    D = MTLCreateSystemDefaultDevice(); Q = [D newCommandQueue];
    // shader geometry is specialised by string substitution (runtime compile is free)
    __gsrc = kSrc;

    {
        char nb[64]; snprintf(nb,sizeof nb,"#define NCOLD     %uu", NCOL);
        size_t pos=__gsrc.find("#define NCOLD     4096u");
        if(pos!=std::string::npos) __gsrc.replace(pos, strlen("#define NCOLD     4096u"), nb);
        NSError* e=nil; MTLCompileOptions* o=[MTLCompileOptions new]; o.fastMathEnabled=YES;
        L = [D newLibraryWithSource:[NSString stringWithUTF8String:__gsrc.c_str()] options:o error:&e];
        if (!L) { printf("SHADER COMPILE FAIL:\n%s\n", e.localizedDescription.UTF8String); return 2; }
    }
    id<MTLComputePipelineState> pInitW=P_(@"k_init_w"), pInitC=P_(@"k_init_conn"),
        pDrive=P_(@"k_drive"), pEmit=P_(@"k_emit"), pTopKS=P_(@"k_topk_serial"), pTopKP=P_(@"k_topk_par"), pLearn=P_(@"k_learn"), pBias=P_(@"k_bias"), pWDiag=P_(@"k_wdiag"), pIndeg=P_(@"k_indeg"), pUntouched=P_(@"k_untouched"), pBsTest=P_(@"k_bstest"), pWithin=P_(@"k_within"),
        pClear=P_(@"k_clear"), pWStat=P_(@"k_wstat");

    // ---- tunables ----
    uint32_t nAnd     = argu(argc,argv,"nand",     4);   // init functional density = 2^-nand
    uint32_t targetC  = argu(argc,argv,"target", 256);   // homeostatic target active columns
    uint32_t K        = argu(argc,argv,"k",        6);
    uint32_t dthr     = argu(argc,argv,"dthr",     2);
    uint32_t boost    = argu(argc,argv,"boost",    4);
    uint32_t mode     = argu(argc,argv,"mode",     1);
    uint32_t EPOCHS   = argu(argc,argv,"epochs",  30);
    uint32_t T        = argu(argc,argv,"T",       10);
    uint32_t CLAMP    = argu(argc,argv,"clampsteps", 5); // cue steps at test; then released
    uint32_t NSTIM    = argu(argc,argv,"stim",    12);
    uint32_t quiet    = argu(argc,argv,"quiet",    0);
    uint32_t seed     = argu(argc,argv,"seed",     0);   // replication seed
    uint32_t split    = argu(argc,argv,"split",    0);   // shared vs exclusive member analysis
    uint32_t unionchar= argu(argc,argv,"unionchar",0);  // characterise the visited union
    uint32_t biasUp   = argu(argc,argv,"biasup",   0);   // 0 = ablate intrinsic homeostasis
    uint32_t biasDown = argu(argc,argv,"biasdown", 1);
    uint32_t biasShift= argu(argc,argv,"biasshift",4);
    uint32_t biasDownEvery = argu(argc,argv,"biasdownevery",1);
    uint32_t inBranches=argu(argc,argv,"inbr",     8);   // branches dedicated to the afferent path
    uint32_t STIM_COLS =argu(argc,argv,"stimcols",128);
    IN_COLS            =argu(argc,argv,"incols",  512);
    NCOL               =argu(argc,argv,"ncol",   4096);
    N = NCOL*NPC; NGRP = NCOL/COLS_PER_GRP;
    PLANE_U4 = (size_t)NCOL*NSLOT*SLOT_U4*32; PLANE_B = PLANE_U4*16;
    uint32_t diag     = argu(argc,argv,"diag",     0);
    uint32_t profile  = argu(argc,argv,"profile",  0);
    uint32_t topkPar  = argu(argc,argv,"topkpar",  1);   // 1 = parallel selection (STEP B)
    uint32_t vtopk    = argu(argc,argv,"vtopk",    0);   // verify parallel == serial bit-exactly
    uint32_t contrast = argu(argc,argv,"contrast", 0);   // C1: contrastive clamped/free phases
    uint32_t bandLo   = argu(argc,argv,"bandlo",   5);
    uint32_t bandHi   = argu(argc,argv,"bandhi",  11);
    uint32_t margGate = argu(argc,argv,"margingate",1);
    uint32_t hetero   = argu(argc,argv,"hetero",   0);
    uint32_t vbs      = argu(argc,argv,"vbs",      0);   // verify bit-serial vs scalar reference
    uint32_t ltdRep   = argu(argc,argv,"ltdrep",   1);   // C1b: LTD volume multiplier
    uint32_t ltdEvery = argu(argc,argv,"ltdevery", 1);   // C1b: apply LTD every Nth free step
    uint32_t kTest    = argu(argc,argv,"ktest",    0);   // READOUT-ONLY k override (0 = use K)
    uint32_t facil    = argu(argc,argv,"facil",    0);   // READOUT-ONLY facilitation bonus
    uint32_t deep     = argu(argc,argv,"deep",     0);   // multiplicity + heavy-tail analysis
    uint32_t wave     = argu(argc,argv,"wave",     0);   // long free-run wave characterisation
    uint32_t waveT    = argu(argc,argv,"wavet",   35);
    // clamped/free split DURING TRAINING. Defaults to T (fully clamped) to keep the
    // legacy protocol; contrastive requires this to be < T or there is no free phase.
    uint32_t trainClamp = argu(argc,argv,"trainclamp", contrast ? CLAMP : T);
    if (biasUp == 1u) {   // "auto": set from the implied target firing rate
        double pr = (double)targetC*K/(double)(NCOL-IN_COLS)/NPC;
        biasUp = (uint32_t)std::max(2.0, (1.0-pr)/pr*biasDown);
    }

    if(!quiet){
    printf("======================================================================\n");
    printf(" AXON v0  -  columnar assembly engine, Apple M1 Pro\n");
    printf("======================================================================\n");
    printf("  neurons %u (%u cols x %u) | branches %u x %u presyn | fan-in %u | %.2f G syn\n",
           N,NCOL,NPC,NSLOT,PRESYN,FANIN,(double)N*FANIN/1e9);
    printf("  weights 4 planes x %.0f MB = %.0f MB | groups %u\n",
           PLANE_B/1048576.0, 4*PLANE_B/1048576.0, NGRP);
    printf("  nand=%u target=%u k=%u dthr=%u mode=%u epochs=%u T=%u clamp=%u stim=%u biasUp=%u inbr=%u stimcols=%u\n\n",
           nAnd,targetC,K,dthr,mode,EPOCHS,T,CLAMP,NSTIM,biasUp,inBranches,STIM_COLS);
    }

    auto mk=[&](size_t n, MTLResourceOptions m){ return [D newBufferWithLength:n options:m]; };
    id<MTLBuffer> W0=mk(PLANE_B,MTLResourceStorageModePrivate), W1=mk(PLANE_B,MTLResourceStorageModePrivate),
                  W2=mk(PLANE_B,MTLResourceStorageModePrivate), W3=mk(PLANE_B,MTLResourceStorageModePrivate);
    id<MTLBuffer> bConn=mk((size_t)NCOL*NSLOT*4, MTLResourceStorageModePrivate);
    id<MTLBuffer> spkA =mk(NCOL*4, MTLResourceStorageModeShared);
    id<MTLBuffer> spkB =mk(NCOL*4, MTLResourceStorageModeShared);
    id<MTLBuffer> bDrv =mk(NCOL*4, MTLResourceStorageModeShared);
    id<MTLBuffer> bBits=mk((size_t)N*4, MTLResourceStorageModePrivate);
    id<MTLBuffer> bAcc =mk((size_t)N*4, MTLResourceStorageModePrivate);
    id<MTLBuffer> bBias=mk((size_t)N*4, MTLResourceStorageModeShared);
    id<MTLBuffer> bMarg=mk(NCOL*4, MTLResourceStorageModeShared);
    id<MTLBuffer> bRefBall=mk(NCOL*4, MTLResourceStorageModeShared);
    id<MTLBuffer> bDriveLog=mk(256*4, MTLResourceStorageModeShared);
    id<MTLBuffer> bWCnt=mk(128, MTLResourceStorageModeShared);
    id<MTLBuffer> bMult=mk((size_t)N*4, MTLResourceStorageModeShared);
    id<MTLBuffer> bStim=mk(NCOL*4, MTLResourceStorageModeShared);
    id<MTLBuffer> bStat=mk(4096, MTLResourceStorageModeShared);
    const uint32_t MAXT=64;
    id<MTLBuffer> bHist=mk((size_t)MAXT*NCOL*4, MTLResourceStorageModeShared);
    id<MTLBuffer> bTheta=mk(64, MTLResourceStorageModeShared);

    { id<MTLCommandBuffer> cb=[Q commandBuffer]; id<MTLComputeCommandEncoder> ce=[cb computeCommandEncoder];
      [ce setComputePipelineState:pInitW];
      [ce setBuffer:W0 offset:0 atIndex:0];[ce setBuffer:W1 offset:0 atIndex:1];
      [ce setBuffer:W2 offset:0 atIndex:2];[ce setBuffer:W3 offset:0 atIndex:3];
      [ce setBytes:&nAnd length:4 atIndex:4];[ce setBytes:&seed length:4 atIndex:5];
      [ce dispatchThreads:MTLSizeMake(PLANE_U4,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
      [ce setComputePipelineState:pInitC];
      [ce setBuffer:bConn offset:0 atIndex:0];
      uint32_t cfg[4] = { NGRP, IN_COLS/COLS_PER_GRP, NSLOT, inBranches };
      [ce setBytes:cfg length:16 atIndex:1];[ce setBytes:&seed length:4 atIndex:2];
      [ce dispatchThreads:MTLSizeMake(NCOL*NSLOT,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
      [ce endEncoding];[cb commit];[cb waitUntilCompleted]; }
    memset(bTheta.contents,0,64); memset(bStat.contents,0,4096);

    auto wstat=[&]()->double{
        memset(bStat.contents,0,4096);
        id<MTLCommandBuffer> cb=[Q commandBuffer]; id<MTLComputeCommandEncoder> ce=[cb computeCommandEncoder];
        [ce setComputePipelineState:pWStat];[ce setBuffer:W3 offset:0 atIndex:0];[ce setBuffer:bStat offset:0 atIndex:1];
        [ce dispatchThreads:MTLSizeMake(PLANE_U4,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [ce endEncoding];[cb commit];[cb waitUntilCompleted];
        return 100.0*((uint32_t*)bStat.contents)[0]/((double)N*FANIN);
    };

    std::mt19937 rng(7 + seed*104729);
    std::vector<std::vector<uint32_t>> stimCols(NSTIM);
    for (uint32_t p=0;p<NSTIM;++p){
        std::vector<uint32_t> all(IN_COLS); for(uint32_t i=0;i<IN_COLS;++i) all[i]=i;
        std::shuffle(all.begin(),all.end(),rng);
        stimCols[p].assign(all.begin(), all.begin()+STIM_COLS);
    }
    auto loadStim=[&](uint32_t p, double keep, uint32_t seed){
        uint32_t* s=(uint32_t*)bStim.contents; memset(s,0,NCOL*4);
        std::mt19937 r(seed); std::uniform_real_distribution<double> u(0,1);
        for (uint32_t c : stimCols[p]) if (u(r) <= keep) s[c] = 0xFFFFFFFFu;
    };

    id<MTLComputePipelineState> __strong pTopKsel[1] = { topkPar ? pTopKP : pTopKS };
    uint32_t topkTGv[1] = { topkPar ? (uint32_t)std::min<NSUInteger>(1024, pTopKP.maxTotalThreadsPerThreadgroup) : 256u };
    #define pTopK  (pTopKsel[0])
    #define topkTG (topkTGv[0])
    Params P{}; P.k=K; P.dthr=dthr; P.boost=boost; P.mode=mode; P.inCols=IN_COLS;
    P.targetCols=targetC; P.biasUp=biasUp; P.biasDown=biasDown; P.biasShift=biasShift;
    P.bandLo=bandLo; P.bandHi=contrast?bandHi:0u; P.marginGate=margGate; P.hetero=hetero; P.ltdRep=ltdRep; P.biasDownEvery=biasDownEvery;

    // T steps, batched into ONE command buffer. clampSteps of cue, then released.
    auto present=[&](uint32_t p, double keep, uint32_t T_, uint32_t clampSteps,
                     bool learn, bool freeze, uint32_t seed, uint32_t useBias=2u)->double{
        // READOUT-ONLY k override: weights frozen, only the winner-take-all width changes.
        P.k = learn ? K : ((kTest != 0u) ? kTest : K);
        P.facil = learn ? 0u : facil;   // readout-only, weights untouched
        // READOUT GATE: intrinsic bias is a learning-time mechanism only.
        // useBias==2 means "auto" = on while learning, off at readout.
        P.useBias = (useBias==2u) ? (learn ? 1u : 0u) : useBias;
        loadStim(p, keep, seed);
        memset(bStat.contents,0,4096);
        P.histOn=1u;
        { id<MTLCommandBuffer> cb=[Q commandBuffer]; id<MTLComputeCommandEncoder> ce=[cb computeCommandEncoder];
          [ce setComputePipelineState:pClear];
          [ce setBuffer:spkA offset:0 atIndex:0];
          [ce dispatchThreads:MTLSizeMake(NCOL,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
          [ce setBuffer:spkB offset:0 atIndex:0];
          [ce dispatchThreads:MTLSizeMake(NCOL,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
          [ce endEncoding];[cb commit];[cb waitUntilCompleted]; }

        auto t0=std::chrono::steady_clock::now();
        id<MTLCommandBuffer> cb=[Q commandBuffer];
        id<MTLComputeCommandEncoder> ce=[cb computeCommandEncoder];
        for (uint32_t t=0;t<T_;++t) {
            id<MTLBuffer> in  = (t&1)?spkB:spkA;
            id<MTLBuffer> out = (t&1)?spkA:spkB;
            P.step=t; P.learn=learn?1u:0u; P.freeze=freeze?1u:0u;
            P.clamp = (t < clampSteps) ? 1u : 0u;
            P.depress = (learn && (t%2==1)) ? 1u : 0u;
            // C1: clamped phase potentiates, free phase depresses the co-active pairs
            P.phase = (!learn || !contrast) ? 0u
                    : ((t < clampSteps) ? 1u : (((t - clampSteps) % ltdEvery == 0u) ? 2u : 3u));
            [ce setComputePipelineState:pDrive];
            [ce setBuffer:W3 offset:0 atIndex:0];  [ce setBuffer:bConn offset:0 atIndex:1];
            [ce setBuffer:in offset:0 atIndex:2];  [ce setBuffer:bDrv offset:0 atIndex:3];
            [ce setBuffer:bBits offset:0 atIndex:4];[ce setBuffer:bAcc offset:0 atIndex:5];
            [ce setBuffer:bBias offset:0 atIndex:7];[ce setBuffer:in offset:0 atIndex:8];
            [ce setBytes:&P length:sizeof(P) atIndex:6];
            [ce dispatchThreads:MTLSizeMake(N,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];

            [ce setComputePipelineState:pTopK];
            [ce setBuffer:bDrv offset:0 atIndex:0];[ce setBuffer:bTheta offset:0 atIndex:1];
            [ce setBytes:&P length:sizeof(P) atIndex:2];
            uint32_t ncol_=NCOL; [ce setBytes:&ncol_ length:4 atIndex:3];
            [ce dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(topkTG,1,1)];

            [ce setComputePipelineState:pEmit];
            [ce setBuffer:bAcc offset:0 atIndex:0];  [ce setBuffer:bDrv offset:0 atIndex:1];
            [ce setBuffer:bTheta offset:0 atIndex:2];[ce setBuffer:bStim offset:0 atIndex:3];
            [ce setBuffer:bBits offset:0 atIndex:4]; [ce setBuffer:out offset:0 atIndex:5];
            [ce setBytes:&P length:sizeof(P) atIndex:6];[ce setBuffer:bStat offset:0 atIndex:7];
            [ce setBuffer:bHist offset:0 atIndex:8];[ce setBuffer:bMarg offset:0 atIndex:9];[ce setBuffer:bRefBall offset:0 atIndex:10];[ce setBuffer:bDriveLog offset:0 atIndex:11];[ce setBuffer:bMult offset:0 atIndex:12];
            [ce dispatchThreads:MTLSizeMake(N,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];

            if (learn) {
                [ce setComputePipelineState:pBias];
                [ce setBuffer:out offset:0 atIndex:0];[ce setBuffer:bBias offset:0 atIndex:1];
                [ce setBytes:&P length:sizeof(P) atIndex:2];
                [ce dispatchThreads:MTLSizeMake(N,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            }

            if (learn) {
                [ce setComputePipelineState:pLearn];
                [ce setBuffer:W0 offset:0 atIndex:0];[ce setBuffer:W1 offset:0 atIndex:1];
                [ce setBuffer:W2 offset:0 atIndex:2];[ce setBuffer:W3 offset:0 atIndex:3];
                [ce setBuffer:bConn offset:0 atIndex:4];[ce setBuffer:in offset:0 atIndex:5];
                [ce setBuffer:out offset:0 atIndex:6];[ce setBuffer:bBits offset:0 atIndex:7];
                [ce setBytes:&P length:sizeof(P) atIndex:8];[ce setBuffer:bMarg offset:0 atIndex:9];[ce setBuffer:bStat offset:0 atIndex:10];
                [ce dispatchThreads:MTLSizeMake(N,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            }
        }
        [ce endEncoding];[cb commit];[cb waitUntilCompleted];
        double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();
        return ms/T_;
    };

    auto assemblyOf=[&](uint32_t T_)->std::vector<uint32_t>{
        const uint32_t* s=(const uint32_t*)((T_&1)?spkB.contents:spkA.contents);
        std::vector<uint32_t> v;
        for (uint32_t c=IN_COLS;c<NCOL;++c) if (s[c]) v.push_back(c);
        return v;
    };
    auto overlap=[&](const std::vector<uint32_t>&a,const std::vector<uint32_t>&b)->double{
        if(a.empty()||b.empty()) return 0.0;
        std::vector<uint32_t> in;
        std::set_intersection(a.begin(),a.end(),b.begin(),b.end(),std::back_inserter(in));
        return (double)in.size()/std::sqrt((double)a.size()*b.size());
    };

    if (vbs) {
        printf("\n========== BIT-SERIAL VERIFICATION vs SCALAR REFERENCE ==========\n");
        const uint32_t NW = 1<<14;                    // uint4 words = 2M synapses
        id<MTLBuffer> t0=mk(NW*16,MTLResourceStorageModeShared), t1=mk(NW*16,MTLResourceStorageModeShared),
                      t2=mk(NW*16,MTLResourceStorageModeShared), t3=mk(NW*16,MTLResourceStorageModeShared),
                      tm=mk(NW*16,MTLResourceStorageModeShared);
        std::mt19937 r(424242);
        auto runOp=[&](uint32_t op,uint32_t lo,uint32_t hi,std::vector<uint8_t>& vin,
                       std::vector<uint8_t>& mk_,std::vector<uint8_t>& vout){
            uint32_t *p0=(uint32_t*)t0.contents,*p1=(uint32_t*)t1.contents,
                     *p2=(uint32_t*)t2.contents,*p3=(uint32_t*)t3.contents,*pm=(uint32_t*)tm.contents;
            size_t nbits=(size_t)NW*4*32;
            vin.assign(nbits,0); mk_.assign(nbits,0);
            memset(p0,0,NW*16);memset(p1,0,NW*16);memset(p2,0,NW*16);memset(p3,0,NW*16);memset(pm,0,NW*16);
            for(size_t i=0;i<nbits;++i){
                uint8_t v=r()&15u, m=r()&1u; vin[i]=v; mk_[i]=m;
                size_t w=i>>5, b=i&31;
                if(v&1) p0[w]|=1u<<b; if(v&2) p1[w]|=1u<<b;
                if(v&4) p2[w]|=1u<<b; if(v&8) p3[w]|=1u<<b;
                if(m)   pm[w]|=1u<<b;
            }
            uint32_t cfg[4]={op,lo,hi,0};
            id<MTLCommandBuffer> cb=[Q commandBuffer]; id<MTLComputeCommandEncoder> ce=[cb computeCommandEncoder];
            [ce setComputePipelineState:pBsTest];
            [ce setBuffer:t0 offset:0 atIndex:0];[ce setBuffer:t1 offset:0 atIndex:1];
            [ce setBuffer:t2 offset:0 atIndex:2];[ce setBuffer:t3 offset:0 atIndex:3];
            [ce setBuffer:tm offset:0 atIndex:4];[ce setBytes:cfg length:16 atIndex:5];
            [ce dispatchThreads:MTLSizeMake(NW,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [ce endEncoding];[cb commit];[cb waitUntilCompleted];
            vout.assign(nbits,0);
            for(size_t i=0;i<nbits;++i){ size_t w=i>>5,b=i&31;
                vout[i]=((p0[w]>>b)&1)|(((p1[w]>>b)&1)<<1)|(((p2[w]>>b)&1)<<2)|(((p3[w]>>b)&1)<<3); }
        };
        std::vector<uint8_t> vin,mk_,vout;
        struct T{const char*n;uint32_t op,lo,hi;};
        T tests[]={{"bs_inc (carry, saturate 15)",0,0,0},{"bs_dec (BORROW, saturate 0)",1,0,0},
                   {"bs_clip [5,11]",2,5,11},{"bs_clip [6,10]",2,6,10},{"bs_clip [4,12]",2,4,12}};
        for (auto& t : tests) {
            runOp(t.op,t.lo,t.hi,vin,mk_,vout);
            size_t bad=0, firstBad=SIZE_MAX;
            for(size_t i=0;i<vin.size();++i){
                int exp;
                if(t.op==0) exp = mk_[i] ? std::min(15,(int)vin[i]+1) : vin[i];
                else if(t.op==1) exp = mk_[i] ? std::max(0,(int)vin[i]-1) : vin[i];
                else exp = std::min((int)t.hi,std::max((int)t.lo,(int)vin[i]));
                if(exp!=(int)vout[i]){ if(bad==0) firstBad=i; bad++; }
            }
            printf("  %-30s %9zu values  mismatches %zu  %s\n", t.n, vin.size(), bad,
                   bad? "FAIL":"EXACT");
            if(bad) printf("      first at i=%zu in=%u mask=%u got=%u\n",
                           firstBad,vin[firstBad],mk_[firstBad],vout[firstBad]);
        }
        printf("================================================================\n");
    }

    if (vtopk) {
        printf("\n=============== STEP B: parallel top-K verification ===============\n");
        printf("  serial TG=256 threads | parallel TG=%u threads (max %lu)\n",
               (uint32_t)std::min<NSUInteger>(1024,pTopKP.maxTotalThreadsPerThreadgroup),
               (unsigned long)pTopKP.maxTotalThreadsPerThreadgroup);
        std::vector<uint32_t> hs((size_t)T*NCOL), hp((size_t)T*NCOL);
        uint32_t mismatch=0, thetaMismatch=0;
        std::vector<uint32_t> thS, thP;
        for (uint32_t q=0;q<NSTIM;++q) {
            pTopK=pTopKS; topkTG=256;
            present(q,1.0,T,CLAMP,false,true,9000+q);
            memcpy(hs.data(), bHist.contents, (size_t)T*NCOL*4);
            thS.push_back(((uint32_t*)bTheta.contents)[0]);
            pTopK=pTopKP; topkTG=(uint32_t)std::min<NSUInteger>(1024,pTopKP.maxTotalThreadsPerThreadgroup);
            present(q,1.0,T,CLAMP,false,true,9000+q);
            memcpy(hp.data(), bHist.contents, (size_t)T*NCOL*4);
            thP.push_back(((uint32_t*)bTheta.contents)[0]);
            for (size_t i=0;i<hs.size();++i) if (hs[i]!=hp[i]) mismatch++;
            if (thS.back()!=thP.back()) thetaMismatch++;
        }
        printf("  winner-set words compared: %zu   MISMATCHES: %u  -> %s\n",
               (size_t)T*NCOL*NSTIM, mismatch, mismatch?"FAIL":"BIT-IDENTICAL");
        printf("  final theta compared: %u stimuli, mismatches %u\n", NSTIM, thetaMismatch);
        // timing
        auto timeIt=[&](bool par)->double{
            pTopK = par?pTopKP:pTopKS;
            topkTG = par?(uint32_t)std::min<NSUInteger>(1024,pTopKP.maxTotalThreadsPerThreadgroup):256u;
            double best=1e30;
            for(int r=0;r<5;++r) best=std::min(best,present(0,1.0,T,CLAMP,false,true,9999));
            return best; };
        double ts=timeIt(false), tp=timeIt(true);
        printf("  full step: serial %.1f us | parallel %.1f us | saved %.1f us (%.1f%% of step)\n",
               ts*1000.0, tp*1000.0, (ts-tp)*1000.0, 100.0*(ts-tp)/ts);
        pTopK = topkPar?pTopKP:pTopKS;
        topkTG = topkPar?(uint32_t)std::min<NSUInteger>(1024,pTopKP.maxTotalThreadsPerThreadgroup):256u;
        printf("==================================================================\n");
    }

    double d0 = wstat();
    if(!quiet) printf("initial functional synapse density: %.3f%%  (fan-in %.0f functional)\n\n",
                      d0, d0/100.0*FANIN);

    // settle homeostasis without learning
    for (uint32_t i=0;i<10;++i) present(i%NSTIM, 1.0, T, T, false, false, 999);
    if(!quiet){
        const uint32_t* st=(const uint32_t*)bStat.contents;
        printf("settled: theta=%u  activeCols/step=%.0f  activeNeurons/step=%.0f\n\n",
               ((uint32_t*)bTheta.contents)[0], (double)st[0]/T, (double)st[1]/T);
    }

    double msAcc=0; int msN=0;
    for (uint32_t ep=0; ep<EPOCHS; ++ep) {
        for (uint32_t p=0;p<NSTIM;++p){ msAcc+=present(p,1.0,T,trainClamp,true,false,1000+p); msN++; }
        if(!quiet && (ep%5==4 || ep==0)){
            const uint32_t* st=(const uint32_t*)bStat.contents;
            double ltp=st[8], ltd=st[9]; uint32_t tp=st[10], td=st[11];
            double wd=wstat();          // NOTE: wstat() memsets bStat, so read counters first
            printf("  ep %3u  wdens %7.3f%%  LTP %10.0f  LTD %10.0f  LTD/LTP %6.3f  learners %u/%u\n",
                   ep+1, wd, ltp, ltd, ltp>0?ltd/ltp:0.0, tp, td);
        }
        if(false){
            const uint32_t* st=(const uint32_t*)bStat.contents;
            printf("  epoch %2u  theta=%6u  activeCols/step=%6.0f  activeNeu/step=%7.0f  branchCoinc=%.2f  wdens=%.3f%%\n",
                   ep+1,((uint32_t*)bTheta.contents)[0],(double)st[0]/T,(double)st[1]/T,
                   st[1]?(double)st[2]/st[1]:0.0, wstat());
        }
    }
    if (msN==0){ msAcc=present(0,1.0,T,T,false,true,777); msN=1; }  // timed, NON-learning
    double msStep = msAcc/msN;

    // ---------------- firing-steps-per-neuron, under TRAINING conditions ----------------
    // The biasUp/T law only follows if firing steps per presentation are roughly
    // CONSTANT rather than proportional to T. Measured here directly, from the spike
    // history of one additional training presentation (perturbs weights by 1/(NSTIM*EPOCHS)).
    {
        present(0, 1.0, T, trainClamp, true, false, 31337);
        const uint32_t* H=(const uint32_t*)bHist.contents;
        std::vector<uint16_t> cnt(N,0), cntC(N,0), cntF(N,0);
        for (uint32_t t=0;t<T;++t)
            for (uint32_t c=IN_COLS;c<NCOL;++c){
                uint32_t b=H[(size_t)t*NCOL+c];
                while(b){ uint32_t l=__builtin_ctz(b); b&=b-1;
                    cnt[c*NPC+l]++; if(t<trainClamp) cntC[c*NPC+l]++; else cntF[c*NPC+l]++; }
            }
        // final-state assembly members: the neurons that actually get ALLOCATED and
        // therefore the ones whose bias determines the veto. Transient one-shot firers
        // dominate the "fired at least once" population and must not be averaged in.
        const uint32_t* Hlast = H + (size_t)(T-1)*NCOL;
        std::vector<uint8_t> isMember(N,0); uint32_t nMem=0;
        for (uint32_t c=IN_COLS;c<NCOL;++c){ uint32_t b=Hlast[c];
            while(b){ uint32_t l=__builtin_ctz(b); b&=b-1; isMember[c*NPC+l]=1; nMem++; } }
        double n=0, sum=0, sumC=0, sumF=0; std::vector<uint16_t> v;
        double nm=0, sm=0, smC=0, smF=0; std::vector<uint16_t> vm;
        for (uint32_t i=IN_COLS*NPC;i<N;++i){
            if(cnt[i]){ n++; sum+=cnt[i]; sumC+=cntC[i]; sumF+=cntF[i]; v.push_back(cnt[i]); }
            if(isMember[i]){ nm++; sm+=cnt[i]; smC+=cntC[i]; smF+=cntF[i]; vm.push_back(cnt[i]); }
        }
        std::sort(v.begin(),v.end()); std::sort(vm.begin(),vm.end());
        printf("\n================= FIRING STEPS PER NEURON (training) =================\n");
        printf("  T=%u  trainClamp=%u (clamped) + %u (free)\n", T, trainClamp, T-trainClamp);
        printf("  neurons firing at least once: %.0f\n", n);
        printf("  mean firing steps  : %.2f   (clamped %.2f + free %.2f)\n",
               n?sum/n:0.0, n?sumC/n:0.0, n?sumF/n:0.0);
        if(!v.empty())
            printf("  p50 %u | p90 %u | max %u   | as fraction of T: %.3f\n",
                   v[v.size()/2], v[(size_t)(0.9*(v.size()-1))], v.back(), n?sum/n/T:0.0);
        printf("  --- restricted to FINAL-STATE ASSEMBLY MEMBERS (the allocated ones) ---\n");
        printf("  members: %.0f | mean firing steps %.2f (clamped %.2f + free %.2f) | fraction of T %.3f\n",
               nm, nm?sm/nm:0.0, nm?smC/nm:0.0, nm?smF/nm:0.0, nm?sm/nm/T:0.0);
        if(!vm.empty()) printf("  p50 %u | p90 %u | max %u\n",
               vm[vm.size()/2], vm[(size_t)(0.9*(vm.size()-1))], vm.back());
        printf("  => bias gain per presentation = biasUp * spikes(members) = %.0f\n",
               nm?biasUp*sm/nm:0.0);
        printf("======================================================================\n");
    }

    // ---- STEP A2: two reference definitions, measured separately ----
    //  CLAMPED  = input on for all T steps, settled, plasticity off. Stimulus-specific
    //             by construction.  <-- the correct reference
    //  FREE     = cue for CLAMP steps then released. This is what was used before, and
    //             if free-run endpoints all collapse it makes self and best-other the
    //             same measurement, which is how best-other could exceed self.
    std::vector<std::vector<uint32_t>> refClampCols(NSTIM), refFreeCols(NSTIM);
    std::vector<std::vector<uint32_t>> refClampBal(NSTIM), refFreeBal(NSTIM);
    std::vector<std::vector<uint32_t>> asm_(NSTIM);
    std::vector<std::vector<uint32_t>> asmSpkHost(NSTIM, std::vector<uint32_t>(NCOL,0));
    for (uint32_t p=0;p<NSTIM;++p){
        present(p,1.0,T,T,false,true,2000+p);                 // CLAMPED throughout
        refClampCols[p]=assemblyOf(T);
        refClampBal[p].assign((const uint32_t*)((T&1)?spkB.contents:spkA.contents),
                              (const uint32_t*)((T&1)?spkB.contents:spkA.contents)+NCOL);
        asm_[p]=refClampCols[p];
        memcpy(asmSpkHost[p].data(), refClampBal[p].data(), NCOL*4);
        for (uint32_t c=0;c<IN_COLS;++c) asmSpkHost[p][c]=0u;

        present(p,1.0,T,CLAMP,false,true,2100+p);              // cue then RELEASED
        refFreeCols[p]=assemblyOf(T);
        refFreeBal[p].assign((const uint32_t*)((T&1)?spkB.contents:spkA.contents),
                             (const uint32_t*)((T&1)?spkB.contents:spkA.contents)+NCOL);
    }

    // cross-talk between trained assemblies
    double sz=0; for(auto&a:refClampCols) sz+=a.size(); sz/=NSTIM;
    double xtalk=0, xtalkF=0; int xn=0;
    for(uint32_t i=0;i<NSTIM;++i) for(uint32_t j=i+1;j<NSTIM;++j){
        xtalk+=overlap(refClampCols[i],refClampCols[j]);
        xtalkF+=overlap(refFreeCols[i],refFreeCols[j]); xn++; }
    xtalk = xn?xtalk/xn:0; xtalkF = xn?xtalkF/xn:0;

    if (diag) {
        id<MTLBuffer> bMemb=mk((size_t)N*4, MTLResourceStorageModeShared);
        id<MTLBuffer> bAsm =mk((size_t)NSTIM*NCOL*4, MTLResourceStorageModeShared);
        id<MTLBuffer> bHist=mk(48*4, MTLResourceStorageModeShared);
        uint32_t* mp=(uint32_t*)bMemb.contents; memset(mp,0,(size_t)N*4);
        uint32_t* ap=(uint32_t*)bAsm.contents;
        for (uint32_t q=0;q<NSTIM;++q) memcpy(ap+(size_t)q*NCOL, asmSpkHost[q].data(), NCOL*4);
        for (uint32_t c=IN_COLS;c<NCOL;++c)
            for (uint32_t l=0;l<NPC;++l){
                uint32_t m=0;
                for (uint32_t q=0;q<std::min(NSTIM,32u);++q) if ((asmSpkHost[q][c]>>l)&1u) m |= (1u<<q);
                mp[c*NPC+l]=m;
            }
        uint32_t nMember=0; for(uint32_t i=0;i<N;++i) if(mp[i]) nMember++;

        const char* selName[2]={"RECURRENT branches","AFFERENT branches"};
        printf("\n======================= STEP 0 DIAGNOSTIC =======================\n");
        printf("4-bit synaptic counter distribution, %u stimuli, %u assembly-member neurons\n",
               NSTIM, nMember);
        for (uint32_t sel=0; sel<2; ++sel) {
            memset(bHist.contents,0,48*4);
            id<MTLCommandBuffer> cb=[Q commandBuffer]; id<MTLComputeCommandEncoder> ce=[cb computeCommandEncoder];
            [ce setComputePipelineState:pWDiag];
            [ce setBuffer:W0 offset:0 atIndex:0];[ce setBuffer:W1 offset:0 atIndex:1];
            [ce setBuffer:W2 offset:0 atIndex:2];[ce setBuffer:W3 offset:0 atIndex:3];
            [ce setBuffer:bConn offset:0 atIndex:4];[ce setBuffer:bMemb offset:0 atIndex:5];
            [ce setBuffer:bAsm offset:0 atIndex:6];
            uint32_t cfg[4]={NSTIM,inBranches,sel,NCOL/4};
            [ce setBytes:cfg length:16 atIndex:7];[ce setBuffer:bHist offset:0 atIndex:8];[ce setBuffer:bMarg offset:0 atIndex:9];[ce setBuffer:bRefBall offset:0 atIndex:10];[ce setBuffer:bDriveLog offset:0 atIndex:11];[ce setBuffer:bMult offset:0 atIndex:12];
            [ce dispatchThreads:MTLSizeMake(N,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [ce endEncoding];[cb commit];[cb waitUntilCompleted];
            const uint32_t* h=(const uint32_t*)bHist.contents;
            const char* cat[3]={"within-assembly","between-assembly","all"};
            printf("\n--- %s ---\n", selName[sel]);
            printf("  %-17s %10s  %6s %6s %6s | %s\n","pair type","count","mean","%at 0","%at 15","histogram v=0..15");
            for (uint32_t c=0;c<3;++c){
                double tot=0,sum=0; for(uint32_t v=0;v<16;++v){ tot+=h[c*16+v]; sum+=(double)v*h[c*16+v]; }
                printf("  %-17s %10.0f  %6.2f %6.1f %6.1f | ", cat[c], tot, tot?sum/tot:0.0,
                       tot?100.0*h[c*16+0]/tot:0.0, tot?100.0*h[c*16+15]/tot:0.0);
                for(uint32_t v=0;v<16;++v) printf("%s%.1f", v?",":"", tot?100.0*h[c*16+v]/tot:0.0);
                printf("\n");
            }
        }
        printf("================================================================\n");
    }

    if(!quiet){
        printf("\n  %.4f ms/timestep -> %.0f steps/s (%.1fx biological real time), wdens %.3f%%\n",
               msStep, 1000.0/msStep, 1.0/msStep, wstat());
        printf("  assembly size %.1f cols (%.2f%% of area) | cross-talk CLAMPED %.3f  FREE %.3f  (chance %.3f)\n\n",
               sz, 100.0*sz/(NCOL-IN_COLS), xtalk, xtalkF, sz/(NCOL-IN_COLS));
        printf("--- pattern completion: cue %u steps, RELEASED %u.  BOTH reference definitions ---\n", CLAMP, T-CLAMP);
        printf("  cue  |  CLAMPED ref: self  other  margin  |  FREE ref: self  other  margin\n");
    }
    // ---------------- per-kernel GPU profile (real hardware timestamps) ----------------
    if (profile) {
        id<MTLCounterSet> tsSet=nil;
        for (id<MTLCounterSet> cs in D.counterSets)
            if ([cs.name isEqualToString:MTLCommonCounterSetTimestamp]) tsSet=cs;
        if (!tsSet || ![D supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]) {
            printf("\n[profile] GPU timestamps unavailable on this device\n");
        } else {
            const uint32_t KPS=5;
            const char* kn[5]={"k_drive","k_topk","k_emit","k_bias","k_learn"};
            NSUInteger NS=(NSUInteger)T*KPS*2;
            MTLCounterSampleBufferDescriptor* csd=[MTLCounterSampleBufferDescriptor new];
            csd.counterSet=tsSet; csd.storageMode=MTLStorageModeShared; csd.sampleCount=NS;
            NSError* ce2=nil;
            id<MTLCounterSampleBuffer> sb=[D newCounterSampleBufferWithDescriptor:csd error:&ce2];
            if(!sb){ printf("\n[profile] sample buffer failed: %s\n", ce2.localizedDescription.UTF8String); }
            else {
                loadStim(0,1.0,5555);
                memset(bStat.contents,0,4096);
                P.learn=1; P.freeze=1; P.histOn=0;
                auto t0=std::chrono::steady_clock::now();
                id<MTLCommandBuffer> cb=[Q commandBuffer];
                NSUInteger si=0;
                for (uint32_t t=0;t<T;++t){
                    id<MTLBuffer> in=(t&1)?spkB:spkA, out=(t&1)?spkA:spkB;
                    P.step=t; P.clamp=(t<CLAMP)?1u:0u; P.depress=(t%2==1)?1u:0u;
                    for (uint32_t kk=0;kk<KPS;++kk){
                        MTLComputePassDescriptor* cpd=[MTLComputePassDescriptor computePassDescriptor];
                        cpd.sampleBufferAttachments[0].sampleBuffer=sb;
                        cpd.sampleBufferAttachments[0].startOfEncoderSampleIndex=si;
                        cpd.sampleBufferAttachments[0].endOfEncoderSampleIndex=si+1;
                        si+=2;
                        id<MTLComputeCommandEncoder> ce=[cb computeCommandEncoderWithDescriptor:cpd];
                        if (kk==0){
                            [ce setComputePipelineState:pDrive];
                            [ce setBuffer:W3 offset:0 atIndex:0];[ce setBuffer:bConn offset:0 atIndex:1];
                            [ce setBuffer:in offset:0 atIndex:2];[ce setBuffer:bDrv offset:0 atIndex:3];
                            [ce setBuffer:bBits offset:0 atIndex:4];[ce setBuffer:bAcc offset:0 atIndex:5];
                            [ce setBuffer:bBias offset:0 atIndex:7];[ce setBuffer:in offset:0 atIndex:8];
                            [ce setBytes:&P length:sizeof(P) atIndex:6];
                            [ce dispatchThreads:MTLSizeMake(N,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
                        } else if (kk==1){
                            [ce setComputePipelineState:pTopK];
                            [ce setBuffer:bDrv offset:0 atIndex:0];[ce setBuffer:bTheta offset:0 atIndex:1];
                            [ce setBytes:&P length:sizeof(P) atIndex:2];
                            uint32_t nc_=NCOL;[ce setBytes:&nc_ length:4 atIndex:3];
                            [ce dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(topkTG,1,1)];
                        } else if (kk==2){
                            [ce setComputePipelineState:pEmit];
                            [ce setBuffer:bAcc offset:0 atIndex:0];[ce setBuffer:bDrv offset:0 atIndex:1];
                            [ce setBuffer:bTheta offset:0 atIndex:2];[ce setBuffer:bStim offset:0 atIndex:3];
                            [ce setBuffer:bBits offset:0 atIndex:4];[ce setBuffer:out offset:0 atIndex:5];
                            [ce setBytes:&P length:sizeof(P) atIndex:6];[ce setBuffer:bStat offset:0 atIndex:7];
                            [ce setBuffer:bHist offset:0 atIndex:8];[ce setBuffer:bMarg offset:0 atIndex:9];[ce setBuffer:bRefBall offset:0 atIndex:10];[ce setBuffer:bDriveLog offset:0 atIndex:11];[ce setBuffer:bMult offset:0 atIndex:12];
                            [ce dispatchThreads:MTLSizeMake(N,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
                        } else if (kk==3){
                            [ce setComputePipelineState:pBias];
                            [ce setBuffer:out offset:0 atIndex:0];[ce setBuffer:bBias offset:0 atIndex:1];
                            [ce setBytes:&P length:sizeof(P) atIndex:2];
                            [ce dispatchThreads:MTLSizeMake(N,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
                        } else {
                            [ce setComputePipelineState:pLearn];
                            [ce setBuffer:W0 offset:0 atIndex:0];[ce setBuffer:W1 offset:0 atIndex:1];
                            [ce setBuffer:W2 offset:0 atIndex:2];[ce setBuffer:W3 offset:0 atIndex:3];
                            [ce setBuffer:bConn offset:0 atIndex:4];[ce setBuffer:in offset:0 atIndex:5];
                            [ce setBuffer:out offset:0 atIndex:6];[ce setBuffer:bBits offset:0 atIndex:7];
                            [ce setBytes:&P length:sizeof(P) atIndex:8];[ce setBuffer:bMarg offset:0 atIndex:9];[ce setBuffer:bStat offset:0 atIndex:10];
                            [ce dispatchThreads:MTLSizeMake(N,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
                        }
                        [ce endEncoding];
                    }
                }
                [cb commit];[cb waitUntilCompleted];
                double wall=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();
                NSData* dat=[sb resolveCounterRange:NSMakeRange(0,NS)];
                const MTLCounterResultTimestamp* ts=(const MTLCounterResultTimestamp*)dat.bytes;
                double acc[5]={0,0,0,0,0}; double tot=0;
                for (uint32_t t=0;t<T;++t) for (uint32_t kk=0;kk<KPS;++kk){
                    NSUInteger i=(t*KPS+kk)*2;
                    double us=(double)(ts[i+1].timestamp-ts[i].timestamp)/1000.0;
                    acc[kk]+=us; tot+=us;
                }
                printf("\n================ PER-KERNEL GPU PROFILE (hardware timestamps) ================\n");
                printf("  %u timesteps, one encoder per kernel, cue %u steps, learning ON\n", T, CLAMP);
                printf("  %-10s %12s %12s   %s\n","kernel","us/step","%% of GPU","");
                for (uint32_t kk=0;kk<KPS;++kk)
                    printf("  %-10s %12.1f %12.1f\n", kn[kk], acc[kk]/T, 100.0*acc[kk]/tot);
                printf("  %-10s %12.1f %12.1f\n","TOTAL",tot/T,100.0);
                printf("  GPU busy %.3f ms/step | wall %.3f ms/step | cmdbuf GPU %.3f ms\n",
                       tot/T/1000.0, wall/T, (cb.GPUEndTime-cb.GPUStartTime)*1e3);
                printf("=============================================================================\n");
            }
        }
    }

    // ---- marginal cost of each kernel, measured INSIDE the production encoder ----
    // (no per-encoder overhead: dispatch kernel k an extra R times and take the slope)
    if (profile) {
        const char* kn[5]={"k_drive","k_topk","k_emit","k_bias","k_learn"};
        auto encodeK=[&](id<MTLComputeCommandEncoder> ce, uint32_t kk,
                         id<MTLBuffer> in, id<MTLBuffer> out){
            if (kk==0){
                [ce setComputePipelineState:pDrive];
                [ce setBuffer:W3 offset:0 atIndex:0];[ce setBuffer:bConn offset:0 atIndex:1];
                [ce setBuffer:in offset:0 atIndex:2];[ce setBuffer:bDrv offset:0 atIndex:3];
                [ce setBuffer:bBits offset:0 atIndex:4];[ce setBuffer:bAcc offset:0 atIndex:5];
                [ce setBuffer:bBias offset:0 atIndex:7];[ce setBuffer:in offset:0 atIndex:8];[ce setBytes:&P length:sizeof(P) atIndex:6];
                [ce dispatchThreads:MTLSizeMake(N,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            } else if (kk==1){
                [ce setComputePipelineState:pTopK];
                [ce setBuffer:bDrv offset:0 atIndex:0];[ce setBuffer:bTheta offset:0 atIndex:1];
                [ce setBytes:&P length:sizeof(P) atIndex:2];
                uint32_t nc_=NCOL;[ce setBytes:&nc_ length:4 atIndex:3];
                [ce dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(topkTG,1,1)];
            } else if (kk==2){
                [ce setComputePipelineState:pEmit];
                [ce setBuffer:bAcc offset:0 atIndex:0];[ce setBuffer:bDrv offset:0 atIndex:1];
                [ce setBuffer:bTheta offset:0 atIndex:2];[ce setBuffer:bStim offset:0 atIndex:3];
                [ce setBuffer:bBits offset:0 atIndex:4];[ce setBuffer:out offset:0 atIndex:5];
                [ce setBytes:&P length:sizeof(P) atIndex:6];[ce setBuffer:bStat offset:0 atIndex:7];
                [ce setBuffer:bHist offset:0 atIndex:8];[ce setBuffer:bMarg offset:0 atIndex:9];[ce setBuffer:bRefBall offset:0 atIndex:10];[ce setBuffer:bDriveLog offset:0 atIndex:11];[ce setBuffer:bMult offset:0 atIndex:12];
                [ce dispatchThreads:MTLSizeMake(N,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            } else if (kk==3){
                [ce setComputePipelineState:pBias];
                [ce setBuffer:out offset:0 atIndex:0];[ce setBuffer:bBias offset:0 atIndex:1];
                [ce setBytes:&P length:sizeof(P) atIndex:2];
                [ce dispatchThreads:MTLSizeMake(N,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            } else {
                [ce setComputePipelineState:pLearn];
                [ce setBuffer:W0 offset:0 atIndex:0];[ce setBuffer:W1 offset:0 atIndex:1];
                [ce setBuffer:W2 offset:0 atIndex:2];[ce setBuffer:W3 offset:0 atIndex:3];
                [ce setBuffer:bConn offset:0 atIndex:4];[ce setBuffer:in offset:0 atIndex:5];
                [ce setBuffer:out offset:0 atIndex:6];[ce setBuffer:bBits offset:0 atIndex:7];
                [ce setBytes:&P length:sizeof(P) atIndex:8];[ce setBuffer:bMarg offset:0 atIndex:9];[ce setBuffer:bStat offset:0 atIndex:10];
                [ce dispatchThreads:MTLSizeMake(N,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            }
        };
        // cumulative ablation in the production encoder: the only honest way to
        // cost a kernel here, since repeated back-to-back dispatches run hot in
        // cache and make a marginal/slope measurement meaningless.
        auto runMask=[&](uint32_t mask)->double{
            loadStim(0,1.0,5555); P.learn=1; P.freeze=1; P.histOn=0;
            double best=1e30;
            for (int r=0;r<5;++r){
                auto t0=std::chrono::steady_clock::now();
                id<MTLCommandBuffer> cb=[Q commandBuffer];
                id<MTLComputeCommandEncoder> ce=[cb computeCommandEncoder];
                for (uint32_t t=0;t<T;++t){
                    id<MTLBuffer> in=(t&1)?spkB:spkA, out=(t&1)?spkA:spkB;
                    P.step=t; P.clamp=(t<CLAMP)?1u:0u; P.depress=(t%2==1)?1u:0u;
                    for (uint32_t kk=0;kk<5;++kk) if (mask & (1u<<kk)) encodeK(ce,kk,in,out);
                }
                [ce endEncoding];[cb commit];[cb waitUntilCompleted];
                double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count()/T;
                best=std::min(best,ms);
            }
            return best;
        };
        struct Row { const char* label; uint32_t mask; };
        Row rows[] = {
            {"k_drive only",                 0b00001},
            {"+ k_topk",                     0b00011},
            {"+ k_emit",                     0b00111},
            {"+ k_bias",                     0b01111},
            {"+ k_learn  (full step)",       0b11111},
        };
        printf("\n============ CUMULATIVE ABLATION, production encoder (us/step) ============\n");
        double prev=0;
        for (auto& r : rows) {
            double ms=runMask(r.mask);
            printf("  %-26s %9.1f us   (delta %+8.1f us)\n", r.label, ms*1000.0,
                   prev==0?ms*1000.0:(ms-prev)*1000.0);
            prev=ms;
        }
        double full=prev, noLearn=runMask(0b01111);
        printf("  ------------------------------------------------------------------\n");
        printf("  plasticity (k_learn) = %.1f us of %.1f us  ->  %.1f%% of the step\n",
               (full-noLearn)*1000.0, full*1000.0, 100.0*(full-noLearn)/full);
        printf("==========================================================================\n");
    }

    // ================================ STEP A1 ================================
    {
        id<MTLBuffer> bInd=mk((size_t)N*4, MTLResourceStorageModeShared);
        { id<MTLCommandBuffer> cb=[Q commandBuffer]; id<MTLComputeCommandEncoder> ce=[cb computeCommandEncoder];
          [ce setComputePipelineState:pIndeg];
          [ce setBuffer:W3 offset:0 atIndex:0];[ce setBuffer:bInd offset:0 atIndex:1];
          [ce setBytes:&inBranches length:4 atIndex:2];
          [ce dispatchThreads:MTLSizeMake(N,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
          [ce endEncoding];[cb commit];[cb waitUntilCompleted]; }
        const uint32_t* ind=(const uint32_t*)bInd.contents;
        std::vector<uint32_t> v; v.reserve(N);
        for (uint32_t c=IN_COLS;c<NCOL;++c) for(uint32_t l=0;l<NPC;++l) v.push_back(ind[c*NPC+l]);
        std::vector<uint32_t> sv=v; std::sort(sv.begin(),sv.end());
        auto pct=[&](double q){ return sv[(size_t)(q*(sv.size()-1))]; };
        double mean=0; for(uint32_t x:v) mean+=x; mean/=v.size();
        double sd=0; for(uint32_t x:v) sd+=(x-mean)*(x-mean); sd=std::sqrt(sd/v.size());
        printf("\n=============================== STEP A1 ===============================\n");
        printf("Functional in-degree over the %u RECURRENT branches (popcount of MSB plane)\n", NSLOT-inBranches);
        printf("  n=%zu neurons   mean %.1f   sd %.1f   cv %.3f\n", v.size(), mean, sd, sd/mean);
        printf("  min %u | p10 %u | p25 %u | p50 %u | p75 %u | p90 %u | p99 %u | p99.9 %u | max %u\n",
               sv.front(),pct(0.10),pct(0.25),pct(0.50),pct(0.75),pct(0.90),pct(0.99),pct(0.999),sv.back());
        printf("  tail ratio p99/p50 = %.2f   max/p50 = %.2f\n",
               (double)pct(0.99)/pct(0.50), (double)sv.back()/pct(0.50));

        // winner identity across stimuli
        auto neuronSet=[&](const std::vector<uint32_t>& bal){
            std::vector<uint32_t> r;
            for (uint32_t c=IN_COLS;c<NCOL;++c){ uint32_t b=bal[c];
                while(b){ uint32_t l=__builtin_ctz(b); b&=b-1; r.push_back(c*NPC+l); } }
            return r; };
        auto jacc=[&](const std::vector<uint32_t>&a,const std::vector<uint32_t>&b){
            if(a.empty()||b.empty()) return 0.0;
            std::vector<uint32_t> i,u;
            std::set_intersection(a.begin(),a.end(),b.begin(),b.end(),std::back_inserter(i));
            std::set_union(a.begin(),a.end(),b.begin(),b.end(),std::back_inserter(u));
            return (double)i.size()/u.size(); };

        for (int which=0; which<2; ++which) {
            const auto& src = which ? refClampBal : refFreeBal;
            std::vector<std::vector<uint32_t>> ns(NSTIM);
            for(uint32_t q=0;q<NSTIM;++q) ns[q]=neuronSet(src[q]);
            double jm=0; int jn=0; double jmax=0;
            for(uint32_t i=0;i<NSTIM;++i) for(uint32_t j=i+1;j<NSTIM;++j){
                double x=jacc(ns[i],ns[j]); jm+=x; jn++; jmax=std::max(jmax,x); }
            std::vector<uint32_t> cnt(N,0);
            for(auto&a:ns) for(uint32_t x:a) cnt[x]++;
            const uint32_t* bias=(const uint32_t*)bBias.contents;
            uint32_t nAll=0,nNone=0; double dAll=0,dNone=0,dAny=0; uint32_t nAny=0;
            double bAll=0,bNone=0,bAny=0;
            for (uint32_t c=IN_COLS;c<NCOL;++c) for(uint32_t l=0;l<NPC;++l){
                uint32_t id=c*NPC+l;
                if(cnt[id]>=NSTIM-1){nAll++; dAll+=ind[id]; bAll+=bias[id];}
                if(cnt[id]==0){nNone++; dNone+=ind[id]; bNone+=bias[id];}
                if(cnt[id]>0){nAny++; dAny+=ind[id]; bAny+=bias[id];} }
            printf("\n  --- winner identity, %s endpoint ---\n", which?"CLAMPED":"FREE-RUN");
            printf("    mean pairwise Jaccard %.3f (max %.3f) over %d pairs, mean set size %.0f\n",
                   jm/jn, jmax, jn, (double)ns[0].size());
            printf("    neurons winning in >=%u/%u stimuli: %6u  in-degree %6.1f  intrinsic-bias %8.1f\n",
                   NSTIM-1, NSTIM, nAll, nAll?dAll/nAll:0.0, nAll?bAll/nAll:0.0);
            printf("    neurons winning in any stimulus  : %6u  in-degree %6.1f  intrinsic-bias %8.1f\n",
                   nAny, nAny?dAny/nAny:0.0, nAny?bAny/nAny:0.0);
            printf("    neurons never winning            : %6u  in-degree %6.1f  intrinsic-bias %8.1f\n",
                   nNone, nNone?dNone/nNone:0.0, nNone?bNone/nNone:0.0);
            printf("    HUB RATIO (winners-in-all / never) = %.3f\n",
                   (nNone&&dNone)?(dAll/std::max(1u,nAll))/(dNone/nNone):0.0);
        }
        printf("======================================================================\n");
    }

    // ================================ STEP A3 + A4 ================================
    {
        auto neuronSet=[&](const std::vector<uint32_t>& bal){
            std::vector<uint32_t> r;
            for (uint32_t c=IN_COLS;c<NCOL;++c){ uint32_t b=bal[c];
                while(b){ uint32_t l=__builtin_ctz(b); b&=b-1; r.push_back(c*NPC+l); } }
            return r; };
        auto jacc=[&](const std::vector<uint32_t>&a,const std::vector<uint32_t>&b){
            if(a.empty()||b.empty()) return 0.0;
            std::vector<uint32_t> i,u;
            std::set_intersection(a.begin(),a.end(),b.begin(),b.end(),std::back_inserter(i));
            std::set_union(a.begin(),a.end(),b.begin(),b.end(),std::back_inserter(u));
            return (double)i.size()/u.size(); };

        // ---------------- A3: chance baseline ----------------
        const double npool = (double)(NCOL-IN_COLS)*NPC;
        std::vector<std::vector<uint32_t>> cns(NSTIM);
        for(uint32_t q=0;q<NSTIM;++q) cns[q]=neuronSet(refClampBal[q]);
        double msz=0; for(auto&a:cns) msz+=a.size(); msz/=NSTIM;
        double eInt = msz*msz/npool;
        double eJac = eInt/(2.0*msz - eInt);
        double jm=0,jmax=0; int jn=0; double totInt=0;
        for(uint32_t i=0;i<NSTIM;++i) for(uint32_t j=i+1;j<NSTIM;++j){
            double x=jacc(cns[i],cns[j]); jm+=x; jn++; jmax=std::max(jmax,x);
            std::vector<uint32_t> ii;
            std::set_intersection(cns[i].begin(),cns[i].end(),cns[j].begin(),cns[j].end(),
                                  std::back_inserter(ii));
            totInt += ii.size(); }
        std::vector<uint32_t> uni;
        for(auto&a:cns) uni.insert(uni.end(),a.begin(),a.end());
        std::sort(uni.begin(),uni.end());
        size_t uniqN = std::unique(uni.begin(),uni.end()) - uni.begin();
        double sumSz = msz*NSTIM;
        printf("\n=============================== STEP A3 ===============================\n");
        printf("Chance baseline for %u assemblies of %.0f neurons from a pool of %.0f\n",
               NSTIM, msz, npool);
        printf("  E[|A n B|] = m^2/n            = %.2f neurons\n", eInt);
        printf("  E[Jaccard] = E/(2m-E)         = %.5f      <-- CHANCE\n", eJac);
        printf("  measured mean pairwise Jaccard = %.5f   (max %.5f)\n", jm/jn, jmax);
        printf("  ratio measured/chance          = %.3f\n", eJac>0?(jm/jn)/eJac:0.0);
        printf("  total pairwise overlap: measured %.0f neurons vs %.0f expected at chance (%.1fx less)\n",
               totInt, eInt*jn, totInt>0?eInt*jn/totInt:1e9);
        printf("  sum of assembly sizes %.0f | union %zu | excess %.0f  -> %s\n",
               sumSz, uniqN, sumSz-(double)uniqN,
               (sumSz-(double)uniqN) < 0.02*sumSz ? "NEAR-EXACT PARTITION" : "overlapping");

        // ---------------- A4: zero the bias at readout only ----------------
        std::vector<std::vector<uint32_t>> refNBCols(NSTIM), refNBBal(NSTIM);
        for (uint32_t q=0;q<NSTIM;++q){
            present(q,1.0,T,T,false,true,2200+q,0u);          // clamped, bias OFF
            refNBCols[q]=assemblyOf(T);
            refNBBal[q].assign((const uint32_t*)((T&1)?spkB.contents:spkA.contents),
                               (const uint32_t*)((T&1)?spkB.contents:spkA.contents)+NCOL);
        }
        double preserve=0; for(uint32_t q=0;q<NSTIM;++q) preserve+=overlap(refNBCols[q],refClampCols[q]);
        preserve/=NSTIM;
        double xNB=0; int xn2=0;
        for(uint32_t i=0;i<NSTIM;++i) for(uint32_t j=i+1;j<NSTIM;++j){ xNB+=overlap(refNBCols[i],refNBCols[j]); xn2++; }
        xNB/=xn2;
        printf("\n=============================== STEP A4 ===============================\n");
        printf("Zero the intrinsic bias at TEST time only. Trained weights unchanged.\n");
        printf("  clamped assembly preserved when bias is zeroed: overlap %.3f\n", preserve);
        printf("  cross-talk of clamped assemblies, bias OFF     : %.3f  (bias ON: %.3f)\n", xNB, xtalk);

        auto colsOf=[&](const uint32_t* sp){ std::vector<uint32_t> v;
            for(uint32_t c=IN_COLS;c<NCOL;++c) if(sp[c]) v.push_back(c); return v; };
        printf("\n  release trajectory, cue 75%%, bias OFF at test (vs bias ON baseline)\n");
        printf("  step  phase   ov(clampRef,biasON)  ov(clampRef,biasOFF)  ov(noBiasRef,biasOFF)\n");
        std::vector<double> a1(T,0),a2(T,0),a3(T,0);
        for (uint32_t q=0;q<NSTIM;++q){
            present(q,0.75,T,CLAMP,false,true,4300+q,1u);
            { const uint32_t* H=(const uint32_t*)bHist.contents;
              for(uint32_t t=0;t<T;++t) a1[t]+=overlap(colsOf(H+(size_t)t*NCOL),refClampCols[q]); }
            present(q,0.75,T,CLAMP,false,true,4300+q,0u);
            { const uint32_t* H=(const uint32_t*)bHist.contents;
              for(uint32_t t=0;t<T;++t){ auto a=colsOf(H+(size_t)t*NCOL);
                a2[t]+=overlap(a,refClampCols[q]); a3[t]+=overlap(a,refNBCols[q]); } }
        }
        for (uint32_t t=0;t<T;++t)
            printf("  %4u  %-5s %19.3f %21.3f %22.3f\n", t, t<CLAMP?"cue":"FREE",
                   a1[t]/NSTIM, a2[t]/NSTIM, a3[t]/NSTIM);

        printf("\n  pattern completion with bias OFF at test  (keep=0 is the no-cue control)\n");
        printf("  cue  |  vs CLAMPED ref(biasON): self  other  margin | vs noBias ref: self  other  margin\n");
        for (double keep : {1.0,0.75,0.5,0.35,0.25,0.15,0.10,0.05,0.02,0.0}) {
            double s1=0,o1=0,s2=0,o2=0;
            for (uint32_t q=0;q<NSTIM;++q){
                present(q,keep,T,CLAMP,false,true,4400+q,0u);
                auto a=assemblyOf(T);
                s1+=overlap(a,refClampCols[q]); s2+=overlap(a,refNBCols[q]);
                double b1=0,b2=0;
                for(uint32_t r=0;r<NSTIM;++r) if(r!=q){
                    b1=std::max(b1,overlap(a,refClampCols[r]));
                    b2=std::max(b2,overlap(a,refNBCols[r])); }
                o1+=b1; o2+=b2; }
            printf("  %3.0f%% |%25.3f %6.3f %+7.3f |%18.3f %6.3f %+7.3f\n",
                   keep*100, s1/NSTIM, o1/NSTIM, s1/NSTIM-o1/NSTIM,
                   s2/NSTIM, o2/NSTIM, s2/NSTIM-o2/NSTIM);
        }
        printf("======================================================================\n");
    }

    // ================================ A4b + A4c ================================
    {
        auto nset=[&](const uint32_t* bal){ std::vector<uint32_t> r;
            for (uint32_t c=IN_COLS;c<NCOL;++c){ uint32_t b=bal[c];
                while(b){ uint32_t l=__builtin_ctz(b); b&=b-1; r.push_back(c*NPC+l);} } return r; };
        auto jac=[&](const std::vector<uint32_t>&a,const std::vector<uint32_t>&b){
            if(a.empty()||b.empty()) return 0.0; std::vector<uint32_t> i,u;
            std::set_intersection(a.begin(),a.end(),b.begin(),b.end(),std::back_inserter(i));
            std::set_union(a.begin(),a.end(),b.begin(),b.end(),std::back_inserter(u));
            return (double)i.size()/u.size(); };

        id<MTLBuffer> bInd=mk((size_t)N*4, MTLResourceStorageModeShared);
        id<MTLBuffer> bUnt=mk((size_t)N*4, MTLResourceStorageModeShared);
        { id<MTLCommandBuffer> cb=[Q commandBuffer]; id<MTLComputeCommandEncoder> ce=[cb computeCommandEncoder];
          [ce setComputePipelineState:pIndeg];
          [ce setBuffer:W3 offset:0 atIndex:0];[ce setBuffer:bInd offset:0 atIndex:1];
          [ce setBytes:&inBranches length:4 atIndex:2];
          [ce dispatchThreads:MTLSizeMake(N,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
          [ce setComputePipelineState:pUntouched];
          [ce setBuffer:W0 offset:0 atIndex:0];[ce setBuffer:W1 offset:0 atIndex:1];
          [ce setBuffer:W2 offset:0 atIndex:2];[ce setBuffer:W3 offset:0 atIndex:3];
          [ce setBuffer:bUnt offset:0 atIndex:4];[ce setBytes:&inBranches length:4 atIndex:5];
          [ce dispatchThreads:MTLSizeMake(N,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
          [ce endEncoding];[cb commit];[cb waitUntilCompleted]; }
        const uint32_t* ind=(const uint32_t*)bInd.contents;
        const uint32_t* unt=(const uint32_t*)bUnt.contents;
        const double RECSYN = (double)(NSLOT-inBranches)*PRESYN;

        const double cues[4]={1.0,0.75,0.35,0.15};
        printf("\n=============================== A4b ===============================\n");
        printf("Is the recalled ~25%% a FIXED subset? (readout gate on: bias off at test)\n");
        std::vector<std::vector<std::vector<uint32_t>>> rec(NSTIM,
            std::vector<std::vector<uint32_t>>(4));
        for (uint32_t q=0;q<NSTIM;++q)
            for (int ci=0;ci<4;++ci){
                present(q,cues[ci],T,CLAMP,false,true,5000+q*16+ci);
                rec[q][ci]=nset((const uint32_t*)((T&1)?spkB.contents:spkA.contents));
            }
        printf("  pairwise Jaccard of recalled sets ACROSS cue levels, same stimulus:\n    ");
        double jall=0; int jn=0;
        for (int a=0;a<4;++a) for(int b=a+1;b<4;++b){
            double m=0; for(uint32_t q=0;q<NSTIM;++q) m+=jac(rec[q][a],rec[q][b]); m/=NSTIM;
            printf("%.0f%%/%.0f%%=%.3f  ", cues[a]*100, cues[b]*100, m); jall+=m; jn++; }
        printf("\n    mean across cue pairs = %.3f\n", jall/jn);

        // core = recalled at every cue level; periphery = in assembly but never recalled
        double coreSz=0, periSz=0, asmSz=0;
        double dCore=0,dPeri=0,uCore=0,uPeri=0; size_t nCore=0,nPeri=0;
        for (uint32_t q=0;q<NSTIM;++q){
            std::vector<uint32_t> core=rec[q][0];
            for(int ci=1;ci<4;++ci){ std::vector<uint32_t> t;
                std::set_intersection(core.begin(),core.end(),rec[q][ci].begin(),rec[q][ci].end(),
                                      std::back_inserter(t)); core=t; }
            std::vector<uint32_t> A = nset(refClampBal[q].data());
            std::vector<uint32_t> coreInA, peri;
            std::set_intersection(core.begin(),core.end(),A.begin(),A.end(),std::back_inserter(coreInA));
            std::set_difference(A.begin(),A.end(),core.begin(),core.end(),std::back_inserter(peri));
            coreSz+=coreInA.size(); periSz+=peri.size(); asmSz+=A.size();
            for(uint32_t x:coreInA){ dCore+=ind[x]; uCore+=100.0*unt[x]/RECSYN; nCore++; }
            for(uint32_t x:peri)   { dPeri+=ind[x]; uPeri+=100.0*unt[x]/RECSYN; nPeri++; }
        }
        printf("\n  assembly %.0f neurons | CORE (recalled at all 4 cues) %.0f (%.1f%%) | periphery %.0f\n",
               asmSz/NSTIM, coreSz/NSTIM, 100.0*coreSz/asmSz, periSz/NSTIM);
        printf("  %-12s %10s %14s %22s\n","group","n","in-degree","%% synapses still at 5/6");
        printf("  %-12s %10zu %14.1f %22.1f\n","CORE",nCore,nCore?dCore/nCore:0.0,nCore?uCore/nCore:0.0);
        printf("  %-12s %10zu %14.1f %22.1f\n","periphery",nPeri,nPeri?dPeri/nPeri:0.0,nPeri?uPeri/nPeri:0.0);
        double allD=0,allU=0; size_t nAll=0;
        for (uint32_t c=IN_COLS;c<NCOL;++c) for(uint32_t l=0;l<NPC;++l){
            allD+=ind[c*NPC+l]; allU+=100.0*unt[c*NPC+l]/RECSYN; nAll++; }
        printf("  %-12s %10zu %14.1f %22.1f\n","all neurons",nAll,allD/nAll,allU/nAll);

        // ---------------- A4c ----------------
        const uint32_t* bs=(const uint32_t*)bBias.contents;
        std::vector<uint32_t> bv; bv.reserve(N);
        for (uint32_t c=IN_COLS;c<NCOL;++c) for(uint32_t l=0;l<NPC;++l) bv.push_back(bs[c*NPC+l]);
        std::sort(bv.begin(),bv.end());
        double bm=0; for(uint32_t x:bv) bm+=x; bm/=bv.size();
        printf("\n=============================== A4c ===============================\n");
        printf("Intrinsic bias growth after %u epochs (unbounded, non-leaky, learning-time only)\n", EPOCHS);
        printf("  mean %.0f | p50 %u | p99 %u | max %u | UINT32_MAX %u | headroom %.3g x\n",
               bm, bv[bv.size()/2], bv[(size_t)(0.99*(bv.size()-1))], bv.back(),
               4294967295u, bv.back()? 4294967295.0/bv.back() : 0.0);
        printf("  growth rate ~= %.1f per epoch (max/epochs) -> epochs to saturate ~ %.3g\n",
               EPOCHS? (double)bv.back()/EPOCHS : 0.0,
               (bv.back()&&EPOCHS)? 4294967295.0/((double)bv.back()/EPOCHS) : 0.0);
        printf("==================================================================\n");
    }

    // ===== CHARACTERISE THE VISITED UNION: what is the trace exploring? =====
    if (unionchar) {
        id<MTLBuffer> bPost=mk((size_t)N*4, MTLResourceStorageModeShared);
        uint32_t* mp=(uint32_t*)bMult.contents; memset(mp,0,(size_t)N*4);
        for (uint32_t q=0;q<NSTIM;++q)
            for (uint32_t c=IN_COLS;c<NCOL;++c){ uint32_t b=refClampBal[q][c];
                while(b){ uint32_t l=__builtin_ctz(b); b&=b-1; mp[c*NPC+l]++; } }
        uint32_t NQ=std::min(NSTIM,12u);
        double nIn=0,nOut=0, fI=0,pI=0, fO=0,pO=0, fA=0,pA=0, mOut=0, mAll=0, nAll=0;
        auto runW=[&](const std::vector<uint32_t>& post, double& f, double& pp){
            uint32_t* pm=(uint32_t*)bPost.contents; memset(pm,0,(size_t)N*4);
            for(uint32_t x:post) pm[x]=1u;
            memset(bWCnt.contents,0,128);
            id<MTLCommandBuffer> cb=[Q commandBuffer]; id<MTLComputeCommandEncoder> ce=[cb computeCommandEncoder];
            [ce setComputePipelineState:pWithin];
            [ce setBuffer:W3 offset:0 atIndex:0];[ce setBuffer:bConn offset:0 atIndex:1];
            [ce setBuffer:bRefBall offset:0 atIndex:2];[ce setBuffer:bRefBall offset:0 atIndex:3];
            uint32_t cfg2[2]={inBranches, IN_COLS}; [ce setBytes:cfg2 length:8 atIndex:4];
            [ce setBuffer:bWCnt offset:0 atIndex:5];[ce setBuffer:bMult offset:0 atIndex:6];
            [ce setBuffer:bPost offset:0 atIndex:7]; uint32_t um=1u; [ce setBytes:&um length:4 atIndex:8];
            [ce dispatchThreads:MTLSizeMake(N,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [ce endEncoding];[cb commit];[cb waitUntilCompleted];
            const uint32_t* w=(const uint32_t*)bWCnt.contents; f+=w[0]; pp+=w[1];
        };
        std::vector<uint32_t> allRec;
        for (uint32_t c=IN_COLS;c<NCOL;++c) for(uint32_t l=0;l<NPC;++l) allRec.push_back(c*NPC+l);
        for (uint32_t q=0;q<NQ;++q){
            memcpy(bRefBall.contents, refClampBal[q].data(), NCOL*4);
            present(q,1.0,35u,CLAMP,false,true,9500+q);
            const uint32_t* H=(const uint32_t*)bHist.contents;
            std::vector<uint32_t> uni;
            for (uint32_t t=CLAMP;t<35u;++t)
                for (uint32_t c=IN_COLS;c<NCOL;++c){ uint32_t b=H[(size_t)t*NCOL+c];
                    while(b){ uint32_t l=__builtin_ctz(b); b&=b-1; uni.push_back(c*NPC+l);} }
            std::sort(uni.begin(),uni.end()); uni.erase(std::unique(uni.begin(),uni.end()),uni.end());
            std::vector<uint32_t> ref;
            for (uint32_t c=IN_COLS;c<NCOL;++c){ uint32_t b=refClampBal[q][c];
                while(b){ uint32_t l=__builtin_ctz(b); b&=b-1; ref.push_back(c*NPC+l);} }
            std::sort(ref.begin(),ref.end());
            std::vector<uint32_t> in_,out_;
            std::set_intersection(uni.begin(),uni.end(),ref.begin(),ref.end(),std::back_inserter(in_));
            std::set_difference(uni.begin(),uni.end(),ref.begin(),ref.end(),std::back_inserter(out_));
            nIn+=in_.size(); nOut+=out_.size();
            runW(in_,fI,pI); runW(out_,fO,pO); runW(allRec,fA,pA);
            for(uint32_t x:out_) mOut+=mp[x];
            for(uint32_t x:allRec) mAll+=mp[x];
            nAll+=allRec.size();
        }
        printf("\nUNIONCHAR %.0f %.0f %.4f %.4f %.4f %.4f %.4f\n",
               nIn/NQ, nOut/NQ, 100.0*fI/pI, 100.0*fO/pO, 100.0*fA/pA,
               nOut>0? mOut/nOut:0.0, nAll>0? mAll/nAll:0.0);
    }

    // ===== SHARED vs EXCLUSIVE members: is flat mean density an averaging artifact? =====
    if (split) {
        uint32_t* mp=(uint32_t*)bMult.contents; memset(mp,0,(size_t)N*4);
        for (uint32_t q=0;q<NSTIM;++q)
            for (uint32_t c=IN_COLS;c<NCOL;++c){ uint32_t b=refClampBal[q][c];
                while(b){ uint32_t l=__builtin_ctz(b); b&=b-1; mp[c*NPC+l]++; } }
        uint32_t NQ=std::min(NSTIM,12u);
        double fS=0,pS=0,fE=0,pE=0,nS=0,nE=0;
        std::vector<double> dS(64,0),cS(64,0),dE(64,0),cE(64,0);
        for (uint32_t q=0;q<NQ;++q){
            memcpy(bRefBall.contents, refClampBal[q].data(), NCOL*4);
            memset(bWCnt.contents,0,128);
            { id<MTLCommandBuffer> cb=[Q commandBuffer]; id<MTLComputeCommandEncoder> ce=[cb computeCommandEncoder];
              [ce setComputePipelineState:pWithin];
              [ce setBuffer:W3 offset:0 atIndex:0];[ce setBuffer:bConn offset:0 atIndex:1];
              [ce setBuffer:bRefBall offset:0 atIndex:2];[ce setBuffer:bRefBall offset:0 atIndex:3];
              uint32_t cfg2[2]={inBranches, IN_COLS}; [ce setBytes:cfg2 length:8 atIndex:4];
              [ce setBuffer:bWCnt offset:0 atIndex:5];[ce setBuffer:bMult offset:0 atIndex:6];
              [ce setBuffer:bMult offset:0 atIndex:7]; uint32_t um0=0; [ce setBytes:&um0 length:4 atIndex:8];
              [ce setBuffer:bMult offset:0 atIndex:7]; uint32_t um1=0; [ce setBytes:&um1 length:4 atIndex:8];
              [ce dispatchThreads:MTLSizeMake(N,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
              [ce endEncoding];[cb commit];[cb waitUntilCompleted]; }
            const uint32_t* w=(const uint32_t*)bWCnt.contents;
            fS+=w[4]; pS+=w[5]; fE+=w[6]; pE+=w[7]; nS+=w[8]; nE+=w[9];
            memset(bDriveLog.contents,0,256*4);
            present(q,1.0,35u,CLAMP,false,true,8800+q);
            const uint32_t* dl=(const uint32_t*)bDriveLog.contents;
            for (uint32_t t=0;t<35u;++t){
                dS[t]+=dl[t*4+0]; cS[t]+=dl[t*4+1]; dE[t]+=dl[t*4+2]; cE[t]+=dl[t*4+3]; }
        }
        printf("\n===== SHARED vs EXCLUSIVE assembly members (NSTIM=%u) =====\n", NSTIM);
        printf("  shared fraction of assembly : %.3f  (%.0f shared / %.0f exclusive neurons)\n",
               (nS+nE)>0? nS/(nS+nE):0.0, nS/NQ, nE/NQ);
        printf("  within-assembly density  SHARED    %.4f%%  (%.0f/%.0f)\n", 100.0*fS/pS, fS, pS);
        printf("  within-assembly density  EXCLUSIVE %.4f%%  (%.0f/%.0f)\n", 100.0*fE/pE, fE, pE);
        printf("  free-phase drive  SHARED   :"); for(uint32_t t=CLAMP;t<35u;++t) printf(" %.0f", cS[t]?dS[t]/cS[t]:0.0);
        printf("\n  free-phase drive  EXCLUSIVE:"); for(uint32_t t=CLAMP;t<35u;++t) printf(" %.0f", cE[t]?dE[t]/cE[t]:0.0);
        printf("\n==========================================================\n");
    }

    // ============ DECISIVE: within-assembly recurrent scaffold vs load ============
    if (deep) {
        uint32_t NQ = std::min(NSTIM, 12u);
        double fIn=0,pIn=0,fOut=0,pOut=0;
        std::vector<double> drvSum(40,0), drvCnt(40,0);
        for (uint32_t q=0;q<NQ;++q) {
            memcpy(bRefBall.contents, refClampBal[q].data(), NCOL*4);
            memset(bWCnt.contents,0,64);
            { id<MTLCommandBuffer> cb=[Q commandBuffer]; id<MTLComputeCommandEncoder> ce=[cb computeCommandEncoder];
              [ce setComputePipelineState:pWithin];
              [ce setBuffer:W3 offset:0 atIndex:0];[ce setBuffer:bConn offset:0 atIndex:1];
              [ce setBuffer:bRefBall offset:0 atIndex:2];[ce setBuffer:bRefBall offset:0 atIndex:3];
              uint32_t cfg2[2]={inBranches, IN_COLS}; [ce setBytes:cfg2 length:8 atIndex:4];
              [ce setBuffer:bWCnt offset:0 atIndex:5];
              [ce dispatchThreads:MTLSizeMake(N,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
              [ce endEncoding];[cb commit];[cb waitUntilCompleted]; }
            const uint32_t* w=(const uint32_t*)bWCnt.contents;
            fIn+=w[0]; pIn+=w[1]; fOut+=w[2]; pOut+=w[3];
            // free-phase drive to this assembly's members, per step
            memset(bDriveLog.contents,0,256*4);
            present(q,1.0,std::min(T+20u,40u),CLAMP,false,true,8600+q);
            const uint32_t* dl=(const uint32_t*)bDriveLog.contents;
            // driveLog is 4 slots/step (shared sum,count, exclusive sum,count); sum both
            // halves. Prior to the shared/exclusive split this array was 2 slots/step and
            // this loop read it with stride 2, which misaligned the printed trace.
            for (uint32_t t=0;t<std::min(T+20u,40u);++t){
                drvSum[t]+=dl[t*4+0]+dl[t*4+2]; drvCnt[t]+=dl[t*4+1]+dl[t*4+3]; }
        }
        printf("\n========= WITHIN-ASSEMBLY RECURRENT SCAFFOLD (NSTIM=%u) =========\n", NSTIM);
        printf("  within-assembly : %.0f functional of %.0f possible = %.4f%%\n", fIn, pIn, 100.0*fIn/pIn);
        printf("  between         : %.0f functional of %.0f possible = %.4f%%\n", fOut, pOut, 100.0*fOut/pOut);
        printf("  enrichment within/between = %.2fx\n", (fOut>0&&pIn>0)? (fIn/pIn)/(fOut/pOut) : 0.0);
        printf("  free-phase drive to assembly members, per step:\n   ");
        uint32_t TT=std::min(T+20u,40u);
        for (uint32_t t=0;t<TT;++t) if(t<CLAMP+12) printf(" t%u:%.0f", t, drvCnt[t]? drvSum[t]/drvCnt[t]:0.0);
        printf("\n================================================================\n");
    }

    // ============ MULTIPLICITY OF THE RETRIEVED POPULATION + HEAVY TAIL ============
    if (deep) {
        auto nsetOf2=[&](const uint32_t* row){ std::vector<uint32_t> v;
            for(uint32_t c=IN_COLS;c<NCOL;++c){ uint32_t b=row[c];
                while(b){ uint32_t l=__builtin_ctz(b); b&=b-1; v.push_back(c*NPC+l);} } return v; };
        // multiplicity = how many reference assemblies contain each neuron
        std::vector<uint16_t> mult(N,0);
        for (uint32_t q=0;q<NSTIM;++q) for (uint32_t x : nsetOf2(refClampBal[q].data())) mult[x]++;
        double popSum=0; uint32_t popN=0, maxMult=0;
        for (uint32_t i=IN_COLS*NPC;i<N;++i) if(mult[i]){ popSum+=mult[i]; popN++; maxMult=std::max<uint32_t>(maxMult,mult[i]); }
        double popMean = popN? popSum/popN : 0.0;

        uint32_t NQ = std::min(NSTIM, 30u);
        double mIn=0,mOut=0; double nIn=0,nOut=0;
        double sumBest=0, sumMeanOther=0, sumSelf=0;
        for (uint32_t q=0;q<NQ;++q) {
            present(q,1.0,T,CLAMP,false,true,8100+q);
            const uint32_t* sp=(const uint32_t*)((T&1)?spkB.contents:spkA.contents);
            std::vector<uint32_t> rec=nsetOf2(sp), refN=nsetOf2(refClampBal[q].data());
            std::vector<uint32_t> in_,out_;
            std::set_intersection(rec.begin(),rec.end(),refN.begin(),refN.end(),std::back_inserter(in_));
            std::set_difference(rec.begin(),rec.end(),refN.begin(),refN.end(),std::back_inserter(out_));
            for(uint32_t x:in_){ mIn+=mult[x]; nIn++; }
            for(uint32_t x:out_){ mOut+=mult[x]; nOut++; }
            std::vector<uint32_t> recCols; for(uint32_t c=IN_COLS;c<NCOL;++c) if(sp[c]) recCols.push_back(c);
            sumSelf += overlap(recCols, refClampCols[q]);
            double best=0, tot=0; uint32_t cnt=0;
            for (uint32_t r=0;r<NSTIM;++r) if(r!=q){ double o=overlap(recCols,refClampCols[r]);
                best=std::max(best,o); tot+=o; cnt++; }
            sumBest+=best; sumMeanOther+= cnt? tot/cnt : 0.0;
        }
        printf("\n============ RETRIEVED-POPULATION MULTIPLICITY (NSTIM=%u) ============\n", NSTIM);
        printf("  population: %u neurons used, mean multiplicity %.2f, max %u\n", popN, popMean, maxMult);
        printf("  recalled AND in reference    : n=%.0f  mean multiplicity %.2f\n", nIn,  nIn? mIn/nIn:0.0);
        printf("  recalled but NOT in reference: n=%.0f  mean multiplicity %.2f  (%.2fx population)\n",
               nOut, nOut? mOut/nOut:0.0, (nOut&&popMean)? (mOut/nOut)/popMean : 0.0);
        printf("  self %.3f | best-other %.3f | MEAN-over-others %.3f  -> %s\n",
               sumSelf/NQ, sumBest/NQ, sumMeanOther/NQ,
               (sumMeanOther/NQ > 0.5*sumBest/NQ) ? "UNIFORM contamination" : "TAIL-dominated");

        // heavy tail: does assembly Jaccard track input overlap, or allocation index?
        auto jac2=[&](const std::vector<uint32_t>&a,const std::vector<uint32_t>&b){
            std::vector<uint32_t> i,u;
            std::set_intersection(a.begin(),a.end(),b.begin(),b.end(),std::back_inserter(i));
            std::set_union(a.begin(),a.end(),b.begin(),b.end(),std::back_inserter(u));
            return u.empty()?0.0:(double)i.size()/u.size(); };
        std::vector<std::vector<uint32_t>> ns(NSTIM);
        for(uint32_t q=0;q<NSTIM;++q){ ns[q]=nsetOf2(refClampBal[q].data()); }
        std::vector<double> J,IO,IX;
        for (uint32_t i=0;i<NSTIM;++i) for (uint32_t j=i+1;j<NSTIM;++j) {
            J.push_back(jac2(ns[i],ns[j]));
            std::vector<uint32_t> a=stimCols[i],b=stimCols[j];
            std::sort(a.begin(),a.end()); std::sort(b.begin(),b.end());
            std::vector<uint32_t> ii; std::set_intersection(a.begin(),a.end(),b.begin(),b.end(),std::back_inserter(ii));
            IO.push_back((double)ii.size());
            IX.push_back((double)std::max(i,j));
        }
        auto pear=[&](std::vector<double>&x,std::vector<double>&y){
            double mx=0,my=0; for(size_t i=0;i<x.size();++i){mx+=x[i];my+=y[i];} mx/=x.size(); my/=y.size();
            double sxy=0,sxx=0,syy=0;
            for(size_t i=0;i<x.size();++i){ double dx=x[i]-mx,dy=y[i]-my; sxy+=dx*dy; sxx+=dx*dx; syy+=dy*dy; }
            return (sxx>0&&syy>0)? sxy/std::sqrt(sxx*syy) : 0.0; };
        std::vector<double> Js=J; std::sort(Js.begin(),Js.end());
        printf("  heavy tail: %zu pairs | mean J %.5f | p99 %.5f | max %.5f\n",
               J.size(), std::accumulate(J.begin(),J.end(),0.0)/J.size(),
               Js[(size_t)(0.99*(Js.size()-1))], Js.back());
        printf("  corr(assemblyJaccard, INPUT overlap)      r = %+.3f\n", pear(J,IO));
        printf("  corr(assemblyJaccard, allocation index)   r = %+.3f\n", pear(J,IX));
        printf("======================================================================\n");
    }

    // ================== WAVE CHARACTERISATION: long free run ==================
    if (wave) {
        uint32_t TW = std::min(waveT, 60u);
        uint32_t NQ = std::min(NSTIM, 12u);
        std::vector<double> ovR(TW,0), nCol(TW,0), nNeu(TW,0), jPrev(TW,0), fracA(TW,0), jPrev2(TW,0);
        double unionSz=0, unionE=0, unionO=0, unionX=0;
        auto nsetOf=[&](const uint32_t* row){ std::vector<uint32_t> v;
            for(uint32_t c=IN_COLS;c<NCOL;++c){ uint32_t b=row[c];
                while(b){ uint32_t l=__builtin_ctz(b); b&=b-1; v.push_back(c*NPC+l);} } return v; };
        auto jc=[&](const std::vector<uint32_t>&a,const std::vector<uint32_t>&b){
            if(a.empty()||b.empty()) return 0.0; std::vector<uint32_t> i,u;
            std::set_intersection(a.begin(),a.end(),b.begin(),b.end(),std::back_inserter(i));
            std::set_union(a.begin(),a.end(),b.begin(),b.end(),std::back_inserter(u));
            return (double)i.size()/u.size(); };
        for (uint32_t q=0;q<NQ;++q) {
            present(q, 1.0, TW, CLAMP, false, true, 7000+q);
            const uint32_t* H=(const uint32_t*)bHist.contents;
            std::vector<uint32_t> refN = nsetOf(refClampBal[q].data());
            std::vector<uint32_t> prev, prev2, uni, uniE, uniO;
            for (uint32_t t=0;t<TW;++t) {
                const uint32_t* row=H+(size_t)t*NCOL;
                std::vector<uint32_t> cur=nsetOf(row);
                uint32_t nc=0; for(uint32_t c=IN_COLS;c<NCOL;++c) if(row[c]) nc++;
                nCol[t]+=nc; nNeu[t]+=cur.size();
                std::vector<uint32_t> ii;
                std::set_intersection(cur.begin(),cur.end(),refN.begin(),refN.end(),std::back_inserter(ii));
                ovR[t]  += (cur.empty()||refN.empty())?0.0:(double)ii.size()/std::sqrt((double)cur.size()*refN.size());
                fracA[t]+= refN.empty()?0.0:(double)ii.size()/refN.size();
                if (t>=CLAMP) { uni.insert(uni.end(),cur.begin(),cur.end());
                    if (t%2u==0u) uniE.insert(uniE.end(),cur.begin(),cur.end());
                    else          uniO.insert(uniO.end(),cur.begin(),cur.end()); }
                jPrev[t]  += t   ? jc(cur,prev)  : 0.0;
                jPrev2[t] += t>1 ? jc(cur,prev2) : 0.0;
                prev2=prev; prev=cur;
            }
            std::sort(uni.begin(),uni.end());
            uni.erase(std::unique(uni.begin(),uni.end()),uni.end());
            unionSz += (double)uni.size();
            std::sort(uniE.begin(),uniE.end()); uniE.erase(std::unique(uniE.begin(),uniE.end()),uniE.end());
            std::sort(uniO.begin(),uniO.end()); uniO.erase(std::unique(uniO.begin(),uniO.end()),uniO.end());
            std::vector<uint32_t> ii;
            std::set_intersection(uniE.begin(),uniE.end(),uniO.begin(),uniO.end(),std::back_inserter(ii));
            unionE += (double)uniE.size(); unionO += (double)uniO.size();
            unionX += (uniE.empty()||uniO.empty())?0.0:(double)ii.size()/std::sqrt((double)uniE.size()*uniO.size());
        }
        printf("\nPARITY %.0f %.4f %.0f %.4f %.0f %.4f %.4f\n",
               unionSz/NQ, unionSz/NQ/1536.0, unionE/NQ, unionE/NQ/1536.0,
               unionO/NQ, unionO/NQ/1536.0, unionX/NQ);   // union uE uEx uO uOx overlap
        printf("\n================= WAVE: %u free steps after a %u-step cue =================\n", TW-CLAMP, CLAMP);
        printf("  step  phase   cols  neurons   ov(clampRef)  fracOfAssembly  J(t,t-1)   J(t,t-2)\n");
        for (uint32_t t=0;t<TW;++t)
            printf("  %4u  %-5s %6.0f %8.0f %13.3f %15.3f %10.3f %10.3f\n",
                   t, t<CLAMP?"cue":"FREE", nCol[t]/NQ, nNeu[t]/NQ, ovR[t]/NQ, fracA[t]/NQ,
                   jPrev[t]/NQ, jPrev2[t]/NQ);
        printf("==========================================================================\n");
    }

    // ---------------- STEP 2 / blocking diagnostic: free-run trajectory ----------------
    {
        auto colsOf=[&](const uint32_t* sp)->std::vector<uint32_t>{
            std::vector<uint32_t> v; for(uint32_t c=IN_COLS;c<NCOL;++c) if(sp[c]) v.push_back(c); return v; };
        printf("\n============ FREE-RUN DIAGNOSTIC (cue %u steps, then released) ============\n", CLAMP);
        printf("  cue=75%%, averaged over %u stimuli.  'margin' = k-th minus (k+1)-th drive\n", NSTIM);
        printf("  step  phase  cols  drive_k  margin  %%marg=0   ov(CLAMPED ref)  ov(FREE ref)\n");
        std::vector<double> ovc(T,0), ovt(T,0), mg(T,0), dk(T,0), tie(T,0), zer(T,0), nc(T,0);
        std::vector<double> ovf(T,0);
        for (uint32_t q=0;q<NSTIM;++q) {
            present(q, 0.75, T, CLAMP, false, true, 4000+q);
            const uint32_t* H=(const uint32_t*)bHist.contents;
            const uint32_t* S=(const uint32_t*)bStat.contents;
            std::vector<uint32_t> ref = colsOf(H + (size_t)(CLAMP-1)*NCOL);
            for (uint32_t t=0;t<T;++t) {
                auto a = colsOf(H + (size_t)t*NCOL);
                ovc[t]+=overlap(a,ref); ovt[t]+=overlap(a,refClampCols[q]); ovf[t]+=overlap(a,refFreeCols[q]);
                const uint32_t* sb=S+64+t*8;
                double n=sb[2]; nc[t]+=n;
                if(n>0){ mg[t]+=sb[0]/n; dk[t]+=sb[1]/n; tie[t]+=100.0*sb[3]/n; zer[t]+=100.0*sb[4]/n; }
            }
        }
        for (uint32_t t=0;t<T;++t)
            printf("  %4u  %-5s %5.0f %8.1f %7.1f %8.1f   %14.3f  %12.3f\n",
                   t, t<CLAMP?"cue":"FREE", nc[t]/NSTIM, dk[t]/NSTIM, mg[t]/NSTIM,
                   tie[t]/NSTIM, ovt[t]/NSTIM, ovf[t]/NSTIM);
        printf("==========================================================================\n");
    }

    double m50=0;
    for (double keep : {1.0,0.75,0.5,0.35,0.25}) {
        double self=0,other=0,selfF=0,otherF=0;
        for (uint32_t p=0;p<NSTIM;++p){
            present(p,keep,T,CLAMP,false,true,3000+p);
            auto a=assemblyOf(T);
            self+=overlap(a,refClampCols[p]); selfF+=overlap(a,refFreeCols[p]);
            double best=0,bestF=0;
            for(uint32_t q=0;q<NSTIM;++q) if(q!=p){
                best=std::max(best,overlap(a,refClampCols[q]));
                bestF=std::max(bestF,overlap(a,refFreeCols[q])); }
            other+=best; otherF+=bestF;
        }
        self/=NSTIM; other/=NSTIM; selfF/=NSTIM; otherF/=NSTIM;
        if(keep==0.5) m50=self-other;
        if(!quiet) printf("  %3.0f%% |  %16.3f %6.3f %+7.3f  |  %11.3f %6.3f %+7.3f\n",
                          keep*100, self, other, self-other, selfF, otherF, selfF-otherF);
    }
    if(quiet) printf("tgt=%-4u k=%-3u inbr=%-3u stimc=%-4u bias=%-4u nand=%u | %7.4f ms %6.0f st/s asm=%-5.0f xtalk=%.3f margin50=%+.3f wd=%.2f\n",
                     targetC,K,inBranches,STIM_COLS,biasUp,nAnd,msStep,1000.0/msStep,sz,xtalk,m50,wstat());
    else printf("\n=== DONE ===\n");
}
return 0;}

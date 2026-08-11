// probe05_prop.mm — the AXON engine core.
//
// Geometry forced by F6 (probe04):
//   one simdgroup = one COLUMN of 32 postsynaptic neurons (lane = neuron)
//   one coalesced uint4 load = 32 postsyn x 128 presyn BINARY weight tile (512 B)
//   tile is skipped iff the 128 presyn neurons are all silent -- simdgroup-uniform
//
// Measures:
//   A. propagation throughput vs fraction of active presynaptic groups
//   B. cost of exact k-winners-take-all inside a column (rank via simd_shuffle)
//   C. correctness of k-WTA + ballot
//
// N = 131072 neurons, fan-in 8192, 1.07 G binary synapses, W = 128 MB.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <cstdio>
#include <chrono>
#include <vector>
#include <algorithm>

static const char* kSrc = R"MSL(
#include <metal_stdlib>
using namespace metal;

inline uint hash32(uint x){ x^=x>>16; x*=0x7feb352du; x^=x>>15; x*=0x846ca68bu; x^=x>>16; return x; }

kernel void warmup(device float* b [[buffer(0)]], constant uint& n [[buffer(1)]],
                   uint g [[thread_position_in_grid]])
{ float x=b[g]; for(uint i=0;i<n;++i) x=fma(x,1.0000001f,0.0000001f); b[g]=x; }

kernel void fill_w(device uint4* w [[buffer(0)]], uint g [[thread_position_in_grid]])
{ w[g] = uint4(hash32(g*4u+0),hash32(g*4u+1),hash32(g*4u+2),hash32(g*4u+3)); }

kernel void fill_conn(device uint* c [[buffer(0)]], constant uint& nGrp [[buffer(1)]],
                      uint g [[thread_position_in_grid]])
{ c[g] = hash32(g*2246822519u) % nGrp; }

// ---------------------------------------------------------------------------
// PROPAGATION
// mode 0: skip silent tiles (block-event-driven)
// mode 1: never skip (dense reference)
// ---------------------------------------------------------------------------
kernel void prop(device const uint4* W      [[buffer(0)]],
                 device const uint*  conn   [[buffer(1)]],
                 device uint*        outAcc [[buffer(2)]],
                 constant uint&      S      [[buffer(3)]],
                 constant uint&      step   [[buffer(4)]],
                 constant uint&      actPM  [[buffer(5)]],   // active groups, per mille
                 constant uint&      mode   [[buffer(6)]],
                 uint gid  [[thread_position_in_grid]],
                 uint lane [[thread_index_in_simdgroup]])
{
    uint col = gid >> 5;
    uint acc = 0;
    uint base = col * S;
    for (uint s = 0; s < S; ++s) {
        uint g = conn[base + s];
        // presyn group activity, derived not loaded, so this measures WEIGHT traffic only
        uint h = hash32(g * 2654435761u + step * 40503u);
        bool active = (h % 1000u) < actPM;
        uint4 sp = uint4(0);
        if (active) sp = uint4(hash32(h+1),hash32(h+2),hash32(h+3),hash32(h+4));
        if (mode == 1 || active) {                       // simdgroup-uniform branch
            uint4 w = W[(base + s) * 32 + lane];
            uint4 pc = popcount(w & sp);
            acc += pc.x + pc.y + pc.z + pc.w;
        }
    }
    outAcc[gid] = acc;
}

// ---------------------------------------------------------------------------
// EXACT k-WINNERS-TAKE-ALL inside a 32-neuron column.
// rank(i) = #{ j : (v_j, j) >lex (v_i, i) };  fire iff rank < k.
// 32 simd_shuffles, no shared memory, deterministic ties. Result is a ballot
// word: one bit per neuron, exactly k bits set.
// ---------------------------------------------------------------------------
inline uint kwta_ballot(uint v, uint lane, uint k)
{
    uint rank = 0;
    for (uint j = 0; j < 32; ++j) {
        uint ov = simd_shuffle(v, j);
        // lexicographic (value, lane) so ties break by lane index -> exactly k winners
        bool greater = (ov > v) || (ov == v && j < lane);
        rank += greater ? 1u : 0u;
    }
    bool fire = rank < k;
    return (uint)((simd_vote::vote_t)simd_ballot(fire));
}

kernel void prop_kwta(device const uint4* W      [[buffer(0)]],
                      device const uint*  conn   [[buffer(1)]],
                      device uint*        outSpk [[buffer(2)]],
                      constant uint&      S      [[buffer(3)]],
                      constant uint&      step   [[buffer(4)]],
                      constant uint&      actPM  [[buffer(5)]],
                      constant uint&      k      [[buffer(6)]],
                      uint gid  [[thread_position_in_grid]],
                      uint lane [[thread_index_in_simdgroup]])
{
    uint col = gid >> 5;
    uint acc = 0, base = col * S;
    for (uint s = 0; s < S; ++s) {
        uint g = conn[base + s];
        uint h = hash32(g * 2654435761u + step * 40503u);
        if ((h % 1000u) < actPM) {
            uint4 sp = uint4(hash32(h+1),hash32(h+2),hash32(h+3),hash32(h+4));
            uint4 w = W[(base + s) * 32 + lane];
            uint4 pc = popcount(w & sp);
            acc += pc.x + pc.y + pc.z + pc.w;
        }
    }
    uint b = kwta_ballot(acc, lane, k);
    if (lane == 0) outSpk[col] = b;
}

// isolated k-WTA cost + correctness (no memory traffic at all)
kernel void kwta_only(device uint* outSpk [[buffer(0)]], constant uint& k [[buffer(1)]],
                      constant uint& reps [[buffer(2)]],
                      uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]])
{
    uint col = gid >> 5;
    uint v = hash32(gid * 2654435761u) & 0xFFFFu;
    uint b = 0;
    for (uint r = 0; r < reps; ++r) { b = kwta_ballot(v + r, lane, k); v ^= (b & 1u); }
    if (lane == 0) outSpk[col] = b;
}
)MSL";

static id<MTLDevice> D; static id<MTLCommandQueue> Q; static id<MTLLibrary> L;
static id<MTLComputePipelineState> P(NSString* n){NSError*e=nil;
    id<MTLComputePipelineState> p=[D newComputePipelineStateWithFunction:[L newFunctionWithName:n] error:&e];
    if(!p){printf("pipe %s: %s\n",n.UTF8String,e.localizedDescription.UTF8String);exit(3);} return p;}

int main(){
@autoreleasepool{
    D=MTLCreateSystemDefaultDevice(); Q=[D newCommandQueue];
    NSError*e=nil; MTLCompileOptions*o=[MTLCompileOptions new]; o.fastMathEnabled=YES;
    L=[D newLibraryWithSource:[NSString stringWithUTF8String:kSrc] options:o error:&e];
    if(!L){printf("COMPILE FAIL %s\n",e.localizedDescription.UTF8String);return 2;}
    id<MTLComputePipelineState> pW=P(@"warmup"), pFW=P(@"fill_w"), pFC=P(@"fill_conn"),
        pProp=P(@"prop"), pPK=P(@"prop_kwta"), pKO=P(@"kwta_only");

    const uint32_t NCOL = 4096;            // columns of 32 neurons
    const uint32_t N    = NCOL*32;         // 131072 neurons
    const uint32_t NGRP = N/128;           // 1024 presyn groups of 128
    const uint32_t S    = 64;              // fan-in slots per column
    const size_t   WBYTES = (size_t)NCOL*S*32*16;

    printf("=== AXON core geometry ===\n");
    printf("  neurons          : %u  (%u columns x 32)\n", N, NCOL);
    printf("  presyn groups    : %u  (x128 neurons)\n", NGRP);
    printf("  slots/column     : %u  -> fan-in %u synapses/neuron\n", S, S*128);
    printf("  total synapses   : %.2f G (binary)\n", (double)N*S*128/1e9);
    printf("  weight memory    : %.1f MB\n", WBYTES/1048576.0);

    id<MTLBuffer> bW   =[D newBufferWithLength:WBYTES options:MTLResourceStorageModePrivate];
    id<MTLBuffer> bConn=[D newBufferWithLength:NCOL*S*4 options:MTLResourceStorageModePrivate];
    id<MTLBuffer> bAcc =[D newBufferWithLength:N*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bSpk =[D newBufferWithLength:NCOL*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bWarm=[D newBufferWithLength:1<<22 options:MTLResourceStorageModePrivate];

    { id<MTLCommandBuffer> cb=[Q commandBuffer]; id<MTLComputeCommandEncoder> ce=[cb computeCommandEncoder];
      [ce setComputePipelineState:pFW]; [ce setBuffer:bW offset:0 atIndex:0];
      [ce dispatchThreads:MTLSizeMake(WBYTES/16,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
      [ce setComputePipelineState:pFC]; [ce setBuffer:bConn offset:0 atIndex:0];
      uint32_t ng=NGRP; [ce setBytes:&ng length:4 atIndex:1];
      [ce dispatchThreads:MTLSizeMake(NCOL*S,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
      [ce endEncoding];[cb commit];[cb waitUntilCompleted]; }

    auto burn=[&](int ms){auto t0=std::chrono::steady_clock::now();
        while(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-t0).count()<ms){
            id<MTLCommandBuffer> cb=[Q commandBuffer];id<MTLComputeCommandEncoder> ce=[cb computeCommandEncoder];
            [ce setComputePipelineState:pW];[ce setBuffer:bWarm offset:0 atIndex:0];
            uint32_t it=20000;[ce setBytes:&it length:4 atIndex:1];
            [ce dispatchThreads:MTLSizeMake(1<<20,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [ce endEncoding];[cb commit];[cb waitUntilCompleted];}};
    printf("\n[warm 2000ms]\n"); burn(2000);

    // ---- A. propagation vs activity ----
    printf("\n=== A. BLOCK-SPARSE BINARY PROPAGATION ===\n");
    printf("  active%%   mode        ms/step   steps/s    GB/s    Gsyn/s   TOPS(bin)\n");
    auto runProp=[&](uint32_t actPM, uint32_t mode, uint32_t steps)->double{
        burn(80);
        auto t0=std::chrono::steady_clock::now();
        id<MTLCommandBuffer> cb=[Q commandBuffer];
        id<MTLComputeCommandEncoder> ce=[cb computeCommandEncoder];
        [ce setComputePipelineState:pProp];
        [ce setBuffer:bW offset:0 atIndex:0];[ce setBuffer:bConn offset:0 atIndex:1];
        [ce setBuffer:bAcc offset:0 atIndex:2];
        uint32_t s=S; [ce setBytes:&s length:4 atIndex:3];
        [ce setBytes:&actPM length:4 atIndex:5];[ce setBytes:&mode length:4 atIndex:6];
        for(uint32_t st=0; st<steps; ++st){
            [ce setBytes:&st length:4 atIndex:4];
            [ce dispatchThreads:MTLSizeMake(N,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        }
        [ce endEncoding];[cb commit];[cb waitUntilCompleted];
        return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count()/steps;
    };
    for (uint32_t actPM : {10u,30u,50u,100u,250u,500u,1000u}) {
        for (uint32_t mode : {0u,1u}) {
            if (mode==1 && actPM!=1000u) continue;                 // dense ref once
            uint32_t steps = (actPM<=100u)?400u:100u;
            double ms = runProp(actPM, mode, steps);
            double frac = (mode==1)?1.0:(actPM/1000.0);
            double bytes = (double)NCOL*S*frac*512.0;
            double syn   = (double)N*S*128.0*frac;
            printf("  %6.1f   %-10s  %7.3f  %8.0f  %6.1f  %8.2f   %7.2f\n",
                   frac*100.0, mode?"dense":"skip", ms, 1000.0/ms,
                   (bytes/1e9)/(ms/1e3), (syn/1e9)/(ms/1e3), (2.0*syn/1e12)/(ms/1e3));
        }
    }

    // ---- B. k-WTA isolated cost ----
    printf("\n=== B. EXACT k-WTA COST (32 simd_shuffle, no memory) ===\n");
    for (uint32_t k : {1u,2u,4u,8u}) {
        uint32_t reps=64, steps=100;
        burn(60);
        auto t0=std::chrono::steady_clock::now();
        id<MTLCommandBuffer> cb=[Q commandBuffer];
        id<MTLComputeCommandEncoder> ce=[cb computeCommandEncoder];
        [ce setComputePipelineState:pKO];
        [ce setBuffer:bSpk offset:0 atIndex:0];
        [ce setBytes:&k length:4 atIndex:1];[ce setBytes:&reps length:4 atIndex:2];
        for(uint32_t st=0;st<steps;++st)
            [ce dispatchThreads:MTLSizeMake(N,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [ce endEncoding];[cb commit];[cb waitUntilCompleted];
        double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();
        double perKwta = (ms/1e3)/((double)steps*reps*NCOL);
        printf("  k=%u : %.3f ms for %u reps x %u cols  ->  %.1f ns per column-kWTA, %.2f G kWTA/s\n",
               k, ms/steps, reps, NCOL, perKwta*1e9, 1.0/perKwta/1e9);
    }

    // ---- C. correctness: exactly k bits set ----
    printf("\n=== C. k-WTA CORRECTNESS (popcount of ballot must equal k) ===\n");
    for (uint32_t k : {1u,3u,5u,8u,16u}) {
        uint32_t reps=1;
        id<MTLCommandBuffer> cb=[Q commandBuffer];
        id<MTLComputeCommandEncoder> ce=[cb computeCommandEncoder];
        [ce setComputePipelineState:pKO];
        [ce setBuffer:bSpk offset:0 atIndex:0];
        [ce setBytes:&k length:4 atIndex:1];[ce setBytes:&reps length:4 atIndex:2];
        [ce dispatchThreads:MTLSizeMake(N,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [ce endEncoding];[cb commit];[cb waitUntilCompleted];
        uint32_t* sp=(uint32_t*)bSpk.contents; uint32_t bad=0;
        for(uint32_t c=0;c<NCOL;++c) if(__builtin_popcount(sp[c])!=(int)k) bad++;
        printf("  k=%2u : %u/%u columns wrong  %s   (sample ballot 0x%08X)\n",
               k, bad, NCOL, bad?"FAIL":"PASS", sp[0]);
    }

    // ---- D. fused propagation + k-WTA ----
    printf("\n=== D. FUSED prop + k-WTA (what a real timestep costs) ===\n");
    printf("  active%%   ms/step   steps/s   x-realtime(1ms bio)\n");
    for (uint32_t actPM : {10u,30u,50u,100u}) {
        uint32_t steps=400, k=4;
        burn(80);
        auto t0=std::chrono::steady_clock::now();
        id<MTLCommandBuffer> cb=[Q commandBuffer];
        id<MTLComputeCommandEncoder> ce=[cb computeCommandEncoder];
        [ce setComputePipelineState:pPK];
        [ce setBuffer:bW offset:0 atIndex:0];[ce setBuffer:bConn offset:0 atIndex:1];
        [ce setBuffer:bSpk offset:0 atIndex:2];
        uint32_t s=S;[ce setBytes:&s length:4 atIndex:3];
        [ce setBytes:&actPM length:4 atIndex:5];[ce setBytes:&k length:4 atIndex:6];
        for(uint32_t st=0;st<steps;++st){ [ce setBytes:&st length:4 atIndex:4];
            [ce dispatchThreads:MTLSizeMake(N,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; }
        [ce endEncoding];[cb commit];[cb waitUntilCompleted];
        double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count()/steps;
        printf("  %6.1f   %7.4f  %8.0f   %6.1fx\n", actPM/10.0, ms, 1000.0/ms, 1.0/ms*1.0);
    }
    printf("\n=== DONE ===\n");
}
return 0;}

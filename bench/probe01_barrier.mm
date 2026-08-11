// probe01_barrier.mm — The load-bearing uncertainty:
//   (a) how many threadgroups are TRULY co-resident (cold vs warm GPU)
//   (b) does a sense-reversing device-wide barrier work at all
//   (c) what does one grid barrier cost in nanoseconds
//
// clang++ -std=c++17 -fobjc-arc -O2 probe01_barrier.mm -framework Metal -framework Foundation -o probe01

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <cstdio>
#include <vector>
#include <chrono>

static const char* kSrc = R"MSL(
#include <metal_stdlib>
using namespace metal;

// Burn kernel: ramp the GPU power state before we measure anything.
kernel void warmup(device float* buf [[buffer(0)]],
                   constant uint& iters [[buffer(1)]],
                   uint gid [[thread_position_in_grid]])
{
    float x = buf[gid];
    for (uint i = 0; i < iters; ++i) x = fma(x, 1.0000001f, 0.0000001f);
    buf[gid] = x;
}

// Residency probe that reports HOW FAR it got, not just pass/fail.
kernel void residency(device atomic_uint* arrive  [[buffer(0)]],
                      device uint*        maxseen [[buffer(1)]],
                      constant uint&      N       [[buffer(2)]],
                      constant uint&      spinCap [[buffer(3)]],
                      uint tgid [[threadgroup_position_in_grid]],
                      uint tid  [[thread_position_in_threadgroup]])
{
    if (tid == 0) {
        atomic_fetch_add_explicit(arrive, 1u, memory_order_relaxed);
        uint seen = 0;
        for (uint i = 0; i < spinCap; ++i) {
            seen = atomic_load_explicit(arrive, memory_order_relaxed);
            if (seen >= N) break;
        }
        maxseen[tgid] = seen;
    }
}

// ---------------------------------------------------------------------------
// Sense-reversing device-wide barrier.
// MSL only guarantees memory_order_relaxed on device atomics, so ordering of
// ordinary device memory around the barrier is enforced with an explicit
// device memory fence (threadgroup_barrier with mem_device) on both sides.
// ---------------------------------------------------------------------------
inline bool grid_barrier(device atomic_uint* count,
                         device atomic_uint* gen,
                         threadgroup uint& ok_tg,
                         uint numTG, uint tid, uint spinCap)
{
    // order all prior device writes from this threadgroup before arrival
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

    bool ok = true;
    if (tid == 0) {
        uint myGen = atomic_load_explicit(gen, memory_order_relaxed);
        uint prev  = atomic_fetch_add_explicit(count, 1u, memory_order_relaxed);
        if (prev + 1 == numTG) {
            // last one in: reset and release everybody
            atomic_store_explicit(count, 0u, memory_order_relaxed);
            atomic_fetch_add_explicit(gen, 1u, memory_order_relaxed);
        } else {
            ok = false;
            for (uint i = 0; i < spinCap; ++i) {
                if (atomic_load_explicit(gen, memory_order_relaxed) != myGen) { ok = true; break; }
            }
        }
        ok_tg = ok ? 1u : 0u;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    return ok_tg != 0u;
}

kernel void barrier_bench(device atomic_uint* count   [[buffer(0)]],
                          device atomic_uint* gen     [[buffer(1)]],
                          device uint*        flags   [[buffer(2)]],
                          device uint*        payload [[buffer(3)]],
                          constant uint&      numTG   [[buffer(4)]],
                          constant uint&      rounds  [[buffer(5)]],
                          constant uint&      spinCap [[buffer(6)]],
                          uint tgid [[threadgroup_position_in_grid]],
                          uint tid  [[thread_position_in_threadgroup]],
                          uint tgsz [[threads_per_threadgroup]])
{
    threadgroup uint ok_tg;
    uint fails = 0;
    uint corrupt = 0;
    for (uint r = 0; r < rounds; ++r) {
        // Each TG writes its own slot; after the barrier every TG must see
        // round r in EVERY slot. This is a real correctness test of the fence,
        // not just a liveness test.
        if (tid == 0) payload[tgid] = r + 1;
        bool ok = grid_barrier(count, gen, ok_tg, numTG, tid, spinCap);
        if (!ok) { fails++; break; }
        if (tid == 0) {
            for (uint j = 0; j < numTG; ++j)
                if (payload[j] != r + 1) corrupt++;
        }
        ok = grid_barrier(count, gen, ok_tg, numTG, tid, spinCap);   // second barrier: don't race the readers
        if (!ok) { fails++; break; }
    }
    if (tid == 0) { flags[tgid*2+0] = fails; flags[tgid*2+1] = corrupt; }
}
)MSL";

static id<MTLDevice> gDev; static id<MTLCommandQueue> gQ; static id<MTLLibrary> gLib;
static id<MTLComputePipelineState> mkPipe(NSString* n) {
    NSError* e = nil;
    id<MTLComputePipelineState> p = [gDev newComputePipelineStateWithFunction:[gLib newFunctionWithName:n] error:&e];
    if (!p) { printf("pipeline %s failed: %s\n", n.UTF8String, e.localizedDescription.UTF8String); exit(3); }
    return p;
}

int main(int argc, char** argv) {
@autoreleasepool {
    gDev = MTLCreateSystemDefaultDevice();
    gQ = [gDev newCommandQueue];
    NSError* err = nil;
    MTLCompileOptions* opts = [MTLCompileOptions new]; opts.fastMathEnabled = YES;
    gLib = [gDev newLibraryWithSource:[NSString stringWithUTF8String:kSrc] options:opts error:&err];
    if (!gLib) { printf("COMPILE FAIL: %s\n", err.localizedDescription.UTF8String); return 2; }

    id<MTLComputePipelineState> pWarm = mkPipe(@"warmup");
    id<MTLComputePipelineState> pRes  = mkPipe(@"residency");
    id<MTLComputePipelineState> pBar  = mkPipe(@"barrier_bench");
    printf("barrier_bench: maxTotalThreadsPerTG=%lu  threadExecWidth=%lu\n",
           (unsigned long)pBar.maxTotalThreadsPerThreadgroup, (unsigned long)pBar.threadExecutionWidth);

    // ---------------- warm the GPU ----------------
    id<MTLBuffer> bWarm = [gDev newBufferWithLength:1<<22 options:MTLResourceStorageModePrivate];
    auto burn = [&](int ms){
        auto t0 = std::chrono::steady_clock::now();
        while (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-t0).count() < ms) {
            id<MTLCommandBuffer> cb = [gQ commandBuffer];
            id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
            [ce setComputePipelineState:pWarm];
            [ce setBuffer:bWarm offset:0 atIndex:0];
            uint32_t it = 20000; [ce setBytes:&it length:4 atIndex:1];
            [ce dispatchThreads:MTLSizeMake(1<<20,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [ce endEncoding]; [cb commit]; [cb waitUntilCompleted];
        }
    };

    // ---------------- residency sweep ----------------
    id<MTLBuffer> bArrive = [gDev newBufferWithLength:64 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bSeen   = [gDev newBufferWithLength:8192*4 options:MTLResourceStorageModeShared];

    auto residency = [&](uint32_t N, uint32_t tgsize)->bool{
        memset(bArrive.contents, 0, 64);
        memset(bSeen.contents, 0, 8192*4);
        uint32_t spinCap = 4000000u;
        id<MTLCommandBuffer> cb = [gQ commandBuffer];
        id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
        [ce setComputePipelineState:pRes];
        [ce setBuffer:bArrive offset:0 atIndex:0];
        [ce setBuffer:bSeen offset:0 atIndex:1];
        [ce setBytes:&N length:4 atIndex:2];
        [ce setBytes:&spinCap length:4 atIndex:3];
        [ce dispatchThreadgroups:MTLSizeMake(N,1,1) threadsPerThreadgroup:MTLSizeMake(tgsize,1,1)];
        [ce endEncoding]; [cb commit]; [cb waitUntilCompleted];
        if (cb.error) return false;
        uint32_t* s = (uint32_t*)bSeen.contents; uint32_t ok=0;
        for (uint32_t i=0;i<N;++i) if (s[i] >= N) ok++;
        return ok == N;
    };

    auto sweep = [&](const char* tag){
        printf("\n=== RESIDENCY SWEEP (%s) ===\n", tag);
        for (uint32_t tgsize : {32u,64u,128u,256u,512u,1024u}) {
            if (tgsize > pRes.maxTotalThreadsPerThreadgroup) continue;
            uint32_t best = 0;
            for (uint32_t N = 1; N <= 4096; N *= 2) {
                if (residency(N, tgsize)) best = N; else break;
            }
            // refine upward linearly from best
            uint32_t refined = best;
            for (uint32_t N = best+1; N <= best*2 && N <= 4096; ++N) {
                if (residency(N, tgsize)) refined = N; else break;
            }
            printf("  tgsize %4u -> %4u co-resident TGs = %6u threads (%5u simdgroups)\n",
                   tgsize, refined, refined*tgsize, refined*tgsize/32);
        }
    };

    sweep("COLD");
    printf("\n[warming GPU 1500 ms ...]\n"); burn(1500);
    sweep("WARM");

    // ---------------- barrier benchmark ----------------
    printf("\n=== DEVICE-WIDE BARRIER: CORRECTNESS + COST ===\n");
    id<MTLBuffer> bCount = [gDev newBufferWithLength:64 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bGen   = [gDev newBufferWithLength:64 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bFlags = [gDev newBufferWithLength:8192*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bPay   = [gDev newBufferWithLength:8192*4 options:MTLResourceStorageModeShared];

    for (uint32_t tgsize : {64u, 256u}) {
      for (uint32_t numTG : {4u, 8u, 16u, 24u, 32u, 48u, 64u}) {
        memset(bCount.contents,0,64); memset(bGen.contents,0,64);
        memset(bFlags.contents,0,8192*4); memset(bPay.contents,0,8192*4);
        uint32_t rounds = 2000, spinCap = 8000000u;
        burn(120);  // keep clocks up
        auto t0 = std::chrono::steady_clock::now();
        id<MTLCommandBuffer> cb = [gQ commandBuffer];
        id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
        [ce setComputePipelineState:pBar];
        [ce setBuffer:bCount offset:0 atIndex:0];
        [ce setBuffer:bGen offset:0 atIndex:1];
        [ce setBuffer:bFlags offset:0 atIndex:2];
        [ce setBuffer:bPay offset:0 atIndex:3];
        [ce setBytes:&numTG length:4 atIndex:4];
        [ce setBytes:&rounds length:4 atIndex:5];
        [ce setBytes:&spinCap length:4 atIndex:6];
        [ce dispatchThreadgroups:MTLSizeMake(numTG,1,1) threadsPerThreadgroup:MTLSizeMake(tgsize,1,1)];
        [ce endEncoding]; [cb commit]; [cb waitUntilCompleted];
        double ms = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();
        uint32_t* f = (uint32_t*)bFlags.contents;
        uint32_t fails=0, corrupt=0;
        for (uint32_t i=0;i<numTG;++i){ fails+=f[i*2]; corrupt+=f[i*2+1]; }
        const char* status = cb.error ? "CB-ERROR" : (fails ? "TIMEOUT" : (corrupt ? "INCOHERENT" : "OK"));
        double nsPerBarrier = (ms*1e6) / (double)(rounds*2);
        printf("  tgsize %4u  TGs %3u : %-10s  %8.2f ms  %8.1f ns/barrier  (fails=%u corrupt=%u)\n",
               tgsize, numTG, status, ms, nsPerBarrier, fails, corrupt);
        if (fails) break;   // beyond residency, no point going higher
      }
      printf("\n");
    }
    printf("=== DONE ===\n");
}
    return 0;
}

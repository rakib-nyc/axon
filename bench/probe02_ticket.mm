// probe02_ticket.mm — ticket-barrier (no reset => no relaxed-atomic reorder hazard)
//
// Answers the decision-relevant question: is a persistent megakernel with a
// device-wide barrier actually CHEAPER than just issuing one dispatch per
// timestep? If not, the whole megakernel idea is dead and we use dispatches.
//
// clang++ -std=c++17 -fobjc-arc -O2 probe02_ticket.mm -framework Metal -framework Foundation -o probe02

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <cstdio>
#include <vector>
#include <chrono>
#include <algorithm>

static const char* kSrc = R"MSL(
#include <metal_stdlib>
using namespace metal;

kernel void warmup(device float* buf [[buffer(0)]], constant uint& iters [[buffer(1)]],
                   uint gid [[thread_position_in_grid]])
{ float x = buf[gid]; for (uint i=0;i<iters;++i) x = fma(x,1.0000001f,0.0000001f); buf[gid]=x; }

kernel void residency(device atomic_uint* arrive [[buffer(0)]], device uint* maxseen [[buffer(1)]],
                      constant uint& N [[buffer(2)]], constant uint& spinCap [[buffer(3)]],
                      uint tgid [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]])
{
    if (tid == 0) {
        atomic_fetch_add_explicit(arrive, 1u, memory_order_relaxed);
        uint seen = 0;
        for (uint i=0;i<spinCap;++i){ seen = atomic_load_explicit(arrive, memory_order_relaxed); if (seen>=N) break; }
        maxseen[tgid] = seen;
    }
}

// ---------------------------------------------------------------------------
// TICKET BARRIER
// One monotonic counter. Arrival takes a ticket; a threadgroup leaves when the
// counter has passed the end of its own round. Nothing is ever reset, so there
// is no second atomic to be reordered against -- the bug class is designed out
// rather than fenced around.
// Invariant: a TG cannot take its (r+1)-th ticket until count >= (r+1)*N,
// which already implies all N TGs arrived in round r. So no early release.
// ---------------------------------------------------------------------------
inline bool grid_barrier(device atomic_uint* count, threadgroup uint& ok_tg,
                         uint numTG, uint tid, uint spinCap)
{
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (tid == 0) {
        uint ticket = atomic_fetch_add_explicit(count, 1u, memory_order_relaxed);
        uint target = (ticket / numTG + 1u) * numTG;
        bool ok = false;
        for (uint i = 0; i < spinCap; ++i) {
            if (atomic_load_explicit(count, memory_order_relaxed) >= target) { ok = true; break; }
        }
        ok_tg = ok ? 1u : 0u;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    return ok_tg != 0u;
}

kernel void barrier_bench(device atomic_uint* count [[buffer(0)]],
                          device uint* flags        [[buffer(1)]],
                          device uint* payload      [[buffer(2)]],
                          constant uint& numTG      [[buffer(3)]],
                          constant uint& rounds     [[buffer(4)]],
                          constant uint& spinCap    [[buffer(5)]],
                          uint tgid [[threadgroup_position_in_grid]],
                          uint tid  [[thread_position_in_threadgroup]])
{
    threadgroup uint ok_tg;
    uint fails = 0, corrupt = 0;
    for (uint r = 0; r < rounds; ++r) {
        if (tid == 0) payload[tgid] = r + 1;
        if (!grid_barrier(count, ok_tg, numTG, tid, spinCap)) { fails++; break; }
        if (tid == 0) for (uint j = 0; j < numTG; ++j) if (payload[j] != r+1) corrupt++;
        if (!grid_barrier(count, ok_tg, numTG, tid, spinCap)) { fails++; break; }
    }
    if (tid == 0) { flags[tgid*2+0]=fails; flags[tgid*2+1]=corrupt; }
}

// Baseline for comparison: the SAME work as one barrier round, but expressed
// as one dispatch per step so Metal's own inter-dispatch barrier does the sync.
kernel void step_kernel(device uint* payload [[buffer(0)]], constant uint& r [[buffer(1)]],
                        uint tgid [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]])
{ if (tid == 0) payload[tgid] = r + 1; }
)MSL";

static id<MTLDevice> gDev; static id<MTLCommandQueue> gQ; static id<MTLLibrary> gLib;
static id<MTLComputePipelineState> mkPipe(NSString* n) {
    NSError* e=nil;
    id<MTLComputePipelineState> p=[gDev newComputePipelineStateWithFunction:[gLib newFunctionWithName:n] error:&e];
    if(!p){printf("pipeline %s: %s\n",n.UTF8String,e.localizedDescription.UTF8String);exit(3);} return p;
}

int main() {
@autoreleasepool {
    gDev = MTLCreateSystemDefaultDevice(); gQ = [gDev newCommandQueue];
    NSError* err=nil; MTLCompileOptions* o=[MTLCompileOptions new]; o.fastMathEnabled=YES;
    gLib=[gDev newLibraryWithSource:[NSString stringWithUTF8String:kSrc] options:o error:&err];
    if(!gLib){printf("COMPILE FAIL: %s\n",err.localizedDescription.UTF8String);return 2;}

    id<MTLComputePipelineState> pWarm=mkPipe(@"warmup"), pRes=mkPipe(@"residency"),
                                pBar=mkPipe(@"barrier_bench"), pStep=mkPipe(@"step_kernel");

    id<MTLBuffer> bWarm=[gDev newBufferWithLength:1<<22 options:MTLResourceStorageModePrivate];
    auto burn=[&](int ms){
        auto t0=std::chrono::steady_clock::now();
        while(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-t0).count()<ms){
            id<MTLCommandBuffer> cb=[gQ commandBuffer];
            id<MTLComputeCommandEncoder> ce=[cb computeCommandEncoder];
            [ce setComputePipelineState:pWarm]; [ce setBuffer:bWarm offset:0 atIndex:0];
            uint32_t it=20000; [ce setBytes:&it length:4 atIndex:1];
            [ce dispatchThreads:MTLSizeMake(1<<20,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [ce endEncoding];[cb commit];[cb waitUntilCompleted];
        }
    };

    // ---- reliable residency: N passes only if it passes ALL trials ----
    id<MTLBuffer> bArr=[gDev newBufferWithLength:64 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bSeen=[gDev newBufferWithLength:8192*4 options:MTLResourceStorageModeShared];
    auto residency=[&](uint32_t N,uint32_t tg)->bool{
        memset(bArr.contents,0,64); memset(bSeen.contents,0,8192*4);
        uint32_t sc=4000000u;
        id<MTLCommandBuffer> cb=[gQ commandBuffer];
        id<MTLComputeCommandEncoder> ce=[cb computeCommandEncoder];
        [ce setComputePipelineState:pRes];
        [ce setBuffer:bArr offset:0 atIndex:0];[ce setBuffer:bSeen offset:0 atIndex:1];
        [ce setBytes:&N length:4 atIndex:2];[ce setBytes:&sc length:4 atIndex:3];
        [ce dispatchThreadgroups:MTLSizeMake(N,1,1) threadsPerThreadgroup:MTLSizeMake(tg,1,1)];
        [ce endEncoding];[cb commit];[cb waitUntilCompleted];
        if(cb.error) return false;
        uint32_t* s=(uint32_t*)bSeen.contents; for(uint32_t i=0;i<N;++i) if(s[i]<N) return false;
        return true;
    };
    printf("[warming GPU 2000 ms]\n"); burn(2000);
    printf("\n=== RELIABLE CO-RESIDENCY (must pass 5/5 trials) ===\n");
    for (uint32_t tg : {32u,64u,128u,256u,512u,1024u}) {
        uint32_t best=0;
        for (uint32_t N=1;N<=2048;++N) {
            bool all=true; for(int t=0;t<5 && all;++t) all = residency(N,tg);
            if(all) best=N; else break;
        }
        printf("  tgsize %4u -> %4u TGs reliably co-resident (%6u threads, %5u simdgroups)\n",
               tg,best,best*tg,best*tg/32);
    }

    // ---- barrier correctness + cost ----
    printf("\n=== TICKET BARRIER: CORRECTNESS + COST ===\n");
    id<MTLBuffer> bCount=[gDev newBufferWithLength:64 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bFlags=[gDev newBufferWithLength:8192*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bPay  =[gDev newBufferWithLength:8192*4 options:MTLResourceStorageModeShared];
    for (uint32_t tg : {64u,256u}) {
      for (uint32_t numTG : {2u,4u,8u,12u,16u,24u,32u}) {
        memset(bCount.contents,0,64);memset(bFlags.contents,0,8192*4);memset(bPay.contents,0,8192*4);
        uint32_t rounds=5000, sc=8000000u;
        burn(100);
        auto t0=std::chrono::steady_clock::now();
        id<MTLCommandBuffer> cb=[gQ commandBuffer];
        id<MTLComputeCommandEncoder> ce=[cb computeCommandEncoder];
        [ce setComputePipelineState:pBar];
        [ce setBuffer:bCount offset:0 atIndex:0];[ce setBuffer:bFlags offset:0 atIndex:1];
        [ce setBuffer:bPay offset:0 atIndex:2];
        [ce setBytes:&numTG length:4 atIndex:3];[ce setBytes:&rounds length:4 atIndex:4];
        [ce setBytes:&sc length:4 atIndex:5];
        [ce dispatchThreadgroups:MTLSizeMake(numTG,1,1) threadsPerThreadgroup:MTLSizeMake(tg,1,1)];
        [ce endEncoding];[cb commit];[cb waitUntilCompleted];
        double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();
        uint32_t* f=(uint32_t*)bFlags.contents; uint32_t fails=0,corrupt=0;
        for(uint32_t i=0;i<numTG;++i){fails+=f[i*2];corrupt+=f[i*2+1];}
        const char* st = cb.error?"CB-ERROR":(fails?"TIMEOUT":(corrupt?"INCOHERENT":"OK"));
        printf("  tgsize %4u TGs %3u : %-10s %8.2f ms  %8.0f ns/barrier  (fails=%u corrupt=%u)\n",
               tg,numTG,st,ms,(ms*1e6)/(rounds*2),fails,corrupt);
        if(fails) break;
      }
      printf("\n");
    }

    // ---- baseline: one dispatch per step (Metal's own barrier) ----
    printf("=== BASELINE: ONE DISPATCH PER TIMESTEP (serial encoder) ===\n");
    for (uint32_t numTG : {8u, 16u, 32u}) {
      for (uint32_t perCB : {1u, 100u, 1000u}) {
        uint32_t steps = 5000; burn(100);
        auto t0=std::chrono::steady_clock::now();
        uint32_t done=0;
        while (done < steps) {
            uint32_t n = std::min(perCB, steps-done);
            id<MTLCommandBuffer> cb=[gQ commandBuffer];
            id<MTLComputeCommandEncoder> ce=[cb computeCommandEncoder];   // serial dispatch type
            [ce setComputePipelineState:pStep];
            [ce setBuffer:bPay offset:0 atIndex:0];
            for (uint32_t i=0;i<n;++i){ uint32_t r=done+i; [ce setBytes:&r length:4 atIndex:1];
                [ce dispatchThreadgroups:MTLSizeMake(numTG,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; }
            [ce endEncoding];[cb commit];[cb waitUntilCompleted];
            done += n;
        }
        double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();
        printf("  TGs %3u  %4u dispatch/cmdbuf : %8.2f ms  %8.0f ns/step\n",
               numTG, perCB, ms, (ms*1e6)/steps);
      }
      printf("\n");
    }
    printf("=== DONE ===\n");
}
    return 0;
}

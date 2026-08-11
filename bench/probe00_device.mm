// probe00_device.mm — Does runtime Metal shader compilation even work without Xcode?
// And what is the real co-residency limit for a persistent megakernel on M1 Pro?
//
// clang++ -std=c++17 -fobjc-arc -O2 probe00_device.mm -framework Metal -framework Foundation -o probe00

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <cstdio>
#include <vector>

static const char* kSrc = R"MSL(
#include <metal_stdlib>
using namespace metal;

// ---- test 1: simd_ballot exists and is 32 lanes wide on Apple GPU ----
kernel void ballot_probe(device uint*   out      [[buffer(0)]],
                         device float*  vals     [[buffer(1)]],
                         uint gid [[thread_position_in_grid]],
                         uint lane [[thread_index_in_simdgroup]],
                         uint sgid [[simdgroup_index_in_threadgroup]])
{
    bool fired = vals[gid] > 0.5f;
    simd_vote v = simd_ballot(fired);
    uint mask = (uint)((simd_vote::vote_t)v);   // low 32 bits = one bit per lane
    if (lane == 0) out[gid / 32] = mask;
}

// ---- test 2: how many threadgroups are actually co-resident? ----
// Each TG announces arrival, then spins waiting for all N to arrive.
// Bounded spin so we can never hang the GPU.
kernel void residency_probe(device atomic_uint* arrive  [[buffer(0)]],
                            device uint*        saw_all [[buffer(1)]],
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
        saw_all[tgid] = (seen >= N) ? 1u : 0u;
    }
}

// ---- test 3: raw dependent-load latency / bandwidth feel ----
kernel void touch(device uint* buf [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
    buf[gid] = buf[gid] * 1664525u + 1013904223u;
}
)MSL";

int main() {
@autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) { printf("NO METAL DEVICE\n"); return 1; }

    printf("=== DEVICE ===\n");
    printf("name                        : %s\n", dev.name.UTF8String);
    printf("registryID                  : %llu\n", dev.registryID);
    printf("unified memory              : %d\n", (int)dev.hasUnifiedMemory);
    printf("recommendedMaxWorkingSetSize: %.2f GB\n", dev.recommendedMaxWorkingSetSize/1073741824.0);
    printf("maxBufferLength             : %.2f GB\n", dev.maxBufferLength/1073741824.0);
    printf("maxThreadgroupMemoryLength  : %lu bytes\n", (unsigned long)dev.maxThreadgroupMemoryLength);
    printf("maxThreadsPerThreadgroup    : %lu x %lu x %lu\n",
           (unsigned long)dev.maxThreadsPerThreadgroup.width,
           (unsigned long)dev.maxThreadsPerThreadgroup.height,
           (unsigned long)dev.maxThreadsPerThreadgroup.depth);
    printf("argumentBuffersSupport      : %ld\n", (long)dev.argumentBuffersSupport);
    for (int f = 1001; f <= 1009; ++f)
        if ([dev supportsFamily:(MTLGPUFamily)f]) printf("supports Apple%d\n", f - 1000);
    if ([dev supportsFamily:MTLGPUFamilyMetal3]) printf("supports Metal3\n");

    printf("\n=== RUNTIME SHADER COMPILE (no Xcode present) ===\n");
    NSError* err = nil;
    MTLCompileOptions* opts = [MTLCompileOptions new];
    opts.fastMathEnabled = YES;
    id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kSrc]
                                           options:opts error:&err];
    if (!lib) { printf("COMPILE FAILED: %s\n", err.localizedDescription.UTF8String); return 2; }
    printf("runtime compile: OK  (functions: %s)\n",
           [[lib.functionNames componentsJoinedByString:@", "] UTF8String]);

    id<MTLCommandQueue> q = [dev newCommandQueue];

    auto mkPipe = [&](NSString* n) -> id<MTLComputePipelineState> {
        NSError* e = nil;
        id<MTLFunction> fn = [lib newFunctionWithName:n];
        id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:fn error:&e];
        if (!p) { printf("pipeline %s failed: %s\n", n.UTF8String, e.localizedDescription.UTF8String); exit(3);}
        return p;
    };

    // ---------- ballot ----------
    id<MTLComputePipelineState> pBallot = mkPipe(@"ballot_probe");
    printf("\n=== BALLOT PIPELINE ===\n");
    printf("threadExecutionWidth        : %lu   <-- SIMD width\n", (unsigned long)pBallot.threadExecutionWidth);
    printf("maxTotalThreadsPerThreadgroup: %lu\n", (unsigned long)pBallot.maxTotalThreadsPerThreadgroup);
    printf("staticThreadgroupMemoryLength: %lu\n", (unsigned long)pBallot.staticThreadgroupMemoryLength);

    const uint32_t Nn = 1024;
    id<MTLBuffer> bVals = [dev newBufferWithLength:Nn*sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bOut  = [dev newBufferWithLength:(Nn/32)*sizeof(uint32_t) options:MTLResourceStorageModeShared];
    float* vp = (float*)bVals.contents;
    // deterministic pattern: lane i fires iff bit i of 0xA5A5A5A5 pattern
    for (uint32_t i = 0; i < Nn; ++i) vp[i] = ((0xA5A5A5A5u >> (i%32)) & 1u) ? 1.0f : 0.0f;
    {
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
        [ce setComputePipelineState:pBallot];
        [ce setBuffer:bOut offset:0 atIndex:0];
        [ce setBuffer:bVals offset:0 atIndex:1];
        [ce dispatchThreads:MTLSizeMake(Nn,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [ce endEncoding]; [cb commit]; [cb waitUntilCompleted];
    }
    uint32_t* op = (uint32_t*)bOut.contents;
    printf("ballot[0] = 0x%08X   (expect 0xA5A5A5A5)  %s\n", op[0],
           op[0]==0xA5A5A5A5u ? "PASS" : "FAIL");
    printf("ballot[7] = 0x%08X\n", op[7]);

    // ---------- residency ----------
    printf("\n=== CO-RESIDENT THREADGROUP LIMIT (persistent megakernel budget) ===\n");
    id<MTLComputePipelineState> pRes = mkPipe(@"residency_probe");
    const uint32_t TGSIZE = 256;
    id<MTLBuffer> bArrive = [dev newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bSaw    = [dev newBufferWithLength:4096*sizeof(uint32_t) options:MTLResourceStorageModeShared];

    auto tryN = [&](uint32_t N)->bool{
        memset(bArrive.contents, 0, sizeof(uint32_t));
        memset(bSaw.contents, 0, 4096*sizeof(uint32_t));
        uint32_t spinCap = 2000000u;
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
        [ce setComputePipelineState:pRes];
        [ce setBuffer:bArrive offset:0 atIndex:0];
        [ce setBuffer:bSaw offset:0 atIndex:1];
        [ce setBytes:&N length:4 atIndex:2];
        [ce setBytes:&spinCap length:4 atIndex:3];
        [ce dispatchThreadgroups:MTLSizeMake(N,1,1) threadsPerThreadgroup:MTLSizeMake(TGSIZE,1,1)];
        [ce endEncoding]; [cb commit]; [cb waitUntilCompleted];
        if (cb.error) { printf("  N=%4u  CB ERROR: %s\n", N, cb.error.localizedDescription.UTF8String); return false; }
        uint32_t* s = (uint32_t*)bSaw.contents; uint32_t ok=0;
        for (uint32_t i=0;i<N && i<4096;++i) ok += s[i];
        return ok == ((N<4096)?N:4096);
    };

    uint32_t best = 0;
    for (uint32_t N : {1u,2u,4u,8u,16u,24u,32u,48u,64u,96u,128u,192u,256u,384u,512u,768u,1024u}) {
        bool ok = tryN(N);
        printf("  TGs=%4u (tgsize %u) all-arrived: %s\n", N, TGSIZE, ok?"YES":"no");
        if (ok) best = N; else break;
    }
    printf("=> max co-resident threadgroups @%u threads: %u  (%u threads total)\n",
           TGSIZE, best, best*TGSIZE);

    printf("\n=== DONE ===\n");
}
    return 0;
}

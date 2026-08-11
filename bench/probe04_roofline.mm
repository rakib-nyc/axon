// probe04_roofline.mm — the memory roofline that decides network scale.
//
// M1 Pro claims ~200 GB/s. What do we actually get for:
//   1. pure sequential streaming read           (roofline ceiling)
//   2. random gather at varying BLOCK GRANULARITY  <-- tests the central claim
//      that 32x32-block structural sparsity beats unstructured sparsity
//   3. read + write (state round-trip, which per-timestep dispatch forces on us)

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <cstdio>
#include <chrono>
#include <vector>

static const char* kSrc = R"MSL(
#include <metal_stdlib>
using namespace metal;

kernel void warmup(device float* b [[buffer(0)]], constant uint& n [[buffer(1)]],
                   uint g [[thread_position_in_grid]])
{ float x=b[g]; for(uint i=0;i<n;++i) x=fma(x,1.0000001f,0.0000001f); b[g]=x; }

inline uint hash32(uint x){
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16; return x;
}

// Sequential streaming read: consecutive threads read consecutive uint4s.
kernel void seq_read(device const uint4* buf [[buffer(0)]],
                     device uint4* out       [[buffer(1)]],
                     constant uint& nVec     [[buffer(2)]],
                     constant uint& iters    [[buffer(3)]],
                     constant uint& gridSz   [[buffer(4)]],
                     uint gid [[thread_position_in_grid]])
{
    uint4 acc = uint4(0);
    for (uint i = 0; i < iters; ++i) {
        uint idx = gid + i * gridSz;
        if (idx < nVec) acc ^= buf[idx];
    }
    if (acc.x == 0xDEADBEEFu) out[gid] = acc;   // never true; defeats DCE
}

// Random gather at block granularity `blkVec` (in uint4 = 16-byte units).
// Each thread picks a random block and reads it contiguously.
kernel void gather(device const uint4* buf [[buffer(0)]],
                   device uint4* out       [[buffer(1)]],
                   constant uint& nBlocks  [[buffer(2)]],
                   constant uint& blkVec   [[buffer(3)]],
                   constant uint& iters    [[buffer(4)]],
                   uint gid [[thread_position_in_grid]])
{
    uint4 acc = uint4(0);
    for (uint i = 0; i < iters; ++i) {
        uint blk = hash32(gid * 2654435761u + i) % nBlocks;
        uint base = blk * blkVec;
        for (uint j = 0; j < blkVec; ++j) acc ^= buf[base + j];
    }
    if (acc.x == 0xDEADBEEFu) out[gid] = acc;
}

// Same, but the whole SIMDGROUP cooperates on one block (coalesced within block).
// This is how a real synaptic fan-out read should be issued.
kernel void gather_simd(device const uint4* buf [[buffer(0)]],
                        device uint4* out       [[buffer(1)]],
                        constant uint& nBlocks  [[buffer(2)]],
                        constant uint& blkVec   [[buffer(3)]],
                        constant uint& iters    [[buffer(4)]],
                        uint gid  [[thread_position_in_grid]],
                        uint lane [[thread_index_in_simdgroup]])
{
    uint4 acc = uint4(0);
    uint sg = gid / 32;
    for (uint i = 0; i < iters; ++i) {
        uint blk = hash32(sg * 2654435761u + i) % nBlocks;
        uint base = blk * blkVec;
        for (uint j = lane; j < blkVec; j += 32) acc ^= buf[base + j];
    }
    if (acc.x == 0xDEADBEEFu) out[gid] = acc;
}

// State round-trip: read a uint4, transform, write it back. What per-timestep
// dispatch forces on neuron state.
kernel void rw(device uint4* buf [[buffer(0)]], constant uint& nVec [[buffer(1)]],
               uint gid [[thread_position_in_grid]])
{ if (gid < nVec) { uint4 v = buf[gid]; buf[gid] = v * 1664525u + 1013904223u; } }
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
    id<MTLComputePipelineState> pW=P(@"warmup"), pSeq=P(@"seq_read"),
        pGat=P(@"gather"), pGatS=P(@"gather_simd"), pRW=P(@"rw");

    const size_t BUF = 512ull<<20;            // 512 MB
    const uint32_t nVec = (uint32_t)(BUF/16); // uint4 count
    id<MTLBuffer> buf=[D newBufferWithLength:BUF options:MTLResourceStorageModePrivate];
    id<MTLBuffer> out=[D newBufferWithLength:1<<22 options:MTLResourceStorageModePrivate];
    id<MTLBuffer> bW =[D newBufferWithLength:1<<22 options:MTLResourceStorageModePrivate];
    printf("buffer: %zu MB (%u uint4)\n", BUF>>20, nVec);

    auto burn=[&](int ms){auto t0=std::chrono::steady_clock::now();
        while(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-t0).count()<ms){
            id<MTLCommandBuffer> cb=[Q commandBuffer];id<MTLComputeCommandEncoder> ce=[cb computeCommandEncoder];
            [ce setComputePipelineState:pW];[ce setBuffer:bW offset:0 atIndex:0];
            uint32_t it=20000;[ce setBytes:&it length:4 atIndex:1];
            [ce dispatchThreads:MTLSizeMake(1<<20,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [ce endEncoding];[cb commit];[cb waitUntilCompleted];}};

    // time a dispatch, best of `reps`
    auto timeIt=[&](id<MTLComputePipelineState> p, uint32_t grid, uint32_t tg,
                    std::vector<std::pair<int,uint32_t>> args, int reps)->double{
        double best=1e30;
        for(int r=0;r<reps;++r){
            auto t0=std::chrono::steady_clock::now();
            id<MTLCommandBuffer> cb=[Q commandBuffer];
            id<MTLComputeCommandEncoder> ce=[cb computeCommandEncoder];
            [ce setComputePipelineState:p];
            [ce setBuffer:buf offset:0 atIndex:0];
            [ce setBuffer:out offset:0 atIndex:1];
            for(auto& a: args){ uint32_t v=a.second; [ce setBytes:&v length:4 atIndex:a.first]; }
            [ce dispatchThreads:MTLSizeMake(grid,1,1) threadsPerThreadgroup:MTLSizeMake(tg,1,1)];
            [ce endEncoding];[cb commit];[cb waitUntilCompleted];
            double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();
            if(ms<best) best=ms;
        }
        return best;
    };

    printf("[warm 2000ms]\n"); burn(2000);

    // ---------- 1. sequential streaming read ----------
    printf("\n=== SEQUENTIAL STREAMING READ (roofline ceiling) ===\n");
    for (uint32_t gridSz : {1u<<18, 1u<<20, 1u<<22}) {
        uint32_t iters = nVec / gridSz;
        double ms = timeIt(pSeq, gridSz, 256, {{2,nVec},{3,iters},{4,gridSz}}, 5);
        double gb = (double)gridSz*iters*16.0/1e9;
        printf("  grid %8u x %4u iters : %7.3f ms  %8.1f GB/s\n", gridSz, iters, ms, gb/(ms/1e3));
    }

    // ---------- 2. random gather vs block granularity ----------
    printf("\n=== RANDOM GATHER vs BLOCK GRANULARITY (thread-per-block) ===\n");
    printf("  (this is the test of 'block-structured sparsity beats unstructured')\n");
    printf("  blkBytes    GB/s   blocks/s     vs-seq\n");
    double seqRef = 0;
    {
        uint32_t gridSz=1u<<20, iters=nVec/gridSz;
        double ms=timeIt(pSeq,gridSz,256,{{2,nVec},{3,iters},{4,gridSz}},5);
        seqRef=((double)gridSz*iters*16.0/1e9)/(ms/1e3);
    }
    for (uint32_t blkVec : {1u,2u,4u,8u,16u,32u,64u,128u}) {
        uint32_t nBlocks = nVec / blkVec;
        uint32_t gridSz  = 1u<<18;
        uint32_t iters   = 64;
        double ms = timeIt(pGat, gridSz, 256, {{2,nBlocks},{3,blkVec},{4,iters}}, 5);
        double bytes = (double)gridSz*iters*blkVec*16.0;
        double gbs = (bytes/1e9)/(ms/1e3);
        printf("  %8u  %6.1f   %9.2fM   %5.1f%%\n",
               blkVec*16, gbs, (double)gridSz*iters/(ms/1e3)/1e6, 100.0*gbs/seqRef);
    }

    printf("\n=== RANDOM GATHER, SIMDGROUP-COOPERATIVE (32 lanes per block) ===\n");
    printf("  blkBytes    GB/s     vs-seq\n");
    for (uint32_t blkVec : {32u,64u,128u,256u}) {
        uint32_t nBlocks = nVec / blkVec;
        uint32_t gridSz  = 1u<<20;
        uint32_t iters   = 16;
        double ms = timeIt(pGatS, gridSz, 256, {{2,nBlocks},{3,blkVec},{4,iters}}, 5);
        double bytes = (double)(gridSz/32)*iters*blkVec*16.0;
        double gbs = (bytes/1e9)/(ms/1e3);
        printf("  %8u  %6.1f     %5.1f%%\n", blkVec*16, gbs, 100.0*gbs/seqRef);
    }

    // ---------- 3. state round-trip ----------
    printf("\n=== READ+WRITE ROUND-TRIP (neuron state per timestep) ===\n");
    for (size_t mb : {1ull, 4ull, 16ull, 64ull, 256ull}) {
        uint32_t n = (uint32_t)((mb<<20)/16);
        double best=1e30;
        for(int r=0;r<7;++r){
            auto t0=std::chrono::steady_clock::now();
            id<MTLCommandBuffer> cb=[Q commandBuffer];
            id<MTLComputeCommandEncoder> ce=[cb computeCommandEncoder];
            [ce setComputePipelineState:pRW];[ce setBuffer:buf offset:0 atIndex:0];
            [ce setBytes:&n length:4 atIndex:1];
            [ce dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [ce endEncoding];[cb commit];[cb waitUntilCompleted];
            double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();
            if(ms<best)best=ms;
        }
        printf("  %4llu MB state : %7.3f ms  %8.1f GB/s (r+w)  -> %8.0f steps/s if this were the only cost\n",
               mb, best, (2.0*(mb<<20)/1e9)/(best/1e3), 1000.0/best);
    }
    printf("\n=== DONE ===\n");
}
return 0;}

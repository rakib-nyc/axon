// probe06_columnar.mm — the central experiment of the project.
//
// Same neuron count, same fan-in, SAME NUMBER OF SPIKES.
// Only difference: are the active neurons scattered at random, or clustered
// into columns? If the columnar thesis is right, clustering should buy a large
// speedup for free, because whole 512 B weight tiles become skippable.
//
// Also sweeps slot granularity (presyn neurons covered per skip-decision) with
// fan-in held constant at 8192, and uses a REAL spike buffer rather than a hash.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <string>
#include <vector>
#include <random>
#include <algorithm>

// ---- shader source, specialized by string substitution (free: runtime compile) ----
static std::string mkSrc(uint32_t SLOT_U4, uint32_t S) {
    char buf[256];
    snprintf(buf, sizeof buf, "#define SLOT_U4 %uu\n#define NSLOT %uu\n", SLOT_U4, S);
    return std::string(buf) + R"MSL(
#include <metal_stdlib>
using namespace metal;

kernel void warmup(device float* b [[buffer(0)]], constant uint& n [[buffer(1)]],
                   uint g [[thread_position_in_grid]])
{ float x=b[g]; for(uint i=0;i<n;++i) x=fma(x,1.0000001f,0.0000001f); b[g]=x; }

inline uint hash32(uint x){ x^=x>>16; x*=0x7feb352du; x^=x>>15; x*=0x846ca68bu; x^=x>>16; return x; }

kernel void fill_w(device uint4* w [[buffer(0)]], uint g [[thread_position_in_grid]])
{ w[g] = uint4(hash32(g*4u),hash32(g*4u+1),hash32(g*4u+2),hash32(g*4u+3)); }

kernel void fill_conn(device uint* c [[buffer(0)]], constant uint& nGrp [[buffer(1)]],
                      uint g [[thread_position_in_grid]])
{ c[g] = hash32(g*2246822519u) % nGrp; }

inline uint kwta_ballot(uint v, uint lane, uint k){
    uint rank = 0;
    for (uint j = 0; j < 32; ++j) {
        uint ov = simd_shuffle(v, j);
        rank += ((ov > v) || (ov == v && j < lane)) ? 1u : 0u;
    }
    return (uint)((simd_vote::vote_t)simd_ballot(rank < k));
}

// One fused timestep: propagate -> adapt -> column gate -> within-column k-WTA -> ballot out.
kernel void step(device const uint4* W        [[buffer(0)]],
                 device const uint*  conn     [[buffer(1)]],
                 device const uint4* spkIn    [[buffer(2)]],
                 device uint*        spkOut   [[buffer(3)]],
                 device uint*        colDrive [[buffer(4)]],
                 constant uint&      k        [[buffer(5)]],
                 constant uint&      theta    [[buffer(6)]],
                 uint gid  [[thread_position_in_grid]],
                 uint lane [[thread_index_in_simdgroup]])
{
    uint col  = gid >> 5;
    uint base = col * NSLOT;
    uint acc  = 0;
    for (uint s = 0; s < NSLOT; ++s) {
        uint g = conn[base + s];
        uint4 sp[SLOT_U4];
        uint any = 0;
        for (uint j = 0; j < SLOT_U4; ++j) {
            sp[j] = spkIn[g * SLOT_U4 + j];
            any |= sp[j].x | sp[j].y | sp[j].z | sp[j].w;
        }
        if (any != 0u) {                                  // simdgroup-uniform skip
            uint wbase = (base + s) * SLOT_U4 * 32u + lane;
            for (uint j = 0; j < SLOT_U4; ++j) {
                uint4 w  = W[wbase + j * 32u];
                uint4 pc = popcount(w & sp[j]);
                acc += pc.x + pc.y + pc.z + pc.w;
            }
        }
    }
    uint drive = simd_sum(acc);                            // column drive
    uint b = kwta_ballot(acc, lane, k);
    if (drive < theta) b = 0u;                             // columnar gate
    if (lane == 0) { spkOut[col] = b; colDrive[col] = drive; }
}
)MSL";
}

static id<MTLDevice> D; static id<MTLCommandQueue> Q;
static id<MTLComputePipelineState> P(id<MTLLibrary> L, NSString* n){NSError*e=nil;
    id<MTLComputePipelineState> p=[D newComputePipelineStateWithFunction:[L newFunctionWithName:n] error:&e];
    if(!p){printf("pipe %s: %s\n",n.UTF8String,e.localizedDescription.UTF8String);exit(3);} return p;}

int main(){
@autoreleasepool{
    D=MTLCreateSystemDefaultDevice(); Q=[D newCommandQueue];

    const uint32_t NCOL = 4096, N = NCOL*32, FANIN = 8192;
    printf("=== AXON columnar experiment ===\n");
    printf("  %u neurons (%u columns x 32), fan-in %u, %.2f G binary synapses, W = %.0f MB\n\n",
           N, NCOL, FANIN, (double)N*FANIN/1e9, (double)N*FANIN/8/1048576.0);

    // shared buffers sized for the largest config
    const size_t WBYTES = (size_t)N * FANIN / 8;
    id<MTLBuffer> bW   =[D newBufferWithLength:WBYTES options:MTLResourceStorageModePrivate];
    id<MTLBuffer> bConn=[D newBufferWithLength:NCOL*64*4 options:MTLResourceStorageModePrivate];
    id<MTLBuffer> bIn  =[D newBufferWithLength:N/8 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bOut =[D newBufferWithLength:NCOL*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bDrv =[D newBufferWithLength:NCOL*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bWarm=[D newBufferWithLength:1<<22 options:MTLResourceStorageModePrivate];

    struct Cfg { uint32_t slotU4, S; };
    std::vector<Cfg> cfgs = {{1,64},{2,32},{4,16},{8,8}};

    // ---- build a spike pattern with EXACTLY `nspk` spikes, clustered or random ----
    auto makeSpikes=[&](uint32_t nspk, bool clustered, uint32_t seed){
        uint32_t* w = (uint32_t*)bIn.contents;
        memset(w, 0, N/8);
        std::mt19937 rng(seed);
        if (!clustered) {
            std::uniform_int_distribution<uint32_t> d(0, N-1);
            uint32_t placed=0;
            while (placed<nspk) { uint32_t n=d(rng);
                if(!((w[n>>5]>>(n&31))&1u)){ w[n>>5]|=1u<<(n&31); placed++; } }
        } else {
            // fill whole 32-neuron columns until the spike budget is spent
            uint32_t ncols = nspk/32;
            std::uniform_int_distribution<uint32_t> d(0, NCOL-1);
            uint32_t placed=0;
            while (placed<ncols) { uint32_t c=d(rng); if(w[c]!=0xFFFFFFFFu){ w[c]=0xFFFFFFFFu; placed++; } }
        }
    };

    // ---- run one config ----
    auto run=[&](Cfg cfg, uint32_t nspk, bool clustered, uint32_t steps)->double{
        NSError*e=nil; MTLCompileOptions*o=[MTLCompileOptions new]; o.fastMathEnabled=YES;
        std::string s = mkSrc(cfg.slotU4, cfg.S);
        id<MTLLibrary> L=[D newLibraryWithSource:[NSString stringWithUTF8String:s.c_str()] options:o error:&e];
        if(!L){printf("COMPILE FAIL: %s\n",e.localizedDescription.UTF8String);exit(2);}
        id<MTLComputePipelineState> pFW=P(L,@"fill_w"), pFC=P(L,@"fill_conn"),
                                    pStep=P(L,@"step"), pWarm=P(L,@"warmup");
        uint32_t nGrp = N / (cfg.slotU4*128);
        { id<MTLCommandBuffer> cb=[Q commandBuffer]; id<MTLComputeCommandEncoder> ce=[cb computeCommandEncoder];
          [ce setComputePipelineState:pFW];[ce setBuffer:bW offset:0 atIndex:0];
          [ce dispatchThreads:MTLSizeMake(WBYTES/16,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
          [ce setComputePipelineState:pFC];[ce setBuffer:bConn offset:0 atIndex:0];
          [ce setBytes:&nGrp length:4 atIndex:1];
          [ce dispatchThreads:MTLSizeMake(NCOL*cfg.S,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
          [ce endEncoding];[cb commit];[cb waitUntilCompleted]; }
        makeSpikes(nspk, clustered, 12345);
        // warm
        { auto t0=std::chrono::steady_clock::now();
          while(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-t0).count()<100){
            id<MTLCommandBuffer> cb=[Q commandBuffer];id<MTLComputeCommandEncoder> ce=[cb computeCommandEncoder];
            [ce setComputePipelineState:pWarm];[ce setBuffer:bWarm offset:0 atIndex:0];
            uint32_t it=20000;[ce setBytes:&it length:4 atIndex:1];
            [ce dispatchThreads:MTLSizeMake(1<<20,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [ce endEncoding];[cb commit];[cb waitUntilCompleted]; } }
        uint32_t k=4, theta=0;
        double best=1e30;
        for(int rep=0;rep<3;++rep){
            auto t0=std::chrono::steady_clock::now();
            id<MTLCommandBuffer> cb=[Q commandBuffer];
            id<MTLComputeCommandEncoder> ce=[cb computeCommandEncoder];
            [ce setComputePipelineState:pStep];
            [ce setBuffer:bW offset:0 atIndex:0];[ce setBuffer:bConn offset:0 atIndex:1];
            [ce setBuffer:bIn offset:0 atIndex:2];[ce setBuffer:bOut offset:0 atIndex:3];
            [ce setBuffer:bDrv offset:0 atIndex:4];
            [ce setBytes:&k length:4 atIndex:5];[ce setBytes:&theta length:4 atIndex:6];
            for(uint32_t st=0;st<steps;++st)
                [ce dispatchThreads:MTLSizeMake(N,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [ce endEncoding];[cb commit];[cb waitUntilCompleted];
            double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count()/steps;
            best=std::min(best,ms);
        }
        return best;
    };

    printf("=== SLOT GRANULARITY x ACTIVITY STRUCTURE (fan-in fixed at 8192) ===\n");
    printf("  presyn/slot = neurons that must ALL be silent to skip a tile\n\n");
    printf("  %-22s | %-17s | %-17s | speedup\n", "config", "RANDOM 1%", "CLUSTERED 1%");
    printf("  %-22s | %8s %8s | %8s %8s |\n", "", "ms/step", "steps/s", "ms/step", "steps/s");
    printf("  ---------------------------------------------------------------------------\n");
    uint32_t nspk = N/100;   // 1% activity, 1310 spikes, identical in both conditions
    for (auto c : cfgs) {
        double rnd = run(c, nspk, false, 300);
        double clu = run(c, nspk, true,  300);
        char nm[64]; snprintf(nm,sizeof nm,"%u slots x %u presyn", c.S, c.slotU4*128);
        printf("  %-22s | %8.4f %8.0f | %8.4f %8.0f | %5.2fx\n",
               nm, rnd, 1000.0/rnd, clu, 1000.0/clu, rnd/clu);
    }

    printf("\n=== ACTIVITY SWEEP, best config, clustered vs random ===\n");
    printf("  activity |   random ms   clustered ms   speedup\n");
    for (double pct : {0.25, 0.5, 1.0, 2.0, 5.0, 10.0}) {
        uint32_t ns = (uint32_t)(N*pct/100.0);
        double rnd = run({1,64}, ns, false, 300);
        double clu = run({1,64}, ns, true,  300);
        printf("  %6.2f%%  | %10.4f   %10.4f    %5.2fx\n", pct, rnd, clu, rnd/clu);
    }
    printf("\n=== DONE ===\n");
}
return 0;}

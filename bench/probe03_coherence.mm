// probe03_coherence.mm — Is device memory coherent ACROSS threadgroups mid-dispatch?
//
// Three variants of the same cross-threadgroup handshake:
//   A: plain device store / plain device load          (what probe02 did)
//   B: atomic_store / atomic_load  (relaxed)
//   C: atomic_store / atomic RMW read (fetch_or 0)     -- forces coherent point
// plus the barrier spin itself in load- vs RMW- form.
//
// If C works and A/B do not, the rule for the whole project is:
//   "cross-threadgroup traffic inside a dispatch must be RMW atomics"
// which is expensive enough that batched dispatches win outright.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <cstdio>
#include <chrono>
#include <algorithm>

static const char* kSrc = R"MSL(
#include <metal_stdlib>
using namespace metal;

kernel void warmup(device float* b [[buffer(0)]], constant uint& n [[buffer(1)]],
                   uint g [[thread_position_in_grid]])
{ float x=b[g]; for(uint i=0;i<n;++i) x=fma(x,1.0000001f,0.0000001f); b[g]=x; }

// spin read as plain atomic load
inline uint rd_load(device atomic_uint* p){ return atomic_load_explicit(p, memory_order_relaxed); }
// spin read forced through the coherent point with a no-op RMW
inline uint rd_rmw (device atomic_uint* p){ return atomic_fetch_or_explicit(p, 0u, memory_order_relaxed); }

inline bool barrier_load(device atomic_uint* c, threadgroup uint& ok, uint N, uint tid, uint cap){
    threadgroup_barrier(mem_flags::mem_device|mem_flags::mem_threadgroup);
    if(tid==0){ uint t=atomic_fetch_add_explicit(c,1u,memory_order_relaxed); uint tgt=(t/N+1u)*N; bool o=false;
        for(uint i=0;i<cap;++i){ if(rd_load(c)>=tgt){o=true;break;} } ok=o?1u:0u; }
    threadgroup_barrier(mem_flags::mem_device|mem_flags::mem_threadgroup);
    return ok!=0u;
}
inline bool barrier_rmw(device atomic_uint* c, threadgroup uint& ok, uint N, uint tid, uint cap){
    threadgroup_barrier(mem_flags::mem_device|mem_flags::mem_threadgroup);
    if(tid==0){ uint t=atomic_fetch_add_explicit(c,1u,memory_order_relaxed); uint tgt=(t/N+1u)*N; bool o=false;
        for(uint i=0;i<cap;++i){ if(rd_rmw(c)>=tgt){o=true;break;} } ok=o?1u:0u; }
    threadgroup_barrier(mem_flags::mem_device|mem_flags::mem_threadgroup);
    return ok!=0u;
}

// variant: 0=plain/plain 1=atomic store+load 2=atomic store+RMW read
// spinMode: 0=load 1=rmw
kernel void coherence(device atomic_uint* count   [[buffer(0)]],
                      device uint*        payload [[buffer(1)]],
                      device atomic_uint* apay    [[buffer(2)]],
                      device uint*        flags   [[buffer(3)]],
                      constant uint&      N       [[buffer(4)]],
                      constant uint&      rounds  [[buffer(5)]],
                      constant uint&      cap     [[buffer(6)]],
                      constant uint&      variant [[buffer(7)]],
                      constant uint&      spinMode[[buffer(8)]],
                      uint tgid [[threadgroup_position_in_grid]],
                      uint tid  [[thread_position_in_threadgroup]])
{
    threadgroup uint ok;
    uint fails=0, corrupt=0, r=0;
    for(r=0;r<rounds;++r){
        uint v=r+1;
        if(tid==0){
            if(variant==0) payload[tgid]=v;
            else           atomic_store_explicit(&apay[tgid], v, memory_order_relaxed);
        }
        bool o = (spinMode==0) ? barrier_load(count,ok,N,tid,cap) : barrier_rmw(count,ok,N,tid,cap);
        if(!o){fails++;break;}
        if(tid==0){
            for(uint j=0;j<N;++j){
                uint got;
                if(variant==0)      got = payload[j];
                else if(variant==1) got = atomic_load_explicit(&apay[j], memory_order_relaxed);
                else                got = atomic_fetch_or_explicit(&apay[j], 0u, memory_order_relaxed);
                if(got!=v) corrupt++;
            }
        }
        o = (spinMode==0) ? barrier_load(count,ok,N,tid,cap) : barrier_rmw(count,ok,N,tid,cap);
        if(!o){fails++;break;}
    }
    if(tid==0){ flags[tgid*3+0]=fails; flags[tgid*3+1]=corrupt; flags[tgid*3+2]=r; }
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
    id<MTLComputePipelineState> pW=P(@"warmup"), pC=P(@"coherence");

    id<MTLBuffer> bW=[D newBufferWithLength:1<<22 options:MTLResourceStorageModePrivate];
    auto burn=[&](int ms){auto t0=std::chrono::steady_clock::now();
        while(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-t0).count()<ms){
            id<MTLCommandBuffer> cb=[Q commandBuffer]; id<MTLComputeCommandEncoder> ce=[cb computeCommandEncoder];
            [ce setComputePipelineState:pW];[ce setBuffer:bW offset:0 atIndex:0];
            uint32_t it=20000;[ce setBytes:&it length:4 atIndex:1];
            [ce dispatchThreads:MTLSizeMake(1<<20,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [ce endEncoding];[cb commit];[cb waitUntilCompleted];}};

    id<MTLBuffer> bC=[D newBufferWithLength:64 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bP=[D newBufferWithLength:8192*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bA=[D newBufferWithLength:8192*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bF=[D newBufferWithLength:8192*4 options:MTLResourceStorageModeShared];

    const char* vname[3]={"plain store / plain load","atomic store / atomic load","atomic store / RMW read"};
    const char* sname[2]={"spin=load","spin=RMW "};
    printf("[warm 1500ms]\n"); burn(1500);
    printf("\n=== CROSS-THREADGROUP COHERENCE INSIDE ONE DISPATCH ===\n");
    printf("    (rounds=2000, tgsize=256)\n\n");
    for(uint32_t spinMode=0; spinMode<2; ++spinMode){
      for(uint32_t variant=0; variant<3; ++variant){
        for(uint32_t N : {2u,4u,8u}){
            memset(bC.contents,0,64);memset(bP.contents,0,8192*4);
            memset(bA.contents,0,8192*4);memset(bF.contents,0,8192*4);
            uint32_t rounds=2000, cap=2000000u;
            burn(80);
            auto t0=std::chrono::steady_clock::now();
            id<MTLCommandBuffer> cb=[Q commandBuffer];
            id<MTLComputeCommandEncoder> ce=[cb computeCommandEncoder];
            [ce setComputePipelineState:pC];
            [ce setBuffer:bC offset:0 atIndex:0];[ce setBuffer:bP offset:0 atIndex:1];
            [ce setBuffer:bA offset:0 atIndex:2];[ce setBuffer:bF offset:0 atIndex:3];
            [ce setBytes:&N length:4 atIndex:4];[ce setBytes:&rounds length:4 atIndex:5];
            [ce setBytes:&cap length:4 atIndex:6];[ce setBytes:&variant length:4 atIndex:7];
            [ce setBytes:&spinMode length:4 atIndex:8];
            [ce dispatchThreadgroups:MTLSizeMake(N,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [ce endEncoding];[cb commit];[cb waitUntilCompleted];
            double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();
            uint32_t* f=(uint32_t*)bF.contents; uint32_t fails=0,corrupt=0,minr=0xffffffff;
            for(uint32_t i=0;i<N;++i){fails+=f[i*3];corrupt+=f[i*3+1];minr=std::min(minr,f[i*3+2]);}
            const char* st=cb.error?"CB-ERR":(fails?"DEADLOCK":(corrupt?"INCOHERENT":"OK"));
            printf("  %s  %-28s N=%u : %-10s rounds_done=%4u corrupt=%6u  %7.1f ns/barrier\n",
                   sname[spinMode], vname[variant], N, st, minr, corrupt,
                   fails?0.0:(ms*1e6)/(rounds*2));
        }
        printf("\n");
      }
    }
    printf("=== DONE ===\n");
}
return 0;}

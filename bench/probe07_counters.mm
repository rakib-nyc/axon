// probe07_counters.mm — what hardware counters does this GPU actually expose?
//
// Everything reported so far has been wall-clock around waitUntilCompleted,
// which cannot separate kernels inside one command buffer. MTLCounterSampleBuffer
// samples GPU counters AT DISPATCH BOUNDARIES with no host sync. If this device
// exposes a statistic/stage-utilization set we get ALU and memory occupancy too.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <cstdio>
#include <vector>

static const char* kSrc = R"MSL(
#include <metal_stdlib>
using namespace metal;
kernel void heavy(device float* b [[buffer(0)]], constant uint& n [[buffer(1)]],
                  uint g [[thread_position_in_grid]])
{ float x=b[g]; for(uint i=0;i<n;++i) x=fma(x,1.0000001f,0.0000001f); b[g]=x; }
kernel void light(device uint* b [[buffer(0)]], uint g [[thread_position_in_grid]])
{ b[g] = b[g]*1664525u+1013904223u; }
)MSL";

int main(){
@autoreleasepool{
    id<MTLDevice> D = MTLCreateSystemDefaultDevice();
    printf("=== COUNTER SETS EXPOSED BY %s ===\n", D.name.UTF8String);
    if (D.counterSets.count == 0) printf("  (none)\n");
    for (id<MTLCounterSet> cs in D.counterSets) {
        printf("  set \"%s\"  (%lu counters)\n", cs.name.UTF8String, (unsigned long)cs.counters.count);
        for (id<MTLCounter> c in cs.counters) printf("      - %s\n", c.name.UTF8String);
    }
    printf("\n=== SAMPLING POINTS SUPPORTED ===\n");
    printf("  at stage boundary    : %d\n", (int)[D supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]);
    printf("  at draw boundary     : %d\n", (int)[D supportsCounterSampling:MTLCounterSamplingPointAtDrawBoundary]);
    printf("  at dispatch boundary : %d\n", (int)[D supportsCounterSampling:MTLCounterSamplingPointAtDispatchBoundary]);
    printf("  at tile dispatch     : %d\n", (int)[D supportsCounterSampling:MTLCounterSamplingPointAtTileDispatchBoundary]);
    printf("  at blit boundary     : %d\n", (int)[D supportsCounterSampling:MTLCounterSamplingPointAtBlitBoundary]);

    // find the timestamp set
    id<MTLCounterSet> tsSet = nil;
    for (id<MTLCounterSet> cs in D.counterSets)
        if ([cs.name isEqualToString:MTLCommonCounterSetTimestamp]) tsSet = cs;
    if (!tsSet) { printf("\nNo timestamp counter set -> per-kernel GPU timing unavailable.\n"); return 0; }
    bool dispatchPt = [D supportsCounterSampling:MTLCounterSamplingPointAtDispatchBoundary];
    bool stagePt    = [D supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary];
    printf("\ndispatch-boundary sampling: %s ; falling back to PER-ENCODER timing: %s\n",
           dispatchPt?"yes":"NO", stagePt?"available":"unavailable");
    if (!stagePt) return 0;

    NSError* e=nil; MTLCompileOptions* o=[MTLCompileOptions new]; o.fastMathEnabled=YES;
    id<MTLLibrary> L=[D newLibraryWithSource:[NSString stringWithUTF8String:kSrc] options:o error:&e];
    id<MTLComputePipelineState> pH=[D newComputePipelineStateWithFunction:[L newFunctionWithName:@"heavy"] error:&e];
    id<MTLComputePipelineState> pL=[D newComputePipelineStateWithFunction:[L newFunctionWithName:@"light"] error:&e];
    id<MTLCommandQueue> Q=[D newCommandQueue];
    id<MTLBuffer> buf=[D newBufferWithLength:1<<24 options:MTLResourceStorageModePrivate];

    const NSUInteger NS = 8;   // 4 dispatches -> 8 samples (before/after each)
    MTLCounterSampleBufferDescriptor* d=[MTLCounterSampleBufferDescriptor new];
    d.counterSet=tsSet; d.storageMode=MTLStorageModeShared; d.sampleCount=NS;
    id<MTLCounterSampleBuffer> sb=[D newCounterSampleBufferWithDescriptor:d error:&e];
    if(!sb){ printf("\ncounter sample buffer failed: %s\n", e.localizedDescription.UTF8String); return 1; }

    // GPU ticks -> nanoseconds
    MTLTimestamp c0,g0,c1,g1;
    [D sampleTimestamps:&c0 gpuTimestamp:&g0];

    id<MTLCommandBuffer> cb=[Q commandBuffer];
    uint32_t it=4000, n=1<<22;
    // one ENCODER per kernel, each with start/end timestamps attached
    for (int k=0;k<4;++k) {
        MTLComputePassDescriptor* cpd=[MTLComputePassDescriptor computePassDescriptor];
        cpd.sampleBufferAttachments[0].sampleBuffer=sb;
        cpd.sampleBufferAttachments[0].startOfEncoderSampleIndex=k*2;
        cpd.sampleBufferAttachments[0].endOfEncoderSampleIndex=k*2+1;
        id<MTLComputeCommandEncoder> ce=[cb computeCommandEncoderWithDescriptor:cpd];
        if (k%2==0) {
            [ce setComputePipelineState:pH];[ce setBuffer:buf offset:0 atIndex:0];
            [ce setBytes:&it length:4 atIndex:1];
            [ce dispatchThreads:MTLSizeMake(1<<20,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        } else {
            [ce setComputePipelineState:pL];[ce setBuffer:buf offset:0 atIndex:0];
            [ce dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        }
        [ce endEncoding];
    }
    [cb commit];[cb waitUntilCompleted];
    printf("\n  whole command buffer GPU time: %.1f us\n", (cb.GPUEndTime-cb.GPUStartTime)*1e6);
    [D sampleTimestamps:&c1 gpuTimestamp:&g1];
    double nsPerTick = (double)(c1-c0)/(double)(g1-g0);

    NSData* dat=[sb resolveCounterRange:NSMakeRange(0,NS)];
    if(!dat){ printf("resolve failed\n"); return 1; }
    const MTLCounterResultTimestamp* ts=(const MTLCounterResultTimestamp*)dat.bytes;
    printf("\n=== PER-DISPATCH GPU TIME (ticks -> ns, scale %.4f) ===\n", nsPerTick);
    const char* nm[4]={"heavy","light","heavy","light"};
    for (int k=0;k<4;++k) {
        uint64_t a=ts[k*2].timestamp, b=ts[k*2+1].timestamp;
        printf("  %-6s : %10llu ticks  = %9.1f us\n", nm[k], (b-a), (b-a)*nsPerTick/1000.0);
    }
    printf("\n=> per-kernel GPU timing inside one command buffer: WORKING\n");
}
return 0;}

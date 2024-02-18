//go:build darwin
#include "gpu_info_darwin.h"

uint64_t getRecommendedMaxVRAM()
{
	id<MTLDevice> device = MTLCreateSystemDefaultDevice();
	NSLog(@"Metal default device %s", [device.name UTF8String]);
	NSLog(@"Recommended max VRAM in byte: %lld", device.recommendedMaxWorkingSetSize);
	return device.recommendedMaxWorkingSetSize;
}


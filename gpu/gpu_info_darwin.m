//go:build darwin
#include "gpu_info_darwin.h"

uint64_t getRecommendedMaxVRAM()
{
	id<MTLDevice> device = MTLCreateSystemDefaultDevice();
	uint64_t result = device.recommendedMaxWorkingSetSize;
	CFRelease(device);
	return result;
}


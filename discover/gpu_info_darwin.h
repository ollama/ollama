#import <Metal/Metal.h>
#include <stdint.h>
uint64_t getRecommendedMaxVRAM();
uint64_t getPhysicalMemory();
uint64_t getFreeMemory();
int getMemoryHeadroom(uint64_t *available, uint64_t *immediate, uint64_t *compressed);

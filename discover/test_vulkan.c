//go:build debug

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gpu_info_vulkan.h"

int main(int argc, char *argv[]) {
    printf("=== Vulkan GPU Info Test ===\n");

    // Default library paths - adjust these based on your system
    char *vk_lib_path = "libvulkan.so.1";

    // Allow command line override of library paths
    if (argc >= 2) {
        vk_lib_path = argv[1];
    }

    printf("Using Vulkan library: %s\n", vk_lib_path);
    printf("\n");

    // Initialize Vulkan
    vk_init_resp_t init_resp = {0};
    init_resp.ch.verbose = 1; // Enable verbose logging

    printf("Initializing Vulkan...\n");
    vk_init(vk_lib_path, &init_resp);

    if (init_resp.err) {
        printf("Error initializing Vulkan: %s\n", init_resp.err);
        free(init_resp.err);
        return 1;
    }

    printf("Successfully initialized Vulkan!\n");
    printf("Number of devices found: %d\n\n", init_resp.num_devices);

    if (init_resp.num_devices == 0) {
        printf("No Vulkan devices found.\n");
        vk_release(init_resp.ch);
        return 0;
    }

    // Check each device
    for (int i = 0; i < init_resp.num_devices; i++) {
        printf("=== Device %d ===\n", i);

        // Check VRAM information
        mem_info_t mem_info = {0};
        vk_check_vram(init_resp.ch, i, &mem_info);

        if (mem_info.err) {
            printf("Error checking device %d VRAM: %s\n", i, mem_info.err);
            free(mem_info.err);
        } else {
            printf("Device ID: %s\n", mem_info.gpu_id);
            printf("Device Name: %s\n", mem_info.gpu_name);
            printf("Total VRAM: %lu bytes (%.2f GB)\n",
                   mem_info.total, (double)mem_info.total / (1024.0 * 1024.0 * 1024.0));
            printf("Free VRAM: %lu bytes (%.2f GB)\n",
                   mem_info.free, (double)mem_info.free / (1024.0 * 1024.0 * 1024.0));
            printf("API Version: %d.%d.%d\n", mem_info.major, mem_info.minor, mem_info.patch);
        }

        printf("\n");
    }

    // Cleanup
    printf("Releasing Vulkan resources...\n");
    vk_release(init_resp.ch);

    printf("Test completed successfully!\n");
    return 0;
}

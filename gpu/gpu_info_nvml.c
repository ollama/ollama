#ifndef __APPLE__  // TODO - maybe consider nvidia support on intel macs?

#include "gpu_info_nvml.h"

void nvml_init(char *nvml_lib_path, nvml_init_resp_t *resp) {
    resp->err = NULL;
    resp->num_devices = 0;
    const int buflen = 256;
    char buf[buflen + 1];
    int i;

    struct lookup {
        char *s;
        void **p;
    } l[] = {
        {"nvmlInit", (void *)&resp->nh.nvmlInit},
        {"nvmlShutdown", (void *)&resp->nh.nvmlShutdown},
        {"nvmlDeviceGetCount", (void *)&resp->nh.nvmlDeviceGetCount},
        {"nvmlDeviceGetHandleByIndex", (void *)&resp->nh.nvmlDeviceGetHandleByIndex},
        {"nvmlDeviceGetMemoryInfo", (void *)&resp->nh.nvmlDeviceGetMemoryInfo},
        {"nvmlDeviceGetCudaComputeCapability", (void *)&resp->nh.nvmlDeviceGetCudaComputeCapability},
        {"nvmlErrorString", (void *)&resp->nh.nvmlErrorString},
        {NULL, NULL},
    };

    resp->nh.handle = LOAD_LIBRARY(nvml_lib_path, RTLD_LAZY);
    if (!resp->nh.handle) {
        char *msg = LOAD_ERR();
        snprintf(buf, buflen, "Unable to load %s library to query for Nvidia GPUs: %s", nvml_lib_path, msg);
        resp->err = strdup(buf);
        return;
    }

    for (i = 0; l[i].s != NULL; i++) {
        *l[i].p = LOAD_SYMBOL(resp->nh.handle, l[i].s);
        if (!*l[i].p) {
            char *msg = LOAD_ERR();
            UNLOAD_LIBRARY(resp->nh.handle);
            resp->nh.handle = NULL;
            snprintf(buf, buflen, "symbol lookup for %s failed: %s", l[i].s, msg);
            resp->err = strdup(buf);
            return;
        }
    }

    nvmlReturn_t ret = (*resp->nh.nvmlInit)();
    if (ret != NVML_SUCCESS) {
        snprintf(buf, buflen, "nvml init failure: %s", (*resp->nh.nvmlErrorString)(ret));
        resp->err = strdup(buf);
        return;
    }

    ret = (*resp->nh.nvmlDeviceGetCount)(&resp->num_devices);
    if (ret != NVML_SUCCESS) {
        snprintf(buf, buflen, "unable to get device count: %s", (*resp->nh.nvmlErrorString)(ret));
        resp->err = strdup(buf);
        return;
    }
}

void nvml_check_vram(nvml_handle_t nh, int device_id, mem_info_t *resp) {
    resp->err = NULL;
    nvmlMemory_t memInfo;
    nvmlReturn_t ret;
    const int buflen = 256;
    char buf[buflen + 1];

    if (nh.handle == NULL) {
        resp->err = strdup("nvml handle isn't initialized");
        return;
    }

    nvmlDevice_t device;
    ret = (*nh.nvmlDeviceGetHandleByIndex)(device_id, &device);
    if (ret != NVML_SUCCESS) {
        snprintf(buf, buflen, "nvml device handle by index failed: %s", (*nh.nvmlErrorString)(ret));
        resp->err = strdup(buf);
        return;
    }

    ret = (*nh.nvmlDeviceGetMemoryInfo)(device, &memInfo);
    if (ret != NVML_SUCCESS) {
        snprintf(buf, buflen, "nvml device memory info lookup failure: %s", (*nh.nvmlErrorString)(ret));
        resp->err = strdup(buf);
        return;
    }

    resp->total = memInfo.total;
    resp->free = memInfo.free;

    int major, minor;
    ret = (*nh.nvmlDeviceGetCudaComputeCapability)(device, &resp->major, &resp->minor);
    if (ret != NVML_SUCCESS) {
        snprintf(buf, buflen, "nvml device compute capability lookup failure: %s", (*nh.nvmlErrorString)(ret));
        resp->err = strdup(buf);
        return;
    }

    printf("[Device %u] NVML totalMem %llu\n", device_id, resp->total);
    printf("[Device %u] NVML freeMem %llu\n", device_id, resp->free);
    printf("[Device %u] NVML major %d\n", device_id, resp->major);
    printf("[Device %u] NVML minor %d\n", device_id, resp->minor);
}

void nvml_release(nvml_handle_t nh) {
    if (nh.handle != NULL) {
        (*nh.nvmlShutdown)();
        UNLOAD_LIBRARY(nh.handle);
        nh.handle = NULL;
    }
}
#endif  // __APPLE__
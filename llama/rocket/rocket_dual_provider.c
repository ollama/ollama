// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * rocket_dual_provider.c — Unified runtime-dispatching submit provider for
 * librocketnpu supporting both:
 *   1. Mainline Linux DRM-accel `rocket` driver (/dev/accel/accel0)
 *   2. Rockchip BSP DRM `rknpu` driver >= 0.9.6 (/dev/dri/renderD128..143)
 *
 * This allows a single libggml-rocket.so build to run on both mainline kernels
 * (6.18+) and vendor Rockchip BSP kernels (Linux 5.10/6.1 with RKNPU >= 0.9.6).
 */
#include <errno.h>
#include <fcntl.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <libdrm/drm.h>
#include <drm/rocket_accel.h>

#include "rocket_npu.h"
#include "rocket_log.h"
#include "rocket_sysfs.h"
#include "rocket_busy_poll.h"
#include "rocket_fence_watch.h"
#include "rocket_rk3576_internal.h"
#include "rocket_hw_profile.h"
#include "rknpu_uapi.h"
#include "rknpu_submit.h"

/* -------------------------------------------------------------------------
 * 1. Mainline `rocket` submit provider (prefixed as rkt_mainline_*)
 * ------------------------------------------------------------------------- */
#define rocket_open                      rkt_mainline_open
#define rocket_close                     rkt_mainline_close
#define rocket_bo_alloc                  rkt_mainline_bo_alloc
#define rocket_bo_alloc32                rkt_mainline_bo_alloc32
#define rocket_bo_ensure32               rkt_mainline_bo_ensure32
#define rocket_bo_free                   rkt_mainline_bo_free
#define rocket_bo_prep                   rkt_mainline_bo_prep
#define rocket_bo_fini                   rkt_mainline_bo_fini
#define rocket_bo_prep_ranges            rkt_mainline_bo_prep_ranges
#define rocket_bo_fini_ranges            rkt_mainline_bo_fini_ranges
#define rocket_bo_ranges_supported       rkt_mainline_bo_ranges_supported
#define rocket_batched_submit_supported  rkt_mainline_batched_submit_supported
#define rocket_batched_submit_native     rkt_mainline_batched_submit_native
#define rocket_ppu_done_supported        rkt_mainline_ppu_done_supported
#define rocket_no_dpu_done_supported     rkt_mainline_no_dpu_done_supported
#define rocket_batch_completion_tracked  rkt_mainline_batch_completion_tracked
#define rocket_submit_batch_atomic       rkt_mainline_submit_batch_atomic
#define rocket_submit_matmul             rkt_mainline_submit_matmul
#define rocket_submit_matmul_flags       rkt_mainline_submit_matmul_flags
#define rocket_submit_tasks              rkt_mainline_submit_tasks
#define rocket_submit_tasks_flags        rkt_mainline_submit_tasks_flags
#define rocket_submit_scratch_size       rkt_mainline_submit_scratch_size
#define rocket_submit_tasks_pre          rkt_mainline_submit_tasks_pre
#define rocket_submit_jobs               rkt_mainline_submit_jobs
#define rocket_submit_ioctl_count        rkt_mainline_submit_ioctl_count
#define rocket_submit_task_count         rkt_mainline_submit_task_count
#define rocket_submit_counters_reset     rkt_mainline_submit_counters_reset

int rkt_mainline_bo_alloc(int fd, size_t size, rocket_bo *out);
int rkt_mainline_bo_alloc32(int fd, size_t size, rocket_bo *out);
int rkt_mainline_bo_ensure32(int fd, rocket_bo *bo, size_t need);
void rkt_mainline_bo_free(int fd, rocket_bo *bo);
int rkt_mainline_bo_prep(int fd, rocket_bo *bo, int dir, uint64_t timeout_ns);
int rkt_mainline_bo_fini(int fd, rocket_bo *bo);
int rkt_mainline_bo_ranges_supported(void);

#include "rocket_npu.c"

#undef rocket_open
#undef rocket_close
#undef rocket_bo_alloc
#undef rocket_bo_alloc32
#undef rocket_bo_ensure32
#undef rocket_bo_free
#undef rocket_bo_prep
#undef rocket_bo_fini
#undef rocket_bo_prep_ranges
#undef rocket_bo_fini_ranges
#undef rocket_bo_ranges_supported
#undef rocket_batched_submit_supported
#undef rocket_batched_submit_native
#undef rocket_ppu_done_supported
#undef rocket_no_dpu_done_supported
#undef rocket_batch_completion_tracked
#undef rocket_submit_batch_atomic
#undef rocket_submit_matmul
#undef rocket_submit_matmul_flags
#undef rocket_submit_tasks
#undef rocket_submit_tasks_flags
#undef rocket_submit_scratch_size
#undef rocket_submit_tasks_pre
#undef rocket_submit_jobs
#undef rocket_submit_ioctl_count
#undef rocket_submit_task_count
#undef rocket_submit_counters_reset

/* -------------------------------------------------------------------------
 * 2. Vendor `rknpu` submit provider (prefixed as rkt_rknpu_*)
 * ------------------------------------------------------------------------- */
#define rocket_open                      rkt_rknpu_open
#define rocket_close                     rkt_rknpu_close
#define rocket_bo_alloc                  rkt_rknpu_bo_alloc
#define rocket_bo_alloc32                rkt_rknpu_bo_alloc32
#define rocket_bo_ensure32               rkt_rknpu_bo_ensure32
#define rocket_bo_free                   rkt_rknpu_bo_free
#define rocket_bo_prep                   rkt_rknpu_bo_prep
#define rocket_bo_fini                   rkt_rknpu_bo_fini
#define rocket_bo_prep_ranges            rkt_rknpu_bo_prep_ranges
#define rocket_bo_fini_ranges            rkt_rknpu_bo_fini_ranges
#define rocket_bo_ranges_supported       rkt_rknpu_bo_ranges_supported
#define rocket_batched_submit_supported  rkt_rknpu_batched_submit_supported
#define rocket_batched_submit_native     rkt_rknpu_batched_submit_native
#define rocket_ppu_done_supported        rkt_rknpu_ppu_done_supported
#define rocket_no_dpu_done_supported     rkt_rknpu_no_dpu_done_supported
#define rocket_batch_completion_tracked  rkt_rknpu_batch_completion_tracked
#define rocket_submit_batch_atomic       rkt_rknpu_submit_batch_atomic
#define rocket_submit_matmul             rkt_rknpu_submit_matmul
#define rocket_submit_matmul_flags       rkt_rknpu_submit_matmul_flags
#define rocket_submit_tasks              rkt_rknpu_submit_tasks
#define rocket_submit_tasks_flags        rkt_rknpu_submit_tasks_flags
#define rocket_submit_scratch_size       rkt_rknpu_submit_scratch_size
#define rocket_submit_tasks_pre          rkt_rknpu_submit_tasks_pre
#define rocket_submit_jobs               rkt_rknpu_submit_jobs
#define rocket_submit_ioctl_count        rkt_rknpu_submit_ioctl_count
#define rocket_submit_task_count         rkt_rknpu_submit_task_count
#define rocket_submit_counters_reset     rkt_rknpu_submit_counters_reset

int rkt_rknpu_bo_alloc(int fd, size_t size, rocket_bo *out);
int rkt_rknpu_bo_alloc32(int fd, size_t size, rocket_bo *out);
int rkt_rknpu_bo_ensure32(int fd, rocket_bo *bo, size_t need);
void rkt_rknpu_bo_free(int fd, rocket_bo *bo);
int rkt_rknpu_bo_prep(int fd, rocket_bo *bo, int dir, uint64_t timeout_ns);
int rkt_rknpu_bo_fini(int fd, rocket_bo *bo);
int rkt_rknpu_bo_ranges_supported(void);

#include "rknpu_provider.c"

#undef rocket_open
#undef rocket_close
#undef rocket_bo_alloc
#undef rocket_bo_alloc32
#undef rocket_bo_ensure32
#undef rocket_bo_free
#undef rocket_bo_prep
#undef rocket_bo_fini
#undef rocket_bo_prep_ranges
#undef rocket_bo_fini_ranges
#undef rocket_bo_ranges_supported
#undef rocket_batched_submit_supported
#undef rocket_batched_submit_native
#undef rocket_ppu_done_supported
#undef rocket_no_dpu_done_supported
#undef rocket_batch_completion_tracked
#undef rocket_submit_batch_atomic
#undef rocket_submit_matmul
#undef rocket_submit_matmul_flags
#undef rocket_submit_tasks
#undef rocket_submit_tasks_flags
#undef rocket_submit_scratch_size
#undef rocket_submit_tasks_pre
#undef rocket_submit_jobs
#undef rocket_submit_ioctl_count
#undef rocket_submit_task_count
#undef rocket_submit_counters_reset

/* -------------------------------------------------------------------------
 * 3. Runtime provider detection & dispatch
 * ------------------------------------------------------------------------- */
enum {
    RKT_PROV_UNRESOLVED = -1,
    RKT_PROV_NONE       =  0,
    RKT_PROV_MAINLINE   =  1,
    RKT_PROV_RKNPU      =  2,
};

static _Atomic int g_active_provider = RKT_PROV_UNRESOLVED;

static int rkt_probe_drm_node(const char *path, char *out_name, size_t out_len, uint32_t *out_rknpu_ver)
{
    int fd = open(path, O_RDWR | O_CLOEXEC);
    if (fd < 0) {
        return -errno;
    }
    memset(out_name, 0, out_len);
    struct drm_version dv;
    memset(&dv, 0, sizeof(dv));
    dv.name = out_name;
    dv.name_len = out_len - 1;
    if (ioctl(fd, DRM_IOCTL_VERSION, &dv) < 0) {
        int e = errno;
        close(fd);
        return -e;
    }
    if (out_rknpu_ver) {
        *out_rknpu_ver = 0;
        if (strcmp(out_name, "rknpu") == 0) {
            struct rknpu_action act;
            memset(&act, 0, sizeof(act));
            act.flags = RKNPU_GET_DRV_VERSION;
            if (ioctl(fd, ioc(RKNPU_ACTION_NR, sizeof(act)), &act) == 0) {
                *out_rknpu_ver = act.value;
            }
        }
    }
    close(fd);
    return 0;
}

/*
 * Quietly probes whether a supported Rockchip NPU kernel driver is accessible.
 * Returns:
 *   RKT_PROV_MAINLINE (1) when mainline DRM-accel `rocket` is present
 *   RKT_PROV_RKNPU    (2) when vendor DRM `rknpu` (>= 0.9.6) is present
 *   RKT_PROV_NONE     (0) when neither is available
 */
int rocket_npu_probe(void)
{
    int p = atomic_load_explicit(&g_active_provider, memory_order_relaxed);
    if (p != RKT_PROV_UNRESOLVED) {
        return p;
    }

    const char *forced_drv = getenv("ROCKET_DRIVER");
    int allow_mainline = 1;
    int allow_rknpu = 1;
    if (forced_drv && *forced_drv) {
        if (strcmp(forced_drv, "rocket") == 0 || strcmp(forced_drv, "mainline") == 0) {
            allow_rknpu = 0;
        } else if (strcmp(forced_drv, "rknpu") == 0 || strcmp(forced_drv, "vendor") == 0 || strcmp(forced_drv, "bsp") == 0) {
            allow_mainline = 0;
        }
    }

    const char *assume = getenv("ROCKET_ASSUME_PRESENT");
    if (assume && *assume && strcmp(assume, "0") != 0) {
        if (!allow_mainline || strcmp(assume, "rknpu") == 0) {
            p = RKT_PROV_RKNPU;
        } else {
            p = RKT_PROV_MAINLINE;
        }
        atomic_store_explicit(&g_active_provider, p, memory_order_relaxed);
        return p;
    }

    char name[64];
    uint32_t rknpu_ver = 0;
    const char *dev = getenv("ROCKET_DEV");
    if (dev && *dev) {
        if (rkt_probe_drm_node(dev, name, sizeof(name), &rknpu_ver) == 0) {
            if (allow_mainline && strcmp(name, "rocket") == 0) {
                p = RKT_PROV_MAINLINE;
            } else if (allow_rknpu && strcmp(name, "rknpu") == 0 &&
                       (rknpu_ver == 0 || rknpu_ver >= RKNPU_DRV_VERSION_MIN)) {
                p = RKT_PROV_RKNPU;
            } else {
                p = RKT_PROV_NONE;
            }
        } else {
            p = RKT_PROV_NONE;
        }
        atomic_store_explicit(&g_active_provider, p, memory_order_relaxed);
        return p;
    }

    if (allow_mainline) {
        int rc = rkt_probe_drm_node("/dev/accel/accel0", name, sizeof(name), NULL);
        if (rc == 0 && strcmp(name, "rocket") == 0) {
            p = RKT_PROV_MAINLINE;
            atomic_store_explicit(&g_active_provider, p, memory_order_relaxed);
            return p;
        }
        if (rc == -EACCES || rc == -EPERM) {
            ROCKET_LOGE("rocket: /dev/accel/accel0 is present but permission was denied (%s); "
                        "add the user to the 'render' group or grant device access\n",
                        strerror(-rc));
        }
    }

    if (allow_rknpu) {
        for (int n = RKNPU_RENDER_FIRST; n <= RKNPU_RENDER_LAST; n++) {
            char path[64];
            snprintf(path, sizeof(path), "/dev/dri/renderD%d", n);
            int rc = rkt_probe_drm_node(path, name, sizeof(name), &rknpu_ver);
            if (rc == 0 && strcmp(name, "rknpu") == 0) {
                if (rknpu_ver == 0 || rknpu_ver >= RKNPU_DRV_VERSION_MIN) {
                    p = RKT_PROV_RKNPU;
                    atomic_store_explicit(&g_active_provider, p, memory_order_relaxed);
                    return p;
                }
            }
            if (rc == -EACCES || rc == -EPERM) {
                char sysfs_drv[128], link_target[256];
                snprintf(sysfs_drv, sizeof(sysfs_drv), "/sys/class/drm/renderD%d/device/driver", n);
                ssize_t l = readlink(sysfs_drv, link_target, sizeof(link_target) - 1);
                if (l > 0) {
                    link_target[l] = '\0';
                    if (strstr(link_target, "RKNPU") != NULL || strstr(link_target, "rknpu") != NULL) {
                        ROCKET_LOGE("rocket: %s (RKNPU) is present but permission was denied (%s); "
                                    "add the user to the 'render' group or grant device access\n",
                                    path, strerror(-rc));
                    }
                }
            }
        }
    }

    p = RKT_PROV_NONE;
    atomic_store_explicit(&g_active_provider, p, memory_order_relaxed);
    return p;
}

static inline int use_rknpu(void)
{
    return rocket_npu_probe() == RKT_PROV_RKNPU;
}

const char *rocket_driver_name(void)
{
    if (use_rknpu()) {
        return "vendor rknpu driver";
    }
    return "mainline rocket driver";
}

int rocket_open(void)
{
    return use_rknpu() ? rkt_rknpu_open() : rkt_mainline_open();
}

void rocket_close(int fd)
{
    if (use_rknpu()) {
        rkt_rknpu_close(fd);
    } else {
        rkt_mainline_close(fd);
    }
}

int rocket_bo_alloc(int fd, size_t size, rocket_bo *bo)
{
    return use_rknpu() ? rkt_rknpu_bo_alloc(fd, size, bo)
                       : rkt_mainline_bo_alloc(fd, size, bo);
}

int rocket_bo_alloc32(int fd, size_t size, rocket_bo *bo)
{
    return use_rknpu() ? rkt_rknpu_bo_alloc32(fd, size, bo)
                       : rkt_mainline_bo_alloc32(fd, size, bo);
}

int rocket_bo_ensure32(int fd, rocket_bo *bo, size_t need)
{
    return use_rknpu() ? rkt_rknpu_bo_ensure32(fd, bo, need)
                       : rkt_mainline_bo_ensure32(fd, bo, need);
}

void rocket_bo_free(int fd, rocket_bo *bo)
{
    if (use_rknpu()) {
        rkt_rknpu_bo_free(fd, bo);
    } else {
        rkt_mainline_bo_free(fd, bo);
    }
}

int rocket_bo_prep(int fd, rocket_bo *bo, int dir, uint64_t timeout_ns)
{
    return use_rknpu() ? rkt_rknpu_bo_prep(fd, bo, dir, timeout_ns)
                       : rkt_mainline_bo_prep(fd, bo, dir, timeout_ns);
}

int rocket_bo_fini(int fd, rocket_bo *bo)
{
    return use_rknpu() ? rkt_rknpu_bo_fini(fd, bo)
                       : rkt_mainline_bo_fini(fd, bo);
}

int rocket_bo_prep_ranges(int fd, rocket_bo *bo, const rocket_bo_range *r,
                          unsigned n, uint64_t timeout_ns)
{
    return use_rknpu() ? rkt_rknpu_bo_prep_ranges(fd, bo, r, n, timeout_ns)
                       : rkt_mainline_bo_prep_ranges(fd, bo, r, n, timeout_ns);
}

int rocket_bo_fini_ranges(int fd, rocket_bo *bo, const rocket_bo_range *r,
                          unsigned n)
{
    return use_rknpu() ? rkt_rknpu_bo_fini_ranges(fd, bo, r, n)
                       : rkt_mainline_bo_fini_ranges(fd, bo, r, n);
}

int rocket_bo_ranges_supported(void)
{
    return use_rknpu() ? rkt_rknpu_bo_ranges_supported()
                       : rkt_mainline_bo_ranges_supported();
}

int rocket_batched_submit_supported(void)
{
    return use_rknpu() ? rkt_rknpu_batched_submit_supported()
                       : rkt_mainline_batched_submit_supported();
}

int rocket_batched_submit_native(void)
{
    return use_rknpu() ? rkt_rknpu_batched_submit_native()
                       : rkt_mainline_batched_submit_native();
}

int rocket_ppu_done_supported(void)
{
    return use_rknpu() ? rkt_rknpu_ppu_done_supported()
                       : rkt_mainline_ppu_done_supported();
}

int rocket_no_dpu_done_supported(void)
{
    return use_rknpu() ? rkt_rknpu_no_dpu_done_supported()
                       : rkt_mainline_no_dpu_done_supported();
}

int rocket_batch_completion_tracked(void)
{
    return use_rknpu() ? rkt_rknpu_batch_completion_tracked()
                       : rkt_mainline_batch_completion_tracked();
}

int rocket_submit_batch_atomic(void)
{
    return use_rknpu() ? rkt_rknpu_submit_batch_atomic()
                       : rkt_mainline_submit_batch_atomic();
}

int rocket_submit_matmul(int fd,
                         const rocket_bo *regcmd_bo, uint32_t regcmd_count,
                         const uint32_t *in_handles, uint32_t n_in,
                         const uint32_t *out_handles, uint32_t n_out,
                         uint32_t timeout_ms)
{
    return use_rknpu() ? rkt_rknpu_submit_matmul(fd, regcmd_bo, regcmd_count, in_handles, n_in, out_handles, n_out, timeout_ms)
                       : rkt_mainline_submit_matmul(fd, regcmd_bo, regcmd_count, in_handles, n_in, out_handles, n_out, timeout_ms);
}

int rocket_submit_matmul_flags(int fd,
                               const rocket_bo *regcmd_bo, uint32_t regcmd_count,
                               const uint32_t *in_handles, uint32_t n_in,
                               const uint32_t *out_handles, uint32_t n_out,
                               uint32_t job_flags)
{
    return use_rknpu() ? rkt_rknpu_submit_matmul_flags(fd, regcmd_bo, regcmd_count, in_handles, n_in, out_handles, n_out, job_flags)
                       : rkt_mainline_submit_matmul_flags(fd, regcmd_bo, regcmd_count, in_handles, n_in, out_handles, n_out, job_flags);
}

int rocket_submit_tasks(int fd,
                        const rocket_task_desc *tasks, uint32_t n_tasks,
                        const uint32_t *in_handles, uint32_t n_in,
                        const uint32_t *out_handles, uint32_t n_out)
{
    return use_rknpu() ? rkt_rknpu_submit_tasks(fd, tasks, n_tasks, in_handles, n_in, out_handles, n_out)
                       : rkt_mainline_submit_tasks(fd, tasks, n_tasks, in_handles, n_in, out_handles, n_out);
}

int rocket_submit_tasks_flags(int fd,
                              const rocket_task_desc *tasks, uint32_t n_tasks,
                              const uint32_t *in_handles, uint32_t n_in,
                              const uint32_t *out_handles, uint32_t n_out,
                              uint32_t job_flags)
{
    return use_rknpu() ? rkt_rknpu_submit_tasks_flags(fd, tasks, n_tasks, in_handles, n_in, out_handles, n_out, job_flags)
                       : rkt_mainline_submit_tasks_flags(fd, tasks, n_tasks, in_handles, n_in, out_handles, n_out, job_flags);
}

size_t rocket_submit_scratch_size(uint32_t max_tasks)
{
    return use_rknpu() ? rkt_rknpu_submit_scratch_size(max_tasks)
                       : rkt_mainline_submit_scratch_size(max_tasks);
}

int rocket_submit_tasks_pre(int fd, void *scratch,
                            const rocket_task_desc *tasks, uint32_t n_tasks,
                            const uint32_t *in_handles, uint32_t n_in,
                            const uint32_t *out_handles, uint32_t n_out,
                            uint32_t job_flags)
{
    return use_rknpu() ? rkt_rknpu_submit_tasks_pre(fd, scratch, tasks, n_tasks, in_handles, n_in, out_handles, n_out, job_flags)
                       : rkt_mainline_submit_tasks_pre(fd, scratch, tasks, n_tasks, in_handles, n_in, out_handles, n_out, job_flags);
}

int rocket_submit_jobs(int fd, const rocket_job_desc *jobs, uint32_t n_jobs)
{
    return use_rknpu() ? rkt_rknpu_submit_jobs(fd, jobs, n_jobs)
                       : rkt_mainline_submit_jobs(fd, jobs, n_jobs);
}

uint64_t rocket_submit_ioctl_count(void)
{
    return rkt_mainline_submit_ioctl_count() + rkt_rknpu_submit_ioctl_count();
}

uint64_t rocket_submit_task_count(void)
{
    return rkt_mainline_submit_task_count() + rkt_rknpu_submit_task_count();
}

void rocket_submit_counters_reset(void)
{
    rkt_mainline_submit_counters_reset();
    rkt_rknpu_submit_counters_reset();
}

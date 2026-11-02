/* SPDX-License-Identifier: MIT */
/*
 * Copyright © 2024 Tomeu Vizoso
 */
#ifndef __DRM_UAPI_ROCKET_ACCEL_H__
#define __DRM_UAPI_ROCKET_ACCEL_H__

#include <libdrm/drm.h>

#if defined(__cplusplus)
extern "C" {
#endif

#define DRM_ROCKET_CREATE_BO			0x00
#define DRM_ROCKET_SUBMIT			0x01
#define DRM_ROCKET_PREP_BO			0x02
#define DRM_ROCKET_FINI_BO			0x03

#define DRM_IOCTL_ROCKET_CREATE_BO		DRM_IOWR(DRM_COMMAND_BASE + DRM_ROCKET_CREATE_BO, struct drm_rocket_create_bo)
#define DRM_IOCTL_ROCKET_SUBMIT			DRM_IOW(DRM_COMMAND_BASE + DRM_ROCKET_SUBMIT, struct drm_rocket_submit)
#define DRM_IOCTL_ROCKET_PREP_BO		DRM_IOW(DRM_COMMAND_BASE + DRM_ROCKET_PREP_BO, struct drm_rocket_prep_bo)
#define DRM_IOCTL_ROCKET_FINI_BO		DRM_IOW(DRM_COMMAND_BASE + DRM_ROCKET_FINI_BO, struct drm_rocket_fini_bo)

struct drm_rocket_create_bo {
	__u32 size;
	__u32 handle;
	__u64 dma_address;
	__u64 offset;
};

struct drm_rocket_prep_bo {
	__u32 handle;
	__u32 reserved;
	__s64 timeout_ns;
};

struct drm_rocket_fini_bo {
	__u32 handle;
	__u32 reserved;
};

struct drm_rocket_task {
	__u32 regcmd;
	__u32 regcmd_count;
};

struct drm_rocket_job {
	__u64 tasks;
	__u64 in_bo_handles;
	__u64 out_bo_handles;
	__u32 task_count;
	__u32 task_struct_size;
	__u32 in_bo_handle_count;
	__u32 out_bo_handle_count;
};

struct drm_rocket_submit {
	__u64 jobs;
	__u32 job_count;
	__u32 job_struct_size;
	__u64 reserved;
};

#if defined(__cplusplus)
}
#endif

#endif /* __DRM_UAPI_ROCKET_ACCEL_H__ */

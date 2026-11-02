/* SPDX-License-Identifier: MIT */
#ifndef OLLAMA_ROCKET_LIBDRM_DRM_H
#define OLLAMA_ROCKET_LIBDRM_DRM_H

#if defined(__has_include) && __has_include(<drm/drm.h>)
#include <drm/drm.h>
#else
#include <linux/ioctl.h>
#include <linux/types.h>
#include <stddef.h>

#define DRM_IOCTL_BASE                  'd'
#define DRM_IO(nr)                      _IO(DRM_IOCTL_BASE, nr)
#define DRM_IOR(nr, type)               _IOR(DRM_IOCTL_BASE, nr, type)
#define DRM_IOW(nr, type)               _IOW(DRM_IOCTL_BASE, nr, type)
#define DRM_IOWR(nr, type)              _IOWR(DRM_IOCTL_BASE, nr, type)
#define DRM_COMMAND_BASE                0x40

struct drm_version {
    int version_major;
    int version_minor;
    int version_patchlevel;
    size_t name_len;
    char *name;
    size_t date_len;
    char *date;
    size_t desc_len;
    char *desc;
};

struct drm_gem_close {
    __u32 handle;
    __u32 pad;
};

#define DRM_IOCTL_VERSION               DRM_IOWR(0x00, struct drm_version)
#define DRM_IOCTL_GEM_CLOSE             DRM_IOW (0x09, struct drm_gem_close)
#endif

#endif /* OLLAMA_ROCKET_LIBDRM_DRM_H */

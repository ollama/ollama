// SPDX-License-Identifier: MIT
// ze_device.hpp — RAII wrapper for ze_device_handle_t.
//
// L0 device handles are NOT reference-counted by the application; they are
// owned by the driver for the lifetime of the process.  Therefore the
// destructor performs NO destroy call.  The wrapper caches the properties
// queried at construction for O(1) access without repeated L0 calls.
#pragma once

#include <cstdint>
#include <string>
#include <cstring>

// Forward-declared L0 types (resolved at runtime via dlsym — no link-time dep).
struct _ze_device_handle_t;
typedef struct _ze_device_handle_t *ze_device_handle_t;

// Capability flags resolved at construction from ze_device_properties_t and
// ze_device_compute_properties_t. Stored as plain integers so ze_device.hpp
// does not need the L0 headers to be visible to its includers.
struct ZeDeviceCaps {
    uint32_t compute_units;   // numEUsPerSubSlice * numSubSlicesPerSlice * numSlices
    uint32_t clock_mhz;       // core_clock_rate
    uint64_t total_memory;    // sum across ze_device_memory_properties_t entries
    uint8_t  supports_fp16;
    uint8_t  supports_int8;
    uint8_t  device_kind;     // 1 = GPU, 2 = NPU  (ze_ollama_device_kind_t values)
    char     name[256];
    char     uuid[37];
};

/**
 * ZeDevice — RAII wrapper for a Level Zero device handle.
 *
 * Non-copyable (device handles are not reference-counted by the application).
 * Move-constructible so it can be stored in std::vector / std::array.
 *
 * The underlying ze_device_handle_t remains valid for the lifetime of its
 * ze_driver_handle_t, so no explicit destroy is needed in the destructor.
 */
class ZeDevice {
public:
    ZeDevice() noexcept : handle_(nullptr), caps_{} {}

    /**
     * Construct from a raw L0 device handle and pre-filled capability record.
     * Called by the factory inside ze_ollama_init after querying device props.
     */
    ZeDevice(ze_device_handle_t h, const ZeDeviceCaps &caps) noexcept
        : handle_(h), caps_(caps) {}

    // Non-copyable — device handles are not reference-counted.
    ZeDevice(const ZeDevice &)            = delete;
    ZeDevice &operator=(const ZeDevice &) = delete;

    // Move-constructible for storage in containers.
    ZeDevice(ZeDevice &&o) noexcept : handle_(o.handle_), caps_(o.caps_) {
        o.handle_ = nullptr;
    }
    ZeDevice &operator=(ZeDevice &&o) noexcept {
        if (this != &o) {
            handle_   = o.handle_;
            caps_     = o.caps_;
            o.handle_ = nullptr;
        }
        return *this;
    }

    // No-op destructor — device lifetime is managed by the driver.
    ~ZeDevice() noexcept = default;

    ze_device_handle_t handle()         const noexcept { return handle_;               }
    const char        *name()           const noexcept { return caps_.name;             }
    const char        *uuid_str()       const noexcept { return caps_.uuid;             }
    uint32_t           compute_units()  const noexcept { return caps_.compute_units;    }
    uint32_t           clock_mhz()      const noexcept { return caps_.clock_mhz;        }
    uint64_t           total_memory()   const noexcept { return caps_.total_memory;     }
    uint8_t            supports_fp16()  const noexcept { return caps_.supports_fp16;    }
    uint8_t            supports_int8()  const noexcept { return caps_.supports_int8;    }
    uint8_t            device_kind()    const noexcept { return caps_.device_kind;      }
    bool               valid()          const noexcept { return handle_ != nullptr;     }

private:
    ze_device_handle_t handle_;
    ZeDeviceCaps       caps_;
};

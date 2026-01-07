#include "ggml.h"
#include "ggml-impl.h"

#ifdef _WIN32
// AMD Device Library eXtra (ADLX)
//
// https://github.com/GPUOpen-LibrariesAndSDKs/ADLX
//
// This Windows-only library provides accurate VRAM reporting for AMD GPUs.
// The runtime DLL is installed with every AMD Driver on Windows, however
// the SDK isn't a part of the HIP SDK packaging.  As such, we avoid including
// the headers from the SDK to simplify building from source.
//
// ADLX relies heavily on function pointer tables.
// Only the minimal set of types are defined below to facilitate
// finding the target AMD GPU(s) and querying their current VRAM usage
// Unused function parameters are commented out to avoid unnecessary type
// definitions.

#include <filesystem>
#include <mutex>

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#  define NOMINMAX
#endif
#include <windows.h>

namespace fs = std::filesystem;

#include <stdio.h>
#include <stdint.h>

// Begin minimal ADLX definitions - derived from tag v1.0 (Dec 2022)
typedef     uint64_t            adlx_uint64;
typedef     uint32_t            adlx_uint32;
typedef     int32_t             adlx_int32;
typedef     adlx_int32          adlx_int;
typedef     adlx_uint32         adlx_uint;
typedef     long                adlx_long;
typedef     uint8_t             adlx_uint8;
typedef enum
{
    ADLX_OK = 0,                    /**< @ENG_START_DOX This result indicates success. @ENG_END_DOX */
    ADLX_ALREADY_ENABLED,           /**< @ENG_START_DOX This result indicates that the asked action is already enabled. @ENG_END_DOX */
    ADLX_ALREADY_INITIALIZED,       /**< @ENG_START_DOX This result indicates that ADLX has a unspecified type of initialization. @ENG_END_DOX */
    ADLX_FAIL,                      /**< @ENG_START_DOX This result indicates an unspecified failure. @ENG_END_DOX */
    ADLX_INVALID_ARGS,              /**< @ENG_START_DOX This result indicates that the arguments are invalid. @ENG_END_DOX */
    ADLX_BAD_VER,                   /**< @ENG_START_DOX This result indicates that the asked version is incompatible with the current version. @ENG_END_DOX */
    ADLX_UNKNOWN_INTERFACE,         /**< @ENG_START_DOX This result indicates that an unknown interface was asked. @ENG_END_DOX */
    ADLX_TERMINATED,                /**< @ENG_START_DOX This result indicates that the calls were made in an interface after ADLX was terminated. @ENG_END_DOX */
    ADLX_ADL_INIT_ERROR,            /**< @ENG_START_DOX This result indicates that the ADL initialization failed. @ENG_END_DOX */
    ADLX_NOT_FOUND,                 /**< @ENG_START_DOX This result indicates that the item is not found. @ENG_END_DOX */
    ADLX_INVALID_OBJECT,            /**< @ENG_START_DOX This result indicates that the method was called into an invalid object. @ENG_END_DOX */
    ADLX_ORPHAN_OBJECTS,            /**< @ENG_START_DOX This result indicates that ADLX was terminated with outstanding ADLX objects. Any interface obtained from ADLX points to invalid memory and calls in their methods will result in unexpected behavior. @ENG_END_DOX */
    ADLX_NOT_SUPPORTED,             /**< @ENG_START_DOX This result indicates that the asked feature is not supported. @ENG_END_DOX */
    ADLX_PENDING_OPERATION,         /**< @ENG_START_DOX This result indicates a failure due to an operation currently in progress. @ENG_END_DOX */
    ADLX_GPU_INACTIVE               /**< @ENG_START_DOX This result indicates that the GPU is inactive. @ENG_END_DOX */
} ADLX_RESULT;
#define ADLX_SUCCEEDED(x) (ADLX_OK == (x) || ADLX_ALREADY_ENABLED == (x) || ADLX_ALREADY_INITIALIZED == (x))
#define ADLX_FAILED(x) (ADLX_OK != (x)  && ADLX_ALREADY_ENABLED != (x) && ADLX_ALREADY_INITIALIZED != (x))
#define ADLX_VER_MAJOR       1
#define ADLX_VER_MINOR       0
#define ADLX_VER_RELEASE     5
#define ADLX_VER_BUILD_NUM   30
#define ADLX_MAKE_FULL_VER(VERSION_MAJOR, VERSION_MINOR, VERSION_RELEASE, VERSION_BUILD_NUM)    ( ((adlx_uint64)(VERSION_MAJOR) << 48ull) | ((adlx_uint64)(VERSION_MINOR) << 32ull) | ((adlx_uint64)(VERSION_RELEASE) << 16ull)  | (adlx_uint64)(VERSION_BUILD_NUM))
#define ADLX_FULL_VERSION ADLX_MAKE_FULL_VER(ADLX_VER_MAJOR, ADLX_VER_MINOR, ADLX_VER_RELEASE, ADLX_VER_BUILD_NUM)
#define ADLX_CORE_LINK          __declspec(dllexport)
#define ADLX_STD_CALL           __stdcall
#define ADLX_CDECL_CALL         __cdecl
#define ADLX_FAST_CALL          __fastcall
#define ADLX_INLINE              __inline
#define ADLX_FORCEINLINE         __forceinline
#define ADLX_NO_VTABLE          __declspec(novtable)

#if defined(__cplusplus)
typedef     bool                adlx_bool;
#else
typedef     adlx_uint8           adlx_bool;
#define     true                1
#define     false               0
#endif

typedef struct IADLXSystem IADLXSystem;
typedef struct IADLXGPUList IADLXGPUList;
typedef struct IADLXGPU IADLXGPU;
typedef struct IADLXInterface IADLXInterface;
typedef struct IADLXPerformanceMonitoringServices IADLXPerformanceMonitoringServices;
typedef struct IADLXGPUMetrics IADLXGPUMetrics;
typedef struct IADLXGPUMetricsSupport IADLXGPUMetricsSupport;

typedef struct IADLXSystemVtbl
{
    // IADLXSystem interface
    ADLX_RESULT (ADLX_STD_CALL *GetHybridGraphicsType)(/* IADLXSystem* pThis, ADLX_HG_TYPE* hgType */);
    ADLX_RESULT (ADLX_STD_CALL *GetGPUs)(IADLXSystem* pThis, IADLXGPUList** ppGPUs); // Used
    ADLX_RESULT (ADLX_STD_CALL *QueryInterface)(/* IADLXSystem* pThis, const wchar_t* interfaceId, void** ppInterface */);
    ADLX_RESULT (ADLX_STD_CALL *GetDisplaysServices)(/* IADLXSystem* pThis, IADLXDisplayServices** ppDispServices */);
    ADLX_RESULT (ADLX_STD_CALL *GetDesktopsServices)(/* IADLXSystem* pThis, IADLXDesktopServices** ppDeskServices */);
    ADLX_RESULT (ADLX_STD_CALL *GetGPUsChangedHandling)(/* IADLXSystem* pThis, IADLXGPUsChangedHandling** ppGPUsChangedHandling */);
    ADLX_RESULT (ADLX_STD_CALL *EnableLog)(/* IADLXSystem* pThis, ADLX_LOG_DESTINATION mode, ADLX_LOG_SEVERITY severity, IADLXLog* pLogger, const wchar_t* fileName */);
    ADLX_RESULT (ADLX_STD_CALL *Get3DSettingsServices)(/* IADLXSystem* pThis, IADLX3DSettingsServices** pp3DSettingsServices */);
    ADLX_RESULT (ADLX_STD_CALL *GetGPUTuningServices)(/* IADLXSystem* pThis, IADLXGPUTuningServices** ppGPUTuningServices */);
    ADLX_RESULT (ADLX_STD_CALL *GetPerformanceMonitoringServices)(IADLXSystem* pThis, IADLXPerformanceMonitoringServices** ppPerformanceMonitoringServices); // Used
    ADLX_RESULT (ADLX_STD_CALL *TotalSystemRAM)(/* IADLXSystem* pThis, adlx_uint* ramMB */);
    ADLX_RESULT (ADLX_STD_CALL *GetI2C)(/* IADLXSystem* pThis, IADLXGPU* pGPU, IADLXI2C** ppI2C */);
} IADLXSystemVtbl;
struct IADLXSystem { const IADLXSystemVtbl *pVtbl; };

typedef struct IADLXGPUVtbl
{
    //IADLXInterface
    adlx_long (ADLX_STD_CALL *Acquire)(/* IADLXGPU* pThis */);
    adlx_long (ADLX_STD_CALL *Release)(IADLXGPU* pThis); // Used
    ADLX_RESULT (ADLX_STD_CALL *QueryInterface)(/* IADLXGPU* pThis, const wchar_t* interfaceId, void** ppInterface */);

    //IADLXGPU
    ADLX_RESULT (ADLX_STD_CALL *VendorId)(/* IADLXGPU* pThis, const char** vendorId */);
    ADLX_RESULT (ADLX_STD_CALL *ASICFamilyType)(/* IADLXGPU* pThis, ADLX_ASIC_FAMILY_TYPE* asicFamilyType */);
    ADLX_RESULT (ADLX_STD_CALL *Type)(/* IADLXGPU* pThis, ADLX_GPU_TYPE* gpuType */);
    ADLX_RESULT (ADLX_STD_CALL *IsExternal)(/* IADLXGPU* pThis, adlx_bool* isExternal */);
    ADLX_RESULT (ADLX_STD_CALL *Name)(/* IADLXGPU* pThis, const char** gpuName */);
    ADLX_RESULT (ADLX_STD_CALL *DriverPath)(/* IADLXGPU* pThis, const char** driverPath */);
    ADLX_RESULT (ADLX_STD_CALL *PNPString)(/* IADLXGPU* pThis, const char** pnpString */);
    ADLX_RESULT (ADLX_STD_CALL *HasDesktops)(/* IADLXGPU* pThis, adlx_bool* hasDesktops */);
    ADLX_RESULT (ADLX_STD_CALL *TotalVRAM)(IADLXGPU* pThis, adlx_uint* vramMB); // Used
    ADLX_RESULT (ADLX_STD_CALL *VRAMType)(/* IADLXGPU* pThis, const char** type */);
    ADLX_RESULT (ADLX_STD_CALL *BIOSInfo)(/* IADLXGPU* pThis, const char** partNumber, const char** version, const char** date */);
    ADLX_RESULT (ADLX_STD_CALL *DeviceId)(/* IADLXGPU* pThis, const char** deviceId */);
    ADLX_RESULT (ADLX_STD_CALL *RevisionId)(/* IADLXGPU* pThis, const char** revisionId */);
    ADLX_RESULT (ADLX_STD_CALL *SubSystemId)(/* IADLXGPU* pThis, const char** subSystemId */);
    ADLX_RESULT (ADLX_STD_CALL *SubSystemVendorId)(/* IADLXGPU* pThis, const char** subSystemVendorId */);
    ADLX_RESULT (ADLX_STD_CALL *UniqueId)(IADLXGPU* pThis, adlx_int* uniqueId); // Used
} IADLXGPUVtbl;
struct IADLXGPU { const IADLXGPUVtbl *pVtbl; };

typedef struct IADLXGPUListVtbl
{
    //IADLXInterface
    adlx_long (ADLX_STD_CALL *Acquire)(/* IADLXGPUList* pThis */);
    adlx_long (ADLX_STD_CALL *Release)(IADLXGPUList* pThis); // Used
    ADLX_RESULT (ADLX_STD_CALL *QueryInterface)(/* IADLXGPUList* pThis, const wchar_t* interfaceId, void** ppInterface */);

    //IADLXList
    adlx_uint (ADLX_STD_CALL *Size)(/* IADLXGPUList* pThis */);
    adlx_uint8 (ADLX_STD_CALL *Empty)(/* IADLXGPUList* pThis */);
    adlx_uint (ADLX_STD_CALL *Begin)(IADLXGPUList* pThis); // Used
    adlx_uint (ADLX_STD_CALL *End)(IADLXGPUList* pThis); // Used
    ADLX_RESULT (ADLX_STD_CALL *At)(/* IADLXGPUList* pThis, const adlx_uint location, IADLXInterface** ppItem */);
    ADLX_RESULT (ADLX_STD_CALL *Clear)(/* IADLXGPUList* pThis */);
    ADLX_RESULT (ADLX_STD_CALL *Remove_Back)(/* IADLXGPUList* pThis */);
    ADLX_RESULT (ADLX_STD_CALL *Add_Back)(/* IADLXGPUList* pThis, IADLXInterface* pItem */);

    //IADLXGPUList
    ADLX_RESULT (ADLX_STD_CALL *At_GPUList)(IADLXGPUList* pThis, const adlx_uint location, IADLXGPU** ppItem); // Used
    ADLX_RESULT (ADLX_STD_CALL *Add_Back_GPUList)(/* IADLXGPUList* pThis, IADLXGPU* pItem */);

} IADLXGPUListVtbl;
struct IADLXGPUList { const IADLXGPUListVtbl *pVtbl; };

typedef struct IADLXPerformanceMonitoringServicesVtbl
{
    //IADLXInterface
    adlx_long (ADLX_STD_CALL *Acquire)(/* IADLXPerformanceMonitoringServices* pThis */);
    adlx_long (ADLX_STD_CALL *Release)(IADLXPerformanceMonitoringServices* pThis); // Used
    ADLX_RESULT (ADLX_STD_CALL *QueryInterface)(/* IADLXPerformanceMonitoringServices* pThis, const wchar_t* interfaceId, void** ppInterface */);

    //IADLXPerformanceMonitoringServices
    ADLX_RESULT (ADLX_STD_CALL *GetSamplingIntervalRange)(/* IADLXPerformanceMonitoringServices* pThis, ADLX_IntRange* range */);
    ADLX_RESULT (ADLX_STD_CALL *SetSamplingInterval)(/* IADLXPerformanceMonitoringServices* pThis, adlx_int intervalMs */);
    ADLX_RESULT (ADLX_STD_CALL *GetSamplingInterval)(/* IADLXPerformanceMonitoringServices* pThis, adlx_int* intervalMs */);
    ADLX_RESULT (ADLX_STD_CALL *GetMaxPerformanceMetricsHistorySizeRange)(/* IADLXPerformanceMonitoringServices* pThis, ADLX_IntRange* range */);
    ADLX_RESULT (ADLX_STD_CALL *SetMaxPerformanceMetricsHistorySize)(/* IADLXPerformanceMonitoringServices* pThis, adlx_int sizeSec */);
    ADLX_RESULT (ADLX_STD_CALL *GetMaxPerformanceMetricsHistorySize)(/* IADLXPerformanceMonitoringServices* pThis, adlx_int* sizeSec */);
    ADLX_RESULT (ADLX_STD_CALL *ClearPerformanceMetricsHistory)(/* IADLXPerformanceMonitoringServices* pThis */);
    ADLX_RESULT (ADLX_STD_CALL *GetCurrentPerformanceMetricsHistorySize)(/* IADLXPerformanceMonitoringServices* pThis, adlx_int* sizeSec */);
    ADLX_RESULT (ADLX_STD_CALL *StartPerformanceMetricsTracking)(/* IADLXPerformanceMonitoringServices* pThis */);
    ADLX_RESULT (ADLX_STD_CALL *StopPerformanceMetricsTracking)(/* IADLXPerformanceMonitoringServices* pThis */);
    ADLX_RESULT (ADLX_STD_CALL *GetAllMetricsHistory)(/* IADLXPerformanceMonitoringServices* pThis, adlx_int startMs, adlx_int stopMs, IADLXAllMetricsList** ppMetricsList */);
    ADLX_RESULT (ADLX_STD_CALL *GetGPUMetricsHistory)(/* IADLXPerformanceMonitoringServices* pThis, IADLXGPU* pGPU, adlx_int startMs, adlx_int stopMs, IADLXGPUMetricsList** ppMetricsList */);
    ADLX_RESULT (ADLX_STD_CALL *GetSystemMetricsHistory)(/* IADLXPerformanceMonitoringServices* pThis, adlx_int startMs, adlx_int stopMs, IADLXSystemMetricsList** ppMetricsList */);
    ADLX_RESULT (ADLX_STD_CALL *GetFPSHistory)(/* IADLXPerformanceMonitoringServices* pThis, adlx_int startMs, adlx_int stopMs, IADLXFPSList** ppMetricsList */);
    ADLX_RESULT (ADLX_STD_CALL *GetCurrentAllMetrics)(/* IADLXPerformanceMonitoringServices* pThis, IADLXAllMetrics** ppMetrics */);
    ADLX_RESULT (ADLX_STD_CALL *GetCurrentGPUMetrics)(IADLXPerformanceMonitoringServices* pThis, IADLXGPU* pGPU, IADLXGPUMetrics** ppMetrics); // Used
    ADLX_RESULT (ADLX_STD_CALL *GetCurrentSystemMetrics)(/* IADLXPerformanceMonitoringServices* pThis, IADLXSystemMetrics** ppMetrics */);
    ADLX_RESULT (ADLX_STD_CALL *GetCurrentFPS)(/* IADLXPerformanceMonitoringServices* pThis, IADLXFPS** ppMetrics */);
    ADLX_RESULT (ADLX_STD_CALL *GetSupportedGPUMetrics)(IADLXPerformanceMonitoringServices* pThis, IADLXGPU* pGPU, IADLXGPUMetricsSupport** ppMetricsSupported); // Used
    ADLX_RESULT (ADLX_STD_CALL *GetSupportedSystemMetrics)(/* IADLXPerformanceMonitoringServices* pThis, IADLXSystemMetricsSupport** ppMetricsSupported */);
}IADLXPerformanceMonitoringServicesVtbl;
struct IADLXPerformanceMonitoringServices { const IADLXPerformanceMonitoringServicesVtbl *pVtbl; };

typedef struct IADLXGPUMetricsSupportVtbl
{
    //IADLXInterface
    adlx_long (ADLX_STD_CALL* Acquire)(/* IADLXGPUMetricsSupport* pThis */);
    adlx_long (ADLX_STD_CALL* Release)(IADLXGPUMetricsSupport* pThis); // Used
    ADLX_RESULT (ADLX_STD_CALL* QueryInterface)(/* IADLXGPUMetricsSupport* pThis, const wchar_t* interfaceId, void** ppInterface */);

    //IADLXGPUMetricsSupport
    ADLX_RESULT (ADLX_STD_CALL* IsSupportedGPUUsage)(/* IADLXGPUMetricsSupport* pThis, adlx_bool* supported */);
    ADLX_RESULT (ADLX_STD_CALL* IsSupportedGPUClockSpeed)(/* IADLXGPUMetricsSupport* pThis, adlx_bool* supported */);
    ADLX_RESULT (ADLX_STD_CALL* IsSupportedGPUVRAMClockSpeed)(/* IADLXGPUMetricsSupport* pThis, adlx_bool* supported */);
    ADLX_RESULT (ADLX_STD_CALL* IsSupportedGPUTemperature)(/* IADLXGPUMetricsSupport* pThis, adlx_bool* supported */);
    ADLX_RESULT (ADLX_STD_CALL* IsSupportedGPUHotspotTemperature)(/* IADLXGPUMetricsSupport* pThis, adlx_bool* supported */);
    ADLX_RESULT (ADLX_STD_CALL* IsSupportedGPUPower)(/* IADLXGPUMetricsSupport* pThis, adlx_bool* supported */);
    ADLX_RESULT (ADLX_STD_CALL* IsSupportedGPUTotalBoardPower)(/* IADLXGPUMetricsSupport* pThis, adlx_bool* supported */);
    ADLX_RESULT (ADLX_STD_CALL* IsSupportedGPUFanSpeed)(/* IADLXGPUMetricsSupport* pThis, adlx_bool* supported */);
    ADLX_RESULT (ADLX_STD_CALL* IsSupportedGPUVRAM)(IADLXGPUMetricsSupport* pThis, adlx_bool* supported); // Used
    ADLX_RESULT (ADLX_STD_CALL* IsSupportedGPUVoltage)(/* IADLXGPUMetricsSupport* pThis, adlx_bool* supported */);

    ADLX_RESULT (ADLX_STD_CALL* GetGPUUsageRange)(/* IADLXGPUMetricsSupport* pThis, adlx_int* minValue, adlx_int* maxValue */);
    ADLX_RESULT (ADLX_STD_CALL* GetGPUClockSpeedRange)(/* IADLXGPUMetricsSupport* pThis, adlx_int* minValue, adlx_int* maxValue */);
    ADLX_RESULT (ADLX_STD_CALL* GetGPUVRAMClockSpeedRange)(/* IADLXGPUMetricsSupport* pThis, adlx_int* minValue, adlx_int* maxValue */);
    ADLX_RESULT (ADLX_STD_CALL* GetGPUTemperatureRange)(/* IADLXGPUMetricsSupport* pThis, adlx_int* minValue, adlx_int* maxValue */);
    ADLX_RESULT (ADLX_STD_CALL* GetGPUHotspotTemperatureRange)(/* IADLXGPUMetricsSupport* pThis, adlx_int* minValue, adlx_int* maxValue */);
    ADLX_RESULT (ADLX_STD_CALL* GetGPUPowerRange)(/* IADLXGPUMetricsSupport* pThis, adlx_int* minValue, adlx_int* maxValue */);
    ADLX_RESULT (ADLX_STD_CALL* GetGPUFanSpeedRange)(/* IADLXGPUMetricsSupport* pThis, adlx_int* minValue, adlx_int* maxValue */);
    ADLX_RESULT (ADLX_STD_CALL* GetGPUVRAMRange)(/* IADLXGPUMetricsSupport* pThis, adlx_int* minValue, adlx_int* maxValue */);
    ADLX_RESULT (ADLX_STD_CALL* GetGPUVoltageRange)(/* IADLXGPUMetricsSupport* pThis, adlx_int* minValue, adlx_int* maxValue */);
    ADLX_RESULT (ADLX_STD_CALL* GetGPUTotalBoardPowerRange)(/* IADLXGPUMetricsSupport* pThis, adlx_int* minValue, adlx_int* maxValue */);
} IADLXGPUMetricsSupportVtbl;
struct IADLXGPUMetricsSupport { const IADLXGPUMetricsSupportVtbl *pVtbl; };

typedef struct IADLXGPUMetricsVtbl
{
    //IADLXInterface
    adlx_long (ADLX_STD_CALL* Acquire)(/* IADLXGPUMetrics* pThis */);
    adlx_long (ADLX_STD_CALL* Release)(IADLXGPUMetrics* pThis); // Used
    ADLX_RESULT (ADLX_STD_CALL* QueryInterface)(/* IADLXGPUMetrics* pThis, const wchar_t* interfaceId, void** ppInterface */);

    //IADLXGPUMetrics
    ADLX_RESULT (ADLX_STD_CALL* TimeStamp)(/* IADLXGPUMetrics* pThis, adlx_int64* ms */);
    ADLX_RESULT (ADLX_STD_CALL* GPUUsage)(/* IADLXGPUMetrics* pThis, adlx_double* data */);
    ADLX_RESULT (ADLX_STD_CALL* GPUClockSpeed)(/* IADLXGPUMetrics* pThis, adlx_int* data */);
    ADLX_RESULT (ADLX_STD_CALL* GPUVRAMClockSpeed)(/* IADLXGPUMetrics* pThis, adlx_int* data */);
    ADLX_RESULT (ADLX_STD_CALL* GPUTemperature)(/* IADLXGPUMetrics* pThis, adlx_double* data */);
    ADLX_RESULT (ADLX_STD_CALL* GPUHotspotTemperature)(/* IADLXGPUMetrics* pThis, adlx_double* data */);
    ADLX_RESULT (ADLX_STD_CALL* GPUPower)(/* IADLXGPUMetrics* pThis, adlx_double* data */);
    ADLX_RESULT (ADLX_STD_CALL* GPUTotalBoardPower)(/* IADLXGPUMetrics* pThis, adlx_double* data */);
    ADLX_RESULT (ADLX_STD_CALL* GPUFanSpeed)(/* IADLXGPUMetrics* pThis, adlx_int* data */);
    ADLX_RESULT (ADLX_STD_CALL* GPUVRAM)(IADLXGPUMetrics* pThis, adlx_int* data); // Used
    ADLX_RESULT (ADLX_STD_CALL* GPUVoltage)(/* IADLXGPUMetrics* pThis, adlx_int* data */);
} IADLXGPUMetricsVtbl;
struct IADLXGPUMetrics { const IADLXGPUMetricsVtbl *pVtbl; };

struct {
  void *handle;
  ADLX_RESULT (*ADLXInitialize)(adlx_uint64 version, IADLXSystem** ppSystem);
  ADLX_RESULT (*ADLXInitializeWithIncompatibleDriver)(adlx_uint64 version, IADLXSystem** ppSystem);
  ADLX_RESULT (*ADLXQueryVersion)(const char** version);
  ADLX_RESULT (*ADLXTerminate)();
  IADLXSystem *sys;
} adlx { NULL, NULL, NULL, NULL, NULL, NULL };
static std::mutex ggml_adlx_lock;

extern "C" {

int ggml_hip_mgmt_init() {
    std::lock_guard<std::mutex> lock(ggml_adlx_lock);
    if (adlx.handle != NULL) {
        // Already initialized
        return 0;
    }
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);
    fs::path libPath = fs::path("\\Windows") / fs::path("System32") / fs::path("amdadlx64.dll");

    adlx.handle = (void*)LoadLibraryW(libPath.wstring().c_str());
    if (adlx.handle == NULL) {
        return ADLX_NOT_FOUND;
    }

    adlx.ADLXInitialize = (ADLX_RESULT (*)(adlx_uint64 version, IADLXSystem **ppSystem)) GetProcAddress((HMODULE)(adlx.handle), "ADLXInitialize");
    adlx.ADLXInitializeWithIncompatibleDriver = (ADLX_RESULT (*)(adlx_uint64 version, IADLXSystem **ppSystem)) GetProcAddress((HMODULE)(adlx.handle), "ADLXInitializeWithIncompatibleDriver");
    adlx.ADLXTerminate = (ADLX_RESULT (*)()) GetProcAddress((HMODULE)(adlx.handle), "ADLXTerminate");
    adlx.ADLXQueryVersion = (ADLX_RESULT (*)(const char **version)) GetProcAddress((HMODULE)(adlx.handle), "ADLXQueryVersion");
    if (adlx.ADLXInitialize == NULL || adlx.ADLXInitializeWithIncompatibleDriver == NULL || adlx.ADLXTerminate == NULL) {
        GGML_LOG_INFO("%s unable to locate required symbols in amdadlx64.dll, falling back to hip free memory reporting", __func__);
        FreeLibrary((HMODULE)(adlx.handle));
        adlx.handle = NULL;
        return ADLX_NOT_FOUND;
    }

    SetErrorMode(old_mode);

    // Aid in troubleshooting...
    if (adlx.ADLXQueryVersion != NULL) {
        const char *version = NULL;
        ADLX_RESULT status = adlx.ADLXQueryVersion(&version);
        if (ADLX_SUCCEEDED(status)) {
            GGML_LOG_DEBUG("%s located ADLX version %s\n", __func__, version);  
        }
    }

    ADLX_RESULT status = adlx.ADLXInitialize(ADLX_FULL_VERSION, &adlx.sys);
    if (ADLX_FAILED(status)) {
        // GGML_LOG_DEBUG("%s failed to initialize ADLX error=%d - attempting with incompatible driver...\n", __func__, status);
        // Try with the incompatible driver
        status = adlx.ADLXInitializeWithIncompatibleDriver(ADLX_FULL_VERSION, &adlx.sys);
        if (ADLX_FAILED(status)) {
            GGML_LOG_INFO("%s failed to initialize ADLX error=%d\n", __func__, status);
            FreeLibrary((HMODULE)(adlx.handle));
            adlx.handle = NULL;
            adlx.sys = NULL;
            return status;
        }
        // GGML_LOG_DEBUG("%s initialized ADLX with incpomatible driver\n", __func__);
    }
    return ADLX_OK;
}

void ggml_hip_mgmt_release() {
    std::lock_guard<std::mutex> lock(ggml_adlx_lock);
    if (adlx.handle == NULL) {
        // Already free
        return;
    }
    ADLX_RESULT status = adlx.ADLXTerminate();
    if (ADLX_FAILED(status)) {
        GGML_LOG_INFO("%s failed to terminate Adlx %d\n", __func__, status);
        // Unload anyway...
    }
    FreeLibrary((HMODULE)(adlx.handle));
    adlx.handle = NULL;
}

#define adlx_gdm_cleanup \
    if (gpuMetricsSupport != NULL) gpuMetricsSupport->pVtbl->Release(gpuMetricsSupport); \
    if (gpuMetrics != NULL) gpuMetrics->pVtbl->Release(gpuMetrics); \
    if (perfMonitoringServices != NULL) perfMonitoringServices->pVtbl->Release(perfMonitoringServices); \
    if (gpus != NULL) gpus->pVtbl->Release(gpus); \
    if (gpu != NULL) gpu->pVtbl->Release(gpu)

int ggml_hip_get_device_memory(const char *id, size_t *free, size_t *total, bool is_integrated_gpu) {
    std::lock_guard<std::mutex> lock(ggml_adlx_lock);
    if (adlx.handle == NULL) {
        GGML_LOG_INFO("%s ADLX was not initialized\n", __func__);
        return ADLX_ADL_INIT_ERROR;
    }
    IADLXGPUMetricsSupport *gpuMetricsSupport = NULL;
    IADLXPerformanceMonitoringServices *perfMonitoringServices = NULL;
    IADLXGPUList* gpus = NULL;
    IADLXGPU* gpu = NULL;
    IADLXGPUMetrics *gpuMetrics = NULL;
    ADLX_RESULT status;

    uint32_t pci_domain, pci_bus, pci_device, pci_function;
    if (sscanf(id, "%04x:%02x:%02x.%x", &pci_domain, &pci_bus, &pci_device, &pci_function) != 4) {
        // TODO - parse other formats?
        GGML_LOG_DEBUG("%s device ID was not a PCI ID %s\n", __func__, id);
        return ADLX_NOT_FOUND;
    }
    status = adlx.sys->pVtbl->GetPerformanceMonitoringServices(adlx.sys, &perfMonitoringServices);
    if (ADLX_FAILED(status)) {
        GGML_LOG_INFO("%s GetPerformanceMonitoringServices failed %d\n", __func__, status);
        return status;
    }

    status = adlx.sys->pVtbl->GetGPUs(adlx.sys, &gpus);
    if (ADLX_FAILED(status)) {
        GGML_LOG_INFO("%s GetGPUs failed %d\n", __func__, status);
        adlx_gdm_cleanup;
        return status;
    }

    // Get GPU list
    for (adlx_uint crt = gpus->pVtbl->Begin(gpus); crt != gpus->pVtbl->End(gpus); ++crt)
    {
        status = gpus->pVtbl->At_GPUList(gpus, crt, &gpu);
        if (ADLX_FAILED(status))
        {
            GGML_LOG_INFO("%s %d] At_GPUList failed %d\n", __func__, crt, status);
            continue;
        }
        adlx_int uniqueID;
        status = gpu->pVtbl->UniqueId(gpu, &uniqueID);
        if (ADLX_FAILED(status)) {
            GGML_LOG_INFO("%s %d] UniqueId lookup failed %d\n", __func__, crt, status);
            gpu->pVtbl->Release(gpu);
            gpu = NULL;
            continue;
        }
        if ((((uniqueID >> 8) & 0xff) != pci_bus) || ((uniqueID & 0xff) != pci_device)) {
            gpu->pVtbl->Release(gpu);
            gpu = NULL;
            continue;
        }
        // Any failures at this point should cause a fall-back to other APIs
        status = perfMonitoringServices->pVtbl->GetSupportedGPUMetrics(perfMonitoringServices, gpu, &gpuMetricsSupport);
        if (ADLX_FAILED(status)) {
            GGML_LOG_INFO("%s GetSupportedGPUMetrics failed %d\n", __func__, status);
            adlx_gdm_cleanup;
            return status;
        }
        status = perfMonitoringServices->pVtbl->GetCurrentGPUMetrics(perfMonitoringServices, gpu, &gpuMetrics);
        if (ADLX_FAILED(status)) {
            GGML_LOG_INFO("%s GetCurrentGPUMetrics failed %d\n", __func__, status);
            adlx_gdm_cleanup;
            return status;
        }

        adlx_bool supported = false;
        status = gpuMetricsSupport->pVtbl->IsSupportedGPUVRAM(gpuMetricsSupport, &supported);
        if (ADLX_FAILED(status)) {
            GGML_LOG_INFO("%s IsSupportedGPUVRAM failed %d\n", __func__, status);
            adlx_gdm_cleanup;
            return status;
        }
        
        adlx_uint totalVRAM = 0;
        status = gpu->pVtbl->TotalVRAM(gpu, &totalVRAM);
        if (ADLX_FAILED(status)) {
            GGML_LOG_INFO("%s TotalVRAM failed %d\n", __func__, status);
            adlx_gdm_cleanup;
            return status;
        }

        adlx_int usedVRAM = 0;
        status = gpuMetrics->pVtbl->GPUVRAM(gpuMetrics, &usedVRAM);
        if (ADLX_FAILED(status)) {
            GGML_LOG_INFO("%s GPUVRAM failed %d\n", __func__, status);
            adlx_gdm_cleanup;
            return status;
        }
        *total = size_t(totalVRAM) * 1024 * 1024;
        *free = size_t(totalVRAM-usedVRAM) * 1024 * 1024;

        adlx_gdm_cleanup;
        return ADLX_OK;
    }
    adlx_gdm_cleanup;
    return ADLX_NOT_FOUND;
}

} // extern "C"

#else // #ifdef _WIN32

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <filesystem>

#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>
#include <glob.h>
namespace fs = std::filesystem;

extern "C" {

int ggml_hip_mgmt_init() {
    return 0;
}
void ggml_hip_mgmt_release() {}
int ggml_hip_get_device_memory(const char *id, size_t *free, size_t *total, bool is_integrated_gpu) {
    GGML_LOG_INFO("%s searching for device %s\n", __func__, id);
    const std::string drmDeviceGlob = "/sys/class/drm/card*/device/uevent";
    const std::string drmTotalMemoryFile = "mem_info_vram_total";
    const std::string drmUsedMemoryFile = "mem_info_vram_used";
    const std::string drmGTTTotalMemoryFile = "mem_info_gtt_total";
    const std::string drmGTTUsedMemoryFile = "mem_info_gtt_used";
    const std::string drmUeventPCISlotLabel = "PCI_SLOT_NAME=";


    glob_t glob_result;
    glob(drmDeviceGlob.c_str(), GLOB_NOSORT, NULL, &glob_result);

    for (size_t i = 0; i < glob_result.gl_pathc; ++i) {
        const char* device_file = glob_result.gl_pathv[i];
        std::ifstream file(device_file);
        if (!file.is_open()) {
            std::cerr << "Failed to open sysfs node" << std::endl;
            globfree(&glob_result);
            return 1;
        }

        std::string line;
        while (std::getline(file, line)) {
            // Check for PCI_SLOT_NAME label
            if (line.find(drmUeventPCISlotLabel) == 0) {
                std::istringstream iss(line.substr(drmUeventPCISlotLabel.size()));
                std::string pciSlot;
                iss >> pciSlot;
                if (pciSlot == std::string(id)) {
                    std::string dir = fs::path(device_file).parent_path().string();

                    std::string totalFile = dir + "/" + drmTotalMemoryFile;
                    std::ifstream totalFileStream(totalFile.c_str());
                    if (!totalFileStream.is_open()) {
                        GGML_LOG_DEBUG("%s Failed to read sysfs node %s\n", __func__, totalFile.c_str());
                        file.close();
                        globfree(&glob_result);
                        return 1;
                    }

                    uint64_t memory;
                    totalFileStream >> memory;

                    std::string usedFile = dir + "/" + drmUsedMemoryFile;
                    std::ifstream usedFileStream(usedFile.c_str());
                    if (!usedFileStream.is_open()) {
                        GGML_LOG_DEBUG("%s Failed to read sysfs node %s\n", __func__, usedFile.c_str());
                        file.close();
                        globfree(&glob_result);
                        return 1;
                    }

                    uint64_t memoryUsed;
                    usedFileStream >> memoryUsed;

                    if (is_integrated_gpu) {
                        std::string totalFile = dir + "/" + drmGTTTotalMemoryFile;
                        std::ifstream totalFileStream(totalFile.c_str());
                        if (!totalFileStream.is_open()) {
                            GGML_LOG_DEBUG("%s Failed to read sysfs node %s\n", __func__, totalFile.c_str());
                            file.close();
                            globfree(&glob_result);
                            return 1;
                        }
                        uint64_t gtt;
                        totalFileStream >> gtt;
                        std::string usedFile = dir + "/" + drmGTTUsedMemoryFile;
                        std::ifstream usedFileStream(usedFile.c_str());
                        if (!usedFileStream.is_open()) {
                            GGML_LOG_DEBUG("%s Failed to read sysfs node %s\n", __func__, usedFile.c_str());
                            file.close();
                            globfree(&glob_result);
                            return 1;
                        }
                        uint64_t gttUsed;
                        usedFileStream >> gttUsed;
                        memory += gtt;
                        memoryUsed += gttUsed;
                    }

                    *total = memory;
                    *free = memory - memoryUsed;

                    file.close();
                    globfree(&glob_result);
                    return 0;
                }
            }
        }

        file.close();
    }
    GGML_LOG_DEBUG("%s unable to find matching device\n", __func__);
    globfree(&glob_result);
    return 1;
}

} // extern "C"

#endif // #ifdef _WIN32
cmake_minimum_required(VERSION 3.20)
project(OllamaMSI NONE)  # No languages needed - we only run external tools

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

# Required: Ollama version
if(NOT DEFINED OLLAMA_VERSION)
    set(OLLAMA_VERSION "0.0.0" CACHE STRING "Ollama version string")
endif()

# Package version (x.y.z only, no prerelease suffix)
if(NOT DEFINED OLLAMA_PKG_VERSION)
    string(REGEX MATCH "^[0-9]+\\.[0-9]+\\.[0-9]+" OLLAMA_PKG_VERSION "${OLLAMA_VERSION}")
    if(NOT OLLAMA_PKG_VERSION)
        set(OLLAMA_PKG_VERSION "0.0.0")
    endif()
endif()

# Dist directory containing built binaries
if(NOT DEFINED OLLAMA_DIST_DIR)
    set(OLLAMA_DIST_DIR "${CMAKE_SOURCE_DIR}/../../dist" CACHE PATH "Path to dist directory")
endif()
get_filename_component(OLLAMA_DIST_DIR "${OLLAMA_DIST_DIR}" ABSOLUTE)

# Base URL for downloading pre-built deps MSIs.
# Defaults to https://ollama.com/download (redirects to latest GitHub release).
# Override via environment for testing: OLLAMA_DEPS_DOWNLOAD_URL=http://localhost:8000
if(DEFINED ENV{OLLAMA_DEPS_DOWNLOAD_URL})
    set(OLLAMA_DEPS_DOWNLOAD_URL "$ENV{OLLAMA_DEPS_DOWNLOAD_URL}")
elseif(NOT OLLAMA_DEPS_DOWNLOAD_URL)
    set(OLLAMA_DEPS_DOWNLOAD_URL "https://ollama.com/download")
endif()
# Strip trailing slash for consistent URL construction
string(REGEX REPLACE "/+$" "" OLLAMA_DEPS_DOWNLOAD_URL "${OLLAMA_DEPS_DOWNLOAD_URL}")

# Force local rebuild of deps MSIs (skip download and reuse).
# Use when published deps are bad and need to be rebuilt from source:
#   cmake -DOLLAMA_DEPS_FORCE_BUILD=ON ...
# Or in CI: set OLLAMA_DEPS_FORCE_BUILD=1 in the environment.
if(DEFINED ENV{OLLAMA_DEPS_FORCE_BUILD})
    set(OLLAMA_DEPS_FORCE_BUILD "$ENV{OLLAMA_DEPS_FORCE_BUILD}")
endif()
if(NOT DEFINED OLLAMA_DEPS_FORCE_BUILD)
    set(OLLAMA_DEPS_FORCE_BUILD OFF)
endif()

# Output directory for built MSIs
set(MSI_OUTPUT_DIR "${OLLAMA_DIST_DIR}" CACHE PATH "Output directory for MSI files")

# Maximum uncompressed content per cabinet file (in MB).  WiX compresses one
# cabinet per thread, so smaller values create more cabs and more parallelism.
# Tradeoff: too small = worse compression ratio + per-cab overhead.
if(NOT DEFINED OLLAMA_CABSIZEMB)
    set(OLLAMA_CABSIZEMB "2" CACHE STRING "Max uncompressed MB per cabinet")
endif()

# --------------------------------------------------------------------------
# Tool detection
# --------------------------------------------------------------------------

find_program(WIX_EXE wix REQUIRED)
message(STATUS "WiX CLI: ${WIX_EXE}")

# Signing auto-detection (same logic as build_windows.ps1 checkEnv)
set(SIGNING_ENABLED FALSE)
if(DEFINED ENV{KEY_CONTAINER} AND EXISTS "${CMAKE_SOURCE_DIR}/../../ollama_inc.crt")
    set(SIGNING_ENABLED TRUE)
    set(OLLAMA_CERT "${CMAKE_SOURCE_DIR}/../../ollama_inc.crt")
    get_filename_component(OLLAMA_CERT "${OLLAMA_CERT}" ABSOLUTE)
    if(DEFINED ENV{SIGN_TOOL})
        set(SIGN_TOOL "$ENV{SIGN_TOOL}")
    else()
        set(SIGN_TOOL "C:/Program Files (x86)/Windows Kits/8.1/bin/x64/signtool.exe")
    endif()
    message(STATUS "Code signing: ENABLED")
    message(STATUS "  Certificate: ${OLLAMA_CERT}")
    message(STATUS "  Sign tool: ${SIGN_TOOL}")
else()
    message(STATUS "Code signing: DISABLED (set KEY_CONTAINER env and provide ollama_inc.crt)")
endif()

find_program(POWERSHELL_EXE pwsh HINTS "C:/Program Files/PowerShell/7")
if(NOT POWERSHELL_EXE)
    find_program(POWERSHELL_EXE powershell)
endif()
if(NOT POWERSHELL_EXE)
    message(FATAL_ERROR "PowerShell not found (pwsh or powershell)")
endif()
message(STATUS "PowerShell: ${POWERSHELL_EXE}")

message(STATUS "Ollama version: ${OLLAMA_VERSION}")
message(STATUS "Package version: ${OLLAMA_PKG_VERSION}")
message(STATUS "Dist directory: ${OLLAMA_DIST_DIR}")
message(STATUS "MSI output: ${MSI_OUTPUT_DIR}")
message(STATUS "Deps download URL: ${OLLAMA_DEPS_DOWNLOAD_URL}")
if(OLLAMA_DEPS_FORCE_BUILD)
    message(STATUS "Deps force build: ON (skipping download, rebuilding from source)")
endif()

# --------------------------------------------------------------------------
# Encode chained-uninstall.ps1 for embedding in core MSI custom action
# --------------------------------------------------------------------------
# The script is kept as a readable .ps1 file in the repo. At configure time
# we encode it to base64 (UTF-16LE) for use with powershell -EncodedCommand.
# The encoded string is passed to WiX as a preprocessor variable.

set(CHAINED_UNINSTALL_SCRIPT "${CMAKE_SOURCE_DIR}/chained-uninstall.ps1")
if(NOT EXISTS "${CHAINED_UNINSTALL_SCRIPT}")
    message(FATAL_ERROR "chained-uninstall.ps1 not found at ${CHAINED_UNINSTALL_SCRIPT}")
endif()
execute_process(
    COMMAND ${POWERSHELL_EXE} -NoProfile -Command
        "[Convert]::ToBase64String([System.Text.Encoding]::Unicode.GetBytes((Get-Content -Raw '${CHAINED_UNINSTALL_SCRIPT}')))"
    OUTPUT_VARIABLE CHAINED_UNINSTALL_ENCODED
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE ENCODE_RESULT
)
if(NOT ENCODE_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to encode chained-uninstall.ps1")
endif()
message(STATUS "Chained uninstall script encoded (${CHAINED_UNINSTALL_SCRIPT})")

# --------------------------------------------------------------------------
# Source directories
# --------------------------------------------------------------------------

set(WXS_DIR "${CMAKE_SOURCE_DIR}")
set(DEPS_WXS_DIR "${CMAKE_SOURCE_DIR}/dependencies")
set(DIST_AMD64 "${OLLAMA_DIST_DIR}/windows-amd64")
set(DIST_ARM64 "${OLLAMA_DIST_DIR}/windows-arm64")

# --------------------------------------------------------------------------
# Helper: append signing command to a custom_command OUTPUT
# --------------------------------------------------------------------------

function(sign_output MSI_PATH)
    if(SIGNING_ENABLED)
        add_custom_command(OUTPUT "${MSI_PATH}" APPEND
            COMMAND "${SIGN_TOOL}" sign /v /fd sha256
                /t http://timestamp.digicert.com
                /f "${OLLAMA_CERT}"
                /csp "Google Cloud KMS Provider"
                /kc "$ENV{KEY_CONTAINER}"
                "${MSI_PATH}"
            VERBATIM
        )
    endif()
endfunction()

# --------------------------------------------------------------------------
# Helper: build a backend or core MSI from WXS sources
#
# Uses add_custom_command(OUTPUT) for file-level dependency tracking.
# The MSI only rebuilds when its WXS sources or EXTRA_DEPENDS change.
# --------------------------------------------------------------------------

function(add_msi_target TARGET_NAME OUTPUT_MSI)
    cmake_parse_arguments(ARG "" "BINDPATH" "SOURCES;DEFINES;EXTRA_DEPENDS" ${ARGN})

    set(MSI_PATH "${MSI_OUTPUT_DIR}/${OUTPUT_MSI}")

    # Build the wix command arguments
    set(WIX_ARGS build -o "${MSI_PATH}")
    foreach(SRC ${ARG_SOURCES})
        list(APPEND WIX_ARGS "${SRC}")
    endforeach()
    list(APPEND WIX_ARGS -d PKG_VERSION=${OLLAMA_PKG_VERSION})
    list(APPEND WIX_ARGS -d CABSIZEMB=${OLLAMA_CABSIZEMB})
    list(APPEND WIX_ARGS -ext WixToolset.Util.wixext)
    if(ARG_BINDPATH)
        list(APPEND WIX_ARGS -bindpath "${ARG_BINDPATH}")
    endif()
    # Add WXS source directory as bindpath so relative paths (e.g., ..\assets\app.ico) resolve
    list(APPEND WIX_ARGS -bindpath "${WXS_DIR}")
    foreach(DEF ${ARG_DEFINES})
        list(APPEND WIX_ARGS -d "${DEF}")
    endforeach()

    add_custom_command(
        OUTPUT "${MSI_PATH}"
        DEPENDS ${ARG_SOURCES} ${ARG_EXTRA_DEPENDS}
        COMMAND ${POWERSHELL_EXE} -NoProfile -Command "Write-Host \"[$(Get-Date -Format 'HH:mm:ss')] Building ${OUTPUT_MSI}...\""
        COMMAND ${WIX_EXE} ${WIX_ARGS}
        COMMAND ${POWERSHELL_EXE} -NoProfile -Command "Write-Host \"[$(Get-Date -Format 'HH:mm:ss')] Finished ${OUTPUT_MSI}\""
        COMMENT "Building ${OUTPUT_MSI}"
        VERBATIM
    )
    sign_output("${MSI_PATH}")

    add_custom_target(${TARGET_NAME} DEPENDS "${MSI_PATH}")

    # Export the output path so downstream targets can DEPENDS on it
    set(${TARGET_NAME}_OUTPUT "${MSI_PATH}" PARENT_SCOPE)
endfunction()

# --------------------------------------------------------------------------
# Helper: build a deps MSI (with download-or-build logic)
#
# The download-or-build decision happens at configure time. They only change
# when the SDK version changes, which requires a cmake reconfigure anyway.
# Downloaded deps use add_custom_target (no-op + sign). Locally-built deps
# use add_custom_command(OUTPUT) for file-level dependency tracking.
#
# Flow:
#   1. Read SDK version from breadcrumb file (e.g., "12.8.1")
#   2. Construct filename: "ollama-cuda-deps-12.8.1.msi"
#   3. Try downloading from OLLAMA_DEPS_DOWNLOAD_URL
#   4. If download succeeds: use it (skip local build)
#   5. If download fails: warn, build locally with WiX
# --------------------------------------------------------------------------

function(add_deps_msi_target TARGET_NAME BACKEND_NAME DEPS_WXS DEPS_GENERATED_WXS
         COMPONENT_GROUP_ID ROOT_DIR_ID BACKEND_PATH EXCLUDE_FILE VERSION_FILE_NAME)

    set(GENERATED_WXS "${CMAKE_BINARY_DIR}/${DEPS_GENERATED_WXS}")
    file(GLOB_RECURSE DEPS_INPUT_FILES LIST_DIRECTORIES false
        "${DIST_AMD64}/lib/ollama/${BACKEND_PATH}/*")

    # Read version from breadcrumb file to create versioned output filename
    set(VERSION_FILE "${DIST_AMD64}/lib/ollama/${BACKEND_PATH}/${VERSION_FILE_NAME}")
    set(DEPS_VERSION "")
    if(EXISTS "${VERSION_FILE}")
        file(STRINGS "${VERSION_FILE}" DEPS_VERSION LIMIT_COUNT 1)
        string(STRIP "${DEPS_VERSION}" DEPS_VERSION)
        message(STATUS "  ${BACKEND_NAME} deps version: ${DEPS_VERSION}")
    endif()

    # Use versioned filename if breadcrumb exists, otherwise generic
    if(DEPS_VERSION)
        set(DEPS_MSI_NAME "ollama-${BACKEND_NAME}-deps-${DEPS_VERSION}.msi")
    else()
        set(DEPS_MSI_NAME "ollama-${BACKEND_NAME}-deps.msi")
        message(WARNING "No version breadcrumb for ${BACKEND_NAME} deps (${VERSION_FILE})")
    endif()
    set(DEPS_MSI_PATH "${MSI_OUTPUT_DIR}/${DEPS_MSI_NAME}")

    # ---- Try downloading or reusing pre-built MSI ----
    #
    # Priority order (unless OLLAMA_DEPS_FORCE_BUILD is set):
    #   1. Download from OLLAMA_DEPS_DOWNLOAD_URL (CI or local test server)
    #   2. Reuse existing local file (incremental build, same SDK version)
    #   3. Build locally from source
    #
    set(DEPS_DOWNLOADED FALSE)
    if(DEPS_VERSION)
        if(OLLAMA_DEPS_FORCE_BUILD)
            message(STATUS "  OLLAMA_DEPS_FORCE_BUILD set, skipping download for ${DEPS_MSI_NAME}")
        else()
            # Always try downloading from the server first.  Short timeout so
            # transient outages don't stall the build — worst case we rebuild
            # deps locally, which is better than blocking a release.
            set(DOWNLOAD_URL "${OLLAMA_DEPS_DOWNLOAD_URL}/${DEPS_MSI_NAME}")
            set(DOWNLOAD_TEMP "${CMAKE_BINARY_DIR}/_download_${DEPS_MSI_NAME}")
            message(STATUS "  Downloading ${DEPS_MSI_NAME} from ${DOWNLOAD_URL}")
            file(DOWNLOAD "${DOWNLOAD_URL}" "${DOWNLOAD_TEMP}"
                STATUS DOWNLOAD_STATUS
                TIMEOUT 15
            )
            list(GET DOWNLOAD_STATUS 0 DOWNLOAD_STATUS_CODE)
            list(GET DOWNLOAD_STATUS 1 DOWNLOAD_ERROR_MSG)
            if(DOWNLOAD_STATUS_CODE EQUAL 0)
                # CMake file(DOWNLOAD) returns status 0 even for HTTP 404 responses.
                # Verify the downloaded file is a plausible MSI (> 1KB).
                # A 404/empty response produces a 0-byte or small file.
                file(SIZE "${DOWNLOAD_TEMP}" DOWNLOADED_SIZE)
                if(DOWNLOADED_SIZE GREATER 1024)
                    file(RENAME "${DOWNLOAD_TEMP}" "${DEPS_MSI_PATH}")
                    message(STATUS "  Downloaded ${DEPS_MSI_NAME} (${DOWNLOADED_SIZE} bytes)")
                    set(DEPS_DOWNLOADED TRUE)
                else()
                    message(STATUS "  ${DEPS_MSI_NAME} not available from server (${DOWNLOADED_SIZE} bytes returned)")
                    file(REMOVE "${DOWNLOAD_TEMP}")
                endif()
            else()
                message(STATUS "  Download failed: ${DOWNLOAD_ERROR_MSG}")
                file(REMOVE "${DOWNLOAD_TEMP}")
            endif()

            # If download didn't work, try reusing an existing local file.
            # Deps only change when the SDK version changes (which changes the filename),
            # so an existing file with the right name is still valid.
            if(NOT DEPS_DOWNLOADED AND EXISTS "${DEPS_MSI_PATH}")
                file(SIZE "${DEPS_MSI_PATH}" EXISTING_SIZE)
                if(EXISTING_SIZE GREATER 1024)
                    message(STATUS "  Reusing existing local ${DEPS_MSI_NAME} (${EXISTING_SIZE} bytes)")
                    set(DEPS_DOWNLOADED TRUE)
                endif()
            endif()
        endif()
    else()
        message(WARNING "  Cannot download ${BACKEND_NAME} deps: no version breadcrumb")
    endif()

    # ---- Create target ----
    if(DEPS_DOWNLOADED)
        # MSI was downloaded or reused from a previous build. Leave it
        # byte-for-byte intact so packages.json can preserve the published hash.
        add_custom_target(${TARGET_NAME}
            COMMENT "Using pre-built ${DEPS_MSI_NAME}"
        )
    else()
        message(STATUS "  Will build ${DEPS_MSI_NAME} locally")

        # Generate the WXS fragment for deps files
        add_custom_command(
            OUTPUT "${GENERATED_WXS}"
            COMMAND ${POWERSHELL_EXE} -ExecutionPolicy Bypass -File
                "${DEPS_WXS_DIR}/generate-backend-deps.ps1"
                -BackendPath "lib/ollama/${BACKEND_PATH}"
                -ComponentGroupId "${COMPONENT_GROUP_ID}"
                -RootDirectoryId "${ROOT_DIR_ID}"
                -DistDir "${OLLAMA_DIST_DIR}"
                -OutputFile "${GENERATED_WXS}"
                -ExcludeFile "${EXCLUDE_FILE}"
            DEPENDS "${DEPS_WXS_DIR}/generate-backend-deps.ps1" ${DEPS_INPUT_FILES}
            COMMENT "Generating deps WXS for ${BACKEND_NAME}"
            VERBATIM
        )

        # Build the deps MSI
        set(DEPS_SOURCES
            "${DEPS_WXS_DIR}/${DEPS_WXS}"
            "${GENERATED_WXS}"
            "${DEPS_WXS_DIR}/Folders.wxs"
        )

        set(WIX_ARGS build -o "${DEPS_MSI_PATH}")
        foreach(SRC ${DEPS_SOURCES})
            list(APPEND WIX_ARGS "${SRC}")
        endforeach()
        # Deps MSIs use the SDK version (from breadcrumb) for their MSI Version
        # so they don't change when only the Ollama version bumps.
        if(DEPS_VERSION)
            list(APPEND WIX_ARGS -d DEPS_VERSION=${DEPS_VERSION})
        else()
            # Fallback to Ollama version if no breadcrumb (shouldn't happen in CI)
            list(APPEND WIX_ARGS -d DEPS_VERSION=${OLLAMA_PKG_VERSION})
        endif()
        list(APPEND WIX_ARGS -d CABSIZEMB=${OLLAMA_CABSIZEMB})
        list(APPEND WIX_ARGS -bindpath "${DIST_AMD64}")

        add_custom_command(
            OUTPUT "${DEPS_MSI_PATH}"
            DEPENDS "${GENERATED_WXS}" ${DEPS_INPUT_FILES} ${DEPS_SOURCES}
            COMMAND ${POWERSHELL_EXE} -NoProfile -Command "Write-Host \"[$(Get-Date -Format 'HH:mm:ss')] Building ${DEPS_MSI_NAME}...\""
            COMMAND ${WIX_EXE} ${WIX_ARGS}
            COMMAND ${POWERSHELL_EXE} -NoProfile -Command "Write-Host \"[$(Get-Date -Format 'HH:mm:ss')] Finished ${DEPS_MSI_NAME}\""
            COMMENT "Building ${DEPS_MSI_NAME}"
            VERBATIM
        )
        sign_output("${DEPS_MSI_PATH}")

        add_custom_target(${TARGET_NAME} DEPENDS "${DEPS_MSI_PATH}")
    endif()

    # Export the output path so downstream targets can use it
    set(${TARGET_NAME}_OUTPUT "${DEPS_MSI_PATH}" PARENT_SCOPE)
endfunction()

# --------------------------------------------------------------------------
# Deps MSI targets
# --------------------------------------------------------------------------

# CUDA v12 deps
add_deps_msi_target(msi-cuda-v12-deps "cuda"
    "cuda-v12-deps.wxs" "cuda-v12-deps-generated.wxs"
    "cuda_v12_deps_Components" "lib_ollama_cuda_v12_Dir"
    "cuda_v12" "ggml-cuda*.dll" "cuda-version.txt")

# CUDA v13 deps
add_deps_msi_target(msi-cuda-v13-deps "cuda-v13"
    "cuda-v13-deps.wxs" "cuda-v13-deps-generated.wxs"
    "cuda_v13_deps_Components" "lib_ollama_cuda_v13_Dir"
    "cuda_v13" "ggml-cuda*.dll" "cuda-version.txt")

# ROCm deps
add_deps_msi_target(msi-rocm-deps "rocm"
    "rocm-deps.wxs" "rocm-deps-generated.wxs"
    "rocm_deps_Components" "lib_ollama_rocm_Dir"
    "rocm" "ggml-hip*.dll" "rocm-version.txt")

# Vulkan deps
add_deps_msi_target(msi-vulkan-deps "vulkan"
    "vulkan-deps.wxs" "vulkan-deps-generated.wxs"
    "vulkan_deps_Components" "lib_ollama_vulkan_Dir"
    "vulkan" "ggml-vulkan*.dll" "vulkan-version.txt")

# MLX CUDA v13 deps
add_deps_msi_target(msi-mlx-deps "mlx"
    "mlx-deps.wxs" "mlx-deps-generated.wxs"
    "mlx_cuda_v13_deps_Components" "lib_ollama_mlx_cuda_v13_Dir"
    "mlx_cuda_v13" "mlx.dll,mlxc.dll" "mlx-version.txt")

# --------------------------------------------------------------------------
# Backend MSI targets
#
# Each backend MSI depends on:
#   - Its .wxs source files (rebuild if WiX definition changes)
#   - The key backend DLL in dist (rebuild if backend binaries change)
#   - Its deps target (ordering: deps must be built/signed first)
# --------------------------------------------------------------------------

# CUDA v12
add_msi_target(msi-cuda-v12 "ollama-cuda-v12.msi"
    SOURCES "${WXS_DIR}/cuda-v12.wxs" "${WXS_DIR}/Folders.wxs"
    BINDPATH "${DIST_AMD64}"
    EXTRA_DEPENDS "${DIST_AMD64}/lib/ollama/cuda_v12/ggml-cuda.dll")
add_dependencies(msi-cuda-v12 msi-cuda-v12-deps)

# CUDA v13
add_msi_target(msi-cuda-v13 "ollama-cuda-v13.msi"
    SOURCES "${WXS_DIR}/cuda-v13.wxs" "${WXS_DIR}/Folders.wxs"
    BINDPATH "${DIST_AMD64}"
    EXTRA_DEPENDS "${DIST_AMD64}/lib/ollama/cuda_v13/ggml-cuda.dll")
add_dependencies(msi-cuda-v13 msi-cuda-v13-deps)

# ROCm
add_msi_target(msi-rocm "ollama-rocm.msi"
    SOURCES "${WXS_DIR}/rocm.wxs" "${WXS_DIR}/Folders.wxs"
    BINDPATH "${DIST_AMD64}"
    EXTRA_DEPENDS "${DIST_AMD64}/lib/ollama/rocm/ggml-hip.dll")
add_dependencies(msi-rocm msi-rocm-deps)

# Vulkan
add_msi_target(msi-vulkan "ollama-vulkan.msi"
    SOURCES "${WXS_DIR}/vulkan.wxs" "${WXS_DIR}/Folders.wxs"
    BINDPATH "${DIST_AMD64}"
    EXTRA_DEPENDS "${DIST_AMD64}/lib/ollama/vulkan/ggml-vulkan.dll")
add_dependencies(msi-vulkan msi-vulkan-deps)

# MLX CUDA v13
add_msi_target(msi-mlx-cuda-v13 "ollama-mlx-cuda-v13.msi"
    SOURCES "${WXS_DIR}/mlx-cuda-v13.wxs" "${WXS_DIR}/Folders.wxs"
    BINDPATH "${DIST_AMD64}"
    EXTRA_DEPENDS "${DIST_AMD64}/lib/ollama/mlx_cuda_v13/mlx.dll"
        "${DIST_AMD64}/lib/ollama/mlx_cuda_v13/mlxc.dll")
add_dependencies(msi-mlx-cuda-v13 msi-mlx-deps)

# --------------------------------------------------------------------------
# Generate packages.json
#
# Uses add_custom_target (always-run) with target-level ordering via
# add_dependencies. This keeps each MSI build in its own MSBuild project,
# preventing the race conditions that occur when MSBuild parallelizes
# custom commands folded into a single project.
#
# The script does compare-and-write: packages.json is only updated when
# content changes. Since msi-core has a file-level dependency on
# packages.json, it only rebuilds when hashes actually differ.
# --------------------------------------------------------------------------

set(PACKAGES_JSON "${OLLAMA_DIST_DIR}/packages.json")

add_custom_target(generate-packages-json
    COMMAND ${POWERSHELL_EXE} -ExecutionPolicy Bypass -File
        "${WXS_DIR}/generate-packages-json.ps1"
        -Version "${OLLAMA_VERSION}"
        -DistDir "${OLLAMA_DIST_DIR}"
    COMMENT "Generating packages.json"
    VERBATIM
)
# Target-level ordering: all backend + deps MSIs must be built before hashing
add_dependencies(generate-packages-json
    msi-cuda-v12 msi-cuda-v13 msi-rocm msi-vulkan msi-mlx-cuda-v13
    msi-cuda-v12-deps msi-cuda-v13-deps msi-rocm-deps msi-vulkan-deps msi-mlx-deps)

# --------------------------------------------------------------------------
# Core MSI targets
#
# Depends on packages.json (file-level) and key payload binaries so the
# MSI rebuilds when either the manifest or the binaries change.
# --------------------------------------------------------------------------

set(CORE_RUNTIME_WXS "${CMAKE_BINARY_DIR}/core-runtime-generated.wxs")
file(GLOB CORE_RUNTIME_FILES LIST_DIRECTORIES false
    "${DIST_AMD64}/lib/ollama/*")
set(CORE_REQUIRED_PAYLOADS
    "${DIST_AMD64}/ollama.exe"
    "${DIST_AMD64}/lib/ollama/llama-server.exe"
    "${OLLAMA_DIST_DIR}/windows-ollama-app-amd64.exe")
string(REPLACE ";" "|" CORE_REQUIRED_PAYLOADS_ARG "${CORE_REQUIRED_PAYLOADS}")
add_custom_target(check-core-payloads
    COMMAND ${CMAKE_COMMAND}
        "-DREQUIRED_PAYLOADS=${CORE_REQUIRED_PAYLOADS_ARG}"
        -P "${WXS_DIR}/check-required-payloads.cmake"
    COMMENT "Checking core MSI payloads"
    VERBATIM
)
add_custom_command(
    OUTPUT "${CORE_RUNTIME_WXS}"
    COMMAND ${POWERSHELL_EXE} -ExecutionPolicy Bypass -File
        "${DEPS_WXS_DIR}/generate-backend-deps.ps1"
        -BackendPath "lib/ollama"
        -ComponentGroupId "core_runtime_Components"
        -RootDirectoryId "lib_ollama_Dir"
        -DistDir "${OLLAMA_DIST_DIR}"
        -OutputFile "${CORE_RUNTIME_WXS}"
        -NoRecurse
    DEPENDS "${DEPS_WXS_DIR}/generate-backend-deps.ps1" ${CORE_RUNTIME_FILES}
    COMMENT "Generating core runtime WXS"
    VERBATIM
)

add_msi_target(msi-core "ollama-core.msi"
    SOURCES "${WXS_DIR}/ollama-core.wxs" "${WXS_DIR}/Folders.wxs" "${CORE_RUNTIME_WXS}"
    BINDPATH "${DIST_AMD64}"
    DEFINES "CHAINED_UNINSTALL_ENCODED=${CHAINED_UNINSTALL_ENCODED}"
    EXTRA_DEPENDS "${PACKAGES_JSON}" "${DIST_AMD64}/ollama.exe"
        "${OLLAMA_DIST_DIR}/windows-ollama-app-amd64.exe" ${CORE_RUNTIME_FILES})
# Target-level ordering: generate-packages-json must run before msi-core.
# The file-level dep on PACKAGES_JSON handles incremental rebuild detection
# (only rebuilds core MSI when packages.json content actually changes).
add_dependencies(msi-core generate-packages-json)
add_dependencies(msi-core check-core-payloads)

# ARM64 core MSI (only if ARM64 binaries exist)
if(EXISTS "${DIST_ARM64}/ollama.exe")
    set(CORE_RUNTIME_ARM64_WXS "${CMAKE_BINARY_DIR}/core-runtime-arm64-generated.wxs")
    file(GLOB CORE_RUNTIME_ARM64_FILES LIST_DIRECTORIES false
        "${DIST_ARM64}/lib/ollama/*")
    set(CORE_REQUIRED_PAYLOADS_ARM64
        "${DIST_ARM64}/ollama.exe"
        "${DIST_ARM64}/lib/ollama/llama-server.exe"
        "${OLLAMA_DIST_DIR}/windows-ollama-app-arm64.exe")
    string(REPLACE ";" "|" CORE_REQUIRED_PAYLOADS_ARM64_ARG "${CORE_REQUIRED_PAYLOADS_ARM64}")
    add_custom_target(check-core-arm64-payloads
        COMMAND ${CMAKE_COMMAND}
            "-DREQUIRED_PAYLOADS=${CORE_REQUIRED_PAYLOADS_ARM64_ARG}"
            -P "${WXS_DIR}/check-required-payloads.cmake"
        COMMENT "Checking ARM64 core MSI payloads"
        VERBATIM
    )
    add_custom_command(
        OUTPUT "${CORE_RUNTIME_ARM64_WXS}"
        COMMAND ${POWERSHELL_EXE} -ExecutionPolicy Bypass -File
            "${DEPS_WXS_DIR}/generate-backend-deps.ps1"
            -BackendPath "lib/ollama"
            -ComponentGroupId "core_runtime_Components"
            -RootDirectoryId "lib_ollama_Dir"
            -DistDir "${OLLAMA_DIST_DIR}"
            -OutputFile "${CORE_RUNTIME_ARM64_WXS}"
            -Arch "arm64"
            -NoRecurse
        DEPENDS "${DEPS_WXS_DIR}/generate-backend-deps.ps1" ${CORE_RUNTIME_ARM64_FILES}
        COMMENT "Generating ARM64 core runtime WXS"
        VERBATIM
    )

    add_msi_target(msi-core-arm64 "ollama-core-arm64.msi"
        SOURCES "${WXS_DIR}/ollama-core-arm64.wxs" "${WXS_DIR}/Folders.wxs" "${CORE_RUNTIME_ARM64_WXS}"
        BINDPATH "${DIST_ARM64}"
        DEFINES "CHAINED_UNINSTALL_ENCODED=${CHAINED_UNINSTALL_ENCODED}"
        EXTRA_DEPENDS "${PACKAGES_JSON}" "${DIST_ARM64}/ollama.exe"
            "${OLLAMA_DIST_DIR}/windows-ollama-app-arm64.exe" ${CORE_RUNTIME_ARM64_FILES})
    add_dependencies(msi-core-arm64 generate-packages-json)
    add_dependencies(msi-core-arm64 check-core-arm64-payloads)
    message(STATUS "ARM64 core MSI: ENABLED")
else()
    message(STATUS "ARM64 core MSI: DISABLED (no ARM64 binaries in ${DIST_ARM64})")
endif()

# --------------------------------------------------------------------------
# Convenience target: build everything
# --------------------------------------------------------------------------

add_custom_target(msi-all
    COMMENT "All MSI packages built"
)
add_dependencies(msi-all msi-core)
if(EXISTS "${DIST_ARM64}/ollama.exe")
    add_dependencies(msi-all msi-core-arm64)
endif()

# --------------------------------------------------------------------------
# Stage install.ps1 (with file-level dependency tracking)
# --------------------------------------------------------------------------

set(INSTALL_SCRIPT_SRC "${CMAKE_SOURCE_DIR}/../../scripts/install.ps1")
if(EXISTS "${INSTALL_SCRIPT_SRC}")
    set(INSTALL_SCRIPT_DIST "${MSI_OUTPUT_DIR}/install.ps1")
    add_custom_command(
        OUTPUT "${INSTALL_SCRIPT_DIST}"
        DEPENDS "${INSTALL_SCRIPT_SRC}"
        COMMAND ${CMAKE_COMMAND} -E copy "${INSTALL_SCRIPT_SRC}" "${INSTALL_SCRIPT_DIST}"
        COMMENT "Staging install.ps1 to dist"
        VERBATIM
    )
    if(SIGNING_ENABLED)
        add_custom_command(OUTPUT "${INSTALL_SCRIPT_DIST}" APPEND
            COMMAND "${SIGN_TOOL}" sign /v /fd sha256
                /t http://timestamp.digicert.com
                /f "${OLLAMA_CERT}"
                /csp "Google Cloud KMS Provider"
                /kc "$ENV{KEY_CONTAINER}"
                "${INSTALL_SCRIPT_DIST}"
            VERBATIM
        )
    endif()
    add_custom_target(stage-install-script DEPENDS "${INSTALL_SCRIPT_DIST}")
    add_dependencies(msi-all stage-install-script)
endif()

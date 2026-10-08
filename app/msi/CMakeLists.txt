cmake_minimum_required(VERSION 3.20)
project(OllamaMSI NONE)  # No languages needed - we only run external tools

function(ollama_msi_product_version OUT_VAR VERSION LABEL)
    string(REGEX MATCH "^[0-9]+\\.[0-9]+\\.[0-9]+" _msi_version "${VERSION}")
    if(NOT _msi_version)
        message(FATAL_ERROR "${LABEL} must start with x.y.z for MSI metadata, got '${VERSION}'")
    endif()
    set(${OUT_VAR} "${_msi_version}" PARENT_SCOPE)
endfunction()

function(ollama_read_version_breadcrumb OUT_VAR VERSION_FILE LABEL)
    set(_version "")
    if(EXISTS "${VERSION_FILE}")
        file(STRINGS "${VERSION_FILE}" _version LIMIT_COUNT 1)
        string(STRIP "${_version}" _version)
        if(NOT _version)
            message(FATAL_ERROR "${LABEL} version breadcrumb is empty: ${VERSION_FILE}")
        endif()
    endif()
    set(${OUT_VAR} "${_version}" PARENT_SCOPE)
endfunction()

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

# Required: Ollama version
if(NOT DEFINED OLLAMA_VERSION)
    set(OLLAMA_VERSION "0.0.0" CACHE STRING "Ollama version string")
endif()

# Windows Installer ProductVersion is numeric x.y.z; prerelease suffixes from
# release tags stay in OLLAMA_VERSION and are stripped here for MSI metadata.
ollama_msi_product_version(OLLAMA_MSI_VERSION "${OLLAMA_VERSION}" "OLLAMA_VERSION")

# Dist directory containing built binaries
if(NOT DEFINED OLLAMA_DIST_DIR)
    set(OLLAMA_DIST_DIR "${CMAKE_SOURCE_DIR}/../../dist" CACHE PATH "Path to dist directory")
endif()
get_filename_component(OLLAMA_DIST_DIR "${OLLAMA_DIST_DIR}" ABSOLUTE)
set(OLLAMA_REPO_ROOT "${CMAKE_SOURCE_DIR}/../..")
get_filename_component(OLLAMA_REPO_ROOT "${OLLAMA_REPO_ROOT}" ABSOLUTE)

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

# Maximum uncompressed content per cabinet file (in MB).
# 200 MB matches WiX's default and avoids generating thousands of tiny cabs.
if(NOT DEFINED OLLAMA_CABSIZEMB)
    set(OLLAMA_CABSIZEMB "200" CACHE STRING "Max uncompressed MB per cabinet")
endif()

# --------------------------------------------------------------------------
# Tool detection
# --------------------------------------------------------------------------

find_program(WIX_EXE wix REQUIRED)
message(STATUS "WiX CLI: ${WIX_EXE}")

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
find_program(MSI_GO_EXECUTABLE go)

message(STATUS "Ollama version: ${OLLAMA_VERSION}")
message(STATUS "MSI product version: ${OLLAMA_MSI_VERSION}")
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

function(add_msi_go_license_target TARGET_NAME ARCH DIST_DIR)
    set(GO_LICENSE_OUTPUT "${DIST_DIR}/lib/ollama/GO_LICENSE")
    if(MSI_GO_EXECUTABLE)
        add_custom_command(
            OUTPUT "${GO_LICENSE_OUTPUT}"
            COMMAND ${CMAKE_COMMAND}
                "-DGO_EXECUTABLE=${MSI_GO_EXECUTABLE}"
                "-DSOURCE_DIR=${OLLAMA_REPO_ROOT}"
                "-DBINARY_DIR=${CMAKE_BINARY_DIR}/go-license-${ARCH}"
                "-DOUTPUT_DIR=${DIST_DIR}/lib/ollama"
                "-DTARGETS=windows/${ARCH}"
                -P "${OLLAMA_REPO_ROOT}/cmake/generate_go_license.cmake"
            DEPENDS
                "${OLLAMA_REPO_ROOT}/cmake/generate_go_license.cmake"
                "${OLLAMA_REPO_ROOT}/go.mod"
                "${OLLAMA_REPO_ROOT}/go.sum"
            COMMENT "Collecting Go licenses for Windows ${ARCH}"
            VERBATIM
        )
        add_custom_target(${TARGET_NAME} DEPENDS "${GO_LICENSE_OUTPUT}")
    elseif(EXISTS "${GO_LICENSE_OUTPUT}")
        add_custom_target(${TARGET_NAME} DEPENDS "${GO_LICENSE_OUTPUT}")
    else()
        message(FATAL_ERROR "GO_LICENSE missing at ${GO_LICENSE_OUTPUT}; install Go or build ollama-go-license before building MSIs")
    endif()
    set(${TARGET_NAME}_OUTPUT "${GO_LICENSE_OUTPUT}" PARENT_SCOPE)
endfunction()

set(OLLAMA_ROCM_BACKEND_DIR "" CACHE STRING "ROCm backend payload directory under lib/ollama")
if(NOT OLLAMA_ROCM_BACKEND_DIR)
    file(GLOB OLLAMA_ROCM_BACKEND_CANDIDATES LIST_DIRECTORIES true
        "${DIST_AMD64}/lib/ollama/rocm_v*")
    list(LENGTH OLLAMA_ROCM_BACKEND_CANDIDATES OLLAMA_ROCM_BACKEND_COUNT)
    if(OLLAMA_ROCM_BACKEND_COUNT GREATER 1)
        message(FATAL_ERROR "Multiple ROCm backend directories found: ${OLLAMA_ROCM_BACKEND_CANDIDATES}")
    elseif(OLLAMA_ROCM_BACKEND_COUNT EQUAL 1)
        list(GET OLLAMA_ROCM_BACKEND_CANDIDATES 0 OLLAMA_ROCM_BACKEND_PATH)
        get_filename_component(OLLAMA_ROCM_BACKEND_DIR "${OLLAMA_ROCM_BACKEND_PATH}" NAME)
    else()
        set(OLLAMA_ROCM_BACKEND_DIR "rocm_v7_1")
    endif()
endif()
if(OLLAMA_ROCM_BACKEND_DIR MATCHES "[/\\\\]")
    message(FATAL_ERROR "OLLAMA_ROCM_BACKEND_DIR must be a directory name, got ${OLLAMA_ROCM_BACKEND_DIR}")
endif()
message(STATUS "ROCm backend directory: ${OLLAMA_ROCM_BACKEND_DIR}")

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
    cmake_parse_arguments(ARG "" "BINDPATH" "SOURCES;DEFINES;EXTRA_DEPENDS;EXTENSIONS" ${ARGN})

    set(MSI_PATH "${MSI_OUTPUT_DIR}/${OUTPUT_MSI}")

    # Build the wix command arguments
    set(WIX_ARGS build -o "${MSI_PATH}")
    foreach(SRC ${ARG_SOURCES})
        list(APPEND WIX_ARGS "${SRC}")
    endforeach()
    list(APPEND WIX_ARGS -d MSI_VERSION=${OLLAMA_MSI_VERSION})
    list(APPEND WIX_ARGS -d CABSIZEMB=${OLLAMA_CABSIZEMB})
    list(APPEND WIX_ARGS -d ROCM_BACKEND_DIR=${OLLAMA_ROCM_BACKEND_DIR})
    foreach(EXTENSION ${ARG_EXTENSIONS})
        list(APPEND WIX_ARGS -ext "${EXTENSION}")
    endforeach()
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
#   1. Read dependency version from breadcrumb file (e.g., "12.8.1")
#   2. Construct filename: "ollama-cuda-deps-12.8.1.msi"
#   3. Try downloading from OLLAMA_DEPS_DOWNLOAD_URL
#   4. If download succeeds: use it (skip local build)
#   5. If download fails: warn, build locally with WiX
# --------------------------------------------------------------------------

function(add_deps_msi_target TARGET_NAME BACKEND_NAME DEPS_WXS DEPS_GENERATED_WXS
         COMPONENT_GROUP_ID ROOT_DIR_ID BACKEND_PATH EXCLUDE_FILE VERSION_FILE_NAME)
    cmake_parse_arguments(ARG "" "DEPS_VERSION" "EXTRA_SOURCES;EXTRA_DEPENDS" ${ARGN})

    set(GENERATED_WXS "${CMAKE_BINARY_DIR}/${DEPS_GENERATED_WXS}")
    file(GLOB_RECURSE DEPS_INPUT_FILES LIST_DIRECTORIES false
        "${DIST_AMD64}/lib/ollama/${BACKEND_PATH}/*")
    list(LENGTH DEPS_INPUT_FILES DEPS_INPUT_FILE_COUNT)

    # Read the dependency breadcrumb to create a versioned output filename.
    set(VERSION_FILE "${DIST_AMD64}/lib/ollama/${BACKEND_PATH}/${VERSION_FILE_NAME}")
    set(DEPS_VERSION "${ARG_DEPS_VERSION}")
    if(NOT DEPS_VERSION)
        ollama_read_version_breadcrumb(DEPS_VERSION "${VERSION_FILE}" "${BACKEND_NAME} deps")
    endif()

    if(NOT DEPS_VERSION AND DEPS_INPUT_FILE_COUNT GREATER 0)
        message(FATAL_ERROR "Missing version breadcrumb for ${BACKEND_NAME} deps: ${VERSION_FILE}")
    endif()

    if(DEPS_VERSION)
        message(STATUS "  ${BACKEND_NAME} deps version: ${DEPS_VERSION}")
    endif()

    # Use versioned filename if breadcrumb exists, otherwise generic
    if(DEPS_VERSION)
        set(DEPS_MSI_NAME "ollama-${BACKEND_NAME}-deps-${DEPS_VERSION}.msi")
        ollama_msi_product_version(DEPS_MSI_VERSION "${DEPS_VERSION}" "${BACKEND_NAME} deps version")
    else()
        set(DEPS_MSI_NAME "ollama-${BACKEND_NAME}-deps.msi")
        set(DEPS_MSI_VERSION "${OLLAMA_MSI_VERSION}")
        message(WARNING "No version breadcrumb for ${BACKEND_NAME} deps (${VERSION_FILE})")
    endif()
    set(DEPS_MSI_PATH "${MSI_OUTPUT_DIR}/${DEPS_MSI_NAME}")

    file(GLOB STALE_DEPS_MSIS LIST_DIRECTORIES false
        "${MSI_OUTPUT_DIR}/ollama-${BACKEND_NAME}-deps*.msi")
    foreach(STALE_DEPS_MSI IN LISTS STALE_DEPS_MSIS)
        if(NOT "${STALE_DEPS_MSI}" STREQUAL "${DEPS_MSI_PATH}")
            message(STATUS "  Removing stale ${BACKEND_NAME} deps MSI: ${STALE_DEPS_MSI}")
            file(REMOVE "${STALE_DEPS_MSI}")
        endif()
    endforeach()

    # ---- Try downloading or reusing pre-built MSI ----
    #
    # Priority order (unless OLLAMA_DEPS_FORCE_BUILD is set):
    #   1. Download from OLLAMA_DEPS_DOWNLOAD_URL (CI or local test server)
    #   2. Build locally from source. The custom command output lets the build
    #      tool skip work when the existing local MSI is still up to date.
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
                # Verify the response is a plausible MSI compound file, not
                # an HTML error body from a mirror or local test server.
                file(SIZE "${DOWNLOAD_TEMP}" DOWNLOADED_SIZE)
                file(READ "${DOWNLOAD_TEMP}" DOWNLOADED_MAGIC LIMIT 8 HEX)
                string(TOLOWER "${DOWNLOADED_MAGIC}" DOWNLOADED_MAGIC)
                if(DOWNLOADED_SIZE GREATER 1024 AND DOWNLOADED_MAGIC STREQUAL "d0cf11e0a1b11ae1")
                    file(REMOVE "${DEPS_MSI_PATH}")
                    file(RENAME "${DOWNLOAD_TEMP}" "${DEPS_MSI_PATH}")
                    message(STATUS "  Downloaded ${DEPS_MSI_NAME} (${DOWNLOADED_SIZE} bytes)")
                    set(DEPS_DOWNLOADED TRUE)
                else()
                    message(STATUS "  ${DEPS_MSI_NAME} not available from server (not a valid MSI, ${DOWNLOADED_SIZE} bytes returned)")
                    file(REMOVE "${DOWNLOAD_TEMP}")
                endif()
            else()
                message(STATUS "  Download failed: ${DOWNLOAD_ERROR_MSG}")
                file(REMOVE "${DOWNLOAD_TEMP}")
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
            COMMAND ${POWERSHELL_EXE} -NoProfile -ExecutionPolicy Bypass -File
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
            ${ARG_EXTRA_SOURCES}
            "${DEPS_WXS_DIR}/Folders.wxs"
        )

        set(WIX_ARGS build -o "${DEPS_MSI_PATH}")
        foreach(SRC ${DEPS_SOURCES})
            list(APPEND WIX_ARGS "${SRC}")
        endforeach()
        # Deps MSIs use the numeric prefix from their payload breadcrumb for
        # MSI metadata so they don't change when only the Ollama version bumps.
        list(APPEND WIX_ARGS -d DEPS_VERSION=${DEPS_MSI_VERSION})
        list(APPEND WIX_ARGS -d CABSIZEMB=${OLLAMA_CABSIZEMB})
        list(APPEND WIX_ARGS -d ROCM_BACKEND_DIR=${OLLAMA_ROCM_BACKEND_DIR})
        list(APPEND WIX_ARGS -bindpath "${DIST_AMD64}")

        add_custom_command(
            OUTPUT "${DEPS_MSI_PATH}"
            DEPENDS "${GENERATED_WXS}" ${DEPS_INPUT_FILES} ${DEPS_SOURCES} ${ARG_EXTRA_DEPENDS}
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

set(MLX_CUDA_V13_DEPS_GENERATED_WXS "${CMAKE_BINARY_DIR}/mlx-cuda-v13-deps-generated.wxs")
file(GLOB_RECURSE MLX_CUDA_V13_DEPS_INPUT_FILES LIST_DIRECTORIES false
    "${DIST_AMD64}/lib/ollama/mlx_cuda_v13/*")
list(LENGTH MLX_CUDA_V13_DEPS_INPUT_FILES MLX_CUDA_V13_DEPS_INPUT_FILE_COUNT)
set(CUDA_V13_VERSION_FILE "${DIST_AMD64}/lib/ollama/cuda_v13/cuda-version.txt")
set(MLX_CUDA_V13_VERSION_FILE "${DIST_AMD64}/lib/ollama/mlx_cuda_v13/mlx-version.txt")
set(CUDA_V13_SHARED_DEPS_VERSION "")
if(MLX_CUDA_V13_DEPS_INPUT_FILE_COUNT GREATER 0)
    ollama_read_version_breadcrumb(CUDA_V13_SHARED_DEPS_VERSION "${MLX_CUDA_V13_VERSION_FILE}" "CUDA v13 shared deps")
    if(NOT CUDA_V13_SHARED_DEPS_VERSION)
        message(FATAL_ERROR "Missing version breadcrumb for CUDA v13 shared deps: ${MLX_CUDA_V13_VERSION_FILE}")
    endif()
else()
    ollama_read_version_breadcrumb(CUDA_V13_SHARED_DEPS_VERSION "${CUDA_V13_VERSION_FILE}" "CUDA v13 deps")
endif()
set(CUDA_V13_SHARED_DEPS_ARGS)
if(CUDA_V13_SHARED_DEPS_VERSION)
    list(APPEND CUDA_V13_SHARED_DEPS_ARGS DEPS_VERSION "${CUDA_V13_SHARED_DEPS_VERSION}")
endif()
add_custom_command(
    OUTPUT "${MLX_CUDA_V13_DEPS_GENERATED_WXS}"
    COMMAND ${POWERSHELL_EXE} -NoProfile -ExecutionPolicy Bypass -File
        "${DEPS_WXS_DIR}/generate-backend-deps.ps1"
        -BackendPath "lib/ollama/mlx_cuda_v13"
        -ComponentGroupId "mlx_cuda_v13_deps_Components"
        -RootDirectoryId "lib_ollama_mlx_cuda_v13_Dir"
        -DistDir "${OLLAMA_DIST_DIR}"
        -OutputFile "${MLX_CUDA_V13_DEPS_GENERATED_WXS}"
        -ExcludeFile "mlx.dll,mlxc.dll,ollama_xgrammar.dll"
    DEPENDS "${DEPS_WXS_DIR}/generate-backend-deps.ps1" ${MLX_CUDA_V13_DEPS_INPUT_FILES}
    COMMENT "Generating MLX deps WXS for CUDA v13"
    VERBATIM
)

# CUDA v13 deps
add_deps_msi_target(msi-cuda-v13-deps "cuda-v13"
    "cuda-v13-deps.wxs" "cuda-v13-deps-generated.wxs"
    "cuda_v13_deps_Components" "lib_ollama_cuda_v13_Dir"
    "cuda_v13" "ggml-cuda*.dll" "cuda-version.txt"
    ${CUDA_V13_SHARED_DEPS_ARGS}
    EXTRA_SOURCES "${MLX_CUDA_V13_DEPS_GENERATED_WXS}"
    EXTRA_DEPENDS ${MLX_CUDA_V13_DEPS_INPUT_FILES})

# ROCm deps
add_deps_msi_target(msi-rocm-deps "rocm"
    "rocm-deps.wxs" "rocm-deps-generated.wxs"
    "rocm_deps_Components" "lib_ollama_rocm_backend_Dir"
    "${OLLAMA_ROCM_BACKEND_DIR}" "ggml-hip*.dll" "rocm-version.txt")

# Vulkan deps
add_deps_msi_target(msi-vulkan-deps "vulkan"
    "vulkan-deps.wxs" "vulkan-deps-generated.wxs"
    "vulkan_deps_Components" "lib_ollama_vulkan_Dir"
    "vulkan" "ggml-vulkan*.dll" "vulkan-version.txt")

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
    EXTRA_DEPENDS "${DIST_AMD64}/lib/ollama/${OLLAMA_ROCM_BACKEND_DIR}/ggml-hip.dll")
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
        "${DIST_AMD64}/lib/ollama/mlx_cuda_v13/mlxc.dll"
        "${DIST_AMD64}/lib/ollama/mlx_cuda_v13/ollama_xgrammar.dll")
add_dependencies(msi-mlx-cuda-v13 msi-cuda-v13-deps)

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
set(PACKAGES_JSON_INPUTS
    "${msi-cuda-v12_OUTPUT}"
    "${msi-cuda-v13_OUTPUT}"
    "${msi-rocm_OUTPUT}"
    "${msi-vulkan_OUTPUT}"
    "${msi-mlx-cuda-v13_OUTPUT}"
    "${msi-cuda-v12-deps_OUTPUT}"
    "${msi-cuda-v13-deps_OUTPUT}"
    "${msi-rocm-deps_OUTPUT}"
    "${msi-vulkan-deps_OUTPUT}")

add_custom_command(
    OUTPUT "${PACKAGES_JSON}"
    COMMAND ${POWERSHELL_EXE} -NoProfile -ExecutionPolicy Bypass -File
        "${WXS_DIR}/generate-packages-json.ps1"
        -Version "${OLLAMA_VERSION}"
        -DistDir "${OLLAMA_DIST_DIR}"
    DEPENDS ${PACKAGES_JSON_INPUTS}
    COMMENT "Generating packages.json"
    VERBATIM
)
add_custom_target(generate-packages-json DEPENDS "${PACKAGES_JSON}")

# --------------------------------------------------------------------------
# Core MSI targets
#
# Depends on packages.json (file-level) and key payload binaries so the
# MSI rebuilds when either the manifest or the binaries change.
# --------------------------------------------------------------------------

set(CORE_RUNTIME_WXS "${CMAKE_BINARY_DIR}/core-runtime-generated.wxs")
add_msi_go_license_target(msi-go-license-amd64 "amd64" "${DIST_AMD64}")
file(GLOB CORE_RUNTIME_FILES LIST_DIRECTORIES false
    "${DIST_AMD64}/lib/ollama/*")
list(APPEND CORE_RUNTIME_FILES "${msi-go-license-amd64_OUTPUT}")
set(CORE_REQUIRED_PAYLOADS
    "${DIST_AMD64}/ollama.exe"
    "${DIST_AMD64}/lib/ollama/llama-server.exe"
    "${msi-go-license-amd64_OUTPUT}"
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
    COMMAND ${POWERSHELL_EXE} -NoProfile -ExecutionPolicy Bypass -File
        "${DEPS_WXS_DIR}/generate-backend-deps.ps1"
        -BackendPath "lib/ollama"
        -ComponentGroupId "core_runtime_Components"
        -RootDirectoryId "lib_ollama_Dir"
        -DistDir "${OLLAMA_DIST_DIR}"
        -OutputFile "${CORE_RUNTIME_WXS}"
        -NoRecurse
    DEPENDS "${DEPS_WXS_DIR}/generate-backend-deps.ps1" ${CORE_RUNTIME_FILES}
        "${msi-go-license-amd64_OUTPUT}"
    COMMENT "Generating core runtime WXS"
    VERBATIM
)

add_msi_target(msi-core "ollama-core.msi"
    SOURCES "${WXS_DIR}/ollama-core.wxs" "${WXS_DIR}/Folders.wxs" "${CORE_RUNTIME_WXS}"
    BINDPATH "${DIST_AMD64}"
    DEFINES "CHAINED_UNINSTALL_ENCODED=${CHAINED_UNINSTALL_ENCODED}"
    EXTENSIONS WixToolset.Util.wixext
    EXTRA_DEPENDS "${PACKAGES_JSON}" "${DIST_AMD64}/ollama.exe"
        "${OLLAMA_DIST_DIR}/windows-ollama-app-amd64.exe" ${CORE_RUNTIME_FILES}
        "${msi-go-license-amd64_OUTPUT}")
# Target-level ordering: generate-packages-json must run before msi-core.
# The file-level dep on PACKAGES_JSON handles incremental rebuild detection
# (only rebuilds core MSI when packages.json content actually changes).
add_dependencies(msi-core generate-packages-json)
add_dependencies(check-core-payloads msi-go-license-amd64)
add_dependencies(msi-core msi-go-license-amd64)
add_dependencies(msi-core check-core-payloads)

# ARM64 core MSI (only if ARM64 binaries exist)
if(EXISTS "${DIST_ARM64}/ollama.exe")
    set(CORE_RUNTIME_ARM64_WXS "${CMAKE_BINARY_DIR}/core-runtime-arm64-generated.wxs")
    add_msi_go_license_target(msi-go-license-arm64 "arm64" "${DIST_ARM64}")
    file(GLOB CORE_RUNTIME_ARM64_FILES LIST_DIRECTORIES false
        "${DIST_ARM64}/lib/ollama/*")
    list(APPEND CORE_RUNTIME_ARM64_FILES "${msi-go-license-arm64_OUTPUT}")
    set(CORE_REQUIRED_PAYLOADS_ARM64
        "${DIST_ARM64}/ollama.exe"
        "${DIST_ARM64}/lib/ollama/llama-server.exe"
        "${msi-go-license-arm64_OUTPUT}"
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
        COMMAND ${POWERSHELL_EXE} -NoProfile -ExecutionPolicy Bypass -File
            "${DEPS_WXS_DIR}/generate-backend-deps.ps1"
            -BackendPath "lib/ollama"
            -ComponentGroupId "core_runtime_Components"
            -RootDirectoryId "lib_ollama_Dir"
            -DistDir "${OLLAMA_DIST_DIR}"
            -OutputFile "${CORE_RUNTIME_ARM64_WXS}"
            -Arch "arm64"
            -NoRecurse
        DEPENDS "${DEPS_WXS_DIR}/generate-backend-deps.ps1" ${CORE_RUNTIME_ARM64_FILES}
            "${msi-go-license-arm64_OUTPUT}"
        COMMENT "Generating ARM64 core runtime WXS"
        VERBATIM
    )

    add_msi_target(msi-core-arm64 "ollama-core-arm64.msi"
        SOURCES "${WXS_DIR}/ollama-core-arm64.wxs" "${WXS_DIR}/Folders.wxs" "${CORE_RUNTIME_ARM64_WXS}"
        BINDPATH "${DIST_ARM64}"
        DEFINES "CHAINED_UNINSTALL_ENCODED=${CHAINED_UNINSTALL_ENCODED}"
        EXTENSIONS WixToolset.Util.wixext
        EXTRA_DEPENDS "${PACKAGES_JSON}" "${DIST_ARM64}/ollama.exe"
            "${OLLAMA_DIST_DIR}/windows-ollama-app-arm64.exe" ${CORE_RUNTIME_ARM64_FILES}
            "${msi-go-license-arm64_OUTPUT}")
    add_dependencies(msi-core-arm64 generate-packages-json)
    add_dependencies(check-core-arm64-payloads msi-go-license-arm64)
    add_dependencies(msi-core-arm64 msi-go-license-arm64)
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

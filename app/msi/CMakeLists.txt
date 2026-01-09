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
# Override via environment for testing: OLLAMA_DOWNLOAD_URL=http://localhost:8000
if(DEFINED ENV{OLLAMA_DOWNLOAD_URL})
    set(OLLAMA_DOWNLOAD_URL "$ENV{OLLAMA_DOWNLOAD_URL}")
elseif(NOT OLLAMA_DOWNLOAD_URL)
    set(OLLAMA_DOWNLOAD_URL "https://ollama.com/download")
endif()
# Strip trailing slash for consistent URL construction
string(REGEX REPLACE "/+$" "" OLLAMA_DOWNLOAD_URL "${OLLAMA_DOWNLOAD_URL}")

# Output directory for built MSIs
set(MSI_OUTPUT_DIR "${OLLAMA_DIST_DIR}" CACHE PATH "Output directory for MSI files")

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
message(STATUS "Download URL: ${OLLAMA_DOWNLOAD_URL}")

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
# Helper: sign an MSI
# --------------------------------------------------------------------------

function(sign_file TARGET_NAME FILE_PATH)
    if(SIGNING_ENABLED)
        add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
            COMMAND "${SIGN_TOOL}" sign /v /fd sha256
                /t http://timestamp.digicert.com
                /f "${OLLAMA_CERT}"
                /csp "Google Cloud KMS Provider"
                /kc "$ENV{KEY_CONTAINER}"
                "${FILE_PATH}"
            COMMENT "Signing ${FILE_PATH}"
            VERBATIM
        )
    endif()
endfunction()

# --------------------------------------------------------------------------
# Helper: build a backend or core MSI from WXS sources
# --------------------------------------------------------------------------

function(add_msi_target TARGET_NAME OUTPUT_MSI WXS_SOURCES BINDPATH)
    set(MSI_PATH "${MSI_OUTPUT_DIR}/${OUTPUT_MSI}")

    # Build the wix command arguments
    set(WIX_ARGS build -o "${MSI_PATH}")
    foreach(SRC ${WXS_SOURCES})
        list(APPEND WIX_ARGS "${SRC}")
    endforeach()
    list(APPEND WIX_ARGS -d PKG_VERSION=${OLLAMA_PKG_VERSION})
    list(APPEND WIX_ARGS -ext WixToolset.Util.wixext)
    if(BINDPATH)
        list(APPEND WIX_ARGS -bindpath "${BINDPATH}")
    endif()
    # Add WXS source directory as bindpath so relative paths (e.g., ..\assets\app.ico) resolve
    list(APPEND WIX_ARGS -bindpath "${WXS_DIR}")
    # Pass any extra -d defines from ARGN
    foreach(EXTRA_DEF ${ARGN})
        list(APPEND WIX_ARGS -d "${EXTRA_DEF}")
    endforeach()

    add_custom_target(${TARGET_NAME}
        COMMAND ${WIX_EXE} ${WIX_ARGS}
        COMMENT "Building ${OUTPUT_MSI}"
        VERBATIM
    )
    sign_file(${TARGET_NAME} "${MSI_PATH}")
endfunction()

# --------------------------------------------------------------------------
# Helper: build a deps MSI (with download-or-build logic)
#
# When OLLAMA_DOWNLOAD_URL is set, tries to download a pre-built deps
# MSI from that URL before building locally. This avoids rebuilding large
# deps MSIs (CUDA SDK, ROCm, etc.) when only the Ollama code changed.
#
# Flow:
#   1. Read SDK version from breadcrumb file (e.g., "12.8.1")
#   2. Construct filename: "ollama-cuda-deps-12.8.1.msi"
#   3. If OLLAMA_DOWNLOAD_URL set: try downloading
#   4. If download succeeds: use it (skip local build)
#   5. If download fails: warn, build locally with WiX
# --------------------------------------------------------------------------

function(add_deps_msi_target TARGET_NAME BACKEND_NAME DEPS_WXS DEPS_GENERATED_WXS
         COMPONENT_GROUP_ID ROOT_DIR_ID BACKEND_PATH EXCLUDE_FILE VERSION_FILE_NAME)

    set(GENERATED_WXS "${CMAKE_BINARY_DIR}/${DEPS_GENERATED_WXS}")

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

    # ---- Try downloading pre-built MSI ----
    set(DEPS_DOWNLOADED FALSE)
    if(DEPS_VERSION)
        set(DOWNLOAD_URL "${OLLAMA_DOWNLOAD_URL}/${DEPS_MSI_NAME}")
        # Download to a temp path first -- if the server is serving from the
        # same directory as MSI_OUTPUT_DIR, file(DOWNLOAD) would truncate the
        # existing file before the server reads it (race condition).
        set(DOWNLOAD_TEMP "${CMAKE_BINARY_DIR}/_download_${DEPS_MSI_NAME}")
        message(STATUS "  Downloading ${DEPS_MSI_NAME} from ${DOWNLOAD_URL}")
        file(DOWNLOAD "${DOWNLOAD_URL}" "${DOWNLOAD_TEMP}"
            STATUS DOWNLOAD_STATUS
            TIMEOUT 120
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
                message(STATUS "  ${DEPS_MSI_NAME} not available from server (${DOWNLOADED_SIZE} bytes returned), will build locally")
                file(REMOVE "${DOWNLOAD_TEMP}")
            endif()
        else()
            message(STATUS "  Download failed for ${DEPS_MSI_NAME}: ${DOWNLOAD_ERROR_MSG}, will build locally")
            file(REMOVE "${DOWNLOAD_TEMP}")
        endif()
    else()
        message(WARNING "  Cannot download ${BACKEND_NAME} deps: no version breadcrumb")
    endif()

    # ---- Create target ----
    if(DEPS_DOWNLOADED)
        # MSI was downloaded -- target is a no-op (just sign if enabled)
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
        list(APPEND WIX_ARGS -d PKG_VERSION=${OLLAMA_PKG_VERSION})
        list(APPEND WIX_ARGS -bindpath "${DIST_AMD64}")

        add_custom_target(${TARGET_NAME}
            DEPENDS "${GENERATED_WXS}"
            COMMAND ${WIX_EXE} ${WIX_ARGS}
            COMMENT "Building ${DEPS_MSI_NAME}"
            VERBATIM
        )
    endif()
    sign_file(${TARGET_NAME} "${DEPS_MSI_PATH}")
endfunction()

# --------------------------------------------------------------------------
# Deps MSI targets
# --------------------------------------------------------------------------

# CUDA v12 deps
add_deps_msi_target(msi-cuda-v12-deps "cuda"
    "cuda-v12-deps.wxs" "cuda-v12-deps-generated.wxs"
    "cuda_v12_deps_Components" "lib_ollama_cuda_v12_Dir"
    "cuda_v12" "ggml-cuda.dll" "cuda-version.txt")

# CUDA v13 deps
add_deps_msi_target(msi-cuda-v13-deps "cuda-v13"
    "cuda-v13-deps.wxs" "cuda-v13-deps-generated.wxs"
    "cuda_v13_deps_Components" "lib_ollama_cuda_v13_Dir"
    "cuda_v13" "ggml-cuda.dll" "cuda-version.txt")

# ROCm deps
add_deps_msi_target(msi-rocm-deps "rocm"
    "rocm-deps.wxs" "rocm-deps-generated.wxs"
    "rocm_deps_Components" "lib_ollama_rocm_Dir"
    "rocm" "ggml-hip.dll" "rocm-version.txt")

# Vulkan deps
add_deps_msi_target(msi-vulkan-deps "vulkan"
    "vulkan-deps.wxs" "vulkan-deps-generated.wxs"
    "vulkan_deps_Components" "lib_ollama_vulkan_Dir"
    "vulkan" "ggml-vulkan.dll" "vulkan-version.txt")

# --------------------------------------------------------------------------
# Backend MSI targets (depend on their deps)
# --------------------------------------------------------------------------

# CUDA v12
add_msi_target(msi-cuda-v12 "ollama-cuda-v12.msi"
    "${WXS_DIR}/cuda-v12.wxs;${WXS_DIR}/Folders.wxs"
    "${DIST_AMD64}")
add_dependencies(msi-cuda-v12 msi-cuda-v12-deps)

# CUDA v13
add_msi_target(msi-cuda-v13 "ollama-cuda-v13.msi"
    "${WXS_DIR}/cuda-v13.wxs;${WXS_DIR}/Folders.wxs"
    "${DIST_AMD64}")
add_dependencies(msi-cuda-v13 msi-cuda-v13-deps)

# ROCm
add_msi_target(msi-rocm "ollama-rocm.msi"
    "${WXS_DIR}/rocm.wxs;${WXS_DIR}/Folders.wxs"
    "${DIST_AMD64}")
add_dependencies(msi-rocm msi-rocm-deps)

# Vulkan
add_msi_target(msi-vulkan "ollama-vulkan.msi"
    "${WXS_DIR}/vulkan.wxs;${WXS_DIR}/Folders.wxs"
    "${DIST_AMD64}")
add_dependencies(msi-vulkan msi-vulkan-deps)

# --------------------------------------------------------------------------
# Generate packages.json (depends on all backend + deps MSIs being built and signed)
# --------------------------------------------------------------------------

add_custom_target(generate-packages-json
    COMMAND ${POWERSHELL_EXE} -ExecutionPolicy Bypass -File
        "${WXS_DIR}/generate-packages-json.ps1"
        -Version "${OLLAMA_VERSION}"
        -DistDir "${OLLAMA_DIST_DIR}"
    COMMENT "Generating packages.json"
    VERBATIM
)
add_dependencies(generate-packages-json
    msi-cuda-v12 msi-cuda-v13 msi-rocm msi-vulkan
    msi-cuda-v12-deps msi-cuda-v13-deps msi-rocm-deps msi-vulkan-deps
)

# --------------------------------------------------------------------------
# Core MSI targets (depend on packages.json)
# --------------------------------------------------------------------------

add_msi_target(msi-core "ollama-core.msi"
    "${WXS_DIR}/ollama-core.wxs;${WXS_DIR}/Folders.wxs"
    "${DIST_AMD64}"
    "CHAINED_UNINSTALL_ENCODED=${CHAINED_UNINSTALL_ENCODED}")
add_dependencies(msi-core generate-packages-json)

# ARM64 core MSI (only if ARM64 binaries exist)
if(EXISTS "${DIST_ARM64}/ollama.exe")
    add_msi_target(msi-core-arm64 "ollama-core-arm64.msi"
        "${WXS_DIR}/ollama-core-arm64.wxs;${WXS_DIR}/Folders.wxs"
        "${DIST_ARM64}"
        "CHAINED_UNINSTALL_ENCODED=${CHAINED_UNINSTALL_ENCODED}")
    add_dependencies(msi-core-arm64 generate-packages-json)
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
# Sign install.ps1 (if signing is enabled)
# --------------------------------------------------------------------------

set(INSTALL_SCRIPT_SRC "${CMAKE_SOURCE_DIR}/../../scripts/install.ps1")
if(EXISTS "${INSTALL_SCRIPT_SRC}")
    set(INSTALL_SCRIPT_DIST "${MSI_OUTPUT_DIR}/install.ps1")
    add_custom_target(stage-install-script
        COMMAND ${CMAKE_COMMAND} -E copy "${INSTALL_SCRIPT_SRC}" "${INSTALL_SCRIPT_DIST}"
        COMMENT "Staging install.ps1 to dist"
        VERBATIM
    )
    if(SIGNING_ENABLED)
        add_custom_target(sign-install-script
            COMMAND "${SIGN_TOOL}" sign /v /fd sha256
                /t http://timestamp.digicert.com
                /f "${OLLAMA_CERT}"
                /csp "Google Cloud KMS Provider"
                /kc "$ENV{KEY_CONTAINER}"
                "${INSTALL_SCRIPT_DIST}"
            COMMENT "Signing install.ps1 in dist"
            VERBATIM
        )
        add_dependencies(sign-install-script stage-install-script)
        add_dependencies(msi-all sign-install-script)
    else()
        add_dependencies(msi-all stage-install-script)
    endif()
endif()

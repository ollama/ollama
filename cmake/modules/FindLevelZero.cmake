# SPDX-License-Identifier: MIT
# FindLevelZero.cmake — Locate Intel Level Zero headers for the Ollama
# GGML Level Zero backend.
#
# CRITICAL (INFO-1 / ADR-L0-005 dlopen discipline):
#   This module intentionally sets ONLY the include path.
#   LevelZero_LIBRARIES is left UNSET to enforce dlopen-only linking.
#   The backend (libggml-level-zero) must NOT link against ze_loader at
#   compile time; it resolves the loader via dlopen at runtime.
#
# Imported target created (if found):
#   LevelZero::LevelZero  — INTERFACE library carrying include directories only.
#
# Variables set:
#   LevelZero_FOUND            — TRUE when headers are located
#   LevelZero_INCLUDE_DIRS     — Path to level_zero/ headers
#   LevelZero_LIBRARIES        — Intentionally UNSET (dlopen discipline)
#
# Search order:
#   1. LEVEL_ZERO_ROOT env var                  (user override, both platforms)
#   2. pkg-config level-zero                    (Linux / macOS)
#   3. /usr/include/level_zero                  (Linux fallback)
#   4. ONEAPI_ROOT\compiler\latest\include      (Windows, Intel oneAPI)
#   5. Standard CMake find_path search paths

cmake_minimum_required(VERSION 3.21)

# -- 1. Honour user-provided root override on both platforms -----------------
if(DEFINED ENV{LEVEL_ZERO_ROOT})
    set(_lz_user_root "$ENV{LEVEL_ZERO_ROOT}")
endif()

# -- 2. Try pkg-config first on Linux / macOS --------------------------------
if(NOT WIN32)
    find_package(PkgConfig QUIET)
    if(PKG_CONFIG_FOUND)
        pkg_check_modules(_LZ_PC QUIET level-zero)
    endif()
endif()

# -- 3. Locate ze_api.h -------------------------------------------------------
find_path(LevelZero_INCLUDE_DIRS
    NAMES level_zero/ze_api.h
    HINTS
        ${_lz_user_root}/include
        ${_LZ_PC_INCLUDE_DIRS}
    PATHS
        # Linux standard paths
        /usr/include
        /usr/local/include
        # Windows Intel oneAPI installation paths
        "$ENV{ONEAPI_ROOT}/compiler/latest/include"
        "$ENV{ONEAPI_ROOT}/include"
        # Common Windows SDK layout
        "C:/Program Files (x86)/Intel/oneAPI/compiler/latest/include"
        "C:/Program Files/Intel/oneAPI/compiler/latest/include"
    PATH_SUFFIXES
        ""
    NO_DEFAULT_PATH
)

# Retry with default CMake paths if the restricted search failed
if(NOT LevelZero_INCLUDE_DIRS)
    find_path(LevelZero_INCLUDE_DIRS
        NAMES level_zero/ze_api.h
    )
endif()

# -- 4. Report result ---------------------------------------------------------
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LevelZero
    REQUIRED_VARS LevelZero_INCLUDE_DIRS
    FAIL_MESSAGE  "Intel Level Zero headers not found. Install level-zero-dev (Linux) or Intel oneAPI Base Toolkit (Windows). Set LEVEL_ZERO_ROOT to override the search path."
)

mark_as_advanced(LevelZero_INCLUDE_DIRS)

# LevelZero_LIBRARIES is intentionally left unset (dlopen discipline).
# Explicitly clear it so downstream consumers cannot accidentally link.
unset(LevelZero_LIBRARIES)
unset(LevelZero_LIBRARIES CACHE)

# -- 5. Create INTERFACE imported target -------------------------------------
if(LevelZero_FOUND AND NOT TARGET LevelZero::LevelZero)
    add_library(LevelZero::LevelZero INTERFACE IMPORTED)
    set_target_properties(LevelZero::LevelZero PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${LevelZero_INCLUDE_DIRS}"
    )
    # No INTERFACE_LINK_LIBRARIES — dlopen discipline enforced here.
endif()

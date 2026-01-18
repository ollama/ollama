# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

include(${CMAKE_ROOT}/Modules/CMakeDetermineCompiler.cmake)
include(${CMAKE_ROOT}/Modules/CMakeParseImplicitLinkInfo.cmake)
include(${CMAKE_ROOT}/Modules/CMakeParseLibraryArchitecture.cmake)

if(NOT ((CMAKE_GENERATOR MATCHES "Make") OR
        (CMAKE_GENERATOR MATCHES "Ninja")))
  message(FATAL_ERROR "HIP language not currently supported by \"${CMAKE_GENERATOR}\" generator")
endif()

if(NOT CMAKE_HIP_PLATFORM)
  execute_process(COMMAND hipconfig --platform
    OUTPUT_VARIABLE _CMAKE_HIPCONFIG_PLATFORM OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _CMAKE_HIPCONFIG_RESULT
    )
  if(_CMAKE_HIPCONFIG_RESULT EQUAL 0 AND _CMAKE_HIPCONFIG_PLATFORM MATCHES "^(nvidia|nvcc)$")
    set(CMAKE_HIP_PLATFORM "nvidia" CACHE STRING "HIP platform" FORCE)
  else()
    set(CMAKE_HIP_PLATFORM "amd" CACHE STRING "HIP platform" FORCE)
  endif()
endif()
if(NOT CMAKE_HIP_PLATFORM MATCHES "^(amd|nvidia)$")
  message(FATAL_ERROR
    "The CMAKE_HIP_PLATFORM has unsupported value:\n"
    " '${CMAKE_HIP_PLATFORM}'\n"
    "It must be 'amd' or 'nvidia'."
    )
endif()

if(NOT CMAKE_HIP_COMPILER)
  set(CMAKE_HIP_COMPILER_INIT NOTFOUND)

  # prefer the environment variable HIPCXX
  if(NOT $ENV{HIPCXX} STREQUAL "")
    get_filename_component(CMAKE_HIP_COMPILER_INIT $ENV{HIPCXX} PROGRAM PROGRAM_ARGS CMAKE_HIP_FLAGS_ENV_INIT)
    if(CMAKE_HIP_FLAGS_ENV_INIT)
      set(CMAKE_HIP_COMPILER_ARG1 "${CMAKE_HIP_FLAGS_ENV_INIT}" CACHE STRING "Arguments to CXX compiler")
    endif()
    if(NOT EXISTS ${CMAKE_HIP_COMPILER_INIT})
      message(FATAL_ERROR "Could not find compiler set in environment variable HIPCXX:\n$ENV{HIPCXX}.\n${CMAKE_HIP_COMPILER_INIT}")
    endif()
  endif()

  # finally list compilers to try
  if(NOT CMAKE_HIP_COMPILER_INIT)
    if(CMAKE_HIP_PLATFORM STREQUAL "nvidia")
      set(CMAKE_HIP_COMPILER_LIST nvcc)
    elseif(CMAKE_HIP_PLATFORM STREQUAL "amd")
      set(CMAKE_HIP_COMPILER_LIST clang++)

      # Look for the Clang coming with ROCm to support HIP.
      execute_process(COMMAND hipconfig --hipclangpath
        OUTPUT_VARIABLE _CMAKE_HIPCONFIG_CLANGPATH
        RESULT_VARIABLE _CMAKE_HIPCONFIG_RESULT
      )
      if(_CMAKE_HIPCONFIG_RESULT EQUAL 0 AND EXISTS "${_CMAKE_HIPCONFIG_CLANGPATH}")
        set(CMAKE_HIP_COMPILER_HINTS "${_CMAKE_HIPCONFIG_CLANGPATH}")
      endif()
    endif()
  endif()

  _cmake_find_compiler(HIP)
else()
  _cmake_find_compiler_path(HIP)
endif()

mark_as_advanced(CMAKE_HIP_COMPILER)

# Build a small source file to identify the compiler.
if(NOT CMAKE_HIP_COMPILER_ID_RUN)
  set(CMAKE_HIP_COMPILER_ID_RUN 1)

  include(${CMAKE_ROOT}/Modules/CMakeDetermineCompilerId.cmake)

  # We determine the vendor to use the right flags for detection right away.
  # The main compiler identification is still needed below to extract other information.
  list(APPEND CMAKE_HIP_COMPILER_ID_VENDORS NVIDIA Clang)
  set(CMAKE_HIP_COMPILER_ID_VENDOR_REGEX_NVIDIA "nvcc: NVIDIA \\(R\\) Cuda compiler driver")
  set(CMAKE_HIP_COMPILER_ID_VENDOR_REGEX_Clang "(clang version)")
  CMAKE_DETERMINE_COMPILER_ID_VENDOR(HIP "--version")

  if(CMAKE_HIP_COMPILER_ID STREQUAL "NVIDIA")
    # Find the CUDA toolkit to get:
    # - CMAKE_HIP_COMPILER_CUDA_TOOLKIT_VERSION
    # - CMAKE_HIP_COMPILER_CUDA_TOOLKIT_ROOT
    # - CMAKE_HIP_COMPILER_CUDA_LIBRARY_ROOT
    # We save them in CMakeHIPCompiler.cmake.
    # Match arguments with cmake_cuda_architectures_all call.
    include(Internal/CMakeCUDAFindToolkit)
    cmake_cuda_find_toolkit(HIP CMAKE_HIP_COMPILER_CUDA_)

    # If the user set CMAKE_HIP_ARCHITECTURES, validate its value.
    include(Internal/CMakeCUDAArchitecturesValidate)
    cmake_cuda_architectures_validate(HIP)

    if(NOT CMAKE_HIP_HOST_COMPILER AND NOT $ENV{HIPHOSTCXX} STREQUAL "")
      get_filename_component(CMAKE_HIP_HOST_COMPILER $ENV{HIPHOSTCXX} PROGRAM)
      if(NOT EXISTS "${CMAKE_HIP_HOST_COMPILER}")
        message(FATAL_ERROR "Could not find compiler set in environment variable HIPHOSTCXX:\n$ENV{HIPHOSTCXX}.\n${CMAKE_HIP_HOST_COMPILER}")
      endif()
    endif()
  endif()

  if(CMAKE_HIP_COMPILER_ID STREQUAL "Clang")
    list(APPEND CMAKE_HIP_COMPILER_ID_TEST_FLAGS_FIRST "-v")
  elseif(CMAKE_HIP_COMPILER_ID STREQUAL "NVIDIA")
    # Tell nvcc to treat .hip files as CUDA sources.
    list(APPEND CMAKE_HIP_COMPILER_ID_TEST_FLAGS_FIRST "-x cu -v")
    if(CMAKE_HIP_HOST_COMPILER)
      string(APPEND CMAKE_HIP_COMPILER_ID_TEST_FLAGS_FIRST " -ccbin=\"${CMAKE_HIP_HOST_COMPILER}\"")
    endif()
  endif()

  # We perform compiler identification for a second time to extract implicit linking info.
  # We need to unset the compiler ID otherwise CMAKE_DETERMINE_COMPILER_ID() doesn't work.
  set(CMAKE_HIP_COMPILER_ID)
  set(CMAKE_HIP_PLATFORM_ID)
  file(READ ${CMAKE_ROOT}/Modules/CMakePlatformId.h.in
    CMAKE_HIP_COMPILER_ID_PLATFORM_CONTENT)

  CMAKE_DETERMINE_COMPILER_ID(HIP HIPFLAGS CMakeHIPCompilerId.hip)

  if(CMAKE_HIP_COMPILER_ID STREQUAL "NVIDIA")
    include(Internal/CMakeCUDAArchitecturesAll)
    # From CMAKE_HIP_COMPILER_CUDA_TOOLKIT_VERSION and CMAKE_HIP_COMPILER_{ID,VERSION}, get:
    # - CMAKE_HIP_ARCHITECTURES_ALL
    # - CMAKE_HIP_ARCHITECTURES_ALL_MAJOR
    # Match arguments with cmake_cuda_find_toolkit call.
    cmake_cuda_architectures_all(HIP CMAKE_HIP_COMPILER_CUDA_)
  endif()

  _cmake_find_compiler_sysroot(HIP)
endif()

if(NOT CMAKE_HIP_COMPILER_ROCM_ROOT AND CMAKE_HIP_COMPILER_ID STREQUAL "Clang")
   execute_process(COMMAND "${CMAKE_HIP_COMPILER}" -v -print-targets
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _CMAKE_HIP_COMPILER_RESULT
    OUTPUT_VARIABLE _CMAKE_HIP_COMPILER_STDOUT
    ERROR_VARIABLE _CMAKE_HIP_COMPILER_STDERR
    )

  if(_CMAKE_HIP_COMPILER_RESULT EQUAL 0 AND _CMAKE_HIP_COMPILER_STDERR MATCHES "Found HIP installation: *([^,]*)[,\n]")
    set(CMAKE_HIP_COMPILER_ROCM_ROOT "${CMAKE_MATCH_1}")
    file(TO_CMAKE_PATH "${CMAKE_HIP_COMPILER_ROCM_ROOT}" CMAKE_HIP_COMPILER_ROCM_ROOT)
  endif()
endif()
if(NOT CMAKE_HIP_COMPILER_ROCM_ROOT)
  execute_process(
    COMMAND hipconfig --rocmpath
    OUTPUT_VARIABLE _CMAKE_HIPCONFIG_ROCMPATH
    RESULT_VARIABLE _CMAKE_HIPCONFIG_RESULT
    )
  if(_CMAKE_HIPCONFIG_RESULT EQUAL 0 AND EXISTS "${_CMAKE_HIPCONFIG_ROCMPATH}")
    set(CMAKE_HIP_COMPILER_ROCM_ROOT "${_CMAKE_HIPCONFIG_ROCMPATH}")
  endif()
endif()
if(NOT CMAKE_HIP_COMPILER_ROCM_ROOT)
  message(FATAL_ERROR "Failed to find ROCm root directory.")
endif()

if(CMAKE_HIP_PLATFORM STREQUAL "amd")
  # For this platform we need the hip-lang cmake package.

  # Normally implicit link information is not detected until ABI detection,
  # but we need to populate CMAKE_HIP_LIBRARY_ARCHITECTURE to find hip-lang.
  cmake_parse_implicit_link_info("${CMAKE_HIP_COMPILER_PRODUCED_OUTPUT}"
    _CMAKE_HIP_COMPILER_ID_IMPLICIT_LIBS
    _CMAKE_HIP_COMPILER_ID_IMPLICIT_DIRS
    _CMAKE_HIP_COMPILER_ID_IMPLICIT_FWKS
    _CMAKE_HIP_COMPILER_ID_IMPLICIT_LOG
    "" LANGUAGE HIP)
  message(CONFIGURE_LOG
    "Parsed HIP implicit link information from compiler id output:\n${_CMAKE_HIP_COMPILER_ID_IMPLICIT_LOG}\n\n")
  cmake_parse_library_architecture(HIP "${_CMAKE_HIP_COMPILER_ID_IMPLICIT_DIRS}" "" CMAKE_HIP_LIBRARY_ARCHITECTURE)
  if(CMAKE_HIP_LIBRARY_ARCHITECTURE)
    message(CONFIGURE_LOG
      "Parsed HIP library architecture from compiler id output: ${CMAKE_HIP_LIBRARY_ARCHITECTURE}\n")
  endif()
  unset(_CMAKE_HIP_COMPILER_ID_IMPLICIT_LIBS)
  unset(_CMAKE_HIP_COMPILER_ID_IMPLICIT_DIRS)
  unset(_CMAKE_HIP_COMPILER_ID_IMPLICIT_FWKS)
  unset(_CMAKE_HIP_COMPILER_ID_IMPLICIT_LOG)

  if(NOT CMAKE_HIP_COMPILER_ROCM_LIB)
    set(_CMAKE_HIP_COMPILER_ROCM_LIB_DIRS
      "${CMAKE_HIP_COMPILER_ROCM_ROOT}/lib"
      "${CMAKE_HIP_COMPILER_ROCM_ROOT}/lib64"
      )
    if(CMAKE_HIP_LIBRARY_ARCHITECTURE)
      list(APPEND _CMAKE_HIP_COMPILER_ROCM_LIB_DIRS "${CMAKE_HIP_COMPILER_ROCM_ROOT}/lib/${CMAKE_HIP_LIBRARY_ARCHITECTURE}")
    endif()
    foreach(dir IN LISTS _CMAKE_HIP_COMPILER_ROCM_LIB_DIRS)
      if(EXISTS "${dir}/cmake/hip-lang/hip-lang-config.cmake")
        set(CMAKE_HIP_COMPILER_ROCM_LIB "${dir}")
        break()
      endif()
    endforeach()
    if(NOT CMAKE_HIP_COMPILER_ROCM_LIB)
      list(TRANSFORM _CMAKE_HIP_COMPILER_ROCM_LIB_DIRS APPEND "/cmake/hip-lang/hip-lang-config.cmake")
      string(REPLACE ";" "\n " _CMAKE_HIP_COMPILER_ROCM_LIB_DIRS "${_CMAKE_HIP_COMPILER_ROCM_LIB_DIRS}")
      message(FATAL_ERROR
        "The ROCm root directory:\n"
        " ${CMAKE_HIP_COMPILER_ROCM_ROOT}\n"
        "does not contain the HIP runtime CMake package, expected at one of:\n"
        " ${_CMAKE_HIP_COMPILER_ROCM_LIB_DIRS}\n"
        )
    endif()
    unset(_CMAKE_HIP_COMPILER_ROCM_LIB_DIRS)
  endif()
  if(NOT DEFINED CMAKE_SIZEOF_VOID_P)
    # We have not yet determined the target ABI but we need 'find_package' to
    # search lib64 directories to find hip-lang CMake package dependencies.
    # This will be replaced by ABI detection later.
    set(CMAKE_HIP_SIZEOF_DATA_PTR 8)
  endif()
endif()

if (NOT _CMAKE_TOOLCHAIN_LOCATION)
  get_filename_component(_CMAKE_TOOLCHAIN_LOCATION "${CMAKE_HIP_COMPILER}" PATH)
endif ()

set(_CMAKE_PROCESSING_LANGUAGE "HIP")
include(CMakeFindBinUtils)
include(Compiler/${CMAKE_HIP_COMPILER_ID}-FindBinUtils OPTIONAL)
unset(_CMAKE_PROCESSING_LANGUAGE)

if(CMAKE_HIP_COMPILER_ID STREQUAL "Clang")
  set(CMAKE_HIP_RUNTIME_LIBRARY_DEFAULT "SHARED")
elseif(CMAKE_HIP_COMPILER_ID STREQUAL "NVIDIA")
  include(Internal/CMakeNVCCParseImplicitInfo)
  # Parse CMAKE_HIP_COMPILER_PRODUCED_OUTPUT to get:
  # - CMAKE_HIP_ARCHITECTURES_DEFAULT
  # - CMAKE_HIP_HOST_IMPLICIT_LINK_DIRECTORIES
  # - CMAKE_HIP_HOST_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES
  # - CMAKE_HIP_HOST_IMPLICIT_LINK_LIBRARIES
  # - CMAKE_HIP_HOST_LINK_LAUNCHER
  # - CMAKE_HIP_RUNTIME_LIBRARY_DEFAULT
  # - CMAKE_HIP_CUDA_TOOLKIT_INCLUDE_DIRECTORIES
  # Match arguments with cmake_nvcc_filter_implicit_info call in CMakeTestHIPCompiler.
  cmake_nvcc_parse_implicit_info(HIP CMAKE_HIP_CUDA_)

  include(Internal/CMakeCUDAFilterImplicitLibs)
  # Filter out implicit link libraries that should not be passed unconditionally.
  cmake_cuda_filter_implicit_libs(CMAKE_HIP_HOST_IMPLICIT_LINK_LIBRARIES)
endif()

if(CMAKE_HIP_COMPILER_SYSROOT)
  string(CONCAT _SET_CMAKE_HIP_COMPILER_SYSROOT
    "set(CMAKE_HIP_COMPILER_SYSROOT \"${CMAKE_HIP_COMPILER_SYSROOT}\")\n"
    "set(CMAKE_COMPILER_SYSROOT \"${CMAKE_HIP_COMPILER_SYSROOT}\")")
else()
  set(_SET_CMAKE_HIP_COMPILER_SYSROOT "")
endif()

if(CMAKE_HIP_COMPILER_ARCHITECTURE_ID)
  set(_SET_CMAKE_HIP_COMPILER_ARCHITECTURE_ID
    "set(CMAKE_HIP_COMPILER_ARCHITECTURE_ID ${CMAKE_HIP_COMPILER_ARCHITECTURE_ID})")
else()
  set(_SET_CMAKE_HIP_COMPILER_ARCHITECTURE_ID "")
endif()

if(MSVC_HIP_ARCHITECTURE_ID)
  set(SET_MSVC_HIP_ARCHITECTURE_ID
    "set(MSVC_HIP_ARCHITECTURE_ID ${MSVC_HIP_ARCHITECTURE_ID})")
endif()

if(CMAKE_HIP_COMPILER_ID STREQUAL "NVIDIA")
  if(NOT "$ENV{CUDAARCHS}" STREQUAL "")
    set(CMAKE_HIP_ARCHITECTURES "$ENV{CUDAARCHS}" CACHE STRING "CUDA architectures")
  endif()

  # If the user did not set CMAKE_HIP_ARCHITECTURES, use the compiler's default.
  if("${CMAKE_HIP_ARCHITECTURES}" STREQUAL "")
    set(CMAKE_HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES_DEFAULT}" CACHE STRING "HIP architectures" FORCE)
    if(NOT CMAKE_HIP_ARCHITECTURES)
      message(FATAL_ERROR "Failed to detect a default HIP architecture.\n\nCompiler output:\n${CMAKE_HIP_COMPILER_PRODUCED_OUTPUT}")
    endif()
  endif()
  unset(CMAKE_HIP_ARCHITECTURES_DEFAULT)
elseif(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
  # Use 'rocm_agent_enumerator' to get the current GPU architecture.
  set(_CMAKE_HIP_ARCHITECTURES)
  find_program(_CMAKE_HIP_ROCM_AGENT_ENUMERATOR
    NAMES rocm_agent_enumerator
    HINTS "${CMAKE_HIP_COMPILER_ROCM_ROOT}/bin"
    NO_CACHE)
  if(_CMAKE_HIP_ROCM_AGENT_ENUMERATOR)
    execute_process(COMMAND "${_CMAKE_HIP_ROCM_AGENT_ENUMERATOR}" -t GPU
      RESULT_VARIABLE _CMAKE_ROCM_AGENT_ENUMERATOR_RESULT
      OUTPUT_VARIABLE _CMAKE_ROCM_AGENT_ENUMERATOR_STDOUT
      ERROR_VARIABLE  _CMAKE_ROCM_AGENT_ENUMERATOR_STDERR
    )
    if(_CMAKE_ROCM_AGENT_ENUMERATOR_RESULT EQUAL 0)
      separate_arguments(_hip_archs NATIVE_COMMAND "${_CMAKE_ROCM_AGENT_ENUMERATOR_STDOUT}")
      foreach(_hip_arch ${_hip_archs})
        if(_hip_arch STREQUAL "gfx000")
          continue()
        endif()
        string(FIND ${_hip_arch} ":" pos)
        if(NOT pos STREQUAL "-1")
          string(SUBSTRING ${_hip_arch} 0 ${pos} _hip_arch)
        endif()
        list(APPEND _CMAKE_HIP_ARCHITECTURES "${_hip_arch}")
      endforeach()
    endif()
    unset(_CMAKE_ROCM_AGENT_ENUMERATOR_RESULT)
    unset(_CMAKE_ROCM_AGENT_ENUMERATOR_STDOUT)
    unset(_CMAKE_ROCM_AGENT_ENUMERATOR_STDERR)
  endif()
  unset(_CMAKE_HIP_ROCM_AGENT_ENUMERATOR)
  if(_CMAKE_HIP_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES "${_CMAKE_HIP_ARCHITECTURES}" CACHE STRING "HIP architectures")
  elseif(CMAKE_HIP_COMPILER_PRODUCED_OUTPUT MATCHES " -target-cpu ([a-z0-9]+) ")
    set(CMAKE_HIP_ARCHITECTURES "${CMAKE_MATCH_1}" CACHE STRING "HIP architectures")
  else()
    message(FATAL_ERROR "Failed to find a default HIP architecture.")
  endif()
  unset(_CMAKE_HIP_ARCHITECTURES)
endif()

# configure variables set in this file for fast reload later on
configure_file(${CMAKE_ROOT}/Modules/CMakeHIPCompiler.cmake.in
  ${CMAKE_PLATFORM_INFO_DIR}/CMakeHIPCompiler.cmake
  @ONLY
  )
set(CMAKE_HIP_COMPILER_ENV_VAR "HIPCXX")

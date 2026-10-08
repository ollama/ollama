include_guard(GLOBAL)

set(OLLAMA_WINDOWS_RUNTIME_DEP_INCLUDE_REGEXES)
set(OLLAMA_WINDOWS_RUNTIME_DEP_DIRS)

if(WIN32)
    set(OLLAMA_WINDOWS_RUNTIME_DEP_INCLUDE_REGEXES
        "msvcp.*\\.dll"
        "vcruntime.*\\.dll"
        "concrt.*\\.dll"
        "vcomp.*\\.dll"
        "libc\\+\\+.*\\.dll"
        "libstdc\\+\\+.*\\.dll"
        "libgcc_s.*\\.dll"
        "libunwind.*\\.dll"
        "libomp.*\\.dll"
        "libwinpthread.*\\.dll"
    )

    if(DEFINED OLLAMA_WINDOWS_RUNTIME_ARCH)
        set(_ollama_windows_runtime_arch "${OLLAMA_WINDOWS_RUNTIME_ARCH}")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(ARM64|arm64|aarch64)$")
        set(_ollama_windows_runtime_arch "arm64")
    else()
        set(_ollama_windows_runtime_arch "x64")
    endif()

    set(_ollama_msvc_redist_roots)
    if(DEFINED ENV{VCToolsRedistDir})
        list(APPEND _ollama_msvc_redist_roots "$ENV{VCToolsRedistDir}")
    endif()
    if(DEFINED ENV{VCINSTALLDIR})
        list(APPEND _ollama_msvc_redist_roots "$ENV{VCINSTALLDIR}/Redist/MSVC")
    endif()
    if(DEFINED ENV{VSINSTALLDIR})
        list(APPEND _ollama_msvc_redist_roots "$ENV{VSINSTALLDIR}/VC/Redist/MSVC")
    endif()
    if(CMAKE_GENERATOR_INSTANCE)
        list(APPEND _ollama_msvc_redist_roots "${CMAKE_GENERATOR_INSTANCE}/VC/Redist/MSVC")
    endif()
    if(CMAKE_CXX_COMPILER)
        cmake_path(GET CMAKE_CXX_COMPILER PARENT_PATH _ollama_msvc_bin_dir)
        cmake_path(GET _ollama_msvc_bin_dir PARENT_PATH _ollama_tmp)
        cmake_path(GET _ollama_tmp PARENT_PATH _ollama_tmp)
        cmake_path(GET _ollama_tmp PARENT_PATH _ollama_msvc_ver_dir)
        list(APPEND _ollama_msvc_redist_roots "${_ollama_msvc_ver_dir}/../../../Redist/MSVC")
    endif()

    set(_ollama_vswhere "$ENV{ProgramFiles\(x86\)}/Microsoft Visual Studio/Installer/vswhere.exe")
    if(EXISTS "${_ollama_vswhere}")
        execute_process(
            COMMAND "${_ollama_vswhere}" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Redist.14.Latest -property installationPath
            OUTPUT_VARIABLE _ollama_vs_install_dir
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(_ollama_vs_install_dir)
            list(APPEND _ollama_msvc_redist_roots "${_ollama_vs_install_dir}/VC/Redist/MSVC")
        endif()
    endif()

    set(_ollama_vc_crt_dirs)
    set(_ollama_vc_openmp_dirs)
    foreach(_ollama_redist_root IN LISTS _ollama_msvc_redist_roots)
        file(TO_CMAKE_PATH "${_ollama_redist_root}" _ollama_redist_root)
        file(GLOB _ollama_root_crt_dirs LIST_DIRECTORIES true
            "${_ollama_redist_root}/${_ollama_windows_runtime_arch}/Microsoft.VC*.CRT"
            "${_ollama_redist_root}/*/${_ollama_windows_runtime_arch}/Microsoft.VC*.CRT")
        file(GLOB _ollama_root_openmp_dirs LIST_DIRECTORIES true
            "${_ollama_redist_root}/${_ollama_windows_runtime_arch}/Microsoft.VC*.OPENMP"
            "${_ollama_redist_root}/*/${_ollama_windows_runtime_arch}/Microsoft.VC*.OPENMP")
        list(APPEND _ollama_vc_crt_dirs ${_ollama_root_crt_dirs})
        list(APPEND _ollama_vc_openmp_dirs ${_ollama_root_openmp_dirs})
    endforeach()

    function(_ollama_latest_runtime_dir out_var)
        set(_dirs ${ARGN})
        if(_dirs)
            list(REMOVE_DUPLICATES _dirs)
            list(SORT _dirs COMPARE NATURAL ORDER DESCENDING)
            list(GET _dirs 0 _latest_dir)
            set(${out_var} "${_latest_dir}" PARENT_SCOPE)
        else()
            set(${out_var} "" PARENT_SCOPE)
        endif()
    endfunction()

    _ollama_latest_runtime_dir(_ollama_vc_crt_dir ${_ollama_vc_crt_dirs})
    _ollama_latest_runtime_dir(_ollama_vc_openmp_dir ${_ollama_vc_openmp_dirs})

    list(APPEND OLLAMA_WINDOWS_RUNTIME_DEP_DIRS
        ${_ollama_vc_crt_dir}
        ${_ollama_vc_openmp_dir})
    list(REMOVE_ITEM OLLAMA_WINDOWS_RUNTIME_DEP_DIRS "")
    list(REMOVE_DUPLICATES OLLAMA_WINDOWS_RUNTIME_DEP_DIRS)

    if(OLLAMA_WINDOWS_RUNTIME_DEP_DIRS)
        message(STATUS "Windows runtime DLL search dirs: ${OLLAMA_WINDOWS_RUNTIME_DEP_DIRS}")
    else()
        message(WARNING "Could not find Windows runtime redistributable directories for ${_ollama_windows_runtime_arch}")
    endif()
endif()

function(ollama_install_windows_runtime_dlls dest)
    if(NOT WIN32)
        return()
    endif()

    cmake_parse_arguments(ARG "" "COMPONENT" "" ${ARGN})

    set(_runtime_dlls)
    foreach(_runtime_dir IN LISTS OLLAMA_WINDOWS_RUNTIME_DEP_DIRS)
        if(EXISTS "${_runtime_dir}")
            file(GLOB _dir_runtime_dlls LIST_DIRECTORIES false "${_runtime_dir}/*.dll")
            list(APPEND _runtime_dlls ${_dir_runtime_dlls})
        endif()
    endforeach()

    if(_runtime_dlls)
        list(REMOVE_DUPLICATES _runtime_dlls)
        set(_component_args)
        if(ARG_COMPONENT)
            set(_component_args COMPONENT "${ARG_COMPONENT}")
        endif()
        install(FILES ${_runtime_dlls}
            DESTINATION "${dest}"
            ${_component_args})
    else()
        message(WARNING "Could not find Windows runtime DLLs to bundle for ${dest}")
    endif()
endfunction()

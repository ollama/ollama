set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR ARM64)

set(_ollama_llvm_mingw_hints)
if(DEFINED ENV{ProgramFiles})
    file(GLOB _ollama_program_files_llvm_mingw_bins
        LIST_DIRECTORIES true
        "$ENV{ProgramFiles}/llvm-mingw-*-x86_64*/bin")
    list(SORT _ollama_program_files_llvm_mingw_bins COMPARE NATURAL ORDER DESCENDING)
    list(APPEND _ollama_llvm_mingw_hints ${_ollama_program_files_llvm_mingw_bins})
endif()
if(DEFINED ENV{LOCALAPPDATA})
    file(GLOB _ollama_winget_llvm_mingw_bins
        LIST_DIRECTORIES true
        "$ENV{LOCALAPPDATA}/Microsoft/WinGet/Packages/MartinStorsjo.LLVM-MinGW*/llvm-mingw-*-x86_64*/bin")
    list(SORT _ollama_winget_llvm_mingw_bins COMPARE NATURAL ORDER DESCENDING)
    list(APPEND _ollama_llvm_mingw_hints ${_ollama_winget_llvm_mingw_bins})
endif()

if(NOT CMAKE_C_COMPILER)
    find_program(CMAKE_C_COMPILER
        NAMES aarch64-w64-mingw32-gcc
        HINTS ${_ollama_llvm_mingw_hints}
        REQUIRED)
endif()

if(NOT CMAKE_CXX_COMPILER)
    find_program(CMAKE_CXX_COMPILER
        NAMES aarch64-w64-mingw32-g++
        HINTS ${_ollama_llvm_mingw_hints}
        REQUIRED)
endif()

get_filename_component(_ollama_llvm_mingw_bin_dir "${CMAKE_CXX_COMPILER}" DIRECTORY)

if(NOT HOST_CXX_COMPILER)
    find_program(_ollama_path_host_cxx
        NAMES clang++ g++
        NO_CMAKE_FIND_ROOT_PATH)
    if(_ollama_path_host_cxx)
        set(HOST_CXX_COMPILER "${_ollama_path_host_cxx}")
    endif()
endif()
if(NOT HOST_CXX_COMPILER)
    find_program(_ollama_mingw_host_cxx
        NAMES x86_64-w64-mingw32-g++
        HINTS "${_ollama_llvm_mingw_bin_dir}"
        REQUIRED)
    if(CMAKE_HOST_WIN32)
        # llama.cpp builds a small host-only UI embedding tool during
        # cross-compiles, but currently models HOST_CXX_COMPILER as only an
        # executable path and has no companion host flags hook. When the host
        # compiler is llvm-mingw, the generated host tool otherwise depends on
        # llvm-mingw runtime DLLs being on PATH. Keep that workaround local and
        # explicit: wrap the compiler only to add -static for this host tool.
        set(_ollama_host_cxx_wrapper "${CMAKE_BINARY_DIR}/ollama-host-cxx.cmd")
        file(TO_NATIVE_PATH "${_ollama_mingw_host_cxx}" _ollama_mingw_host_cxx_native)
        file(WRITE "${_ollama_host_cxx_wrapper}"
            "@echo off\r\n"
            "\"${_ollama_mingw_host_cxx_native}\" -static %*\r\n")
        set(HOST_CXX_COMPILER "${_ollama_host_cxx_wrapper}")
    else()
        set(HOST_CXX_COMPILER "${_ollama_mingw_host_cxx}")
    endif()
endif()
set(HOST_CXX_COMPILER "${HOST_CXX_COMPILER}" CACHE FILEPATH "Host C++ compiler for build-time tools" FORCE)

string(PREPEND CMAKE_C_FLAGS_INIT "-D_WIN32_WINNT=0x0A00 ")
string(PREPEND CMAKE_CXX_FLAGS_INIT "-D_WIN32_WINNT=0x0A00 ")

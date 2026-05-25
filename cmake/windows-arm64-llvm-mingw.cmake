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
    find_program(HOST_CXX_COMPILER
        NAMES x86_64-w64-mingw32-g++ clang++ g++
        HINTS "${_ollama_llvm_mingw_bin_dir}"
        REQUIRED)
    set(HOST_CXX_COMPILER "${HOST_CXX_COMPILER}" CACHE FILEPATH "Host C++ compiler for build-time tools")
endif()

string(PREPEND CMAKE_C_FLAGS_INIT "-D_WIN32_WINNT=0x0A00 ")
string(PREPEND CMAKE_CXX_FLAGS_INIT "-D_WIN32_WINNT=0x0A00 ")

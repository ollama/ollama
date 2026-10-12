cmake_minimum_required(VERSION 3.24)

project(OllamaXGrammar CXX)

# cmake/mlx provides the pinned xgrammar target and XGRAMMAR_VERSION.
if(NOT TARGET xgrammar)
    set(XGRAMMAR_VERSION "unknown" CACHE STRING "XGrammar release reported by the library")
    find_path(XGRAMMAR_INCLUDE_DIR xgrammar/xgrammar.h REQUIRED)
    find_path(DLPACK_INCLUDE_DIR dlpack/dlpack.h REQUIRED)
    find_library(XGRAMMAR_LIBRARY xgrammar REQUIRED)
    add_library(xgrammar UNKNOWN IMPORTED)
    set_target_properties(xgrammar PROPERTIES
        IMPORTED_LOCATION "${XGRAMMAR_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${XGRAMMAR_INCLUDE_DIR};${DLPACK_INCLUDE_DIR}")
endif()

find_package(Threads REQUIRED)

add_library(ollama_xgrammar SHARED xgrammar.cpp)
target_compile_features(ollama_xgrammar PRIVATE cxx_std_17)
target_include_directories(ollama_xgrammar PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_definitions(ollama_xgrammar PRIVATE
    OLLAMA_XGRAMMAR_BUILD=1
    OLLAMA_XGRAMMAR_VERSION="${XGRAMMAR_VERSION}")
# Export only the ollama_xgrammar_* API; xgrammar's own C++ symbols stay hidden.
set_target_properties(ollama_xgrammar PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)
if(APPLE)
    target_link_options(ollama_xgrammar PRIVATE "LINKER:-exported_symbol,_ollama_xgrammar_*")
elseif(UNIX)
    target_link_options(ollama_xgrammar PRIVATE "LINKER:--exclude-libs,ALL")
endif()
target_link_libraries(ollama_xgrammar PRIVATE xgrammar Threads::Threads)

# cmake/mlx installs the library with the rest of the MLX payload.
if(PROJECT_IS_TOP_LEVEL)
    include(GNUInstallDirs)
    install(TARGETS ollama_xgrammar
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()

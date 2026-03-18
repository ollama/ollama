cmake_minimum_required(VERSION 3.19)

if(WIN32)
    message(STATUS "Building DirectML backend")

    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        # Parallel build object files
        add_definitions(/MP)
    endif()

    ggml_add_backend_library(ggml-directml
        ggml-directml.cpp
        ../../include/ggml-directml.h
    )

    # Use the latest DirectML API version from the Windows SDK
    target_compile_definitions(ggml-directml PRIVATE DML_TARGET_VERSION_USE_LATEST)

    # Link against D3D12 and DXGI (system libraries available on all Windows 10+)
    target_link_libraries(ggml-directml PRIVATE d3d12 dxgi)

    # DirectML.dll is loaded dynamically at runtime (not linked at compile time)
    # This allows the binary to run on systems without DirectML installed

    target_include_directories(ggml-directml PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
else()
    message(STATUS "DirectML backend is only available on Windows, skipping")
endif()

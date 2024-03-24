
set(TARGET ns_ext_server)
if (WIN32)
    add_library(${TARGET} SHARED ext_server.cpp)
else()
    add_library(${TARGET} STATIC ext_server.cpp)
endif()

target_include_directories(${TARGET} PUBLIC . ../neural_speed ../neural_speed/neural_speed ../neural_speed/bestla ../neural_speed/neural_speed/core)
target_compile_features(${TARGET} PUBLIC cxx_std_11)
target_link_libraries(${TARGET} PUBLIC bestla)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
install(TARGETS ns_ext_server LIBRARY)


# llama.cpp owns video sampling and timestamps. Ship its decoder tools with the
# native payload so video input never depends on a user-installed executable.
include(ExternalProject)
find_program(OLLAMA_FFMPEG_BASH bash HINTS "C:/msys64/usr/bin" REQUIRED)
find_program(OLLAMA_FFMPEG_MAKE NAMES gmake make HINTS "C:/msys64/usr/bin" REQUIRED)
set(_ffmpeg_version 9.0.2)
set(_ffmpeg_archive "ffmpeg-${_ffmpeg_version}.tar.xz")
set(_ffmpeg_stage "${CMAKE_BINARY_DIR}/ffmpeg-bin")
set(_ffmpeg_arch "${CMAKE_SYSTEM_PROCESSOR}")
if(APPLE AND CMAKE_OSX_ARCHITECTURES)
    set(_ffmpeg_arch "${CMAKE_OSX_ARCHITECTURES}")
endif()
if(_ffmpeg_arch MATCHES "^(AMD64|amd64|x64)$")
    set(_ffmpeg_arch x86_64)
elseif(_ffmpeg_arch MATCHES "^(ARM64|arm64)$")
    set(_ffmpeg_arch aarch64)
endif()
string(TOLOWER "${CMAKE_SYSTEM_NAME}" _ffmpeg_os)
if(WIN32)
    set(_ffmpeg_os mingw32)
endif()
file(MAKE_DIRECTORY "${_ffmpeg_stage}")
ExternalProject_Add(ollama-ffmpeg
    URL "https://ffmpeg.org/releases/${_ffmpeg_archive}"
    URL_HASH SHA256=8c3850283eb25fa026482078a04051e0be17347b09ef81a0849bec15a96e002e
    DOWNLOAD_DIR "${CMAKE_BINARY_DIR}/../downloads"
    DOWNLOAD_NO_EXTRACT TRUE
    SOURCE_DIR "${CMAKE_BINARY_DIR}/ffmpeg-src"
    # Keep the exact unmodified source alongside the tools, including in offline
    # installs. It also provides corresponding source for the LGPL programs.
    CONFIGURE_COMMAND ${CMAKE_COMMAND} -E tar xf <DOWNLOADED_FILE>
        COMMAND "${OLLAMA_FFMPEG_BASH}" "${CMAKE_CURRENT_LIST_DIR}/../scripts/configure_ffmpeg.sh"
            "<BINARY_DIR>/ffmpeg-${_ffmpeg_version}" "${CMAKE_C_COMPILER}" "${_ffmpeg_os}"
            "${_ffmpeg_arch}" "${CMAKE_OSX_DEPLOYMENT_TARGET}" "${MSVC}"
    BUILD_COMMAND "${OLLAMA_FFMPEG_MAKE}" -C "<BINARY_DIR>/ffmpeg-${_ffmpeg_version}" -j4 ffmpeg${CMAKE_EXECUTABLE_SUFFIX} ffprobe${CMAKE_EXECUTABLE_SUFFIX}
    INSTALL_COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "<BINARY_DIR>/ffmpeg-${_ffmpeg_version}/ffmpeg${CMAKE_EXECUTABLE_SUFFIX}"
        "<BINARY_DIR>/ffmpeg-${_ffmpeg_version}/ffprobe${CMAKE_EXECUTABLE_SUFFIX}"
        "${_ffmpeg_stage}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "<BINARY_DIR>/ffmpeg-${_ffmpeg_version}/COPYING.LGPLv2.1" "${_ffmpeg_stage}/FFMPEG_LICENSE"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different <DOWNLOADED_FILE> "${_ffmpeg_stage}/FFMPEG_SOURCE.tar.xz"
    BUILD_BYPRODUCTS "${_ffmpeg_stage}/ffmpeg${CMAKE_EXECUTABLE_SUFFIX}" "${_ffmpeg_stage}/ffprobe${CMAKE_EXECUTABLE_SUFFIX}"
    USES_TERMINAL_CONFIGURE TRUE
    USES_TERMINAL_BUILD TRUE)
ExternalProject_Add_StepDependencies(ollama-ffmpeg configure "${CMAKE_CURRENT_LIST_DIR}/../scripts/configure_ffmpeg.sh")
install(PROGRAMS "${_ffmpeg_stage}/ffmpeg${CMAKE_EXECUTABLE_SUFFIX}" "${_ffmpeg_stage}/ffprobe${CMAKE_EXECUTABLE_SUFFIX}"
    DESTINATION "${OLLAMA_LIB_DIR}" COMPONENT llama-server)
install(FILES "${_ffmpeg_stage}/FFMPEG_LICENSE" "${_ffmpeg_stage}/FFMPEG_SOURCE.tar.xz"
    DESTINATION "${OLLAMA_LIB_DIR}" COMPONENT llama-server)
install(FILES "${CMAKE_CURRENT_LIST_DIR}/../scripts/configure_ffmpeg.sh"
    DESTINATION "${OLLAMA_LIB_DIR}" RENAME FFMPEG_NOTICE COMPONENT llama-server)

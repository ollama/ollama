# Exercise the shipped tools with no system FFmpeg on PATH. Fixtures contain two
# red frames followed by two blue frames; MP4 metadata is at the end of the file.
if(WIN32)
    set(_suffix .exe)
endif()
set(_fixtures "${CMAKE_CURRENT_LIST_DIR}/../integration/testdata/video")
foreach(_clip red-blue.mp4 red-blue.webm red-blue-hevc.mp4 red-blue-audio.mp4)
    execute_process(COMMAND "${CMAKE_COMMAND}" -E env "PATH="
        "${TOOL_DIR}/ffprobe${_suffix}" -v error -select_streams v:0
        -show_entries stream=width,height -of csv=p=0 -i cache:pipe:0
        INPUT_FILE "${_fixtures}/${_clip}" OUTPUT_VARIABLE _dimensions
        RESULT_VARIABLE _result ERROR_VARIABLE _error)
    string(STRIP "${_dimensions}" _dimensions)
    if(NOT _result EQUAL 0 OR NOT _dimensions STREQUAL "32,32")
        message(FATAL_ERROR "${_clip}: probe failed (${_result}): ${_dimensions} ${_error}")
    endif()
    set(_raw "${CMAKE_CURRENT_BINARY_DIR}/video-test.rgb")
    execute_process(COMMAND "${CMAKE_COMMAND}" -E env "PATH="
        "${TOOL_DIR}/ffmpeg${_suffix}" -nostdin -loglevel error
        -read_ahead_limit -1 -i cache:pipe:0 -vf fps=2
        -frames:v 4 -f rawvideo -pix_fmt rgb24 pipe:1
        INPUT_FILE "${_fixtures}/${_clip}" OUTPUT_FILE "${_raw}"
        RESULT_VARIABLE _result ERROR_VARIABLE _error)
    file(SIZE "${_raw}" _size)
    file(READ "${_raw}" _first OFFSET 0 LIMIT 3 HEX)
    file(READ "${_raw}" _last OFFSET 9216 LIMIT 3 HEX)
    file(REMOVE "${_raw}")
    if(NOT _result EQUAL 0 OR NOT _size EQUAL 12288)
        message(FATAL_ERROR "${_clip}: decode failed (${_result}): ${_size} bytes ${_error}")
    endif()
    math(EXPR _red "0x${_first} >> 16")
    math(EXPR _blue "0x${_last} & 255")
    math(EXPR _first_other "0x${_first} & 65535")
    math(EXPR _last_other "0x${_last} >> 8")
    if(_red LESS 200 OR _blue LESS 200 OR _first_other GREATER 5140 OR _last_other GREATER 5140)
        message(FATAL_ERROR "${_clip}: frame order/colors changed: ${_first}, ${_last}")
    endif()
    message(STATUS "${_clip}: bundled probe and temporal decode passed")
endforeach()

# MTMD also uses FFmpeg to decode WebP once subprocess support is enabled.
set(_raw "${CMAKE_CURRENT_BINARY_DIR}/video-test.rgb")
execute_process(COMMAND "${CMAKE_COMMAND}" -E env "PATH="
    "${TOOL_DIR}/ffmpeg${_suffix}" -nostdin -loglevel error
    -i pipe:0 -frames:v 1 -f rawvideo -pix_fmt rgb24 pipe:1
    INPUT_FILE "${_fixtures}/red.webp" OUTPUT_FILE "${_raw}"
    RESULT_VARIABLE _result ERROR_VARIABLE _error)
file(SIZE "${_raw}" _size)
file(REMOVE "${_raw}")
if(NOT _result EQUAL 0 OR NOT _size EQUAL 3072)
    message(FATAL_ERROR "WebP regression: ${_result}, ${_size} bytes, ${_error}")
endif()
# Omni video models ask for a mono float32 soundtrack through the same tools.
execute_process(COMMAND "${CMAKE_COMMAND}" -E env "PATH="
    "${TOOL_DIR}/ffmpeg${_suffix}" -nostdin -loglevel error -read_ahead_limit -1
    -i cache:pipe:0 -map 0:a:0 -vn -ac 1 -ar 16000 -f f32le pipe:1
    INPUT_FILE "${_fixtures}/red-blue-audio.mp4" OUTPUT_FILE "${_raw}"
    RESULT_VARIABLE _result ERROR_VARIABLE _error)
file(SIZE "${_raw}" _size)
file(REMOVE "${_raw}")
if(NOT _result EQUAL 0 OR _size LESS 128000)
    message(FATAL_ERROR "Soundtrack decode failed: ${_result}, ${_size} bytes, ${_error}")
endif()
message(STATUS "Bundled WebP and soundtrack decode passed")

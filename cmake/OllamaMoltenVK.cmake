function(_ollama_extract_moltenvk archive output_dir)
    string(TOLOWER "${archive}" _lower_archive)
    set(_tar_args "xf")
    if (_lower_archive MATCHES "\\.(tar\\.gz|tgz)$")
        set(_tar_args "xzf")
    elseif (_lower_archive MATCHES "\\.tar\\.xz$")
        set(_tar_args "xJf")
    elseif (_lower_archive MATCHES "\\.tar\\.bz2$")
        set(_tar_args "xjf")
    endif()

    execute_process(
        COMMAND ${CMAKE_COMMAND} -E tar ${_tar_args} "${archive}"
        WORKING_DIRECTORY "${output_dir}"
        RESULT_VARIABLE _tar_result
    )

    if (NOT _tar_result EQUAL 0)
        message(FATAL_ERROR "Failed to extract ${archive} (cmake -E tar exited with ${_tar_result})")
    endif()
endfunction()

function(_ollama_locate_moltenvk_root search_dir out_var)
    set(_candidates)
    file(GLOB _possible "${search_dir}/MoltenVK*")
    list(APPEND _candidates ${_possible})
    list(APPEND _candidates "${search_dir}/MoltenVK")
    list(REMOVE_DUPLICATES _candidates)

    foreach(candidate IN LISTS _candidates)
        if (NOT IS_DIRECTORY "${candidate}")
            continue()
        endif()

        set(_candidate_root "${candidate}")
        if (EXISTS "${candidate}/MoltenVK")
            set(_candidate_root "${candidate}/MoltenVK")
        endif()

        if (EXISTS "${_candidate_root}/include" AND EXISTS "${_candidate_root}/dynamic")
            set(${out_var} "${_candidate_root}" PARENT_SCOPE)
            return()
        endif()
    endforeach()

    set(${out_var} "" PARENT_SCOPE)
endfunction()

function(ollama_configure_moltenvk)
    if (NOT CMAKE_SYSTEM_NAME STREQUAL "Darwin")
        message(FATAL_ERROR "OLLAMA_FETCH_MOLTENVK is only supported on macOS")
    endif()

    set(_molten_url "${OLLAMA_MOLTENVK_URL}")
    if (NOT _molten_url)
        if (NOT OLLAMA_MOLTENVK_VERSION)
            message(FATAL_ERROR "OLLAMA_MOLTENVK_VERSION must be set when OLLAMA_FETCH_MOLTENVK=ON")
        endif()
        set(_molten_url "https://github.com/KhronosGroup/MoltenVK/releases/download/v${OLLAMA_MOLTENVK_VERSION}/MoltenVK-macos.tar")
    endif()
    get_filename_component(_archive_name "${_molten_url}" NAME)
    if (NOT _archive_name)
        set(_archive_name "MoltenVK-${OLLAMA_MOLTENVK_VERSION}.tar")
    endif()

    set(_sdk_root "${CMAKE_BINARY_DIR}/moltenvk-sdk")
    set(_signature_file "${_sdk_root}/.ollama-moltenvk")
    set(_requested_signature "${OLLAMA_MOLTENVK_VERSION}|${_molten_url}")

    set(_needs_download FALSE)
    if (NOT EXISTS "${_sdk_root}/include")
        set(_needs_download TRUE)
    elseif (NOT EXISTS "${_signature_file}")
        set(_needs_download TRUE)
    else()
        file(READ "${_signature_file}" _installed_signature)
        string(STRIP "${_installed_signature}" _installed_signature)
        if (NOT _installed_signature STREQUAL _requested_signature)
            set(_needs_download TRUE)
        endif()
    endif()

    if (_needs_download)
        file(REMOVE_RECURSE "${_sdk_root}")

        set(_download_dir "${CMAKE_BINARY_DIR}/CMakeFiles/moltenvk-download")
        file(REMOVE_RECURSE "${_download_dir}")
        file(MAKE_DIRECTORY "${_download_dir}")

        set(_archive_path "${_download_dir}/${_archive_name}")
        set(_download_args SHOW_PROGRESS)
        if (OLLAMA_MOLTENVK_SHA256)
            list(APPEND _download_args EXPECTED_HASH "SHA256=${OLLAMA_MOLTENVK_SHA256}")
        endif()

        message(STATUS "Downloading MoltenVK from ${_molten_url}")
        file(DOWNLOAD "${_molten_url}" "${_archive_path}" ${_download_args})
        _ollama_extract_moltenvk("${_archive_path}" "${_download_dir}")

        _ollama_locate_moltenvk_root("${_download_dir}" _molten_root)
        if (NOT _molten_root)
            message(FATAL_ERROR "Unable to find MoltenVK payload within ${_archive_name}")
        endif()

        get_filename_component(_sdk_parent "${_sdk_root}" DIRECTORY)
        file(MAKE_DIRECTORY "${_sdk_parent}")
        file(RENAME "${_molten_root}" "${_sdk_root}")

        file(WRITE "${_signature_file}" "${_requested_signature}\n")
        file(REMOVE_RECURSE "${_download_dir}")
    endif()

    if (NOT EXISTS "${_sdk_root}/include")
        message(FATAL_ERROR "MoltenVK include directory not found at ${_sdk_root}")
    endif()

    file(GLOB _molten_libraries "${_sdk_root}/dynamic/dylib/macOS/libMoltenVK.dylib")
    if (NOT _molten_libraries)
        message(FATAL_ERROR "MoltenVK library not found under ${_sdk_root}")
    endif()
    list(GET _molten_libraries 0 _molten_library)

    set(OLLAMA_MOLTENVK_SDK_DIR "${_sdk_root}" CACHE PATH "Cached MoltenVK SDK directory" FORCE)
    mark_as_advanced(OLLAMA_MOLTENVK_SDK_DIR)

    set(Vulkan_INCLUDE_DIR "${_sdk_root}/include" CACHE PATH "MoltenVK include directory" FORCE)
    set(Vulkan_LIBRARY "${_molten_library}" CACHE FILEPATH "MoltenVK Vulkan library" FORCE)
    set(Vulkan_LIBRARIES "${_molten_library}" CACHE FILEPATH "MoltenVK Vulkan library" FORCE)

    if (NOT DEFINED ENV{VULKAN_SDK})
        set(ENV{VULKAN_SDK} "${_sdk_root}")
    endif()

    message(STATUS "Using MoltenVK from ${_sdk_root}")
endfunction()

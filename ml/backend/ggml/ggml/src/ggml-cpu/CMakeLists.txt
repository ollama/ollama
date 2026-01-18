function(ggml_add_cpu_backend_features cpu_name arch)
    # The feature detection code is compiled as a separate target so that
    # it can be built without the architecture flags
    # Since multiple variants of the CPU backend may be included in the same
    # build, using set_source_files_properties() to set the arch flags is not possible
    set(GGML_CPU_FEATS_NAME ${cpu_name}-feats)
    add_library(${GGML_CPU_FEATS_NAME} OBJECT ggml-cpu/arch/${arch}/cpu-feats.cpp)
    target_include_directories(${GGML_CPU_FEATS_NAME} PRIVATE . ../include)
    target_compile_definitions(${GGML_CPU_FEATS_NAME} PRIVATE ${ARGN})
    target_compile_definitions(${GGML_CPU_FEATS_NAME} PRIVATE GGML_BACKEND_DL GGML_BACKEND_BUILD GGML_BACKEND_SHARED)
    set_target_properties(${GGML_CPU_FEATS_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(${cpu_name} PRIVATE ${GGML_CPU_FEATS_NAME})
endfunction()

function(ggml_add_cpu_backend_variant_impl tag_name)
    if (tag_name)
        set(GGML_CPU_NAME ggml-cpu-${tag_name})
    else()
        set(GGML_CPU_NAME ggml-cpu)
    endif()

    ggml_add_backend_library(${GGML_CPU_NAME})

    list (APPEND GGML_CPU_SOURCES
        ggml-cpu/ggml-cpu.c
        ggml-cpu/ggml-cpu.cpp
        ggml-cpu/repack.cpp
        ggml-cpu/repack.h
        ggml-cpu/hbm.cpp
        ggml-cpu/hbm.h
        ggml-cpu/quants.c
        ggml-cpu/quants.h
        ggml-cpu/traits.cpp
        ggml-cpu/traits.h
        ggml-cpu/amx/amx.cpp
        ggml-cpu/amx/amx.h
        ggml-cpu/amx/mmq.cpp
        ggml-cpu/amx/mmq.h
        ggml-cpu/ggml-cpu-impl.h
        ggml-cpu/common.h
        ggml-cpu/binary-ops.h
        ggml-cpu/binary-ops.cpp
        ggml-cpu/unary-ops.h
        ggml-cpu/unary-ops.cpp
        ggml-cpu/simd-mappings.h
        ggml-cpu/vec.h
        ggml-cpu/vec.cpp
        ggml-cpu/ops.h
        ggml-cpu/ops.cpp
        )

    target_compile_features(${GGML_CPU_NAME} PRIVATE c_std_11 cxx_std_17)
    target_include_directories(${GGML_CPU_NAME} PRIVATE . ggml-cpu)

    if (APPLE AND GGML_ACCELERATE)
        find_library(ACCELERATE_FRAMEWORK Accelerate)
        if (ACCELERATE_FRAMEWORK)
            message(STATUS "Accelerate framework found")

            target_compile_definitions(${GGML_CPU_NAME} PRIVATE GGML_USE_ACCELERATE)
            target_compile_definitions(${GGML_CPU_NAME} PRIVATE ACCELERATE_NEW_LAPACK)
            target_compile_definitions(${GGML_CPU_NAME} PRIVATE ACCELERATE_LAPACK_ILP64)

            target_link_libraries(${GGML_CPU_NAME} PRIVATE ${ACCELERATE_FRAMEWORK})
        else()
            message(WARNING "Accelerate framework not found")
        endif()
    endif()

    if (GGML_OPENMP)
        find_package(OpenMP)
        if (OpenMP_FOUND)
            set(GGML_OPENMP_ENABLED "ON" CACHE INTERNAL "")
            target_compile_definitions(${GGML_CPU_NAME} PRIVATE GGML_USE_OPENMP)

            target_link_libraries(${GGML_CPU_NAME} PRIVATE OpenMP::OpenMP_C OpenMP::OpenMP_CXX)
        else()
            set(GGML_OPENMP_ENABLED "OFF" CACHE INTERNAL "")
            message(WARNING "OpenMP not found")
        endif()
    endif()

    if (GGML_LLAMAFILE)
        target_compile_definitions(${GGML_CPU_NAME} PRIVATE GGML_USE_LLAMAFILE)

        list(APPEND GGML_CPU_SOURCES
                    ggml-cpu/llamafile/sgemm.cpp
                    ggml-cpu/llamafile/sgemm.h)
    endif()

    if (GGML_CPU_HBM)
        find_library(memkind memkind REQUIRED)

        message(STATUS "Using memkind for CPU HBM")

        target_compile_definitions(${GGML_CPU_NAME} PRIVATE GGML_USE_CPU_HBM)

        target_link_libraries(${GGML_CPU_NAME} PUBLIC memkind)
    endif()

    if (GGML_SYSTEM_ARCH STREQUAL "ARM")
        message(STATUS "ARM detected")
        list(APPEND GGML_CPU_SOURCES
            ggml-cpu/arch/arm/quants.c
            ggml-cpu/arch/arm/repack.cpp
            )

        if (MSVC AND NOT CMAKE_C_COMPILER_ID STREQUAL "Clang")
            message(FATAL_ERROR "MSVC is not supported for ARM, use clang")
        else()
            check_cxx_compiler_flag(-mfp16-format=ieee GGML_COMPILER_SUPPORTS_FP16_FORMAT_I3E)
            if (NOT "${GGML_COMPILER_SUPPORTS_FP16_FORMAT_I3E}" STREQUAL "")
                list(APPEND ARCH_FLAGS -mfp16-format=ieee)
            endif()

            if (GGML_NATIVE)
                # -mcpu=native does not always enable all the features in some compilers,
                # so we check for them manually and enable them if available

                execute_process(
                    COMMAND ${CMAKE_C_COMPILER} -mcpu=native -E -v -
                    INPUT_FILE "/dev/null"
                    OUTPUT_QUIET
                    ERROR_VARIABLE ARM_MCPU
                    RESULT_VARIABLE ARM_MCPU_RESULT
                )
                if (NOT ARM_MCPU_RESULT)
                    string(REGEX MATCH "-mcpu=[^ ']+" ARM_MCPU_FLAG "${ARM_MCPU}")
                    string(REGEX MATCH "-march=[^ ']+" ARM_MARCH_FLAG "${ARM_MCPU}")

                    # on some old GCC we need to read -march=
                    if (ARM_MARCH_FLAG AND NOT "${ARM_MARCH_FLAG}" STREQUAL "-march=native")
                        set(ARM_NATIVE_FLAG "${ARM_MARCH_FLAG}")
                    elseif(ARM_MCPU_FLAG AND NOT "${ARM_MCPU_FLAG}" STREQUAL "-mcpu=native")
                        set(ARM_NATIVE_FLAG "${ARM_MCPU_FLAG}")
                    endif()
                endif()

                if ("${ARM_NATIVE_FLAG}" STREQUAL "")
                    set(ARM_NATIVE_FLAG -mcpu=native)
                    message(WARNING "ARM -march/-mcpu not found, -mcpu=native will be used")
                else()
                    message(STATUS "ARM detected flags: ${ARM_NATIVE_FLAG}")
                endif()

                include(CheckCXXSourceRuns)

                macro(check_arm_feature tag feature code)
                    set(CMAKE_REQUIRED_FLAGS_SAVE ${CMAKE_REQUIRED_FLAGS})
                    set(CMAKE_REQUIRED_FLAGS "${ARM_NATIVE_FLAG}+${tag}")
                    check_cxx_source_runs("${code}" GGML_MACHINE_SUPPORTS_${tag})
                    if (GGML_MACHINE_SUPPORTS_${tag})
                        set(ARM_NATIVE_FLAG_FIX "${ARM_NATIVE_FLAG_FIX}+${tag}")
                    else()
                        set(CMAKE_REQUIRED_FLAGS "${ARM_NATIVE_FLAG}+no${tag}")
                        check_cxx_source_compiles("int main() { return 0; }" GGML_MACHINE_SUPPORTS_no${tag})
                        if (GGML_MACHINE_SUPPORTS_no${tag})
                            set(ARM_NATIVE_FLAG_FIX "${ARM_NATIVE_FLAG_FIX}+no${tag}")
                            list(APPEND ARCH_FLAGS -U__ARM_FEATURE_${feature})
                        endif()
                    endif()
                    set(CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS_SAVE})
                endmacro()

                check_arm_feature(dotprod DOTPROD     "#include <arm_neon.h>\nint main() { int8x16_t _a, _b; volatile int32x4_t _s = vdotq_s32(_s, _a, _b); return 0; }")
                check_arm_feature(i8mm    MATMUL_INT8 "#include <arm_neon.h>\nint main() { int8x16_t _a, _b; volatile int32x4_t _s = vmmlaq_s32(_s, _a, _b); return 0; }")
                check_arm_feature(sve     SVE         "#include <arm_sve.h>\nint main()  { svfloat32_t _a, _b; volatile svfloat32_t _c = svadd_f32_z(svptrue_b8(), _a, _b); return 0; }")
                check_arm_feature(sme     SME         "#include <arm_sme.h>\n__arm_locally_streaming int main() { __asm__ volatile(\"smstart; smstop;\"); return 0; }")

                list(APPEND ARCH_FLAGS "${ARM_NATIVE_FLAG}${ARM_NATIVE_FLAG_FIX}")
            else()
                if (GGML_CPU_ARM_ARCH)
                    list(APPEND ARCH_FLAGS -march=${GGML_CPU_ARM_ARCH})
                elseif(GGML_CPU_ALL_VARIANTS)
                    # Begin with the lowest baseline
                    set(ARM_MCPU "armv8-a")
                    set(ARCH_TAGS "")
                    set(ARCH_DEFINITIONS "")

                    # When a feature is selected, bump the MCPU to the first
                    # version that supported it
                    if (GGML_INTERNAL_DOTPROD)
                        set(ARM_MCPU "armv8.2-a")
                        set(ARCH_TAGS "${ARCH_TAGS}+dotprod")
                        list(APPEND ARCH_DEFINITIONS GGML_USE_DOTPROD)
                    endif()
                    if (GGML_INTERNAL_FP16_VECTOR_ARITHMETIC)
                        set(ARM_MCPU "armv8.2-a")
                        set(ARCH_TAGS "${ARCH_TAGS}+fp16")
                        list(APPEND ARCH_DEFINITIONS GGML_USE_FP16_VECTOR_ARITHMETIC)
                    endif()
                    if (GGML_INTERNAL_SVE)
                        set(ARM_MCPU "armv8.2-a")
                        set(ARCH_TAGS "${ARCH_TAGS}+sve")
                        list(APPEND ARCH_DEFINITIONS GGML_USE_SVE)
                    endif()
                    if (GGML_INTERNAL_MATMUL_INT8)
                        set(ARM_MCPU "armv8.6-a")
                        set(ARCH_TAGS "${ARCH_TAGS}+i8mm")
                        list(APPEND ARCH_DEFINITIONS GGML_USE_MATMUL_INT8)
                    endif()
                    if (GGML_INTERNAL_SVE2)
                        set(ARM_MCPU "armv8.6-a")
                        set(ARCH_TAGS "${ARCH_TAGS}+sve2")
                        list(APPEND ARCH_DEFINITIONS GGML_USE_SVE2)
                    endif()
                    if (GGML_INTERNAL_NOSVE)
                        set(ARCH_TAGS "${ARCH_TAGS}+nosve")
                    endif()
                    if (GGML_INTERNAL_SME)
                        set(ARM_MCPU "armv9.2-a")
                        set(ARCH_TAGS "${ARCH_TAGS}+sme")
                        list(APPEND ARCH_DEFINITIONS GGML_USE_SME)
                    endif()
                    list(APPEND ARCH_FLAGS "-march=${ARM_MCPU}${ARCH_TAGS}")
                    ggml_add_cpu_backend_features(${GGML_CPU_NAME} arm ${ARCH_DEFINITIONS})
                endif()
            endif()

            message(STATUS "Checking for ARM features using flags:")
            foreach(flag IN LISTS ARCH_FLAGS)
                message(STATUS "  ${flag}")
            endforeach()

            include(CheckCXXSourceCompiles)
            set(CMAKE_REQUIRED_FLAGS_SAVE ${CMAKE_REQUIRED_FLAGS})
            string(REPLACE ";" " " ARCH_FLAGS_STR "${ARCH_FLAGS}")
            set(CMAKE_REQUIRED_FLAGS "${ARCH_FLAGS_STR}")
            foreach(feature DOTPROD SVE MATMUL_INT8 FMA FP16_VECTOR_ARITHMETIC SME)
                set(ARM_FEATURE "HAVE_${feature}")
                check_cxx_source_compiles(
                    "
                    #if !defined(__ARM_FEATURE_${feature})
                    #  error \"Feature ${feature} is not defined\"
                    #endif
                    int main() { return 0; }
                    "
                    ${ARM_FEATURE}
                )
            endforeach()
            set(CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS_SAVE})
        endif()
    elseif (GGML_SYSTEM_ARCH STREQUAL "x86")
        message(STATUS "x86 detected")
        list(APPEND GGML_CPU_SOURCES
            ggml-cpu/arch/x86/quants.c
            ggml-cpu/arch/x86/repack.cpp
            )

        if (MSVC)
            # instruction set detection for MSVC only
            if (GGML_NATIVE)
                include(ggml-cpu/cmake/FindSIMD.cmake)
            endif ()
            if (GGML_AVX512)
                list(APPEND ARCH_FLAGS /arch:AVX512)
                # /arch:AVX512 includes: __AVX512F__, __AVX512CD__, __AVX512BW__, __AVX512DQ__, and __AVX512VL__
                # MSVC has no compile-time flags enabling specific
                # AVX512 extensions, neither it defines the
                # macros corresponding to the extensions.
                # Do it manually.
                list(APPEND ARCH_DEFINITIONS GGML_AVX512)
                if (GGML_AVX512_VBMI)
                    list(APPEND ARCH_DEFINITIONS __AVX512VBMI__)
                    if (CMAKE_C_COMPILER_ID STREQUAL "Clang")
                        list(APPEND ARCH_FLAGS -mavx512vbmi)
                    endif()
                endif()
                if (GGML_AVX512_VNNI)
                    list(APPEND ARCH_DEFINITIONS __AVX512VNNI__ GGML_AVX512_VNNI)
                    if (CMAKE_C_COMPILER_ID STREQUAL "Clang")
                        list(APPEND ARCH_FLAGS -mavx512vnni)
                    endif()
                endif()
                if (GGML_AVX512_BF16)
                    list(APPEND ARCH_DEFINITIONS __AVX512BF16__ GGML_AVX512_BF16)
                    if (CMAKE_C_COMPILER_ID STREQUAL "Clang")
                        list(APPEND ARCH_FLAGS -mavx512bf16)
                    endif()
                endif()
                if (GGML_AMX_TILE)
                    list(APPEND ARCH_DEFINITIONS __AMX_TILE__ GGML_AMX_TILE)
                endif()
                if (GGML_AMX_INT8)
                    list(APPEND ARCH_DEFINITIONS __AMX_INT8__ GGML_AMX_INT8)
                endif()
                if (GGML_AMX_BF16)
                    list(APPEND ARCH_DEFINITIONS __AMX_BF16__ GGML_AMX_BF16)
                endif()
            elseif (GGML_AVX2)
                list(APPEND ARCH_FLAGS /arch:AVX2)
                list(APPEND ARCH_DEFINITIONS GGML_AVX2 GGML_FMA GGML_F16C)
            elseif (GGML_AVX)
                list(APPEND ARCH_FLAGS /arch:AVX)
                list(APPEND ARCH_DEFINITIONS GGML_AVX)
            elseif (GGML_SSE42)
                list(APPEND ARCH_FLAGS /arch:SSE4.2)
                list(APPEND ARCH_DEFINITIONS GGML_SSE42)
            endif()
            if (GGML_AVX_VNNI)
                list(APPEND ARCH_DEFINITIONS __AVXVNNI__ GGML_AVX_VNNI)
            endif()
            if (GGML_BMI2)
                # MSVC does not define macro __BMI2__
                list(APPEND ARCH_DEFINITIONS __BMI2__ GGML_BMI2)
            endif()
        else ()
            if (GGML_NATIVE)
                list(APPEND ARCH_FLAGS -march=native)
            else ()
                if (GGML_SSE42)
                    list(APPEND ARCH_FLAGS -msse4.2)
                    list(APPEND ARCH_DEFINITIONS GGML_SSE42)
                endif()
                if (GGML_F16C)
                    list(APPEND ARCH_FLAGS -mf16c)
                    list(APPEND ARCH_DEFINITIONS GGML_F16C)
                endif()
                if (GGML_FMA)
                    list(APPEND ARCH_FLAGS -mfma)
                    list(APPEND ARCH_DEFINITIONS GGML_FMA)
                endif()
                if (GGML_BMI2)
                    list(APPEND ARCH_FLAGS -mbmi2)
                    list(APPEND ARCH_DEFINITIONS GGML_BMI2)
                endif()
                if (GGML_AVX)
                    list(APPEND ARCH_FLAGS -mavx)
                    list(APPEND ARCH_DEFINITIONS GGML_AVX)
                endif()
                if (GGML_AVX2)
                    list(APPEND ARCH_FLAGS -mavx2)
                    list(APPEND ARCH_DEFINITIONS GGML_AVX2)
                endif()
                if (GGML_AVX_VNNI)
                    list(APPEND ARCH_FLAGS -mavxvnni)
                    list(APPEND ARCH_DEFINITIONS GGML_AVX_VNNI)
                endif()
                if (GGML_AVX512)
                    list(APPEND ARCH_FLAGS -mavx512f)
                    list(APPEND ARCH_FLAGS -mavx512cd)
                    list(APPEND ARCH_FLAGS -mavx512vl)
                    list(APPEND ARCH_FLAGS -mavx512dq)
                    list(APPEND ARCH_FLAGS -mavx512bw)
                    list(APPEND ARCH_DEFINITIONS GGML_AVX512)
                endif()
                if (GGML_AVX512_VBMI)
                    list(APPEND ARCH_FLAGS -mavx512vbmi)
                    list(APPEND ARCH_DEFINITIONS GGML_AVX512_VBMI)
                endif()
                if (GGML_AVX512_VNNI)
                    list(APPEND ARCH_FLAGS -mavx512vnni)
                    list(APPEND ARCH_DEFINITIONS GGML_AVX512_VNNI)
                endif()
                if (GGML_AVX512_BF16)
                    list(APPEND ARCH_FLAGS -mavx512bf16)
                    list(APPEND ARCH_DEFINITIONS GGML_AVX512_BF16)
                endif()
                if (GGML_AMX_TILE)
                    list(APPEND ARCH_FLAGS -mamx-tile)
                    list(APPEND ARCH_DEFINITIONS GGML_AMX_TILE)
                endif()
                if (GGML_AMX_INT8)
                    list(APPEND ARCH_FLAGS -mamx-int8)
                    list(APPEND ARCH_DEFINITIONS GGML_AMX_INT8)
                endif()
                if (GGML_AMX_BF16)
                    list(APPEND ARCH_FLAGS -mamx-bf16)
                    list(APPEND ARCH_DEFINITIONS GGML_AMX_BF16)
                endif()
            endif()
        endif()

        if (GGML_BACKEND_DL)
            if (GGML_NATIVE)
                # the feature check relies on ARCH_DEFINITIONS, but it is not set with GGML_NATIVE
                message(FATAL_ERROR "GGML_NATIVE is not compatible with GGML_BACKEND_DL, consider using GGML_CPU_ALL_VARIANTS")
            endif()
            ggml_add_cpu_backend_features(${GGML_CPU_NAME} x86 ${ARCH_DEFINITIONS})
        endif()
    elseif (GGML_SYSTEM_ARCH STREQUAL "PowerPC")
        message(STATUS "PowerPC detected")
        list(APPEND GGML_CPU_SOURCES ggml-cpu/arch/powerpc/quants.c)
        if (GGML_NATIVE)
            if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "ppc64")
                file(READ "/proc/cpuinfo" POWER10_M)
            elseif (${CMAKE_SYSTEM_PROCESSOR} MATCHES "powerpc")
                execute_process(COMMAND bash -c "prtconf |grep 'Implementation' | head -n 1" OUTPUT_VARIABLE POWER10_M)
            endif()

            string(TOUPPER "${POWER10_M}" POWER10_M_UPPER)
            string(REGEX MATCHALL "POWER *([0-9]+)" MATCHED_STRING "${POWER10_M_UPPER}")
            string(REGEX REPLACE "POWER *([0-9]+)" "\\1" EXTRACTED_NUMBER "${MATCHED_STRING}")

            if (EXTRACTED_NUMBER GREATER_EQUAL 10)
                list(APPEND ARCH_FLAGS -mcpu=power10)
            elseif (EXTRACTED_NUMBER EQUAL 9)
                list(APPEND ARCH_FLAGS -mcpu=power9)
            elseif (${CMAKE_SYSTEM_PROCESSOR} MATCHES "ppc64le")
                list(APPEND ARCH_FLAGS -mcpu=powerpc64le -mtune=native)
            else()
                list(APPEND ARCH_FLAGS -mcpu=native -mtune=native -mpowerpc64)
            endif()
        elseif(GGML_CPU_ALL_VARIANTS)
            # Begin with the lowest baseline
            set(ARCH_DEFINITIONS "")

            # When a feature is selected, bump the MCPU to the first
            # version that supported it
            foreach(PVER RANGE 7 11)
                if(DEFINED GGML_INTERNAL_POWER${PVER})
                    set(POWERPC_MCPU "power${PVER}")
                    list(APPEND ARCH_DEFINITIONS GGML_USE_POWER${PVER})
                endif()
            endforeach()
            if (GGML_INTERNAL_VSX)
                list(APPEND ARCH_DEFINITIONS GGML_USE_VSX)
                list(APPEND ARCH_FLAGS -mvsx)
            endif()

            if (DEFINED POWERPC_MCPU)
                list(APPEND ARCH_FLAGS -mcpu=${POWERPC_MCPU})
            endif()
            ggml_add_cpu_backend_features(${GGML_CPU_NAME} powerpc ${ARCH_DEFINITIONS})
        else()
            if (GGML_CPU_POWERPC_CPUTYPE)
                list(APPEND ARCH_FLAGS -mcpu=${GGML_CPU_POWERPC_CPUTYPE})
            endif()
        endif()
    elseif (GGML_SYSTEM_ARCH STREQUAL "loongarch64")
        message(STATUS "loongarch64 detected")
        list(APPEND GGML_CPU_SOURCES ggml-cpu/arch/loongarch/quants.c)

        list(APPEND ARCH_FLAGS -march=loongarch64)
        if (GGML_LASX)
            list(APPEND ARCH_FLAGS -mlasx)
        endif()
        if (GGML_LSX)
            list(APPEND ARCH_FLAGS -mlsx)
        endif()
    elseif (GGML_SYSTEM_ARCH STREQUAL "riscv64")
        message(STATUS "riscv64 detected")
        list(APPEND GGML_CPU_SOURCES
            ggml-cpu/arch/riscv/quants.c
            ggml-cpu/arch/riscv/repack.cpp
            )
        if (GGML_CPU_RISCV64_SPACEMIT)
            target_compile_definitions(${GGML_CPU_NAME} PRIVATE GGML_USE_CPU_RISCV64_SPACEMIT ${RISCV64_SPACEMIT_IME_SPEC})
            list(APPEND GGML_CPU_SOURCES
                ggml-cpu/spacemit/ime.cpp
                ggml-cpu/spacemit/ime.h
                ggml-cpu/spacemit/ime1_kernels.cpp
                ggml-cpu/spacemit/ime_kernels.h
            )
        endif()
        if(NOT GGML_CPU_ALL_VARIANTS)
            set(MARCH_STR "rv64gc")
            if (GGML_RV_ZFH)
                string(APPEND MARCH_STR "_zfh")
            endif()

            if (GGML_XTHEADVECTOR)
                string(APPEND MARCH_STR "_xtheadvector")
            elseif (GGML_RVV)
                string(APPEND MARCH_STR "_v")
                if (GGML_RV_ZVFH)
                    string(APPEND MARCH_STR "_zvfh")
                endif()
                if (GGML_RV_ZVFBFWMA)
                    string(APPEND MARCH_STR "_zvfbfwma")
                endif()
            endif()
            if (GGML_RV_ZICBOP)
                string(APPEND MARCH_STR "_zicbop")
            endif()
            if (GGML_RV_ZIHINTPAUSE)
                string(APPEND MARCH_STR "_zihintpause")
            endif()
            list(APPEND ARCH_FLAGS "-march=${MARCH_STR}" -mabi=lp64d)
        else()
            # Begin with the lowest baseline
            set(ARCH_DEFINITIONS "")

            if (GGML_INTERNAL_RVV)
                message(STATUS "RVV enabled")
                list(APPEND ARCH_DEFINITIONS GGML_USE_RVV)
                list(APPEND ARCH_FLAGS -march=rv64gc_v -mabi=lp64d)
            endif()

            ggml_add_cpu_backend_features(${GGML_CPU_NAME} riscv ${ARCH_DEFINITIONS})
        endif()
    elseif (GGML_SYSTEM_ARCH STREQUAL "s390x")
        message(STATUS "s390x detected")
        list(APPEND GGML_CPU_SOURCES
            ggml-cpu/arch/s390/quants.c)

        # for native compilation
        if (GGML_NATIVE)
            # check machine level to determine target
            file(READ "/proc/cpuinfo" CPUINFO_CONTENTS)
            string(REGEX REPLACE "machine[ \t\r\n]*=[ \t\r\n]*([0-9]+)" "\\1" S390X_M ${CPUINFO_CONTENTS})

            # TODO: Separation to determine activation of VX/VXE/VXE2
            if (${S390X_M} MATCHES "8561|8562")
                message(STATUS "z15 target")
                list(APPEND ARCH_FLAGS -march=z15)
            elseif (${S390X_M} MATCHES "3931")
                message(STATUS "z16 target")
                list(APPEND ARCH_FLAGS -march=z16)
            elseif (${S390X_M} MATCHES "9175|9176")
                # NOTE: Only available from GCC 15.1.0 onwards. Any z17 machine with compile issues must first verify their GCC version.
                #       binutils must also be updated to the latest for the -march=z17 flag to work. Otherwise, use -march=arch15.
                message(STATUS "z17 target")
                list(APPEND ARCH_FLAGS -march=arch15)
            else()
                message(STATUS "Unknown target")
                message(WARNING "Unknown target. If you are compiling for z14 and earlier, you might have to add -DGGML_VXE=OFF.")
                list(APPEND ARCH_FLAGS -march=native -mtune=native)
            endif()
        # for cross-compilation
        elseif(GGML_CPU_ALL_VARIANTS)
            # range through IBM z15 to z17
            # NOTE: update when a new hardware level is released
            foreach (ZHW RANGE 15 17)
                if(DEFINED GGML_INTERNAL_Z${ZHW})
                    message(STATUS "z${ZHW} cross-compile target")
                    list(APPEND ARCH_FLAGS -march=z${ZHW})
                endif()
            endforeach()
        endif()

        if (GGML_VXE OR GGML_INTERNAL_VXE2)
            message(STATUS "VXE2 enabled")
            list(APPEND ARCH_FLAGS -mvx -mzvector)
            list(APPEND ARCH_DEFINITIONS GGML_USE_VXE2)
        endif()

        if (GGML_INTERNAL_NNPA)
            message(STATUS "NNPA enabled")
            list(APPEND ARCH_DEFINITIONS GGML_USE_NNPA)
        endif()

        ggml_add_cpu_backend_features(${GGML_CPU_NAME} s390 ${ARCH_DEFINITIONS})
    elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "wasm")
        message(STATUS "Wasm detected")
        list (APPEND GGML_CPU_SOURCES ggml-cpu/arch/wasm/quants.c)
    else()
        message(WARNING "Unknown CPU architecture. Falling back to generic implementations.")
        list(APPEND ARCH_FLAGS -DGGML_CPU_GENERIC)
    endif()

    if (GGML_CPU_REPACK)
        target_compile_definitions(${GGML_CPU_NAME} PRIVATE GGML_USE_CPU_REPACK)
    endif()

    if (GGML_CPU_KLEIDIAI)
        message(STATUS "Using KleidiAI optimized kernels if applicable")

        # Disable the KleidiAI tests
        set(KLEIDIAI_BUILD_TESTS  OFF)

        # Fetch KleidiAI sources:
        include(FetchContent)
        set(KLEIDIAI_COMMIT_TAG "v1.16.0")
        set(KLEIDIAI_DOWNLOAD_URL "https://github.com/ARM-software/kleidiai/archive/refs/tags/${KLEIDIAI_COMMIT_TAG}.tar.gz")
        set(KLEIDIAI_ARCHIVE_MD5  "0a9e9008adb6031f9e8cf70dff4a3321")

        if (POLICY CMP0135)
            cmake_policy(SET CMP0135 NEW)
        endif()

        FetchContent_Declare(KleidiAI_Download
            URL ${KLEIDIAI_DOWNLOAD_URL}
            DOWNLOAD_EXTRACT_TIMESTAMP NEW
            URL_HASH MD5=${KLEIDIAI_ARCHIVE_MD5})

        FetchContent_MakeAvailable(KleidiAI_Download)
        FetchContent_GetProperties(KleidiAI_Download
            SOURCE_DIR  KLEIDIAI_SRC
            POPULATED   KLEIDIAI_POPULATED)

        if (NOT KLEIDIAI_POPULATED)
            message(FATAL_ERROR "KleidiAI source downloaded failed.")
        endif()

        add_compile_definitions(GGML_USE_CPU_KLEIDIAI)

        # Remove kleidiai target after fetching it
        if (TARGET kleidiai)
            set_target_properties(kleidiai PROPERTIES EXCLUDE_FROM_ALL TRUE)
        endif()

        list(APPEND GGML_CPU_SOURCES
            ggml-cpu/kleidiai/kleidiai.cpp
            ggml-cpu/kleidiai/kernels.cpp
            ggml-cpu/kleidiai/kleidiai.h
            ggml-cpu/kleidiai/kernels.h
            )

        # KleidiAI
        include_directories(
            ${KLEIDIAI_SRC}/
            ${KLEIDIAI_SRC}/kai/
            ${KLEIDIAI_SRC}/kai/ukernels/
            ${KLEIDIAI_SRC}/kai/ukernels/matmul/
            ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/
            ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/
            ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_fp32_bf16p_bf16p/
            ${KLEIDIAI_SRC}/kai/ukernels/matmul/pack/)

        set(ARCH_FLAGS_TEMP "${ARCH_FLAGS}")
        if (NOT ARCH_FLAGS_TEMP)
            string(REGEX MATCH "-march=[^ ]+" ARCH_FLAGS_TEMP "${CMAKE_C_FLAGS}")
        endif()
        string(FIND "${ARCH_FLAGS_TEMP}" "+dotprod" DOTPROD_ENABLED)
        string(FIND "${ARCH_FLAGS_TEMP}" "+i8mm" I8MM_ENABLED)
        string(FIND "${ARCH_FLAGS_TEMP}" "+sme" SME_ENABLED)
        string(FIND "${ARCH_FLAGS_TEMP}" "+sve" SVE_ENABLED)

        set(PRIVATE_ARCH_FLAGS ${ARCH_FLAGS_TEMP})

        list(APPEND GGML_KLEIDIAI_SOURCES
            ${KLEIDIAI_SRC}/kai/ukernels/matmul/pack/kai_lhs_quant_pack_qsi8d32p_f32.c
            ${KLEIDIAI_SRC}/kai/ukernels/matmul/pack/kai_lhs_quant_pack_qsi8d32p4x8sb_f32_neon.c
            ${KLEIDIAI_SRC}/kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4c32ps1s0scalef16_qsu4c32s16s0_neon.c
            ${KLEIDIAI_SRC}/kai/ukernels/matmul/pack/kai_lhs_quant_pack_qsi8d32p_f32_neon.c
            ${KLEIDIAI_SRC}/kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0.c
            ${KLEIDIAI_SRC}/kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.c
            ${KLEIDIAI_SRC}/kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi8cxp_qsi8cx_neon.c)

        if (NOT DOTPROD_ENABLED MATCHES -1)
            list(APPEND GGML_KLEIDIAI_SOURCES
                ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod.c
                ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod.c
                ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod.c
                ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod.c
                ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod.c
                ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod.c)
        endif()

        if (NOT I8MM_ENABLED MATCHES -1)
            list(APPEND GGML_KLEIDIAI_SOURCES
                ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm.c
                ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm.c)
        endif()

        if (NOT SME_ENABLED MATCHES -1)
            list(APPEND GGML_KLEIDIAI_SOURCES
                ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme2_mopa.c
                ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot.c
                ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa.c
                ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa_asm.S
                ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot.c
                ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot_asm.S
                ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_fp32_bf16p_bf16p/kai_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa.c
                ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_fp32_bf16p_bf16p/kai_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa_asm.S
                ${KLEIDIAI_SRC}/kai/ukernels/matmul/pack/kai_lhs_pack_bf16p2vlx2_f32_sme.c
                ${KLEIDIAI_SRC}/kai/ukernels/matmul/pack/kai_rhs_pack_kxn_bf16p2vlx2b_f32_x32_sme.c
                ${KLEIDIAI_SRC}/kai/kai_common_sme_asm.S)
            set(PRIVATE_ARCH_FLAGS "-fno-tree-vectorize;${PRIVATE_ARCH_FLAGS}+sve+sve2")
        endif()

        if (NOT SVE_ENABLED MATCHES -1)
            list(APPEND GGML_KLEIDIAI_SOURCES
                ${KLEIDIAI_SRC}/kai/kai_common_sve_asm.S
                ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod_asm.S
                ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod.c
                ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm_asm.S
                ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm.c)
        endif()

        set_source_files_properties(${GGML_KLEIDIAI_SOURCES} PROPERTIES COMPILE_OPTIONS "${PRIVATE_ARCH_FLAGS}")
        list(APPEND GGML_CPU_SOURCES ${GGML_KLEIDIAI_SOURCES})
    endif()

    message(STATUS "Adding CPU backend variant ${GGML_CPU_NAME}: ${ARCH_FLAGS} ${ARCH_DEFINITIONS}")
    target_sources(${GGML_CPU_NAME} PRIVATE ${GGML_CPU_SOURCES})
    target_compile_options(${GGML_CPU_NAME} PRIVATE ${ARCH_FLAGS})
    target_compile_definitions(${GGML_CPU_NAME} PRIVATE ${ARCH_DEFINITIONS})

    if (EMSCRIPTEN)
        set_target_properties(${GGML_CPU_NAME} PROPERTIES COMPILE_FLAGS "-msimd128")
    endif()

    if (CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
        # The compiler automatically enables "-ffast-math" which can cause NaNs in tests due to "-fassociative-math"
        target_compile_options(${GGML_CPU_NAME} PRIVATE "-fno-associative-math")
    endif()
endfunction()

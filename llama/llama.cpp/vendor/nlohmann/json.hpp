//     __ _____ _____ _____
//  __|  |   __|     |   | |  JSON for Modern C++
// |  |  |__   |  |  | | | |  version 3.12.0
// |_____|_____|_____|_|___|  https://github.com/nlohmann/json
//
// SPDX-FileCopyrightText: 2013 - 2025 Niels Lohmann <https://nlohmann.me>
// SPDX-License-Identifier: MIT

/****************************************************************************\
 * Note on documentation: The source files contain links to the online      *
 * documentation of the public API at https://json.nlohmann.me. This URL    *
 * contains the most recent documentation and should also be applicable to  *
 * previous versions; documentation for deprecated functions is not         *
 * removed, but marked deprecated. See "Generate documentation" section in  *
 * file docs/README.md.                                                     *
\****************************************************************************/

#ifndef INCLUDE_NLOHMANN_JSON_HPP_
#define INCLUDE_NLOHMANN_JSON_HPP_

#include <algorithm> // all_of, find, for_each
#include <cstddef> // nullptr_t, ptrdiff_t, size_t
#include <functional> // hash, less
#include <initializer_list> // initializer_list
#ifndef JSON_NO_IO
    #include <iosfwd> // istream, ostream
#endif  // JSON_NO_IO
#include <iterator> // random_access_iterator_tag
#include <memory> // unique_ptr
#include <string> // string, stoi, to_string
#include <utility> // declval, forward, move, pair, swap
#include <vector> // vector

// #include <nlohmann/adl_serializer.hpp>
//     __ _____ _____ _____
//  __|  |   __|     |   | |  JSON for Modern C++
// |  |  |__   |  |  | | | |  version 3.12.0
// |_____|_____|_____|_|___|  https://github.com/nlohmann/json
//
// SPDX-FileCopyrightText: 2013 - 2025 Niels Lohmann <https://nlohmann.me>
// SPDX-License-Identifier: MIT



#include <utility>

// #include <nlohmann/detail/abi_macros.hpp>
//     __ _____ _____ _____
//  __|  |   __|     |   | |  JSON for Modern C++
// |  |  |__   |  |  | | | |  version 3.12.0
// |_____|_____|_____|_|___|  https://github.com/nlohmann/json
//
// SPDX-FileCopyrightText: 2013 - 2025 Niels Lohmann <https://nlohmann.me>
// SPDX-License-Identifier: MIT



// This file contains all macro definitions affecting or depending on the ABI

#ifndef JSON_SKIP_LIBRARY_VERSION_CHECK
    #if defined(NLOHMANN_JSON_VERSION_MAJOR) && defined(NLOHMANN_JSON_VERSION_MINOR) && defined(NLOHMANN_JSON_VERSION_PATCH)
        #if NLOHMANN_JSON_VERSION_MAJOR != 3 || NLOHMANN_JSON_VERSION_MINOR != 12 || NLOHMANN_JSON_VERSION_PATCH != 0
            #warning "Already included a different version of the library!"
        #endif
    #endif
#endif

#define NLOHMANN_JSON_VERSION_MAJOR 3   // NOLINT(modernize-macro-to-enum)
#define NLOHMANN_JSON_VERSION_MINOR 12  // NOLINT(modernize-macro-to-enum)
#define NLOHMANN_JSON_VERSION_PATCH 0   // NOLINT(modernize-macro-to-enum)

#ifndef JSON_DIAGNOSTICS
    #define JSON_DIAGNOSTICS 0
#endif

#ifndef JSON_DIAGNOSTIC_POSITIONS
    #define JSON_DIAGNOSTIC_POSITIONS 0
#endif

#ifndef JSON_USE_LEGACY_DISCARDED_VALUE_COMPARISON
    #define JSON_USE_LEGACY_DISCARDED_VALUE_COMPARISON 0
#endif

#if JSON_DIAGNOSTICS
    #define NLOHMANN_JSON_ABI_TAG_DIAGNOSTICS _diag
#else
    #define NLOHMANN_JSON_ABI_TAG_DIAGNOSTICS
#endif

#if JSON_DIAGNOSTIC_POSITIONS
    #define NLOHMANN_JSON_ABI_TAG_DIAGNOSTIC_POSITIONS _dp
#else
    #define NLOHMANN_JSON_ABI_TAG_DIAGNOSTIC_POSITIONS
#endif

#if JSON_USE_LEGACY_DISCARDED_VALUE_COMPARISON
    #define NLOHMANN_JSON_ABI_TAG_LEGACY_DISCARDED_VALUE_COMPARISON _ldvcmp
#else
    #define NLOHMANN_JSON_ABI_TAG_LEGACY_DISCARDED_VALUE_COMPARISON
#endif

#ifndef NLOHMANN_JSON_NAMESPACE_NO_VERSION
    #define NLOHMANN_JSON_NAMESPACE_NO_VERSION 0
#endif

// Construct the namespace ABI tags component
#define NLOHMANN_JSON_ABI_TAGS_CONCAT_EX(a, b, c) json_abi ## a ## b ## c
#define NLOHMANN_JSON_ABI_TAGS_CONCAT(a, b, c) \
    NLOHMANN_JSON_ABI_TAGS_CONCAT_EX(a, b, c)

#define NLOHMANN_JSON_ABI_TAGS                                       \
    NLOHMANN_JSON_ABI_TAGS_CONCAT(                                   \
            NLOHMANN_JSON_ABI_TAG_DIAGNOSTICS,                       \
            NLOHMANN_JSON_ABI_TAG_LEGACY_DISCARDED_VALUE_COMPARISON, \
            NLOHMANN_JSON_ABI_TAG_DIAGNOSTIC_POSITIONS)

// Construct the namespace version component
#define NLOHMANN_JSON_NAMESPACE_VERSION_CONCAT_EX(major, minor, patch) \
    _v ## major ## _ ## minor ## _ ## patch
#define NLOHMANN_JSON_NAMESPACE_VERSION_CONCAT(major, minor, patch) \
    NLOHMANN_JSON_NAMESPACE_VERSION_CONCAT_EX(major, minor, patch)

#if NLOHMANN_JSON_NAMESPACE_NO_VERSION
#define NLOHMANN_JSON_NAMESPACE_VERSION
#else
#define NLOHMANN_JSON_NAMESPACE_VERSION                                 \
    NLOHMANN_JSON_NAMESPACE_VERSION_CONCAT(NLOHMANN_JSON_VERSION_MAJOR, \
                                           NLOHMANN_JSON_VERSION_MINOR, \
                                           NLOHMANN_JSON_VERSION_PATCH)
#endif

// Combine namespace components
#define NLOHMANN_JSON_NAMESPACE_CONCAT_EX(a, b) a ## b
#define NLOHMANN_JSON_NAMESPACE_CONCAT(a, b) \
    NLOHMANN_JSON_NAMESPACE_CONCAT_EX(a, b)

#ifndef NLOHMANN_JSON_NAMESPACE
#define NLOHMANN_JSON_NAMESPACE               \
    nlohmann::NLOHMANN_JSON_NAMESPACE_CONCAT( \
            NLOHMANN_JSON_ABI_TAGS,           \
            NLOHMANN_JSON_NAMESPACE_VERSION)
#endif

#ifndef NLOHMANN_JSON_NAMESPACE_BEGIN
#define NLOHMANN_JSON_NAMESPACE_BEGIN                \
    namespace nlohmann                               \
    {                                                \
    inline namespace NLOHMANN_JSON_NAMESPACE_CONCAT( \
                NLOHMANN_JSON_ABI_TAGS,              \
                NLOHMANN_JSON_NAMESPACE_VERSION)     \
    {
#endif

#ifndef NLOHMANN_JSON_NAMESPACE_END
#define NLOHMANN_JSON_NAMESPACE_END                                     \
    }  /* namespace (inline namespace) NOLINT(readability/namespace) */ \
    }  // namespace nlohmann
#endif

// #include <nlohmann/detail/conversions/from_json.hpp>
//     __ _____ _____ _____
//  __|  |   __|     |   | |  JSON for Modern C++
// |  |  |__   |  |  | | | |  version 3.12.0
// |_____|_____|_____|_|___|  https://github.com/nlohmann/json
//
// SPDX-FileCopyrightText: 2013 - 2025 Niels Lohmann <https://nlohmann.me>
// SPDX-License-Identifier: MIT



#include <algorithm> // transform
#include <array> // array
#include <forward_list> // forward_list
#include <iterator> // inserter, front_inserter, end
#include <map> // map
#ifdef JSON_HAS_CPP_17
    #include <optional> // optional
#endif
#include <string> // string
#include <tuple> // tuple, make_tuple
#include <type_traits> // is_arithmetic, is_same, is_enum, underlying_type, is_convertible
#include <unordered_map> // unordered_map
#include <utility> // pair, declval
#include <valarray> // valarray

// #include <nlohmann/detail/exceptions.hpp>
//     __ _____ _____ _____
//  __|  |   __|     |   | |  JSON for Modern C++
// |  |  |__   |  |  | | | |  version 3.12.0
// |_____|_____|_____|_|___|  https://github.com/nlohmann/json
//
// SPDX-FileCopyrightText: 2013 - 2025 Niels Lohmann <https://nlohmann.me>
// SPDX-License-Identifier: MIT



#include <cstddef> // nullptr_t
#include <exception> // exception
#if JSON_DIAGNOSTICS
    #include <numeric> // accumulate
#endif
#include <stdexcept> // runtime_error
#include <string> // to_string
#include <vector> // vector

// #include <nlohmann/detail/value_t.hpp>
//     __ _____ _____ _____
//  __|  |   __|     |   | |  JSON for Modern C++
// |  |  |__   |  |  | | | |  version 3.12.0
// |_____|_____|_____|_|___|  https://github.com/nlohmann/json
//
// SPDX-FileCopyrightText: 2013 - 2025 Niels Lohmann <https://nlohmann.me>
// SPDX-License-Identifier: MIT



#include <array> // array
#include <cstddef> // size_t
#include <cstdint> // uint8_t
#include <string> // string

// #include <nlohmann/detail/macro_scope.hpp>
//     __ _____ _____ _____
//  __|  |   __|     |   | |  JSON for Modern C++
// |  |  |__   |  |  | | | |  version 3.12.0
// |_____|_____|_____|_|___|  https://github.com/nlohmann/json
//
// SPDX-FileCopyrightText: 2013 - 2025 Niels Lohmann <https://nlohmann.me>
// SPDX-License-Identifier: MIT



#include <utility> // declval, pair
// #include <nlohmann/detail/meta/detected.hpp>
//     __ _____ _____ _____
//  __|  |   __|     |   | |  JSON for Modern C++
// |  |  |__   |  |  | | | |  version 3.12.0
// |_____|_____|_____|_|___|  https://github.com/nlohmann/json
//
// SPDX-FileCopyrightText: 2013 - 2025 Niels Lohmann <https://nlohmann.me>
// SPDX-License-Identifier: MIT



#include <type_traits>

// #include <nlohmann/detail/meta/void_t.hpp>
//     __ _____ _____ _____
//  __|  |   __|     |   | |  JSON for Modern C++
// |  |  |__   |  |  | | | |  version 3.12.0
// |_____|_____|_____|_|___|  https://github.com/nlohmann/json
//
// SPDX-FileCopyrightText: 2013 - 2025 Niels Lohmann <https://nlohmann.me>
// SPDX-License-Identifier: MIT



// #include <nlohmann/detail/abi_macros.hpp>


NLOHMANN_JSON_NAMESPACE_BEGIN
namespace detail
{

template<typename ...Ts> struct make_void
{
    using type = void;
};
template<typename ...Ts> using void_t = typename make_void<Ts...>::type;

}  // namespace detail
NLOHMANN_JSON_NAMESPACE_END


NLOHMANN_JSON_NAMESPACE_BEGIN
namespace detail
{

// https://en.cppreference.com/w/cpp/experimental/is_detected
struct nonesuch
{
    nonesuch() = delete;
    ~nonesuch() = delete;
    nonesuch(nonesuch const&) = delete;
    nonesuch(nonesuch const&&) = delete;
    void operator=(nonesuch const&) = delete;
    void operator=(nonesuch&&) = delete;
};

template<class Default,
         class AlwaysVoid,
         template<class...> class Op,
         class... Args>
struct detector
{
    using value_t = std::false_type;
    using type = Default;
};

template<class Default, template<class...> class Op, class... Args>
struct detector<Default, void_t<Op<Args...>>, Op, Args...>
{
    using value_t = std::true_type;
    using type = Op<Args...>;
};

template<template<class...> class Op, class... Args>
using is_detected = typename detector<nonesuch, void, Op, Args...>::value_t;

template<template<class...> class Op, class... Args>
struct is_detected_lazy : is_detected<Op, Args...> { };

template<template<class...> class Op, class... Args>
using detected_t = typename detector<nonesuch, void, Op, Args...>::type;

template<class Default, template<class...> class Op, class... Args>
using detected_or = detector<Default, void, Op, Args...>;

template<class Default, template<class...> class Op, class... Args>
using detected_or_t = typename detected_or<Default, Op, Args...>::type;

template<class Expected, template<class...> class Op, class... Args>
using is_detected_exact = std::is_same<Expected, detected_t<Op, Args...>>;

template<class To, template<class...> class Op, class... Args>
using is_detected_convertible =
    std::is_convertible<detected_t<Op, Args...>, To>;

}  // namespace detail
NLOHMANN_JSON_NAMESPACE_END

// #include <nlohmann/thirdparty/hedley/hedley.hpp>


//     __ _____ _____ _____
//  __|  |   __|     |   | |  JSON for Modern C++
// |  |  |__   |  |  | | | |  version 3.12.0
// |_____|_____|_____|_|___|  https://github.com/nlohmann/json
//
// SPDX-FileCopyrightText: 2013 - 2025 Niels Lohmann <https://nlohmann.me>
// SPDX-FileCopyrightText: 2016 - 2021 Evan Nemerson <evan@nemerson.com>
// SPDX-License-Identifier: MIT

/* Hedley - https://nemequ.github.io/hedley
 * Created by Evan Nemerson <evan@nemerson.com>
 */

#if !defined(JSON_HEDLEY_VERSION) || (JSON_HEDLEY_VERSION < 15)
#if defined(JSON_HEDLEY_VERSION)
    #undef JSON_HEDLEY_VERSION
#endif
#define JSON_HEDLEY_VERSION 15

#if defined(JSON_HEDLEY_STRINGIFY_EX)
    #undef JSON_HEDLEY_STRINGIFY_EX
#endif
#define JSON_HEDLEY_STRINGIFY_EX(x) #x

#if defined(JSON_HEDLEY_STRINGIFY)
    #undef JSON_HEDLEY_STRINGIFY
#endif
#define JSON_HEDLEY_STRINGIFY(x) JSON_HEDLEY_STRINGIFY_EX(x)

#if defined(JSON_HEDLEY_CONCAT_EX)
    #undef JSON_HEDLEY_CONCAT_EX
#endif
#define JSON_HEDLEY_CONCAT_EX(a,b) a##b

#if defined(JSON_HEDLEY_CONCAT)
    #undef JSON_HEDLEY_CONCAT
#endif
#define JSON_HEDLEY_CONCAT(a,b) JSON_HEDLEY_CONCAT_EX(a,b)

#if defined(JSON_HEDLEY_CONCAT3_EX)
    #undef JSON_HEDLEY_CONCAT3_EX
#endif
#define JSON_HEDLEY_CONCAT3_EX(a,b,c) a##b##c

#if defined(JSON_HEDLEY_CONCAT3)
    #undef JSON_HEDLEY_CONCAT3
#endif
#define JSON_HEDLEY_CONCAT3(a,b,c) JSON_HEDLEY_CONCAT3_EX(a,b,c)

#if defined(JSON_HEDLEY_VERSION_ENCODE)
    #undef JSON_HEDLEY_VERSION_ENCODE
#endif
#define JSON_HEDLEY_VERSION_ENCODE(major,minor,revision) (((major) * 1000000) + ((minor) * 1000) + (revision))

#if defined(JSON_HEDLEY_VERSION_DECODE_MAJOR)
    #undef JSON_HEDLEY_VERSION_DECODE_MAJOR
#endif
#define JSON_HEDLEY_VERSION_DECODE_MAJOR(version) ((version) / 1000000)

#if defined(JSON_HEDLEY_VERSION_DECODE_MINOR)
    #undef JSON_HEDLEY_VERSION_DECODE_MINOR
#endif
#define JSON_HEDLEY_VERSION_DECODE_MINOR(version) (((version) % 1000000) / 1000)

#if defined(JSON_HEDLEY_VERSION_DECODE_REVISION)
    #undef JSON_HEDLEY_VERSION_DECODE_REVISION
#endif
#define JSON_HEDLEY_VERSION_DECODE_REVISION(version) ((version) % 1000)

#if defined(JSON_HEDLEY_GNUC_VERSION)
    #undef JSON_HEDLEY_GNUC_VERSION
#endif
#if defined(__GNUC__) && defined(__GNUC_PATCHLEVEL__)
    #define JSON_HEDLEY_GNUC_VERSION JSON_HEDLEY_VERSION_ENCODE(__GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__)
#elif defined(__GNUC__)
    #define JSON_HEDLEY_GNUC_VERSION JSON_HEDLEY_VERSION_ENCODE(__GNUC__, __GNUC_MINOR__, 0)
#endif

#if defined(JSON_HEDLEY_GNUC_VERSION_CHECK)
    #undef JSON_HEDLEY_GNUC_VERSION_CHECK
#endif
#if defined(JSON_HEDLEY_GNUC_VERSION)
    #define JSON_HEDLEY_GNUC_VERSION_CHECK(major,minor,patch) (JSON_HEDLEY_GNUC_VERSION >= JSON_HEDLEY_VERSION_ENCODE(major, minor, patch))
#else
    #define JSON_HEDLEY_GNUC_VERSION_CHECK(major,minor,patch) (0)
#endif

#if defined(JSON_HEDLEY_MSVC_VERSION)
    #undef JSON_HEDLEY_MSVC_VERSION
#endif
#if defined(_MSC_FULL_VER) && (_MSC_FULL_VER >= 140000000) && !defined(__ICL)
    #define JSON_HEDLEY_MSVC_VERSION JSON_HEDLEY_VERSION_ENCODE(_MSC_FULL_VER / 10000000, (_MSC_FULL_VER % 10000000) / 100000, (_MSC_FULL_VER % 100000) / 100)
#elif defined(_MSC_FULL_VER) && !defined(__ICL)
    #define JSON_HEDLEY_MSVC_VERSION JSON_HEDLEY_VERSION_ENCODE(_MSC_FULL_VER / 1000000, (_MSC_FULL_VER % 1000000) / 10000, (_MSC_FULL_VER % 10000) / 10)
#elif defined(_MSC_VER) && !defined(__ICL)
    #define JSON_HEDLEY_MSVC_VERSION JSON_HEDLEY_VERSION_ENCODE(_MSC_VER / 100, _MSC_VER % 100, 0)
#endif

#if defined(JSON_HEDLEY_MSVC_VERSION_CHECK)
    #undef JSON_HEDLEY_MSVC_VERSION_CHECK
#endif
#if !defined(JSON_HEDLEY_MSVC_VERSION)
    #define JSON_HEDLEY_MSVC_VERSION_CHECK(major,minor,patch) (0)
#elif defined(_MSC_VER) && (_MSC_VER >= 1400)
    #define JSON_HEDLEY_MSVC_VERSION_CHECK(major,minor,patch) (_MSC_FULL_VER >= ((major * 10000000) + (minor * 100000) + (patch)))
#elif defined(_MSC_VER) && (_MSC_VER >= 1200)
    #define JSON_HEDLEY_MSVC_VERSION_CHECK(major,minor,patch) (_MSC_FULL_VER >= ((major * 1000000) + (minor * 10000) + (patch)))
#else
    #define JSON_HEDLEY_MSVC_VERSION_CHECK(major,minor,patch) (_MSC_VER >= ((major * 100) + (minor)))
#endif

#if defined(JSON_HEDLEY_INTEL_VERSION)
    #undef JSON_HEDLEY_INTEL_VERSION
#endif
#if defined(__INTEL_COMPILER) && defined(__INTEL_COMPILER_UPDATE) && !defined(__ICL)
    #define JSON_HEDLEY_INTEL_VERSION JSON_HEDLEY_VERSION_ENCODE(__INTEL_COMPILER / 100, __INTEL_COMPILER % 100, __INTEL_COMPILER_UPDATE)
#elif defined(__INTEL_COMPILER) && !defined(__ICL)
    #define JSON_HEDLEY_INTEL_VERSION JSON_HEDLEY_VERSION_ENCODE(__INTEL_COMPILER / 100, __INTEL_COMPILER % 100, 0)
#endif

#if defined(JSON_HEDLEY_INTEL_VERSION_CHECK)
    #undef JSON_HEDLEY_INTEL_VERSION_CHECK
#endif
#if defined(JSON_HEDLEY_INTEL_VERSION)
    #define JSON_HEDLEY_INTEL_VERSION_CHECK(major,minor,patch) (JSON_HEDLEY_INTEL_VERSION >= JSON_HEDLEY_VERSION_ENCODE(major, minor, patch))
#else
    #define JSON_HEDLEY_INTEL_VERSION_CHECK(major,minor,patch) (0)
#endif

#if defined(JSON_HEDLEY_INTEL_CL_VERSION)
    #undef JSON_HEDLEY_INTEL_CL_VERSION
#endif
#if defined(__INTEL_COMPILER) && defined(__INTEL_COMPILER_UPDATE) && defined(__ICL)
    #define JSON_HEDLEY_INTEL_CL_VERSION JSON_HEDLEY_VERSION_ENCODE(__INTEL_COMPILER, __INTEL_COMPILER_UPDATE, 0)
#endif

#if defined(JSON_HEDLEY_INTEL_CL_VERSION_CHECK)
    #undef JSON_HEDLEY_INTEL_CL_VERSION_CHECK
#endif
#if defined(JSON_HEDLEY_INTEL_CL_VERSION)
    #define JSON_HEDLEY_INTEL_CL_VERSION_CHECK(major,minor,patch) (JSON_HEDLEY_INTEL_CL_VERSION >= JSON_HEDLEY_VERSION_ENCODE(major, minor, patch))
#else
    #define JSON_HEDLEY_INTEL_CL_VERSION_CHECK(major,minor,patch) (0)
#endif

#if defined(JSON_HEDLEY_PGI_VERSION)
    #undef JSON_HEDLEY_PGI_VERSION
#endif
#if defined(__PGI) && defined(__PGIC__) && defined(__PGIC_MINOR__) && defined(__PGIC_PATCHLEVEL__)
    #define JSON_HEDLEY_PGI_VERSION JSON_HEDLEY_VERSION_ENCODE(__PGIC__, __PGIC_MINOR__, __PGIC_PATCHLEVEL__)
#endif

#if defined(JSON_HEDLEY_PGI_VERSION_CHECK)
    #undef JSON_HEDLEY_PGI_VERSION_CHECK
#endif
#if defined(JSON_HEDLEY_PGI_VERSION)
    #define JSON_HEDLEY_PGI_VERSION_CHECK(major,minor,patch) (JSON_HEDLEY_PGI_VERSION >= JSON_HEDLEY_VERSION_ENCODE(major, minor, patch))
#else
    #define JSON_HEDLEY_PGI_VERSION_CHECK(major,minor,patch) (0)
#endif

#if defined(JSON_HEDLEY_SUNPRO_VERSION)
    #undef JSON_HEDLEY_SUNPRO_VERSION
#endif
#if defined(__SUNPRO_C) && (__SUNPRO_C > 0x1000)
    #define JSON_HEDLEY_SUNPRO_VERSION JSON_HEDLEY_VERSION_ENCODE((((__SUNPRO_C >> 16) & 0xf) * 10) + ((__SUNPRO_C >> 12) & 0xf), (((__SUNPRO_C >> 8) & 0xf) * 10) + ((__SUNPRO_C >> 4) & 0xf), (__SUNPRO_C & 0xf) * 10)
#elif defined(__SUNPRO_C)
    #define JSON_HEDLEY_SUNPRO_VERSION JSON_HEDLEY_VERSION_ENCODE((__SUNPRO_C >> 8) & 0xf, (__SUNPRO_C >> 4) & 0xf, (__SUNPRO_C) & 0xf)
#elif defined(__SUNPRO_CC) && (__SUNPRO_CC > 0x1000)
    #define JSON_HEDLEY_SUNPRO_VERSION JSON_HEDLEY_VERSION_ENCODE((((__SUNPRO_CC >> 16) & 0xf) * 10) + ((__SUNPRO_CC >> 12) & 0xf), (((__SUNPRO_CC >> 8) & 0xf) * 10) + ((__SUNPRO_CC >> 4) & 0xf), (__SUNPRO_CC & 0xf) * 10)
#elif defined(__SUNPRO_CC)
    #define JSON_HEDLEY_SUNPRO_VERSION JSON_HEDLEY_VERSION_ENCODE((__SUNPRO_CC >> 8) & 0xf, (__SUNPRO_CC >> 4) & 0xf, (__SUNPRO_CC) & 0xf)
#endif

#if defined(JSON_HEDLEY_SUNPRO_VERSION_CHECK)
    #undef JSON_HEDLEY_SUNPRO_VERSION_CHECK
#endif
#if defined(JSON_HEDLEY_SUNPRO_VERSION)
    #define JSON_HEDLEY_SUNPRO_VERSION_CHECK(major,minor,patch) (JSON_HEDLEY_SUNPRO_VERSION >= JSON_HEDLEY_VERSION_ENCODE(major, minor, patch))
#else
    #define JSON_HEDLEY_SUNPRO_VERSION_CHECK(major,minor,patch) (0)
#endif

#if defined(JSON_HEDLEY_EMSCRIPTEN_VERSION)
    #undef JSON_HEDLEY_EMSCRIPTEN_VERSION
#endif
#if defined(__EMSCRIPTEN__)
    #define JSON_HEDLEY_EMSCRIPTEN_VERSION JSON_HEDLEY_VERSION_ENCODE(__EMSCRIPTEN_major__, __EMSCRIPTEN_minor__, __EMSCRIPTEN_tiny__)
#endif

#if defined(JSON_HEDLEY_EMSCRIPTEN_VERSION_CHECK)
    #undef JSON_HEDLEY_EMSCRIPTEN_VERSION_CHECK
#endif
#if defined(JSON_HEDLEY_EMSCRIPTEN_VERSION)
    #define JSON_HEDLEY_EMSCRIPTEN_VERSION_CHECK(major,minor,patch) (JSON_HEDLEY_EMSCRIPTEN_VERSION >= JSON_HEDLEY_VERSION_ENCODE(major, minor, patch))
#else
    #define JSON_HEDLEY_EMSCRIPTEN_VERSION_CHECK(major,minor,patch) (0)
#endif

#if defined(JSON_HEDLEY_ARM_VERSION)
    #undef JSON_HEDLEY_ARM_VERSION
#endif
#if defined(__CC_ARM) && defined(__ARMCOMPILER_VERSION)
    #define JSON_HEDLEY_ARM_VERSION JSON_HEDLEY_VERSION_ENCODE(__ARMCOMPILER_VERSION / 1000000, (__ARMCOMPILER_VERSION % 1000000) / 10000, (__ARMCOMPILER_VERSION % 10000) / 100)
#elif defined(__CC_ARM) && defined(__ARMCC_VERSION)
    #define JSON_HEDLEY_ARM_VERSION JSON_HEDLEY_VERSION_ENCODE(__ARMCC_VERSION / 1000000, (__ARMCC_VERSION % 1000000) / 10000, (__ARMCC_VERSION % 10000) / 100)
#endif

#if defined(JSON_HEDLEY_ARM_VERSION_CHECK)
    #undef JSON_HEDLEY_ARM_VERSION_CHECK
#endif
#if defined(JSON_HEDLEY_ARM_VERSION)
    #define JSON_HEDLEY_ARM_VERSION_CHECK(major,minor,patch) (JSON_HEDLEY_ARM_VERSION >= JSON_HEDLEY_VERSION_ENCODE(major, minor, patch))
#else
    #define JSON_HEDLEY_ARM_VERSION_CHECK(major,minor,patch) (0)
#endif

#if defined(JSON_HEDLEY_IBM_VERSION)
    #undef JSON_HEDLEY_IBM_VERSION
#endif
#if defined(__ibmxl__)
    #define JSON_HEDLEY_IBM_VERSION JSON_HEDLEY_VERSION_ENCODE(__ibmxl_version__, __ibmxl_release__, __ibmxl_modification__)
#elif defined(__xlC__) && defined(__xlC_ver__)
    #define JSON_HEDLEY_IBM_VERSION JSON_HEDLEY_VERSION_ENCODE(__xlC__ >> 8, __xlC__ & 0xff, (__xlC_ver__ >> 8) & 0xff)
#elif defined(__xlC__)
    #define JSON_HEDLEY_IBM_VERSION JSON_HEDLEY_VERSION_ENCODE(__xlC__ >> 8, __xlC__ & 0xff, 0)
#endif

#if defined(JSON_HEDLEY_IBM_VERSION_CHECK)
    #undef JSON_HEDLEY_IBM_VERSION_CHECK
#endif
#if defined(JSON_HEDLEY_IBM_VERSION)
    #define JSON_HEDLEY_IBM_VERSION_CHECK(major,minor,patch) (JSON_HEDLEY_IBM_VERSION >= JSON_HEDLEY_VERSION_ENCODE(major, minor, patch))
#else
    #define JSON_HEDLEY_IBM_VERSION_CHECK(major,minor,patch) (0)
#endif

#if defined(JSON_HEDLEY_TI_VERSION)
    #undef JSON_HEDLEY_TI_VERSION
#endif
#if \
    defined(__TI_COMPILER_VERSION__) && \
    ( \
      defined(__TMS470__) || defined(__TI_ARM__) || \
      defined(__MSP430__) || \
      defined(__TMS320C2000__) \
    )
#if (__TI_COMPILER_VERSION__ >= 16000000)
    #define JSON_HEDLEY_TI_VERSION JSON_HEDLEY_VERSION_ENCODE(__TI_COMPILER_VERSION__ / 1000000, (__TI_COMPILER_VERSION__ % 1000000) / 1000, (__TI_COMPILER_VERSION__ % 1000))
#endif
#endif

#if defined(JSON_HEDLEY_TI_VERSION_CHECK)
    #undef JSON_HEDLEY_TI_VERSION_CHECK
#endif
#if defined(JSON_HEDLEY_TI_VERSION)
    #define JSON_HEDLEY_TI_VERSION_CHECK(major,minor,patch) (JSON_HEDLEY_TI_VERSION >= JSON_HEDLEY_VERSION_ENCODE(major, minor, patch))
#else
    #define JSON_HEDLEY_TI_VERSION_CHECK(major,minor,patch) (0)
#endif

#if defined(JSON_HEDLEY_TI_CL2000_VERSION)
    #undef JSON_HEDLEY_TI_CL2000_VERSION
#endif
#if defined(__TI_COMPILER_VERSION__) && defined(__TMS320C2000__)
    #define JSON_HEDLEY_TI_CL2000_VERSION JSON_HEDLEY_VERSION_ENCODE(__TI_COMPILER_VERSION__ / 1000000, (__TI_COMPILER_VERSION__ % 1000000) / 1000, (__TI_COMPILER_VERSION__ % 1000))
#endif

#if defined(JSON_HEDLEY_TI_CL2000_VERSION_CHECK)
    #undef JSON_HEDLEY_TI_CL2000_VERSION_CHECK
#endif
#if defined(JSON_HEDLEY_TI_CL2000_VERSION)
    #define JSON_HEDLEY_TI_CL2000_VERSION_CHECK(major,minor,patch) (JSON_HEDLEY_TI_CL2000_VERSION >= JSON_HEDLEY_VERSION_ENCODE(major, minor, patch))
#else
    #define JSON_HEDLEY_TI_CL2000_VERSION_CHECK(major,minor,patch) (0)
#endif

#if defined(JSON_HEDLEY_TI_CL430_VERSION)
    #undef JSON_HEDLEY_TI_CL430_VERSION
#endif
#if defined(__TI_COMPILER_VERSION__) && defined(__MSP430__)
    #define JSON_HEDLEY_TI_CL430_VERSION JSON_HEDLEY_VERSION_ENCODE(__TI_COMPILER_VERSION__ / 1000000, (__TI_COMPILER_VERSION__ % 1000000) / 1000, (__TI_COMPILER_VERSION__ % 1000))
#endif

#if defined(JSON_HEDLEY_TI_CL430_VERSION_CHECK)
    #undef JSON_HEDLEY_TI_CL430_VERSION_CHECK
#endif
#if defined(JSON_HEDLEY_TI_CL430_VERSION)
    #define JSON_HEDLEY_TI_CL430_VERSION_CHECK(major,minor,patch) (JSON_HEDLEY_TI_CL430_VERSION >= JSON_HEDLEY_VERSION_ENCODE(major, minor, patch))
#else
    #define JSON_HEDLEY_TI_CL430_VERSION_CHECK(major,minor,patch) (0)
#endif

#if defined(JSON_HEDLEY_TI_ARMCL_VERSION)
    #undef JSON_HEDLEY_TI_ARMCL_VERSION
#endif
#if defined(__TI_COMPILER_VERSION__) && (defined(__TMS470__) || defined(__TI_ARM__))
    #define JSON_HEDLEY_TI_ARMCL_VERSION JSON_HEDLEY_VERSION_ENCODE(__TI_COMPILER_VERSION__ / 1000000, (__TI_COMPILER_VERSION__ % 1000000) / 1000, (__TI_COMPILER_VERSION__ % 1000))
#endif

#if defined(JSON_HEDLEY_TI_ARMCL_VERSION_CHECK)
    #undef JSON_HEDLEY_TI_ARMCL_VERSION_CHECK
#endif
#if defined(JSON_HEDLEY_TI_ARMCL_VERSION)
    #define JSON_HEDLEY_TI_ARMCL_VERSION_CHECK(major,minor,patch) (JSON_HEDLEY_TI_ARMCL_VERSION >= JSON_HEDLEY_VERSION_ENCODE(major, minor, patch))
#else
    #define JSON_HEDLEY_TI_ARMCL_VERSION_CHECK(major,minor,patch) (0)
#endif

#if defined(JSON_HEDLEY_TI_CL6X_VERSION)
    #undef JSON_HEDLEY_TI_CL6X_VERSION
#endif
#if defined(__TI_COMPILER_VERSION__) && defined(__TMS320C6X__)
    #define JSON_HEDLEY_TI_CL6X_VERSION JSON_HEDLEY_VERSION_ENCODE(__TI_COMPILER_VERSION__ / 1000000, (__TI_COMPILER_VERSION__ % 1000000) / 1000, (__TI_COMPILER_VERSION__ % 1000))
#endif

#if defined(JSON_HEDLEY_TI_CL6X_VERSION_CHECK)
    #undef JSON_HEDLEY_TI_CL6X_VERSION_CHECK
#endif
#if defined(JSON_HEDLEY_TI_CL6X_VERSION)
    #define JSON_HEDLEY_TI_CL6X_VERSION_CHECK(major,minor,patch) (JSON_HEDLEY_TI_CL6X_VERSION >= JSON_HEDLEY_VERSION_ENCODE(major, minor, patch))
#else
    #define JSON_HEDLEY_TI_CL6X_VERSION_CHECK(major,minor,patch) (0)
#endif

#if defined(JSON_HEDLEY_TI_CL7X_VERSION)
    #undef JSON_HEDLEY_TI_CL7X_VERSION
#endif
#if defined(__TI_COMPILER_VERSION__) && defined(__C7000__)
    #define JSON_HEDLEY_TI_CL7X_VERSION JSON_HEDLEY_VERSION_ENCODE(__TI_COMPILER_VERSION__ / 1000000, (__TI_COMPILER_VERSION__ % 1000000) / 1000, (__TI_COMPILER_VERSION__ % 1000))
#endif

#if defined(JSON_HEDLEY_TI_CL7X_VERSION_CHECK)
    #undef JSON_HEDLEY_TI_CL7X_VERSION_CHECK
#endif
#if defined(JSON_HEDLEY_TI_CL7X_VERSION)
    #define JSON_HEDLEY_TI_CL7X_VERSION_CHECK(major,minor,patch) (JSON_HEDLEY_TI_CL7X_VERSION >= JSON_HEDLEY_VERSION_ENCODE(major, minor, patch))
#else
    #define JSON_HEDLEY_TI_CL7X_VERSION_CHECK(major,minor,patch) (0)
#endif

#if defined(JSON_HEDLEY_TI_CLPRU_VERSION)
    #undef JSON_HEDLEY_TI_CLPRU_VERSION
#endif
#if defined(__TI_COMPILER_VERSION__) && defined(__PRU__)
    #define JSON_HEDLEY_TI_CLPRU_VERSION JSON_HEDLEY_VERSION_ENCODE(__TI_COMPILER_VERSION__ / 1000000, (__TI_COMPILER_VERSION__ % 1000000) / 1000, (__TI_COMPILER_VERSION__ % 1000))
#endif

#if defined(JSON_HEDLEY_TI_CLPRU_VERSION_CHECK)
    #undef JSON_HEDLEY_TI_CLPRU_VERSION_CHECK
#endif
#if defined(JSON_HEDLEY_TI_CLPRU_VERSION)
    #define JSON_HEDLEY_TI_CLPRU_VERSION_CHECK(major,minor,patch) (JSON_HEDLEY_TI_CLPRU_VERSION >= JSON_HEDLEY_VERSION_ENCODE(major, minor, patch))
#else
    #define JSON_HEDLEY_TI_CLPRU_VERSION_CHECK(major,minor,patch) (0)
#endif

#if defined(JSON_HEDLEY_CRAY_VERSION)
    #undef JSON_HEDLEY_CRAY_VERSION
#endif
#if defined(_CRAYC)
    #if defined(_RELEASE_PATCHLEVEL)
        #define JSON_HEDLEY_CRAY_VERSION JSON_HEDLEY_VERSION_ENCODE(_RELEASE_MAJOR, _RELEASE_MINOR, _RELEASE_PATCHLEVEL)
    #else
        #define JSON_HEDLEY_CRAY_VERSION JSON_HEDLEY_VERSION_ENCODE(_RELEASE_MAJOR, _RELEASE_MINOR, 0)
    #endif
#endif

#if defined(JSON_HEDLEY_CRAY_VERSION_CHECK)
    #undef JSON_HEDLEY_CRAY_VERSION_CHECK
#endif
#if defined(JSON_HEDLEY_CRAY_VERSION)
    #define JSON_HEDLEY_CRAY_VERSION_CHECK(major,minor,patch) (JSON_HEDLEY_CRAY_VERSION >= JSON_HEDLEY_VERSION_ENCODE(major, minor, patch))
#else
    #define JSON_HEDLEY_CRAY_VERSION_CHECK(major,minor,patch) (0)
#endif

#if defined(JSON_HEDLEY_IAR_VERSION)
    #undef JSON_HEDLEY_IAR_VERSION
#endif
#if defined(__IAR_SYSTEMS_ICC__)
    #if __VER__ > 1000
        #define JSON_HEDLEY_IAR_VERSION JSON_HEDLEY_VERSION_ENCODE((__VER__ / 1000000), ((__VER__ / 1000) % 1000), (__VER__ % 1000))
    #else
        #define JSON_HEDLEY_IAR_VERSION JSON_HEDLEY_VERSION_ENCODE(__VER__ / 100, __VER__ % 100, 0)
    #endif
#endif

#if defined(JSON_HEDLEY_IAR_VERSION_CHECK)
    #undef JSON_HEDLEY_IAR_VERSION_CHECK
#endif
#if defined(JSON_HEDLEY_IAR_VERSION)
    #define JSON_HEDLEY_IAR_VERSION_CHECK(major,minor,patch) (JSON_HEDLEY_IAR_VERSION >= JSON_HEDLEY_VERSION_ENCODE(major, minor, patch))
#else
    #define JSON_HEDLEY_IAR_VERSION_CHECK(major,minor,patch) (0)
#endif

#if defined(JSON_HEDLEY_TINYC_VERSION)
    #undef JSON_HEDLEY_TINYC_VERSION
#endif
#if defined(__TINYC__)
    #define JSON_HEDLEY_TINYC_VERSION JSON_HEDLEY_VERSION_ENCODE(__TINYC__ / 1000, (__TINYC__ / 100) % 10, __TINYC__ % 100)
#endif

#if defined(JSON_HEDLEY_TINYC_VERSION_CHECK)
    #undef JSON_HEDLEY_TINYC_VERSION_CHECK
#endif
#if defined(JSON_HEDLEY_TINYC_VERSION)
    #define JSON_HEDLEY_TINYC_VERSION_CHECK(major,minor,patch) (JSON_HEDLEY_TINYC_VERSION >= JSON_HEDLEY_VERSION_ENCODE(major, minor, patch))
#else
    #define JSON_HEDLEY_TINYC_VERSION_CHECK(major,minor,patch) (0)
#endif

#if defined(JSON_HEDLEY_DMC_VERSION)
    #undef JSON_HEDLEY_DMC_VERSION
#endif
#if defined(__DMC__)
    #define JSON_HEDLEY_DMC_VERSION JSON_HEDLEY_VERSION_ENCODE(__DMC__ >> 8, (__DMC__ >> 4) & 0xf, __DMC__ & 0xf)
#endif

#if defined(JSON_HEDLEY_DMC_VERSION_CHECK)
    #undef JSON_HEDLEY_DMC_VERSION_CHECK
#endif
#if defined(JSON_HEDLEY_DMC_VERSION)
    #define JSON_HEDLEY_DMC_VERSION_CHECK(major,minor,patch) (JSON_HEDLEY_DMC_VERSION >= JSON_HEDLEY_VERSION_ENCODE(major, minor, patch))
#else
    #define JSON_HEDLEY_DMC_VERSION_CHECK(major,minor,patch) (0)
#endif

#if defined(JSON_HEDLEY_COMPCERT_VERSION)
    #undef JSON_HEDLEY_COMPCERT_VERSION
#endif
#if defined(__COMPCERT_VERSION__)
    #define JSON_HEDLEY_COMPCERT_VERSION JSON_HEDLEY_VERSION_ENCODE(__COMPCERT_VERSION__ / 10000, (__COMPCERT_VERSION__ / 100) % 100, __COMPCERT_VERSION__ % 100)
#endif

#if defined(JSON_HEDLEY_COMPCERT_VERSION_CHECK)
    #undef JSON_HEDLEY_COMPCERT_VERSION_CHECK
#endif
#if defined(JSON_HEDLEY_COMPCERT_VERSION)
    #define JSON_HEDLEY_COMPCERT_VERSION_CHECK(major,minor,patch) (JSON_HEDLEY_COMPCERT_VERSION >= JSON_HEDLEY_VERSION_ENCODE(major, minor, patch))
#else
    #define JSON_HEDLEY_COMPCERT_VERSION_CHECK(major,minor,patch) (0)
#endif

#if defined(JSON_HEDLEY_PELLES_VERSION)
    #undef JSON_HEDLEY_PELLES_VERSION
#endif
#if defined(__POCC__)
    #define JSON_HEDLEY_PELLES_VERSION JSON_HEDLEY_VERSION_ENCODE(__POCC__ / 100, __POCC__ % 100, 0)
#endif

#if defined(JSON_HEDLEY_PELLES_VERSION_CHECK)
    #undef JSON_HEDLEY_PELLES_VERSION_CHECK
#endif
#if defined(JSON_HEDLEY_PELLES_VERSION)
    #define JSON_HEDLEY_PELLES_VERSION_CHECK(major,minor,patch) (JSON_HEDLEY_PELLES_VERSION >= JSON_HEDLEY_VERSION_ENCODE(major, minor, patch))
#else
    #define JSON_HEDLEY_PELLES_VERSION_CHECK(major,minor,patch) (0)
#endif

#if defined(JSON_HEDLEY_MCST_LCC_VERSION)
    #undef JSON_HEDLEY_MCST_LCC_VERSION
#endif
#if defined(__LCC__) && defined(__LCC_MINOR__)
    #define JSON_HEDLEY_MCST_LCC_VERSION JSON_HEDLEY_VERSION_ENCODE(__LCC__ / 100, __LCC__ % 100, __LCC_MINOR__)
#endif

#if defined(JSON_HEDLEY_MCST_LCC_VERSION_CHECK)
    #undef JSON_HEDLEY_MCST_LCC_VERSION_CHECK
#endif
#if defined(JSON_HEDLEY_MCST_LCC_VERSION)
    #define JSON_HEDLEY_MCST_LCC_VERSION_CHECK(major,minor,patch) (JSON_HEDLEY_MCST_LCC_VERSION >= JSON_HEDLEY_VERSION_ENCODE(major, minor, patch))
#else
    #define JSON_HEDLEY_MCST_LCC_VERSION_CHECK(major,minor,patch) (0)
#endif

#if defined(JSON_HEDLEY_GCC_VERSION)
    #undef JSON_HEDLEY_GCC_VERSION
#endif
#if \
    defined(JSON_HEDLEY_GNUC_VERSION) && \
    !defined(__clang__) && \
    !defined(JSON_HEDLEY_INTEL_VERSION) && \
    !defined(JSON_HEDLEY_PGI_VERSION) && \
    !defined(JSON_HEDLEY_ARM_VERSION) && \
    !defined(JSON_HEDLEY_CRAY_VERSION) && \
    !defined(JSON_HEDLEY_TI_VERSION) && \
    !defined(JSON_HEDLEY_TI_ARMCL_VERSION) && \
    !defined(JSON_HEDLEY_TI_CL430_VERSION) && \
    !defined(JSON_HEDLEY_TI_CL2000_VERSION) && \
    !defined(JSON_HEDLEY_TI_CL6X_VERSION) && \
    !defined(JSON_HEDLEY_TI_CL7X_VERSION) && \
    !defined(JSON_HEDLEY_TI_CLPRU_VERSION) && \
    !defined(__COMPCERT__) && \
    !defined(JSON_HEDLEY_MCST_LCC_VERSION)
    #define JSON_HEDLEY_GCC_VERSION JSON_HEDLEY_GNUC_VERSION
#endif

#if defined(JSON_HEDLEY_GCC_VERSION_CHECK)
    #undef JSON_HEDLEY_GCC_VERSION_CHECK
#endif
#if defined(JSON_HEDLEY_GCC_VERSION)
    #define JSON_HEDLEY_GCC_VERSION_CHECK(major,minor,patch) (JSON_HEDLEY_GCC_VERSION >= JSON_HEDLEY_VERSION_ENCODE(major, minor, patch))
#else
    #define JSON_HEDLEY_GCC_VERSION_CHECK(major,minor,patch) (0)
#endif

#if defined(JSON_HEDLEY_HAS_ATTRIBUTE)
    #undef JSON_HEDLEY_HAS_ATTRIBUTE
#endif
#if \
  defined(__has_attribute) && \
  ( \
    (!defined(JSON_HEDLEY_IAR_VERSION) || JSON_HEDLEY_IAR_VERSION_CHECK(8,5,9)) \
  )
#  define JSON_HEDLEY_HAS_ATTRIBUTE(attribute) __has_attribute(attribute)
#else
#  define JSON_HEDLEY_HAS_ATTRIBUTE(attribute) (0)
#endif

#if defined(JSON_HEDLEY_GNUC_HAS_ATTRIBUTE)
    #undef JSON_HEDLEY_GNUC_HAS_ATTRIBUTE
#endif
#if defined(__has_attribute)
    #define JSON_HEDLEY_GNUC_HAS_ATTRIBUTE(attribute,major,minor,patch) JSON_HEDLEY_HAS_ATTRIBUTE(attribute)
#else
    #define JSON_HEDLEY_GNUC_HAS_ATTRIBUTE(attribute,major,minor,patch) JSON_HEDLEY_GNUC_VERSION_CHECK(major,minor,patch)
#endif

#if defined(JSON_HEDLEY_GCC_HAS_ATTRIBUTE)
    #undef JSON_HEDLEY_GCC_HAS_ATTRIBUTE
#endif
#if defined(__has_attribute)
    #define JSON_HEDLEY_GCC_HAS_ATTRIBUTE(attribute,major,minor,patch) JSON_HEDLEY_HAS_ATTRIBUTE(attribute)
#else
    #define JSON_HEDLEY_GCC_HAS_ATTRIBUTE(attribute,major,minor,patch) JSON_HEDLEY_GCC_VERSION_CHECK(major,minor,patch)
#endif

#if defined(JSON_HEDLEY_HAS_CPP_ATTRIBUTE)
    #undef JSON_HEDLEY_HAS_CPP_ATTRIBUTE
#endif
#if \
    defined(__has_cpp_attribute) && \
    defined(__cplusplus) && \
    (!defined(JSON_HEDLEY_SUNPRO_VERSION) || JSON_HEDLEY_SUNPRO_VERSION_CHECK(5,15,0))
    #define JSON_HEDLEY_HAS_CPP_ATTRIBUTE(attribute) __has_cpp_attribute(attribute)
#else
    #define JSON_HEDLEY_HAS_CPP_ATTRIBUTE(attribute) (0)
#endif

#if defined(JSON_HEDLEY_HAS_CPP_ATTRIBUTE_NS)
    #undef JSON_HEDLEY_HAS_CPP_ATTRIBUTE_NS
#endif
#if !defined(__cplusplus) || !defined(__has_cpp_attribute)
    #define JSON_HEDLEY_HAS_CPP_ATTRIBUTE_NS(ns,attribute) (0)
#elif \
    !defined(JSON_HEDLEY_PGI_VERSION) && \
    !defined(JSON_HEDLEY_IAR_VERSION) && \
    (!defined(JSON_HEDLEY_SUNPRO_VERSION) || JSON_HEDLEY_SUNPRO_VERSION_CHECK(5,15,0)) && \
    (!defined(JSON_HEDLEY_MSVC_VERSION) || JSON_HEDLEY_MSVC_VERSION_CHECK(19,20,0))
    #define JSON_HEDLEY_HAS_CPP_ATTRIBUTE_NS(ns,attribute) JSON_HEDLEY_HAS_CPP_ATTRIBUTE(ns::attribute)
#else
    #define JSON_HEDLEY_HAS_CPP_ATTRIBUTE_NS(ns,attribute) (0)
#endif

#if defined(JSON_HEDLEY_GNUC_HAS_CPP_ATTRIBUTE)
    #undef JSON_HEDLEY_GNUC_HAS_CPP_ATTRIBUTE
#endif
#if defined(__has_cpp_attribute) && defined(__cplusplus)
    #define JSON_HEDLEY_GNUC_HAS_CPP_ATTRIBUTE(attribute,major,minor,patch) __has_cpp_attribute(attribute)
#else
    #define JSON_HEDLEY_GNUC_HAS_CPP_ATTRIBUTE(attribute,major,minor,patch) JSON_HEDLEY_GNUC_VERSION_CHECK(major,minor,patch)
#endif

#if defined(JSON_HEDLEY_GCC_HAS_CPP_ATTRIBUTE)
    #undef JSON_HEDLEY_GCC_HAS_CPP_ATTRIBUTE
#endif
#if defined(__has_cpp_attribute) && defined(__cplusplus)
    #define JSON_HEDLEY_GCC_HAS_CPP_ATTRIBUTE(attribute,major,minor,patch) __has_cpp_attribute(attribute)
#else
    #define JSON_HEDLEY_GCC_HAS_CPP_ATTRIBUTE(attribute,major,minor,patch) JSON_HEDLEY_GCC_VERSION_CHECK(major,minor,patch)
#endif

#if defined(JSON_HEDLEY_HAS_BUILTIN)
    #undef JSON_HEDLEY_HAS_BUILTIN
#endif
#if defined(__has_builtin)
    #define JSON_HEDLEY_HAS_BUILTIN(builtin) __has_builtin(builtin)
#else
    #define JSON_HEDLEY_HAS_BUILTIN(builtin) (0)
#endif

#if defined(JSON_HEDLEY_GNUC_HAS_BUILTIN)
    #undef JSON_HEDLEY_GNUC_HAS_BUILTIN
#endif
#if defined(__has_builtin)
    #define JSON_HEDLEY_GNUC_HAS_BUILTIN(builtin,major,minor,patch) __has_builtin(builtin)
#else
    #define JSON_HEDLEY_GNUC_HAS_BUILTIN(builtin,major,minor,patch) JSON_HEDLEY_GNUC_VERSION_CHECK(major,minor,patch)
#endif

#if defined(JSON_HEDLEY_GCC_HAS_BUILTIN)
    #undef JSON_HEDLEY_GCC_HAS_BUILTIN
#endif
#if defined(__has_builtin)
    #define JSON_HEDLEY_GCC_HAS_BUILTIN(builtin,major,minor,patch) __has_builtin(builtin)
#else
    #define JSON_HEDLEY_GCC_HAS_BUILTIN(builtin,major,minor,patch) JSON_HEDLEY_GCC_VERSION_CHECK(major,minor,patch)
#endif

#if defined(JSON_HEDLEY_HAS_FEATURE)
    #undef JSON_HEDLEY_HAS_FEATURE
#endif
#if defined(__has_feature)
    #define JSON_HEDLEY_HAS_FEATURE(feature) __has_feature(feature)
#else
    #define JSON_HEDLEY_HAS_FEATURE(feature) (0)
#endif

#if defined(JSON_HEDLEY_GNUC_HAS_FEATURE)
    #undef JSON_HEDLEY_GNUC_HAS_FEATURE
#endif
#if defined(__has_feature)
    #define JSON_HEDLEY_GNUC_HAS_FEATURE(feature,major,minor,patch) __has_feature(feature)
#else
    #define JSON_HEDLEY_GNUC_HAS_FEATURE(feature,major,minor,patch) JSON_HEDLEY_GNUC_VERSION_CHECK(major,minor,patch)
#endif

#if defined(JSON_HEDLEY_GCC_HAS_FEATURE)
    #undef JSON_HEDLEY_GCC_HAS_FEATURE
#endif
#if defined(__has_feature)
    #define JSON_HEDLEY_GCC_HAS_FEATURE(feature,major,minor,patch) __has_feature(feature)
#else
    #define JSON_HEDLEY_GCC_HAS_FEATURE(feature,major,minor,patch) JSON_HEDLEY_GCC_VERSION_CHECK(major,minor,patch)
#endif

#if defined(JSON_HEDLEY_HAS_EXTENSION)
    #undef JSON_HEDLEY_HAS_EXTENSION
#endif
#if defined(__has_extension)
    #define JSON_HEDLEY_HAS_EXTENSION(extension) __has_extension(extension)
#else
    #define JSON_HEDLEY_HAS_EXTENSION(extension) (0)
#endif

#if defined(JSON_HEDLEY_GNUC_HAS_EXTENSION)
    #undef JSON_HEDLEY_GNUC_HAS_EXTENSION
#endif
#if defined(__has_extension)
    #define JSON_HEDLEY_GNUC_HAS_EXTENSION(extension,major,minor,patch) __has_extension(extension)
#else
    #define JSON_HEDLEY_GNUC_HAS_EXTENSION(extension,major,minor,patch) JSON_HEDLEY_GNUC_VERSION_CHECK(major,minor,patch)
#endif

#if defined(JSON_HEDLEY_GCC_HAS_EXTENSION)
    #undef JSON_HEDLEY_GCC_HAS_EXTENSION
#endif
#if defined(__has_extension)
    #define JSON_HEDLEY_GCC_HAS_EXTENSION(extension,major,minor,patch) __has_extension(extension)
#else
    #define JSON_HEDLEY_GCC_HAS_EXTENSION(extension,major,minor,patch) JSON_HEDLEY_GCC_VERSION_CHECK(major,minor,patch)
#endif

#if defined(JSON_HEDLEY_HAS_DECLSPEC_ATTRIBUTE)
    #undef JSON_HEDLEY_HAS_DECLSPEC_ATTRIBUTE
#endif
#if defined(__has_declspec_attribute)
    #define JSON_HEDLEY_HAS_DECLSPEC_ATTRIBUTE(attribute) __has_declspec_attribute(attribute)
#else
    #define JSON_HEDLEY_HAS_DECLSPEC_ATTRIBUTE(attribute) (0)
#endif

#if defined(JSON_HEDLEY_GNUC_HAS_DECLSPEC_ATTRIBUTE)
    #undef JSON_HEDLEY_GNUC_HAS_DECLSPEC_ATTRIBUTE
#endif
#if defined(__has_declspec_attribute)
    #define JSON_HEDLEY_GNUC_HAS_DECLSPEC_ATTRIBUTE(attribute,major,minor,patch) __has_declspec_attribute(attribute)
#else
    #define JSON_HEDLEY_GNUC_HAS_DECLSPEC_ATTRIBUTE(attribute,major,minor,patch) JSON_HEDLEY_GNUC_VERSION_CHECK(major,minor,patch)
#endif

#if defined(JSON_HEDLEY_GCC_HAS_DECLSPEC_ATTRIBUTE)
    #undef JSON_HEDLEY_GCC_HAS_DECLSPEC_ATTRIBUTE
#endif
#if defined(__has_declspec_attribute)
    #define JSON_HEDLEY_GCC_HAS_DECLSPEC_ATTRIBUTE(attribute,major,minor,patch) __has_declspec_attribute(attribute)
#else
    #define JSON_HEDLEY_GCC_HAS_DECLSPEC_ATTRIBUTE(attribute,major,minor,patch) JSON_HEDLEY_GCC_VERSION_CHECK(major,minor,patch)
#endif

#if defined(JSON_HEDLEY_HAS_WARNING)
    #undef JSON_HEDLEY_HAS_WARNING
#endif
#if defined(__has_warning)
    #define JSON_HEDLEY_HAS_WARNING(warning) __has_warning(warning)
#else
    #define JSON_HEDLEY_HAS_WARNING(warning) (0)
#endif

#if defined(JSON_HEDLEY_GNUC_HAS_WARNING)
    #undef JSON_HEDLEY_GNUC_HAS_WARNING
#endif
#if defined(__has_warning)
    #define JSON_HEDLEY_GNUC_HAS_WARNING(warning,major,minor,patch) __has_warning(warning)
#else
    #define JSON_HEDLEY_GNUC_HAS_WARNING(warning,major,minor,patch) JSON_HEDLEY_GNUC_VERSION_CHECK(major,minor,patch)
#endif

#if defined(JSON_HEDLEY_GCC_HAS_WARNING)
    #undef JSON_HEDLEY_GCC_HAS_WARNING
#endif
#if defined(__has_warning)
    #define JSON_HEDLEY_GCC_HAS_WARNING(warning,major,minor,patch) __has_warning(warning)
#else
    #define JSON_HEDLEY_GCC_HAS_WARNING(warning,major,minor,patch) JSON_HEDLEY_GCC_VERSION_CHECK(major,minor,patch)
#endif

#if \
    (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L)) || \
    defined(__clang__) || \
    JSON_HEDLEY_GCC_VERSION_CHECK(3,0,0) || \
    JSON_HEDLEY_INTEL_VERSION_CHECK(13,0,0) || \
    JSON_HEDLEY_IAR_VERSION_CHECK(8,0,0) || \
    JSON_HEDLEY_PGI_VERSION_CHECK(18,4,0) || \
    JSON_HEDLEY_ARM_VERSION_CHECK(4,1,0) || \
    JSON_HEDLEY_TI_VERSION_CHECK(15,12,0) || \
    JSON_HEDLEY_TI_ARMCL_VERSION_CHECK(4,7,0) || \
    JSON_HEDLEY_TI_CL430_VERSION_CHECK(2,0,1) || \
    JSON_HEDLEY_TI_CL2000_VERSION_CHECK(6,1,0) || \
    JSON_HEDLEY_TI_CL6X_VERSION_CHECK(7,0,0) || \
    JSON_HEDLEY_TI_CL7X_VERSION_CHECK(1,2,0) || \
    JSON_HEDLEY_TI_CLPRU_VERSION_CHECK(2,1,0) || \
    JSON_HEDLEY_CRAY_VERSION_CHECK(5,0,0) || \
    JSON_HEDLEY_TINYC_VERSION_CHECK(0,9,17) || \
    JSON_HEDLEY_SUNPRO_VERSION_CHECK(8,0,0) || \
    (JSON_HEDLEY_IBM_VERSION_CHECK(10,1,0) && defined(__C99_PRAGMA_OPERATOR))
    #define JSON_HEDLEY_PRAGMA(value) _Pragma(#value)
#elif JSON_HEDLEY_MSVC_VERSION_CHECK(15,0,0)
    #define JSON_HEDLEY_PRAGMA(value) __pragma(value)
#else
    #define JSON_HEDLEY_PRAGMA(value)
#endif

#if defined(JSON_HEDLEY_DIAGNOSTIC_PUSH)
    #undef JSON_HEDLEY_DIAGNOSTIC_PUSH
#endif
#if defined(JSON_HEDLEY_DIAGNOSTIC_POP)
    #undef JSON_HEDLEY_DIAGNOSTIC_POP
#endif
#if defined(__clang__)
    #define JSON_HEDLEY_DIAGNOSTIC_PUSH _Pragma("clang diagnostic push")
    #define JSON_HEDLEY_DIAGNOSTIC_POP _Pragma("clang diagnostic pop")
#elif JSON_HEDLEY_INTEL_VERSION_CHECK(13,0,0)
    #define JSON_HEDLEY_DIAGNOSTIC_PUSH _Pragma("warning(push)")
    #define JSON_HEDLEY_DIAGNOSTIC_POP _Pragma("warning(pop)")
#elif JSON_HEDLEY_GCC_VERSION_CHECK(4,6,0)
    #define JSON_HEDLEY_DIAGNOSTIC_PUSH _Pragma("GCC diagnostic push")
    #define JSON_HEDLEY_DIAGNOSTIC_POP _Pragma("GCC diagnostic pop")
#elif \
    JSON_HEDLEY_MSVC_VERSION_CHECK(15,0,0) || \
    JSON_HEDLEY_INTEL_CL_VERSION_CHECK(2021,1,0)
    #define JSON_HEDLEY_DIAGNOSTIC_PUSH __pragma(warning(push))
    #define JSON_HEDLEY_DIAGNOSTIC_POP __pragma(warning(pop))
#elif JSON_HEDLEY_ARM_VERSION_CHECK(5,6,0)
    #define JSON_HEDLEY_DIAGNOSTIC_PUSH _Pragma("push")
    #define JSON_HEDLEY_DIAGNOSTIC_POP _Pragma("pop")
#elif \
    JSON_HEDLEY_TI_VERSION_CHECK(15,12,0) || \
    JSON_HEDLEY_TI_ARMCL_VERSION_CHECK(5,2,0) || \
    JSON_HEDLEY_TI_CL430_VERSION_CHECK(4,4,0) || \
    JSON_HEDLEY_TI_CL6X_VERSION_CHECK(8,1,0) || \
    JSON_HEDLEY_TI_CL7X_VERSION_CHECK(1,2,0) || \
    JSON_HEDLEY_TI_CLPRU_VERSION_CHECK(2,1,0)
    #define JSON_HEDLEY_DIAGNOSTIC_PUSH _Pragma("diag_push")
    #define JSON_HEDLEY_DIAGNOSTIC_POP _Pragma("diag_pop")
#elif JSON_HEDLEY_PELLES_VERSION_CHECK(2,90,0)
    #define JSON_HEDLEY_DIAGNOSTIC_PUSH _Pragma("warning(push)")
    #define JSON_HEDLEY_DIAGNOSTIC_POP _Pragma("warning(pop)")
#else
    #define JSON_HEDLEY_DIAGNOSTIC_PUSH
    #define JSON_HEDLEY_DIAGNOSTIC_POP
#endif

/* JSON_HEDLEY_DIAGNOSTIC_DISABLE_CPP98_COMPAT_WRAP_ is for
   HEDLEY INTERNAL USE ONLY.  API subject to change without notice. */
#if defined(JSON_HEDLEY_DIAGNOSTIC_DISABLE_CPP98_COMPAT_WRAP_)
    #undef JSON_HEDLEY_DIAGNOSTIC_DISABLE_CPP98_COMPAT_WRAP_
#endif
#if defined(__cplusplus)
#  if JSON_HEDLEY_HAS_WARNING("-Wc++98-compat")
#    if JSON_HEDLEY_HAS_WARNING("-Wc++17-extensions")
#      if JSON_HEDLEY_HAS_WARNING("-Wc++1z-extensions")
#        define JSON_HEDLEY_DIAGNOSTIC_DISABLE_CPP98_COMPAT_WRAP_(xpr) \
    JSON_HEDLEY_DIAGNOSTIC_PUSH \
    _Pragma("clang diagnostic ignored \"-Wc++98-compat\"") \
    _Pragma("clang diagnostic ignored \"-Wc++17-extensions\"") \
    _Pragma("clang diagnostic ignored \"-Wc++1z-extensions\"") \
    xpr \
    JSON_HEDLEY_DIAGNOSTIC_POP
#      else
#        define JSON_HEDLEY_DIAGNOSTIC_DISABLE_CPP98_COMPAT_WRAP_(xpr) \
    JSON_HEDLEY_DIAGNOSTIC_PUSH \
    _Pragma("clang diagnostic ignored \"-Wc++98-compat\"") \
    _Pragma("clang diagnostic ignored \"-Wc++17-extensions\"") \
    xpr \
    JSON_HEDLEY_DIAGNOSTIC_POP
#      endif
#    else
#      define JSON_HEDLEY_DIAGNOSTIC_DISABLE_CPP98_COMPAT_WRAP_(xpr) \
    JSON_HEDLEY_DIAGNOSTIC_PUSH \
    _Pragma("clang diagnostic ignored \"-Wc++98-compat\"") \
    xpr \
    JSON_HEDLEY_DIAGNOSTIC_POP
#    endif
#  endif
#endif
#if !defined(JSON_HEDLEY_DIAGNOSTIC_DISABLE_CPP98_COMPAT_WRAP_)
    #define JSON_HEDLEY_DIAGNOSTIC_DISABLE_CPP98_COMPAT_WRAP_(x) x
#endif

#if defined(JSON_HEDLEY_CONST_CAST)
    #undef JSON_HEDLEY_CONST_CAST
#endif
#if defined(__cplusplus)
#  define JSON_HEDLEY_CONST_CAST(T, expr) (const_cast<T>(expr))
#elif \
  JSON_HEDLEY_HAS_WARNING("-Wcast-qual") || \
  JSON_HEDLEY_GCC_VERSION_CHECK(4,6,0) || \
  JSON_HEDLEY_INTEL_VERSION_CHECK(13,0,0)
#  define JSON_HEDLEY_CONST_CAST(T, expr) (__extension__ ({ \
        JSON_HEDLEY_DIAGNOSTIC_PUSH \
        JSON_HEDLEY_DIAGNOSTIC_DISABLE_CAST_QUAL \
        ((T) (expr)); \
        JSON_HEDLEY_DIAGNOSTIC_POP \
    }))
#else
#  define JSON_HEDLEY_CONST_CAST(T, expr) ((T) (expr))
#endif

#if defined(JSON_HEDLEY_REINTERPRET_CAST)
    #undef JSON_HEDLEY_REINTERPRET_CAST
#endif
#if defined(__cplusplus)
    #define JSON_HEDLEY_REINTERPRET_CAST(T, expr) (reinterpret_cast<T>(expr))
#else
    #define JSON_HEDLEY_REINTERPRET_CAST(T, expr) ((T) (expr))
#endif

#if defined(JSON_HEDLEY_STATIC_CAST)
    #undef JSON_HEDLEY_STATIC_CAST
#endif
#if defined(__cplusplus)
    #define JSON_HEDLEY_STATIC_CAST(T, expr) (static_cast<T>(expr))
#else
    #define JSON_HEDLEY_STATIC_CAST(T, expr) ((T) (expr))
#endif

#if defined(JSON_HEDLEY_CPP_CAST)
    #undef JSON_HEDLEY_CPP_CAST
#endif
#if defined(__cplusplus)
#  if JSON_HEDLEY_HAS_WARNING("-Wold-style-cast")
#    define JSON_HEDLEY_CPP_CAST(T, expr) \
    JSON_HEDLEY_DIAGNOSTIC_PUSH \
    _Pragma("clang diagnostic ignored \"-Wold-style-cast\"") \
    ((T) (expr)) \
    JSON_HEDLEY_DIAGNOSTIC_POP
#  elif JSON_HEDLEY_IAR_VERSION_CHECK(8,3,0)
#    define JSON_HEDLEY_CPP_CAST(T, expr) \
    JSON_HEDLEY_DIAGNOSTIC_PUSH \
    _Pragma("diag_suppress=Pe137") \
    JSON_HEDLEY_DIAGNOSTIC_POP
#  else
#    define JSON_HEDLEY_CPP_CAST(T, expr) ((T) (expr))
#  endif
#else
#  define JSON_HEDLEY_CPP_CAST(T, expr) (expr)
#endif

#if defined(JSON_HEDLEY_DIAGNOSTIC_DISABLE_DEPRECATED)
    #undef JSON_HEDLEY_DIAGNOSTIC_DISABLE_DEPRECATED
#endif
#if JSON_HEDLEY_HAS_WARNING("-Wdeprecated-declarations")
    #define JSON_HEDLEY_DIAGNOSTIC_DISABLE_DEPRECATED _Pragma("clang diagnostic ignored \"-Wdeprecated-declarations\"")
#elif JSON_HEDLEY_INTEL_VERSION_CHECK(13,0,0)
    #define JSON_HEDLEY_DIAGNOSTIC_DISABLE_DEPRECATED _Pragma("warning(disable:1478 1786)")
#elif JSON_HEDLEY_INTEL_CL_VERSION_CHECK(2021,1,0)
    #define JSON_HEDLEY_DIAGNOSTIC_DISABLE_DEPRECATED __pragma(warning(disable:1478 1786))
#elif JSON_HEDLEY_PGI_VERSION_CHECK(20,7,0)
    #define JSON_HEDLEY_DIAGNOSTIC_DISABLE_DEPRECATED _Pragma("diag_suppress 1215,1216,1444,1445")
#elif JSON_HEDLEY_PGI_VERSION_CHECK(17,10,0)
    #define JSON_HEDLEY_DIAGNOSTIC_DISABLE_DEPRECATED _Pragma("diag_suppress 1215,1444")
#elif JSON_HEDLEY_GCC_VERSION_CHECK(4,3,0)
    #define JSON_HEDLEY_DIAGNOSTIC_DISABLE_DEPRECATED _Pragma("GCC diagnostic ignored \"-Wdeprecated-declarations\"")
#elif JSON_HEDLEY_MSVC_VERSION_CHECK(15,0,0)
    #define JSON_HEDLEY_DIAGNOSTIC_DISABLE_DEPRECATED __pragma(warning(disable:4996))
#elif JSON_HEDLEY_MCST_LCC_VERSION_CHECK(1,25,10)
    #define JSON_HEDLEY_DIAGNOSTIC_DISABLE_DEPRECATED _Pragma("diag_suppress 1215,1444")
#elif \
    JSON_HEDLEY_TI_VERSION_CHECK(15,12,0) || \
    (JSON_HEDLEY_TI_ARMCL_VERSION_CHECK(4,8,0) && defined(__TI_GNU_ATTRIBUTE_SUPPORT__)) || \
    JSON_HEDLEY_TI_ARMCL_VERSION_CHECK(5,2,0) || \
    (JSON_HEDLEY_TI_CL2000_VERSION_CHECK(6,0,0) && defined(__TI_GNU_ATTRIBUTE_SUPPORT__)) || \
    JSON_HEDLEY_TI_CL2000_VERSION_CHECK(6,4,0) || \
    (JSON_HEDLEY_TI_CL430_VERSION_CHECK(4,0,0) && defined(__TI_GNU_ATTRIBUTE_SUPPORT__)) || \
    JSON_HEDLEY_TI_CL430_VERSION_CHECK(4,3,0) || \
    (JSON_HEDLEY_TI_CL6X_VERSION_CHECK(7,2,0) && defined(__TI_GNU_ATTRIBUTE_SUPPORT__)) || \
    JSON_HEDLEY_TI_CL6X_VERSION_CHECK(7,5,0) || \
    JSON_HEDLEY_TI_CL7X_VERSION_CHECK(1,2,0) || \
    JSON_HEDLEY_TI_CLPRU_VERSION_CHECK(2,1,0)
    #define JSON_HEDLEY_DIAGNOSTIC_DISABLE_DEPRECATED _Pragma("diag_suppress 1291,1718")
#elif JSON_HEDLEY_SUNPRO_VERSION_CHECK(5,13,0) && !defined(__cplusplus)
    #define JSON_HEDLEY_DIAGNOSTIC_DISABLE_DEPRECATED _Pragma("error_messages(off,E_DEPRECATED_ATT,E_DEPRECATED_ATT_MESS)")
#elif JSON_HEDLEY_SUNPRO_VERSION_CHECK(5,13,0) && defined(__cplusplus)
    #define JSON_HEDLEY_DIAGNOSTIC_DISABLE_DEPRECATED _Pragma("error_messages(off,symdeprecated,symdeprecated2)")
#elif JSON_HEDLEY_IAR_VERSION_CHECK(8,0,0)
    #define JSON_HEDLEY_DIAGNOSTIC_DISABLE_DEPRECATED _Pragma("diag_suppress=Pe1444,Pe1215")
#elif JSON_HEDLEY_PELLES_VERSION_CHECK(2,90,0)
    #define JSON_HEDLEY_DIAGNOSTIC_DISABLE_DEPRECATED _Pragma("warn(disable:2241)")
#else
    #define JSON_HEDLEY_DIAGNOSTIC_DISABLE_DEPRECATED
#endif

#if defined(JSON_HEDLEY_DIAGNOSTIC_DISABLE_UNKNOWN_PRAGMAS)
    #undef JSON_HEDLEY_DIAGNOSTIC_DISABLE_UNKNOWN_PRAGMAS
#endif
#if JSON_HEDLEY_HAS_WARNING("-Wunknown-pragmas")
    #define JSON_HEDLEY_DIAGNOSTIC_DISABLE_UNKNOWN_PRAGMAS _Pragma("clang diagnostic ignored \"-Wunknown-pragmas\"")
#elif JSON_HEDLEY_INTEL_VERSION_CHECK(13,0,0)
    #define JSON_HEDLEY_DIAGNOSTIC_DISABLE_UNKNOWN_PRAGMAS _Pragma("warning(disable:161)")
#elif JSON_HEDLEY_INTEL_CL_VERSION_CHECK(2021,1,0)
    #define JSON_HEDLEY_DIAGNOSTIC_DISABLE_UNKNOWN_PRAGMAS __pragma(warning(disable:161))
#elif JSON_HEDLEY_PGI_VERSION_CHECK(17,10,0)
    #define JSON_HEDLEY_DIAGNOSTIC_DISABLE_UNKNOWN_PRAGMAS _Pragma("diag_suppress 1675")
#elif JSON_HEDLEY_GCC_VERSION_CHECK(4,3,0)
    #define JSON_HEDLEY_DIAGNOSTIC_DISABLE_UNKNOWN_PRAGMAS _Pragma("GCC diagnostic ignored \"-Wunknown-pragmas\"")
#elif JSON_HEDLEY_MSVC_VERSION_CHECK(15,0,0)
    #define JSON_HEDLEY_DIAGNOSTIC_DISABLE_UNKNOWN_PRAGMAS __pragma(warning(disable:4068))
#elif \
    JSON_HEDLEY_TI_VERSION_CHECK(16,9,0) || \
    JSON_HEDLEY_TI_CL6X_VERSION_CHECK(8,0,0) || \
    JSON_HEDLEY_TI_CL7X_VERSION_CHECK(1,2,0) || \
    JSON_HEDLEY_TI_CLPRU_VERSION_CHECK(2,3,0)
    #define JSON_HEDLEY_DIAGNOSTIC_DISABLE_UNKNOWN_PRAGMAS _Pragma("diag_suppress 163")
#elif JSON_HEDLEY_TI_CL6X_VERSION_CHECK(8,0,0)
    #define JSON_HEDLEY_DIAGNOSTIC_DISABLE_UNKNOWN_PRAGMAS _Pragma("diag_suppress 163")
#elif JSON_HEDLEY_IAR_VERSION_CHECK(8,0,0)
    #define JSON_HEDLEY_DIAGNOSTIC_DISABLE_UNKNOWN_PRAGMAS _Pragma("diag_suppress=Pe161")
#elif JSON_HEDLEY_MCST_LCC_VERSION_CHECK(1,25,10)
    #define JSON_HEDLEY_DIAGNOSTIC_DISABLE_UNKNOWN_PRAGMAS _Pragma("diag_suppress 161")
#else
    #define JSON_HEDLEY_DIAGNOSTIC_DISABLE_UNKNOWN_PRAGMAS
#endif

#if defined(JSON_HEDLEY_DIAGNOSTIC_DISABLE_UNKNOWN_CPP_ATTRIBUTES)
    #undef JSON_HEDLEY_DIAGNOSTIC_DISABLE_UNKNOWN_CPP_ATTRIBUTES
#endif
#if JSON_HEDLEY_HAS_WARNING("-Wunknown-attributes")
    #define JSON_HEDLEY_DIAGNOSTIC_DISABLE_UNKNOWN_CPP_ATTRIBUTES _Pragma("clang diagnostic ignored \"-Wunknown-attributes\"")
#elif JSON_HEDLEY_GCC_VERSION_CHECK(4,6,0)
    #define JSON_HEDLEY_DIAGNOSTIC_DISABLE_UNKNOWN_CPP_ATTRIBUTES _Pragma("GCC diagnostic ignored \"-Wdeprecated-declarations\"")
#elif JSON_HEDLEY_INTEL_VERSION_CHECK(17,0,0)
    #define JSON_HEDLEY_DIAGNOSTIC_DISABLE_UNKNOWN_CPP_ATTRIBUTES _Pragma("warning(disable:1292)")
#elif JSON_HEDLEY_INTEL_CL_VERSION_CHECK(2021,1,0)
    #define JSON_HEDLEY_DIAGNOSTIC_DISABLE_UNKNOWN_CPP_ATTRIBUTES __pragma(warning(disable:1292))
#elif JSON_HEDLEY_MSVC_VERSION_CHECK(19,0,0)
    #define JSON_HEDLEY_DIAGNOSTIC_DISABLE_UNKNOWN_CPP_ATTRIBUTES __pragma(warning(disable:5030))
#elif JSON_HEDLEY_PGI_VERSION_CHECK(20,7,0)
    #define JSON_HEDLEY_DIAGNOSTIC_DISABLE_UNKNOWN_CPP_ATTRIBUTES _Pragma("diag_suppress 1097,1098")
#elif JSON_HEDLEY_PGI_VERSION_CHECK(17,10,0)
    #define JSON_HEDLEY_DIAGNOSTIC_DISABLE_UNKNOWN_CPP_ATTRIBUTES _Pragma("diag_suppress 1097")
#elif JSON_HEDLEY_SUNPRO_VERSION_CHECK(5,14,0) && defined(__cplusplus)
    #define JSON_HEDLEY_DIAGNOSTIC_DISABLE_UNKNOWN_CPP_ATTRIBUTES _Pragma("error_messages(off,attrskipunsup)")
#elif \
    JSON_HEDLEY_TI_VERSION_CHECK(18,1,0) || \
    JSON_HEDLEY_TI_CL6X_VERSION_CHECK(8,3,0) || \
    JSON_HEDLEY_TI_CL7X_VERSION_CHECK(1,2,0)
    #define JSON_HEDLEY_DIAGNOSTIC_DISABLE_UNKNOWN_CPP_ATTRIBUTES _Pragma("diag_suppress 1173")
#elif JSON_HEDLEY_IAR_VERSION_CHECK(8,0,0)
    #define JSON_HEDLEY_DIAGNOSTIC_DISABLE_UNKNOWN_CPP_ATTRIBUTES _Pragma("diag_suppress=Pe1097")
#elif JSON_HEDLEY_MCST_LCC_VERSION_CHECK(1,25,10)
    #define JSON_HEDLEY_DIAGNOSTIC_DISABLE_UNKNOWN_CPP_ATTRIBUTES _Pragma("diag_suppress 1097")
#else
    #define JSON_HEDLEY_DIAGNOSTIC_DISABLE_UNKNOWN_CPP_ATTRIBUTES
#endif

#if defined(JSON_HEDLEY_DIAGNOSTIC_DISABLE_CAST_QUAL)
    #undef JSON_HEDLEY_DIAGNOSTIC_DISABLE_CAST_QUAL
#endif
#if JSON_HEDLEY_HAS_WARNING("-Wcast-qual")
    #define JSON_HEDLEY_DIAGNOSTIC_DISABLE_CAST_QUAL _Pragma("clang diagnostic ignored \"-Wcast-qual\"")
#elif JSON_HEDLEY_INTEL_VERSION_CHECK(13,0,0)
    #define JSON_HEDLEY_DIAGNOSTIC_DISABLE_CAST_QUAL _Pragma("warning(disable:2203 2331)")
#elif JSON_HEDLEY_GCC_VERSION_CHECK(3,0,0)
    #define JSON_HEDLEY_DIAGNOSTIC_DISABLE_CAST_QUAL _Pragma("GCC diagnostic ignored \"-Wcast-qual\"")
#else
    #define JSON_HEDLEY_DIAGNOSTIC_DISABLE_CAST_QUAL
#endif

#if defined(JSON_HEDLEY_DIAGNOSTIC_DISABLE_UNUSED_FUNCTION)
    #undef JSON_HEDLEY_DIAGNOSTIC_DISABLE_UNUSED_FUNCTION
#endif
#if JSON_HEDLEY_HAS_WARNING("-Wunused-function")
    #define JSON_HEDLEY_DIAGNOSTIC_DISABLE_UNUSED_FUNCTION _Pragma("clang diagnostic ignored \"-Wunused-function\"")
#elif JSON_HEDLEY_GCC_VERSION_CHECK(3,4,0)
    #define JSON_HEDLEY_DIAGNOSTIC_DISABLE_UNUSED_FUNCTION _Pragma("GCC diagnostic ignored \"-Wunused-function\"")
#elif JSON_HEDLEY_MSVC_VERSION_CHECK(1,0,0)
    #define JSON_HEDLEY_DIAGNOSTIC_DISABLE_UNUSED_FUNCTION __pragma(warning(disable:4505))
#elif JSON_HEDLEY_MCST_LCC_VERSION_CHECK(1,25,10)
    #define JSON_HEDLEY_DIAGNOSTIC_DISABLE_UNUSED_FUNCTION _Pragma("diag_suppress 3142")
#else
    #define JSON_HEDLEY_DIAGNOSTIC_DISABLE_UNUSED_FUNCTION
#endif

#if defined(JSON_HEDLEY_DEPRECATED)
    #undef JSON_HEDLEY_DEPRECATED
#endif
#if defined(JSON_HEDLEY_DEPRECATED_FOR)
    #undef JSON_HEDLEY_DEPRECATED_FOR
#endif
#if \
    JSON_HEDLEY_MSVC_VERSION_CHECK(14,0,0) || \
    JSON_HEDLEY_INTEL_CL_VERSION_CHECK(2021,1,0)
    #define JSON_HEDLEY_DEPRECATED(since) __declspec(deprecated("Since " # since))
    #define JSON_HEDLEY_DEPRECATED_FOR(since, replacement) __declspec(deprecated("Since " #since "; use " #replacement))
#elif \
    (JSON_HEDLEY_HAS_EXTENSION(attribute_deprecated_with_message) && !defined(JSON_HEDLEY_IAR_VERSION)) || \
    JSON_HEDLEY_GCC_VERSION_CHECK(4,5,0) || \
    JSON_HEDLEY_INTEL_VERSION_CHECK(13,0,0) || \
    JSON_HEDLEY_ARM_VERSION_CHECK(5,6,0) || \
    JSON_HEDLEY_SUNPRO_VERSION_CHECK(5,13,0) || \
    JSON_HEDLEY_PGI_VERSION_CHECK(17,10,0) || \
    JSON_HEDLEY_TI_VERSION_CHECK(18,1,0) || \
    JSON_HEDLEY_TI_ARMCL_VERSION_CHECK(18,1,0) || \
    JSON_HEDLEY_TI_CL6X_VERSION_CHECK(8,3,0) || \
    JSON_HEDLEY_TI_CL7X_VERSION_CHECK(1,2,0) || \
    JSON_HEDLEY_TI_CLPRU_VERSION_CHECK(2,3,0) || \
    JSON_HEDLEY_MCST_LCC_VERSION_CHECK(1,25,10)
    #define JSON_HEDLEY_DEPRECATED(since) __attribute__((__deprecated__("Since " #since)))
    #define JSON_HEDLEY_DEPRECATED_FOR(since, replacement) __attribute__((__deprecated__("Since " #since "; use " #replacement)))
#elif defined(__cplusplus) && (__cplusplus >= 201402L)
    #define JSON_HEDLEY_DEPRECATED(since) JSON_HEDLEY_DIAGNOSTIC_DISABLE_CPP98_COMPAT_WRAP_([[deprecated("Since " #since)]])
    #define JSON_HEDLEY_DEPRECATED_FOR(since, replacement) JSON_HEDLEY_DIAGNOSTIC_DISABLE_CPP98_COMPAT_WRAP_([[deprecated("Since " #since "; use " #replacement)]])
#elif \
    JSON_HEDLEY_HAS_ATTRIBUTE(deprecated) || \
    JSON_HEDLEY_GCC_VERSION_CHECK(3,1,0) || \
    JSON_HEDLEY_ARM_VERSION_CHECK(4,1,0) || \
    JSON_HEDLEY_TI_VERSION_CHECK(15,12,0) || \
    (JSON_HEDLEY_TI_ARMCL_VERSION_CHECK(4,8,0) && defined(__TI_GNU_ATTRIBUTE_SUPPORT__)) || \
    JSON_HEDLEY_TI_ARMCL_VERSION_CHECK(5,2,0) || \
    (JSON_HEDLEY_TI_CL2000_VERSION_CHECK(6,0,0) && defined(__TI_GNU_ATTRIBUTE_SUPPORT__)) || \
    JSON_HEDLEY_TI_CL2000_VERSION_CHECK(6,4,0) || \
    (JSON_HEDLEY_TI_CL430_VERSION_CHECK(4,0,0) && defined(__TI_GNU_ATTRIBUTE_SUPPORT__)) || \
    JSON_HEDLEY_TI_CL430_VERSION_CHECK(4,3,0) || \
    (JSON_HEDLEY_TI_CL6X_VERSION_CHECK(7,2,0) && defined(__TI_GNU_ATTRIBUTE_SUPPORT__)) || \
    JSON_HEDLEY_TI_CL6X_VERSION_CHECK(7,5,0) || \
    JSON_HEDLEY_TI_CL7X_VERSION_CHECK(1,2,0) || \
    JSON_HEDLEY_TI_CLPRU_VERSION_CHECK(2,1,0) || \
    JSON_HEDLEY_MCST_LCC_VERSION_CHECK(1,25,10) || \
    JSON_HEDLEY_IAR_VERSION_CHECK(8,10,0)
    #define JSON_HEDLEY_DEPRECATED(since) __attribute__((__deprecated__))
    #define JSON_HEDLEY_DEPRECATED_FOR(since, replacement) __attribute__((__deprecated__))
#elif \
    JSON_HEDLEY_MSVC_VERSION_CHECK(13,10,0) || \
    JSON_HEDLEY_PELLES_VERSION_CHECK(6,50,0) || \
    JSON_HEDLEY_INTEL_CL_VERSION_CHECK(2021,1,0)
    #define JSON_HEDLEY_DEPRECATED(since) __declspec(deprecated)
    #define JSON_HEDLEY_DEPRECATED_FOR(since, replacement) __declspec(deprecated)
#elif JSON_HEDLEY_IAR_VERSION_CHECK(8,0,0)
    #define JSON_HEDLEY_DEPRECATED(since) _Pragma("deprecated")
    #define JSON_HEDLEY_DEPRECATED_FOR(since, replacement) _Pragma("deprecated")
#else
    #define JSON_HEDLEY_DEPRECATED(since)
    #define JSON_HEDLEY_DEPRECATED_FOR(since, replacement)
#endif

#if defined(JSON_HEDLEY_UNAVAILABLE)
    #undef JSON_HEDLEY_UNAVAILABLE
#endif
#if \
    JSON_HEDLEY_HAS_ATTRIBUTE(warning) || \
    JSON_HEDLEY_GCC_VERSION_CHECK(4,3,0) || \
    JSON_HEDLEY_INTEL_VERSION_CHECK(13,0,0) || \
    JSON_HEDLEY_MCST_LCC_VERSION_CHECK(1,25,10)
    #define JSON_HEDLEY_UNAVAILABLE(available_since) __attribute__((__warning__("Not available until " #available_since)))
#else
    #define JSON_HEDLEY_UNAVAILABLE(available_since)
#endif

#if defined(JSON_HEDLEY_WARN_UNUSED_RESULT)
    #undef JSON_HEDLEY_WARN_UNUSED_RESULT
#endif
#if defined(JSON_HEDLEY_WARN_UNUSED_RESULT_MSG)
    #undef JSON_HEDLEY_WARN_UNUSED_RESULT_MSG
#endif
#if \
    JSON_HEDLEY_HAS_ATTRIBUTE(warn_unused_result) || \
    JSON_HEDLEY_GCC_VERSION_CHECK(3,4,0) || \
    JSON_HEDLEY_INTEL_VERSION_CHECK(13,0,0) || \
    JSON_HEDLEY_TI_VERSION_CHECK(15,12,0) || \
    (JSON_HEDLEY_TI_ARMCL_VERSION_CHECK(4,8,0) && defined(__TI_GNU_ATTRIBUTE_SUPPORT__)) || \
    JSON_HEDLEY_TI_ARMCL_VERSION_CHECK(5,2,0) || \
    (JSON_HEDLEY_TI_CL2000_VERSION_CHECK(6,0,0) && defined(__TI_GNU_ATTRIBUTE_SUPPORT__)) || \
    JSON_HEDLEY_TI_CL2000_VERSION_CHECK(6,4,0) || \
    (JSON_HEDLEY_TI_CL430_VERSION_CHECK(4,0,0) && defined(__TI_GNU_ATTRIBUTE_SUPPORT__)) || \
    JSON_HEDLEY_TI_CL430_VERSION_CHECK(4,3,0) || \
    (JSON_HEDLEY_TI_CL6X_VERSION_CHECK(7,2,0) && defined(__TI_GNU_ATTRIBUTE_SUPPORT__)) || \
    JSON_HEDLEY_TI_CL6X_VERSION_CHECK(7,5,0) || \
    JSON_HEDLEY_TI_CL7X_VERSION_CHECK(1,2,0) || \
    JSON_HEDLEY_TI_CLPRU_VERSION_CHECK(2,1,0) || \
    (JSON_HEDLEY_SUNPRO_VERSION_CHECK(5,15,0) && defined(__cplusplus)) || \
    JSON_HEDLEY_PGI_VERSION_CHECK(17,10,0) || \
    JSON_HEDLEY_MCST_LCC_VERSION_CHECK(1,25,10)
    #define JSON_HEDLEY_WARN_UNUSED_RESULT __attribute__((__warn_unused_result__))
    #define JSON_HEDLEY_WARN_UNUSED_RESULT_MSG(msg) __attribute__((__warn_unused_result__))
#elif (JSON_HEDLEY_HAS_CPP_ATTRIBUTE(nodiscard) >= 201907L)
    #define JSON_HEDLEY_WARN_UNUSED_RESULT JSON_HEDLEY_DIAGNOSTIC_DISABLE_CPP98_COMPAT_WRAP_([[nodiscard]])
    #define JSON_HEDLEY_WARN_UNUSED_RESULT_MSG(msg) JSON_HEDLEY_DIAGNOSTIC_DISABLE_CPP98_COMPAT_WRAP_([[nodiscard(msg)]])
#elif JSON_HEDLEY_HAS_CPP_ATTRIBUTE(nodiscard)
    #define JSON_HEDLEY_WARN_UNUSED_RESULT JSON_HEDLEY_DIAGNOSTIC_DISABLE_CPP98_COMPAT_WRAP_([[nodiscard]])
    #define JSON_HEDLEY_WARN_UNUSED_RESULT_MSG(msg) JSON_HEDLEY_DIAGNOSTIC_DISABLE_CPP98_COMPAT_WRAP_([[nodiscard]])
#elif defined(_Check_return_) /* SAL */
    #define JSON_HEDLEY_WARN_UNUSED_RESULT _Check_return_
    #define JSON_HEDLEY_WARN_UNUSED_RESULT_MSG(msg) _Check_return_
#else
    #define JSON_HEDLEY_WARN_UNUSED_RESULT
    #define JSON_HEDLEY_WARN_UNUSED_RESULT_MSG(msg)
#endif

#if defined(JSON_HEDLEY_SENTINEL)
    #undef JSON_HEDLEY_SENTINEL
#endif
#if \
    JSON_HEDLEY_HAS_ATTRIBUTE(sentinel) || \
    JSON_HEDLEY_GCC_VERSION_CHECK(4,0,0) || \
    JSON_HEDLEY_INTEL_VERSION_CHECK(13,0,0) || \
    JSON_HEDLEY_ARM_VERSION_CHECK(5,4,0) || \
    JSON_HEDLEY_MCST_LCC_VERSION_CHECK(1,25,10)
    #define JSON_HEDLEY_SENTINEL(position) __attribute__((__sentinel__(position)))
#else
    #define JSON_HEDLEY_SENTINEL(position)
#endif

#if defined(JSON_HEDLEY_NO_RETURN)
    #undef JSON_HEDLEY_NO_RETURN
#endif
#if JSON_HEDLEY_IAR_VERSION_CHECK(8,0,0)
    #define JSON_HEDLEY_NO_RETURN __noreturn
#elif \
    JSON_HEDLEY_INTEL_VERSION_CHECK(13,0,0) || \
    JSON_HEDLEY_MCST_LCC_VERSION_CHECK(1,25,10)
    #define JSON_HEDLEY_NO_RETURN __attribute__((__noreturn__))
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
    #define JSON_HEDLEY_NO_RETURN _Noreturn
#elif defined(__cplusplus) && (__cplusplus >= 201103L)
    #define JSON_HEDLEY_NO_RETURN JSON_HEDLEY_DIAGNOSTIC_DISABLE_CPP98_COMPAT_WRAP_([[noreturn]])
#elif \
    JSON_HEDLEY_HAS_ATTRIBUTE(noreturn) || \
    JSON_HEDLEY_GCC_VERSION_CHECK(3,2,0) || \
    JSON_HEDLEY_SUNPRO_VERSION_CHECK(5,11,0) || \
    JSON_HEDLEY_ARM_VERSION_CHECK(4,1,0) || \
    JSON_HEDLEY_IBM_VERSION_CHECK(10,1,0) || \
    JSON_HEDLEY_TI_VERSION_CHECK(15,12,0) || \
    (JSON_HEDLEY_TI_ARMCL_VERSION_CHECK(4,8,0) && defined(__TI_GNU_ATTRIBUTE_SUPPORT__)) || \
    JSON_HEDLEY_TI_ARMCL_VERSION_CHECK(5,2,0) || \
    (JSON_HEDLEY_TI_CL2000_VERSION_CHECK(6,0,0) && defined(__TI_GNU_ATTRIBUTE_SUPPORT__)) || \
    JSON_HEDLEY_TI_CL2000_VERSION_CHECK(6,4,0) || \
    (JSON_HEDLEY_TI_CL430_VERSION_CHECK(4,0,0) && defined(__TI_GNU_ATTRIBUTE_SUPPORT__)) || \
    JSON_HEDLEY_TI_CL430_VERSION_CHECK(4,3,0) || \
    (JSON_HEDLEY_TI_CL6X_VERSION_CHECK(7,2,0) && defined(__TI_GNU_ATTRIBUTE_SUPPORT__)) || \
    JSON_HEDLEY_TI_CL6X_VERSION_CHECK(7,5,0) || \
    JSON_HEDLEY_TI_CL7X_VERSION_CHECK(1,2,0) || \
    JSON_HEDLEY_TI_CLPRU_VERSION_CHECK(2,1,0) || \
    JSON_HEDLEY_IAR_VERSION_CHECK(8,10,0)
    #define JSON_HEDLEY_NO_RETURN __attribute__((__noreturn__))
#elif JSON_HEDLEY_SUNPRO_VERSION_CHECK(5,10,0)
    #define JSON_HEDLEY_NO_RETURN _Pragma("does_not_return")
#elif \
    JSON_HEDLEY_MSVC_VERSION_CHECK(13,10,0) || \
    JSON_HEDLEY_INTEL_CL_VERSION_CHECK(2021,1,0)
    #define JSON_HEDLEY_NO_RETURN __declspec(noreturn)
#elif JSON_HEDLEY_TI_CL6X_VERSION_CHECK(6,0,0) && defined(__cplusplus)
    #define JSON_HEDLEY_NO_RETURN _Pragma("FUNC_NEVER_RETURNS;")
#elif JSON_HEDLEY_COMPCERT_VERSION_CHECK(3,2,0)
    #define JSON_HEDLEY_NO_RETURN __attribute((noreturn))
#elif JSON_HEDLEY_PELLES_VERSION_CHECK(9,0,0)
    #define JSON_HEDLEY_NO_RETURN __declspec(noreturn)
#else
    #define JSON_HEDLEY_NO_RETURN
#endif

#if defined(JSON_HEDLEY_NO_ESCAPE)
    #undef JSON_HEDLEY_NO_ESCAPE
#endif
#if JSON_HEDLEY_HAS_ATTRIBUTE(noescape)
    #define JSON_HEDLEY_NO_ESCAPE __attribute__((__noescape__))
#else
    #define JSON_HEDLEY_NO_ESCAPE
#endif

#if defined(JSON_HEDLEY_UNREACHABLE)
    #undef JSON_HEDLEY_UNREACHABLE
#endif
#if defined(JSON_HEDLEY_UNREACHABLE_RETURN)
    #undef JSON_HEDLEY_UNREACHABLE_RETURN
#endif
#if defined(JSON_HEDLEY_ASSUME)
    #undef JSON_HEDLEY_ASSUME
#endif
#if \
    JSON_HEDLEY_MSVC_VERSION_CHECK(13,10,0) || \
    JSON_HEDLEY_INTEL_VERSION_CHECK(13,0,0) || \
    JSON_HEDLEY_INTEL_CL_VERSION_CHECK(2021,1,0)
    #define JSON_HEDLEY_ASSUME(expr) __assume(expr)
#elif JSON_HEDLEY_HAS_BUILTIN(__builtin_assume)
    #define JSON_HEDLEY_ASSUME(expr) __builtin_assume(expr)
#elif \
    JSON_HEDLEY_TI_CL2000_VERSION_CHECK(6,2,0) || \
    JSON_HEDLEY_TI_CL6X_VERSION_CHECK(4,0,0)
    #if defined(__cplusplus)
        #define JSON_HEDLEY_ASSUME(expr) std::_nassert(expr)
    #else
        #define JSON_HEDLEY_ASSUME(expr) _nassert(expr)
    #endif
#endif
#if \
    (JSON_HEDLEY_HAS_BUILTIN(__builtin_unreachable) && (!defined(JSON_HEDLEY_ARM_VERSION))) || \
    JSON_HEDLEY_GCC_VERSION_CHECK(4,5,0) || \
    JSON_HEDLEY_PGI_VERSION_CHECK(18,10,0) || \
    JSON_HEDLEY_INTEL_VERSION_CHECK(13,0,0) || \
    JSON_HEDLEY_IBM_VERSION_CHECK(13,1,5) || \
    JSON_HEDLEY_CRAY_VERSION_CHECK(10,0,0) || \
    JSON_HEDLEY_MCST_LCC_VERSION_CHECK(1,25,10)
    #define JSON_HEDLEY_UNREACHABLE() __builtin_unreachable()
#elif defined(JSON_HEDLEY_ASSUME)
    #define JSON_HEDLEY_UNREACHABLE() JSON_HEDLEY_ASSUME(0)
#endif
#if !defined(JSON_HEDLEY_ASSUME)
    #if defined(JSON_HEDLEY_UNREACHABLE)
        #define JSON_HEDLEY_ASSUME(expr) JSON_HEDLEY_STATIC_CAST(void, ((expr) ? 1 : (JSON_HEDLEY_UNREACHABLE(), 1)))
    #else
        #define JSON_HEDLEY_ASSUME(expr) JSON_HEDLEY_STATIC_CAST(void, expr)
    #endif
#endif
#if defined(JSON_HEDLEY_UNREACHABLE)
    #if  \
        JSON_HEDLEY_TI_CL2000_VERSION_CHECK(6,2,0) || \
        JSON_HEDLEY_TI_CL6X_VERSION_CHECK(4,0,0)
        #define JSON_HEDLEY_UNREACHABLE_RETURN(value) return (JSON_HEDLEY_STATIC_CAST(void, JSON_HEDLEY_ASSUME(0)), (value))
    #else
        #define JSON_HEDLEY_UNREACHABLE_RETURN(value) JSON_HEDLEY_UNREACHABLE()
    #endif
#else
    #define JSON_HEDLEY_UNREACHABLE_RETURN(value) return (value)
#endif
#if !defined(JSON_HEDLEY_UNREACHABLE)
    #define JSON_HEDLEY_UNREACHABLE() JSON_HEDLEY_ASSUME(0)
#endif

JSON_HEDLEY_DIAGNOSTIC_PUSH
#if JSON_HEDLEY_HAS_WARNING("-Wpedantic")
    #pragma clang diagnostic ignored "-Wpedantic"
#endif
#if JSON_HEDLEY_HAS_WARNING("-Wc++98-compat-pedantic") && defined(__cplusplus)
    #pragma clang diagnostic ignored "-Wc++98-compat-pedantic"
#endif
#if JSON_HEDLEY_GCC_HAS_WARNING("-Wvariadic-macros",4,0,0)
    #if defined(__clang__)
        #pragma clang diagnostic ignored "-Wvariadic-macros"
    #elif defined(JSON_HEDLEY_GCC_VERSION)
        #pragma GCC diagnostic ignored "-Wvariadic-macros"
    #endif
#endif
#if defined(JSON_HEDLEY_NON_NULL)
    #undef JSON_HEDLEY_NON_NULL
#endif
#if \
    JSON_HEDLEY_HAS_ATTRIBUTE(nonnull) || \
    JSON_HEDLEY_GCC_VERSION_CHECK(3,3,0) || \
    JSON_HEDLEY_INTEL_VERSION_CHECK(13,0,0) || \
    JSON_HEDLEY_ARM_VERSION_CHECK(4,1,0)
    #define JSON_HEDLEY_NON_NULL(...) __attribute__((__nonnull__(__VA_ARGS__)))
#else
    #define JSON_HEDLEY_NON_NULL(...)
#endif
JSON_HEDLEY_DIAGNOSTIC_POP

#if defined(JSON_HEDLEY_PRINTF_FORMAT)
    #undef JSON_HEDLEY_PRINTF_FORMAT
#endif
#if defined(__MINGW32__) && JSON_HEDLEY_GCC_HAS_ATTRIBUTE(format,4,4,0) && !defined(__USE_MINGW_ANSI_STDIO)
    #define JSON_HEDLEY_PRINTF_FORMAT(string_idx,first_to_check) __attribute__((__format__(ms_printf, string_idx, first_to_check)))
#elif defined(__MINGW32__) && JSON_HEDLEY_GCC_HAS_ATTRIBUTE(format,4,4,0) && defined(__USE_MINGW_ANSI_STDIO)
    #define JSON_HEDLEY_PRINTF_FORMAT(string_idx,first_to_check) __attribute__((__format__(gnu_printf, string_idx, first_to_check)))
#elif \
    JSON_HEDLEY_HAS_ATTRIBUTE(format) || \
    JSON_HEDLEY_GCC_VERSION_CHECK(3,1,0) || \
    JSON_HEDLEY_INTEL_VERSION_CHECK(13,0,0) || \
    JSON_HEDLEY_ARM_VERSION_CHECK(5,6,0) || \
    JSON_HEDLEY_IBM_VERSION_CHECK(10,1,0) || \
    JSON_HEDLEY_TI_VERSION_CHECK(15,12,0) || \
    (JSON_HEDLEY_TI_ARMCL_VERSION_CHECK(4,8,0) && defined(__TI_GNU_ATTRIBUTE_SUPPORT__)) || \
    JSON_HEDLEY_TI_ARMCL_VERSION_CHECK(5,2,0) || \
    (JSON_HEDLEY_TI_CL2000_VERSION_CHECK(6,0,0) && defined(__TI_GNU_ATTRIBUTE_SUPPORT__)) || \
    JSON_HEDLEY_TI_CL2000_VERSION_CHECK(6,4,0) || \
    (JSON_HEDLEY_TI_CL430_VERSION_CHECK(4,0,0) && defined(__TI_GNU_ATTRIBUTE_SUPPORT__)) || \
    JSON_HEDLEY_TI_CL430_VERSION_CHECK(4,3,0) || \
    (JSON_HEDLEY_TI_CL6X_VERSION_CHECK(7,2,0) && defined(__TI_GNU_ATTRIBUTE_SUPPORT__)) || \
    JSON_HEDLEY_TI_CL6X_VERSION_CHECK(7,5,0) || \
    JSON_HEDLEY_TI_CL7X_VERSION_CHECK(1,2,0) || \
    JSON_HEDLEY_TI_CLPRU_VERSION_CHECK(2,1,0) || \
    JSON_HEDLEY_MCST_LCC_VERSION_CHECK(1,25,10)
    #define JSON_HEDLEY_PRINTF_FORMAT(string_idx,first_to_check) __attribute__((__format__(__printf__, string_idx, first_to_check)))
#elif JSON_HEDLEY_PELLES_VERSION_CHECK(6,0,0)
    #define JSON_HEDLEY_PRINTF_FORMAT(string_idx,first_to_check) __declspec(vaformat(printf,string_idx,first_to_check))
#else
    #define JSON_HEDLEY_PRINTF_FORMAT(string_idx,first_to_check)
#endif

#if defined(JSON_HEDLEY_CONSTEXPR)
    #undef JSON_HEDLEY_CONSTEXPR
#endif
#if defined(__cplusplus)
    #if __cplusplus >= 201103L
        #define JSON_HEDLEY_CONSTEXPR JSON_HEDLEY_DIAGNOSTIC_DISABLE_CPP98_COMPAT_WRAP_(constexpr)
    #endif
#endif
#if !defined(JSON_HEDLEY_CONSTEXPR)
    #define JSON_HEDLEY_CONSTEXPR
#endif

#if defined(JSON_HEDLEY_PREDICT)
    #undef JSON_HEDLEY_PREDICT
#endif
#if defined(JSON_HEDLEY_LIKELY)
    #undef JSON_HEDLEY_LIKELY
#endif
#if defined(JSON_HEDLEY_UNLIKELY)
    #undef JSON_HEDLEY_UNLIKELY
#endif
#if defined(JSON_HEDLEY_UNPREDICTABLE)
    #undef JSON_HEDLEY_UNPREDICTABLE
#endif
#if JSON_HEDLEY_HAS_BUILTIN(__builtin_unpredictable)
    #define JSON_HEDLEY_UNPREDICTABLE(expr) __builtin_unpredictable((expr))
#endif
#if \
  (JSON_HEDLEY_HAS_BUILTIN(__builtin_expect_with_probability) && !defined(JSON_HEDLEY_PGI_VERSION)) || \
  JSON_HEDLEY_GCC_VERSION_CHECK(9,0,0) || \
  JSON_HEDLEY_MCST_LCC_VERSION_CHECK(1,25,10)
#  define JSON_HEDLEY_PREDICT(expr, value, probability) __builtin_expect_with_probability(  (expr), (value), (probability))
#  define JSON_HEDLEY_PREDICT_TRUE(expr, probability)   __builtin_expect_with_probability(!!(expr),    1   , (probability))
#  define JSON_HEDLEY_PREDICT_FALSE(expr, probability)  __builtin_expect_with_probability(!!(expr),    0   , (probability))
#  define JSON_HEDLEY_LIKELY(expr)                      __builtin_expect                 (!!(expr),    1                  )
#  define JSON_HEDLEY_UNLIKELY(expr)                    __builtin_expect                 (!!(expr),    0                  )
#elif \
  (JSON_HEDLEY_HAS_BUILTIN(__builtin_expect) && !defined(JSON_HEDLEY_INTEL_CL_VERSION)) || \
  JSON_HEDLEY_GCC_VERSION_CHECK(3,0,0) || \
  JSON_HEDLEY_INTEL_VERSION_CHECK(13,0,0) || \
  (JSON_HEDLEY_SUNPRO_VERSION_CHECK(5,15,0) && defined(__cplusplus)) || \
  JSON_HEDLEY_ARM_VERSION_CHECK(4,1,0) || \
  JSON_HEDLEY_IBM_VERSION_CHECK(10,1,0) || \
  JSON_HEDLEY_TI_VERSION_CHECK(15,12,0) || \
  JSON_HEDLEY_TI_ARMCL_VERSION_CHECK(4,7,0) || \
  JSON_HEDLEY_TI_CL430_VERSION_CHECK(3,1,0) || \
  JSON_HEDLEY_TI_CL2000_VERSION_CHECK(6,1,0) || \
  JSON_HEDLEY_TI_CL6X_VERSION_CHECK(6,1,0) || \
  JSON_HEDLEY_TI_CL7X_VERSION_CHECK(1,2,0) || \
  JSON_HEDLEY_TI_CLPRU_VERSION_CHECK(2,1,0) || \
  JSON_HEDLEY_TINYC_VERSION_CHECK(0,9,27) || \
  JSON_HEDLEY_CRAY_VERSION_CHECK(8,1,0) || \
  JSON_HEDLEY_MCST_LCC_VERSION_CHECK(1,25,10)
#  define JSON_HEDLEY_PREDICT(expr, expected, probability) \
    (((probability) >= 0.9) ? __builtin_expect((expr), (expected)) : (JSON_HEDLEY_STATIC_CAST(void, expected), (expr)))
#  define JSON_HEDLEY_PREDICT_TRUE(expr, probability) \
    (__extension__ ({ \
        double hedley_probability_ = (probability); \
        ((hedley_probability_ >= 0.9) ? __builtin_expect(!!(expr), 1) : ((hedley_probability_ <= 0.1) ? __builtin_expect(!!(expr), 0) : !!(expr))); \
    }))
#  define JSON_HEDLEY_PREDICT_FALSE(expr, probability) \
    (__extension__ ({ \
        double hedley_probability_ = (probability); \
        ((hedley_probability_ >= 0.9) ? __builtin_expect(!!(expr), 0) : ((hedley_probability_ <= 0.1) ? __builtin_expect(!!(expr), 1) : !!(expr))); \
    }))
#  define JSON_HEDLEY_LIKELY(expr)   __builtin_expect(!!(expr), 1)
#  define JSON_HEDLEY_UNLIKELY(expr) __builtin_expect(!!(expr), 0)
#else
#  define JSON_HEDLEY_PREDICT(expr, expected, probability) (JSON_HEDLEY_STATIC_CAST(void, expected), (expr))
#  define JSON_HEDLEY_PREDICT_TRUE(expr, probability) (!!(expr))
#  define JSON_HEDLEY_PREDICT_FALSE(expr, probability) (!!(expr))
#  define JSON_HEDLEY_LIKELY(expr) (!!(expr))
#  define JSON_HEDLEY_UNLIKELY(expr) (!!(expr))
#endif
#if !defined(JSON_HEDLEY_UNPREDICTABLE)
    #define JSON_HEDLEY_UNPREDICTABLE(expr) JSON_HEDLEY_PREDICT(expr, 1, 0.5)
#endif

#if defined(JSON_HEDLEY_MALLOC)
    #undef JSON_HEDLEY_MALLOC
#endif
#if \
    JSON_HEDLEY_HAS_ATTRIBUTE(malloc) || \
    JSON_HEDLEY_GCC_VERSION_CHECK(3,1,0) || \
    JSON_HEDLEY_INTEL_VERSION_CHECK(13,0,0) || \
    JSON_HEDLEY_SUNPRO_VERSION_CHECK(5,11,0) || \
    JSON_HEDLEY_ARM_VERSION_CHECK(4,1,0) || \
    JSON_HEDLEY_IBM_VERSION_CHECK(12,1,0) || \
    JSON_HEDLEY_TI_VERSION_CHECK(15,12,0) || \
    (JSON_HEDLEY_TI_ARMCL_VERSION_CHECK(4,8,0) && defined(__TI_GNU_ATTRIBUTE_SUPPORT__)) || \
    JSON_HEDLEY_TI_ARMCL_VERSION_CHECK(5,2,0) || \
    (JSON_HEDLEY_TI_CL2000_VERSION_CHECK(6,0,0) && defined(__TI_GNU_ATTRIBUTE_SUPPORT__)) || \
    JSON_HEDLEY_TI_CL2000_VERSION_CHECK(6,4,0) || \
    (JSON_HEDLEY_TI_CL430_VERSION_CHECK(4,0,0) && defined(__TI_GNU_ATTRIBUTE_SUPPORT__)) || \
    JSON_HEDLEY_TI_CL430_VERSION_CHECK(4,3,0) || \
    (JSON_HEDLEY_TI_CL6X_VERSION_CHECK(7,2,0) && defined(__TI_GNU_ATTRIBUTE_SUPPORT__)) || \
    JSON_HEDLEY_TI_CL6X_VERSION_CHECK(7,5,0) || \
    JSON_HEDLEY_TI_CL7X_VERSION_CHECK(1,2,0) || \
    JSON_HEDLEY_TI_CLPRU_VERSION_CHECK(2,1,0) || \
    JSON_HEDLEY_MCST_LCC_VERSION_CHECK(1,25,10)
    #define JSON_HEDLEY_MALLOC __attribute__((__malloc__))
#elif JSON_HEDLEY_SUNPRO_VERSION_CHECK(5,10,0)
    #define JSON_HEDLEY_MALLOC _Pragma("returns_new_memory")
#elif \
    JSON_HEDLEY_MSVC_VERSION_CHECK(14,0,0) || \
    JSON_HEDLEY_INTEL_CL_VERSION_CHECK(2021,1,0)
    #define JSON_HEDLEY_MALLOC __declspec(restrict)
#else
    #define JSON_HEDLEY_MALLOC
#endif

#if defined(JSON_HEDLEY_PURE)
    #undef JSON_HEDLEY_PURE
#endif
#if \
  JSON_HEDLEY_HAS_ATTRIBUTE(pure) || \
  JSON_HEDLEY_GCC_VERSION_CHECK(2,96,0) || \
  JSON_HEDLEY_INTEL_VERSION_CHECK(13,0,0) || \
  JSON_HEDLEY_SUNPRO_VERSION_CHECK(5,11,0) || \
  JSON_HEDLEY_ARM_VERSION_CHECK(4,1,0) || \
  JSON_HEDLEY_IBM_VERSION_CHECK(10,1,0) || \
  JSON_HEDLEY_TI_VERSION_CHECK(15,12,0) || \
  (JSON_HEDLEY_TI_ARMCL_VERSION_CHECK(4,8,0) && defined(__TI_GNU_ATTRIBUTE_SUPPORT__)) || \
  JSON_HEDLEY_TI_ARMCL_VERSION_CHECK(5,2,0) || \
  (JSON_HEDLEY_TI_CL2000_VERSION_CHECK(6,0,0) && defined(__TI_GNU_ATTRIBUTE_SUPPORT__)) || \
  JSON_HEDLEY_TI_CL2000_VERSION_CHECK(6,4,0) || \
  (JSON_HEDLEY_TI_CL430_VERSION_CHECK(4,0,0) && defined(__TI_GNU_ATTRIBUTE_SUPPORT__)) || \
  JSON_HEDLEY_TI_CL430_VERSION_CHECK(4,3,0) || \
  (JSON_HEDLEY_TI_CL6X_VERSION_CHECK(7,2,0) && defined(__TI_GNU_ATTRIBUTE_SUPPORT__)) || \
  JSON_HEDLEY_TI_CL6X_VERSION_CHECK(7,5,0) || \
  JSON_HEDLEY_TI_CL7X_VERSION_CHECK(1,2,0) || \
  JSON_HEDLEY_TI_CLPRU_VERSION_CHECK(2,1,0) || \
  JSON_HEDLEY_PGI_VERSION_CHECK(17,10,0) || \
  JSON_HEDLEY_MCST_LCC_VERSION_CHECK(1,25,10)
#  define JSON_HEDLEY_PURE __attribute__((__pure__))
#elif JSON_HEDLEY_SUNPRO_VERSION_CHECK(5,10,0)
#  define JSON_HEDLEY_PURE _Pragma("does_not_write_global_data")
#elif defined(__cplusplus) && \
    ( \
      JSON_HEDLEY_TI_CL430_VERSION_CHECK(2,0,1) || \
      JSON_HEDLEY_TI_CL6X_VERSION_CHECK(4,0,0) || \
      JSON_HEDLEY_TI_CL7X_VERSION_CHECK(1,2,0) \
    )
#  define JSON_HEDLEY_PURE _Pragma("FUNC_IS_PURE;")
#else
#  define JSON_HEDLEY_PURE
#endif

#if defined(JSON_HEDLEY_CONST)
    #undef JSON_HEDLEY_CONST
#endif
#if \
    JSON_HEDLEY_HAS_ATTRIBUTE(const) || \
    JSON_HEDLEY_GCC_VERSION_CHECK(2,5,0) || \
    JSON_HEDLEY_INTEL_VERSION_CHECK(13,0,0) || \
    JSON_HEDLEY_SUNPRO_VERSION_CHECK(5,11,0) || \
    JSON_HEDLEY_ARM_VERSION_CHECK(4,1,0) || \
    JSON_HEDLEY_IBM_VERSION_CHECK(10,1,0) || \
    JSON_HEDLEY_TI_VERSION_CHECK(15,12,0) || \
    (JSON_HEDLEY_TI_ARMCL_VERSION_CHECK(4,8,0) && defined(__TI_GNU_ATTRIBUTE_SUPPORT__)) || \
    JSON_HEDLEY_TI_ARMCL_VERSION_CHECK(5,2,0) || \
    (JSON_HEDLEY_TI_CL2000_VERSION_CHECK(6,0,0) && defined(__TI_GNU_ATTRIBUTE_SUPPORT__)) || \
    JSON_HEDLEY_TI_CL2000_VERSION_CHECK(6,4,0) || \
    (JSON_HEDLEY_TI_CL430_VERSION_CHECK(4,0,0) && defined(__TI_GNU_ATTRIBUTE_SUPPORT__)) || \
    JSON_HEDLEY_TI_CL430_VERSION_CHECK(4,3,0) || \
    (JSON_HEDLEY_TI_CL6X_VERSION_CHECK(7,2,0) && defined(__TI_GNU_ATTRIBUTE_SUPPORT__)) || \
    JSON_HEDLEY_TI_CL6X_VERSION_CHECK(7,5,0) || \
    JSON_HEDLEY_TI_CL7X_VERSION_CHECK(1,2,0) || \
    JSON_HEDLEY_TI_CLPRU_VERSION_CHECK(2,1,0) || \
    JSON_HEDLEY_PGI_VERSION_CHECK(17,10,0) || \
    JSON_HEDLEY_MCST_LCC_VERSION_CHECK(1,25,10)
    #define JSON_HEDLEY_CONST __attribute__((__const__))
#elif \
    JSON_HEDLEY_SUNPRO_VERSION_CHECK(5,10,0)
    #define JSON_HEDLEY_CONST _Pragma("no_side_effect")
#else
    #define JSON_HEDLEY_CONST JSON_HEDLEY_PURE
#endif

#if defined(JSON_HEDLEY_RESTRICT)
    #undef JSON_HEDLEY_RESTRICT
#endif
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) && !defined(__cplusplus)
    #define JSON_HEDLEY_RESTRICT restrict
#elif \
    JSON_HEDLEY_GCC_VERSION_CHECK(3,1,0) || \
    JSON_HEDLEY_MSVC_VERSION_CHECK(14,0,0) || \
    JSON_HEDLEY_INTEL_VERSION_CHECK(13,0,0) || \
    JSON_HEDLEY_INTEL_CL_VERSION_CHECK(2021,1,0) || \
    JSON_HEDLEY_ARM_VERSION_CHECK(4,1,0) || \
    JSON_HEDLEY_IBM_VERSION_CHECK(10,1,0) || \
    JSON_HEDLEY_PGI_VERSION_CHECK(17,10,0) || \
    JSON_HEDLEY_TI_CL430_VERSION_CHECK(4,3,0) || \
    JSON_HEDLEY_TI_CL2000_VERSION_CHECK(6,2,4) || \
    JSON_HEDLEY_TI_CL6X_VERSION_CHECK(8,1,0) || \
    JSON_HEDLEY_TI_CL7X_VERSION_CHECK(1,2,0) || \
    (JSON_HEDLEY_SUNPRO_VERSION_CHECK(5,14,0) && defined(__cplusplus)) || \
    JSON_HEDLEY_IAR_VERSION_CHECK(8,0,0) || \
    defined(__clang__) || \
    JSON_HEDLEY_MCST_LCC_VERSION_CHECK(1,25,10)
    #define JSON_HEDLEY_RESTRICT __restrict
#elif JSON_HEDLEY_SUNPRO_VERSION_CHECK(5,3,0) && !defined(__cplusplus)
    #define JSON_HEDLEY_RESTRICT _Restrict
#else
    #define JSON_HEDLEY_RESTRICT
#endif

#if defined(JSON_HEDLEY_INLINE)
    #undef JSON_HEDLEY_INLINE
#endif
#if \
    (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L)) || \
    (defined(__cplusplus) && (__cplusplus >= 199711L))
    #define JSON_HEDLEY_INLINE inline
#elif \
    defined(JSON_HEDLEY_GCC_VERSION) || \
    JSON_HEDLEY_ARM_VERSION_CHECK(6,2,0)
    #define JSON_HEDLEY_INLINE __inline__
#elif \
    JSON_HEDLEY_MSVC_VERSION_CHECK(12,0,0) || \
    JSON_HEDLEY_INTEL_CL_VERSION_CHECK(2021,1,0) || \
    JSON_HEDLEY_ARM_VERSION_CHECK(4,1,0) || \
    JSON_HEDLEY_TI_ARMCL_VERSION_CHECK(5,1,0) || \
    JSON_HEDLEY_TI_CL430_VERSION_CHECK(3,1,0) || \
    JSON_HEDLEY_TI_CL2000_VERSION_CHECK(6,2,0) || \
    JSON_HEDLEY_TI_CL6X_VERSION_CHECK(8,0,0) || \
    JSON_HEDLEY_TI_CL7X_VERSION_CHECK(1,2,0) || \
    JSON_HEDLEY_TI_CLPRU_VERSION_CHECK(2,1,0) || \
    JSON_HEDLEY_MCST_LCC_VERSION_CHECK(1,25,10)
    #define JSON_HEDLEY_INLINE __inline
#else
    #define JSON_HEDLEY_INLINE
#endif

#if defined(JSON_HEDLEY_ALWAYS_INLINE)
    #undef JSON_HEDLEY_ALWAYS_INLINE
#endif
#if \
  JSON_HEDLEY_HAS_ATTRIBUTE(always_inline) || \
  JSON_HEDLEY_GCC_VERSION_CHECK(4,0,0) || \
  JSON_HEDLEY_INTEL_VERSION_CHECK(13,0,0) || \
  JSON_HEDLEY_SUNPRO_VERSION_CHECK(5,11,0) || \
  JSON_HEDLEY_ARM_VERSION_CHECK(4,1,0) || \
  JSON_HEDLEY_IBM_VERSION_CHECK(10,1,0) || \
  JSON_HEDLEY_TI_VERSION_CHECK(15,12,0) || \
  (JSON_HEDLEY_TI_ARMCL_VERSION_CHECK(4,8,0) && defined(__TI_GNU_ATTRIBUTE_SUPPORT__)) || \
  JSON_HEDLEY_TI_ARMCL_VERSION_CHECK(5,2,0) || \
  (JSON_HEDLEY_TI_CL2000_VERSION_CHECK(6,0,0) && defined(__TI_GNU_ATTRIBUTE_SUPPORT__)) || \
  JSON_HEDLEY_TI_CL2000_VERSION_CHECK(6,4,0) || \
  (JSON_HEDLEY_TI_CL430_VERSION_CHECK(4,0,0) && defined(__TI_GNU_ATTRIBUTE_SUPPORT__)) || \
  JSON_HEDLEY_TI_CL430_VERSION_CHECK(4,3,0) || \
  (JSON_HEDLEY_TI_CL6X_VERSION_CHECK(7,2,0) && defined(__TI_GNU_ATTRIBUTE_SUPPORT__)) || \
  JSON_HEDLEY_TI_CL6X_VERSION_CHECK(7,5,0) || \
  JSON_HEDLEY_TI_CL7X_VERSION_CHECK(1,2,0) || \
  JSON_HEDLEY_TI_CLPRU_VERSION_CHECK(2,1,0) || \
  JSON_HEDLEY_MCST_LCC_VERSION_CHECK(1,25,10) || \
  JSON_HEDLEY_IAR_VERSION_CHECK(8,10,0)
#  define JSON_HEDLEY_ALWAYS_INLINE __attribute__((__always_inline__)) JSON_HEDLEY_INLINE
#elif \
  JSON_HEDLEY_MSVC_VERSION_CHECK(12,0,0) || \
  JSON_HEDLEY_INTEL_CL_VERSION_CHECK(2021,1,0)
#  define JSON_HEDLEY_ALWAYS_INLINE __forceinline
#elif defined(__cplusplus) && \
    ( \
      JSON_HEDLEY_TI_ARMCL_VERSION_CHECK(5,2,0) || \
      JSON_HEDLEY_TI_CL430_VERSION_CHECK(4,3,0) || \
      JSON_HEDLEY_TI_CL2000_VERSION_CHECK(6,4,0) || \
      JSON_HEDLEY_TI_CL6X_VERSION_CHECK(6,1,0) || \
      JSON_HEDLEY_TI_CL7X_VERSION_CHECK(1,2,0) || \
      JSON_HEDLEY_TI_CLPRU_VERSION_CHECK(2,1,0) \
    )
#  define JSON_HEDLEY_ALWAYS_INLINE _Pragma("FUNC_ALWAYS_INLINE;")
#elif JSON_HEDLEY_IAR_VERSION_CHECK(8,0,0)
#  define JSON_HEDLEY_ALWAYS_INLINE _Pragma("inline=forced")
#else
#  define JSON_HEDLEY_ALWAYS_INLINE JSON_HEDLEY_INLINE
#endif

#if defined(JSON_HEDLEY_NEVER_INLINE)
    #undef JSON_HEDLEY_NEVER_INLINE
#endif
#if \
    JSON_HEDLEY_HAS_ATTRIBUTE(noinline) || \
    JSON_HEDLEY_GCC_VERSION_CHECK(4,0,0) || \
    JSON_HEDLEY_INTEL_VERSION_CHECK(13,0,0) || \
    JSON_HEDLEY_SUNPRO_VERSION_CHECK(5,11,0) || \
    JSON_HEDLEY_ARM_VERSION_CHECK(4,1,0) || \
    JSON_HEDLEY_IBM_VERSION_CHECK(10,1,0) || \
    JSON_HEDLEY_TI_VERSION_CHECK(15,12,0) || \
    (JSON_HEDLEY_TI_ARMCL_VERSION_CHECK(4,8,0) && defined(__TI_GNU_ATTRIBUTE_SUPPORT__)) || \
    JSON_HEDLEY_TI_ARMCL_VERSION_CHECK(5,2,0) || \
    (JSON_HEDLEY_TI_CL2000_VERSION_CHECK(6,0,0) && defined(__TI_GNU_ATTRIBUTE_SUPPORT__)) || \
    JSON_HEDLEY_TI_CL2000_VERSION_CHECK(6,4,0) || \
    (JSON_HEDLEY_TI_CL430_VERSION_CHECK(4,0,0) && defined(__TI_GNU_ATTRIBUTE_SUPPORT__)) || \
    JSON_HEDLEY_TI_CL430_VERSION_CHECK(4,3,0) || \
    (JSON_HEDLEY_TI_CL6X_VERSION_CHECK(7,2,0) && defined(__TI_GNU_ATTRIBUTE_SUPPORT__)) || \
    JSON_HEDLEY_TI_CL6X_VERSION_CHECK(7,5,0) || \
    JSON_HEDLEY_TI_CL7X_VERSION_CHECK(1,2,0) || \
    JSON_HEDLEY_TI_CLPRU_VERSION_CHECK(2,1,0) || \
    JSON_HEDLEY_MCST_LCC_VERSION_CHECK(1,25,10) || \
    JSON_HEDLEY_IAR_VERSION_CHECK(8,10,0)
    #define JSON_HEDLEY_NEVER_INLINE __attribute__((__noinline__))
#elif \
    JSON_HEDLEY_MSVC_VERSION_CHECK(13,10,0) || \
    JSON_HEDLEY_INTEL_CL_VERSION_CHECK(2021,1,0)
    #define JSON_HEDLEY_NEVER_INLINE __declspec(noinline)
#elif JSON_HEDLEY_PGI_VERSION_CHECK(10,2,0)
    #define JSON_HEDLEY_NEVER_INLINE _Pragma("noinline")
#elif JSON_HEDLEY_TI_CL6X_VERSION_CHECK(6,0,0) && defined(__cplusplus)
    #define JSON_HEDLEY_NEVER_INLINE _Pragma("FUNC_CANNOT_INLINE;")
#elif JSON_HEDLEY_IAR_VERSION_CHECK(8,0,0)
    #define JSON_HEDLEY_NEVER_INLINE _Pragma("inline=never")
#elif JSON_HEDLEY_COMPCERT_VERSION_CHECK(3,2,0)
    #define JSON_HEDLEY_NEVER_INLINE __attribute((noinline))
#elif JSON_HEDLEY_PELLES_VERSION_CHECK(9,0,0)
    #define JSON_HEDLEY_NEVER_INLINE __declspec(noinline)
#else
    #define JSON_HEDLEY_NEVER_INLINE
#endif

#if defined(JSON_HEDLEY_PRIVATE)
    #undef JSON_HEDLEY_PRIVATE
#endif
#if defined(JSON_HEDLEY_PUBLIC)
    #undef JSON_HEDLEY_PUBLIC
#endif
#if defined(JSON_HEDLEY_IMPORT)
    #undef JSON_HEDLEY_IMPORT
#endif
#if defined(_WIN32) || defined(__CYGWIN__)
#  define JSON_HEDLEY_PRIVATE
#  define JSON_HEDLEY_PUBLIC   __declspec(dllexport)
#  define JSON_HEDLEY_IMPORT   __declspec(dllimport)
#else
#  if \
    JSON_HEDLEY_HAS_ATTRIBUTE(visibility) || \
    JSON_HEDLEY_GCC_VERSION_CHECK(3,3,0) || \
    JSON_HEDLEY_SUNPRO_VERSION_CHECK(5,11,0) || \
    JSON_HEDLEY_INTEL_VERSION_CHECK(13,0,0) || \
    JSON_HEDLEY_ARM_VERSION_CHECK(4,1,0) || \
    JSON_HEDLEY_IBM_VERSION_CHECK(13,1,0) || \
    ( \
      defined(__TI_EABI__) && \
      ( \
        (JSON_HEDLEY_TI_CL6X_VERSION_CHECK(7,2,0) && defined(__TI_GNU_ATTRIBUTE_SUPPORT__)) || \
        JSON_HEDLEY_TI_CL6X_VERSION_CHECK(7,5,0) \
      ) \
    ) || \
    JSON_HEDLEY_MCST_LCC_VERSION_CHECK(1,25,10)
#    define JSON_HEDLEY_PRIVATE __attribute__((__visibility__("hidden")))
#    define JSON_HEDLEY_PUBLIC  __attribute__((__visibility__("default")))
#  else
#    define JSON_HEDLEY_PRIVATE
#    define JSON_HEDLEY_PUBLIC
#  endif
#  define JSON_HEDLEY_IMPORT    extern
#endif

#if defined(JSON_HEDLEY_NO_THROW)
    #undef JSON_HEDLEY_NO_THROW
#endif
#if \
    JSON_HEDLEY_HAS_ATTRIBUTE(nothrow) || \
    JSON_HEDLEY_GCC_VERSION_CHECK(3,3,0) || \
    JSON_HEDLEY_INTEL_VERSION_CHECK(13,0,0) || \
    JSON_HEDLEY_MCST_LCC_VERSION_CHECK(1,25,10)
    #define JSON_HEDLEY_NO_THROW __attribute__((__nothrow__))
#elif \
    JSON_HEDLEY_MSVC_VERSION_CHECK(13,1,0) || \
    JSON_HEDLEY_INTEL_CL_VERSION_CHECK(2021,1,0) || \
    JSON_HEDLEY_ARM_VERSION_CHECK(4,1,0)
    #define JSON_HEDLEY_NO_THROW __declspec(nothrow)
#else
    #define JSON_HEDLEY_NO_THROW
#endif

#if defined(JSON_HEDLEY_FALL_THROUGH)
    #undef JSON_HEDLEY_FALL_THROUGH
#endif
#if \
    JSON_HEDLEY_HAS_ATTRIBUTE(fallthrough) || \
    JSON_HEDLEY_GCC_VERSION_CHECK(7,0,0) || \
    JSON_HEDLEY_MCST_LCC_VERSION_CHECK(1,25,10)
    #define JSON_HEDLEY_FALL_THROUGH __attribute__((__fallthrough__))
#elif JSON_HEDLEY_HAS_CPP_ATTRIBUTE_NS(clang,fallthrough)
    #define JSON_HEDLEY_FALL_THROUGH JSON_HEDLEY_DIAGNOSTIC_DISABLE_CPP98_COMPAT_WRAP_([[clang::fallthrough]])
#elif JSON_HEDLEY_HAS_CPP_ATTRIBUTE(fallthrough)
    #define JSON_HEDLEY_FALL_THROUGH JSON_HEDLEY_DIAGNOSTIC_DISABLE_CPP98_COMPAT_WRAP_([[fallthrough]])
#elif defined(__fallthrough) /* SAL */
    #define JSON_HEDLEY_FALL_THROUGH __fallthrough
#else
    #define JSON_HEDLEY_FALL_THROUGH
#endif

#if defined(JSON_HEDLEY_RETURNS_NON_NULL)
    #undef JSON_HEDLEY_RETURNS_NON_NULL
#endif
#if \
    JSON_HEDLEY_HAS_ATTRIBUTE(returns_nonnull) || \
    JSON_HEDLEY_GCC_VERSION_CHECK(4,9,0) || \
    JSON_HEDLEY_MCST_LCC_VERSION_CHECK(1,25,10)
    #define JSON_HEDLEY_RETURNS_NON_NULL __attribute__((__returns_nonnull__))
#elif defined(_Ret_notnull_) /* SAL */
    #define JSON_HEDLEY_RETURNS_NON_NULL _Ret_notnull_
#else
    #define JSON_HEDLEY_RETURNS_NON_NULL
#endif

#if defined(JSON_HEDLEY_ARRAY_PARAM)
    #undef JSON_HEDLEY_ARRAY_PARAM
#endif
#if \
    defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) && \
    !defined(__STDC_NO_VLA__) && \
    !defined(__cplusplus) && \
    !defined(JSON_HEDLEY_PGI_VERSION) && \
    !defined(JSON_HEDLEY_TINYC_VERSION)
    #define JSON_HEDLEY_ARRAY_PARAM(name) (name)
#else
    #define JSON_HEDLEY_ARRAY_PARAM(name)
#endif

#if defined(JSON_HEDLEY_IS_CONSTANT)
    #undef JSON_HEDLEY_IS_CONSTANT
#endif
#if defined(JSON_HEDLEY_REQUIRE_CONSTEXPR)
    #undef JSON_HEDLEY_REQUIRE_CONSTEXPR
#endif
/* JSON_HEDLEY_IS_CONSTEXPR_ is for
   HEDLEY INTERNAL USE ONLY.  API subject to change without notice. */
#if defined(JSON_HEDLEY_IS_CONSTEXPR_)
    #undef JSON_HEDLEY_IS_CONSTEXPR_
#endif
#if \
    JSON_HEDLEY_HAS_BUILTIN(__builtin_constant_p) || \
    JSON_HEDLEY_GCC_VERSION_CHECK(3,4,0) || \
    JSON_HEDLEY_INTEL_VERSION_CHECK(13,0,0) || \
    JSON_HEDLEY_TINYC_VERSION_CHECK(0,9,19) || \
    JSON_HEDLEY_ARM_VERSION_CHECK(4,1,0) || \
    JSON_HEDLEY_IBM_VERSION_CHECK(13,1,0) || \
    JSON_HEDLEY_TI_CL6X_VERSION_CHECK(6,1,0) || \
    (JSON_HEDLEY_SUNPRO_VERSION_CHECK(5,10,0) && !defined(__cplusplus)) || \
    JSON_HEDLEY_CRAY_VERSION_CHECK(8,1,0) || \
    JSON_HEDLEY_MCST_LCC_VERSION_CHECK(1,25,10)
    #define JSON_HEDLEY_IS_CONSTANT(expr) __builtin_constant_p(expr)
#endif
#if !defined(__cplusplus)
#  if \
       JSON_HEDLEY_HAS_BUILTIN(__builtin_types_compatible_p) || \
       JSON_HEDLEY_GCC_VERSION_CHECK(3,4,0) || \
       JSON_HEDLEY_INTEL_VERSION_CHECK(13,0,0) || \
       JSON_HEDLEY_IBM_VERSION_CHECK(13,1,0) || \
       JSON_HEDLEY_CRAY_VERSION_CHECK(8,1,0) || \
       JSON_HEDLEY_ARM_VERSION_CHECK(5,4,0) || \
       JSON_HEDLEY_TINYC_VERSION_CHECK(0,9,24)
#if defined(__INTPTR_TYPE__)
    #define JSON_HEDLEY_IS_CONSTEXPR_(expr) __builtin_types_compatible_p(__typeof__((1 ? (void*) ((__INTPTR_TYPE__) ((expr) * 0)) : (int*) 0)), int*)
#else
    #include <stdint.h>
    #define JSON_HEDLEY_IS_CONSTEXPR_(expr) __builtin_types_compatible_p(__typeof__((1 ? (void*) ((intptr_t) ((expr) * 0)) : (int*) 0)), int*)
#endif
#  elif \
       ( \
          defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L) && \
          !defined(JSON_HEDLEY_SUNPRO_VERSION) && \
          !defined(JSON_HEDLEY_PGI_VERSION) && \
          !defined(JSON_HEDLEY_IAR_VERSION)) || \
       (JSON_HEDLEY_HAS_EXTENSION(c_generic_selections) && !defined(JSON_HEDLEY_IAR_VERSION)) || \
       JSON_HEDLEY_GCC_VERSION_CHECK(4,9,0) || \
       JSON_HEDLEY_INTEL_VERSION_CHECK(17,0,0) || \
       JSON_HEDLEY_IBM_VERSION_CHECK(12,1,0) || \
       JSON_HEDLEY_ARM_VERSION_CHECK(5,3,0)
#if defined(__INTPTR_TYPE__)
    #define JSON_HEDLEY_IS_CONSTEXPR_(expr) _Generic((1 ? (void*) ((__INTPTR_TYPE__) ((expr) * 0)) : (int*) 0), int*: 1, void*: 0)
#else
    #include <stdint.h>
    #define JSON_HEDLEY_IS_CONSTEXPR_(expr) _Generic((1 ? (void*) ((intptr_t) * 0) : (int*) 0), int*: 1, void*: 0)
#endif
#  elif \
       defined(JSON_HEDLEY_GCC_VERSION) || \
       defined(JSON_HEDLEY_INTEL_VERSION) || \
       defined(JSON_HEDLEY_TINYC_VERSION) || \
       defined(JSON_HEDLEY_TI_ARMCL_VERSION) || \
       JSON_HEDLEY_TI_CL430_VERSION_CHECK(18,12,0) || \
       defined(JSON_HEDLEY_TI_CL2000_VERSION) || \
       defined(JSON_HEDLEY_TI_CL6X_VERSION) || \
       defined(JSON_HEDLEY_TI_CL7X_VERSION) || \
       defined(JSON_HEDLEY_TI_CLPRU_VERSION) || \
       defined(__clang__)
#    define JSON_HEDLEY_IS_CONSTEXPR_(expr) ( \
        sizeof(void) != \
        sizeof(*( \
                  1 ? \
                  ((void*) ((expr) * 0L) ) : \
((struct { char v[sizeof(void) * 2]; } *) 1) \
                ) \
              ) \
                                            )
#  endif
#endif
#if defined(JSON_HEDLEY_IS_CONSTEXPR_)
    #if !defined(JSON_HEDLEY_IS_CONSTANT)
        #define JSON_HEDLEY_IS_CONSTANT(expr) JSON_HEDLEY_IS_CONSTEXPR_(expr)
    #endif
    #define JSON_HEDLEY_REQUIRE_CONSTEXPR(expr) (JSON_HEDLEY_IS_CONSTEXPR_(expr) ? (expr) : (-1))
#else
    #if !defined(JSON_HEDLEY_IS_CONSTANT)
        #define JSON_HEDLEY_IS_CONSTANT(expr) (0)
    #endif
    #define JSON_HEDLEY_REQUIRE_CONSTEXPR(expr) (expr)
#endif

#if defined(JSON_HEDLEY_BEGIN_C_DECLS)
    #undef JSON_HEDLEY_BEGIN_C_DECLS
#endif
#if defined(JSON_HEDLEY_END_C_DECLS)
    #undef JSON_HEDLEY_END_C_DECLS
#endif
#if defined(JSON_HEDLEY_C_DECL)
    #undef JSON_HEDLEY_C_DECL
#endif
#if defined(__cplusplus)
    #define JSON_HEDLEY_BEGIN_C_DECLS extern "C" {
    #define JSON_HEDLEY_END_C_DECLS }
    #define JSON_HEDLEY_C_DECL extern "C"
#else
    #define JSON_HEDLEY_BEGIN_C_DECLS
    #define JSON_HEDLEY_END_C_DECLS
    #define JSON_HEDLEY_C_DECL
#endif

#if defined(JSON_HEDLEY_STATIC_ASSERT)
    #undef JSON_HEDLEY_STATIC_ASSERT
#endif
#if \
  !defined(__cplusplus) && ( \
      (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)) || \
      (JSON_HEDLEY_HAS_FEATURE(c_static_assert) && !defined(JSON_HEDLEY_INTEL_CL_VERSION)) || \
      JSON_HEDLEY_GCC_VERSION_CHECK(6,0,0) || \
      JSON_HEDLEY_INTEL_VERSION_CHECK(13,0,0) || \
      defined(_Static_assert) \
    )
#  define JSON_HEDLEY_STATIC_ASSERT(expr, message) _Static_assert(expr, message)
#elif \
  (defined(__cplusplus) && (__cplusplus >= 201103L)) || \
  JSON_HEDLEY_MSVC_VERSION_CHECK(16,0,0) || \
  JSON_HEDLEY_INTEL_CL_VERSION_CHECK(2021,1,0)
#  define JSON_HEDLEY_STATIC_ASSERT(expr, message) JSON_HEDLEY_DIAGNOSTIC_DISABLE_CPP98_COMPAT_WRAP_(static_assert(expr, message))
#else
#  define JSON_HEDLEY_STATIC_ASSERT(expr, message)
#endif

#if defined(JSON_HEDLEY_NULL)
    #undef JSON_HEDLEY_NULL
#endif
#if defined(__cplusplus)
    #if __cplusplus >= 201103L
        #define JSON_HEDLEY_NULL JSON_HEDLEY_DIAGNOSTIC_DISABLE_CPP98_COMPAT_WRAP_(nullptr)
    #elif defined(NULL)
        #define JSON_HEDLEY_NULL NULL
    #else
        #define JSON_HEDLEY_NULL JSON_HEDLEY_STATIC_CAST(void*, 0)
    #endif
#elif defined(NULL)
    #define JSON_HEDLEY_NULL NULL
#else
    #define JSON_HEDLEY_NULL ((void*) 0)
#endif

#if defined(JSON_HEDLEY_MESSAGE)
    #undef JSON_HEDLEY_MESSAGE
#endif
#if JSON_HEDLEY_HAS_WARNING("-Wunknown-pragmas")
#  define JSON_HEDLEY_MESSAGE(msg) \
    JSON_HEDLEY_DIAGNOSTIC_PUSH \
    JSON_HEDLEY_DIAGNOSTIC_DISABLE_UNKNOWN_PRAGMAS \
    JSON_HEDLEY_PRAGMA(message msg) \
    JSON_HEDLEY_DIAGNOSTIC_POP
#elif \
  JSON_HEDLEY_GCC_VERSION_CHECK(4,4,0) || \
  JSON_HEDLEY_INTEL_VERSION_CHECK(13,0,0)
#  define JSON_HEDLEY_MESSAGE(msg) JSON_HEDLEY_PRAGMA(message msg)
#elif JSON_HEDLEY_CRAY_VERSION_CHECK(5,0,0)
#  define JSON_HEDLEY_MESSAGE(msg) JSON_HEDLEY_PRAGMA(_CRI message msg)
#elif JSON_HEDLEY_IAR_VERSION_CHECK(8,0,0)
#  define JSON_HEDLEY_MESSAGE(msg) JSON_HEDLEY_PRAGMA(message(msg))
#elif JSON_HEDLEY_PELLES_VERSION_CHECK(2,0,0)
#  define JSON_HEDLEY_MESSAGE(msg) JSON_HEDLEY_PRAGMA(message(msg))
#else
#  define JSON_HEDLEY_MESSAGE(msg)
#endif

#if defined(JSON_HEDLEY_WARNING)
    #undef JSON_HEDLEY_WARNING
#endif
#if JSON_HEDLEY_HAS_WARNING("-Wunknown-pragmas")
#  define JSON_HEDLEY_WARNING(msg) \
    JSON_HEDLEY_DIAGNOSTIC_PUSH \
    JSON_HEDLEY_DIAGNOSTIC_DISABLE_UNKNOWN_PRAGMAS \
    JSON_HEDLEY_PRAGMA(clang warning msg) \
    JSON_HEDLEY_DIAGNOSTIC_POP
#elif \
  JSON_HEDLEY_GCC_VERSION_CHECK(4,8,0) || \
  JSON_HEDLEY_PGI_VERSION_CHECK(18,4,0) || \
  JSON_HEDLEY_INTEL_VERSION_CHECK(13,0,0)
#  define JSON_HEDLEY_WARNING(msg) JSON_HEDLEY_PRAGMA(GCC warning msg)
#elif \
  JSON_HEDLEY_MSVC_VERSION_CHECK(15,0,0) || \
  JSON_HEDLEY_INTEL_CL_VERSION_CHECK(2021,1,0)
#  define JSON_HEDLEY_WARNING(msg) JSON_HEDLEY_PRAGMA(message(msg))
#else
#  define JSON_HEDLEY_WARNING(msg) JSON_HEDLEY_MESSAGE(msg)
#endif

#if defined(JSON_HEDLEY_REQUIRE)
    #undef JSON_HEDLEY_REQUIRE
#endif
#if defined(JSON_HEDLEY_REQUIRE_MSG)
    #undef JSON_HEDLEY_REQUIRE_MSG
#endif
#if JSON_HEDLEY_HAS_ATTRIBUTE(diagnose_if)
#  if JSON_HEDLEY_HAS_WARNING("-Wgcc-compat")
#    define JSON_HEDLEY_REQUIRE(expr) \
    JSON_HEDLEY_DIAGNOSTIC_PUSH \
    _Pragma("clang diagnostic ignored \"-Wgcc-compat\"") \
    __attribute__((diagnose_if(!(expr), #expr, "error"))) \
    JSON_HEDLEY_DIAGNOSTIC_POP
#    define JSON_HEDLEY_REQUIRE_MSG(expr,msg) \
    JSON_HEDLEY_DIAGNOSTIC_PUSH \
    _Pragma("clang diagnostic ignored \"-Wgcc-compat\"") \
    __attribute__((diagnose_if(!(expr), msg, "error"))) \
    JSON_HEDLEY_DIAGNOSTIC_POP
#  else
#    define JSON_HEDLEY_REQUIRE(expr) __attribute__((diagnose_if(!(expr), #expr, "error")))
#    define JSON_HEDLEY_REQUIRE_MSG(expr,msg) __attribute__((diagnose_if(!(expr), msg, "error")))
#  endif
#else
#  define JSON_HEDLEY_REQUIRE(expr)
#  define JSON_HEDLEY_REQUIRE_MSG(expr,msg)
#endif

#if defined(JSON_HEDLEY_FLAGS)
    #undef JSON_HEDLEY_FLAGS
#endif
#if JSON_HEDLEY_HAS_ATTRIBUTE(flag_enum) && (!defined(__cplusplus) || JSON_HEDLEY_HAS_WARNING("-Wbitfield-enum-conversion"))
    #define JSON_HEDLEY_FLAGS __attribute__((__flag_enum__))
#else
    #define JSON_HEDLEY_FLAGS
#endif

#if defined(JSON_HEDLEY_FLAGS_CAST)
    #undef JSON_HEDLEY_FLAGS_CAST
#endif
#if JSON_HEDLEY_INTEL_VERSION_CHECK(19,0,0)
#  define JSON_HEDLEY_FLAGS_CAST(T, expr) (__extension__ ({ \
        JSON_HEDLEY_DIAGNOSTIC_PUSH \
        _Pragma("warning(disable:188)") \
        ((T) (expr)); \
        JSON_HEDLEY_DIAGNOSTIC_POP \
    }))
#else
#  define JSON_HEDLEY_FLAGS_CAST(T, expr) JSON_HEDLEY_STATIC_CAST(T, expr)
#endif

#if defined(JSON_HEDLEY_EMPTY_BASES)
    #undef JSON_HEDLEY_EMPTY_BASES
#endif
#if \
    (JSON_HEDLEY_MSVC_VERSION_CHECK(19,0,23918) && !JSON_HEDLEY_MSVC_VERSION_CHECK(20,0,0)) || \
    JSON_HEDLEY_INTEL_CL_VERSION_CHECK(2021,1,0)
    #define JSON_HEDLEY_EMPTY_BASES __declspec(empty_bases)
#else
    #define JSON_HEDLEY_EMPTY_BASES
#endif

/* Remaining macros are deprecated. */

#if defined(JSON_HEDLEY_GCC_NOT_CLANG_VERSION_CHECK)
    #undef JSON_HEDLEY_GCC_NOT_CLANG_VERSION_CHECK
#endif
#if defined(__clang__)
    #define JSON_HEDLEY_GCC_NOT_CLANG_VERSION_CHECK(major,minor,patch) (0)
#else
    #define JSON_HEDLEY_GCC_NOT_CLANG_VERSION_CHECK(major,minor,patch) JSON_HEDLEY_GCC_VERSION_CHECK(major,minor,patch)
#endif

#if defined(JSON_HEDLEY_CLANG_HAS_ATTRIBUTE)
    #undef JSON_HEDLEY_CLANG_HAS_ATTRIBUTE
#endif
#define JSON_HEDLEY_CLANG_HAS_ATTRIBUTE(attribute) JSON_HEDLEY_HAS_ATTRIBUTE(attribute)

#if defined(JSON_HEDLEY_CLANG_HAS_CPP_ATTRIBUTE)
    #undef JSON_HEDLEY_CLANG_HAS_CPP_ATTRIBUTE
#endif
#define JSON_HEDLEY_CLANG_HAS_CPP_ATTRIBUTE(attribute) JSON_HEDLEY_HAS_CPP_ATTRIBUTE(attribute)

#if defined(JSON_HEDLEY_CLANG_HAS_BUILTIN)
    #undef JSON_HEDLEY_CLANG_HAS_BUILTIN
#endif
#define JSON_HEDLEY_CLANG_HAS_BUILTIN(builtin) JSON_HEDLEY_HAS_BUILTIN(builtin)

#if defined(JSON_HEDLEY_CLANG_HAS_FEATURE)
    #undef JSON_HEDLEY_CLANG_HAS_FEATURE
#endif
#define JSON_HEDLEY_CLANG_HAS_FEATURE(feature) JSON_HEDLEY_HAS_FEATURE(feature)

#if defined(JSON_HEDLEY_CLANG_HAS_EXTENSION)
    #undef JSON_HEDLEY_CLANG_HAS_EXTENSION
#endif
#define JSON_HEDLEY_CLANG_HAS_EXTENSION(extension) JSON_HEDLEY_HAS_EXTENSION(extension)

#if defined(JSON_HEDLEY_CLANG_HAS_DECLSPEC_DECLSPEC_ATTRIBUTE)
    #undef JSON_HEDLEY_CLANG_HAS_DECLSPEC_DECLSPEC_ATTRIBUTE
#endif
#define JSON_HEDLEY_CLANG_HAS_DECLSPEC_ATTRIBUTE(attribute) JSON_HEDLEY_HAS_DECLSPEC_ATTRIBUTE(attribute)

#if defined(JSON_HEDLEY_CLANG_HAS_WARNING)
    #undef JSON_HEDLEY_CLANG_HAS_WARNING
#endif
#define JSON_HEDLEY_CLANG_HAS_WARNING(warning) JSON_HEDLEY_HAS_WARNING(warning)

#endif /* !defined(JSON_HEDLEY_VERSION) || (JSON_HEDLEY_VERSION < X) */


// This file contains all internal macro definitions (except those affecting ABI)
// You MUST include macro_unscope.hpp at the end of json.hpp to undef all of them

// #include <nlohmann/detail/abi_macros.hpp>


// exclude unsupported compilers
#if !defined(JSON_SKIP_UNSUPPORTED_COMPILER_CHECK)
    #if defined(__clang__)
        #if (__clang_major__ * 10000 + __clang_minor__ * 100 + __clang_patchlevel__) < 30400
            #error "unsupported Clang version - see https://github.com/nlohmann/json#supported-compilers"
        #endif
    #elif defined(__GNUC__) && !(defined(__ICC) || defined(__INTEL_COMPILER))
        #if (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__) < 40800
            #error "unsupported GCC version - see https://github.com/nlohmann/json#supported-compilers"
        #endif
    #endif
#endif

// C++ language standard detection
// if the user manually specified the used c++ version this is skipped
#if !defined(JSON_HAS_CPP_23) && !defined(JSON_HAS_CPP_20) && !defined(JSON_HAS_CPP_17) && !defined(JSON_HAS_CPP_14) && !defined(JSON_HAS_CPP_11)
    #if (defined(__cplusplus) && __cplusplus > 202002L) || (defined(_MSVC_LANG) && _MSVC_LANG > 202002L)
        #define JSON_HAS_CPP_23
        #define JSON_HAS_CPP_20
        #define JSON_HAS_CPP_17
        #define JSON_HAS_CPP_14
    #elif (defined(__cplusplus) && __cplusplus > 201703L) || (defined(_MSVC_LANG) && _MSVC_LANG > 201703L)
        #define JSON_HAS_CPP_20
        #define JSON_HAS_CPP_17
        #define JSON_HAS_CPP_14
    #elif (defined(__cplusplus) && __cplusplus > 201402L) || (defined(_HAS_CXX17) && _HAS_CXX17 == 1) // fix for issue #464
        #define JSON_HAS_CPP_17
        #define JSON_HAS_CPP_14
    #elif (defined(__cplusplus) && __cplusplus > 201103L) || (defined(_HAS_CXX14) && _HAS_CXX14 == 1)
        #define JSON_HAS_CPP_14
    #endif
    // the cpp 11 flag is always specified because it is the minimal required version
    #define JSON_HAS_CPP_11
#endif

#ifdef __has_include
    #if __has_include(<version>)
        #include <version>
    #endif
#endif

#if !defined(JSON_HAS_FILESYSTEM) && !defined(JSON_HAS_EXPERIMENTAL_FILESYSTEM)
    #ifdef JSON_HAS_CPP_17
        #if defined(__cpp_lib_filesystem)
            #define JSON_HAS_FILESYSTEM 1
        #elif defined(__cpp_lib_experimental_filesystem)
            #define JSON_HAS_EXPERIMENTAL_FILESYSTEM 1
        #elif !defined(__has_include)
            #define JSON_HAS_EXPERIMENTAL_FILESYSTEM 1
        #elif __has_include(<filesystem>)
            #define JSON_HAS_FILESYSTEM 1
        #elif __has_include(<experimental/filesystem>)
            #define JSON_HAS_EXPERIMENTAL_FILESYSTEM 1
        #endif

        // std::filesystem does not work on MinGW GCC 8: https://sourceforge.net/p/mingw-w64/bugs/737/
        #if defined(__MINGW32__) && defined(__GNUC__) && __GNUC__ == 8
            #undef JSON_HAS_FILESYSTEM
            #undef JSON_HAS_EXPERIMENTAL_FILESYSTEM
        #endif

        // no filesystem support before GCC 8: https://en.cppreference.com/w/cpp/compiler_support
        #if defined(__GNUC__) && !defined(__clang__) && __GNUC__ < 8
            #undef JSON_HAS_FILESYSTEM
            #undef JSON_HAS_EXPERIMENTAL_FILESYSTEM
        #endif

        // no filesystem support before Clang 7: https://en.cppreference.com/w/cpp/compiler_support
        #if defined(__clang_major__) && __clang_major__ < 7
            #undef JSON_HAS_FILESYSTEM
            #undef JSON_HAS_EXPERIMENTAL_FILESYSTEM
        #endif

        // no filesystem support before MSVC 19.14: https://en.cppreference.com/w/cpp/compiler_support
        #if defined(_MSC_VER) && _MSC_VER < 1914
            #undef JSON_HAS_FILESYSTEM
            #undef JSON_HAS_EXPERIMENTAL_FILESYSTEM
        #endif

        // no filesystem support before iOS 13
        #if defined(__IPHONE_OS_VERSION_MIN_REQUIRED) && __IPHONE_OS_VERSION_MIN_REQUIRED < 130000
            #undef JSON_HAS_FILESYSTEM
            #undef JSON_HAS_EXPERIMENTAL_FILESYSTEM
        #endif

        // no filesystem support before macOS Catalina
        #if defined(__MAC_OS_X_VERSION_MIN_REQUIRED) && __MAC_OS_X_VERSION_MIN_REQUIRED < 101500
            #undef JSON_HAS_FILESYSTEM
            #undef JSON_HAS_EXPERIMENTAL_FILESYSTEM
        #endif
    #endif
#endif

#ifndef JSON_HAS_EXPERIMENTAL_FILESYSTEM
    #define JSON_HAS_EXPERIMENTAL_FILESYSTEM 0
#endif

#ifndef JSON_HAS_FILESYSTEM
    #define JSON_HAS_FILESYSTEM 0
#endif

#ifndef JSON_HAS_THREE_WAY_COMPARISON
    #if defined(__cpp_impl_three_way_comparison) && __cpp_impl_three_way_comparison >= 201907L \
        && defined(__cpp_lib_three_way_comparison) && __cpp_lib_three_way_comparison >= 201907L
        #define JSON_HAS_THREE_WAY_COMPARISON 1
    #else
        #define JSON_HAS_THREE_WAY_COMPARISON 0
    #endif
#endif

#ifndef JSON_HAS_RANGES
    // ranges header shipping in GCC 11.1.0 (released 2021-04-27) has syntax error
    #if defined(__GLIBCXX__) && __GLIBCXX__ == 20210427
        #define JSON_HAS_RANGES 0
    #elif defined(__cpp_lib_ranges)
        #define JSON_HAS_RANGES 1
    #else
        #define JSON_HAS_RANGES 0
    #endif
#endif

#ifndef JSON_HAS_STATIC_RTTI
    #if !defined(_HAS_STATIC_RTTI) || _HAS_STATIC_RTTI != 0
        #define JSON_HAS_STATIC_RTTI 1
    #else
        #define JSON_HAS_STATIC_RTTI 0
    #endif
#endif

#ifdef JSON_HAS_CPP_17
    #define JSON_INLINE_VARIABLE inline
#else
    #define JSON_INLINE_VARIABLE
#endif

#if JSON_HEDLEY_HAS_ATTRIBUTE(no_unique_address)
    #define JSON_NO_UNIQUE_ADDRESS [[no_unique_address]]
#else
    #define JSON_NO_UNIQUE_ADDRESS
#endif

// disable documentation warnings on clang
#if defined(__clang__)
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wdocumentation"
    #pragma clang diagnostic ignored "-Wdocumentation-unknown-command"
#endif

// allow disabling exceptions
#if (defined(__cpp_exceptions) || defined(__EXCEPTIONS) || defined(_CPPUNWIND)) && !defined(JSON_NOEXCEPTION)
    #define JSON_THROW(exception) throw exception
    #define JSON_TRY try
    #define JSON_CATCH(exception) catch(exception)
    #define JSON_INTERNAL_CATCH(exception) catch(exception)
#else
    #include <cstdlib>
    #define JSON_THROW(exception) std::abort()
    #define JSON_TRY if(true)
    #define JSON_CATCH(exception) if(false)
    #define JSON_INTERNAL_CATCH(exception) if(false)
#endif

// override exception macros
#if defined(JSON_THROW_USER)
    #undef JSON_THROW
    #define JSON_THROW JSON_THROW_USER
#endif
#if defined(JSON_TRY_USER)
    #undef JSON_TRY
    #define JSON_TRY JSON_TRY_USER
#endif
#if defined(JSON_CATCH_USER)
    #undef JSON_CATCH
    #define JSON_CATCH JSON_CATCH_USER
    #undef JSON_INTERNAL_CATCH
    #define JSON_INTERNAL_CATCH JSON_CATCH_USER
#endif
#if defined(JSON_INTERNAL_CATCH_USER)
    #undef JSON_INTERNAL_CATCH
    #define JSON_INTERNAL_CATCH JSON_INTERNAL_CATCH_USER
#endif

// allow overriding assert
#if !defined(JSON_ASSERT)
    #include <cassert> // assert
    #define JSON_ASSERT(x) assert(x)
#endif

// allow to access some private functions (needed by the test suite)
#if defined(JSON_TESTS_PRIVATE)
    #define JSON_PRIVATE_UNLESS_TESTED public
#else
    #define JSON_PRIVATE_UNLESS_TESTED private
#endif

/*!
@brief macro to briefly define a mapping between an enum and JSON
@def NLOHMANN_JSON_SERIALIZE_ENUM
@since version 3.4.0
*/
#define NLOHMANN_JSON_SERIALIZE_ENUM(ENUM_TYPE, ...)                                            \
    template<typename BasicJsonType>                                                            \
    inline void to_json(BasicJsonType& j, const ENUM_TYPE& e)                                   \
    {                                                                                           \
        /* NOLINTNEXTLINE(modernize-type-traits) we use C++11 */                                \
        static_assert(std::is_enum<ENUM_TYPE>::value, #ENUM_TYPE " must be an enum!");          \
        /* NOLINTNEXTLINE(modernize-avoid-c-arrays) we don't want to depend on <array> */       \
        static const std::pair<ENUM_TYPE, BasicJsonType> m[] = __VA_ARGS__;                     \
        auto it = std::find_if(std::begin(m), std::end(m),                                      \
                               [e](const std::pair<ENUM_TYPE, BasicJsonType>& ej_pair) -> bool  \
        {                                                                                       \
            return ej_pair.first == e;                                                          \
        });                                                                                     \
        j = ((it != std::end(m)) ? it : std::begin(m))->second;                                 \
    }                                                                                           \
    template<typename BasicJsonType>                                                            \
    inline void from_json(const BasicJsonType& j, ENUM_TYPE& e)                                 \
    {                                                                                           \
        /* NOLINTNEXTLINE(modernize-type-traits) we use C++11 */                                \
        static_assert(std::is_enum<ENUM_TYPE>::value, #ENUM_TYPE " must be an enum!");          \
        /* NOLINTNEXTLINE(modernize-avoid-c-arrays) we don't want to depend on <array> */       \
        static const std::pair<ENUM_TYPE, BasicJsonType> m[] = __VA_ARGS__;                     \
        auto it = std::find_if(std::begin(m), std::end(m),                                      \
                               [&j](const std::pair<ENUM_TYPE, BasicJsonType>& ej_pair) -> bool \
        {                                                                                       \
            return ej_pair.second == j;                                                         \
        });                                                                                     \
        e = ((it != std::end(m)) ? it : std::begin(m))->first;                                  \
    }

// Ugly macros to avoid uglier copy-paste when specializing basic_json. They
// may be removed in the future once the class is split.

#define NLOHMANN_BASIC_JSON_TPL_DECLARATION                                \
    template<template<typename, typename, typename...> class ObjectType,   \
             template<typename, typename...> class ArrayType,              \
             class StringType, class BooleanType, class NumberIntegerType, \
             class NumberUnsignedType, class NumberFloatType,              \
             template<typename> class AllocatorType,                       \
             template<typename, typename = void> class JSONSerializer,     \
             class BinaryType,                                             \
             class CustomBaseClass>

#define NLOHMANN_BASIC_JSON_TPL                                            \
    basic_json<ObjectType, ArrayType, StringType, BooleanType,             \
    NumberIntegerType, NumberUnsignedType, NumberFloatType,                \
    AllocatorType, JSONSerializer, BinaryType, CustomBaseClass>

// Macros to simplify conversion from/to types

#define NLOHMANN_JSON_EXPAND( x ) x
#define NLOHMANN_JSON_GET_MACRO(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42, _43, _44, _45, _46, _47, _48, _49, _50, _51, _52, _53, _54, _55, _56, _57, _58, _59, _60, _61, _62, _63, _64, NAME,...) NAME
#define NLOHMANN_JSON_PASTE(...) NLOHMANN_JSON_EXPAND(NLOHMANN_JSON_GET_MACRO(__VA_ARGS__, \
        NLOHMANN_JSON_PASTE64, \
        NLOHMANN_JSON_PASTE63, \
        NLOHMANN_JSON_PASTE62, \
        NLOHMANN_JSON_PASTE61, \
        NLOHMANN_JSON_PASTE60, \
        NLOHMANN_JSON_PASTE59, \
        NLOHMANN_JSON_PASTE58, \
        NLOHMANN_JSON_PASTE57, \
        NLOHMANN_JSON_PASTE56, \
        NLOHMANN_JSON_PASTE55, \
        NLOHMANN_JSON_PASTE54, \
        NLOHMANN_JSON_PASTE53, \
        NLOHMANN_JSON_PASTE52, \
        NLOHMANN_JSON_PASTE51, \
        NLOHMANN_JSON_PASTE50, \
        NLOHMANN_JSON_PASTE49, \
        NLOHMANN_JSON_PASTE48, \
        NLOHMANN_JSON_PASTE47, \
        NLOHMANN_JSON_PASTE46, \
        NLOHMANN_JSON_PASTE45, \
        NLOHMANN_JSON_PASTE44, \
        NLOHMANN_JSON_PASTE43, \
        NLOHMANN_JSON_PASTE42, \
        NLOHMANN_JSON_PASTE41, \
        NLOHMANN_JSON_PASTE40, \
        NLOHMANN_JSON_PASTE39, \
        NLOHMANN_JSON_PASTE38, \
        NLOHMANN_JSON_PASTE37, \
        NLOHMANN_JSON_PASTE36, \
        NLOHMANN_JSON_PASTE35, \
        NLOHMANN_JSON_PASTE34, \
        NLOHMANN_JSON_PASTE33, \
        NLOHMANN_JSON_PASTE32, \
        NLOHMANN_JSON_PASTE31, \
        NLOHMANN_JSON_PASTE30, \
        NLOHMANN_JSON_PASTE29, \
        NLOHMANN_JSON_PASTE28, \
        NLOHMANN_JSON_PASTE27, \
        NLOHMANN_JSON_PASTE26, \
        NLOHMANN_JSON_PASTE25, \
        NLOHMANN_JSON_PASTE24, \
        NLOHMANN_JSON_PASTE23, \
        NLOHMANN_JSON_PASTE22, \
        NLOHMANN_JSON_PASTE21, \
        NLOHMANN_JSON_PASTE20, \
        NLOHMANN_JSON_PASTE19, \
        NLOHMANN_JSON_PASTE18, \
        NLOHMANN_JSON_PASTE17, \
        NLOHMANN_JSON_PASTE16, \
        NLOHMANN_JSON_PASTE15, \
        NLOHMANN_JSON_PASTE14, \
        NLOHMANN_JSON_PASTE13, \
        NLOHMANN_JSON_PASTE12, \
        NLOHMANN_JSON_PASTE11, \
        NLOHMANN_JSON_PASTE10, \
        NLOHMANN_JSON_PASTE9, \
        NLOHMANN_JSON_PASTE8, \
        NLOHMANN_JSON_PASTE7, \
        NLOHMANN_JSON_PASTE6, \
        NLOHMANN_JSON_PASTE5, \
        NLOHMANN_JSON_PASTE4, \
        NLOHMANN_JSON_PASTE3, \
        NLOHMANN_JSON_PASTE2, \
        NLOHMANN_JSON_PASTE1)(__VA_ARGS__))
#define NLOHMANN_JSON_PASTE2(func, v1) func(v1)
#define NLOHMANN_JSON_PASTE3(func, v1, v2) NLOHMANN_JSON_PASTE2(func, v1) NLOHMANN_JSON_PASTE2(func, v2)
#define NLOHMANN_JSON_PASTE4(func, v1, v2, v3) NLOHMANN_JSON_PASTE2(func, v1) NLOHMANN_JSON_PASTE3(func, v2, v3)
#define NLOHMANN_JSON_PASTE5(func, v1, v2, v3, v4) NLOHMANN_JSON_PASTE2(func, v1) NLOHMANN_JSON_PASTE4(func, v2, v3, v4)
#define NLOHMANN_JSON_PASTE6(func, v1, v2, v3, v4, v5) NLOHMANN_JSON_PASTE2(func, v1) NLOHMANN_JSON_PASTE5(func, v2, v3, v4, v5)
#define NLOHMANN_JSON_PASTE7(func, v1, v2, v3, v4, v5, v6) NLOHMANN_JSON_PASTE2(func, v1) NLOHMANN_JSON_PASTE6(func, v2, v3, v4, v5, v6)
#define NLOHMANN_JSON_PASTE8(func, v1, v2, v3, v4, v5, v6, v7) NLOHMANN_JSON_PASTE2(func, v1) NLOHMANN_JSON_PASTE7(func, v2, v3, v4, v5, v6, v7)
#define NLOHMANN_JSON_PASTE9(func, v1, v2, v3, v4, v5, v6, v7, v8) NLOHMANN_JSON_PASTE2(func, v1) NLOHMANN_JSON_PASTE8(func, v2, v3, v4, v5, v6, v7, v8)
#define NLOHMANN_JSON_PASTE10(func, v1, v2, v3, v4, v5, v6, v7, v8, v9) NLOHMANN_JSON_PASTE2(func, v1) NLOHMANN_JSON_PASTE9(func, v2, v3, v4, v5, v6, v7, v8, v9)
#define NLOHMANN_JSON_PASTE11(func, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10) NLOHMANN_JSON_PASTE2(func, v1) NLOHMANN_JSON_PASTE10(func, v2, v3, v4, v5, v6, v7, v8, v9, v10)
#define NLOHMANN_JSON_PASTE12(func, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11) NLOHMANN_JSON_PASTE2(func, v1) NLOHMANN_JSON_PASTE11(func, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11)
#define NLOHMANN_JSON_PASTE13(func, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12) NLOHMANN_JSON_PASTE2(func, v1) NLOHMANN_JSON_PASTE12(func, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12)
#define NLOHMANN_JSON_PASTE14(func, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13) NLOHMANN_JSON_PASTE2(func, v1) NLOHMANN_JSON_PASTE13(func, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13)
#define NLOHMANN_JSON_PASTE15(func, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14) NLOHMANN_JSON_PASTE2(func, v1) NLOHMANN_JSON_PASTE14(func, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14)
#define NLOHMANN_JSON_PASTE16(func, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15) NLOHMANN_JSON_PASTE2(func, v1) NLOHMANN_JSON_PASTE15(func, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15)
#define NLOHMANN_JSON_PASTE17(func, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16) NLOHMANN_JSON_PASTE2(func, v1) NLOHMANN_JSON_PASTE16(func, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16)
#define NLOHMANN_JSON_PASTE18(func, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17) NLOHMANN_JSON_PASTE2(func, v1) NLOHMANN_JSON_PASTE17(func, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17)
#define NLOHMANN_JSON_PASTE19(func, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18) NLOHMANN_JSON_PASTE2(func, v1) NLOHMANN_JSON_PASTE18(func, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18)
#define NLOHMANN_JSON_PASTE20(func, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19) NLOHMANN_JSON_PASTE2(func, v1) NLOHMANN_JSON_PASTE19(func, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19)
#define NLOHMANN_JSON_PASTE21(func, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20) NLOHMANN_JSON_PASTE2(func, v1) NLOHMANN_JSON_PASTE20(func, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20)
#define NLOHMANN_JSON_PASTE22(func, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21) NLOHMANN_JSON_PASTE2(func, v1) NLOHMANN_JSON_PASTE21(func, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21)
#define NLOHMANN_JSON_PASTE23(func, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22) NLOHMANN_JSON_PASTE2(func, v1) NLOHMANN_JSON_PASTE22(func, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22)
#define NLOHMANN_JSON_PASTE24(func, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23) NLOHMANN_JSON_PASTE2(func, v1) NLOHMANN_JSON_PASTE23(func, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23)
#define NLOHMANN_JSON_PASTE25(func, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24) NLOHMANN_JSON_PASTE2(func, v1) NLOHMANN_JSON_PASTE24(func, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24)
#define NLOHMANN_JSON_PASTE26(func, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25) NLOHMANN_JSON_PASTE2(func, v1) NLOHMANN_JSON_PASTE25(func, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25)
#define NLOHMANN_JSON_PASTE27(func, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26) NLOHMANN_JSON_PASTE2(func, v1) NLOHMANN_JSON_PASTE26(func, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26)
#define NLOHMANN_JSON_PASTE28(func, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27) NLOHMANN_JSON_PASTE2(func, v1) NLOHMANN_JSON_PASTE27(func, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27)
#define NLOHMANN_JSON_PASTE29(func, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28) NLOHMANN_JSON_PASTE2(func, v1) NLOHMANN_JSON_PASTE28(func, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28)
#define NLOHMANN_JSON_PASTE30(func, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29) NLOHMANN_JSON_PASTE2(func, v1) NLOHMANN_JSON_PASTE29(func, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29)
#define NLOHMANN_JSON_PASTE31(func, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30) NLOHMANN_JSON_PASTE2(func, v1) NLOHMANN_JSON_PASTE30(func, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30)
#define NLOHMANN_JSON_PASTE32(func, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31) NLOHMANN_JSON_PASTE2(func, v1) NLOHMANN_JSON_PASTE31(func, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31)
#define NLOHMANN_JSON_PASTE33(func, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32) NLOHMANN_JSON_PASTE2(func, v1) NLOHMANN_JSON_PASTE32(func, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32)
#define NLOHMANN_JSON_PASTE34(func, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33) NLOHMANN_JSON_PASTE2(func, v1) NLOHMANN_JSON_PASTE33(func, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33)
#define NLOHMANN_JSON_PASTE35(func, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34) NLOHMANN_JSON_PASTE2(func, v1) NLOHMANN_JSON_PASTE34(func, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34)
#define NLOHMANN_JSON_PASTE36(func, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35) NLOHMANN_JSON_PASTE2(func, v1) NLOHMANN_JSON_PASTE35(func, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35)
#define NLOHMANN_JSON_PASTE37(func, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36) NLOHMANN_JSON_PASTE2(func, v1) NLOHMANN_JSON_PASTE36(func, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36)
#define NLOHMANN_JSON_PASTE38(func, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37) NLOHMANN_JSON_PASTE2(func, v1) NLOHMANN_JSON_PASTE37(func, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37)
#define NLOHMANN_JSON_PASTE39(func, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38) NLOHMANN_JSON_PASTE2(func, v1) NLOHMANN_JSON_PASTE38(func, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38)
#define NLOHMANN_JSON_PASTE40(func, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39) NLOHMANN_JSON_PASTE2(func, v1) NLOHMANN_JSON_PASTE39(func, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39)
#define NLOHMANN_JSON_PASTE41(func, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40) NLOHMANN_JSON_PASTE2(func, v1) NLOHMANN_JSON_PASTE40(func, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40)
#define NLOHMANN_JSON_PASTE42(func, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41) NLOHMANN_JSON_PASTE2(func, v1) NLOHMANN_JSON_PASTE41(func, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41)
#define NLOHMANN_JSON_PASTE43(func, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42) NLOHMANN_JSON_PASTE2(func, v1) NLOHMANN_JSON_PASTE42(func, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42)
#define NLOHMANN_JSON_PASTE44(func, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43) NLOHMANN_JSON_PASTE2(func, v1) NLOHMANN_JSON_PASTE43(func, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43)
#define NLOHMANN_JSON_PASTE45(func, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43, v44) NLOHMANN_JSON_PASTE2(func, v1) NLOHMANN_JSON_PASTE44(func, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43, v44)
#define NLOHMANN_JSON_PASTE46(func, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43, v44, v45) NLOHMANN_JSON_PASTE2(func, v1) NLOHMANN_JSON_PASTE45(func, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43, v44, v45)
#define NLOHMANN_JSON_PASTE47(func, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43, v44, v45, v46) NLOHMANN_JSON_PASTE2(func, v1) NLOHMANN_JSON_PASTE46(func, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43, v44, v45, v46)
#define NLOHMANN_JSON_PASTE48(func, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43, v44, v45, v46, v47) NLOHMANN_JSON_PASTE2(func, v1) NLOHMANN_JSON_PASTE47(func, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43, v44, v45, v46, v47)
#define NLOHMANN_JSON_PASTE49(func, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43, v44, v45, v46, v47, v48) NLOHMANN_JSON_PASTE2(func, v1) NLOHMANN_JSON_PASTE48(func, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43, v44, v45, v46, v47, v48)
#define NLOHMANN_JSON_PASTE50(func, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43, v44, v45, v46, v47, v48, v49) NLOHMANN_JSON_PASTE2(func, v1) NLOHMANN_JSON_PASTE49(func, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43, v44, v45, v46, v47, v48, v49)
#define NLOHMANN_JSON_PASTE51(func, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43, v44, v45, v46, v47, v48, v49, v50) NLOHMANN_JSON_PASTE2(func, v1) NLOHMANN_JSON_PASTE50(func, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43, v44, v45, v46, v47, v48, v49, v50)
#define NLOHMANN_JSON_PASTE52(func, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43, v44, v45, v46, v47, v48, v49, v50, v51) NLOHMANN_JSON_PASTE2(func, v1) NLOHMANN_JSON_PASTE51(func, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43, v44, v45, v46, v47, v48, v49, v50, v51)
#define NLOHMANN_JSON_PASTE53(func, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43, v44, v45, v46, v47, v48, v49, v50, v51, v52) NLOHMANN_JSON_PASTE2(func, v1) NLOHMANN_JSON_PASTE52(func, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43, v44, v45, v46, v47, v48, v49, v50, v51, v52)
#define NLOHMANN_JSON_PASTE54(func, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43, v44, v45, v46, v47, v48, v49, v50, v51, v52, v53) NLOHMANN_JSON_PASTE2(func, v1) NLOHMANN_JSON_PASTE53(func, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43, v44, v45, v46, v47, v48, v49, v50, v51, v52, v53)
#define NLOHMANN_JSON_PASTE55(func, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43, v44, v45, v46, v47, v48, v49, v50, v51, v52, v53, v54) NLOHMANN_JSON_PASTE2(func, v1) NLOHMANN_JSON_PASTE54(func, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43, v44, v45, v46, v47, v48, v49, v50, v51, v52, v53, v54)
#define NLOHMANN_JSON_PASTE56(func, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43, v44, v45, v46, v47, v48, v49, v50, v51, v52, v53, v54, v55) NLOHMANN_JSON_PASTE2(func, v1) NLOHMANN_JSON_PASTE55(func, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43, v44, v45, v46, v47, v48, v49, v50, v51, v52, v53, v54, v55)
#define NLOHMANN_JSON_PASTE57(func, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43, v44, v45, v46, v47, v48, v49, v50, v51, v52, v53, v54, v55, v56) NLOHMANN_JSON_PASTE2(func, v1) NLOHMANN_JSON_PASTE56(func, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43, v44, v45, v46, v47, v48, v49, v50, v51, v52, v53, v54, v55, v56)
#define NLOHMANN_JSON_PASTE58(func, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43, v44, v45, v46, v47, v48, v49, v50, v51, v52, v53, v54, v55, v56, v57) NLOHMANN_JSON_PASTE2(func, v1) NLOHMANN_JSON_PASTE57(func, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43, v44, v45, v46, v47, v48, v49, v50, v51, v52, v53, v54, v55, v56, v57)
#define NLOHMANN_JSON_PASTE59(func, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43, v44, v45, v46, v47, v48, v49, v50, v51, v52, v53, v54, v55, v56, v57, v58) NLOHMANN_JSON_PASTE2(func, v1) NLOHMANN_JSON_PASTE58(func, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43, v44, v45, v46, v47, v48, v49, v50, v51, v52, v53, v54, v55, v56, v57, v58)
#define NLOHMANN_JSON_PASTE60(func, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43, v44, v45, v46, v47, v48, v49, v50, v51, v52, v53, v54, v55, v56, v57, v58, v59) NLOHMANN_JSON_PASTE2(func, v1) NLOHMANN_JSON_PASTE59(func, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43, v44, v45, v46, v47, v48, v49, v50, v51, v52, v53, v54, v55, v56, v57, v58, v59)
#define NLOHMANN_JSON_PASTE61(func, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43, v44, v45, v46, v47, v48, v49, v50, v51, v52, v53, v54, v55, v56, v57, v58, v59, v60) NLOHMANN_JSON_PASTE2(func, v1) NLOHMANN_JSON_PASTE60(func, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43, v44, v45, v46, v47, v48, v49, v50, v51, v52, v53, v54, v55, v56, v57, v58, v59, v60)
#define NLOHMANN_JSON_PASTE62(func, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43, v44, v45, v46, v47, v48, v49, v50, v51, v52, v53, v54, v55, v56, v57, v58, v59, v60, v61) NLOHMANN_JSON_PASTE2(func, v1) NLOHMANN_JSON_PASTE61(func, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43, v44, v45, v46, v47, v48, v49, v50, v51, v52, v53, v54, v55, v56, v57, v58, v59, v60, v61)
#define NLOHMANN_JSON_PASTE63(func, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43, v44, v45, v46, v47, v48, v49, v50, v51, v52, v53, v54, v55, v56, v57, v58, v59, v60, v61, v62) NLOHMANN_JSON_PASTE2(func, v1) NLOHMANN_JSON_PASTE62(func, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43, v44, v45, v46, v47, v48, v49, v50, v51, v52, v53, v54, v55, v56, v57, v58, v59, v60, v61, v62)
#define NLOHMANN_JSON_PASTE64(func, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43, v44, v45, v46, v47, v48, v49, v50, v51, v52, v53, v54, v55, v56, v57, v58, v59, v60, v61, v62, v63) NLOHMANN_JSON_PASTE2(func, v1) NLOHMANN_JSON_PASTE63(func, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43, v44, v45, v46, v47, v48, v49, v50, v51, v52, v53, v54, v55, v56, v57, v58, v59, v60, v61, v62, v63)

#define NLOHMANN_JSON_TO(v1) nlohmann_json_j[#v1] = nlohmann_json_t.v1;
#define NLOHMANN_JSON_FROM(v1) nlohmann_json_j.at(#v1).get_to(nlohmann_json_t.v1);
#define NLOHMANN_JSON_FROM_WITH_DEFAULT(v1) nlohmann_json_t.v1 = !nlohmann_json_j.is_null() ? nlohmann_json_j.value(#v1, nlohmann_json_default_obj.v1) : nlohmann_json_default_obj.v1;

/*!
@brief macro
@def NLOHMANN_DEFINE_TYPE_INTRUSIVE
@since version 3.9.0
@sa https://json.nlohmann.me/api/macros/nlohmann_define_type_intrusive/
*/
#define NLOHMANN_DEFINE_TYPE_INTRUSIVE(Type, ...)  \
    template<typename BasicJsonType, nlohmann::detail::enable_if_t<nlohmann::detail::is_basic_json<BasicJsonType>::value, int> = 0> \
    friend void to_json(BasicJsonType& nlohmann_json_j, const Type& nlohmann_json_t) { NLOHMANN_JSON_EXPAND(NLOHMANN_JSON_PASTE(NLOHMANN_JSON_TO, __VA_ARGS__)) } \
    template<typename BasicJsonType, nlohmann::detail::enable_if_t<nlohmann::detail::is_basic_json<BasicJsonType>::value, int> = 0> \
    friend void from_json(const BasicJsonType& nlohmann_json_j, Type& nlohmann_json_t) { NLOHMANN_JSON_EXPAND(NLOHMANN_JSON_PASTE(NLOHMANN_JSON_FROM, __VA_ARGS__)) }

/*!
@brief macro
@def NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT
@since version 3.11.0
@sa https://json.nlohmann.me/api/macros/nlohmann_define_type_intrusive/
*/
#define NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(Type, ...)  \
    template<typename BasicJsonType, nlohmann::detail::enable_if_t<nlohmann::detail::is_basic_json<BasicJsonType>::value, int> = 0> \
    friend void to_json(BasicJsonType& nlohmann_json_j, const Type& nlohmann_json_t) { NLOHMANN_JSON_EXPAND(NLOHMANN_JSON_PASTE(NLOHMANN_JSON_TO, __VA_ARGS__)) } \
    template<typename BasicJsonType, nlohmann::detail::enable_if_t<nlohmann::detail::is_basic_json<BasicJsonType>::value, int> = 0> \
    friend void from_json(const BasicJsonType& nlohmann_json_j, Type& nlohmann_json_t) { const Type nlohmann_json_default_obj{}; NLOHMANN_JSON_EXPAND(NLOHMANN_JSON_PASTE(NLOHMANN_JSON_FROM_WITH_DEFAULT, __VA_ARGS__)) }

/*!
@brief macro
@def NLOHMANN_DEFINE_TYPE_INTRUSIVE_ONLY_SERIALIZE
@since version 3.11.3
@sa https://json.nlohmann.me/api/macros/nlohmann_define_type_intrusive/
*/
#define NLOHMANN_DEFINE_TYPE_INTRUSIVE_ONLY_SERIALIZE(Type, ...)  \
    template<typename BasicJsonType, nlohmann::detail::enable_if_t<nlohmann::detail::is_basic_json<BasicJsonType>::value, int> = 0> \
    friend void to_json(BasicJsonType& nlohmann_json_j, const Type& nlohmann_json_t) { NLOHMANN_JSON_EXPAND(NLOHMANN_JSON_PASTE(NLOHMANN_JSON_TO, __VA_ARGS__)) }

/*!
@brief macro
@def NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE
@since version 3.9.0
@sa https://json.nlohmann.me/api/macros/nlohmann_define_type_non_intrusive/
*/
#define NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Type, ...)  \
    template<typename BasicJsonType, nlohmann::detail::enable_if_t<nlohmann::detail::is_basic_json<BasicJsonType>::value, int> = 0> \
    void to_json(BasicJsonType& nlohmann_json_j, const Type& nlohmann_json_t) { NLOHMANN_JSON_EXPAND(NLOHMANN_JSON_PASTE(NLOHMANN_JSON_TO, __VA_ARGS__)) } \
    template<typename BasicJsonType, nlohmann::detail::enable_if_t<nlohmann::detail::is_basic_json<BasicJsonType>::value, int> = 0> \
    void from_json(const BasicJsonType& nlohmann_json_j, Type& nlohmann_json_t) { NLOHMANN_JSON_EXPAND(NLOHMANN_JSON_PASTE(NLOHMANN_JSON_FROM, __VA_ARGS__)) }

/*!
@brief macro
@def NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT
@since version 3.11.0
@sa https://json.nlohmann.me/api/macros/nlohmann_define_type_non_intrusive/
*/
#define NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(Type, ...)  \
    template<typename BasicJsonType, nlohmann::detail::enable_if_t<nlohmann::detail::is_basic_json<BasicJsonType>::value, int> = 0> \
    void to_json(BasicJsonType& nlohmann_json_j, const Type& nlohmann_json_t) { NLOHMANN_JSON_EXPAND(NLOHMANN_JSON_PASTE(NLOHMANN_JSON_TO, __VA_ARGS__)) } \
    template<typename BasicJsonType, nlohmann::detail::enable_if_t<nlohmann::detail::is_basic_json<BasicJsonType>::value, int> = 0> \
    void from_json(const BasicJsonType& nlohmann_json_j, Type& nlohmann_json_t) { const Type nlohmann_json_default_obj{}; NLOHMANN_JSON_EXPAND(NLOHMANN_JSON_PASTE(NLOHMANN_JSON_FROM_WITH_DEFAULT, __VA_ARGS__)) }

/*!
@brief macro
@def NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_ONLY_SERIALIZE
@since version 3.11.3
@sa https://json.nlohmann.me/api/macros/nlohmann_define_type_non_intrusive/
*/
#define NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_ONLY_SERIALIZE(Type, ...)  \
    template<typename BasicJsonType, nlohmann::detail::enable_if_t<nlohmann::detail::is_basic_json<BasicJsonType>::value, int> = 0> \
    void to_json(BasicJsonType& nlohmann_json_j, const Type& nlohmann_json_t) { NLOHMANN_JSON_EXPAND(NLOHMANN_JSON_PASTE(NLOHMANN_JSON_TO, __VA_ARGS__)) }

/*!
@brief macro
@def NLOHMANN_DEFINE_DERIVED_TYPE_INTRUSIVE
@since version 3.12.0
@sa https://json.nlohmann.me/api/macros/nlohmann_define_derived_type/
*/
#define NLOHMANN_DEFINE_DERIVED_TYPE_INTRUSIVE(Type, BaseType, ...)  \
    template<typename BasicJsonType, nlohmann::detail::enable_if_t<nlohmann::detail::is_basic_json<BasicJsonType>::value, int> = 0> \
    friend void to_json(BasicJsonType& nlohmann_json_j, const Type& nlohmann_json_t) { nlohmann::to_json(nlohmann_json_j, static_cast<const BaseType &>(nlohmann_json_t)); NLOHMANN_JSON_EXPAND(NLOHMANN_JSON_PASTE(NLOHMANN_JSON_TO, __VA_ARGS__)) } \
    template<typename BasicJsonType, nlohmann::detail::enable_if_t<nlohmann::detail::is_basic_json<BasicJsonType>::value, int> = 0> \
    friend void from_json(const BasicJsonType& nlohmann_json_j, Type& nlohmann_json_t) { nlohmann::from_json(nlohmann_json_j, static_cast<BaseType&>(nlohmann_json_t)); NLOHMANN_JSON_EXPAND(NLOHMANN_JSON_PASTE(NLOHMANN_JSON_FROM, __VA_ARGS__)) }

/*!
@brief macro
@def NLOHMANN_DEFINE_DERIVED_TYPE_INTRUSIVE_WITH_DEFAULT
@since version 3.12.0
@sa https://json.nlohmann.me/api/macros/nlohmann_define_derived_type/
*/
#define NLOHMANN_DEFINE_DERIVED_TYPE_INTRUSIVE_WITH_DEFAULT(Type, BaseType, ...)  \
    template<typename BasicJsonType, nlohmann::detail::enable_if_t<nlohmann::detail::is_basic_json<BasicJsonType>::value, int> = 0> \
    friend void to_json(BasicJsonType& nlohmann_json_j, const Type& nlohmann_json_t) { nlohmann::to_json(nlohmann_json_j, static_cast<const BaseType&>(nlohmann_json_t)); NLOHMANN_JSON_EXPAND(NLOHMANN_JSON_PASTE(NLOHMANN_JSON_TO, __VA_ARGS__)) } \
    template<typename BasicJsonType, nlohmann::detail::enable_if_t<nlohmann::detail::is_basic_json<BasicJsonType>::value, int> = 0> \
    friend void from_json(const BasicJsonType& nlohmann_json_j, Type& nlohmann_json_t) { nlohmann::from_json(nlohmann_json_j, static_cast<BaseType&>(nlohmann_json_t)); const Type nlohmann_json_default_obj{}; NLOHMANN_JSON_EXPAND(NLOHMANN_JSON_PASTE(NLOHMANN_JSON_FROM_WITH_DEFAULT, __VA_ARGS__)) }

/*!
@brief macro
@def NLOHMANN_DEFINE_DERIVED_TYPE_INTRUSIVE_ONLY_SERIALIZE
@since version 3.12.0
@sa https://json.nlohmann.me/api/macros/nlohmann_define_derived_type/
*/
#define NLOHMANN_DEFINE_DERIVED_TYPE_INTRUSIVE_ONLY_SERIALIZE(Type, BaseType, ...)  \
    template<typename BasicJsonType, nlohmann::detail::enable_if_t<nlohmann::detail::is_basic_json<BasicJsonType>::value, int> = 0> \
    friend void to_json(BasicJsonType& nlohmann_json_j, const Type& nlohmann_json_t) { nlohmann::to_json(nlohmann_json_j, static_cast<const BaseType &>(nlohmann_json_t)); NLOHMANN_JSON_EXPAND(NLOHMANN_JSON_PASTE(NLOHMANN_JSON_TO, __VA_ARGS__)) }

/*!
@brief macro
@def NLOHMANN_DEFINE_DERIVED_TYPE_NON_INTRUSIVE
@since version 3.12.0
@sa https://json.nlohmann.me/api/macros/nlohmann_define_derived_type/
*/
#define NLOHMANN_DEFINE_DERIVED_TYPE_NON_INTRUSIVE(Type, BaseType, ...)  \
    template<typename BasicJsonType, nlohmann::detail::enable_if_t<nlohmann::detail::is_basic_json<BasicJsonType>::value, int> = 0> \
    void to_json(BasicJsonType& nlohmann_json_j, const Type& nlohmann_json_t) { nlohmann::to_json(nlohmann_json_j, static_cast<const BaseType &>(nlohmann_json_t)); NLOHMANN_JSON_EXPAND(NLOHMANN_JSON_PASTE(NLOHMANN_JSON_TO, __VA_ARGS__)) } \
    template<typename BasicJsonType, nlohmann::detail::enable_if_t<nlohmann::detail::is_basic_json<BasicJsonType>::value, int> = 0> \
    void from_json(const BasicJsonType& nlohmann_json_j, Type& nlohmann_json_t) { nlohmann::from_json(nlohmann_json_j, static_cast<BaseType&>(nlohmann_json_t)); NLOHMANN_JSON_EXPAND(NLOHMANN_JSON_PASTE(NLOHMANN_JSON_FROM, __VA_ARGS__)) }

/*!
@brief macro
@def NLOHMANN_DEFINE_DERIVED_TYPE_NON_INTRUSIVE_WITH_DEFAULT
@since version 3.12.0
@sa https://json.nlohmann.me/api/macros/nlohmann_define_derived_type/
*/
#define NLOHMANN_DEFINE_DERIVED_TYPE_NON_INTRUSIVE_WITH_DEFAULT(Type, BaseType, ...)  \
    template<typename BasicJsonType, nlohmann::detail::enable_if_t<nlohmann::detail::is_basic_json<BasicJsonType>::value, int> = 0> \
    void to_json(BasicJsonType& nlohmann_json_j, const Type& nlohmann_json_t) { nlohmann::to_json(nlohmann_json_j, static_cast<const BaseType &>(nlohmann_json_t)); NLOHMANN_JSON_EXPAND(NLOHMANN_JSON_PASTE(NLOHMANN_JSON_TO, __VA_ARGS__)) } \
    template<typename BasicJsonType, nlohmann::detail::enable_if_t<nlohmann::detail::is_basic_json<BasicJsonType>::value, int> = 0> \
    void from_json(const BasicJsonType& nlohmann_json_j, Type& nlohmann_json_t) { nlohmann::from_json(nlohmann_json_j, static_cast<BaseType&>(nlohmann_json_t)); const Type nlohmann_json_default_obj{}; NLOHMANN_JSON_EXPAND(NLOHMANN_JSON_PASTE(NLOHMANN_JSON_FROM_WITH_DEFAULT, __VA_ARGS__)) }

/*!
@brief macro
@def NLOHMANN_DEFINE_DERIVED_TYPE_NON_INTRUSIVE_ONLY_SERIALIZE
@since version 3.12.0
@sa https://json.nlohmann.me/api/macros/nlohmann_define_derived_type/
*/
#define NLOHMANN_DEFINE_DERIVED_TYPE_NON_INTRUSIVE_ONLY_SERIALIZE(Type, BaseType, ...)  \
    template<typename BasicJsonType, nlohmann::detail::enable_if_t<nlohmann::detail::is_basic_json<BasicJsonType>::value, int> = 0> \
    void to_json(BasicJsonType& nlohmann_json_j, const Type& nlohmann_json_t) { nlohmann::to_json(nlohmann_json_j, static_cast<const BaseType &>(nlohmann_json_t)); NLOHMANN_JSON_EXPAND(NLOHMANN_JSON_PASTE(NLOHMANN_JSON_TO, __VA_ARGS__)) }

// inspired from https://stackoverflow.com/a/26745591
// allows calling any std function as if (e.g., with begin):
// using std::begin; begin(x);
//
// it allows using the detected idiom to retrieve the return type
// of such an expression
#define NLOHMANN_CAN_CALL_STD_FUNC_IMPL(std_name)                                 \
    namespace detail {                                                            \
    using std::std_name;                                                          \
    \
    template<typename... T>                                                       \
    using result_of_##std_name = decltype(std_name(std::declval<T>()...));        \
    }                                                                             \
    \
    namespace detail2 {                                                           \
    struct std_name##_tag                                                         \
    {                                                                             \
    };                                                                            \
    \
    template<typename... T>                                                       \
    std_name##_tag std_name(T&&...);                                              \
    \
    template<typename... T>                                                       \
    using result_of_##std_name = decltype(std_name(std::declval<T>()...));        \
    \
    template<typename... T>                                                       \
    struct would_call_std_##std_name                                              \
    {                                                                             \
        static constexpr auto const value = ::nlohmann::detail::                  \
                                            is_detected_exact<std_name##_tag, result_of_##std_name, T...>::value; \
    };                                                                            \
    } /* namespace detail2 */ \
    \
    template<typename... T>                                                       \
    struct would_call_std_##std_name : detail2::would_call_std_##std_name<T...>   \
    {                                                                             \
    }

#ifndef JSON_USE_IMPLICIT_CONVERSIONS
    #define JSON_USE_IMPLICIT_CONVERSIONS 1
#endif

#if JSON_USE_IMPLICIT_CONVERSIONS
    #define JSON_EXPLICIT
#else
    #define JSON_EXPLICIT explicit
#endif

#ifndef JSON_DISABLE_ENUM_SERIALIZATION
    #define JSON_DISABLE_ENUM_SERIALIZATION 0
#endif

#ifndef JSON_USE_GLOBAL_UDLS
    #define JSON_USE_GLOBAL_UDLS 1
#endif

#if JSON_HAS_THREE_WAY_COMPARISON
    #include <compare> // partial_ordering
#endif

NLOHMANN_JSON_NAMESPACE_BEGIN
namespace detail
{

///////////////////////////
// JSON type enumeration //
///////////////////////////

/*!
@brief the JSON type enumeration

This enumeration collects the different JSON types. It is internally used to
distinguish the stored values, and the functions @ref basic_json::is_null(),
@ref basic_json::is_object(), @ref basic_json::is_array(),
@ref basic_json::is_string(), @ref basic_json::is_boolean(),
@ref basic_json::is_number() (with @ref basic_json::is_number_integer(),
@ref basic_json::is_number_unsigned(), and @ref basic_json::is_number_float()),
@ref basic_json::is_discarded(), @ref basic_json::is_primitive(), and
@ref basic_json::is_structured() rely on it.

@note There are three enumeration entries (number_integer, number_unsigned, and
number_float), because the library distinguishes these three types for numbers:
@ref basic_json::number_unsigned_t is used for unsigned integers,
@ref basic_json::number_integer_t is used for signed integers, and
@ref basic_json::number_float_t is used for floating-point numbers or to
approximate integers which do not fit in the limits of their respective type.

@sa see @ref basic_json::basic_json(const value_t value_type) -- create a JSON
value with the default value for a given type

@since version 1.0.0
*/
enum class value_t : std::uint8_t
{
    null,             ///< null value
    object,           ///< object (unordered set of name/value pairs)
    array,            ///< array (ordered collection of values)
    string,           ///< string value
    boolean,          ///< boolean value
    number_integer,   ///< number value (signed integer)
    number_unsigned,  ///< number value (unsigned integer)
    number_float,     ///< number value (floating-point)
    binary,           ///< binary array (ordered collection of bytes)
    discarded         ///< discarded by the parser callback function
};

/*!
@brief comparison operator for JSON types

Returns an ordering that is similar to Python:
- order: null < boolean < number < object < array < string < binary
- furthermore, each type is not smaller than itself
- discarded values are not comparable
- binary is represented as a b"" string in python and directly comparable to a
  string; however, making a binary array directly comparable with a string would
  be surprising behavior in a JSON file.

@since version 1.0.0
*/
#if JSON_HAS_THREE_WAY_COMPARISON
    inline std::partial_ordering operator<=>(const value_t lhs, const value_t rhs) noexcept // *NOPAD*
#else
    inline bool operator<(const value_t lhs, const value_t rhs) noexcept
#endif
{
    static constexpr std::array<std::uint8_t, 9> order = {{
            0 /* null */, 3 /* object */, 4 /* array */, 5 /* string */,
            1 /* boolean */, 2 /* integer */, 2 /* unsigned */, 2 /* float */,
            6 /* binary */
        }
    };

    const auto l_index = static_cast<std::size_t>(lhs);
    const auto r_index = static_cast<std::size_t>(rhs);
#if JSON_HAS_THREE_WAY_COMPARISON
    if (l_index < order.size() && r_index < order.size())
    {
        return order[l_index] <=> order[r_index]; // *NOPAD*
    }
    return std::partial_ordering::unordered;
#else
    return l_index < order.size() && r_index < order.size() && order[l_index] < order[r_index];
#endif
}

// GCC selects the built-in operator< over an operator rewritten from
// a user-defined spaceship operator
// Clang, MSVC, and ICC select the rewritten candidate
// (see GCC bug https://gcc.gnu.org/bugzilla/show_bug.cgi?id=105200)
#if JSON_HAS_THREE_WAY_COMPARISON && defined(__GNUC__)
inline bool operator<(const value_t lhs, const value_t rhs) noexcept
{
    return std::is_lt(lhs <=> rhs); // *NOPAD*
}
#endif

}  // namespace detail
NLOHMANN_JSON_NAMESPACE_END

// #include <nlohmann/detail/string_escape.hpp>
//     __ _____ _____ _____
//  __|  |   __|     |   | |  JSON for Modern C++
// |  |  |__   |  |  | | | |  version 3.12.0
// |_____|_____|_____|_|___|  https://github.com/nlohmann/json
//
// SPDX-FileCopyrightText: 2013 - 2025 Niels Lohmann <https://nlohmann.me>
// SPDX-License-Identifier: MIT



// #include <nlohmann/detail/abi_macros.hpp>


NLOHMANN_JSON_NAMESPACE_BEGIN
namespace detail
{

/*!
@brief replace all occurrences of a substring by another string

@param[in,out] s  the string to manipulate; changed so that all
               occurrences of @a f are replaced with @a t
@param[in]     f  the substring to replace with @a t
@param[in]     t  the string to replace @a f

@pre The search string @a f must not be empty. **This precondition is
enforced with an assertion.**

@since version 2.0.0
*/
template<typename StringType>
inline void replace_substring(StringType& s, const StringType& f,
                              const StringType& t)
{
    JSON_ASSERT(!f.empty());
    for (auto pos = s.find(f);                // find first occurrence of f
            pos != StringType::npos;          // make sure f was found
            s.replace(pos, f.size(), t),      // replace with t, and
            pos = s.find(f, pos + t.size()))  // find next occurrence of f
    {}
}

/*!
 * @brief string escaping as described in RFC 6901 (Sect. 4)
 * @param[in] s string to escape
 * @return    escaped string
 *
 * Note the order of escaping "~" to "~0" and "/" to "~1" is important.
 */
template<typename StringType>
inline StringType escape(StringType s)
{
    replace_substring(s, StringType{"~"}, StringType{"~0"});
    replace_substring(s, StringType{"/"}, StringType{"~1"});
    return s;
}

/*!
 * @brief string unescaping as described in RFC 6901 (Sect. 4)
 * @param[in] s string to unescape
 * @return    unescaped string
 *
 * Note the order of escaping "~1" to "/" and "~0" to "~" is important.
 */
template<typename StringType>
static void unescape(StringType& s)
{
    replace_substring(s, StringType{"~1"}, StringType{"/"});
    replace_substring(s, StringType{"~0"}, StringType{"~"});
}

}  // namespace detail
NLOHMANN_JSON_NAMESPACE_END

// #include <nlohmann/detail/input/position_t.hpp>
//     __ _____ _____ _____
//  __|  |   __|     |   | |  JSON for Modern C++
// |  |  |__   |  |  | | | |  version 3.12.0
// |_____|_____|_____|_|___|  https://github.com/nlohmann/json
//
// SPDX-FileCopyrightText: 2013 - 2025 Niels Lohmann <https://nlohmann.me>
// SPDX-License-Identifier: MIT



#include <cstddef> // size_t

// #include <nlohmann/detail/abi_macros.hpp>


NLOHMANN_JSON_NAMESPACE_BEGIN
namespace detail
{

/// struct to capture the start position of the current token
struct position_t
{
    /// the total number of characters read
    std::size_t chars_read_total = 0;
    /// the number of characters read in the current line
    std::size_t chars_read_current_line = 0;
    /// the number of lines read
    std::size_t lines_read = 0;

    /// conversion to size_t to preserve SAX interface
    constexpr operator size_t() const
    {
        return chars_read_total;
    }
};

}  // namespace detail
NLOHMANN_JSON_NAMESPACE_END

// #include <nlohmann/detail/macro_scope.hpp>

// #include <nlohmann/detail/meta/cpp_future.hpp>
//     __ _____ _____ _____
//  __|  |   __|     |   | |  JSON for Modern C++
// |  |  |__   |  |  | | | |  version 3.12.0
// |_____|_____|_____|_|___|  https://github.com/nlohmann/json
//
// SPDX-FileCopyrightText: 2013 - 2025 Niels Lohmann <https://nlohmann.me>
// SPDX-FileCopyrightText: 2018 The Abseil Authors
// SPDX-License-Identifier: MIT



#include <array> // array
#include <cstddef> // size_t
#include <type_traits> // conditional, enable_if, false_type, integral_constant, is_constructible, is_integral, is_same, remove_cv, remove_reference, true_type
#include <utility> // index_sequence, make_index_sequence, index_sequence_for

// #include <nlohmann/detail/macro_scope.hpp>


NLOHMANN_JSON_NAMESPACE_BEGIN
namespace detail
{

template<typename T>
using uncvref_t = typename std::remove_cv<typename std::remove_reference<T>::type>::type;

#ifdef JSON_HAS_CPP_14

// the following utilities are natively available in C++14
using std::enable_if_t;
using std::index_sequence;
using std::make_index_sequence;
using std::index_sequence_for;

#else

// alias templates to reduce boilerplate
template<bool B, typename T = void>
using enable_if_t = typename std::enable_if<B, T>::type;

// The following code is taken from https://github.com/abseil/abseil-cpp/blob/10cb35e459f5ecca5b2ff107635da0bfa41011b4/absl/utility/utility.h
// which is part of Google Abseil (https://github.com/abseil/abseil-cpp), licensed under the Apache License 2.0.

//// START OF CODE FROM GOOGLE ABSEIL

// integer_sequence
//
// Class template representing a compile-time integer sequence. An instantiation
// of `integer_sequence<T, Ints...>` has a sequence of integers encoded in its
// type through its template arguments (which is a common need when
// working with C++11 variadic templates). `absl::integer_sequence` is designed
// to be a drop-in replacement for C++14's `std::integer_sequence`.
//
// Example:
//
//   template< class T, T... Ints >
//   void user_function(integer_sequence<T, Ints...>);
//
//   int main()
//   {
//     // user_function's `T` will be deduced to `int` and `Ints...`
//     // will be deduced to `0, 1, 2, 3, 4`.
//     user_function(make_integer_sequence<int, 5>());
//   }
template <typename T, T... Ints>
struct integer_sequence
{
    using value_type = T;
    static constexpr std::size_t size() noexcept
    {
        return sizeof...(Ints);
    }
};

// index_sequence
//
// A helper template for an `integer_sequence` of `size_t`,
// `absl::index_sequence` is designed to be a drop-in replacement for C++14's
// `std::index_sequence`.
template <size_t... Ints>
using index_sequence = integer_sequence<size_t, Ints...>;

namespace utility_internal
{

template <typename Seq, size_t SeqSize, size_t Rem>
struct Extend;

// Note that SeqSize == sizeof...(Ints). It's passed explicitly for efficiency.
template <typename T, T... Ints, size_t SeqSize>
struct Extend<integer_sequence<T, Ints...>, SeqSize, 0>
{
    using type = integer_sequence < T, Ints..., (Ints + SeqSize)... >;
};

template <typename T, T... Ints, size_t SeqSize>
struct Extend<integer_sequence<T, Ints...>, SeqSize, 1>
{
    using type = integer_sequence < T, Ints..., (Ints + SeqSize)..., 2 * SeqSize >;
};

// Recursion helper for 'make_integer_sequence<T, N>'.
// 'Gen<T, N>::type' is an alias for 'integer_sequence<T, 0, 1, ... N-1>'.
template <typename T, size_t N>
struct Gen
{
    using type =
        typename Extend < typename Gen < T, N / 2 >::type, N / 2, N % 2 >::type;
};

template <typename T>
struct Gen<T, 0>
{
    using type = integer_sequence<T>;
};

}  // namespace utility_internal

// Compile-time sequences of integers

// make_integer_sequence
//
// This template alias is equivalent to
// `integer_sequence<int, 0, 1, ..., N-1>`, and is designed to be a drop-in
// replacement for C++14's `std::make_integer_sequence`.
template <typename T, T N>
using make_integer_sequence = typename utility_internal::Gen<T, N>::type;

// make_index_sequence
//
// This template alias is equivalent to `index_sequence<0, 1, ..., N-1>`,
// and is designed to be a drop-in replacement for C++14's
// `std::make_index_sequence`.
template <size_t N>
using make_index_sequence = make_integer_sequence<size_t, N>;

// index_sequence_for
//
// Converts a typename pack into an index sequence of the same length, and
// is designed to be a drop-in replacement for C++14's
// `std::index_sequence_for()`
template <typename... Ts>
using index_sequence_for = make_index_sequence<sizeof...(Ts)>;

//// END OF CODE FROM GOOGLE ABSEIL

#endif

// dispatch utility (taken from ranges-v3)
template<unsigned N> struct priority_tag : priority_tag < N - 1 > {};
template<> struct priority_tag<0> {};

// taken from ranges-v3
template<typename T>
struct static_const
{
    static JSON_INLINE_VARIABLE constexpr T value{};
};

#ifndef JSON_HAS_CPP_17
    template<typename T>
    constexpr T static_const<T>::value;
#endif

template<typename T, typename... Args>
constexpr std::array<T, sizeof...(Args)> make_array(Args&& ... args)
{
    return std::array<T, sizeof...(Args)> {{static_cast<T>(std::forward<Args>(args))...}};
}

}  // namespace detail
NLOHMANN_JSON_NAMESPACE_END

// #include <nlohmann/detail/meta/type_traits.hpp>
//     __ _____ _____ _____
//  __|  |   __|     |   | |  JSON for Modern C++
// |  |  |__   |  |  | | | |  version 3.12.0
// |_____|_____|_____|_|___|  https://github.com/nlohmann/json
//
// SPDX-FileCopyrightText: 2013 - 2025 Niels Lohmann <https://nlohmann.me>
// SPDX-License-Identifier: MIT



#include <limits> // numeric_limits
#include <string> // char_traits
#include <tuple> // tuple
#include <type_traits> // false_type, is_constructible, is_integral, is_same, true_type
#include <utility> // declval

// #include <nlohmann/detail/iterators/iterator_traits.hpp>
//     __ _____ _____ _____
//  __|  |   __|     |   | |  JSON for Modern C++
// |  |  |__   |  |  | | | |  version 3.12.0
// |_____|_____|_____|_|___|  https://github.com/nlohmann/json
//
// SPDX-FileCopyrightText: 2013 - 2025 Niels Lohmann <https://nlohmann.me>
// SPDX-License-Identifier: MIT



#include <iterator> // random_access_iterator_tag

// #include <nlohmann/detail/abi_macros.hpp>

// #include <nlohmann/detail/meta/void_t.hpp>

// #include <nlohmann/detail/meta/cpp_future.hpp>


NLOHMANN_JSON_NAMESPACE_BEGIN
namespace detail
{

template<typename It, typename = void>
struct iterator_types {};

template<typename It>
struct iterator_types <
    It,
    void_t<typename It::difference_type, typename It::value_type, typename It::pointer,
    typename It::reference, typename It::iterator_category >>
{
    using difference_type = typename It::difference_type;
    using value_type = typename It::value_type;
    using pointer = typename It::pointer;
    using reference = typename It::reference;
    using iterator_category = typename It::iterator_category;
};

// This is required as some compilers implement std::iterator_traits in a way that
// doesn't work with SFINAE. See https://github.com/nlohmann/json/issues/1341.
template<typename T, typename = void>
struct iterator_traits
{
};

template<typename T>
struct iterator_traits < T, enable_if_t < !std::is_pointer<T>::value >>
    : iterator_types<T>
{
};

template<typename T>
struct iterator_traits<T*, enable_if_t<std::is_object<T>::value>>
{
    using iterator_category = std::random_access_iterator_tag;
    using value_type = T;
    using difference_type = ptrdiff_t;
    using pointer = T*;
    using reference = T&;
};

}  // namespace detail
NLOHMANN_JSON_NAMESPACE_END

// #include <nlohmann/detail/macro_scope.hpp>

// #include <nlohmann/detail/meta/call_std/begin.hpp>
//     __ _____ _____ _____
//  __|  |   __|     |   | |  JSON for Modern C++
// |  |  |__   |  |  | | | |  version 3.12.0
// |_____|_____|_____|_|___|  https://github.com/nlohmann/json
//
// SPDX-FileCopyrightText: 2013 - 2025 Niels Lohmann <https://nlohmann.me>
// SPDX-License-Identifier: MIT



// #include <nlohmann/detail/macro_scope.hpp>


NLOHMANN_JSON_NAMESPACE_BEGIN

NLOHMANN_CAN_CALL_STD_FUNC_IMPL(begin);

NLOHMANN_JSON_NAMESPACE_END

// #include <nlohmann/detail/meta/call_std/end.hpp>
//     __ _____ _____ _____
//  __|  |   __|     |   | |  JSON for Modern C++
// |  |  |__   |  |  | | | |  version 3.12.0
// |_____|_____|_____|_|___|  https://github.com/nlohmann/json
//
// SPDX-FileCopyrightText: 2013 - 2025 Niels Lohmann <https://nlohmann.me>
// SPDX-License-Identifier: MIT



// #include <nlohmann/detail/macro_scope.hpp>


NLOHMANN_JSON_NAMESPACE_BEGIN

NLOHMANN_CAN_CALL_STD_FUNC_IMPL(end);

NLOHMANN_JSON_NAMESPACE_END

// #include <nlohmann/detail/meta/cpp_future.hpp>

// #include <nlohmann/detail/meta/detected.hpp>

// #include <nlohmann/json_fwd.hpp>
//     __ _____ _____ _____
//  __|  |   __|     |   | |  JSON for Modern C++
// |  |  |__   |  |  | | | |  version 3.12.0
// |_____|_____|_____|_|___|  https://github.com/nlohmann/json
//
// SPDX-FileCopyrightText: 2013 - 2025 Niels Lohmann <https://nlohmann.me>
// SPDX-License-Identifier: MIT

#ifndef INCLUDE_NLOHMANN_JSON_FWD_HPP_
    #define INCLUDE_NLOHMANN_JSON_FWD_HPP_

    #include <cstdint> // int64_t, uint64_t
    #include <map> // map
    #include <memory> // allocator
    #include <string> // string
    #include <vector> // vector

    // #include <nlohmann/detail/abi_macros.hpp>


    /*!
    @brief namespace for Niels Lohmann
    @see https://github.com/nlohmann
    @since version 1.0.0
    */
    NLOHMANN_JSON_NAMESPACE_BEGIN

    /*!
    @brief default JSONSerializer template argument

    This serializer ignores the template arguments and uses ADL
    ([argument-dependent lookup](https://en.cppreference.com/w/cpp/language/adl))
    for serialization.
    */
    template<typename T = void, typename SFINAE = void>
    struct adl_serializer;

    /// a class to store JSON values
    /// @sa https://json.nlohmann.me/api/basic_json/
    template<template<typename U, typename V, typename... Args> class ObjectType =
    std::map,
    template<typename U, typename... Args> class ArrayType = std::vector,
    class StringType = std::string, class BooleanType = bool,
    class NumberIntegerType = std::int64_t,
    class NumberUnsignedType = std::uint64_t,
    class NumberFloatType = double,
    template<typename U> class AllocatorType = std::allocator,
    template<typename T, typename SFINAE = void> class JSONSerializer =
    adl_serializer,
    class BinaryType = std::vector<std::uint8_t>, // cppcheck-suppress syntaxError
    class CustomBaseClass = void>
    class basic_json;

    /// @brief JSON Pointer defines a string syntax for identifying a specific value within a JSON document
    /// @sa https://json.nlohmann.me/api/json_pointer/
    template<typename RefStringType>
    class json_pointer;

    /*!
    @brief default specialization
    @sa https://json.nlohmann.me/api/json/
    */
    using json = basic_json<>;

    /// @brief a minimal map-like container that preserves insertion order
    /// @sa https://json.nlohmann.me/api/ordered_map/
    template<class Key, class T, class IgnoredLess, class Allocator>
    struct ordered_map;

    /// @brief specialization that maintains the insertion order of object keys
    /// @sa https://json.nlohmann.me/api/ordered_json/
    using ordered_json = basic_json<nlohmann::ordered_map>;

    NLOHMANN_JSON_NAMESPACE_END

#endif  // INCLUDE_NLOHMANN_JSON_FWD_HPP_


NLOHMANN_JSON_NAMESPACE_BEGIN
/*!
@brief detail namespace with internal helper functions

This namespace collects functions that should not be exposed,
implementations of some @ref basic_json methods, and meta-programming helpers.

@since version 2.1.0
*/
namespace detail
{

/////////////
// helpers //
/////////////

// Note to maintainers:
//
// Every trait in this file expects a non CV-qualified type.
// The only exceptions are in the 'aliases for detected' section
// (i.e. those of the form: decltype(T::member_function(std::declval<T>())))
//
// In this case, T has to be properly CV-qualified to constraint the function arguments
// (e.g. to_json(BasicJsonType&, const T&))

template<typename> struct is_basic_json : std::false_type {};

NLOHMANN_BASIC_JSON_TPL_DECLARATION
struct is_basic_json<NLOHMANN_BASIC_JSON_TPL> : std::true_type {};

// used by exceptions create() member functions
// true_type for pointer to possibly cv-qualified basic_json or std::nullptr_t
// false_type otherwise
template<typename BasicJsonContext>
struct is_basic_json_context :
    std::integral_constant < bool,
    is_basic_json<typename std::remove_cv<typename std::remove_pointer<BasicJsonContext>::type>::type>::value
    || std::is_same<BasicJsonContext, std::nullptr_t>::value >
{};

//////////////////////
// json_ref helpers //
//////////////////////

template<typename>
class json_ref;

template<typename>
struct is_json_ref : std::false_type {};

template<typename T>
struct is_json_ref<json_ref<T>> : std::true_type {};

//////////////////////////
// aliases for detected //
//////////////////////////

template<typename T>
using mapped_type_t = typename T::mapped_type;

template<typename T>
using key_type_t = typename T::key_type;

template<typename T>
using value_type_t = typename T::value_type;

template<typename T>
using difference_type_t = typename T::difference_type;

template<typename T>
using pointer_t = typename T::pointer;

template<typename T>
using reference_t = typename T::reference;

template<typename T>
using iterator_category_t = typename T::iterator_category;

template<typename T, typename... Args>
using to_json_function = decltype(T::to_json(std::declval<Args>()...));

template<typename T, typename... Args>
using from_json_function = decltype(T::from_json(std::declval<Args>()...));

template<typename T, typename U>
using get_template_function = decltype(std::declval<T>().template get<U>());

// trait checking if JSONSerializer<T>::from_json(json const&, udt&) exists
template<typename BasicJsonType, typename T, typename = void>
struct has_from_json : std::false_type {};

// trait checking if j.get<T> is valid
// use this trait instead of std::is_constructible or std::is_convertible,
// both rely on, or make use of implicit conversions, and thus fail when T
// has several constructors/operator= (see https://github.com/nlohmann/json/issues/958)
template <typename BasicJsonType, typename T>
struct is_getable
{
    static constexpr bool value = is_detected<get_template_function, const BasicJsonType&, T>::value;
};

template<typename BasicJsonType, typename T>
struct has_from_json < BasicJsonType, T, enable_if_t < !is_basic_json<T>::value >>
{
    using serializer = typename BasicJsonType::template json_serializer<T, void>;

    static constexpr bool value =
        is_detected_exact<void, from_json_function, serializer,
        const BasicJsonType&, T&>::value;
};

// This trait checks if JSONSerializer<T>::from_json(json const&) exists
// this overload is used for non-default-constructible user-defined-types
template<typename BasicJsonType, typename T, typename = void>
struct has_non_default_from_json : std::false_type {};

template<typename BasicJsonType, typename T>
struct has_non_default_from_json < BasicJsonType, T, enable_if_t < !is_basic_json<T>::value >>
{
    using serializer = typename BasicJsonType::template json_serializer<T, void>;

    static constexpr bool value =
        is_detected_exact<T, from_json_function, serializer,
        const BasicJsonType&>::value;
};

// This trait checks if BasicJsonType::json_serializer<T>::to_json exists
// Do not evaluate the trait when T is a basic_json type, to avoid template instantiation infinite recursion.
template<typename BasicJsonType, typename T, typename = void>
struct has_to_json : std::false_type {};

template<typename BasicJsonType, typename T>
struct has_to_json < BasicJsonType, T, enable_if_t < !is_basic_json<T>::value >>
{
    using serializer = typename BasicJsonType::template json_serializer<T, void>;

    static constexpr bool value =
        is_detected_exact<void, to_json_function, serializer, BasicJsonType&,
        T>::value;
};

template<typename T>
using detect_key_compare = typename T::key_compare;

template<typename T>
struct has_key_compare : std::integral_constant<bool, is_detected<detect_key_compare, T>::value> {};

// obtains the actual object key comparator
template<typename BasicJsonType>
struct actual_object_comparator
{
    using object_t = typename BasicJsonType::object_t;
    using object_comparator_t = typename BasicJsonType::default_object_comparator_t;
    using type = typename std::conditional < has_key_compare<object_t>::value,
          typename object_t::key_compare, object_comparator_t>::type;
};

template<typename BasicJsonType>
using actual_object_comparator_t = typename actual_object_comparator<BasicJsonType>::type;

/////////////////
// char_traits //
/////////////////

// Primary template of char_traits calls std char_traits
template<typename T>
struct char_traits : std::char_traits<T>
{};

// Explicitly define char traits for unsigned char since it is not standard
template<>
struct char_traits<unsigned char> : std::char_traits<char>
{
    using char_type = unsigned char;
    using int_type = uint64_t;

    // Redefine to_int_type function
    static int_type to_int_type(char_type c) noexcept
    {
        return static_cast<int_type>(c);
    }

    static char_type to_char_type(int_type i) noexcept
    {
        return static_cast<char_type>(i);
    }

    static constexpr int_type eof() noexcept
    {
        return static_cast<int_type>(std::char_traits<char>::eof());
    }
};

// Explicitly define char traits for signed char since it is not standard
template<>
struct char_traits<signed char> : std::char_traits<char>
{
    using char_type = signed char;
    using int_type = uint64_t;

    // Redefine to_int_type function
    static int_type to_int_type(char_type c) noexcept
    {
        return static_cast<int_type>(c);
    }

    static char_type to_char_type(int_type i) noexcept
    {
        return static_cast<char_type>(i);
    }

    static constexpr int_type eof() noexcept
    {
        return static_cast<int_type>(std::char_traits<char>::eof());
    }
};

///////////////////
// is_ functions //
///////////////////

// https://en.cppreference.com/w/cpp/types/conjunction
template<class...> struct conjunction : std::true_type { };
template<class B> struct conjunction<B> : B { };
template<class B, class... Bn>
struct conjunction<B, Bn...>
: std::conditional<static_cast<bool>(B::value), conjunction<Bn...>, B>::type {};

// https://en.cppreference.com/w/cpp/types/negation
template<class B> struct negation : std::integral_constant < bool, !B::value > { };

// Reimplementation of is_constructible and is_default_constructible, due to them being broken for
// std::pair and std::tuple until LWG 2367 fix (see https://cplusplus.github.io/LWG/lwg-defects.html#2367).
// This causes compile errors in e.g. clang 3.5 or gcc 4.9.
template <typename T>
struct is_default_constructible : std::is_default_constructible<T> {};

template <typename T1, typename T2>
struct is_default_constructible<std::pair<T1, T2>>
    : conjunction<is_default_constructible<T1>, is_default_constructible<T2>> {};

template <typename T1, typename T2>
struct is_default_constructible<const std::pair<T1, T2>>
    : conjunction<is_default_constructible<T1>, is_default_constructible<T2>> {};

template <typename... Ts>
struct is_default_constructible<std::tuple<Ts...>>
    : conjunction<is_default_constructible<Ts>...> {};

template <typename... Ts>
struct is_default_constructible<const std::tuple<Ts...>>
    : conjunction<is_default_constructible<Ts>...> {};

template <typename T, typename... Args>
struct is_constructible : std::is_constructible<T, Args...> {};

template <typename T1, typename T2>
struct is_constructible<std::pair<T1, T2>> : is_default_constructible<std::pair<T1, T2>> {};

template <typename T1, typename T2>
struct is_constructible<const std::pair<T1, T2>> : is_default_constructible<const std::pair<T1, T2>> {};

template <typename... Ts>
struct is_constructible<std::tuple<Ts...>> : is_default_constructible<std::tuple<Ts...>> {};

template <typename... Ts>
struct is_constructible<const std::tuple<Ts...>> : is_default_constructible<const std::tuple<Ts...>> {};

template<typename T, typename = void>
struct is_iterator_traits : std::false_type {};

template<typename T>
struct is_iterator_traits<iterator_traits<T>>
{
  private:
    using traits = iterator_traits<T>;

  public:
    static constexpr auto value =
        is_detected<value_type_t, traits>::value &&
        is_detected<difference_type_t, traits>::value &&
        is_detected<pointer_t, traits>::value &&
        is_detected<iterator_category_t, traits>::value &&
        is_detected<reference_t, traits>::value;
};

template<typename T>
struct is_range
{
  private:
    using t_ref = typename std::add_lvalue_reference<T>::type;

    using iterator = detected_t<result_of_begin, t_ref>;
    using sentinel = detected_t<result_of_end, t_ref>;

    // to be 100% correct, it should use https://en.cppreference.com/w/cpp/iterator/input_or_output_iterator
    // and https://en.cppreference.com/w/cpp/iterator/sentinel_for
    // but reimplementing these would be too much work, as a lot of other concepts are used underneath
    static constexpr auto is_iterator_begin =
        is_iterator_traits<iterator_traits<iterator>>::value;

  public:
    static constexpr bool value = !std::is_same<iterator, nonesuch>::value && !std::is_same<sentinel, nonesuch>::value && is_iterator_begin;
};

template<typename R>
using iterator_t = enable_if_t<is_range<R>::value, result_of_begin<decltype(std::declval<R&>())>>;

template<typename T>
using range_value_t = value_type_t<iterator_traits<iterator_t<T>>>;

// The following implementation of is_complete_type is taken from
// https://blogs.msdn.microsoft.com/vcblog/2015/12/02/partial-support-for-expression-sfinae-in-vs-2015-update-1/
// and is written by Xiang Fan who agreed to using it in this library.

template<typename T, typename = void>
struct is_complete_type : std::false_type {};

template<typename T>
struct is_complete_type<T, decltype(void(sizeof(T)))> : std::true_type {};

template<typename BasicJsonType, typename CompatibleObjectType,
         typename = void>
struct is_compatible_object_type_impl : std::false_type {};

template<typename BasicJsonType, typename CompatibleObjectType>
struct is_compatible_object_type_impl <
    BasicJsonType, CompatibleObjectType,
    enable_if_t < is_detected<mapped_type_t, CompatibleObjectType>::value&&
    is_detected<key_type_t, CompatibleObjectType>::value >>
{
    using object_t = typename BasicJsonType::object_t;

    // macOS's is_constructible does not play well with nonesuch...
    static constexpr bool value =
        is_constructible<typename object_t::key_type,
        typename CompatibleObjectType::key_type>::value &&
        is_constructible<typename object_t::mapped_type,
        typename CompatibleObjectType::mapped_type>::value;
};

template<typename BasicJsonType, typename CompatibleObjectType>
struct is_compatible_object_type
    : is_compatible_object_type_impl<BasicJsonType, CompatibleObjectType> {};

template<typename BasicJsonType, typename ConstructibleObjectType,
         typename = void>
struct is_constructible_object_type_impl : std::false_type {};

template<typename BasicJsonType, typename ConstructibleObjectType>
struct is_constructible_object_type_impl <
    BasicJsonType, ConstructibleObjectType,
    enable_if_t < is_detected<mapped_type_t, ConstructibleObjectType>::value&&
    is_detected<key_type_t, ConstructibleObjectType>::value >>
{
    using object_t = typename BasicJsonType::object_t;

    static constexpr bool value =
        (is_default_constructible<ConstructibleObjectType>::value &&
         (std::is_move_assignable<ConstructibleObjectType>::value ||
          std::is_copy_assignable<ConstructibleObjectType>::value) &&
         (is_constructible<typename ConstructibleObjectType::key_type,
          typename object_t::key_type>::value &&
          std::is_same <
          typename object_t::mapped_type,
          typename ConstructibleObjectType::mapped_type >::value)) ||
        (has_from_json<BasicJsonType,
         typename ConstructibleObjectType::mapped_type>::value ||
         has_non_default_from_json <
         BasicJsonType,
         typename ConstructibleObjectType::mapped_type >::value);
};

template<typename BasicJsonType, typename ConstructibleObjectType>
struct is_constructible_object_type
    : is_constructible_object_type_impl<BasicJsonType,
      ConstructibleObjectType> {};

template<typename BasicJsonType, typename CompatibleStringType>
struct is_compatible_string_type
{
    static constexpr auto value =
        is_constructible<typename BasicJsonType::string_t, CompatibleStringType>::value;
};

template<typename BasicJsonType, typename ConstructibleStringType>
struct is_constructible_string_type
{
    // launder type through decltype() to fix compilation failure on ICPC
#ifdef __INTEL_COMPILER
    using laundered_type = decltype(std::declval<ConstructibleStringType>());
#else
    using laundered_type = ConstructibleStringType;
#endif

    static constexpr auto value =
        conjunction <
        is_constructible<laundered_type, typename BasicJsonType::string_t>,
        is_detected_exact<typename BasicJsonType::string_t::value_type,
        value_type_t, laundered_type >>::value;
};

template<typename BasicJsonType, typename CompatibleArrayType, typename = void>
struct is_compatible_array_type_impl : std::false_type {};

template<typename BasicJsonType, typename CompatibleArrayType>
struct is_compatible_array_type_impl <
    BasicJsonType, CompatibleArrayType,
    enable_if_t <
    is_detected<iterator_t, CompatibleArrayType>::value&&
    is_iterator_traits<iterator_traits<detected_t<iterator_t, CompatibleArrayType>>>::value&&
// special case for types like std::filesystem::path whose iterator's value_type are themselves
// c.f. https://github.com/nlohmann/json/pull/3073
    !std::is_same<CompatibleArrayType, detected_t<range_value_t, CompatibleArrayType>>::value >>
{
    static constexpr bool value =
        is_constructible<BasicJsonType,
        range_value_t<CompatibleArrayType>>::value;
};

template<typename BasicJsonType, typename CompatibleArrayType>
struct is_compatible_array_type
    : is_compatible_array_type_impl<BasicJsonType, CompatibleArrayType> {};

template<typename BasicJsonType, typename ConstructibleArrayType, typename = void>
struct is_constructible_array_type_impl : std::false_type {};

template<typename BasicJsonType, typename ConstructibleArrayType>
struct is_constructible_array_type_impl <
    BasicJsonType, ConstructibleArrayType,
    enable_if_t<std::is_same<ConstructibleArrayType,
    typename BasicJsonType::value_type>::value >>
            : std::true_type {};

template<typename BasicJsonType, typename ConstructibleArrayType>
struct is_constructible_array_type_impl <
    BasicJsonType, ConstructibleArrayType,
    enable_if_t < !std::is_same<ConstructibleArrayType,
    typename BasicJsonType::value_type>::value&&
    !is_compatible_string_type<BasicJsonType, ConstructibleArrayType>::value&&
    is_default_constructible<ConstructibleArrayType>::value&&
(std::is_move_assignable<ConstructibleArrayType>::value ||
 std::is_copy_assignable<ConstructibleArrayType>::value)&&
is_detected<iterator_t, ConstructibleArrayType>::value&&
is_iterator_traits<iterator_traits<detected_t<iterator_t, ConstructibleArrayType>>>::value&&
is_detected<range_value_t, ConstructibleArrayType>::value&&
// special case for types like std::filesystem::path whose iterator's value_type are themselves
// c.f. https://github.com/nlohmann/json/pull/3073
!std::is_same<ConstructibleArrayType, detected_t<range_value_t, ConstructibleArrayType>>::value&&
is_complete_type <
detected_t<range_value_t, ConstructibleArrayType >>::value >>
{
    using value_type = range_value_t<ConstructibleArrayType>;

    static constexpr bool value =
        std::is_same<value_type,
        typename BasicJsonType::array_t::value_type>::value ||
        has_from_json<BasicJsonType,
        value_type>::value ||
        has_non_default_from_json <
        BasicJsonType,
        value_type >::value;
};

template<typename BasicJsonType, typename ConstructibleArrayType>
struct is_constructible_array_type
    : is_constructible_array_type_impl<BasicJsonType, ConstructibleArrayType> {};

template<typename RealIntegerType, typename CompatibleNumberIntegerType,
         typename = void>
struct is_compatible_integer_type_impl : std::false_type {};

template<typename RealIntegerType, typename CompatibleNumberIntegerType>
struct is_compatible_integer_type_impl <
    RealIntegerType, CompatibleNumberIntegerType,
    enable_if_t < std::is_integral<RealIntegerType>::value&&
    std::is_integral<CompatibleNumberIntegerType>::value&&
    !std::is_same<bool, CompatibleNumberIntegerType>::value >>
{
    // is there an assert somewhere on overflows?
    using RealLimits = std::numeric_limits<RealIntegerType>;
    using CompatibleLimits = std::numeric_limits<CompatibleNumberIntegerType>;

    static constexpr auto value =
        is_constructible<RealIntegerType,
        CompatibleNumberIntegerType>::value &&
        CompatibleLimits::is_integer &&
        RealLimits::is_signed == CompatibleLimits::is_signed;
};

template<typename RealIntegerType, typename CompatibleNumberIntegerType>
struct is_compatible_integer_type
    : is_compatible_integer_type_impl<RealIntegerType,
      CompatibleNumberIntegerType> {};

template<typename BasicJsonType, typename CompatibleType, typename = void>
struct is_compatible_type_impl: std::false_type {};

template<typename BasicJsonType, typename CompatibleType>
struct is_compatible_type_impl <
    BasicJsonType, CompatibleType,
    enable_if_t<is_complete_type<CompatibleType>::value >>
{
    static constexpr bool value =
        has_to_json<BasicJsonType, CompatibleType>::value;
};

template<typename BasicJsonType, typename CompatibleType>
struct is_compatible_type
    : is_compatible_type_impl<BasicJsonType, CompatibleType> {};

template<typename T1, typename T2>
struct is_constructible_tuple : std::false_type {};

template<typename T1, typename... Args>
struct is_constructible_tuple<T1, std::tuple<Args...>> : conjunction<is_constructible<T1, Args>...> {};

template<typename BasicJsonType, typename T>
struct is_json_iterator_of : std::false_type {};

template<typename BasicJsonType>
struct is_json_iterator_of<BasicJsonType, typename BasicJsonType::iterator> : std::true_type {};

template<typename BasicJsonType>
struct is_json_iterator_of<BasicJsonType, typename BasicJsonType::const_iterator> : std::true_type
{};

// checks if a given type T is a template specialization of Primary
template<template <typename...> class Primary, typename T>
struct is_specialization_of : std::false_type {};

template<template <typename...> class Primary, typename... Args>
struct is_specialization_of<Primary, Primary<Args...>> : std::true_type {};

template<typename T>
using is_json_pointer = is_specialization_of<::nlohmann::json_pointer, uncvref_t<T>>;

// checks if A and B are comparable using Compare functor
template<typename Compare, typename A, typename B, typename = void>
struct is_comparable : std::false_type {};

template<typename Compare, typename A, typename B>
struct is_comparable<Compare, A, B, void_t<
decltype(std::declval<Compare>()(std::declval<A>(), std::declval<B>())),
decltype(std::declval<Compare>()(std::declval<B>(), std::declval<A>()))
>> : std::true_type {};

template<typename T>
using detect_is_transparent = typename T::is_transparent;

// type trait to check if KeyType can be used as object key (without a BasicJsonType)
// see is_usable_as_basic_json_key_type below
template<typename Comparator, typename ObjectKeyType, typename KeyTypeCVRef, bool RequireTransparentComparator = true,
         bool ExcludeObjectKeyType = RequireTransparentComparator, typename KeyType = uncvref_t<KeyTypeCVRef>>
using is_usable_as_key_type = typename std::conditional <
                              is_comparable<Comparator, ObjectKeyType, KeyTypeCVRef>::value
                              && !(ExcludeObjectKeyType && std::is_same<KeyType,
                                   ObjectKeyType>::value)
                              && (!RequireTransparentComparator
                                  || is_detected <detect_is_transparent, Comparator>::value)
                              && !is_json_pointer<KeyType>::value,
                              std::true_type,
                              std::false_type >::type;

// type trait to check if KeyType can be used as object key
// true if:
//   - KeyType is comparable with BasicJsonType::object_t::key_type
//   - if ExcludeObjectKeyType is true, KeyType is not BasicJsonType::object_t::key_type
//   - the comparator is transparent or RequireTransparentComparator is false
//   - KeyType is not a JSON iterator or json_pointer
template<typename BasicJsonType, typename KeyTypeCVRef, bool RequireTransparentComparator = true,
         bool ExcludeObjectKeyType = RequireTransparentComparator, typename KeyType = uncvref_t<KeyTypeCVRef>>
using is_usable_as_basic_json_key_type = typename std::conditional <
    is_usable_as_key_type<typename BasicJsonType::object_comparator_t,
    typename BasicJsonType::object_t::key_type, KeyTypeCVRef,
    RequireTransparentComparator, ExcludeObjectKeyType>::value
    && !is_json_iterator_of<BasicJsonType, KeyType>::value,
    std::true_type,
    std::false_type >::type;

template<typename ObjectType, typename KeyType>
using detect_erase_with_key_type = decltype(std::declval<ObjectType&>().erase(std::declval<KeyType>()));

// type trait to check if object_t has an erase() member functions accepting KeyType
template<typename BasicJsonType, typename KeyType>
using has_erase_with_key_type = typename std::conditional <
                                is_detected <
                                detect_erase_with_key_type,
                                typename BasicJsonType::object_t, KeyType >::value,
                                std::true_type,
                                std::false_type >::type;

// a naive helper to check if a type is an ordered_map (exploits the fact that
// ordered_map inherits capacity() from std::vector)
template <typename T>
struct is_ordered_map
{
    using one = char;

    struct two
    {
        char x[2]; // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays)
    };

    template <typename C> static one test( decltype(&C::capacity) ) ;
    template <typename C> static two test(...);

    enum { value = sizeof(test<T>(nullptr)) == sizeof(char) }; // NOLINT(cppcoreguidelines-pro-type-vararg,hicpp-vararg)
};

// to avoid useless casts (see https://github.com/nlohmann/json/issues/2893#issuecomment-889152324)
template < typename T, typename U, enable_if_t < !std::is_same<T, U>::value, int > = 0 >
T conditional_static_cast(U value)
{
    return static_cast<T>(value);
}

template<typename T, typename U, enable_if_t<std::is_same<T, U>::value, int> = 0>
T conditional_static_cast(U value)
{
    return value;
}

template<typename... Types>
using all_integral = conjunction<std::is_integral<Types>...>;

template<typename... Types>
using all_signed = conjunction<std::is_signed<Types>...>;

template<typename... Types>
using all_unsigned = conjunction<std::is_unsigned<Types>...>;

// there's a disjunction trait in another PR; replace when merged
template<typename... Types>
using same_sign = std::integral_constant < bool,
      all_signed<Types...>::value || all_unsigned<Types...>::value >;

template<typename OfType, typename T>
using never_out_of_range = std::integral_constant < bool,
      (std::is_signed<OfType>::value && (sizeof(T) < sizeof(OfType)))
      || (same_sign<OfType, T>::value && sizeof(OfType) == sizeof(T)) >;

template<typename OfType, typename T,
         bool OfTypeSigned = std::is_signed<OfType>::value,
         bool TSigned = std::is_signed<T>::value>
struct value_in_range_of_impl2;

template<typename OfType, typename T>
struct value_in_range_of_impl2<OfType, T, false, false>
{
    static constexpr bool test(T val)
    {
        using CommonType = typename std::common_type<OfType, T>::type;
        return static_cast<CommonType>(val) <= static_cast<CommonType>((std::numeric_limits<OfType>::max)());
    }
};

template<typename OfType, typename T>
struct value_in_range_of_impl2<OfType, T, true, false>
{
    static constexpr bool test(T val)
    {
        using CommonType = typename std::common_type<OfType, T>::type;
        return static_cast<CommonType>(val) <= static_cast<CommonType>((std::numeric_limits<OfType>::max)());
    }
};

template<typename OfType, typename T>
struct value_in_range_of_impl2<OfType, T, false, true>
{
    static constexpr bool test(T val)
    {
        using CommonType = typename std::common_type<OfType, T>::type;
        return val >= 0 && static_cast<CommonType>(val) <= static_cast<CommonType>((std::numeric_limits<OfType>::max)());
    }
};

template<typename OfType, typename T>
struct value_in_range_of_impl2<OfType, T, true, true>
{
    static constexpr bool test(T val)
    {
        using CommonType = typename std::common_type<OfType, T>::type;
        return static_cast<CommonType>(val) >= static_cast<CommonType>((std::numeric_limits<OfType>::min)())
               && static_cast<CommonType>(val) <= static_cast<CommonType>((std::numeric_limits<OfType>::max)());
    }
};

template<typename OfType, typename T,
         bool NeverOutOfRange = never_out_of_range<OfType, T>::value,
         typename = detail::enable_if_t<all_integral<OfType, T>::value>>
struct value_in_range_of_impl1;

template<typename OfType, typename T>
struct value_in_range_of_impl1<OfType, T, false>
{
    static constexpr bool test(T val)
    {
        return value_in_range_of_impl2<OfType, T>::test(val);
    }
};

template<typename OfType, typename T>
struct value_in_range_of_impl1<OfType, T, true>
{
    static constexpr bool test(T /*val*/)
    {
        return true;
    }
};

template<typename OfType, typename T>
constexpr bool value_in_range_of(T val)
{
    return value_in_range_of_impl1<OfType, T>::test(val);
}

template<bool Value>
using bool_constant = std::integral_constant<bool, Value>;

///////////////////////////////////////////////////////////////////////////////
// is_c_string
///////////////////////////////////////////////////////////////////////////////

namespace impl
{

template<typename T>
constexpr bool is_c_string()
{
    using TUnExt = typename std::remove_extent<T>::type;
    using TUnCVExt = typename std::remove_cv<TUnExt>::type;
    using TUnPtr = typename std::remove_pointer<T>::type;
    using TUnCVPtr = typename std::remove_cv<TUnPtr>::type;
    return
        (std::is_array<T>::value && std::is_same<TUnCVExt, char>::value)
        || (std::is_pointer<T>::value && std::is_same<TUnCVPtr, char>::value);
}

}  // namespace impl

// checks whether T is a [cv] char */[cv] char[] C string
template<typename T>
struct is_c_string : bool_constant<impl::is_c_string<T>()> {};

template<typename T>
using is_c_string_uncvref = is_c_string<uncvref_t<T>>;

///////////////////////////////////////////////////////////////////////////////
// is_transparent
///////////////////////////////////////////////////////////////////////////////

namespace impl
{

template<typename T>
constexpr bool is_transparent()
{
    return is_detected<detect_is_transparent, T>::value;
}

}  // namespace impl

// checks whether T has a member named is_transparent
template<typename T>
struct is_transparent : bool_constant<impl::is_transparent<T>()> {};

///////////////////////////////////////////////////////////////////////////////

}  // namespace detail
NLOHMANN_JSON_NAMESPACE_END

// #include <nlohmann/detail/string_concat.hpp>
//     __ _____ _____ _____
//  __|  |   __|     |   | |  JSON for Modern C++
// |  |  |__   |  |  | | | |  version 3.12.0
// |_____|_____|_____|_|___|  https://github.com/nlohmann/json
//
// SPDX-FileCopyrightText: 2013 - 2025 Niels Lohmann <https://nlohmann.me>
// SPDX-License-Identifier: MIT



#include <cstring> // strlen
#include <string> // string
#include <utility> // forward

// #include <nlohmann/detail/meta/cpp_future.hpp>

// #include <nlohmann/detail/meta/detected.hpp>


NLOHMANN_JSON_NAMESPACE_BEGIN
namespace detail
{

inline std::size_t concat_length()
{
    return 0;
}

template<typename... Args>
inline std::size_t concat_length(const char* cstr, const Args& ... rest);

template<typename StringType, typename... Args>
inline std::size_t concat_length(const StringType& str, const Args& ... rest);

template<typename... Args>
inline std::size_t concat_length(const char /*c*/, const Args& ... rest)
{
    return 1 + concat_length(rest...);
}

template<typename... Args>
inline std::size_t concat_length(const char* cstr, const Args& ... rest)
{
    // cppcheck-suppress ignoredReturnValue
    return ::strlen(cstr) + concat_length(rest...);
}

template<typename StringType, typename... Args>
inline std::size_t concat_length(const StringType& str, const Args& ... rest)
{
    return str.size() + concat_length(rest...);
}

template<typename OutStringType>
inline void concat_into(OutStringType& /*out*/)
{}

template<typename StringType, typename Arg>
using string_can_append = decltype(std::declval<StringType&>().append(std::declval < Arg && > ()));

template<typename StringType, typename Arg>
using detect_string_can_append = is_detected<string_can_append, StringType, Arg>;

template<typename StringType, typename Arg>
using string_can_append_op = decltype(std::declval<StringType&>() += std::declval < Arg && > ());

template<typename StringType, typename Arg>
using detect_string_can_append_op = is_detected<string_can_append_op, StringType, Arg>;

template<typename StringType, typename Arg>
using string_can_append_iter = decltype(std::declval<StringType&>().append(std::declval<const Arg&>().begin(), std::declval<const Arg&>().end()));

template<typename StringType, typename Arg>
using detect_string_can_append_iter = is_detected<string_can_append_iter, StringType, Arg>;

template<typename StringType, typename Arg>
using string_can_append_data = decltype(std::declval<StringType&>().append(std::declval<const Arg&>().data(), std::declval<const Arg&>().size()));

template<typename StringType, typename Arg>
using detect_string_can_append_data = is_detected<string_can_append_data, StringType, Arg>;

template < typename OutStringType, typename Arg, typename... Args,
           enable_if_t < !detect_string_can_append<OutStringType, Arg>::value
                         && detect_string_can_append_op<OutStringType, Arg>::value, int > = 0 >
inline void concat_into(OutStringType& out, Arg && arg, Args && ... rest);

template < typename OutStringType, typename Arg, typename... Args,
           enable_if_t < !detect_string_can_append<OutStringType, Arg>::value
                         && !detect_string_can_append_op<OutStringType, Arg>::value
                         && detect_string_can_append_iter<OutStringType, Arg>::value, int > = 0 >
inline void concat_into(OutStringType& out, const Arg& arg, Args && ... rest);

template < typename OutStringType, typename Arg, typename... Args,
           enable_if_t < !detect_string_can_append<OutStringType, Arg>::value
                         && !detect_string_can_append_op<OutStringType, Arg>::value
                         && !detect_string_can_append_iter<OutStringType, Arg>::value
                         && detect_string_can_append_data<OutStringType, Arg>::value, int > = 0 >
inline void concat_into(OutStringType& out, const Arg& arg, Args && ... rest);

template<typename OutStringType, typename Arg, typename... Args,
         enable_if_t<detect_string_can_append<OutStringType, Arg>::value, int> = 0>
inline void concat_into(OutStringType& out, Arg && arg, Args && ... rest)
{
    out.append(std::forward<Arg>(arg));
    concat_into(out, std::forward<Args>(rest)...);
}

template < typename OutStringType, typename Arg, typename... Args,
           enable_if_t < !detect_string_can_append<OutStringType, Arg>::value
                         && detect_string_can_append_op<OutStringType, Arg>::value, int > >
inline void concat_into(OutStringType& out, Arg&& arg, Args&& ... rest)
{
    out += std::forward<Arg>(arg);
    concat_into(out, std::forward<Args>(rest)...);
}

template < typename OutStringType, typename Arg, typename... Args,
           enable_if_t < !detect_string_can_append<OutStringType, Arg>::value
                         && !detect_string_can_append_op<OutStringType, Arg>::value
                         && detect_string_can_append_iter<OutStringType, Arg>::value, int > >
inline void concat_into(OutStringType& out, const Arg& arg, Args&& ... rest)
{
    out.append(arg.begin(), arg.end());
    concat_into(out, std::forward<Args>(rest)...);
}

template < typename OutStringType, typename Arg, typename... Args,
           enable_if_t < !detect_string_can_append<OutStringType, Arg>::value
                         && !detect_string_can_append_op<OutStringType, Arg>::value
                         && !detect_string_can_append_iter<OutStringType, Arg>::value
                         && detect_string_can_append_data<OutStringType, Arg>::value, int > >
inline void concat_into(OutStringType& out, const Arg& arg, Args&& ... rest)
{
    out.append(arg.data(), arg.size());
    concat_into(out, std::forward<Args>(rest)...);
}

template<typename OutStringType = std::string, typename... Args>
inline OutStringType concat(Args && ... args)
{
    OutStringType str;
    str.reserve(concat_length(args...));
    concat_into(str, std::forward<Args>(args)...);
    return str;
}

}  // namespace detail
NLOHMANN_JSON_NAMESPACE_END


// With -Wweak-vtables, Clang will complain about the exception classes as they
// have no out-of-line virtual method definitions and their vtable will be
// emitted in every translation unit. This issue cannot be fixed with a
// header-only library as there is no implementation file to move these
// functions to. As a result, we suppress this warning here to avoid client
// code to stumble over this. See https://github.com/nlohmann/json/issues/4087
// for a discussion.
#if defined(__clang__)
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wweak-vtables"
#endif

NLOHMANN_JSON_NAMESPACE_BEGIN
namespace detail
{

////////////////
// exceptions //
////////////////

/// @brief general exception of the @ref basic_json class
/// @sa https://json.nlohmann.me/api/basic_json/exception/
class exception : public std::exception
{
  public:
    /// returns the explanatory string
    const char* what() const noexcept override
    {
        return m.what();
    }

    /// the id of the exception
    const int id; // NOLINT(cppcoreguidelines-non-private-member-variables-in-classes)

  protected:
    JSON_HEDLEY_NON_NULL(3)
    exception(int id_, const char* what_arg) : id(id_), m(what_arg) {} // NOLINT(bugprone-throw-keyword-missing)

    static std::string name(const std::string& ename, int id_)
    {
        return concat("[json.exception.", ename, '.', std::to_string(id_), "] ");
    }

    static std::string diagnostics(std::nullptr_t /*leaf_element*/)
    {
        return "";
    }

    template<typename BasicJsonType>
    static std::string diagnostics(const BasicJsonType* leaf_element)
    {
#if JSON_DIAGNOSTICS
        std::vector<std::string> tokens;
        for (const auto* current = leaf_element; current != nullptr && current->m_parent != nullptr; current = current->m_parent)
        {
            switch (current->m_parent->type())
            {
                case value_t::array:
                {
                    for (std::size_t i = 0; i < current->m_parent->m_data.m_value.array->size(); ++i)
                    {
                        if (&current->m_parent->m_data.m_value.array->operator[](i) == current)
                        {
                            tokens.emplace_back(std::to_string(i));
                            break;
                        }
                    }
                    break;
                }

                case value_t::object:
                {
                    for (const auto& element : *current->m_parent->m_data.m_value.object)
                    {
                        if (&element.second == current)
                        {
                            tokens.emplace_back(element.first.c_str());
                            break;
                        }
                    }
                    break;
                }

                case value_t::null: // LCOV_EXCL_LINE
                case value_t::string: // LCOV_EXCL_LINE
                case value_t::boolean: // LCOV_EXCL_LINE
                case value_t::number_integer: // LCOV_EXCL_LINE
                case value_t::number_unsigned: // LCOV_EXCL_LINE
                case value_t::number_float: // LCOV_EXCL_LINE
                case value_t::binary: // LCOV_EXCL_LINE
                case value_t::discarded: // LCOV_EXCL_LINE
                default:   // LCOV_EXCL_LINE
                    break; // LCOV_EXCL_LINE
            }
        }

        if (tokens.empty())
        {
            return "";
        }

        auto str = std::accumulate(tokens.rbegin(), tokens.rend(), std::string{},
                                   [](const std::string & a, const std::string & b)
        {
            return concat(a, '/', detail::escape(b));
        });

        return concat('(', str, ") ", get_byte_positions(leaf_element));
#else
        return get_byte_positions(leaf_element);
#endif
    }

  private:
    /// an exception object as storage for error messages
    std::runtime_error m;
#if JSON_DIAGNOSTIC_POSITIONS
    template<typename BasicJsonType>
    static std::string get_byte_positions(const BasicJsonType* leaf_element)
    {
        if ((leaf_element->start_pos() != std::string::npos) && (leaf_element->end_pos() != std::string::npos))
        {
            return concat("(bytes ", std::to_string(leaf_element->start_pos()), "-", std::to_string(leaf_element->end_pos()), ") ");
        }
        return "";
    }
#else
    template<typename BasicJsonType>
    static std::string get_byte_positions(const BasicJsonType* leaf_element)
    {
        static_cast<void>(leaf_element);
        return "";
    }
#endif
};

/// @brief exception indicating a parse error
/// @sa https://json.nlohmann.me/api/basic_json/parse_error/
class parse_error : public exception
{
  public:
    /*!
    @brief create a parse error exception
    @param[in] id_       the id of the exception
    @param[in] pos       the position where the error occurred (or with
                         chars_read_total=0 if the position cannot be
                         determined)
    @param[in] what_arg  the explanatory string
    @return parse_error object
    */
    template<typename BasicJsonContext, enable_if_t<is_basic_json_context<BasicJsonContext>::value, int> = 0>
    static parse_error create(int id_, const position_t& pos, const std::string& what_arg, BasicJsonContext context)
    {
        const std::string w = concat(exception::name("parse_error", id_), "parse error",
                                     position_string(pos), ": ", exception::diagnostics(context), what_arg);
        return {id_, pos.chars_read_total, w.c_str()};
    }

    template<typename BasicJsonContext, enable_if_t<is_basic_json_context<BasicJsonContext>::value, int> = 0>
    static parse_error create(int id_, std::size_t byte_, const std::string& what_arg, BasicJsonContext context)
    {
        const std::string w = concat(exception::name("parse_error", id_), "parse error",
                                     (byte_ != 0 ? (concat(" at byte ", std::to_string(byte_))) : ""),
                                     ": ", exception::diagnostics(context), what_arg);
        return {id_, byte_, w.c_str()};
    }

    /*!
    @brief byte index of the parse error

    The byte index of the last read character in the input file.

    @note For an input with n bytes, 1 is the index of the first character and
          n+1 is the index of the terminating null byte or the end of file.
          This also holds true when reading a byte vector (CBOR or MessagePack).
    */
    const std::size_t byte;

  private:
    parse_error(int id_, std::size_t byte_, const char* what_arg)
        : exception(id_, what_arg), byte(byte_) {}

    static std::string position_string(const position_t& pos)
    {
        return concat(" at line ", std::to_string(pos.lines_read + 1),
                      ", column ", std::to_string(pos.chars_read_current_line));
    }
};

/// @brief exception indicating errors with iterators
/// @sa https://json.nlohmann.me/api/basic_json/invalid_iterator/
class invalid_iterator : public exception
{
  public:
    template<typename BasicJsonContext, enable_if_t<is_basic_json_context<BasicJsonContext>::value, int> = 0>
    static invalid_iterator create(int id_, const std::string& what_arg, BasicJsonContext context)
    {
        const std::string w = concat(exception::name("invalid_iterator", id_), exception::diagnostics(context), what_arg);
        return {id_, w.c_str()};
    }

  private:
    JSON_HEDLEY_NON_NULL(3)
    invalid_iterator(int id_, const char* what_arg)
        : exception(id_, what_arg) {}
};

/// @brief exception indicating executing a member function with a wrong type
/// @sa https://json.nlohmann.me/api/basic_json/type_error/
class type_error : public exception
{
  public:
    template<typename BasicJsonContext, enable_if_t<is_basic_json_context<BasicJsonContext>::value, int> = 0>
    static type_error create(int id_, const std::string& what_arg, BasicJsonContext context)
    {
        const std::string w = concat(exception::name("type_error", id_), exception::diagnostics(context), what_arg);
        return {id_, w.c_str()};
    }

  private:
    JSON_HEDLEY_NON_NULL(3)
    type_error(int id_, const char* what_arg) : exception(id_, what_arg) {}
};

/// @brief exception indicating access out of the defined range
/// @sa https://json.nlohmann.me/api/basic_json/out_of_range/
class out_of_range : public exception
{
  public:
    template<typename BasicJsonContext, enable_if_t<is_basic_json_context<BasicJsonContext>::value, int> = 0>
    static out_of_range create(int id_, const std::string& what_arg, BasicJsonContext context)
    {
        const std::string w = concat(exception::name("out_of_range", id_), exception::diagnostics(context), what_arg);
        return {id_, w.c_str()};
    }

  private:
    JSON_HEDLEY_NON_NULL(3)
    out_of_range(int id_, const char* what_arg) : exception(id_, what_arg) {}
};

/// @brief exception indicating other library errors
/// @sa https://json.nlohmann.me/api/basic_json/other_error/
class other_error : public exception
{
  public:
    template<typename BasicJsonContext, enable_if_t<is_basic_json_context<BasicJsonContext>::value, int> = 0>
    static other_error create(int id_, const std::string& what_arg, BasicJsonContext context)
    {
        const std::string w = concat(exception::name("other_error", id_), exception::diagnostics(context), what_arg);
        return {id_, w.c_str()};
    }

  private:
    JSON_HEDLEY_NON_NULL(3)
    other_error(int id_, const char* what_arg) : exception(id_, what_arg) {}
};

}  // namespace detail
NLOHMANN_JSON_NAMESPACE_END

#if defined(__clang__)
    #pragma clang diagnostic pop
#endif

// #include <nlohmann/detail/macro_scope.hpp>

// #include <nlohmann/detail/meta/cpp_future.hpp>

// #include <nlohmann/detail/meta/identity_tag.hpp>
//     __ _____ _____ _____
//  __|  |   __|     |   | |  JSON for Modern C++
// |  |  |__   |  |  | | | |  version 3.12.0
// |_____|_____|_____|_|___|  https://github.com/nlohmann/json
//
// SPDX-FileCopyrightText: 2013 - 2025 Niels Lohmann <https://nlohmann.me>
// SPDX-License-Identifier: MIT



// #include <nlohmann/detail/abi_macros.hpp>


NLOHMANN_JSON_NAMESPACE_BEGIN
namespace detail
{

// dispatching helper struct
template <class T> struct identity_tag {};

}  // namespace detail
NLOHMANN_JSON_NAMESPACE_END

// #include <nlohmann/detail/meta/std_fs.hpp>
//     __ _____ _____ _____
//  __|  |   __|     |   | |  JSON for Modern C++
// |  |  |__   |  |  | | | |  version 3.12.0
// |_____|_____|_____|_|___|  https://github.com/nlohmann/json
//
// SPDX-FileCopyrightText: 2013 - 2025 Niels Lohmann <https://nlohmann.me>
// SPDX-License-Identifier: MIT



// #include <nlohmann/detail/macro_scope.hpp>


#if JSON_HAS_EXPERIMENTAL_FILESYSTEM
#include <experimental/filesystem>
NLOHMANN_JSON_NAMESPACE_BEGIN
namespace detail
{
namespace std_fs = std::experimental::filesystem;
}  // namespace detail
NLOHMANN_JSON_NAMESPACE_END
#elif JSON_HAS_FILESYSTEM
#include <filesystem> // NOLINT(build/c++17)
NLOHMANN_JSON_NAMESPACE_BEGIN
namespace detail
{
namespace std_fs = std::filesystem;
}  // namespace detail
NLOHMANN_JSON_NAMESPACE_END
#endif

// #include <nlohmann/detail/meta/type_traits.hpp>

// #include <nlohmann/detail/string_concat.hpp>

// #include <nlohmann/detail/value_t.hpp>


NLOHMANN_JSON_NAMESPACE_BEGIN
namespace detail
{

template<typename BasicJsonType>
inline void from_json(const BasicJsonType& j, typename std::nullptr_t& n)
{
    if (JSON_HEDLEY_UNLIKELY(!j.is_null()))
    {
        JSON_THROW(type_error::create(302, concat("type must be null, but is ", j.type_name()), &j));
    }
    n = nullptr;
}

#ifdef JSON_HAS_CPP_17
#ifndef JSON_USE_IMPLICIT_CONVERSIONS
template<typename BasicJsonType, typename T>
void from_json(const BasicJsonType& j, std::optional<T>& opt)
{
    if (j.is_null())
    {
        opt = std::nullopt;
    }
    else
    {
        opt.emplace(j.template get<T>());
    }
}

#endif // JSON_USE_IMPLICIT_CONVERSIONS
#endif // JSON_HAS_CPP_17

// overloads for basic_json template parameters
template < typename BasicJsonType, typename ArithmeticType,
           enable_if_t < std::is_arithmetic<ArithmeticType>::value&&
                         !std::is_same<ArithmeticType, typename BasicJsonType::boolean_t>::value,
                         int > = 0 >
void get_arithmetic_value(const BasicJsonType& j, ArithmeticType& val)
{
    switch (static_cast<value_t>(j))
    {
        case value_t::number_unsigned:
        {
            val = static_cast<ArithmeticType>(*j.template get_ptr<const typename BasicJsonType::number_unsigned_t*>());
            break;
        }
        case value_t::number_integer:
        {
            val = static_cast<ArithmeticType>(*j.template get_ptr<const typename BasicJsonType::number_integer_t*>());
            break;
        }
        case value_t::number_float:
        {
            val = static_cast<ArithmeticType>(*j.template get_ptr<const typename BasicJsonType::number_float_t*>());
            break;
        }

        case value_t::null:
        case value_t::object:
        case value_t::array:
        case value_t::string:
        case value_t::boolean:
        case value_t::binary:
        case value_t::discarded:
        default:
            JSON_THROW(type_error::create(302, concat("type must be number, but is ", j.type_name()), &j));
    }
}

template<typename BasicJsonType>
inline void from_json(const BasicJsonType& j, typename BasicJsonType::boolean_t& b)
{
    if (JSON_HEDLEY_UNLIKELY(!j.is_boolean()))
    {
        JSON_THROW(type_error::create(302, concat("type must be boolean, but is ", j.type_name()), &j));
    }
    b = *j.template get_ptr<const typename BasicJsonType::boolean_t*>();
}

template<typename BasicJsonType>
inline void from_json(const BasicJsonType& j, typename BasicJsonType::string_t& s)
{
    if (JSON_HEDLEY_UNLIKELY(!j.is_string()))
    {
        JSON_THROW(type_error::create(302, concat("type must be string, but is ", j.type_name()), &j));
    }
    s = *j.template get_ptr<const typename BasicJsonType::string_t*>();
}

template <
    typename BasicJsonType, typename StringType,
    enable_if_t <
        std::is_assignable<StringType&, const typename BasicJsonType::string_t>::value
        && is_detected_exact<typename BasicJsonType::string_t::value_type, value_type_t, StringType>::value
        && !std::is_same<typename BasicJsonType::string_t, StringType>::value
        && !is_json_ref<StringType>::value, int > = 0 >
inline void from_json(const BasicJsonType& j, StringType& s)
{
    if (JSON_HEDLEY_UNLIKELY(!j.is_string()))
    {
        JSON_THROW(type_error::create(302, concat("type must be string, but is ", j.type_name()), &j));
    }

    s = *j.template get_ptr<const typename BasicJsonType::string_t*>();
}

template<typename BasicJsonType>
inline void from_json(const BasicJsonType& j, typename BasicJsonType::number_float_t& val)
{
    get_arithmetic_value(j, val);
}

template<typename BasicJsonType>
inline void from_json(const BasicJsonType& j, typename BasicJsonType::number_unsigned_t& val)
{
    get_arithmetic_value(j, val);
}

template<typename BasicJsonType>
inline void from_json(const BasicJsonType& j, typename BasicJsonType::number_integer_t& val)
{
    get_arithmetic_value(j, val);
}

#if !JSON_DISABLE_ENUM_SERIALIZATION
template<typename BasicJsonType, typename EnumType,
         enable_if_t<std::is_enum<EnumType>::value, int> = 0>
inline void from_json(const BasicJsonType& j, EnumType& e)
{
    typename std::underlying_type<EnumType>::type val;
    get_arithmetic_value(j, val);
    e = static_cast<EnumType>(val);
}
#endif  // JSON_DISABLE_ENUM_SERIALIZATION

// forward_list doesn't have an insert method
template<typename BasicJsonType, typename T, typename Allocator,
         enable_if_t<is_getable<BasicJsonType, T>::value, int> = 0>
inline void from_json(const BasicJsonType& j, std::forward_list<T, Allocator>& l)
{
    if (JSON_HEDLEY_UNLIKELY(!j.is_array()))
    {
        JSON_THROW(type_error::create(302, concat("type must be array, but is ", j.type_name()), &j));
    }
    l.clear();
    std::transform(j.rbegin(), j.rend(),
                   std::front_inserter(l), [](const BasicJsonType & i)
    {
        return i.template get<T>();
    });
}

// valarray doesn't have an insert method
template<typename BasicJsonType, typename T,
         enable_if_t<is_getable<BasicJsonType, T>::value, int> = 0>
inline void from_json(const BasicJsonType& j, std::valarray<T>& l)
{
    if (JSON_HEDLEY_UNLIKELY(!j.is_array()))
    {
        JSON_THROW(type_error::create(302, concat("type must be array, but is ", j.type_name()), &j));
    }
    l.resize(j.size());
    std::transform(j.begin(), j.end(), std::begin(l),
                   [](const BasicJsonType & elem)
    {
        return elem.template get<T>();
    });
}

template<typename BasicJsonType, typename T, std::size_t N>
auto from_json(const BasicJsonType& j, T (&arr)[N])  // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays)
-> decltype(j.template get<T>(), void())
{
    for (std::size_t i = 0; i < N; ++i)
    {
        arr[i] = j.at(i).template get<T>();
    }
}

template<typename BasicJsonType, typename T, std::size_t N1, std::size_t N2>
auto from_json(const BasicJsonType& j, T (&arr)[N1][N2])  // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays)
-> decltype(j.template get<T>(), void())
{
    for (std::size_t i1 = 0; i1 < N1; ++i1)
    {
        for (std::size_t i2 = 0; i2 < N2; ++i2)
        {
            arr[i1][i2] = j.at(i1).at(i2).template get<T>();
        }
    }
}

template<typename BasicJsonType, typename T, std::size_t N1, std::size_t N2, std::size_t N3>
auto from_json(const BasicJsonType& j, T (&arr)[N1][N2][N3])  // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays)
-> decltype(j.template get<T>(), void())
{
    for (std::size_t i1 = 0; i1 < N1; ++i1)
    {
        for (std::size_t i2 = 0; i2 < N2; ++i2)
        {
            for (std::size_t i3 = 0; i3 < N3; ++i3)
            {
                arr[i1][i2][i3] = j.at(i1).at(i2).at(i3).template get<T>();
            }
        }
    }
}

template<typename BasicJsonType, typename T, std::size_t N1, std::size_t N2, std::size_t N3, std::size_t N4>
auto from_json(const BasicJsonType& j, T (&arr)[N1][N2][N3][N4])  // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays)
-> decltype(j.template get<T>(), void())
{
    for (std::size_t i1 = 0; i1 < N1; ++i1)
    {
        for (std::size_t i2 = 0; i2 < N2; ++i2)
        {
            for (std::size_t i3 = 0; i3 < N3; ++i3)
            {
                for (std::size_t i4 = 0; i4 < N4; ++i4)
                {
                    arr[i1][i2][i3][i4] = j.at(i1).at(i2).at(i3).at(i4).template get<T>();
                }
            }
        }
    }
}

template<typename BasicJsonType>
inline void from_json_array_impl(const BasicJsonType& j, typename BasicJsonType::array_t& arr, priority_tag<3> /*unused*/)
{
    arr = *j.template get_ptr<const typename BasicJsonType::array_t*>();
}

template<typename BasicJsonType, typename T, std::size_t N>
auto from_json_array_impl(const BasicJsonType& j, std::array<T, N>& arr,
                          priority_tag<2> /*unused*/)
-> decltype(j.template get<T>(), void())
{
    for (std::size_t i = 0; i < N; ++i)
    {
        arr[i] = j.at(i).template get<T>();
    }
}

template<typename BasicJsonType, typename ConstructibleArrayType,
         enable_if_t<
             std::is_assignable<ConstructibleArrayType&, ConstructibleArrayType>::value,
             int> = 0>
auto from_json_array_impl(const BasicJsonType& j, ConstructibleArrayType& arr, priority_tag<1> /*unused*/)
-> decltype(
    arr.reserve(std::declval<typename ConstructibleArrayType::size_type>()),
    j.template get<typename ConstructibleArrayType::value_type>(),
    void())
{
    using std::end;

    ConstructibleArrayType ret;
    ret.reserve(j.size());
    std::transform(j.begin(), j.end(),
                   std::inserter(ret, end(ret)), [](const BasicJsonType & i)
    {
        // get<BasicJsonType>() returns *this, this won't call a from_json
        // method when value_type is BasicJsonType
        return i.template get<typename ConstructibleArrayType::value_type>();
    });
    arr = std::move(ret);
}

template<typename BasicJsonType, typename ConstructibleArrayType,
         enable_if_t<
             std::is_assignable<ConstructibleArrayType&, ConstructibleArrayType>::value,
             int> = 0>
inline void from_json_array_impl(const BasicJsonType& j, ConstructibleArrayType& arr,
                                 priority_tag<0> /*unused*/)
{
    using std::end;

    ConstructibleArrayType ret;
    std::transform(
        j.begin(), j.end(), std::inserter(ret, end(ret)),
        [](const BasicJsonType & i)
    {
        // get<BasicJsonType>() returns *this, this won't call a from_json
        // method when value_type is BasicJsonType
        return i.template get<typename ConstructibleArrayType::value_type>();
    });
    arr = std::move(ret);
}

template < typename BasicJsonType, typename ConstructibleArrayType,
           enable_if_t <
               is_constructible_array_type<BasicJsonType, ConstructibleArrayType>::value&&
               !is_constructible_object_type<BasicJsonType, ConstructibleArrayType>::value&&
               !is_constructible_string_type<BasicJsonType, ConstructibleArrayType>::value&&
               !std::is_same<ConstructibleArrayType, typename BasicJsonType::binary_t>::value&&
               !is_basic_json<ConstructibleArrayType>::value,
               int > = 0 >
auto from_json(const BasicJsonType& j, ConstructibleArrayType& arr)
-> decltype(from_json_array_impl(j, arr, priority_tag<3> {}),
j.template get<typename ConstructibleArrayType::value_type>(),
void())
{
    if (JSON_HEDLEY_UNLIKELY(!j.is_array()))
    {
        JSON_THROW(type_error::create(302, concat("type must be array, but is ", j.type_name()), &j));
    }

    from_json_array_impl(j, arr, priority_tag<3> {});
}

template < typename BasicJsonType, typename T, std::size_t... Idx >
std::array<T, sizeof...(Idx)> from_json_inplace_array_impl(BasicJsonType&& j,
                     identity_tag<std::array<T, sizeof...(Idx)>> /*unused*/, index_sequence<Idx...> /*unused*/)
{
    return { { std::forward<BasicJsonType>(j).at(Idx).template get<T>()... } };
}

template < typename BasicJsonType, typename T, std::size_t N >
auto from_json(BasicJsonType&& j, identity_tag<std::array<T, N>> tag)
-> decltype(from_json_inplace_array_impl(std::forward<BasicJsonType>(j), tag, make_index_sequence<N> {}))
{
    if (JSON_HEDLEY_UNLIKELY(!j.is_array()))
    {
        JSON_THROW(type_error::create(302, concat("type must be array, but is ", j.type_name()), &j));
    }

    return from_json_inplace_array_impl(std::forward<BasicJsonType>(j), tag, make_index_sequence<N> {});
}

template<typename BasicJsonType>
inline void from_json(const BasicJsonType& j, typename BasicJsonType::binary_t& bin)
{
    if (JSON_HEDLEY_UNLIKELY(!j.is_binary()))
    {
        JSON_THROW(type_error::create(302, concat("type must be binary, but is ", j.type_name()), &j));
    }

    bin = *j.template get_ptr<const typename BasicJsonType::binary_t*>();
}

template<typename BasicJsonType, typename ConstructibleObjectType,
         enable_if_t<is_constructible_object_type<BasicJsonType, ConstructibleObjectType>::value, int> = 0>
inline void from_json(const BasicJsonType& j, ConstructibleObjectType& obj)
{
    if (JSON_HEDLEY_UNLIKELY(!j.is_object()))
    {
        JSON_THROW(type_error::create(302, concat("type must be object, but is ", j.type_name()), &j));
    }

    ConstructibleObjectType ret;
    const auto* inner_object = j.template get_ptr<const typename BasicJsonType::object_t*>();
    using value_type = typename ConstructibleObjectType::value_type;
    std::transform(
        inner_object->begin(), inner_object->end(),
        std::inserter(ret, ret.begin()),
        [](typename BasicJsonType::object_t::value_type const & p)
    {
        return value_type(p.first, p.second.template get<typename ConstructibleObjectType::mapped_type>());
    });
    obj = std::move(ret);
}

// overload for arithmetic types, not chosen for basic_json template arguments
// (BooleanType, etc..); note: Is it really necessary to provide explicit
// overloads for boolean_t etc. in case of a custom BooleanType which is not
// an arithmetic type?
template < typename BasicJsonType, typename ArithmeticType,
           enable_if_t <
               std::is_arithmetic<ArithmeticType>::value&&
               !std::is_same<ArithmeticType, typename BasicJsonType::number_unsigned_t>::value&&
               !std::is_same<ArithmeticType, typename BasicJsonType::number_integer_t>::value&&
               !std::is_same<ArithmeticType, typename BasicJsonType::number_float_t>::value&&
               !std::is_same<ArithmeticType, typename BasicJsonType::boolean_t>::value,
               int > = 0 >
inline void from_json(const BasicJsonType& j, ArithmeticType& val)
{
    switch (static_cast<value_t>(j))
    {
        case value_t::number_unsigned:
        {
            val = static_cast<ArithmeticType>(*j.template get_ptr<const typename BasicJsonType::number_unsigned_t*>());
            break;
        }
        case value_t::number_integer:
        {
            val = static_cast<ArithmeticType>(*j.template get_ptr<const typename BasicJsonType::number_integer_t*>());
            break;
        }
        case value_t::number_float:
        {
            val = static_cast<ArithmeticType>(*j.template get_ptr<const typename BasicJsonType::number_float_t*>());
            break;
        }
        case value_t::boolean:
        {
            val = static_cast<ArithmeticType>(*j.template get_ptr<const typename BasicJsonType::boolean_t*>());
            break;
        }

        case value_t::null:
        case value_t::object:
        case value_t::array:
        case value_t::string:
        case value_t::binary:
        case value_t::discarded:
        default:
            JSON_THROW(type_error::create(302, concat("type must be number, but is ", j.type_name()), &j));
    }
}

template<typename BasicJsonType, typename... Args, std::size_t... Idx>
std::tuple<Args...> from_json_tuple_impl_base(BasicJsonType&& j, index_sequence<Idx...> /*unused*/)
{
    return std::make_tuple(std::forward<BasicJsonType>(j).at(Idx).template get<Args>()...);
}

template<typename BasicJsonType>
std::tuple<> from_json_tuple_impl_base(BasicJsonType& /*unused*/, index_sequence<> /*unused*/)
{
    return {};
}

template < typename BasicJsonType, class A1, class A2 >
std::pair<A1, A2> from_json_tuple_impl(BasicJsonType&& j, identity_tag<std::pair<A1, A2>> /*unused*/, priority_tag<0> /*unused*/)
{
    return {std::forward<BasicJsonType>(j).at(0).template get<A1>(),
            std::forward<BasicJsonType>(j).at(1).template get<A2>()};
}

template<typename BasicJsonType, typename A1, typename A2>
inline void from_json_tuple_impl(BasicJsonType&& j, std::pair<A1, A2>& p, priority_tag<1> /*unused*/)
{
    p = from_json_tuple_impl(std::forward<BasicJsonType>(j), identity_tag<std::pair<A1, A2>> {}, priority_tag<0> {});
}

template<typename BasicJsonType, typename... Args>
std::tuple<Args...> from_json_tuple_impl(BasicJsonType&& j, identity_tag<std::tuple<Args...>> /*unused*/, priority_tag<2> /*unused*/)
{
    return from_json_tuple_impl_base<BasicJsonType, Args...>(std::forward<BasicJsonType>(j), index_sequence_for<Args...> {});
}

template<typename BasicJsonType, typename... Args>
inline void from_json_tuple_impl(BasicJsonType&& j, std::tuple<Args...>& t, priority_tag<3> /*unused*/)
{
    t = from_json_tuple_impl_base<BasicJsonType, Args...>(std::forward<BasicJsonType>(j), index_sequence_for<Args...> {});
}

template<typename BasicJsonType, typename TupleRelated>
auto from_json(BasicJsonType&& j, TupleRelated&& t)
-> decltype(from_json_tuple_impl(std::forward<BasicJsonType>(j), std::forward<TupleRelated>(t), priority_tag<3> {}))
{
    if (JSON_HEDLEY_UNLIKELY(!j.is_array()))
    {
        JSON_THROW(type_error::create(302, concat("type must be array, but is ", j.type_name()), &j));
    }

    return from_json_tuple_impl(std::forward<BasicJsonType>(j), std::forward<TupleRelated>(t), priority_tag<3> {});
}

template < typename BasicJsonType, typename Key, typename Value, typename Compare, typename Allocator,
           typename = enable_if_t < !std::is_constructible <
                                        typename BasicJsonType::string_t, Key >::value >>
inline void from_json(const BasicJsonType& j, std::map<Key, Value, Compare, Allocator>& m)
{
    if (JSON_HEDLEY_UNLIKELY(!j.is_array()))
    {
        JSON_THROW(type_error::create(302, concat("type must be array, but is ", j.type_name()), &j));
    }
    m.clear();
    for (const auto& p : j)
    {
        if (JSON_HEDLEY_UNLIKELY(!p.is_array()))
        {
            JSON_THROW(type_error::create(302, concat("type must be array, but is ", p.type_name()), &j));
        }
        m.emplace(p.at(0).template get<Key>(), p.at(1).template get<Value>());
    }
}

template < typename BasicJsonType, typename Key, typename Value, typename Hash, typename KeyEqual, typename Allocator,
           typename = enable_if_t < !std::is_constructible <
                                        typename BasicJsonType::string_t, Key >::value >>
inline void from_json(const BasicJsonType& j, std::unordered_map<Key, Value, Hash, KeyEqual, Allocator>& m)
{
    if (JSON_HEDLEY_UNLIKELY(!j.is_array()))
    {
        JSON_THROW(type_error::create(302, concat("type must be array, but is ", j.type_name()), &j));
    }
    m.clear();
    for (const auto& p : j)
    {
        if (JSON_HEDLEY_UNLIKELY(!p.is_array()))
        {
            JSON_THROW(type_error::create(302, concat("type must be array, but is ", p.type_name()), &j));
        }
        m.emplace(p.at(0).template get<Key>(), p.at(1).template get<Value>());
    }
}

#if JSON_HAS_FILESYSTEM || JSON_HAS_EXPERIMENTAL_FILESYSTEM
template<typename BasicJsonType>
inline void from_json(const BasicJsonType& j, std_fs::path& p)
{
    if (JSON_HEDLEY_UNLIKELY(!j.is_string()))
    {
        JSON_THROW(type_error::create(302, concat("type must be string, but is ", j.type_name()), &j));
    }
    const auto& s = *j.template get_ptr<const typename BasicJsonType::string_t*>();
#ifdef JSON_HAS_CPP_20
    p = std_fs::path(std::u8string_view(reinterpret_cast<const char8_t*>(s.data()), s.size()));
#else
    p = std_fs::u8path(s); // accepts UTF-8 encoded std::string in C++17, deprecated in C++20
#endif
}
#endif

struct from_json_fn
{
    template<typename BasicJsonType, typename T>
    auto operator()(const BasicJsonType& j, T&& val) const
    noexcept(noexcept(from_json(j, std::forward<T>(val))))
    -> decltype(from_json(j, std::forward<T>(val)))
    {
        return from_json(j, std::forward<T>(val));
    }
};

}  // namespace detail

#ifndef JSON_HAS_CPP_17
/// namespace to hold default `from_json` function
/// to see why this is required:
/// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2015/n4381.html
namespace // NOLINT(cert-dcl59-cpp,fuchsia-header-anon-namespaces,google-build-namespaces)
{
#endif
JSON_INLINE_VARIABLE constexpr const auto& from_json = // NOLINT(misc-definitions-in-headers)
    detail::static_const<detail::from_json_fn>::value;
#ifndef JSON_HAS_CPP_17
}  // namespace
#endif

NLOHMANN_JSON_NAMESPACE_END

// #include <nlohmann/detail/conversions/to_json.hpp>
//     __ _____ _____ _____
//  __|  |   __|     |   | |  JSON for Modern C++
// |  |  |__   |  |  | | | |  version 3.12.0
// |_____|_____|_____|_|___|  https://github.com/nlohmann/json
//
// SPDX-FileCopyrightText: 2013 - 2025 Niels Lohmann <https://nlohmann.me>
// SPDX-License-Identifier: MIT



// #include <nlohmann/detail/macro_scope.hpp>
// JSON_HAS_CPP_17
#ifdef JSON_HAS_CPP_17
    #include <optional> // optional
#endif

#include <algorithm> // copy
#include <iterator> // begin, end
#include <string> // string
#include <tuple> // tuple, get
#include <type_traits> // is_same, is_constructible, is_floating_point, is_enum, underlying_type
#include <utility> // move, forward, declval, pair
#include <valarray> // valarray
#include <vector> // vector

// #include <nlohmann/detail/iterators/iteration_proxy.hpp>
//     __ _____ _____ _____
//  __|  |   __|     |   | |  JSON for Modern C++
// |  |  |__   |  |  | | | |  version 3.12.0
// |_____|_____|_____|_|___|  https://github.com/nlohmann/json
//
// SPDX-FileCopyrightText: 2013 - 2025 Niels Lohmann <https://nlohmann.me>
// SPDX-License-Identifier: MIT



#include <cstddef> // size_t
#include <iterator> // forward_iterator_tag
#include <tuple> // tuple_size, get, tuple_element
#include <utility> // move

#if JSON_HAS_RANGES
    #include <ranges> // enable_borrowed_range
#endif

// #include <nlohmann/detail/abi_macros.hpp>

// #include <nlohmann/detail/meta/type_traits.hpp>

// #include <nlohmann/detail/string_utils.hpp>
//     __ _____ _____ _____
//  __|  |   __|     |   | |  JSON for Modern C++
// |  |  |__   |  |  | | | |  version 3.12.0
// |_____|_____|_____|_|___|  https://github.com/nlohmann/json
//
// SPDX-FileCopyrightText: 2013 - 2025 Niels Lohmann <https://nlohmann.me>
// SPDX-License-Identifier: MIT



#include <cstddef> // size_t
#include <string> // string, to_string

// #include <nlohmann/detail/abi_macros.hpp>


NLOHMANN_JSON_NAMESPACE_BEGIN
namespace detail
{

template<typename StringType>
void int_to_string(StringType& target, std::size_t value)
{
    // For ADL
    using std::to_string;
    target = to_string(value);
}

template<typename StringType>
StringType to_string(std::size_t value)
{
    StringType result;
    int_to_string(result, value);
    return result;
}

}  // namespace detail
NLOHMANN_JSON_NAMESPACE_END

// #include <nlohmann/detail/value_t.hpp>


NLOHMANN_JSON_NAMESPACE_BEGIN
namespace detail
{

template<typename IteratorType> class iteration_proxy_value
{
  public:
    using difference_type = std::ptrdiff_t;
    using value_type = iteration_proxy_value;
    using pointer = value_type *;
    using reference = value_type &;
    using iterator_category = std::forward_iterator_tag;
    using string_type = typename std::remove_cv< typename std::remove_reference<decltype( std::declval<IteratorType>().key() ) >::type >::type;

  private:
    /// the iterator
    IteratorType anchor{};
    /// an index for arrays (used to create key names)
    std::size_t array_index = 0;
    /// last stringified array index
    mutable std::size_t array_index_last = 0;
    /// a string representation of the array index
    mutable string_type array_index_str = "0";
    /// an empty string (to return a reference for primitive values)
    string_type empty_str{};

  public:
    explicit iteration_proxy_value() = default;
    explicit iteration_proxy_value(IteratorType it, std::size_t array_index_ = 0)
    noexcept(std::is_nothrow_move_constructible<IteratorType>::value
             && std::is_nothrow_default_constructible<string_type>::value)
        : anchor(std::move(it))
        , array_index(array_index_)
    {}

    iteration_proxy_value(iteration_proxy_value const&) = default;
    iteration_proxy_value& operator=(iteration_proxy_value const&) = default;
    // older GCCs are a bit fussy and require explicit noexcept specifiers on defaulted functions
    iteration_proxy_value(iteration_proxy_value&&)
    noexcept(std::is_nothrow_move_constructible<IteratorType>::value
             && std::is_nothrow_move_constructible<string_type>::value) = default; // NOLINT(hicpp-noexcept-move,performance-noexcept-move-constructor,cppcoreguidelines-noexcept-move-operations)
    iteration_proxy_value& operator=(iteration_proxy_value&&)
    noexcept(std::is_nothrow_move_assignable<IteratorType>::value
             && std::is_nothrow_move_assignable<string_type>::value) = default; // NOLINT(hicpp-noexcept-move,performance-noexcept-move-constructor,cppcoreguidelines-noexcept-move-operations)
    ~iteration_proxy_value() = default;

    /// dereference operator (needed for range-based for)
    const iteration_proxy_value& operator*() const
    {
        return *this;
    }

    /// increment operator (needed for range-based for)
    iteration_proxy_value& operator++()
    {
        ++anchor;
        ++array_index;

        return *this;
    }

    iteration_proxy_value operator++(int)& // NOLINT(cert-dcl21-cpp)
    {
        auto tmp = iteration_proxy_value(anchor, array_index);
        ++anchor;
        ++array_index;
        return tmp;
    }

    /// equality operator (needed for InputIterator)
    bool operator==(const iteration_proxy_value& o) const
    {
        return anchor == o.anchor;
    }

    /// inequality operator (needed for range-based for)
    bool operator!=(const iteration_proxy_value& o) const
    {
        return anchor != o.anchor;
    }

    /// return key of the iterator
    const string_type& key() const
    {
        JSON_ASSERT(anchor.m_object != nullptr);

        switch (anchor.m_object->type())
        {
            // use integer array index as key
            case value_t::array:
            {
                if (array_index != array_index_last)
                {
                    int_to_string( array_index_str, array_index );
                    array_index_last = array_index;
                }
                return array_index_str;
            }

            // use key from the object
            case value_t::object:
                return anchor.key();

            // use an empty key for all primitive types
            case value_t::null:
            case value_t::string:
            case value_t::boolean:
            case value_t::number_integer:
            case value_t::number_unsigned:
            case value_t::number_float:
            case value_t::binary:
            case value_t::discarded:
            default:
                return empty_str;
        }
    }

    /// return value of the iterator
    typename IteratorType::reference value() const
    {
        return anchor.value();
    }
};

/// proxy class for the items() function
template<typename IteratorType> class iteration_proxy
{
  private:
    /// the container to iterate
    typename IteratorType::pointer container = nullptr;

  public:
    explicit iteration_proxy() = default;

    /// construct iteration proxy from a container
    explicit iteration_proxy(typename IteratorType::reference cont) noexcept
        : container(&cont) {}

    iteration_proxy(iteration_proxy const&) = default;
    iteration_proxy& operator=(iteration_proxy const&) = default;
    iteration_proxy(iteration_proxy&&) noexcept = default;
    iteration_proxy& operator=(iteration_proxy&&) noexcept = default;
    ~iteration_proxy() = default;

    /// return iterator begin (needed for range-based for)
    iteration_proxy_value<IteratorType> begin() const noexcept
    {
        return iteration_proxy_value<IteratorType>(container->begin());
    }

    /// return iterator end (needed for range-based for)
    iteration_proxy_value<IteratorType> end() const noexcept
    {
        return iteration_proxy_value<IteratorType>(container->end());
    }
};

// Structured Bindings Support
// For further reference see https://blog.tartanllama.xyz/structured-bindings/
// And see https://github.com/nlohmann/json/pull/1391
template<std::size_t N, typename IteratorType, enable_if_t<N == 0, int> = 0>
auto get(const nlohmann::detail::iteration_proxy_value<IteratorType>& i) -> decltype(i.key())
{
    return i.key();
}
// Structured Bindings Support
// For further reference see https://blog.tartanllama.xyz/structured-bindings/
// And see https://github.com/nlohmann/json/pull/1391
template<std::size_t N, typename IteratorType, enable_if_t<N == 1, int> = 0>
auto get(const nlohmann::detail::iteration_proxy_value<IteratorType>& i) -> decltype(i.value())
{
    return i.value();
}

}  // namespace detail
NLOHMANN_JSON_NAMESPACE_END

// The Addition to the STD Namespace is required to add
// Structured Bindings Support to the iteration_proxy_value class
// For further reference see https://blog.tartanllama.xyz/structured-bindings/
// And see https://github.com/nlohmann/json/pull/1391
namespace std
{

#if defined(__clang__)
    // Fix: https://github.com/nlohmann/json/issues/1401
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wmismatched-tags"
#endif
template<typename IteratorType>
class tuple_size<::nlohmann::detail::iteration_proxy_value<IteratorType>> // NOLINT(cert-dcl58-cpp)
    : public std::integral_constant<std::size_t, 2> {};

template<std::size_t N, typename IteratorType>
class tuple_element<N, ::nlohmann::detail::iteration_proxy_value<IteratorType >> // NOLINT(cert-dcl58-cpp)
{
  public:
    using type = decltype(
                     get<N>(std::declval <
                            ::nlohmann::detail::iteration_proxy_value<IteratorType >> ()));
};
#if defined(__clang__)
    #pragma clang diagnostic pop
#endif

}  // namespace std

#if JSON_HAS_RANGES
    template <typename IteratorType>
    inline constexpr bool ::std::ranges::enable_borrowed_range<::nlohmann::detail::iteration_proxy<IteratorType>> = true;
#endif

// #include <nlohmann/detail/meta/cpp_future.hpp>

// #include <nlohmann/detail/meta/std_fs.hpp>

// #include <nlohmann/detail/meta/type_traits.hpp>

// #include <nlohmann/detail/value_t.hpp>


NLOHMANN_JSON_NAMESPACE_BEGIN
namespace detail
{

//////////////////
// constructors //
//////////////////

/*
 * Note all external_constructor<>::construct functions need to call
 * j.m_data.m_value.destroy(j.m_data.m_type) to avoid a memory leak in case j contains an
 * allocated value (e.g., a string). See bug issue
 * https://github.com/nlohmann/json/issues/2865 for more information.
 */

template<value_t> struct external_constructor;

template<>
struct external_constructor<value_t::boolean>
{
    template<typename BasicJsonType>
    static void construct(BasicJsonType& j, typename BasicJsonType::boolean_t b) noexcept
    {
        j.m_data.m_value.destroy(j.m_data.m_type);
        j.m_data.m_type = value_t::boolean;
        j.m_data.m_value = b;
        j.assert_invariant();
    }
};

template<>
struct external_constructor<value_t::string>
{
    template<typename BasicJsonType>
    static void construct(BasicJsonType& j, const typename BasicJsonType::string_t& s)
    {
        j.m_data.m_value.destroy(j.m_data.m_type);
        j.m_data.m_type = value_t::string;
        j.m_data.m_value = s;
        j.assert_invariant();
    }

    template<typename BasicJsonType>
    static void construct(BasicJsonType& j, typename BasicJsonType::string_t&& s)
    {
        j.m_data.m_value.destroy(j.m_data.m_type);
        j.m_data.m_type = value_t::string;
        j.m_data.m_value = std::move(s);
        j.assert_invariant();
    }

    template < typename BasicJsonType, typename CompatibleStringType,
               enable_if_t < !std::is_same<CompatibleStringType, typename BasicJsonType::string_t>::value,
                             int > = 0 >
    static void construct(BasicJsonType& j, const CompatibleStringType& str)
    {
        j.m_data.m_value.destroy(j.m_data.m_type);
        j.m_data.m_type = value_t::string;
        j.m_data.m_value.string = j.template create<typename BasicJsonType::string_t>(str);
        j.assert_invariant();
    }
};

template<>
struct external_constructor<value_t::binary>
{
    template<typename BasicJsonType>
    static void construct(BasicJsonType& j, const typename BasicJsonType::binary_t& b)
    {
        j.m_data.m_value.destroy(j.m_data.m_type);
        j.m_data.m_type = value_t::binary;
        j.m_data.m_value = typename BasicJsonType::binary_t(b);
        j.assert_invariant();
    }

    template<typename BasicJsonType>
    static void construct(BasicJsonType& j, typename BasicJsonType::binary_t&& b)
    {
        j.m_data.m_value.destroy(j.m_data.m_type);
        j.m_data.m_type = value_t::binary;
        j.m_data.m_value = typename BasicJsonType::binary_t(std::move(b));
        j.assert_invariant();
    }
};

template<>
struct external_constructor<value_t::number_float>
{
    template<typename BasicJsonType>
    static void construct(BasicJsonType& j, typename BasicJsonType::number_float_t val) noexcept
    {
        j.m_data.m_value.destroy(j.m_data.m_type);
        j.m_data.m_type = value_t::number_float;
        j.m_data.m_value = val;
        j.assert_invariant();
    }
};

template<>
struct external_constructor<value_t::number_unsigned>
{
    template<typename BasicJsonType>
    static void construct(BasicJsonType& j, typename BasicJsonType::number_unsigned_t val) noexcept
    {
        j.m_data.m_value.destroy(j.m_data.m_type);
        j.m_data.m_type = value_t::number_unsigned;
        j.m_data.m_value = val;
        j.assert_invariant();
    }
};

template<>
struct external_constructor<value_t::number_integer>
{
    template<typename BasicJsonType>
    static void construct(BasicJsonType& j, typename BasicJsonType::number_integer_t val) noexcept
    {
        j.m_data.m_value.destroy(j.m_data.m_type);
        j.m_data.m_type = value_t::number_integer;
        j.m_data.m_value = val;
        j.assert_invariant();
    }
};

template<>
struct external_constructor<value_t::array>
{
    template<typename BasicJsonType>
    static void construct(BasicJsonType& j, const typename BasicJsonType::array_t& arr)
    {
        j.m_data.m_value.destroy(j.m_data.m_type);
        j.m_data.m_type = value_t::array;
        j.m_data.m_value = arr;
        j.set_parents();
        j.assert_invariant();
    }

    template<typename BasicJsonType>
    static void construct(BasicJsonType& j, typename BasicJsonType::array_t&& arr)
    {
        j.m_data.m_value.destroy(j.m_data.m_type);
        j.m_data.m_type = value_t::array;
        j.m_data.m_value = std::move(arr);
        j.set_parents();
        j.assert_invariant();
    }

    template < typename BasicJsonType, typename CompatibleArrayType,
               enable_if_t < !std::is_same<CompatibleArrayType, typename BasicJsonType::array_t>::value,
                             int > = 0 >
    static void construct(BasicJsonType& j, const CompatibleArrayType& arr)
    {
        using std::begin;
        using std::end;

        j.m_data.m_value.destroy(j.m_data.m_type);
        j.m_data.m_type = value_t::array;
        j.m_data.m_value.array = j.template create<typename BasicJsonType::array_t>(begin(arr), end(arr));
        j.set_parents();
        j.assert_invariant();
    }

    template<typename BasicJsonType>
    static void construct(BasicJsonType& j, const std::vector<bool>& arr)
    {
        j.m_data.m_value.destroy(j.m_data.m_type);
        j.m_data.m_type = value_t::array;
        j.m_data.m_value = value_t::array;
        j.m_data.m_value.array->reserve(arr.size());
        for (const bool x : arr)
        {
            j.m_data.m_value.array->push_back(x);
            j.set_parent(j.m_data.m_value.array->back());
        }
        j.assert_invariant();
    }

    template<typename BasicJsonType, typename T,
             enable_if_t<std::is_convertible<T, BasicJsonType>::value, int> = 0>
    static void construct(BasicJsonType& j, const std::valarray<T>& arr)
    {
        j.m_data.m_value.destroy(j.m_data.m_type);
        j.m_data.m_type = value_t::array;
        j.m_data.m_value = value_t::array;
        j.m_data.m_value.array->resize(arr.size());
        if (arr.size() > 0)
        {
            std::copy(std::begin(arr), std::end(arr), j.m_data.m_value.array->begin());
        }
        j.set_parents();
        j.assert_invariant();
    }
};

template<>
struct external_constructor<value_t::object>
{
    template<typename BasicJsonType>
    static void construct(BasicJsonType& j, const typename BasicJsonType::object_t& obj)
    {
        j.m_data.m_value.destroy(j.m_data.m_type);
        j.m_data.m_type = value_t::object;
        j.m_data.m_value = obj;
        j.set_parents();
        j.assert_invariant();
    }

    template<typename BasicJsonType>
    static void construct(BasicJsonType& j, typename BasicJsonType::object_t&& obj)
    {
        j.m_data.m_value.destroy(j.m_data.m_type);
        j.m_data.m_type = value_t::object;
        j.m_data.m_value = std::move(obj);
        j.set_parents();
        j.assert_invariant();
    }

    template < typename BasicJsonType, typename CompatibleObjectType,
               enable_if_t < !std::is_same<CompatibleObjectType, typename BasicJsonType::object_t>::value, int > = 0 >
    static void construct(BasicJsonType& j, const CompatibleObjectType& obj)
    {
        using std::begin;
        using std::end;

        j.m_data.m_value.destroy(j.m_data.m_type);
        j.m_data.m_type = value_t::object;
        j.m_data.m_value.object = j.template create<typename BasicJsonType::object_t>(begin(obj), end(obj));
        j.set_parents();
        j.assert_invariant();
    }
};

/////////////
// to_json //
/////////////

#ifdef JSON_HAS_CPP_17
template<typename BasicJsonType, typename T,
         enable_if_t<std::is_constructible<BasicJsonType, T>::value, int> = 0>
void to_json(BasicJsonType& j, const std::optional<T>& opt)
{
    if (opt.has_value())
    {
        j = *opt;
    }
    else
    {
        j = nullptr;
    }
}
#endif

template<typename BasicJsonType, typename T,
         enable_if_t<std::is_same<T, typename BasicJsonType::boolean_t>::value, int> = 0>
inline void to_json(BasicJsonType& j, T b) noexcept
{
    external_constructor<value_t::boolean>::construct(j, b);
}

template < typename BasicJsonType, typename BoolRef,
           enable_if_t <
               ((std::is_same<std::vector<bool>::reference, BoolRef>::value
                 && !std::is_same <std::vector<bool>::reference, typename BasicJsonType::boolean_t&>::value)
                || (std::is_same<std::vector<bool>::const_reference, BoolRef>::value
                    && !std::is_same <detail::uncvref_t<std::vector<bool>::const_reference>,
                                      typename BasicJsonType::boolean_t >::value))
               && std::is_convertible<const BoolRef&, typename BasicJsonType::boolean_t>::value, int > = 0 >
inline void to_json(BasicJsonType& j, const BoolRef& b) noexcept
{
    external_constructor<value_t::boolean>::construct(j, static_cast<typename BasicJsonType::boolean_t>(b));
}

template<typename BasicJsonType, typename CompatibleString,
         enable_if_t<std::is_constructible<typename BasicJsonType::string_t, CompatibleString>::value, int> = 0>
inline void to_json(BasicJsonType& j, const CompatibleString& s)
{
    external_constructor<value_t::string>::construct(j, s);
}

template<typename BasicJsonType>
inline void to_json(BasicJsonType& j, typename BasicJsonType::string_t&& s)
{
    external_constructor<value_t::string>::construct(j, std::move(s));
}

template<typename BasicJsonType, typename FloatType,
         enable_if_t<std::is_floating_point<FloatType>::value, int> = 0>
inline void to_json(BasicJsonType& j, FloatType val) noexcept
{
    external_constructor<value_t::number_float>::construct(j, static_cast<typename BasicJsonType::number_float_t>(val));
}

template<typename BasicJsonType, typename CompatibleNumberUnsignedType,
         enable_if_t<is_compatible_integer_type<typename BasicJsonType::number_unsigned_t, CompatibleNumberUnsignedType>::value, int> = 0>
inline void to_json(BasicJsonType& j, CompatibleNumberUnsignedType val) noexcept
{
    external_constructor<value_t::number_unsigned>::construct(j, static_cast<typename BasicJsonType::number_unsigned_t>(val));
}

template<typename BasicJsonType, typename CompatibleNumberIntegerType,
         enable_if_t<is_compatible_integer_type<typename BasicJsonType::number_integer_t, CompatibleNumberIntegerType>::value, int> = 0>
inline void to_json(BasicJsonType& j, CompatibleNumberIntegerType val) noexcept
{
    external_constructor<value_t::number_integer>::construct(j, static_cast<typename BasicJsonType::number_integer_t>(val));
}

#if !JSON_DISABLE_ENUM_SERIALIZATION
template<typename BasicJsonType, typename EnumType,
         enable_if_t<std::is_enum<EnumType>::value, int> = 0>
inline void to_json(BasicJsonType& j, EnumType e) noexcept
{
    using underlying_type = typename std::underlying_type<EnumType>::type;
    static constexpr value_t integral_value_t = std::is_unsigned<underlying_type>::value ? value_t::number_unsigned : value_t::number_integer;
    external_constructor<integral_value_t>::construct(j, static_cast<underlying_type>(e));
}
#endif  // JSON_DISABLE_ENUM_SERIALIZATION

template<typename BasicJsonType>
inline void to_json(BasicJsonType& j, const std::vector<bool>& e)
{
    external_constructor<value_t::array>::construct(j, e);
}

template < typename BasicJsonType, typename CompatibleArrayType,
           enable_if_t < is_compatible_array_type<BasicJsonType,
                         CompatibleArrayType>::value&&
                         !is_compatible_object_type<BasicJsonType, CompatibleArrayType>::value&&
                         !is_compatible_string_type<BasicJsonType, CompatibleArrayType>::value&&
                         !std::is_same<typename BasicJsonType::binary_t, CompatibleArrayType>::value&&
                         !is_basic_json<CompatibleArrayType>::value,
                         int > = 0 >
inline void to_json(BasicJsonType& j, const CompatibleArrayType& arr)
{
    external_constructor<value_t::array>::construct(j, arr);
}

template<typename BasicJsonType>
inline void to_json(BasicJsonType& j, const typename BasicJsonType::binary_t& bin)
{
    external_constructor<value_t::binary>::construct(j, bin);
}

template<typename BasicJsonType, typename T,
         enable_if_t<std::is_convertible<T, BasicJsonType>::value, int> = 0>
inline void to_json(BasicJsonType& j, const std::valarray<T>& arr)
{
    external_constructor<value_t::array>::construct(j, std::move(arr));
}

template<typename BasicJsonType>
inline void to_json(BasicJsonType& j, typename BasicJsonType::array_t&& arr)
{
    external_constructor<value_t::array>::construct(j, std::move(arr));
}

template < typename BasicJsonType, typename CompatibleObjectType,
           enable_if_t < is_compatible_object_type<BasicJsonType, CompatibleObjectType>::value&& !is_basic_json<CompatibleObjectType>::value, int > = 0 >
inline void to_json(BasicJsonType& j, const CompatibleObjectType& obj)
{
    external_constructor<value_t::object>::construct(j, obj);
}

template<typename BasicJsonType>
inline void to_json(BasicJsonType& j, typename BasicJsonType::object_t&& obj)
{
    external_constructor<value_t::object>::construct(j, std::move(obj));
}

template <
    typename BasicJsonType, typename T, std::size_t N,
    enable_if_t < !std::is_constructible<typename BasicJsonType::string_t,
                  const T(&)[N]>::value, // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays)
                  int > = 0 >
inline void to_json(BasicJsonType& j, const T(&arr)[N]) // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays)
{
    external_constructor<value_t::array>::construct(j, arr);
}

template < typename BasicJsonType, typename T1, typename T2, enable_if_t < std::is_constructible<BasicJsonType, T1>::value&& std::is_constructible<BasicJsonType, T2>::value, int > = 0 >
inline void to_json(BasicJsonType& j, const std::pair<T1, T2>& p)
{
    j = { p.first, p.second };
}

// for https://github.com/nlohmann/json/pull/1134
template<typename BasicJsonType, typename T,
         enable_if_t<std::is_same<T, iteration_proxy_value<typename BasicJsonType::iterator>>::value, int> = 0>
inline void to_json(BasicJsonType& j, const T& b)
{
    j = { {b.key(), b.value()} };
}

template<typename BasicJsonType, typename Tuple, std::size_t... Idx>
inline void to_json_tuple_impl(BasicJsonType& j, const Tuple& t, index_sequence<Idx...> /*unused*/)
{
    j = { std::get<Idx>(t)... };
}

template<typename BasicJsonType, typename Tuple>
inline void to_json_tuple_impl(BasicJsonType& j, const Tuple& /*unused*/, index_sequence<> /*unused*/)
{
    using array_t = typename BasicJsonType::array_t;
    j = array_t();
}

template<typename BasicJsonType, typename T, enable_if_t<is_constructible_tuple<BasicJsonType, T>::value, int > = 0>
inline void to_json(BasicJsonType& j, const T& t)
{
    to_json_tuple_impl(j, t, make_index_sequence<std::tuple_size<T>::value> {});
}

#if JSON_HAS_FILESYSTEM || JSON_HAS_EXPERIMENTAL_FILESYSTEM
template<typename BasicJsonType>
inline void to_json(BasicJsonType& j, const std_fs::path& p)
{
#ifdef JSON_HAS_CPP_20
    const std::u8string s = p.u8string();
    j = std::string(s.begin(), s.end());
#else
    j = p.u8string(); // returns std::string in C++17
#endif
}
#endif

struct to_json_fn
{
    template<typename BasicJsonType, typename T>
    auto operator()(BasicJsonType& j, T&& val) const noexcept(noexcept(to_json(j, std::forward<T>(val))))
    -> decltype(to_json(j, std::forward<T>(val)), void())
    {
        return to_json(j, std::forward<T>(val));
    }
};
}  // namespace detail

#ifndef JSON_HAS_CPP_17
/// namespace to hold default `to_json` function
/// to see why this is required:
/// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2015/n4381.html
namespace // NOLINT(cert-dcl59-cpp,fuchsia-header-anon-namespaces,google-build-namespaces)
{
#endif
JSON_INLINE_VARIABLE constexpr const auto& to_json = // NOLINT(misc-definitions-in-headers)
    detail::static_const<detail::to_json_fn>::value;
#ifndef JSON_HAS_CPP_17
}  // namespace
#endif

NLOHMANN_JSON_NAMESPACE_END

// #include <nlohmann/detail/meta/identity_tag.hpp>


NLOHMANN_JSON_NAMESPACE_BEGIN

/// @sa https://json.nlohmann.me/api/adl_serializer/
template<typename ValueType, typename>
struct adl_serializer
{
    /// @brief convert a JSON value to any value type
    /// @sa https://json.nlohmann.me/api/adl_serializer/from_json/
    template<typename BasicJsonType, typename TargetType = ValueType>
    static auto from_json(BasicJsonType && j, TargetType& val) noexcept(
        noexcept(::nlohmann::from_json(std::forward<BasicJsonType>(j), val)))
    -> decltype(::nlohmann::from_json(std::forward<BasicJsonType>(j), val), void())
    {
        ::nlohmann::from_json(std::forward<BasicJsonType>(j), val);
    }

    /// @brief convert a JSON value to any value type
    /// @sa https://json.nlohmann.me/api/adl_serializer/from_json/
    template<typename BasicJsonType, typename TargetType = ValueType>
    static auto from_json(BasicJsonType && j) noexcept(
    noexcept(::nlohmann::from_json(std::forward<BasicJsonType>(j), detail::identity_tag<TargetType> {})))
    -> decltype(::nlohmann::from_json(std::forward<BasicJsonType>(j), detail::identity_tag<TargetType> {}))
    {
        return ::nlohmann::from_json(std::forward<BasicJsonType>(j), detail::identity_tag<TargetType> {});
    }

    /// @brief convert any value type to a JSON value
    /// @sa https://json.nlohmann.me/api/adl_serializer/to_json/
    template<typename BasicJsonType, typename TargetType = ValueType>
    static auto to_json(BasicJsonType& j, TargetType && val) noexcept(
        noexcept(::nlohmann::to_json(j, std::forward<TargetType>(val))))
    -> decltype(::nlohmann::to_json(j, std::forward<TargetType>(val)), void())
    {
        ::nlohmann::to_json(j, std::forward<TargetType>(val));
    }
};

NLOHMANN_JSON_NAMESPACE_END

// #include <nlohmann/byte_container_with_subtype.hpp>
//     __ _____ _____ _____
//  __|  |   __|     |   | |  JSON for Modern C++
// |  |  |__   |  |  | | | |  version 3.12.0
// |_____|_____|_____|_|___|  https://github.com/nlohmann/json
//
// SPDX-FileCopyrightText: 2013 - 2025 Niels Lohmann <https://nlohmann.me>
// SPDX-License-Identifier: MIT



#include <cstdint> // uint8_t, uint64_t
#include <tuple> // tie
#include <utility> // move

// #include <nlohmann/detail/abi_macros.hpp>


NLOHMANN_JSON_NAMESPACE_BEGIN

/// @brief an internal type for a backed binary type
/// @sa https://json.nlohmann.me/api/byte_container_with_subtype/
template<typename BinaryType>
class byte_container_with_subtype : public BinaryType
{
  public:
    using container_type = BinaryType;
    using subtype_type = std::uint64_t;

    /// @sa https://json.nlohmann.me/api/byte_container_with_subtype/byte_container_with_subtype/
    byte_container_with_subtype() noexcept(noexcept(container_type()))
        : container_type()
    {}

    /// @sa https://json.nlohmann.me/api/byte_container_with_subtype/byte_container_with_subtype/
    byte_container_with_subtype(const container_type& b) noexcept(noexcept(container_type(b)))
        : container_type(b)
    {}

    /// @sa https://json.nlohmann.me/api/byte_container_with_subtype/byte_container_with_subtype/
    byte_container_with_subtype(container_type&& b) noexcept(noexcept(container_type(std::move(b))))
        : container_type(std::move(b))
    {}

    /// @sa https://json.nlohmann.me/api/byte_container_with_subtype/byte_container_with_subtype/
    byte_container_with_subtype(const container_type& b, subtype_type subtype_) noexcept(noexcept(container_type(b)))
        : container_type(b)
        , m_subtype(subtype_)
        , m_has_subtype(true)
    {}

    /// @sa https://json.nlohmann.me/api/byte_container_with_subtype/byte_container_with_subtype/
    byte_container_with_subtype(container_type&& b, subtype_type subtype_) noexcept(noexcept(container_type(std::move(b))))
        : container_type(std::move(b))
        , m_subtype(subtype_)
        , m_has_subtype(true)
    {}

    bool operator==(const byte_container_with_subtype& rhs) const
    {
        return std::tie(static_cast<const BinaryType&>(*this), m_subtype, m_has_subtype) ==
               std::tie(static_cast<const BinaryType&>(rhs), rhs.m_subtype, rhs.m_has_subtype);
    }

    bool operator!=(const byte_container_with_subtype& rhs) const
    {
        return !(rhs == *this);
    }

    /// @brief sets the binary subtype
    /// @sa https://json.nlohmann.me/api/byte_container_with_subtype/set_subtype/
    void set_subtype(subtype_type subtype_) noexcept
    {
        m_subtype = subtype_;
        m_has_subtype = true;
    }

    /// @brief return the binary subtype
    /// @sa https://json.nlohmann.me/api/byte_container_with_subtype/subtype/
    constexpr subtype_type subtype() const noexcept
    {
        return m_has_subtype ? m_subtype : static_cast<subtype_type>(-1);
    }

    /// @brief return whether the value has a subtype
    /// @sa https://json.nlohmann.me/api/byte_container_with_subtype/has_subtype/
    constexpr bool has_subtype() const noexcept
    {
        return m_has_subtype;
    }

    /// @brief clears the binary subtype
    /// @sa https://json.nlohmann.me/api/byte_container_with_subtype/clear_subtype/
    void clear_subtype() noexcept
    {
        m_subtype = 0;
        m_has_subtype = false;
    }

  private:
    subtype_type m_subtype = 0;
    bool m_has_subtype = false;
};

NLOHMANN_JSON_NAMESPACE_END

// #include <nlohmann/detail/conversions/from_json.hpp>

// #include <nlohmann/detail/conversions/to_json.hpp>

// #include <nlohmann/detail/exceptions.hpp>

// #include <nlohmann/detail/hash.hpp>
//     __ _____ _____ _____
//  __|  |   __|     |   | |  JSON for Modern C++
// |  |  |__   |  |  | | | |  version 3.12.0
// |_____|_____|_____|_|___|  https://github.com/nlohmann/json
//
// SPDX-FileCopyrightText: 2013 - 2025 Niels Lohmann <https://nlohmann.me>
// SPDX-License-Identifier: MIT



#include <cstdint> // uint8_t
#include <cstddef> // size_t
#include <functional> // hash

// #include <nlohmann/detail/abi_macros.hpp>

// #include <nlohmann/detail/value_t.hpp>


NLOHMANN_JSON_NAMESPACE_BEGIN
namespace detail
{

// boost::hash_combine
inline std::size_t combine(std::size_t seed, std::size_t h) noexcept
{
    seed ^= h + 0x9e3779b9 + (seed << 6U) + (seed >> 2U);
    return seed;
}

/*!
@brief hash a JSON value

The hash function tries to rely on std::hash where possible. Furthermore, the
type of the JSON value is taken into account to have different hash values for
null, 0, 0U, and false, etc.

@tparam BasicJsonType basic_json specialization
@param j JSON value to hash
@return hash value of j
*/
template<typename BasicJsonType>
std::size_t hash(const BasicJsonType& j)
{
    using string_t = typename BasicJsonType::string_t;
    using number_integer_t = typename BasicJsonType::number_integer_t;
    using number_unsigned_t = typename BasicJsonType::number_unsigned_t;
    using number_float_t = typename BasicJsonType::number_float_t;

    const auto type = static_cast<std::size_t>(j.type());
    switch (j.type())
    {
        case BasicJsonType::value_t::null:
        case BasicJsonType::value_t::discarded:
        {
            return combine(type, 0);
        }

        case BasicJsonType::value_t::object:
        {
            auto seed = combine(type, j.size());
            for (const auto& element : j.items())
            {
                const auto h = std::hash<string_t> {}(element.key());
                seed = combine(seed, h);
                seed = combine(seed, hash(element.value()));
            }
            return seed;
        }

        case BasicJsonType::value_t::array:
        {
            auto seed = combine(type, j.size());
            for (const auto& element : j)
            {
                seed = combine(seed, hash(element));
            }
            return seed;
        }

        case BasicJsonType::value_t::string:
        {
            const auto h = std::hash<string_t> {}(j.template get_ref<const string_t&>());
            return combine(type, h);
        }

        case BasicJsonType::value_t::boolean:
        {
            const auto h = std::hash<bool> {}(j.template get<bool>());
            return combine(type, h);
        }

        case BasicJsonType::value_t::number_integer:
        {
            const auto h = std::hash<number_integer_t> {}(j.template get<number_integer_t>());
            return combine(type, h);
        }

        case BasicJsonType::value_t::number_unsigned:
        {
            const auto h = std::hash<number_unsigned_t> {}(j.template get<number_unsigned_t>());
            return combine(type, h);
        }

        case BasicJsonType::value_t::number_float:
        {
            const auto h = std::hash<number_float_t> {}(j.template get<number_float_t>());
            return combine(type, h);
        }

        case BasicJsonType::value_t::binary:
        {
            auto seed = combine(type, j.get_binary().size());
            const auto h = std::hash<bool> {}(j.get_binary().has_subtype());
            seed = combine(seed, h);
            seed = combine(seed, static_cast<std::size_t>(j.get_binary().subtype()));
            for (const auto byte : j.get_binary())
            {
                seed = combine(seed, std::hash<std::uint8_t> {}(byte));
            }
            return seed;
        }

        default:                   // LCOV_EXCL_LINE
            JSON_ASSERT(false); // NOLINT(cert-dcl03-c,hicpp-static-assert,misc-static-assert) LCOV_EXCL_LINE
            return 0;              // LCOV_EXCL_LINE
    }
}

}  // namespace detail
NLOHMANN_JSON_NAMESPACE_END

// #include <nlohmann/detail/input/binary_reader.hpp>
//     __ _____ _____ _____
//  __|  |   __|     |   | |  JSON for Modern C++
// |  |  |__   |  |  | | | |  version 3.12.0
// |_____|_____|_____|_|___|  https://github.com/nlohmann/json
//
// SPDX-FileCopyrightText: 2013 - 2025 Niels Lohmann <https://nlohmann.me>
// SPDX-License-Identifier: MIT



#include <algorithm> // generate_n
#include <array> // array
#include <cmath> // ldexp
#include <cstddef> // size_t
#include <cstdint> // uint8_t, uint16_t, uint32_t, uint64_t
#include <cstdio> // snprintf
#include <cstring> // memcpy
#include <iterator> // back_inserter
#include <limits> // numeric_limits
#include <string> // char_traits, string
#include <utility> // make_pair, move
#include <vector> // vector
#ifdef __cpp_lib_byteswap
    #include <bit>  //byteswap
#endif

// #include <nlohmann/detail/exceptions.hpp>

// #include <nlohmann/detail/input/input_adapters.hpp>
//     __ _____ _____ _____
//  __|  |   __|     |   | |  JSON for Modern C++
// |  |  |__   |  |  | | | |  version 3.12.0
// |_____|_____|_____|_|___|  https://github.com/nlohmann/json
//
// SPDX-FileCopyrightText: 2013 - 2025 Niels Lohmann <https://nlohmann.me>
// SPDX-License-Identifier: MIT



#include <array> // array
#include <cstddef> // size_t
#include <cstring> // strlen
#include <iterator> // begin, end, iterator_traits, random_access_iterator_tag, distance, next
#include <memory> // shared_ptr, make_shared, addressof
#include <numeric> // accumulate
#include <string> // string, char_traits
#include <type_traits> // enable_if, is_base_of, is_pointer, is_integral, remove_pointer
#include <utility> // pair, declval

#ifndef JSON_NO_IO
    #include <cstdio>   // FILE *
    #include <istream>  // istream
#endif                  // JSON_NO_IO

// #include <nlohmann/detail/exceptions.hpp>

// #include <nlohmann/detail/iterators/iterator_traits.hpp>

// #include <nlohmann/detail/macro_scope.hpp>

// #include <nlohmann/detail/meta/type_traits.hpp>


NLOHMANN_JSON_NAMESPACE_BEGIN
namespace detail
{

/// the supported input formats
enum class input_format_t { json, cbor, msgpack, ubjson, bson, bjdata };

////////////////////
// input adapters //
////////////////////

#ifndef JSON_NO_IO
/*!
Input adapter for stdio file access. This adapter read only 1 byte and do not use any
 buffer. This adapter is a very low level adapter.
*/
class file_input_adapter
{
  public:
    using char_type = char;

    JSON_HEDLEY_NON_NULL(2)
    explicit file_input_adapter(std::FILE* f) noexcept
        : m_file(f)
    {
        JSON_ASSERT(m_file != nullptr);
    }

    // make class move-only
    file_input_adapter(const file_input_adapter&) = delete;
    file_input_adapter(file_input_adapter&&) noexcept = default;
    file_input_adapter& operator=(const file_input_adapter&) = delete;
    file_input_adapter& operator=(file_input_adapter&&) = delete;
    ~file_input_adapter() = default;

    std::char_traits<char>::int_type get_character() noexcept
    {
        return std::fgetc(m_file);
    }

    // returns the number of characters successfully read
    template<class T>
    std::size_t get_elements(T* dest, std::size_t count = 1)
    {
        return fread(dest, 1, sizeof(T) * count, m_file);
    }

  private:
    /// the file pointer to read from
    std::FILE* m_file;
};

/*!
Input adapter for a (caching) istream. Ignores a UFT Byte Order Mark at
beginning of input. Does not support changing the underlying std::streambuf
in mid-input. Maintains underlying std::istream and std::streambuf to support
subsequent use of standard std::istream operations to process any input
characters following those used in parsing the JSON input.  Clears the
std::istream flags; any input errors (e.g., EOF) will be detected by the first
subsequent call for input from the std::istream.
*/
class input_stream_adapter
{
  public:
    using char_type = char;

    ~input_stream_adapter()
    {
        // clear stream flags; we use underlying streambuf I/O, do not
        // maintain ifstream flags, except eof
        if (is != nullptr)
        {
            is->clear(is->rdstate() & std::ios::eofbit);
        }
    }

    explicit input_stream_adapter(std::istream& i)
        : is(&i), sb(i.rdbuf())
    {}

    // delete because of pointer members
    input_stream_adapter(const input_stream_adapter&) = delete;
    input_stream_adapter& operator=(input_stream_adapter&) = delete;
    input_stream_adapter& operator=(input_stream_adapter&&) = delete;

    input_stream_adapter(input_stream_adapter&& rhs) noexcept
        : is(rhs.is), sb(rhs.sb)
    {
        rhs.is = nullptr;
        rhs.sb = nullptr;
    }

    // std::istream/std::streambuf use std::char_traits<char>::to_int_type, to
    // ensure that std::char_traits<char>::eof() and the character 0xFF do not
    // end up as the same value, e.g. 0xFFFFFFFF.
    std::char_traits<char>::int_type get_character()
    {
        auto res = sb->sbumpc();
        // set eof manually, as we don't use the istream interface.
        if (JSON_HEDLEY_UNLIKELY(res == std::char_traits<char>::eof()))
        {
            is->clear(is->rdstate() | std::ios::eofbit);
        }
        return res;
    }

    template<class T>
    std::size_t get_elements(T* dest, std::size_t count = 1)
    {
        auto res = static_cast<std::size_t>(sb->sgetn(reinterpret_cast<char*>(dest), static_cast<std::streamsize>(count * sizeof(T))));
        if (JSON_HEDLEY_UNLIKELY(res < count * sizeof(T)))
        {
            is->clear(is->rdstate() | std::ios::eofbit);
        }
        return res;
    }

  private:
    /// the associated input stream
    std::istream* is = nullptr;
    std::streambuf* sb = nullptr;
};
#endif  // JSON_NO_IO

// General-purpose iterator-based adapter. It might not be as fast as
// theoretically possible for some containers, but it is extremely versatile.
template<typename IteratorType>
class iterator_input_adapter
{
  public:
    using char_type = typename std::iterator_traits<IteratorType>::value_type;

    iterator_input_adapter(IteratorType first, IteratorType last)
        : current(std::move(first)), end(std::move(last))
    {}

    typename char_traits<char_type>::int_type get_character()
    {
        if (JSON_HEDLEY_LIKELY(current != end))
        {
            auto result = char_traits<char_type>::to_int_type(*current);
            std::advance(current, 1);
            return result;
        }

        return char_traits<char_type>::eof();
    }

    // for general iterators, we cannot really do something better than falling back to processing the range one-by-one
    template<class T>
    std::size_t get_elements(T* dest, std::size_t count = 1)
    {
        auto* ptr = reinterpret_cast<char*>(dest);
        for (std::size_t read_index = 0; read_index < count * sizeof(T); ++read_index)
        {
            if (JSON_HEDLEY_LIKELY(current != end))
            {
                ptr[read_index] = static_cast<char>(*current);
                std::advance(current, 1);
            }
            else
            {
                return read_index;
            }
        }
        return count * sizeof(T);
    }

  private:
    IteratorType current;
    IteratorType end;

    template<typename BaseInputAdapter, size_t T>
    friend struct wide_string_input_helper;

    bool empty() const
    {
        return current == end;
    }
};

template<typename BaseInputAdapter, size_t T>
struct wide_string_input_helper;

template<typename BaseInputAdapter>
struct wide_string_input_helper<BaseInputAdapter, 4>
{
    // UTF-32
    static void fill_buffer(BaseInputAdapter& input,
                            std::array<std::char_traits<char>::int_type, 4>& utf8_bytes,
                            size_t& utf8_bytes_index,
                            size_t& utf8_bytes_filled)
    {
        utf8_bytes_index = 0;

        if (JSON_HEDLEY_UNLIKELY(input.empty()))
        {
            utf8_bytes[0] = std::char_traits<char>::eof();
            utf8_bytes_filled = 1;
        }
        else
        {
            // get the current character
            const auto wc = input.get_character();

            // UTF-32 to UTF-8 encoding
            if (wc < 0x80)
            {
                utf8_bytes[0] = static_cast<std::char_traits<char>::int_type>(wc);
                utf8_bytes_filled = 1;
            }
            else if (wc <= 0x7FF)
            {
                utf8_bytes[0] = static_cast<std::char_traits<char>::int_type>(0xC0u | ((static_cast<unsigned int>(wc) >> 6u) & 0x1Fu));
                utf8_bytes[1] = static_cast<std::char_traits<char>::int_type>(0x80u | (static_cast<unsigned int>(wc) & 0x3Fu));
                utf8_bytes_filled = 2;
            }
            else if (wc <= 0xFFFF)
            {
                utf8_bytes[0] = static_cast<std::char_traits<char>::int_type>(0xE0u | ((static_cast<unsigned int>(wc) >> 12u) & 0x0Fu));
                utf8_bytes[1] = static_cast<std::char_traits<char>::int_type>(0x80u | ((static_cast<unsigned int>(wc) >> 6u) & 0x3Fu));
                utf8_bytes[2] = static_cast<std::char_traits<char>::int_type>(0x80u | (static_cast<unsigned int>(wc) & 0x3Fu));
                utf8_bytes_filled = 3;
            }
            else if (wc <= 0x10FFFF)
            {
                utf8_bytes[0] = static_cast<std::char_traits<char>::int_type>(0xF0u | ((static_cast<unsigned int>(wc) >> 18u) & 0x07u));
                utf8_bytes[1] = static_cast<std::char_traits<char>::int_type>(0x80u | ((static_cast<unsigned int>(wc) >> 12u) & 0x3Fu));
                utf8_bytes[2] = static_cast<std::char_traits<char>::int_type>(0x80u | ((static_cast<unsigned int>(wc) >> 6u) & 0x3Fu));
                utf8_bytes[3] = static_cast<std::char_traits<char>::int_type>(0x80u | (static_cast<unsigned int>(wc) & 0x3Fu));
                utf8_bytes_filled = 4;
            }
            else
            {
                // unknown character
                utf8_bytes[0] = static_cast<std::char_traits<char>::int_type>(wc);
                utf8_bytes_filled = 1;
            }
        }
    }
};

template<typename BaseInputAdapter>
struct wide_string_input_helper<BaseInputAdapter, 2>
{
    // UTF-16
    static void fill_buffer(BaseInputAdapter& input,
                            std::array<std::char_traits<char>::int_type, 4>& utf8_bytes,
                            size_t& utf8_bytes_index,
                            size_t& utf8_bytes_filled)
    {
        utf8_bytes_index = 0;

        if (JSON_HEDLEY_UNLIKELY(input.empty()))
        {
            utf8_bytes[0] = std::char_traits<char>::eof();
            utf8_bytes_filled = 1;
        }
        else
        {
            // get the current character
            const auto wc = input.get_character();

            // UTF-16 to UTF-8 encoding
            if (wc < 0x80)
            {
                utf8_bytes[0] = static_cast<std::char_traits<char>::int_type>(wc);
                utf8_bytes_filled = 1;
            }
            else if (wc <= 0x7FF)
            {
                utf8_bytes[0] = static_cast<std::char_traits<char>::int_type>(0xC0u | ((static_cast<unsigned int>(wc) >> 6u)));
                utf8_bytes[1] = static_cast<std::char_traits<char>::int_type>(0x80u | (static_cast<unsigned int>(wc) & 0x3Fu));
                utf8_bytes_filled = 2;
            }
            else if (0xD800 > wc || wc >= 0xE000)
            {
                utf8_bytes[0] = static_cast<std::char_traits<char>::int_type>(0xE0u | ((static_cast<unsigned int>(wc) >> 12u)));
                utf8_bytes[1] = static_cast<std::char_traits<char>::int_type>(0x80u | ((static_cast<unsigned int>(wc) >> 6u) & 0x3Fu));
                utf8_bytes[2] = static_cast<std::char_traits<char>::int_type>(0x80u | (static_cast<unsigned int>(wc) & 0x3Fu));
                utf8_bytes_filled = 3;
            }
            else
            {
                if (JSON_HEDLEY_UNLIKELY(!input.empty()))
                {
                    const auto wc2 = static_cast<unsigned int>(input.get_character());
                    const auto charcode = 0x10000u + (((static_cast<unsigned int>(wc) & 0x3FFu) << 10u) | (wc2 & 0x3FFu));
                    utf8_bytes[0] = static_cast<std::char_traits<char>::int_type>(0xF0u | (charcode >> 18u));
                    utf8_bytes[1] = static_cast<std::char_traits<char>::int_type>(0x80u | ((charcode >> 12u) & 0x3Fu));
                    utf8_bytes[2] = static_cast<std::char_traits<char>::int_type>(0x80u | ((charcode >> 6u) & 0x3Fu));
                    utf8_bytes[3] = static_cast<std::char_traits<char>::int_type>(0x80u | (charcode & 0x3Fu));
                    utf8_bytes_filled = 4;
                }
                else
                {
                    utf8_bytes[0] = static_cast<std::char_traits<char>::int_type>(wc);
                    utf8_bytes_filled = 1;
                }
            }
        }
    }
};

// Wraps another input adapter to convert wide character types into individual bytes.
template<typename BaseInputAdapter, typename WideCharType>
class wide_string_input_adapter
{
  public:
    using char_type = char;

    wide_string_input_adapter(BaseInputAdapter base)
        : base_adapter(base) {}

    typename std::char_traits<char>::int_type get_character() noexcept
    {
        // check if buffer needs to be filled
        if (utf8_bytes_index == utf8_bytes_filled)
        {
            fill_buffer<sizeof(WideCharType)>();

            JSON_ASSERT(utf8_bytes_filled > 0);
            JSON_ASSERT(utf8_bytes_index == 0);
        }

        // use buffer
        JSON_ASSERT(utf8_bytes_filled > 0);
        JSON_ASSERT(utf8_bytes_index < utf8_bytes_filled);
        return utf8_bytes[utf8_bytes_index++];
    }

    // parsing binary with wchar doesn't make sense, but since the parsing mode can be runtime, we need something here
    template<class T>
    std::size_t get_elements(T* /*dest*/, std::size_t /*count*/ = 1)
    {
        JSON_THROW(parse_error::create(112, 1, "wide string type cannot be interpreted as binary data", nullptr));
    }

  private:
    BaseInputAdapter base_adapter;

    template<size_t T>
    void fill_buffer()
    {
        wide_string_input_helper<BaseInputAdapter, T>::fill_buffer(base_adapter, utf8_bytes, utf8_bytes_index, utf8_bytes_filled);
    }

    /// a buffer for UTF-8 bytes
    std::array<std::char_traits<char>::int_type, 4> utf8_bytes = {{0, 0, 0, 0}};

    /// index to the utf8_codes array for the next valid byte
    std::size_t utf8_bytes_index = 0;
    /// number of valid bytes in the utf8_codes array
    std::size_t utf8_bytes_filled = 0;
};

template<typename IteratorType, typename Enable = void>
struct iterator_input_adapter_factory
{
    using iterator_type = IteratorType;
    using char_type = typename std::iterator_traits<iterator_type>::value_type;
    using adapter_type = iterator_input_adapter<iterator_type>;

    static adapter_type create(IteratorType first, IteratorType last)
    {
        return adapter_type(std::move(first), std::move(last));
    }
};

template<typename T>
struct is_iterator_of_multibyte
{
    using value_type = typename std::iterator_traits<T>::value_type;
    enum
    {
        value = sizeof(value_type) > 1
    };
};

template<typename IteratorType>
struct iterator_input_adapter_factory<IteratorType, enable_if_t<is_iterator_of_multibyte<IteratorType>::value>>
{
    using iterator_type = IteratorType;
    using char_type = typename std::iterator_traits<iterator_type>::value_type;
    using base_adapter_type = iterator_input_adapter<iterator_type>;
    using adapter_type = wide_string_input_adapter<base_adapter_type, char_type>;

    static adapter_type create(IteratorType first, IteratorType last)
    {
        return adapter_type(base_adapter_type(std::move(first), std::move(last)));
    }
};

// General purpose iterator-based input
template<typename IteratorType>
typename iterator_input_adapter_factory<IteratorType>::adapter_type input_adapter(IteratorType first, IteratorType last)
{
    using factory_type = iterator_input_adapter_factory<IteratorType>;
    return factory_type::create(first, last);
}

// Convenience shorthand from container to iterator
// Enables ADL on begin(container) and end(container)
// Encloses the using declarations in namespace for not to leak them to outside scope

namespace container_input_adapter_factory_impl
{

using std::begin;
using std::end;

template<typename ContainerType, typename Enable = void>
struct container_input_adapter_factory {};

template<typename ContainerType>
struct container_input_adapter_factory< ContainerType,
       void_t<decltype(begin(std::declval<ContainerType>()), end(std::declval<ContainerType>()))>>
       {
           using adapter_type = decltype(input_adapter(begin(std::declval<ContainerType>()), end(std::declval<ContainerType>())));

           static adapter_type create(const ContainerType& container)
{
    return input_adapter(begin(container), end(container));
}
       };

}  // namespace container_input_adapter_factory_impl

template<typename ContainerType>
typename container_input_adapter_factory_impl::container_input_adapter_factory<ContainerType>::adapter_type input_adapter(const ContainerType& container)
{
    return container_input_adapter_factory_impl::container_input_adapter_factory<ContainerType>::create(container);
}

// specialization for std::string
using string_input_adapter_type = decltype(input_adapter(std::declval<std::string>()));

#ifndef JSON_NO_IO
// Special cases with fast paths
inline file_input_adapter input_adapter(std::FILE* file)
{
    if (file == nullptr)
    {
        JSON_THROW(parse_error::create(101, 0, "attempting to parse an empty input; check that your input string or stream contains the expected JSON", nullptr));
    }
    return file_input_adapter(file);
}

inline input_stream_adapter input_adapter(std::istream& stream)
{
    return input_stream_adapter(stream);
}

inline input_stream_adapter input_adapter(std::istream&& stream)
{
    return input_stream_adapter(stream);
}
#endif  // JSON_NO_IO

using contiguous_bytes_input_adapter = decltype(input_adapter(std::declval<const char*>(), std::declval<const char*>()));

// Null-delimited strings, and the like.
template < typename CharT,
           typename std::enable_if <
               std::is_pointer<CharT>::value&&
               !std::is_array<CharT>::value&&
               std::is_integral<typename std::remove_pointer<CharT>::type>::value&&
               sizeof(typename std::remove_pointer<CharT>::type) == 1,
               int >::type = 0 >
contiguous_bytes_input_adapter input_adapter(CharT b)
{
    if (b == nullptr)
    {
        JSON_THROW(parse_error::create(101, 0, "attempting to parse an empty input; check that your input string or stream contains the expected JSON", nullptr));
    }
    auto length = std::strlen(reinterpret_cast<const char*>(b));
    const auto* ptr = reinterpret_cast<const char*>(b);
    return input_adapter(ptr, ptr + length); // cppcheck-suppress[nullPointerArithmeticRedundantCheck]
}

template<typename T, std::size_t N>
auto input_adapter(T (&array)[N]) -> decltype(input_adapter(array, array + N)) // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays)
{
    return input_adapter(array, array + N);
}

// This class only handles inputs of input_buffer_adapter type.
// It's required so that expressions like {ptr, len} can be implicitly cast
// to the correct adapter.
class span_input_adapter
{
  public:
    template < typename CharT,
               typename std::enable_if <
                   std::is_pointer<CharT>::value&&
                   std::is_integral<typename std::remove_pointer<CharT>::type>::value&&
                   sizeof(typename std::remove_pointer<CharT>::type) == 1,
                   int >::type = 0 >
    span_input_adapter(CharT b, std::size_t l)
        : ia(reinterpret_cast<const char*>(b), reinterpret_cast<const char*>(b) + l) {}

    template<class IteratorType,
             typename std::enable_if<
                 std::is_same<typename iterator_traits<IteratorType>::iterator_category, std::random_access_iterator_tag>::value,
                 int>::type = 0>
    span_input_adapter(IteratorType first, IteratorType last)
        : ia(input_adapter(first, last)) {}

    contiguous_bytes_input_adapter&& get()
    {
        return std::move(ia); // NOLINT(hicpp-move-const-arg,performance-move-const-arg)
    }

  private:
    contiguous_bytes_input_adapter ia;
};

}  // namespace detail
NLOHMANN_JSON_NAMESPACE_END

// #include <nlohmann/detail/input/json_sax.hpp>
//     __ _____ _____ _____
//  __|  |   __|     |   | |  JSON for Modern C++
// |  |  |__   |  |  | | | |  version 3.12.0
// |_____|_____|_____|_|___|  https://github.com/nlohmann/json
//
// SPDX-FileCopyrightText: 2013 - 2025 Niels Lohmann <https://nlohmann.me>
// SPDX-License-Identifier: MIT



#include <cstddef>
#include <string> // string
#include <type_traits> // enable_if_t
#include <utility> // move
#include <vector> // vector

// #include <nlohmann/detail/exceptions.hpp>

// #include <nlohmann/detail/input/lexer.hpp>
//     __ _____ _____ _____
//  __|  |   __|     |   | |  JSON for Modern C++
// |  |  |__   |  |  | | | |  version 3.12.0
// |_____|_____|_____|_|___|  https://github.com/nlohmann/json
//
// SPDX-FileCopyrightText: 2013 - 2025 Niels Lohmann <https://nlohmann.me>
// SPDX-License-Identifier: MIT



#include <array> // array
#include <clocale> // localeconv
#include <cstddef> // size_t
#include <cstdio> // snprintf
#include <cstdlib> // strtof, strtod, strtold, strtoll, strtoull
#include <initializer_list> // initializer_list
#include <string> // char_traits, string
#include <utility> // move
#include <vector> // vector

// #include <nlohmann/detail/input/input_adapters.hpp>

// #include <nlohmann/detail/input/position_t.hpp>

// #include <nlohmann/detail/macro_scope.hpp>

// #include <nlohmann/detail/meta/type_traits.hpp>


NLOHMANN_JSON_NAMESPACE_BEGIN
namespace detail
{

///////////
// lexer //
///////////

template<typename BasicJsonType>
class lexer_base
{
  public:
    /// token types for the parser
    enum class token_type
    {
        uninitialized,    ///< indicating the scanner is uninitialized
        literal_true,     ///< the `true` literal
        literal_false,    ///< the `false` literal
        literal_null,     ///< the `null` literal
        value_string,     ///< a string -- use get_string() for actual value
        value_unsigned,   ///< an unsigned integer -- use get_number_unsigned() for actual value
        value_integer,    ///< a signed integer -- use get_number_integer() for actual value
        value_float,      ///< an floating point number -- use get_number_float() for actual value
        begin_array,      ///< the character for array begin `[`
        begin_object,     ///< the character for object begin `{`
        end_array,        ///< the character for array end `]`
        end_object,       ///< the character for object end `}`
        name_separator,   ///< the name separator `:`
        value_separator,  ///< the value separator `,`
        parse_error,      ///< indicating a parse error
        end_of_input,     ///< indicating the end of the input buffer
        literal_or_value  ///< a literal or the begin of a value (only for diagnostics)
    };

    /// return name of values of type token_type (only used for errors)
    JSON_HEDLEY_RETURNS_NON_NULL
    JSON_HEDLEY_CONST
    static const char* token_type_name(const token_type t) noexcept
    {
        switch (t)
        {
            case token_type::uninitialized:
                return "<uninitialized>";
            case token_type::literal_true:
                return "true literal";
            case token_type::literal_false:
                return "false literal";
            case token_type::literal_null:
                return "null literal";
            case token_type::value_string:
                return "string literal";
            case token_type::value_unsigned:
            case token_type::value_integer:
            case token_type::value_float:
                return "number literal";
            case token_type::begin_array:
                return "'['";
            case token_type::begin_object:
                return "'{'";
            case token_type::end_array:
                return "']'";
            case token_type::end_object:
                return "'}'";
            case token_type::name_separator:
                return "':'";
            case token_type::value_separator:
                return "','";
            case token_type::parse_error:
                return "<parse error>";
            case token_type::end_of_input:
                return "end of input";
            case token_type::literal_or_value:
                return "'[', '{', or a literal";
            // LCOV_EXCL_START
            default: // catch non-enum values
                return "unknown token";
                // LCOV_EXCL_STOP
        }
    }
};
/*!
@brief lexical analysis

This class organizes the lexical analysis during JSON deserialization.
*/
template<typename BasicJsonType, typename InputAdapterType>
class lexer : public lexer_base<BasicJsonType>
{
    using number_integer_t = typename BasicJsonType::number_integer_t;
    using number_unsigned_t = typename BasicJsonType::number_unsigned_t;
    using number_float_t = typename BasicJsonType::number_float_t;
    using string_t = typename BasicJsonType::string_t;
    using char_type = typename InputAdapterType::char_type;
    using char_int_type = typename char_traits<char_type>::int_type;

  public:
    using token_type = typename lexer_base<BasicJsonType>::token_type;

    explicit lexer(InputAdapterType&& adapter, bool ignore_comments_ = false) noexcept
        : ia(std::move(adapter))
        , ignore_comments(ignore_comments_)
        , decimal_point_char(static_cast<char_int_type>(get_decimal_point()))
    {}

    // delete because of pointer members
    lexer(const lexer&) = delete;
    lexer(lexer&&) = default; // NOLINT(hicpp-noexcept-move,performance-noexcept-move-constructor)
    lexer& operator=(lexer&) = delete;
    lexer& operator=(lexer&&) = default; // NOLINT(hicpp-noexcept-move,performance-noexcept-move-constructor)
    ~lexer() = default;

  private:
    /////////////////////
    // locales
    /////////////////////

    /// return the locale-dependent decimal point
    JSON_HEDLEY_PURE
    static char get_decimal_point() noexcept
    {
        const auto* loc = localeconv();
        JSON_ASSERT(loc != nullptr);
        return (loc->decimal_point == nullptr) ? '.' : *(loc->decimal_point);
    }

    /////////////////////
    // scan functions
    /////////////////////

    /*!
    @brief get codepoint from 4 hex characters following `\u`

    For input "\u c1 c2 c3 c4" the codepoint is:
      (c1 * 0x1000) + (c2 * 0x0100) + (c3 * 0x0010) + c4
    = (c1 << 12) + (c2 << 8) + (c3 << 4) + (c4 << 0)

    Furthermore, the possible characters '0'..'9', 'A'..'F', and 'a'..'f'
    must be converted to the integers 0x0..0x9, 0xA..0xF, 0xA..0xF, resp. The
    conversion is done by subtracting the offset (0x30, 0x37, and 0x57)
    between the ASCII value of the character and the desired integer value.

    @return codepoint (0x0000..0xFFFF) or -1 in case of an error (e.g. EOF or
            non-hex character)
    */
    int get_codepoint()
    {
        // this function only makes sense after reading `\u`
        JSON_ASSERT(current == 'u');
        int codepoint = 0;

        const auto factors = { 12u, 8u, 4u, 0u };
        for (const auto factor : factors)
        {
            get();

            if (current >= '0' && current <= '9')
            {
                codepoint += static_cast<int>((static_cast<unsigned int>(current) - 0x30u) << factor);
            }
            else if (current >= 'A' && current <= 'F')
            {
                codepoint += static_cast<int>((static_cast<unsigned int>(current) - 0x37u) << factor);
            }
            else if (current >= 'a' && current <= 'f')
            {
                codepoint += static_cast<int>((static_cast<unsigned int>(current) - 0x57u) << factor);
            }
            else
            {
                return -1;
            }
        }

        JSON_ASSERT(0x0000 <= codepoint && codepoint <= 0xFFFF);
        return codepoint;
    }

    /*!
    @brief check if the next byte(s) are inside a given range

    Adds the current byte and, for each passed range, reads a new byte and
    checks if it is inside the range. If a violation was detected, set up an
    error message and return false. Otherwise, return true.

    @param[in] ranges  list of integers; interpreted as list of pairs of
                       inclusive lower and upper bound, respectively

    @pre The passed list @a ranges must have 2, 4, or 6 elements; that is,
         1, 2, or 3 pairs. This precondition is enforced by an assertion.

    @return true if and only if no range violation was detected
    */
    bool next_byte_in_range(std::initializer_list<char_int_type> ranges)
    {
        JSON_ASSERT(ranges.size() == 2 || ranges.size() == 4 || ranges.size() == 6);
        add(current);

        for (auto range = ranges.begin(); range != ranges.end(); ++range)
        {
            get();
            if (JSON_HEDLEY_LIKELY(*range <= current && current <= *(++range))) // NOLINT(bugprone-inc-dec-in-conditions)
            {
                add(current);
            }
            else
            {
                error_message = "invalid string: ill-formed UTF-8 byte";
                return false;
            }
        }

        return true;
    }

    /*!
    @brief scan a string literal

    This function scans a string according to Sect. 7 of RFC 8259. While
    scanning, bytes are escaped and copied into buffer token_buffer. Then the
    function returns successfully, token_buffer is *not* null-terminated (as it
    may contain \0 bytes), and token_buffer.size() is the number of bytes in the
    string.

    @return token_type::value_string if string could be successfully scanned,
            token_type::parse_error otherwise

    @note In case of errors, variable error_message contains a textual
          description.
    */
    token_type scan_string()
    {
        // reset token_buffer (ignore opening quote)
        reset();

        // we entered the function by reading an open quote
        JSON_ASSERT(current == '\"');

        while (true)
        {
            // get next character
            switch (get())
            {
                // end of file while parsing string
                case char_traits<char_type>::eof():
                {
                    error_message = "invalid string: missing closing quote";
                    return token_type::parse_error;
                }

                // closing quote
                case '\"':
                {
                    return token_type::value_string;
                }

                // escapes
                case '\\':
                {
                    switch (get())
                    {
                        // quotation mark
                        case '\"':
                            add('\"');
                            break;
                        // reverse solidus
                        case '\\':
                            add('\\');
                            break;
                        // solidus
                        case '/':
                            add('/');
                            break;
                        // backspace
                        case 'b':
                            add('\b');
                            break;
                        // form feed
                        case 'f':
                            add('\f');
                            break;
                        // line feed
                        case 'n':
                            add('\n');
                            break;
                        // carriage return
                        case 'r':
                            add('\r');
                            break;
                        // tab
                        case 't':
                            add('\t');
                            break;

                        // unicode escapes
                        case 'u':
                        {
                            const int codepoint1 = get_codepoint();
                            int codepoint = codepoint1; // start with codepoint1

                            if (JSON_HEDLEY_UNLIKELY(codepoint1 == -1))
                            {
                                error_message = "invalid string: '\\u' must be followed by 4 hex digits";
                                return token_type::parse_error;
                            }

                            // check if code point is a high surrogate
                            if (0xD800 <= codepoint1 && codepoint1 <= 0xDBFF)
                            {
                                // expect next \uxxxx entry
                                if (JSON_HEDLEY_LIKELY(get() == '\\' && get() == 'u'))
                                {
                                    const int codepoint2 = get_codepoint();

                                    if (JSON_HEDLEY_UNLIKELY(codepoint2 == -1))
                                    {
                                        error_message = "invalid string: '\\u' must be followed by 4 hex digits";
                                        return token_type::parse_error;
                                    }

                                    // check if codepoint2 is a low surrogate
                                    if (JSON_HEDLEY_LIKELY(0xDC00 <= codepoint2 && codepoint2 <= 0xDFFF))
                                    {
                                        // overwrite codepoint
                                        codepoint = static_cast<int>(
                                                        // high surrogate occupies the most significant 22 bits
                                                        (static_cast<unsigned int>(codepoint1) << 10u)
                                                        // low surrogate occupies the least significant 15 bits
                                                        + static_cast<unsigned int>(codepoint2)
                                                        // there is still the 0xD800, 0xDC00 and 0x10000 noise
                                                        // in the result, so we have to subtract with:
                                                        // (0xD800 << 10) + DC00 - 0x10000 = 0x35FDC00
                                                        - 0x35FDC00u);
                                    }
                                    else
                                    {
                                        error_message = "invalid string: surrogate U+D800..U+DBFF must be followed by U+DC00..U+DFFF";
                                        return token_type::parse_error;
                                    }
                                }
                                else
                                {
                                    error_message = "invalid string: surrogate U+D800..U+DBFF must be followed by U+DC00..U+DFFF";
                                    return token_type::parse_error;
                                }
                            }
                            else
                            {
                                if (JSON_HEDLEY_UNLIKELY(0xDC00 <= codepoint1 && codepoint1 <= 0xDFFF))
                                {
                                    error_message = "invalid string: surrogate U+DC00..U+DFFF must follow U+D800..U+DBFF";
                                    return token_type::parse_error;
                                }
                            }

                            // result of the above calculation yields a proper codepoint
                            JSON_ASSERT(0x00 <= codepoint && codepoint <= 0x10FFFF);

                            // translate codepoint into bytes
                            if (codepoint < 0x80)
                            {
                                // 1-byte characters: 0xxxxxxx (ASCII)
                                add(static_cast<char_int_type>(codepoint));
                            }
                            else if (codepoint <= 0x7FF)
                            {
                                // 2-byte characters: 110xxxxx 10xxxxxx
                                add(static_cast<char_int_type>(0xC0u | (static_cast<unsigned int>(codepoint) >> 6u)));
                                add(static_cast<char_int_type>(0x80u | (static_cast<unsigned int>(codepoint) & 0x3Fu)));
                            }
                            else if (codepoint <= 0xFFFF)
                            {
                                // 3-byte characters: 1110xxxx 10xxxxxx 10xxxxxx
                                add(static_cast<char_int_type>(0xE0u | (static_cast<unsigned int>(codepoint) >> 12u)));
                                add(static_cast<char_int_type>(0x80u | ((static_cast<unsigned int>(codepoint) >> 6u) & 0x3Fu)));
                                add(static_cast<char_int_type>(0x80u | (static_cast<unsigned int>(codepoint) & 0x3Fu)));
                            }
                            else
                            {
                                // 4-byte characters: 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
                                add(static_cast<char_int_type>(0xF0u | (static_cast<unsigned int>(codepoint) >> 18u)));
                                add(static_cast<char_int_type>(0x80u | ((static_cast<unsigned int>(codepoint) >> 12u) & 0x3Fu)));
                                add(static_cast<char_int_type>(0x80u | ((static_cast<unsigned int>(codepoint) >> 6u) & 0x3Fu)));
                                add(static_cast<char_int_type>(0x80u | (static_cast<unsigned int>(codepoint) & 0x3Fu)));
                            }

                            break;
                        }

                        // other characters after escape
                        default:
                            error_message = "invalid string: forbidden character after backslash";
                            return token_type::parse_error;
                    }

                    break;
                }

                // invalid control characters
                case 0x00:
                {
                    error_message = "invalid string: control character U+0000 (NUL) must be escaped to \\u0000";
                    return token_type::parse_error;
                }

                case 0x01:
                {
                    error_message = "invalid string: control character U+0001 (SOH) must be escaped to \\u0001";
                    return token_type::parse_error;
                }

                case 0x02:
                {
                    error_message = "invalid string: control character U+0002 (STX) must be escaped to \\u0002";
                    return token_type::parse_error;
                }

                case 0x03:
                {
                    error_message = "invalid string: control character U+0003 (ETX) must be escaped to \\u0003";
                    return token_type::parse_error;
                }

                case 0x04:
                {
                    error_message = "invalid string: control character U+0004 (EOT) must be escaped to \\u0004";
                    return token_type::parse_error;
                }

                case 0x05:
                {
                    error_message = "invalid string: control character U+0005 (ENQ) must be escaped to \\u0005";
                    return token_type::parse_error;
                }

                case 0x06:
                {
                    error_message = "invalid string: control character U+0006 (ACK) must be escaped to \\u0006";
                    return token_type::parse_error;
                }

                case 0x07:
                {
                    error_message = "invalid string: control character U+0007 (BEL) must be escaped to \\u0007";
                    return token_type::parse_error;
                }

                case 0x08:
                {
                    error_message = "invalid string: control character U+0008 (BS) must be escaped to \\u0008 or \\b";
                    return token_type::parse_error;
                }

                case 0x09:
                {
                    error_message = "invalid string: control character U+0009 (HT) must be escaped to \\u0009 or \\t";
                    return token_type::parse_error;
                }

                case 0x0A:
                {
                    error_message = "invalid string: control character U+000A (LF) must be escaped to \\u000A or \\n";
                    return token_type::parse_error;
                }

                case 0x0B:
                {
                    error_message = "invalid string: control character U+000B (VT) must be escaped to \\u000B";
                    return token_type::parse_error;
                }

                case 0x0C:
                {
                    error_message = "invalid string: control character U+000C (FF) must be escaped to \\u000C or \\f";
                    return token_type::parse_error;
                }

                case 0x0D:
                {
                    error_message = "invalid string: control character U+000D (CR) must be escaped to \\u000D or \\r";
                    return token_type::parse_error;
                }

                case 0x0E:
                {
                    error_message = "invalid string: control character U+000E (SO) must be escaped to \\u000E";
                    return token_type::parse_error;
                }

                case 0x0F:
                {
                    error_message = "invalid string: control character U+000F (SI) must be escaped to \\u000F";
                    return token_type::parse_error;
                }

                case 0x10:
                {
                    error_message = "invalid string: control character U+0010 (DLE) must be escaped to \\u0010";
                    return token_type::parse_error;
                }

                case 0x11:
                {
                    error_message = "invalid string: control character U+0011 (DC1) must be escaped to \\u0011";
                    return token_type::parse_error;
                }

                case 0x12:
                {
                    error_message = "invalid string: control character U+0012 (DC2) must be escaped to \\u0012";
                    return token_type::parse_error;
                }

                case 0x13:
                {
                    error_message = "invalid string: control character U+0013 (DC3) must be escaped to \\u0013";
                    return token_type::parse_error;
                }

                case 0x14:
                {
                    error_message = "invalid string: control character U+0014 (DC4) must be escaped to \\u0014";
                    return token_type::parse_error;
                }

                case 0x15:
                {
                    error_message = "invalid string: control character U+0015 (NAK) must be escaped to \\u0015";
                    return token_type::parse_error;
                }

                case 0x16:
                {
                    error_message = "invalid string: control character U+0016 (SYN) must be escaped to \\u0016";
                    return token_type::parse_error;
                }

                case 0x17:
                {
                    error_message = "invalid string: control character U+0017 (ETB) must be escaped to \\u0017";
                    return token_type::parse_error;
                }

                case 0x18:
                {
                    error_message = "invalid string: control character U+0018 (CAN) must be escaped to \\u0018";
                    return token_type::parse_error;
                }

                case 0x19:
                {
                    error_message = "invalid string: control character U+0019 (EM) must be escaped to \\u0019";
                    return token_type::parse_error;
                }

                case 0x1A:
                {
                    error_message = "invalid string: control character U+001A (SUB) must be escaped to \\u001A";
                    return token_type::parse_error;
                }

                case 0x1B:
                {
                    error_message = "invalid string: control character U+001B (ESC) must be escaped to \\u001B";
                    return token_type::parse_error;
                }

                case 0x1C:
                {
                    error_message = "invalid string: control character U+001C (FS) must be escaped to \\u001C";
                    return token_type::parse_error;
                }

                case 0x1D:
                {
                    error_message = "invalid string: control character U+001D (GS) must be escaped to \\u001D";
                    return token_type::parse_error;
                }

                case 0x1E:
                {
                    error_message = "invalid string: control character U+001E (RS) must be escaped to \\u001E";
                    return token_type::parse_error;
                }

                case 0x1F:
                {
                    error_message = "invalid string: control character U+001F (US) must be escaped to \\u001F";
                    return token_type::parse_error;
                }

                // U+0020..U+007F (except U+0022 (quote) and U+005C (backspace))
                case 0x20:
                case 0x21:
                case 0x23:
                case 0x24:
                case 0x25:
                case 0x26:
                case 0x27:
                case 0x28:
                case 0x29:
                case 0x2A:
                case 0x2B:
                case 0x2C:
                case 0x2D:
                case 0x2E:
                case 0x2F:
                case 0x30:
                case 0x31:
                case 0x32:
                case 0x33:
                case 0x34:
                case 0x35:
                case 0x36:
                case 0x37:
                case 0x38:
                case 0x39:
                case 0x3A:
                case 0x3B:
                case 0x3C:
                case 0x3D:
                case 0x3E:
                case 0x3F:
                case 0x40:
                case 0x41:
                case 0x42:
                case 0x43:
                case 0x44:
                case 0x45:
                case 0x46:
                case 0x47:
                case 0x48:
                case 0x49:
                case 0x4A:
                case 0x4B:
                case 0x4C:
                case 0x4D:
                case 0x4E:
                case 0x4F:
                case 0x50:
                case 0x51:
                case 0x52:
                case 0x53:
                case 0x54:
                case 0x55:
                case 0x56:
                case 0x57:
                case 0x58:
                case 0x59:
                case 0x5A:
                case 0x5B:
                case 0x5D:
                case 0x5E:
                case 0x5F:
                case 0x60:
                case 0x61:
                case 0x62:
                case 0x63:
                case 0x64:
                case 0x65:
                case 0x66:
                case 0x67:
                case 0x68:
                case 0x69:
                case 0x6A:
                case 0x6B:
                case 0x6C:
                case 0x6D:
                case 0x6E:
                case 0x6F:
                case 0x70:
                case 0x71:
                case 0x72:
                case 0x73:
                case 0x74:
                case 0x75:
                case 0x76:
                case 0x77:
                case 0x78:
                case 0x79:
                case 0x7A:
                case 0x7B:
                case 0x7C:
                case 0x7D:
                case 0x7E:
                case 0x7F:
                {
                    add(current);
                    break;
                }

                // U+0080..U+07FF: bytes C2..DF 80..BF
                case 0xC2:
                case 0xC3:
                case 0xC4:
                case 0xC5:
                case 0xC6:
                case 0xC7:
                case 0xC8:
                case 0xC9:
                case 0xCA:
                case 0xCB:
                case 0xCC:
                case 0xCD:
                case 0xCE:
                case 0xCF:
                case 0xD0:
                case 0xD1:
                case 0xD2:
                case 0xD3:
                case 0xD4:
                case 0xD5:
                case 0xD6:
                case 0xD7:
                case 0xD8:
                case 0xD9:
                case 0xDA:
                case 0xDB:
                case 0xDC:
                case 0xDD:
                case 0xDE:
                case 0xDF:
                {
                    if (JSON_HEDLEY_UNLIKELY(!next_byte_in_range({0x80, 0xBF})))
                    {
                        return token_type::parse_error;
                    }
                    break;
                }

                // U+0800..U+0FFF: bytes E0 A0..BF 80..BF
                case 0xE0:
                {
                    if (JSON_HEDLEY_UNLIKELY(!(next_byte_in_range({0xA0, 0xBF, 0x80, 0xBF}))))
                    {
                        return token_type::parse_error;
                    }
                    break;
                }

                // U+1000..U+CFFF: bytes E1..EC 80..BF 80..BF
                // U+E000..U+FFFF: bytes EE..EF 80..BF 80..BF
                case 0xE1:
                case 0xE2:
                case 0xE3:
                case 0xE4:
                case 0xE5:
                case 0xE6:
                case 0xE7:
                case 0xE8:
                case 0xE9:
                case 0xEA:
                case 0xEB:
                case 0xEC:
                case 0xEE:
                case 0xEF:
                {
                    if (JSON_HEDLEY_UNLIKELY(!(next_byte_in_range({0x80, 0xBF, 0x80, 0xBF}))))
                    {
                        return token_type::parse_error;
                    }
                    break;
                }

                // U+D000..U+D7FF: bytes ED 80..9F 80..BF
                case 0xED:
                {
                    if (JSON_HEDLEY_UNLIKELY(!(next_byte_in_range({0x80, 0x9F, 0x80, 0xBF}))))
                    {
                        return token_type::parse_error;
                    }
                    break;
                }

                // U+10000..U+3FFFF F0 90..BF 80..BF 80..BF
                case 0xF0:
                {
                    if (JSON_HEDLEY_UNLIKELY(!(next_byte_in_range({0x90, 0xBF, 0x80, 0xBF, 0x80, 0xBF}))))
                    {
                        return token_type::parse_error;
                    }
                    break;
                }

                // U+40000..U+FFFFF F1..F3 80..BF 80..BF 80..BF
                case 0xF1:
                case 0xF2:
                case 0xF3:
                {
                    if (JSON_HEDLEY_UNLIKELY(!(next_byte_in_range({0x80, 0xBF, 0x80, 0xBF, 0x80, 0xBF}))))
                    {
                        return token_type::parse_error;
                    }
                    break;
                }

                // U+100000..U+10FFFF F4 80..8F 80..BF 80..BF
                case 0xF4:
                {
                    if (JSON_HEDLEY_UNLIKELY(!(next_byte_in_range({0x80, 0x8F, 0x80, 0xBF, 0x80, 0xBF}))))
                    {
                        return token_type::parse_error;
                    }
                    break;
                }

                // remaining bytes (80..C1 and F5..FF) are ill-formed
                default:
                {
                    error_message = "invalid string: ill-formed UTF-8 byte";
                    return token_type::parse_error;
                }
            }
        }
    }

    /*!
     * @brief scan a comment
     * @return whether comment could be scanned successfully
     */
    bool scan_comment()
    {
        switch (get())
        {
            // single-line comments skip input until a newline or EOF is read
            case '/':
            {
                while (true)
                {
                    switch (get())
                    {
                        case '\n':
                        case '\r':
                        case char_traits<char_type>::eof():
                        case '\0':
                            return true;

                        default:
                            break;
                    }
                }
            }

            // multi-line comments skip input until */ is read
            case '*':
            {
                while (true)
                {
                    switch (get())
                    {
                        case char_traits<char_type>::eof():
                        case '\0':
                        {
                            error_message = "invalid comment; missing closing '*/'";
                            return false;
                        }

                        case '*':
                        {
                            switch (get())
                            {
                                case '/':
                                    return true;

                                default:
                                {
                                    unget();
                                    continue;
                                }
                            }
                        }

                        default:
                            continue;
                    }
                }
            }

            // unexpected character after reading '/'
            default:
            {
                error_message = "invalid comment; expecting '/' or '*' after '/'";
                return false;
            }
        }
    }

    JSON_HEDLEY_NON_NULL(2)
    static void strtof(float& f, const char* str, char** endptr) noexcept
    {
        f = std::strtof(str, endptr);
    }

    JSON_HEDLEY_NON_NULL(2)
    static void strtof(double& f, const char* str, char** endptr) noexcept
    {
        f = std::strtod(str, endptr);
    }

    JSON_HEDLEY_NON_NULL(2)
    static void strtof(long double& f, const char* str, char** endptr) noexcept
    {
        f = std::strtold(str, endptr);
    }

    /*!
    @brief scan a number literal

    This function scans a string according to Sect. 6 of RFC 8259.

    The function is realized with a deterministic finite state machine derived
    from the grammar described in RFC 8259. Starting in state "init", the
    input is read and used to determined the next state. Only state "done"
    accepts the number. State "error" is a trap state to model errors. In the
    table below, "anything" means any character but the ones listed before.

    state    | 0        | 1-9      | e E      | +       | -       | .        | anything
    ---------|----------|----------|----------|---------|---------|----------|-----------
    init     | zero     | any1     | [error]  | [error] | minus   | [error]  | [error]
    minus    | zero     | any1     | [error]  | [error] | [error] | [error]  | [error]
    zero     | done     | done     | exponent | done    | done    | decimal1 | done
    any1     | any1     | any1     | exponent | done    | done    | decimal1 | done
    decimal1 | decimal2 | decimal2 | [error]  | [error] | [error] | [error]  | [error]
    decimal2 | decimal2 | decimal2 | exponent | done    | done    | done     | done
    exponent | any2     | any2     | [error]  | sign    | sign    | [error]  | [error]
    sign     | any2     | any2     | [error]  | [error] | [error] | [error]  | [error]
    any2     | any2     | any2     | done     | done    | done    | done     | done

    The state machine is realized with one label per state (prefixed with
    "scan_number_") and `goto` statements between them. The state machine
    contains cycles, but any cycle can be left when EOF is read. Therefore,
    the function is guaranteed to terminate.

    During scanning, the read bytes are stored in token_buffer. This string is
    then converted to a signed integer, an unsigned integer, or a
    floating-point number.

    @return token_type::value_unsigned, token_type::value_integer, or
            token_type::value_float if number could be successfully scanned,
            token_type::parse_error otherwise

    @note The scanner is independent of the current locale. Internally, the
          locale's decimal point is used instead of `.` to work with the
          locale-dependent converters.
    */
    token_type scan_number()  // lgtm [cpp/use-of-goto] `goto` is used in this function to implement the number-parsing state machine described above. By design, any finite input will eventually reach the "done" state or return token_type::parse_error. In each intermediate state, 1 byte of the input is appended to the token_buffer vector, and only the already initialized variables token_buffer, number_type, and error_message are manipulated.
    {
        // reset token_buffer to store the number's bytes
        reset();

        // the type of the parsed number; initially set to unsigned; will be
        // changed if minus sign, decimal point or exponent is read
        token_type number_type = token_type::value_unsigned;

        // state (init): we just found out we need to scan a number
        switch (current)
        {
            case '-':
            {
                add(current);
                goto scan_number_minus;
            }

            case '0':
            {
                add(current);
                goto scan_number_zero;
            }

            case '1':
            case '2':
            case '3':
            case '4':
            case '5':
            case '6':
            case '7':
            case '8':
            case '9':
            {
                add(current);
                goto scan_number_any1;
            }

            // all other characters are rejected outside scan_number()
            default:            // LCOV_EXCL_LINE
                JSON_ASSERT(false); // NOLINT(cert-dcl03-c,hicpp-static-assert,misc-static-assert) LCOV_EXCL_LINE
        }

scan_number_minus:
        // state: we just parsed a leading minus sign
        number_type = token_type::value_integer;
        switch (get())
        {
            case '0':
            {
                add(current);
                goto scan_number_zero;
            }

            case '1':
            case '2':
            case '3':
            case '4':
            case '5':
            case '6':
            case '7':
            case '8':
            case '9':
            {
                add(current);
                goto scan_number_any1;
            }

            default:
            {
                error_message = "invalid number; expected digit after '-'";
                return token_type::parse_error;
            }
        }

scan_number_zero:
        // state: we just parse a zero (maybe with a leading minus sign)
        switch (get())
        {
            case '.':
            {
                add(decimal_point_char);
                decimal_point_position = token_buffer.size() - 1;
                goto scan_number_decimal1;
            }

            case 'e':
            case 'E':
            {
                add(current);
                goto scan_number_exponent;
            }

            default:
                goto scan_number_done;
        }

scan_number_any1:
        // state: we just parsed a number 0-9 (maybe with a leading minus sign)
        switch (get())
        {
            case '0':
            case '1':
            case '2':
            case '3':
            case '4':
            case '5':
            case '6':
            case '7':
            case '8':
            case '9':
            {
                add(current);
                goto scan_number_any1;
            }

            case '.':
            {
                add(decimal_point_char);
                decimal_point_position = token_buffer.size() - 1;
                goto scan_number_decimal1;
            }

            case 'e':
            case 'E':
            {
                add(current);
                goto scan_number_exponent;
            }

            default:
                goto scan_number_done;
        }

scan_number_decimal1:
        // state: we just parsed a decimal point
        number_type = token_type::value_float;
        switch (get())
        {
            case '0':
            case '1':
            case '2':
            case '3':
            case '4':
            case '5':
            case '6':
            case '7':
            case '8':
            case '9':
            {
                add(current);
                goto scan_number_decimal2;
            }

            default:
            {
                error_message = "invalid number; expected digit after '.'";
                return token_type::parse_error;
            }
        }

scan_number_decimal2:
        // we just parsed at least one number after a decimal point
        switch (get())
        {
            case '0':
            case '1':
            case '2':
            case '3':
            case '4':
            case '5':
            case '6':
            case '7':
            case '8':
            case '9':
            {
                add(current);
                goto scan_number_decimal2;
            }

            case 'e':
            case 'E':
            {
                add(current);
                goto scan_number_exponent;
            }

            default:
                goto scan_number_done;
        }

scan_number_exponent:
        // we just parsed an exponent
        number_type = token_type::value_float;
        switch (get())
        {
            case '+':
            case '-':
            {
                add(current);
                goto scan_number_sign;
            }

            case '0':
            case '1':
            case '2':
            case '3':
            case '4':
            case '5':
            case '6':
            case '7':
            case '8':
            case '9':
            {
                add(current);
                goto scan_number_any2;
            }

            default:
            {
                error_message =
                    "invalid number; expected '+', '-', or digit after exponent";
                return token_type::parse_error;
            }
        }

scan_number_sign:
        // we just parsed an exponent sign
        switch (get())
        {
            case '0':
            case '1':
            case '2':
            case '3':
            case '4':
            case '5':
            case '6':
            case '7':
            case '8':
            case '9':
            {
                add(current);
                goto scan_number_any2;
            }

            default:
            {
                error_message = "invalid number; expected digit after exponent sign";
                return token_type::parse_error;
            }
        }

scan_number_any2:
        // we just parsed a number after the exponent or exponent sign
        switch (get())
        {
            case '0':
            case '1':
            case '2':
            case '3':
            case '4':
            case '5':
            case '6':
            case '7':
            case '8':
            case '9':
            {
                add(current);
                goto scan_number_any2;
            }

            default:
                goto scan_number_done;
        }

scan_number_done:
        // unget the character after the number (we only read it to know that
        // we are done scanning a number)
        unget();

        char* endptr = nullptr; // NOLINT(misc-const-correctness,cppcoreguidelines-pro-type-vararg,hicpp-vararg)
        errno = 0;

        // try to parse integers first and fall back to floats
        if (number_type == token_type::value_unsigned)
        {
            const auto x = std::strtoull(token_buffer.data(), &endptr, 10);

            // we checked the number format before
            JSON_ASSERT(endptr == token_buffer.data() + token_buffer.size());

            if (errno != ERANGE)
            {
                value_unsigned = static_cast<number_unsigned_t>(x);
                if (value_unsigned == x)
                {
                    return token_type::value_unsigned;
                }
            }
        }
        else if (number_type == token_type::value_integer)
        {
            const auto x = std::strtoll(token_buffer.data(), &endptr, 10);

            // we checked the number format before
            JSON_ASSERT(endptr == token_buffer.data() + token_buffer.size());

            if (errno != ERANGE)
            {
                value_integer = static_cast<number_integer_t>(x);
                if (value_integer == x)
                {
                    return token_type::value_integer;
                }
            }
        }

        // this code is reached if we parse a floating-point number or if an
        // integer conversion above failed
        strtof(value_float, token_buffer.data(), &endptr);

        // we checked the number format before
        JSON_ASSERT(endptr == token_buffer.data() + token_buffer.size());

        return token_type::value_float;
    }

    /*!
    @param[in] literal_text  the literal text to expect
    @param[in] length        the length of the passed literal text
    @param[in] return_type   the token type to return on success
    */
    JSON_HEDLEY_NON_NULL(2)
    token_type scan_literal(const char_type* literal_text, const std::size_t length,
                            token_type return_type)
    {
        JSON_ASSERT(char_traits<char_type>::to_char_type(current) == literal_text[0]);
        for (std::size_t i = 1; i < length; ++i)
        {
            if (JSON_HEDLEY_UNLIKELY(char_traits<char_type>::to_char_type(get()) != literal_text[i]))
            {
                error_message = "invalid literal";
                return token_type::parse_error;
            }
        }
        return return_type;
    }

    /////////////////////
    // input management
    /////////////////////

    /// reset token_buffer; current character is beginning of token
    void reset() noexcept
    {
        token_buffer.clear();
        token_string.clear();
        decimal_point_position = std::string::npos;
        token_string.push_back(char_traits<char_type>::to_char_type(current));
    }

    /*
    @brief get next character from the input

    This function provides the interface to the used input adapter. It does
    not throw in case the input reached EOF, but returns a
    `char_traits<char>::eof()` in that case.  Stores the scanned characters
    for use in error messages.

    @return character read from the input
    */
    char_int_type get()
    {
        ++position.chars_read_total;
        ++position.chars_read_current_line;

        if (next_unget)
        {
            // just reset the next_unget variable and work with current
            next_unget = false;
        }
        else
        {
            current = ia.get_character();
        }

        if (JSON_HEDLEY_LIKELY(current != char_traits<char_type>::eof()))
        {
            token_string.push_back(char_traits<char_type>::to_char_type(current));
        }

        if (current == '\n')
        {
            ++position.lines_read;
            position.chars_read_current_line = 0;
        }

        return current;
    }

    /*!
    @brief unget current character (read it again on next get)

    We implement unget by setting variable next_unget to true. The input is not
    changed - we just simulate ungetting by modifying chars_read_total,
    chars_read_current_line, and token_string. The next call to get() will
    behave as if the unget character is read again.
    */
    void unget()
    {
        next_unget = true;

        --position.chars_read_total;

        // in case we "unget" a newline, we have to also decrement the lines_read
        if (position.chars_read_current_line == 0)
        {
            if (position.lines_read > 0)
            {
                --position.lines_read;
            }
        }
        else
        {
            --position.chars_read_current_line;
        }

        if (JSON_HEDLEY_LIKELY(current != char_traits<char_type>::eof()))
        {
            JSON_ASSERT(!token_string.empty());
            token_string.pop_back();
        }
    }

    /// add a character to token_buffer
    void add(char_int_type c)
    {
        token_buffer.push_back(static_cast<typename string_t::value_type>(c));
    }

  public:
    /////////////////////
    // value getters
    /////////////////////

    /// return integer value
    constexpr number_integer_t get_number_integer() const noexcept
    {
        return value_integer;
    }

    /// return unsigned integer value
    constexpr number_unsigned_t get_number_unsigned() const noexcept
    {
        return value_unsigned;
    }

    /// return floating-point value
    constexpr number_float_t get_number_float() const noexcept
    {
        return value_float;
    }

    /// return current string value (implicitly resets the token; useful only once)
    string_t& get_string()
    {
        // translate decimal points from locale back to '.' (#4084)
        if (decimal_point_char != '.' && decimal_point_position != std::string::npos)
        {
            token_buffer[decimal_point_position] = '.';
        }
        return token_buffer;
    }

    /////////////////////
    // diagnostics
    /////////////////////

    /// return position of last read token
    constexpr position_t get_position() const noexcept
    {
        return position;
    }

    /// return the last read token (for errors only).  Will never contain EOF
    /// (an arbitrary value that is not a valid char value, often -1), because
    /// 255 may legitimately occur.  May contain NUL, which should be escaped.
    std::string get_token_string() const
    {
        // escape control characters
        std::string result;
        for (const auto c : token_string)
        {
            if (static_cast<unsigned char>(c) <= '\x1F')
            {
                // escape control characters
                std::array<char, 9> cs{{}};
                static_cast<void>((std::snprintf)(cs.data(), cs.size(), "<U+%.4X>", static_cast<unsigned char>(c))); // NOLINT(cppcoreguidelines-pro-type-vararg,hicpp-vararg)
                result += cs.data();
            }
            else
            {
                // add character as is
                result.push_back(static_cast<std::string::value_type>(c));
            }
        }

        return result;
    }

    /// return syntax error message
    JSON_HEDLEY_RETURNS_NON_NULL
    constexpr const char* get_error_message() const noexcept
    {
        return error_message;
    }

    /////////////////////
    // actual scanner
    /////////////////////

    /*!
    @brief skip the UTF-8 byte order mark
    @return true iff there is no BOM or the correct BOM has been skipped
    */
    bool skip_bom()
    {
        if (get() == 0xEF)
        {
            // check if we completely parse the BOM
            return get() == 0xBB && get() == 0xBF;
        }

        // the first character is not the beginning of the BOM; unget it to
        // process is later
        unget();
        return true;
    }

    void skip_whitespace()
    {
        do
        {
            get();
        }
        while (current == ' ' || current == '\t' || current == '\n' || current == '\r');
    }

    token_type scan()
    {
        // initially, skip the BOM
        if (position.chars_read_total == 0 && !skip_bom())
        {
            error_message = "invalid BOM; must be 0xEF 0xBB 0xBF if given";
            return token_type::parse_error;
        }

        // read next character and ignore whitespace
        skip_whitespace();

        // ignore comments
        while (ignore_comments && current == '/')
        {
            if (!scan_comment())
            {
                return token_type::parse_error;
            }

            // skip following whitespace
            skip_whitespace();
        }

        switch (current)
        {
            // structural characters
            case '[':
                return token_type::begin_array;
            case ']':
                return token_type::end_array;
            case '{':
                return token_type::begin_object;
            case '}':
                return token_type::end_object;
            case ':':
                return token_type::name_separator;
            case ',':
                return token_type::value_separator;

            // literals
            case 't':
            {
                std::array<char_type, 4> true_literal = {{static_cast<char_type>('t'), static_cast<char_type>('r'), static_cast<char_type>('u'), static_cast<char_type>('e')}};
                return scan_literal(true_literal.data(), true_literal.size(), token_type::literal_true);
            }
            case 'f':
            {
                std::array<char_type, 5> false_literal = {{static_cast<char_type>('f'), static_cast<char_type>('a'), static_cast<char_type>('l'), static_cast<char_type>('s'), static_cast<char_type>('e')}};
                return scan_literal(false_literal.data(), false_literal.size(), token_type::literal_false);
            }
            case 'n':
            {
                std::array<char_type, 4> null_literal = {{static_cast<char_type>('n'), static_cast<char_type>('u'), static_cast<char_type>('l'), static_cast<char_type>('l')}};
                return scan_literal(null_literal.data(), null_literal.size(), token_type::literal_null);
            }

            // string
            case '\"':
                return scan_string();

            // number
            case '-':
            case '0':
            case '1':
            case '2':
            case '3':
            case '4':
            case '5':
            case '6':
            case '7':
            case '8':
            case '9':
                return scan_number();

            // end of input (the null byte is needed when parsing from
            // string literals)
            case '\0':
            case char_traits<char_type>::eof():
                return token_type::end_of_input;

            // error
            default:
                error_message = "invalid literal";
                return token_type::parse_error;
        }
    }

  private:
    /// input adapter
    InputAdapterType ia;

    /// whether comments should be ignored (true) or signaled as errors (false)
    const bool ignore_comments = false;

    /// the current character
    char_int_type current = char_traits<char_type>::eof();

    /// whether the next get() call should just return current
    bool next_unget = false;

    /// the start position of the current token
    position_t position {};

    /// raw input token string (for error messages)
    std::vector<char_type> token_string {};

    /// buffer for variable-length tokens (numbers, strings)
    string_t token_buffer {};

    /// a description of occurred lexer errors
    const char* error_message = "";

    // number values
    number_integer_t value_integer = 0;
    number_unsigned_t value_unsigned = 0;
    number_float_t value_float = 0;

    /// the decimal point
    const char_int_type decimal_point_char = '.';
    /// the position of the decimal point in the input
    std::size_t decimal_point_position = std::string::npos;
};

}  // namespace detail
NLOHMANN_JSON_NAMESPACE_END

// #include <nlohmann/detail/macro_scope.hpp>

// #include <nlohmann/detail/string_concat.hpp>

NLOHMANN_JSON_NAMESPACE_BEGIN

/*!
@brief SAX interface

This class describes the SAX interface used by @ref nlohmann::json::sax_parse.
Each function is called in different situations while the input is parsed. The
boolean return value informs the parser whether to continue processing the
input.
*/
template<typename BasicJsonType>
struct json_sax
{
    using number_integer_t = typename BasicJsonType::number_integer_t;
    using number_unsigned_t = typename BasicJsonType::number_unsigned_t;
    using number_float_t = typename BasicJsonType::number_float_t;
    using string_t = typename BasicJsonType::string_t;
    using binary_t = typename BasicJsonType::binary_t;

    /*!
    @brief a null value was read
    @return whether parsing should proceed
    */
    virtual bool null() = 0;

    /*!
    @brief a boolean value was read
    @param[in] val  boolean value
    @return whether parsing should proceed
    */
    virtual bool boolean(bool val) = 0;

    /*!
    @brief an integer number was read
    @param[in] val  integer value
    @return whether parsing should proceed
    */
    virtual bool number_integer(number_integer_t val) = 0;

    /*!
    @brief an unsigned integer number was read
    @param[in] val  unsigned integer value
    @return whether parsing should proceed
    */
    virtual bool number_unsigned(number_unsigned_t val) = 0;

    /*!
    @brief a floating-point number was read
    @param[in] val  floating-point value
    @param[in] s    raw token value
    @return whether parsing should proceed
    */
    virtual bool number_float(number_float_t val, const string_t& s) = 0;

    /*!
    @brief a string value was read
    @param[in] val  string value
    @return whether parsing should proceed
    @note It is safe to move the passed string value.
    */
    virtual bool string(string_t& val) = 0;

    /*!
    @brief a binary value was read
    @param[in] val  binary value
    @return whether parsing should proceed
    @note It is safe to move the passed binary value.
    */
    virtual bool binary(binary_t& val) = 0;

    /*!
    @brief the beginning of an object was read
    @param[in] elements  number of object elements or -1 if unknown
    @return whether parsing should proceed
    @note binary formats may report the number of elements
    */
    virtual bool start_object(std::size_t elements) = 0;

    /*!
    @brief an object key was read
    @param[in] val  object key
    @return whether parsing should proceed
    @note It is safe to move the passed string.
    */
    virtual bool key(string_t& val) = 0;

    /*!
    @brief the end of an object was read
    @return whether parsing should proceed
    */
    virtual bool end_object() = 0;

    /*!
    @brief the beginning of an array was read
    @param[in] elements  number of array elements or -1 if unknown
    @return whether parsing should proceed
    @note binary formats may report the number of elements
    */
    virtual bool start_array(std::size_t elements) = 0;

    /*!
    @brief the end of an array was read
    @return whether parsing should proceed
    */
    virtual bool end_array() = 0;

    /*!
    @brief a parse error occurred
    @param[in] position    the position in the input where the error occurs
    @param[in] last_token  the last read token
    @param[in] ex          an exception object describing the error
    @return whether parsing should proceed (must return false)
    */
    virtual bool parse_error(std::size_t position,
                             const std::string& last_token,
                             const detail::exception& ex) = 0;

    json_sax() = default;
    json_sax(const json_sax&) = default;
    json_sax(json_sax&&) noexcept = default;
    json_sax& operator=(const json_sax&) = default;
    json_sax& operator=(json_sax&&) noexcept = default;
    virtual ~json_sax() = default;
};

namespace detail
{
constexpr std::size_t unknown_size()
{
    return (std::numeric_limits<std::size_t>::max)();
}

/*!
@brief SAX implementation to create a JSON value from SAX events

This class implements the @ref json_sax interface and processes the SAX events
to create a JSON value which makes it basically a DOM parser. The structure or
hierarchy of the JSON value is managed by the stack `ref_stack` which contains
a pointer to the respective array or object for each recursion depth.

After successful parsing, the value that is passed by reference to the
constructor contains the parsed value.

@tparam BasicJsonType  the JSON type
*/
template<typename BasicJsonType, typename InputAdapterType>
class json_sax_dom_parser
{
  public:
    using number_integer_t = typename BasicJsonType::number_integer_t;
    using number_unsigned_t = typename BasicJsonType::number_unsigned_t;
    using number_float_t = typename BasicJsonType::number_float_t;
    using string_t = typename BasicJsonType::string_t;
    using binary_t = typename BasicJsonType::binary_t;
    using lexer_t = lexer<BasicJsonType, InputAdapterType>;

    /*!
    @param[in,out] r  reference to a JSON value that is manipulated while
                       parsing
    @param[in] allow_exceptions_  whether parse errors yield exceptions
    */
    explicit json_sax_dom_parser(BasicJsonType& r, const bool allow_exceptions_ = true, lexer_t* lexer_ = nullptr)
        : root(r), allow_exceptions(allow_exceptions_), m_lexer_ref(lexer_)
    {}

    // make class move-only
    json_sax_dom_parser(const json_sax_dom_parser&) = delete;
    json_sax_dom_parser(json_sax_dom_parser&&) = default; // NOLINT(hicpp-noexcept-move,performance-noexcept-move-constructor)
    json_sax_dom_parser& operator=(const json_sax_dom_parser&) = delete;
    json_sax_dom_parser& operator=(json_sax_dom_parser&&) = default; // NOLINT(hicpp-noexcept-move,performance-noexcept-move-constructor)
    ~json_sax_dom_parser() = default;

    bool null()
    {
        handle_value(nullptr);
        return true;
    }

    bool boolean(bool val)
    {
        handle_value(val);
        return true;
    }

    bool number_integer(number_integer_t val)
    {
        handle_value(val);
        return true;
    }

    bool number_unsigned(number_unsigned_t val)
    {
        handle_value(val);
        return true;
    }

    bool number_float(number_float_t val, const string_t& /*unused*/)
    {
        handle_value(val);
        return true;
    }

    bool string(string_t& val)
    {
        handle_value(val);
        return true;
    }

    bool binary(binary_t& val)
    {
        handle_value(std::move(val));
        return true;
    }

    bool start_object(std::size_t len)
    {
        ref_stack.push_back(handle_value(BasicJsonType::value_t::object));

#if JSON_DIAGNOSTIC_POSITIONS
        // Manually set the start position of the object here.
        // Ensure this is after the call to handle_value to ensure correct start position.
        if (m_lexer_ref)
        {
            // Lexer has read the first character of the object, so
            // subtract 1 from the position to get the correct start position.
            ref_stack.back()->start_position = m_lexer_ref->get_position() - 1;
        }
#endif

        if (JSON_HEDLEY_UNLIKELY(len != detail::unknown_size() && len > ref_stack.back()->max_size()))
        {
            JSON_THROW(out_of_range::create(408, concat("excessive object size: ", std::to_string(len)), ref_stack.back()));
        }

        return true;
    }

    bool key(string_t& val)
    {
        JSON_ASSERT(!ref_stack.empty());
        JSON_ASSERT(ref_stack.back()->is_object());

        // add null at given key and store the reference for later
        object_element = &(ref_stack.back()->m_data.m_value.object->operator[](val));
        return true;
    }

    bool end_object()
    {
        JSON_ASSERT(!ref_stack.empty());
        JSON_ASSERT(ref_stack.back()->is_object());

#if JSON_DIAGNOSTIC_POSITIONS
        if (m_lexer_ref)
        {
            // Lexer's position is past the closing brace, so set that as the end position.
            ref_stack.back()->end_position = m_lexer_ref->get_position();
        }
#endif

        ref_stack.back()->set_parents();
        ref_stack.pop_back();
        return true;
    }

    bool start_array(std::size_t len)
    {
        ref_stack.push_back(handle_value(BasicJsonType::value_t::array));

#if JSON_DIAGNOSTIC_POSITIONS
        // Manually set the start position of the array here.
        // Ensure this is after the call to handle_value to ensure correct start position.
        if (m_lexer_ref)
        {
            ref_stack.back()->start_position = m_lexer_ref->get_position() - 1;
        }
#endif

        if (JSON_HEDLEY_UNLIKELY(len != detail::unknown_size() && len > ref_stack.back()->max_size()))
        {
            JSON_THROW(out_of_range::create(408, concat("excessive array size: ", std::to_string(len)), ref_stack.back()));
        }

        return true;
    }

    bool end_array()
    {
        JSON_ASSERT(!ref_stack.empty());
        JSON_ASSERT(ref_stack.back()->is_array());

#if JSON_DIAGNOSTIC_POSITIONS
        if (m_lexer_ref)
        {
            // Lexer's position is past the closing bracket, so set that as the end position.
            ref_stack.back()->end_position = m_lexer_ref->get_position();
        }
#endif

        ref_stack.back()->set_parents();
        ref_stack.pop_back();
        return true;
    }

    template<class Exception>
    bool parse_error(std::size_t /*unused*/, const std::string& /*unused*/,
                     const Exception& ex)
    {
        errored = true;
        static_cast<void>(ex);
        if (allow_exceptions)
        {
            JSON_THROW(ex);
        }
        return false;
    }

    constexpr bool is_errored() const
    {
        return errored;
    }

  private:

#if JSON_DIAGNOSTIC_POSITIONS
    void handle_diagnostic_positions_for_json_value(BasicJsonType& v)
    {
        if (m_lexer_ref)
        {
            // Lexer has read past the current field value, so set the end position to the current position.
            // The start position will be set below based on the length of the string representation
            // of the value.
            v.end_position = m_lexer_ref->get_position();

            switch (v.type())
            {
                case value_t::boolean:
                {
                    // 4 and 5 are the string length of "true" and "false"
                    v.start_position = v.end_position - (v.m_data.m_value.boolean ? 4 : 5);
                    break;
                }

                case value_t::null:
                {
                    // 4 is the string length of "null"
                    v.start_position = v.end_position - 4;
                    break;
                }

                case value_t::string:
                {
                    // include the length of the quotes, which is 2
                    v.start_position = v.end_position - v.m_data.m_value.string->size() - 2;
                    break;
                }

                // As we handle the start and end positions for values created during parsing,
                // we do not expect the following value type to be called. Regardless, set the positions
                // in case this is created manually or through a different constructor. Exclude from lcov
                // since the exact condition of this switch is esoteric.
                // LCOV_EXCL_START
                case value_t::discarded:
                {
                    v.end_position = std::string::npos;
                    v.start_position = v.end_position;
                    break;
                }
                // LCOV_EXCL_STOP
                case value_t::binary:
                case value_t::number_integer:
                case value_t::number_unsigned:
                case value_t::number_float:
                {
                    v.start_position = v.end_position - m_lexer_ref->get_string().size();
                    break;
                }
                case value_t::object:
                case value_t::array:
                {
                    // object and array are handled in start_object() and start_array() handlers
                    // skip setting the values here.
                    break;
                }
                default: // LCOV_EXCL_LINE
                    // Handle all possible types discretely, default handler should never be reached.
                    JSON_ASSERT(false); // NOLINT(cert-dcl03-c,hicpp-static-assert,misc-static-assert,-warnings-as-errors) LCOV_EXCL_LINE
            }
        }
    }
#endif

    /*!
    @invariant If the ref stack is empty, then the passed value will be the new
               root.
    @invariant If the ref stack contains a value, then it is an array or an
               object to which we can add elements
    */
    template<typename Value>
    JSON_HEDLEY_RETURNS_NON_NULL
    BasicJsonType* handle_value(Value&& v)
    {
        if (ref_stack.empty())
        {
            root = BasicJsonType(std::forward<Value>(v));

#if JSON_DIAGNOSTIC_POSITIONS
            handle_diagnostic_positions_for_json_value(root);
#endif

            return &root;
        }

        JSON_ASSERT(ref_stack.back()->is_array() || ref_stack.back()->is_object());

        if (ref_stack.back()->is_array())
        {
            ref_stack.back()->m_data.m_value.array->emplace_back(std::forward<Value>(v));

#if JSON_DIAGNOSTIC_POSITIONS
            handle_diagnostic_positions_for_json_value(ref_stack.back()->m_data.m_value.array->back());
#endif

            return &(ref_stack.back()->m_data.m_value.array->back());
        }

        JSON_ASSERT(ref_stack.back()->is_object());
        JSON_ASSERT(object_element);
        *object_element = BasicJsonType(std::forward<Value>(v));

#if JSON_DIAGNOSTIC_POSITIONS
        handle_diagnostic_positions_for_json_value(*object_element);
#endif

        return object_element;
    }

    /// the parsed JSON value
    BasicJsonType& root;
    /// stack to model hierarchy of values
    std::vector<BasicJsonType*> ref_stack {};
    /// helper to hold the reference for the next object element
    BasicJsonType* object_element = nullptr;
    /// whether a syntax error occurred
    bool errored = false;
    /// whether to throw exceptions in case of errors
    const bool allow_exceptions = true;
    /// the lexer reference to obtain the current position
    lexer_t* m_lexer_ref = nullptr;
};

template<typename BasicJsonType, typename InputAdapterType>
class json_sax_dom_callback_parser
{
  public:
    using number_integer_t = typename BasicJsonType::number_integer_t;
    using number_unsigned_t = typename BasicJsonType::number_unsigned_t;
    using number_float_t = typename BasicJsonType::number_float_t;
    using string_t = typename BasicJsonType::string_t;
    using binary_t = typename BasicJsonType::binary_t;
    using parser_callback_t = typename BasicJsonType::parser_callback_t;
    using parse_event_t = typename BasicJsonType::parse_event_t;
    using lexer_t = lexer<BasicJsonType, InputAdapterType>;

    json_sax_dom_callback_parser(BasicJsonType& r,
                                 parser_callback_t cb,
                                 const bool allow_exceptions_ = true,
                                 lexer_t* lexer_ = nullptr)
        : root(r), callback(std::move(cb)), allow_exceptions(allow_exceptions_), m_lexer_ref(lexer_)
    {
        keep_stack.push_back(true);
    }

    // make class move-only
    json_sax_dom_callback_parser(const json_sax_dom_callback_parser&) = delete;
    json_sax_dom_callback_parser(json_sax_dom_callback_parser&&) = default; // NOLINT(hicpp-noexcept-move,performance-noexcept-move-constructor)
    json_sax_dom_callback_parser& operator=(const json_sax_dom_callback_parser&) = delete;
    json_sax_dom_callback_parser& operator=(json_sax_dom_callback_parser&&) = default; // NOLINT(hicpp-noexcept-move,performance-noexcept-move-constructor)
    ~json_sax_dom_callback_parser() = default;

    bool null()
    {
        handle_value(nullptr);
        return true;
    }

    bool boolean(bool val)
    {
        handle_value(val);
        return true;
    }

    bool number_integer(number_integer_t val)
    {
        handle_value(val);
        return true;
    }

    bool number_unsigned(number_unsigned_t val)
    {
        handle_value(val);
        return true;
    }

    bool number_float(number_float_t val, const string_t& /*unused*/)
    {
        handle_value(val);
        return true;
    }

    bool string(string_t& val)
    {
        handle_value(val);
        return true;
    }

    bool binary(binary_t& val)
    {
        handle_value(std::move(val));
        return true;
    }

    bool start_object(std::size_t len)
    {
        // check callback for object start
        const bool keep = callback(static_cast<int>(ref_stack.size()), parse_event_t::object_start, discarded);
        keep_stack.push_back(keep);

        auto val = handle_value(BasicJsonType::value_t::object, true);
        ref_stack.push_back(val.second);

        if (ref_stack.back())
        {

#if JSON_DIAGNOSTIC_POSITIONS
            // Manually set the start position of the object here.
            // Ensure this is after the call to handle_value to ensure correct start position.
            if (m_lexer_ref)
            {
                // Lexer has read the first character of the object, so
                // subtract 1 from the position to get the correct start position.
                ref_stack.back()->start_position = m_lexer_ref->get_position() - 1;
            }
#endif

            // check object limit
            if (JSON_HEDLEY_UNLIKELY(len != detail::unknown_size() && len > ref_stack.back()->max_size()))
            {
                JSON_THROW(out_of_range::create(408, concat("excessive object size: ", std::to_string(len)), ref_stack.back()));
            }
        }
        return true;
    }

    bool key(string_t& val)
    {
        BasicJsonType k = BasicJsonType(val);

        // check callback for key
        const bool keep = callback(static_cast<int>(ref_stack.size()), parse_event_t::key, k);
        key_keep_stack.push_back(keep);

        // add discarded value at given key and store the reference for later
        if (keep && ref_stack.back())
        {
            object_element = &(ref_stack.back()->m_data.m_value.object->operator[](val) = discarded);
        }

        return true;
    }

    bool end_object()
    {
        if (ref_stack.back())
        {
            if (!callback(static_cast<int>(ref_stack.size()) - 1, parse_event_t::object_end, *ref_stack.back()))
            {
                // discard object
                *ref_stack.back() = discarded;

#if JSON_DIAGNOSTIC_POSITIONS
                // Set start/end positions for discarded object.
                handle_diagnostic_positions_for_json_value(*ref_stack.back());
#endif
            }
            else
            {

#if JSON_DIAGNOSTIC_POSITIONS
                if (m_lexer_ref)
                {
                    // Lexer's position is past the closing brace, so set that as the end position.
                    ref_stack.back()->end_position = m_lexer_ref->get_position();
                }
#endif

                ref_stack.back()->set_parents();
            }
        }

        JSON_ASSERT(!ref_stack.empty());
        JSON_ASSERT(!keep_stack.empty());
        ref_stack.pop_back();
        keep_stack.pop_back();

        if (!ref_stack.empty() && ref_stack.back() && ref_stack.back()->is_structured())
        {
            // remove discarded value
            for (auto it = ref_stack.back()->begin(); it != ref_stack.back()->end(); ++it)
            {
                if (it->is_discarded())
                {
                    ref_stack.back()->erase(it);
                    break;
                }
            }
        }

        return true;
    }

    bool start_array(std::size_t len)
    {
        const bool keep = callback(static_cast<int>(ref_stack.size()), parse_event_t::array_start, discarded);
        keep_stack.push_back(keep);

        auto val = handle_value(BasicJsonType::value_t::array, true);
        ref_stack.push_back(val.second);

        if (ref_stack.back())
        {

#if JSON_DIAGNOSTIC_POSITIONS
            // Manually set the start position of the array here.
            // Ensure this is after the call to handle_value to ensure correct start position.
            if (m_lexer_ref)
            {
                // Lexer has read the first character of the array, so
                // subtract 1 from the position to get the correct start position.
                ref_stack.back()->start_position = m_lexer_ref->get_position() - 1;
            }
#endif

            // check array limit
            if (JSON_HEDLEY_UNLIKELY(len != detail::unknown_size() && len > ref_stack.back()->max_size()))
            {
                JSON_THROW(out_of_range::create(408, concat("excessive array size: ", std::to_string(len)), ref_stack.back()));
            }
        }

        return true;
    }

    bool end_array()
    {
        bool keep = true;

        if (ref_stack.back())
        {
            keep = callback(static_cast<int>(ref_stack.size()) - 1, parse_event_t::array_end, *ref_stack.back());
            if (keep)
            {

#if JSON_DIAGNOSTIC_POSITIONS
                if (m_lexer_ref)
                {
                    // Lexer's position is past the closing bracket, so set that as the end position.
                    ref_stack.back()->end_position = m_lexer_ref->get_position();
                }
#endif

                ref_stack.back()->set_parents();
            }
            else
            {
                // discard array
                *ref_stack.back() = discarded;

#if JSON_DIAGNOSTIC_POSITIONS
                // Set start/end positions for discarded array.
                handle_diagnostic_positions_for_json_value(*ref_stack.back());
#endif
            }
        }

        JSON_ASSERT(!ref_stack.empty());
        JSON_ASSERT(!keep_stack.empty());
        ref_stack.pop_back();
        keep_stack.pop_back();

        // remove discarded value
        if (!keep && !ref_stack.empty() && ref_stack.back()->is_array())
        {
            ref_stack.back()->m_data.m_value.array->pop_back();
        }

        return true;
    }

    template<class Exception>
    bool parse_error(std::size_t /*unused*/, const std::string& /*unused*/,
                     const Exception& ex)
    {
        errored = true;
        static_cast<void>(ex);
        if (allow_exceptions)
        {
            JSON_THROW(ex);
        }
        return false;
    }

    constexpr bool is_errored() const
    {
        return errored;
    }

  private:

#if JSON_DIAGNOSTIC_POSITIONS
    void handle_diagnostic_positions_for_json_value(BasicJsonType& v)
    {
        if (m_lexer_ref)
        {
            // Lexer has read past the current field value, so set the end position to the current position.
            // The start position will be set below based on the length of the string representation
            // of the value.
            v.end_position = m_lexer_ref->get_position();

            switch (v.type())
            {
                case value_t::boolean:
                {
                    // 4 and 5 are the string length of "true" and "false"
                    v.start_position = v.end_position - (v.m_data.m_value.boolean ? 4 : 5);
                    break;
                }

                case value_t::null:
                {
                    // 4 is the string length of "null"
                    v.start_position = v.end_position - 4;
                    break;
                }

                case value_t::string:
                {
                    // include the length of the quotes, which is 2
                    v.start_position = v.end_position - v.m_data.m_value.string->size() - 2;
                    break;
                }

                case value_t::discarded:
                {
                    v.end_position = std::string::npos;
                    v.start_position = v.end_position;
                    break;
                }

                case value_t::binary:
                case value_t::number_integer:
                case value_t::number_unsigned:
                case value_t::number_float:
                {
                    v.start_position = v.end_position - m_lexer_ref->get_string().size();
                    break;
                }

                case value_t::object:
                case value_t::array:
                {
                    // object and array are handled in start_object() and start_array() handlers
                    // skip setting the values here.
                    break;
                }
                default: // LCOV_EXCL_LINE
                    // Handle all possible types discretely, default handler should never be reached.
                    JSON_ASSERT(false); // NOLINT(cert-dcl03-c,hicpp-static-assert,misc-static-assert,-warnings-as-errors) LCOV_EXCL_LINE
            }
        }
    }
#endif

    /*!
    @param[in] v  value to add to the JSON value we build during parsing
    @param[in] skip_callback  whether we should skip calling the callback
               function; this is required after start_array() and
               start_object() SAX events, because otherwise we would call the
               callback function with an empty array or object, respectively.

    @invariant If the ref stack is empty, then the passed value will be the new
               root.
    @invariant If the ref stack contains a value, then it is an array or an
               object to which we can add elements

    @return pair of boolean (whether value should be kept) and pointer (to the
            passed value in the ref_stack hierarchy; nullptr if not kept)
    */
    template<typename Value>
    std::pair<bool, BasicJsonType*> handle_value(Value&& v, const bool skip_callback = false)
    {
        JSON_ASSERT(!keep_stack.empty());

        // do not handle this value if we know it would be added to a discarded
        // container
        if (!keep_stack.back())
        {
            return {false, nullptr};
        }

        // create value
        auto value = BasicJsonType(std::forward<Value>(v));

#if JSON_DIAGNOSTIC_POSITIONS
        handle_diagnostic_positions_for_json_value(value);
#endif

        // check callback
        const bool keep = skip_callback || callback(static_cast<int>(ref_stack.size()), parse_event_t::value, value);

        // do not handle this value if we just learnt it shall be discarded
        if (!keep)
        {
            return {false, nullptr};
        }

        if (ref_stack.empty())
        {
            root = std::move(value);
            return {true, & root};
        }

        // skip this value if we already decided to skip the parent
        // (https://github.com/nlohmann/json/issues/971#issuecomment-413678360)
        if (!ref_stack.back())
        {
            return {false, nullptr};
        }

        // we now only expect arrays and objects
        JSON_ASSERT(ref_stack.back()->is_array() || ref_stack.back()->is_object());

        // array
        if (ref_stack.back()->is_array())
        {
            ref_stack.back()->m_data.m_value.array->emplace_back(std::move(value));
            return {true, & (ref_stack.back()->m_data.m_value.array->back())};
        }

        // object
        JSON_ASSERT(ref_stack.back()->is_object());
        // check if we should store an element for the current key
        JSON_ASSERT(!key_keep_stack.empty());
        const bool store_element = key_keep_stack.back();
        key_keep_stack.pop_back();

        if (!store_element)
        {
            return {false, nullptr};
        }

        JSON_ASSERT(object_element);
        *object_element = std::move(value);
        return {true, object_element};
    }

    /// the parsed JSON value
    BasicJsonType& root;
    /// stack to model hierarchy of values
    std::vector<BasicJsonType*> ref_stack {};
    /// stack to manage which values to keep
    std::vector<bool> keep_stack {}; // NOLINT(readability-redundant-member-init)
    /// stack to manage which object keys to keep
    std::vector<bool> key_keep_stack {}; // NOLINT(readability-redundant-member-init)
    /// helper to hold the reference for the next object element
    BasicJsonType* object_element = nullptr;
    /// whether a syntax error occurred
    bool errored = false;
    /// callback function
    const parser_callback_t callback = nullptr;
    /// whether to throw exceptions in case of errors
    const bool allow_exceptions = true;
    /// a discarded value for the callback
    BasicJsonType discarded = BasicJsonType::value_t::discarded;
    /// the lexer reference to obtain the current position
    lexer_t* m_lexer_ref = nullptr;
};

template<typename BasicJsonType>
class json_sax_acceptor
{
  public:
    using number_integer_t = typename BasicJsonType::number_integer_t;
    using number_unsigned_t = typename BasicJsonType::number_unsigned_t;
    using number_float_t = typename BasicJsonType::number_float_t;
    using string_t = typename BasicJsonType::string_t;
    using binary_t = typename BasicJsonType::binary_t;

    bool null()
    {
        return true;
    }

    bool boolean(bool /*unused*/)
    {
        return true;
    }

    bool number_integer(number_integer_t /*unused*/)
    {
        return true;
    }

    bool number_unsigned(number_unsigned_t /*unused*/)
    {
        return true;
    }

    bool number_float(number_float_t /*unused*/, const string_t& /*unused*/)
    {
        return true;
    }

    bool string(string_t& /*unused*/)
    {
        return true;
    }

    bool binary(binary_t& /*unused*/)
    {
        return true;
    }

    bool start_object(std::size_t /*unused*/ = detail::unknown_size())
    {
        return true;
    }

    bool key(string_t& /*unused*/)
    {
        return true;
    }

    bool end_object()
    {
        return true;
    }

    bool start_array(std::size_t /*unused*/ = detail::unknown_size())
    {
        return true;
    }

    bool end_array()
    {
        return true;
    }

    bool parse_error(std::size_t /*unused*/, const std::string& /*unused*/, const detail::exception& /*unused*/)
    {
        return false;
    }
};

}  // namespace detail
NLOHMANN_JSON_NAMESPACE_END

// #include <nlohmann/detail/input/lexer.hpp>

// #include <nlohmann/detail/macro_scope.hpp>

// #include <nlohmann/detail/meta/is_sax.hpp>
//     __ _____ _____ _____
//  __|  |   __|     |   | |  JSON for Modern C++
// |  |  |__   |  |  | | | |  version 3.12.0
// |_____|_____|_____|_|___|  https://github.com/nlohmann/json
//
// SPDX-FileCopyrightText: 2013 - 2025 Niels Lohmann <https://nlohmann.me>
// SPDX-License-Identifier: MIT



#include <cstdint> // size_t
#include <utility> // declval
#include <string> // string

// #include <nlohmann/detail/abi_macros.hpp>

// #include <nlohmann/detail/meta/detected.hpp>

// #include <nlohmann/detail/meta/type_traits.hpp>


NLOHMANN_JSON_NAMESPACE_BEGIN
namespace detail
{

template<typename T>
using null_function_t = decltype(std::declval<T&>().null());

template<typename T>
using boolean_function_t =
    decltype(std::declval<T&>().boolean(std::declval<bool>()));

template<typename T, typename Integer>
using number_integer_function_t =
    decltype(std::declval<T&>().number_integer(std::declval<Integer>()));

template<typename T, typename Unsigned>
using number_unsigned_function_t =
    decltype(std::declval<T&>().number_unsigned(std::declval<Unsigned>()));

template<typename T, typename Float, typename String>
using number_float_function_t = decltype(std::declval<T&>().number_float(
                                    std::declval<Float>(), std::declval<const String&>()));

template<typename T, typename String>
using string_function_t =
    decltype(std::declval<T&>().string(std::declval<String&>()));

template<typename T, typename Binary>
using binary_function_t =
    decltype(std::declval<T&>().binary(std::declval<Binary&>()));

template<typename T>
using start_object_function_t =
    decltype(std::declval<T&>().start_object(std::declval<std::size_t>()));

template<typename T, typename String>
using key_function_t =
    decltype(std::declval<T&>().key(std::declval<String&>()));

template<typename T>
using end_object_function_t = decltype(std::declval<T&>().end_object());

template<typename T>
using start_array_function_t =
    decltype(std::declval<T&>().start_array(std::declval<std::size_t>()));

template<typename T>
using end_array_function_t = decltype(std::declval<T&>().end_array());

template<typename T, typename Exception>
using parse_error_function_t = decltype(std::declval<T&>().parse_error(
        std::declval<std::size_t>(), std::declval<const std::string&>(),
        std::declval<const Exception&>()));

template<typename SAX, typename BasicJsonType>
struct is_sax
{
  private:
    static_assert(is_basic_json<BasicJsonType>::value,
                  "BasicJsonType must be of type basic_json<...>");

    using number_integer_t = typename BasicJsonType::number_integer_t;
    using number_unsigned_t = typename BasicJsonType::number_unsigned_t;
    using number_float_t = typename BasicJsonType::number_float_t;
    using string_t = typename BasicJsonType::string_t;
    using binary_t = typename BasicJsonType::binary_t;
    using exception_t = typename BasicJsonType::exception;

  public:
    static constexpr bool value =
        is_detected_exact<bool, null_function_t, SAX>::value &&
        is_detected_exact<bool, boolean_function_t, SAX>::value &&
        is_detected_exact<bool, number_integer_function_t, SAX, number_integer_t>::value &&
        is_detected_exact<bool, number_unsigned_function_t, SAX, number_unsigned_t>::value &&
        is_detected_exact<bool, number_float_function_t, SAX, number_float_t, string_t>::value &&
        is_detected_exact<bool, string_function_t, SAX, string_t>::value &&
        is_detected_exact<bool, binary_function_t, SAX, binary_t>::value &&
        is_detected_exact<bool, start_object_function_t, SAX>::value &&
        is_detected_exact<bool, key_function_t, SAX, string_t>::value &&
        is_detected_exact<bool, end_object_function_t, SAX>::value &&
        is_detected_exact<bool, start_array_function_t, SAX>::value &&
        is_detected_exact<bool, end_array_function_t, SAX>::value &&
        is_detected_exact<bool, parse_error_function_t, SAX, exception_t>::value;
};

template<typename SAX, typename BasicJsonType>
struct is_sax_static_asserts
{
  private:
    static_assert(is_basic_json<BasicJsonType>::value,
                  "BasicJsonType must be of type basic_json<...>");

    using number_integer_t = typename BasicJsonType::number_integer_t;
    using number_unsigned_t = typename BasicJsonType::number_unsigned_t;
    using number_float_t = typename BasicJsonType::number_float_t;
    using string_t = typename BasicJsonType::string_t;
    using binary_t = typename BasicJsonType::binary_t;
    using exception_t = typename BasicJsonType::exception;

  public:
    static_assert(is_detected_exact<bool, null_function_t, SAX>::value,
                  "Missing/invalid function: bool null()");
    static_assert(is_detected_exact<bool, boolean_function_t, SAX>::value,
                  "Missing/invalid function: bool boolean(bool)");
    static_assert(is_detected_exact<bool, boolean_function_t, SAX>::value,
                  "Missing/invalid function: bool boolean(bool)");
    static_assert(
        is_detected_exact<bool, number_integer_function_t, SAX,
        number_integer_t>::value,
        "Missing/invalid function: bool number_integer(number_integer_t)");
    static_assert(
        is_detected_exact<bool, number_unsigned_function_t, SAX,
        number_unsigned_t>::value,
        "Missing/invalid function: bool number_unsigned(number_unsigned_t)");
    static_assert(is_detected_exact<bool, number_float_function_t, SAX,
                  number_float_t, string_t>::value,
                  "Missing/invalid function: bool number_float(number_float_t, const string_t&)");
    static_assert(
        is_detected_exact<bool, string_function_t, SAX, string_t>::value,
        "Missing/invalid function: bool string(string_t&)");
    static_assert(
        is_detected_exact<bool, binary_function_t, SAX, binary_t>::value,
        "Missing/invalid function: bool binary(binary_t&)");
    static_assert(is_detected_exact<bool, start_object_function_t, SAX>::value,
                  "Missing/invalid function: bool start_object(std::size_t)");
    static_assert(is_detected_exact<bool, key_function_t, SAX, string_t>::value,
                  "Missing/invalid function: bool key(string_t&)");
    static_assert(is_detected_exact<bool, end_object_function_t, SAX>::value,
                  "Missing/invalid function: bool end_object()");
    static_assert(is_detected_exact<bool, start_array_function_t, SAX>::value,
                  "Missing/invalid function: bool start_array(std::size_t)");
    static_assert(is_detected_exact<bool, end_array_function_t, SAX>::value,
                  "Missing/invalid function: bool end_array()");
    static_assert(
        is_detected_exact<bool, parse_error_function_t, SAX, exception_t>::value,
        "Missing/invalid function: bool parse_error(std::size_t, const "
        "std::string&, const exception&)");
};

}  // namespace detail
NLOHMANN_JSON_NAMESPACE_END

// #include <nlohmann/detail/meta/type_traits.hpp>

// #include <nlohmann/detail/string_concat.hpp>

// #include <nlohmann/detail/value_t.hpp>


NLOHMANN_JSON_NAMESPACE_BEGIN
namespace detail
{

/// how to treat CBOR tags
enum class cbor_tag_handler_t
{
    error,   ///< throw a parse_error exception in case of a tag
    ignore,  ///< ignore tags
    store    ///< store tags as binary type
};

/*!
@brief determine system byte order

@return true if and only if system's byte order is little endian

@note from https://stackoverflow.com/a/1001328/266378
*/
static inline bool little_endianness(int num = 1) noexcept
{
    return *reinterpret_cast<char*>(&num) == 1;
}

///////////////////
// binary reader //
///////////////////

/*!
@brief deserialization of CBOR, MessagePack, and UBJSON values
*/
template<typename BasicJsonType, typename InputAdapterType, typename SAX = json_sax_dom_parser<BasicJsonType, InputAdapterType>>
class binary_reader
{
    using number_integer_t = typename BasicJsonType::number_integer_t;
    using number_unsigned_t = typename BasicJsonType::number_unsigned_t;
    using number_float_t = typename BasicJsonType::number_float_t;
    using string_t = typename BasicJsonType::string_t;
    using binary_t = typename BasicJsonType::binary_t;
    using json_sax_t = SAX;
    using char_type = typename InputAdapterType::char_type;
    using char_int_type = typename char_traits<char_type>::int_type;

  public:
    /*!
    @brief create a binary reader

    @param[in] adapter  input adapter to read from
    */
    explicit binary_reader(InputAdapterType&& adapter, const input_format_t format = input_format_t::json) noexcept : ia(std::move(adapter)), input_format(format)
    {
        (void)detail::is_sax_static_asserts<SAX, BasicJsonType> {};
    }

    // make class move-only
    binary_reader(const binary_reader&) = delete;
    binary_reader(binary_reader&&) = default; // NOLINT(hicpp-noexcept-move,performance-noexcept-move-constructor)
    binary_reader& operator=(const binary_reader&) = delete;
    binary_reader& operator=(binary_reader&&) = default; // NOLINT(hicpp-noexcept-move,performance-noexcept-move-constructor)
    ~binary_reader() = default;

    /*!
    @param[in] format  the binary format to parse
    @param[in] sax_    a SAX event processor
    @param[in] strict  whether to expect the input to be consumed completed
    @param[in] tag_handler  how to treat CBOR tags

    @return whether parsing was successful
    */
    JSON_HEDLEY_NON_NULL(3)
    bool sax_parse(const input_format_t format,
                   json_sax_t* sax_,
                   const bool strict = true,
                   const cbor_tag_handler_t tag_handler = cbor_tag_handler_t::error)
    {
        sax = sax_;
        bool result = false;

        switch (format)
        {
            case input_format_t::bson:
                result = parse_bson_internal();
                break;

            case input_format_t::cbor:
                result = parse_cbor_internal(true, tag_handler);
                break;

            case input_format_t::msgpack:
                result = parse_msgpack_internal();
                break;

            case input_format_t::ubjson:
            case input_format_t::bjdata:
                result = parse_ubjson_internal();
                break;

            case input_format_t::json: // LCOV_EXCL_LINE
            default:            // LCOV_EXCL_LINE
                JSON_ASSERT(false); // NOLINT(cert-dcl03-c,hicpp-static-assert,misc-static-assert) LCOV_EXCL_LINE
        }

        // strict mode: next byte must be EOF
        if (result && strict)
        {
            if (input_format == input_format_t::ubjson || input_format == input_format_t::bjdata)
            {
                get_ignore_noop();
            }
            else
            {
                get();
            }

            if (JSON_HEDLEY_UNLIKELY(current != char_traits<char_type>::eof()))
            {
                return sax->parse_error(chars_read, get_token_string(), parse_error::create(110, chars_read,
                                        exception_message(input_format, concat("expected end of input; last byte: 0x", get_token_string()), "value"), nullptr));
            }
        }

        return result;
    }

  private:
    //////////
    // BSON //
    //////////

    /*!
    @brief Reads in a BSON-object and passes it to the SAX-parser.
    @return whether a valid BSON-value was passed to the SAX parser
    */
    bool parse_bson_internal()
    {
        std::int32_t document_size{};
        get_number<std::int32_t, true>(input_format_t::bson, document_size);

        if (JSON_HEDLEY_UNLIKELY(!sax->start_object(detail::unknown_size())))
        {
            return false;
        }

        if (JSON_HEDLEY_UNLIKELY(!parse_bson_element_list(/*is_array*/false)))
        {
            return false;
        }

        return sax->end_object();
    }

    /*!
    @brief Parses a C-style string from the BSON input.
    @param[in,out] result  A reference to the string variable where the read
                            string is to be stored.
    @return `true` if the \x00-byte indicating the end of the string was
             encountered before the EOF; false` indicates an unexpected EOF.
    */
    bool get_bson_cstr(string_t& result)
    {
        auto out = std::back_inserter(result);
        while (true)
        {
            get();
            if (JSON_HEDLEY_UNLIKELY(!unexpect_eof(input_format_t::bson, "cstring")))
            {
                return false;
            }
            if (current == 0x00)
            {
                return true;
            }
            *out++ = static_cast<typename string_t::value_type>(current);
        }
    }

    /*!
    @brief Parses a zero-terminated string of length @a len from the BSON
           input.
    @param[in] len  The length (including the zero-byte at the end) of the
                    string to be read.
    @param[in,out] result  A reference to the string variable where the read
                            string is to be stored.
    @tparam NumberType The type of the length @a len
    @pre len >= 1
    @return `true` if the string was successfully parsed
    */
    template<typename NumberType>
    bool get_bson_string(const NumberType len, string_t& result)
    {
        if (JSON_HEDLEY_UNLIKELY(len < 1))
        {
            auto last_token = get_token_string();
            return sax->parse_error(chars_read, last_token, parse_error::create(112, chars_read,
                                    exception_message(input_format_t::bson, concat("string length must be at least 1, is ", std::to_string(len)), "string"), nullptr));
        }

        return get_string(input_format_t::bson, len - static_cast<NumberType>(1), result) && get() != char_traits<char_type>::eof();
    }

    /*!
    @brief Parses a byte array input of length @a len from the BSON input.
    @param[in] len  The length of the byte array to be read.
    @param[in,out] result  A reference to the binary variable where the read
                            array is to be stored.
    @tparam NumberType The type of the length @a len
    @pre len >= 0
    @return `true` if the byte array was successfully parsed
    */
    template<typename NumberType>
    bool get_bson_binary(const NumberType len, binary_t& result)
    {
        if (JSON_HEDLEY_UNLIKELY(len < 0))
        {
            auto last_token = get_token_string();
            return sax->parse_error(chars_read, last_token, parse_error::create(112, chars_read,
                                    exception_message(input_format_t::bson, concat("byte array length cannot be negative, is ", std::to_string(len)), "binary"), nullptr));
        }

        // All BSON binary values have a subtype
        std::uint8_t subtype{};
        get_number<std::uint8_t>(input_format_t::bson, subtype);
        result.set_subtype(subtype);

        return get_binary(input_format_t::bson, len, result);
    }

    /*!
    @brief Read a BSON document element of the given @a element_type.
    @param[in] element_type The BSON element type, c.f. http://bsonspec.org/spec.html
    @param[in] element_type_parse_position The position in the input stream,
               where the `element_type` was read.
    @warning Not all BSON element types are supported yet. An unsupported
             @a element_type will give rise to a parse_error.114:
             Unsupported BSON record type 0x...
    @return whether a valid BSON-object/array was passed to the SAX parser
    */
    bool parse_bson_element_internal(const char_int_type element_type,
                                     const std::size_t element_type_parse_position)
    {
        switch (element_type)
        {
            case 0x01: // double
            {
                double number{};
                return get_number<double, true>(input_format_t::bson, number) && sax->number_float(static_cast<number_float_t>(number), "");
            }

            case 0x02: // string
            {
                std::int32_t len{};
                string_t value;
                return get_number<std::int32_t, true>(input_format_t::bson, len) && get_bson_string(len, value) && sax->string(value);
            }

            case 0x03: // object
            {
                return parse_bson_internal();
            }

            case 0x04: // array
            {
                return parse_bson_array();
            }

            case 0x05: // binary
            {
                std::int32_t len{};
                binary_t value;
                return get_number<std::int32_t, true>(input_format_t::bson, len) && get_bson_binary(len, value) && sax->binary(value);
            }

            case 0x08: // boolean
            {
                return sax->boolean(get() != 0);
            }

            case 0x0A: // null
            {
                return sax->null();
            }

            case 0x10: // int32
            {
                std::int32_t value{};
                return get_number<std::int32_t, true>(input_format_t::bson, value) && sax->number_integer(value);
            }

            case 0x12: // int64
            {
                std::int64_t value{};
                return get_number<std::int64_t, true>(input_format_t::bson, value) && sax->number_integer(value);
            }

            case 0x11: // uint64
            {
                std::uint64_t value{};
                return get_number<std::uint64_t, true>(input_format_t::bson, value) && sax->number_unsigned(value);
            }

            default: // anything else not supported (yet)
            {
                std::array<char, 3> cr{{}};
                static_cast<void>((std::snprintf)(cr.data(), cr.size(), "%.2hhX", static_cast<unsigned char>(element_type))); // NOLINT(cppcoreguidelines-pro-type-vararg,hicpp-vararg)
                const std::string cr_str{cr.data()};
                return sax->parse_error(element_type_parse_position, cr_str,
                                        parse_error::create(114, element_type_parse_position, concat("Unsupported BSON record type 0x", cr_str), nullptr));
            }
        }
    }

    /*!
    @brief Read a BSON element list (as specified in the BSON-spec)

    The same binary layout is used for objects and arrays, hence it must be
    indicated with the argument @a is_array which one is expected
    (true --> array, false --> object).

    @param[in] is_array Determines if the element list being read is to be
                        treated as an object (@a is_array == false), or as an
                        array (@a is_array == true).
    @return whether a valid BSON-object/array was passed to the SAX parser
    */
    bool parse_bson_element_list(const bool is_array)
    {
        string_t key;

        while (auto element_type = get())
        {
            if (JSON_HEDLEY_UNLIKELY(!unexpect_eof(input_format_t::bson, "element list")))
            {
                return false;
            }

            const std::size_t element_type_parse_position = chars_read;
            if (JSON_HEDLEY_UNLIKELY(!get_bson_cstr(key)))
            {
                return false;
            }

            if (!is_array && !sax->key(key))
            {
                return false;
            }

            if (JSON_HEDLEY_UNLIKELY(!parse_bson_element_internal(element_type, element_type_parse_position)))
            {
                return false;
            }

            // get_bson_cstr only appends
            key.clear();
        }

        return true;
    }

    /*!
    @brief Reads an array from the BSON input and passes it to the SAX-parser.
    @return whether a valid BSON-array was passed to the SAX parser
    */
    bool parse_bson_array()
    {
        std::int32_t document_size{};
        get_number<std::int32_t, true>(input_format_t::bson, document_size);

        if (JSON_HEDLEY_UNLIKELY(!sax->start_array(detail::unknown_size())))
        {
            return false;
        }

        if (JSON_HEDLEY_UNLIKELY(!parse_bson_element_list(/*is_array*/true)))
        {
            return false;
        }

        return sax->end_array();
    }

    //////////
    // CBOR //
    //////////

    /*!
    @param[in] get_char  whether a new character should be retrieved from the
                         input (true) or whether the last read character should
                         be considered instead (false)
    @param[in] tag_handler how CBOR tags should be treated

    @return whether a valid CBOR value was passed to the SAX parser
    */
    bool parse_cbor_internal(const bool get_char,
                             const cbor_tag_handler_t tag_handler)
    {
        switch (get_char ? get() : current)
        {
            // EOF
            case char_traits<char_type>::eof():
                return unexpect_eof(input_format_t::cbor, "value");

            // Integer 0x00..0x17 (0..23)
            case 0x00:
            case 0x01:
            case 0x02:
            case 0x03:
            case 0x04:
            case 0x05:
            case 0x06:
            case 0x07:
            case 0x08:
            case 0x09:
            case 0x0A:
            case 0x0B:
            case 0x0C:
            case 0x0D:
            case 0x0E:
            case 0x0F:
            case 0x10:
            case 0x11:
            case 0x12:
            case 0x13:
            case 0x14:
            case 0x15:
            case 0x16:
            case 0x17:
                return sax->number_unsigned(static_cast<number_unsigned_t>(current));

            case 0x18: // Unsigned integer (one-byte uint8_t follows)
            {
                std::uint8_t number{};
                return get_number(input_format_t::cbor, number) && sax->number_unsigned(number);
            }

            case 0x19: // Unsigned integer (two-byte uint16_t follows)
            {
                std::uint16_t number{};
                return get_number(input_format_t::cbor, number) && sax->number_unsigned(number);
            }

            case 0x1A: // Unsigned integer (four-byte uint32_t follows)
            {
                std::uint32_t number{};
                return get_number(input_format_t::cbor, number) && sax->number_unsigned(number);
            }

            case 0x1B: // Unsigned integer (eight-byte uint64_t follows)
            {
                std::uint64_t number{};
                return get_number(input_format_t::cbor, number) && sax->number_unsigned(number);
            }

            // Negative integer -1-0x00..-1-0x17 (-1..-24)
            case 0x20:
            case 0x21:
            case 0x22:
            case 0x23:
            case 0x24:
            case 0x25:
            case 0x26:
            case 0x27:
            case 0x28:
            case 0x29:
            case 0x2A:
            case 0x2B:
            case 0x2C:
            case 0x2D:
            case 0x2E:
            case 0x2F:
            case 0x30:
            case 0x31:
            case 0x32:
            case 0x33:
            case 0x34:
            case 0x35:
            case 0x36:
            case 0x37:
                return sax->number_integer(static_cast<std::int8_t>(0x20 - 1 - current));

            case 0x38: // Negative integer (one-byte uint8_t follows)
            {
                std::uint8_t number{};
                return get_number(input_format_t::cbor, number) && sax->number_integer(static_cast<number_integer_t>(-1) - number);
            }

            case 0x39: // Negative integer -1-n (two-byte uint16_t follows)
            {
                std::uint16_t number{};
                return get_number(input_format_t::cbor, number) && sax->number_integer(static_cast<number_integer_t>(-1) - number);
            }

            case 0x3A: // Negative integer -1-n (four-byte uint32_t follows)
            {
                std::uint32_t number{};
                return get_number(input_format_t::cbor, number) && sax->number_integer(static_cast<number_integer_t>(-1) - number);
            }

            case 0x3B: // Negative integer -1-n (eight-byte uint64_t follows)
            {
                std::uint64_t number{};
                return get_number(input_format_t::cbor, number) && sax->number_integer(static_cast<number_integer_t>(-1)
                        - static_cast<number_integer_t>(number));
            }

            // Binary data (0x00..0x17 bytes follow)
            case 0x40:
            case 0x41:
            case 0x42:
            case 0x43:
            case 0x44:
            case 0x45:
            case 0x46:
            case 0x47:
            case 0x48:
            case 0x49:
            case 0x4A:
            case 0x4B:
            case 0x4C:
            case 0x4D:
            case 0x4E:
            case 0x4F:
            case 0x50:
            case 0x51:
            case 0x52:
            case 0x53:
            case 0x54:
            case 0x55:
            case 0x56:
            case 0x57:
            case 0x58: // Binary data (one-byte uint8_t for n follows)
            case 0x59: // Binary data (two-byte uint16_t for n follow)
            case 0x5A: // Binary data (four-byte uint32_t for n follow)
            case 0x5B: // Binary data (eight-byte uint64_t for n follow)
            case 0x5F: // Binary data (indefinite length)
            {
                binary_t b;
                return get_cbor_binary(b) && sax->binary(b);
            }

            // UTF-8 string (0x00..0x17 bytes follow)
            case 0x60:
            case 0x61:
            case 0x62:
            case 0x63:
            case 0x64:
            case 0x65:
            case 0x66:
            case 0x67:
            case 0x68:
            case 0x69:
            case 0x6A:
            case 0x6B:
            case 0x6C:
            case 0x6D:
            case 0x6E:
            case 0x6F:
            case 0x70:
            case 0x71:
            case 0x72:
            case 0x73:
            case 0x74:
            case 0x75:
            case 0x76:
            case 0x77:
            case 0x78: // UTF-8 string (one-byte uint8_t for n follows)
            case 0x79: // UTF-8 string (two-byte uint16_t for n follow)
            case 0x7A: // UTF-8 string (four-byte uint32_t for n follow)
            case 0x7B: // UTF-8 string (eight-byte uint64_t for n follow)
            case 0x7F: // UTF-8 string (indefinite length)
            {
                string_t s;
                return get_cbor_string(s) && sax->string(s);
            }

            // array (0x00..0x17 data items follow)
            case 0x80:
            case 0x81:
            case 0x82:
            case 0x83:
            case 0x84:
            case 0x85:
            case 0x86:
            case 0x87:
            case 0x88:
            case 0x89:
            case 0x8A:
            case 0x8B:
            case 0x8C:
            case 0x8D:
            case 0x8E:
            case 0x8F:
            case 0x90:
            case 0x91:
            case 0x92:
            case 0x93:
            case 0x94:
            case 0x95:
            case 0x96:
            case 0x97:
                return get_cbor_array(
                           conditional_static_cast<std::size_t>(static_cast<unsigned int>(current) & 0x1Fu), tag_handler);

            case 0x98: // array (one-byte uint8_t for n follows)
            {
                std::uint8_t len{};
                return get_number(input_format_t::cbor, len) && get_cbor_array(static_cast<std::size_t>(len), tag_handler);
            }

            case 0x99: // array (two-byte uint16_t for n follow)
            {
                std::uint16_t len{};
                return get_number(input_format_t::cbor, len) && get_cbor_array(static_cast<std::size_t>(len), tag_handler);
            }

            case 0x9A: // array (four-byte uint32_t for n follow)
            {
                std::uint32_t len{};
                return get_number(input_format_t::cbor, len) && get_cbor_array(conditional_static_cast<std::size_t>(len), tag_handler);
            }

            case 0x9B: // array (eight-byte uint64_t for n follow)
            {
                std::uint64_t len{};
                return get_number(input_format_t::cbor, len) && get_cbor_array(conditional_static_cast<std::size_t>(len), tag_handler);
            }

            case 0x9F: // array (indefinite length)
                return get_cbor_array(detail::unknown_size(), tag_handler);

            // map (0x00..0x17 pairs of data items follow)
            case 0xA0:
            case 0xA1:
            case 0xA2:
            case 0xA3:
            case 0xA4:
            case 0xA5:
            case 0xA6:
            case 0xA7:
            case 0xA8:
            case 0xA9:
            case 0xAA:
            case 0xAB:
            case 0xAC:
            case 0xAD:
            case 0xAE:
            case 0xAF:
            case 0xB0:
            case 0xB1:
            case 0xB2:
            case 0xB3:
            case 0xB4:
            case 0xB5:
            case 0xB6:
            case 0xB7:
                return get_cbor_object(conditional_static_cast<std::size_t>(static_cast<unsigned int>(current) & 0x1Fu), tag_handler);

            case 0xB8: // map (one-byte uint8_t for n follows)
            {
                std::uint8_t len{};
                return get_number(input_format_t::cbor, len) && get_cbor_object(static_cast<std::size_t>(len), tag_handler);
            }

            case 0xB9: // map (two-byte uint16_t for n follow)
            {
                std::uint16_t len{};
                return get_number(input_format_t::cbor, len) && get_cbor_object(static_cast<std::size_t>(len), tag_handler);
            }

            case 0xBA: // map (four-byte uint32_t for n follow)
            {
                std::uint32_t len{};
                return get_number(input_format_t::cbor, len) && get_cbor_object(conditional_static_cast<std::size_t>(len), tag_handler);
            }

            case 0xBB: // map (eight-byte uint64_t for n follow)
            {
                std::uint64_t len{};
                return get_number(input_format_t::cbor, len) && get_cbor_object(conditional_static_cast<std::size_t>(len), tag_handler);
            }

            case 0xBF: // map (indefinite length)
                return get_cbor_object(detail::unknown_size(), tag_handler);

            case 0xC6: // tagged item
            case 0xC7:
            case 0xC8:
            case 0xC9:
            case 0xCA:
            case 0xCB:
            case 0xCC:
            case 0xCD:
            case 0xCE:
            case 0xCF:
            case 0xD0:
            case 0xD1:
            case 0xD2:
            case 0xD3:
            case 0xD4:
            case 0xD8: // tagged item (1 bytes follow)
            case 0xD9: // tagged item (2 bytes follow)
            case 0xDA: // tagged item (4 bytes follow)
            case 0xDB: // tagged item (8 bytes follow)
            {
                switch (tag_handler)
                {
                    case cbor_tag_handler_t::error:
                    {
                        auto last_token = get_token_string();
                        return sax->parse_error(chars_read, last_token, parse_error::create(112, chars_read,
                                                exception_message(input_format_t::cbor, concat("invalid byte: 0x", last_token), "value"), nullptr));
                    }

                    case cbor_tag_handler_t::ignore:
                    {
                        // ignore binary subtype
                        switch (current)
                        {
                            case 0xD8:
                            {
                                std::uint8_t subtype_to_ignore{};
                                get_number(input_format_t::cbor, subtype_to_ignore);
                                break;
                            }
                            case 0xD9:
                            {
                                std::uint16_t subtype_to_ignore{};
                                get_number(input_format_t::cbor, subtype_to_ignore);
                                break;
                            }
                            case 0xDA:
                            {
                                std::uint32_t subtype_to_ignore{};
                                get_number(input_format_t::cbor, subtype_to_ignore);
                                break;
                            }
                            case 0xDB:
                            {
                                std::uint64_t subtype_to_ignore{};
                                get_number(input_format_t::cbor, subtype_to_ignore);
                                break;
                            }
                            default:
                                break;
                        }
                        return parse_cbor_internal(true, tag_handler);
                    }

                    case cbor_tag_handler_t::store:
                    {
                        binary_t b;
                        // use binary subtype and store in binary container
                        switch (current)
                        {
                            case 0xD8:
                            {
                                std::uint8_t subtype{};
                                get_number(input_format_t::cbor, subtype);
                                b.set_subtype(detail::conditional_static_cast<typename binary_t::subtype_type>(subtype));
                                break;
                            }
                            case 0xD9:
                            {
                                std::uint16_t subtype{};
                                get_number(input_format_t::cbor, subtype);
                                b.set_subtype(detail::conditional_static_cast<typename binary_t::subtype_type>(subtype));
                                break;
                            }
                            case 0xDA:
                            {
                                std::uint32_t subtype{};
                                get_number(input_format_t::cbor, subtype);
                                b.set_subtype(detail::conditional_static_cast<typename binary_t::subtype_type>(subtype));
                                break;
                            }
                            case 0xDB:
                            {
                                std::uint64_t subtype{};
                                get_number(input_format_t::cbor, subtype);
                                b.set_subtype(detail::conditional_static_cast<typename binary_t::subtype_type>(subtype));
                                break;
                            }
                            default:
                                return parse_cbor_internal(true, tag_handler);
                        }
                        get();
                        return get_cbor_binary(b) && sax->binary(b);
                    }

                    default:                 // LCOV_EXCL_LINE
                        JSON_ASSERT(false); // NOLINT(cert-dcl03-c,hicpp-static-assert,misc-static-assert) LCOV_EXCL_LINE
                        return false;        // LCOV_EXCL_LINE
                }
            }

            case 0xF4: // false
                return sax->boolean(false);

            case 0xF5: // true
                return sax->boolean(true);

            case 0xF6: // null
                return sax->null();

            case 0xF9: // Half-Precision Float (two-byte IEEE 754)
            {
                const auto byte1_raw = get();
                if (JSON_HEDLEY_UNLIKELY(!unexpect_eof(input_format_t::cbor, "number")))
                {
                    return false;
                }
                const auto byte2_raw = get();
                if (JSON_HEDLEY_UNLIKELY(!unexpect_eof(input_format_t::cbor, "number")))
                {
                    return false;
                }

                const auto byte1 = static_cast<unsigned char>(byte1_raw);
                const auto byte2 = static_cast<unsigned char>(byte2_raw);

                // code from RFC 7049, Appendix D, Figure 3:
                // As half-precision floating-point numbers were only added
                // to IEEE 754 in 2008, today's programming platforms often
                // still only have limited support for them. It is very
                // easy to include at least decoding support for them even
                // without such support. An example of a small decoder for
                // half-precision floating-point numbers in the C language
                // is shown in Fig. 3.
                const auto half = static_cast<unsigned int>((byte1 << 8u) + byte2);
                const double val = [&half]
                {
                    const int exp = (half >> 10u) & 0x1Fu;
                    const unsigned int mant = half & 0x3FFu;
                    JSON_ASSERT(0 <= exp&& exp <= 32);
                    JSON_ASSERT(mant <= 1024);
                    switch (exp)
                    {
                        case 0:
                            return std::ldexp(mant, -24);
                        case 31:
                            return (mant == 0)
                            ? std::numeric_limits<double>::infinity()
                            : std::numeric_limits<double>::quiet_NaN();
                        default:
                            return std::ldexp(mant + 1024, exp - 25);
                    }
                }();
                return sax->number_float((half & 0x8000u) != 0
                                         ? static_cast<number_float_t>(-val)
                                         : static_cast<number_float_t>(val), "");
            }

            case 0xFA: // Single-Precision Float (four-byte IEEE 754)
            {
                float number{};
                return get_number(input_format_t::cbor, number) && sax->number_float(static_cast<number_float_t>(number), "");
            }

            case 0xFB: // Double-Precision Float (eight-byte IEEE 754)
            {
                double number{};
                return get_number(input_format_t::cbor, number) && sax->number_float(static_cast<number_float_t>(number), "");
            }

            default: // anything else (0xFF is handled inside the other types)
            {
                auto last_token = get_token_string();
                return sax->parse_error(chars_read, last_token, parse_error::create(112, chars_read,
                                        exception_message(input_format_t::cbor, concat("invalid byte: 0x", last_token), "value"), nullptr));
            }
        }
    }

    /*!
    @brief reads a CBOR string

    This function first reads starting bytes to determine the expected
    string length and then copies this number of bytes into a string.
    Additionally, CBOR's strings with indefinite lengths are supported.

    @param[out] result  created string

    @return whether string creation completed
    */
    bool get_cbor_string(string_t& result)
    {
        if (JSON_HEDLEY_UNLIKELY(!unexpect_eof(input_format_t::cbor, "string")))
        {
            return false;
        }

        switch (current)
        {
            // UTF-8 string (0x00..0x17 bytes follow)
            case 0x60:
            case 0x61:
            case 0x62:
            case 0x63:
            case 0x64:
            case 0x65:
            case 0x66:
            case 0x67:
            case 0x68:
            case 0x69:
            case 0x6A:
            case 0x6B:
            case 0x6C:
            case 0x6D:
            case 0x6E:
            case 0x6F:
            case 0x70:
            case 0x71:
            case 0x72:
            case 0x73:
            case 0x74:
            case 0x75:
            case 0x76:
            case 0x77:
            {
                return get_string(input_format_t::cbor, static_cast<unsigned int>(current) & 0x1Fu, result);
            }

            case 0x78: // UTF-8 string (one-byte uint8_t for n follows)
            {
                std::uint8_t len{};
                return get_number(input_format_t::cbor, len) && get_string(input_format_t::cbor, len, result);
            }

            case 0x79: // UTF-8 string (two-byte uint16_t for n follow)
            {
                std::uint16_t len{};
                return get_number(input_format_t::cbor, len) && get_string(input_format_t::cbor, len, result);
            }

            case 0x7A: // UTF-8 string (four-byte uint32_t for n follow)
            {
                std::uint32_t len{};
                return get_number(input_format_t::cbor, len) && get_string(input_format_t::cbor, len, result);
            }

            case 0x7B: // UTF-8 string (eight-byte uint64_t for n follow)
            {
                std::uint64_t len{};
                return get_number(input_format_t::cbor, len) && get_string(input_format_t::cbor, len, result);
            }

            case 0x7F: // UTF-8 string (indefinite length)
            {
                while (get() != 0xFF)
                {
                    string_t chunk;
                    if (!get_cbor_string(chunk))
                    {
                        return false;
                    }
                    result.append(chunk);
                }
                return true;
            }

            default:
            {
                auto last_token = get_token_string();
                return sax->parse_error(chars_read, last_token, parse_error::create(113, chars_read,
                                        exception_message(input_format_t::cbor, concat("expected length specification (0x60-0x7B) or indefinite string type (0x7F); last byte: 0x", last_token), "string"), nullptr));
            }
        }
    }

    /*!
    @brief reads a CBOR byte array

    This function first reads starting bytes to determine the expected
    byte array length and then copies this number of bytes into the byte array.
    Additionally, CBOR's byte arrays with indefinite lengths are supported.

    @param[out] result  created byte array

    @return whether byte array creation completed
    */
    bool get_cbor_binary(binary_t& result)
    {
        if (JSON_HEDLEY_UNLIKELY(!unexpect_eof(input_format_t::cbor, "binary")))
        {
            return false;
        }

        switch (current)
        {
            // Binary data (0x00..0x17 bytes follow)
            case 0x40:
            case 0x41:
            case 0x42:
            case 0x43:
            case 0x44:
            case 0x45:
            case 0x46:
            case 0x47:
            case 0x48:
            case 0x49:
            case 0x4A:
            case 0x4B:
            case 0x4C:
            case 0x4D:
            case 0x4E:
            case 0x4F:
            case 0x50:
            case 0x51:
            case 0x52:
            case 0x53:
            case 0x54:
            case 0x55:
            case 0x56:
            case 0x57:
            {
                return get_binary(input_format_t::cbor, static_cast<unsigned int>(current) & 0x1Fu, result);
            }

            case 0x58: // Binary data (one-byte uint8_t for n follows)
            {
                std::uint8_t len{};
                return get_number(input_format_t::cbor, len) &&
                       get_binary(input_format_t::cbor, len, result);
            }

            case 0x59: // Binary data (two-byte uint16_t for n follow)
            {
                std::uint16_t len{};
                return get_number(input_format_t::cbor, len) &&
                       get_binary(input_format_t::cbor, len, result);
            }

            case 0x5A: // Binary data (four-byte uint32_t for n follow)
            {
                std::uint32_t len{};
                return get_number(input_format_t::cbor, len) &&
                       get_binary(input_format_t::cbor, len, result);
            }

            case 0x5B: // Binary data (eight-byte uint64_t for n follow)
            {
                std::uint64_t len{};
                return get_number(input_format_t::cbor, len) &&
                       get_binary(input_format_t::cbor, len, result);
            }

            case 0x5F: // Binary data (indefinite length)
            {
                while (get() != 0xFF)
                {
                    binary_t chunk;
                    if (!get_cbor_binary(chunk))
                    {
                        return false;
                    }
                    result.insert(result.end(), chunk.begin(), chunk.end());
                }
                return true;
            }

            default:
            {
                auto last_token = get_token_string();
                return sax->parse_error(chars_read, last_token, parse_error::create(113, chars_read,
                                        exception_message(input_format_t::cbor, concat("expected length specification (0x40-0x5B) or indefinite binary array type (0x5F); last byte: 0x", last_token), "binary"), nullptr));
            }
        }
    }

    /*!
    @param[in] len  the length of the array or detail::unknown_size() for an
                    array of indefinite size
    @param[in] tag_handler how CBOR tags should be treated
    @return whether array creation completed
    */
    bool get_cbor_array(const std::size_t len,
                        const cbor_tag_handler_t tag_handler)
    {
        if (JSON_HEDLEY_UNLIKELY(!sax->start_array(len)))
        {
            return false;
        }

        if (len != detail::unknown_size())
        {
            for (std::size_t i = 0; i < len; ++i)
            {
                if (JSON_HEDLEY_UNLIKELY(!parse_cbor_internal(true, tag_handler)))
                {
                    return false;
                }
            }
        }
        else
        {
            while (get() != 0xFF)
            {
                if (JSON_HEDLEY_UNLIKELY(!parse_cbor_internal(false, tag_handler)))
                {
                    return false;
                }
            }
        }

        return sax->end_array();
    }

    /*!
    @param[in] len  the length of the object or detail::unknown_size() for an
                    object of indefinite size
    @param[in] tag_handler how CBOR tags should be treated
    @return whether object creation completed
    */
    bool get_cbor_object(const std::size_t len,
                         const cbor_tag_handler_t tag_handler)
    {
        if (JSON_HEDLEY_UNLIKELY(!sax->start_object(len)))
        {
            return false;
        }

        if (len != 0)
        {
            string_t key;
            if (len != detail::unknown_size())
            {
                for (std::size_t i = 0; i < len; ++i)
                {
                    get();
                    if (JSON_HEDLEY_UNLIKELY(!get_cbor_string(key) || !sax->key(key)))
                    {
                        return false;
                    }

                    if (JSON_HEDLEY_UNLIKELY(!parse_cbor_internal(true, tag_handler)))
                    {
                        return false;
                    }
                    key.clear();
                }
            }
            else
            {
                while (get() != 0xFF)
                {
                    if (JSON_HEDLEY_UNLIKELY(!get_cbor_string(key) || !sax->key(key)))
                    {
                        return false;
                    }

                    if (JSON_HEDLEY_UNLIKELY(!parse_cbor_internal(true, tag_handler)))
                    {
                        return false;
                    }
                    key.clear();
                }
            }
        }

        return sax->end_object();
    }

    /////////////
    // MsgPack //
    /////////////

    /*!
    @return whether a valid MessagePack value was passed to the SAX parser
    */
    bool parse_msgpack_internal()
    {
        switch (get())
        {
            // EOF
            case char_traits<char_type>::eof():
                return unexpect_eof(input_format_t::msgpack, "value");

            // positive fixint
            case 0x00:
            case 0x01:
            case 0x02:
            case 0x03:
            case 0x04:
            case 0x05:
            case 0x06:
            case 0x07:
            case 0x08:
            case 0x09:
            case 0x0A:
            case 0x0B:
            case 0x0C:
            case 0x0D:
            case 0x0E:
            case 0x0F:
            case 0x10:
            case 0x11:
            case 0x12:
            case 0x13:
            case 0x14:
            case 0x15:
            case 0x16:
            case 0x17:
            case 0x18:
            case 0x19:
            case 0x1A:
            case 0x1B:
            case 0x1C:
            case 0x1D:
            case 0x1E:
            case 0x1F:
            case 0x20:
            case 0x21:
            case 0x22:
            case 0x23:
            case 0x24:
            case 0x25:
            case 0x26:
            case 0x27:
            case 0x28:
            case 0x29:
            case 0x2A:
            case 0x2B:
            case 0x2C:
            case 0x2D:
            case 0x2E:
            case 0x2F:
            case 0x30:
            case 0x31:
            case 0x32:
            case 0x33:
            case 0x34:
            case 0x35:
            case 0x36:
            case 0x37:
            case 0x38:
            case 0x39:
            case 0x3A:
            case 0x3B:
            case 0x3C:
            case 0x3D:
            case 0x3E:
            case 0x3F:
            case 0x40:
            case 0x41:
            case 0x42:
            case 0x43:
            case 0x44:
            case 0x45:
            case 0x46:
            case 0x47:
            case 0x48:
            case 0x49:
            case 0x4A:
            case 0x4B:
            case 0x4C:
            case 0x4D:
            case 0x4E:
            case 0x4F:
            case 0x50:
            case 0x51:
            case 0x52:
            case 0x53:
            case 0x54:
            case 0x55:
            case 0x56:
            case 0x57:
            case 0x58:
            case 0x59:
            case 0x5A:
            case 0x5B:
            case 0x5C:
            case 0x5D:
            case 0x5E:
            case 0x5F:
            case 0x60:
            case 0x61:
            case 0x62:
            case 0x63:
            case 0x64:
            case 0x65:
            case 0x66:
            case 0x67:
            case 0x68:
            case 0x69:
            case 0x6A:
            case 0x6B:
            case 0x6C:
            case 0x6D:
            case 0x6E:
            case 0x6F:
            case 0x70:
            case 0x71:
            case 0x72:
            case 0x73:
            case 0x74:
            case 0x75:
            case 0x76:
            case 0x77:
            case 0x78:
            case 0x79:
            case 0x7A:
            case 0x7B:
            case 0x7C:
            case 0x7D:
            case 0x7E:
            case 0x7F:
                return sax->number_unsigned(static_cast<number_unsigned_t>(current));

            // fixmap
            case 0x80:
            case 0x81:
            case 0x82:
            case 0x83:
            case 0x84:
            case 0x85:
            case 0x86:
            case 0x87:
            case 0x88:
            case 0x89:
            case 0x8A:
            case 0x8B:
            case 0x8C:
            case 0x8D:
            case 0x8E:
            case 0x8F:
                return get_msgpack_object(conditional_static_cast<std::size_t>(static_cast<unsigned int>(current) & 0x0Fu));

            // fixarray
            case 0x90:
            case 0x91:
            case 0x92:
            case 0x93:
            case 0x94:
            case 0x95:
            case 0x96:
            case 0x97:
            case 0x98:
            case 0x99:
            case 0x9A:
            case 0x9B:
            case 0x9C:
            case 0x9D:
            case 0x9E:
            case 0x9F:
                return get_msgpack_array(conditional_static_cast<std::size_t>(static_cast<unsigned int>(current) & 0x0Fu));

            // fixstr
            case 0xA0:
            case 0xA1:
            case 0xA2:
            case 0xA3:
            case 0xA4:
            case 0xA5:
            case 0xA6:
            case 0xA7:
            case 0xA8:
            case 0xA9:
            case 0xAA:
            case 0xAB:
            case 0xAC:
            case 0xAD:
            case 0xAE:
            case 0xAF:
            case 0xB0:
            case 0xB1:
            case 0xB2:
            case 0xB3:
            case 0xB4:
            case 0xB5:
            case 0xB6:
            case 0xB7:
            case 0xB8:
            case 0xB9:
            case 0xBA:
            case 0xBB:
            case 0xBC:
            case 0xBD:
            case 0xBE:
            case 0xBF:
            case 0xD9: // str 8
            case 0xDA: // str 16
            case 0xDB: // str 32
            {
                string_t s;
                return get_msgpack_string(s) && sax->string(s);
            }

            case 0xC0: // nil
                return sax->null();

            case 0xC2: // false
                return sax->boolean(false);

            case 0xC3: // true
                return sax->boolean(true);

            case 0xC4: // bin 8
            case 0xC5: // bin 16
            case 0xC6: // bin 32
            case 0xC7: // ext 8
            case 0xC8: // ext 16
            case 0xC9: // ext 32
            case 0xD4: // fixext 1
            case 0xD5: // fixext 2
            case 0xD6: // fixext 4
            case 0xD7: // fixext 8
            case 0xD8: // fixext 16
            {
                binary_t b;
                return get_msgpack_binary(b) && sax->binary(b);
            }

            case 0xCA: // float 32
            {
                float number{};
                return get_number(input_format_t::msgpack, number) && sax->number_float(static_cast<number_float_t>(number), "");
            }

            case 0xCB: // float 64
            {
                double number{};
                return get_number(input_format_t::msgpack, number) && sax->number_float(static_cast<number_float_t>(number), "");
            }

            case 0xCC: // uint 8
            {
                std::uint8_t number{};
                return get_number(input_format_t::msgpack, number) && sax->number_unsigned(number);
            }

            case 0xCD: // uint 16
            {
                std::uint16_t number{};
                return get_number(input_format_t::msgpack, number) && sax->number_unsigned(number);
            }

            case 0xCE: // uint 32
            {
                std::uint32_t number{};
                return get_number(input_format_t::msgpack, number) && sax->number_unsigned(number);
            }

            case 0xCF: // uint 64
            {
                std::uint64_t number{};
                return get_number(input_format_t::msgpack, number) && sax->number_unsigned(number);
            }

            case 0xD0: // int 8
            {
                std::int8_t number{};
                return get_number(input_format_t::msgpack, number) && sax->number_integer(number);
            }

            case 0xD1: // int 16
            {
                std::int16_t number{};
                return get_number(input_format_t::msgpack, number) && sax->number_integer(number);
            }

            case 0xD2: // int 32
            {
                std::int32_t number{};
                return get_number(input_format_t::msgpack, number) && sax->number_integer(number);
            }

            case 0xD3: // int 64
            {
                std::int64_t number{};
                return get_number(input_format_t::msgpack, number) && sax->number_integer(number);
            }

            case 0xDC: // array 16
            {
                std::uint16_t len{};
                return get_number(input_format_t::msgpack, len) && get_msgpack_array(static_cast<std::size_t>(len));
            }

            case 0xDD: // array 32
            {
                std::uint32_t len{};
                return get_number(input_format_t::msgpack, len) && get_msgpack_array(conditional_static_cast<std::size_t>(len));
            }

            case 0xDE: // map 16
            {
                std::uint16_t len{};
                return get_number(input_format_t::msgpack, len) && get_msgpack_object(static_cast<std::size_t>(len));
            }

            case 0xDF: // map 32
            {
                std::uint32_t len{};
                return get_number(input_format_t::msgpack, len) && get_msgpack_object(conditional_static_cast<std::size_t>(len));
            }

            // negative fixint
            case 0xE0:
            case 0xE1:
            case 0xE2:
            case 0xE3:
            case 0xE4:
            case 0xE5:
            case 0xE6:
            case 0xE7:
            case 0xE8:
            case 0xE9:
            case 0xEA:
            case 0xEB:
            case 0xEC:
            case 0xED:
            case 0xEE:
            case 0xEF:
            case 0xF0:
            case 0xF1:
            case 0xF2:
            case 0xF3:
            case 0xF4:
            case 0xF5:
            case 0xF6:
            case 0xF7:
            case 0xF8:
            case 0xF9:
            case 0xFA:
            case 0xFB:
            case 0xFC:
            case 0xFD:
            case 0xFE:
            case 0xFF:
                return sax->number_integer(static_cast<std::int8_t>(current));

            default: // anything else
            {
                auto last_token = get_token_string();
                return sax->parse_error(chars_read, last_token, parse_error::create(112, chars_read,
                                        exception_message(input_format_t::msgpack, concat("invalid byte: 0x", last_token), "value"), nullptr));
            }
        }
    }

    /*!
    @brief reads a MessagePack string

    This function first reads starting bytes to determine the expected
    string length and then copies this number of bytes into a string.

    @param[out] result  created string

    @return whether string creation completed
    */
    bool get_msgpack_string(string_t& result)
    {
        if (JSON_HEDLEY_UNLIKELY(!unexpect_eof(input_format_t::msgpack, "string")))
        {
            return false;
        }

        switch (current)
        {
            // fixstr
            case 0xA0:
            case 0xA1:
            case 0xA2:
            case 0xA3:
            case 0xA4:
            case 0xA5:
            case 0xA6:
            case 0xA7:
            case 0xA8:
            case 0xA9:
            case 0xAA:
            case 0xAB:
            case 0xAC:
            case 0xAD:
            case 0xAE:
            case 0xAF:
            case 0xB0:
            case 0xB1:
            case 0xB2:
            case 0xB3:
            case 0xB4:
            case 0xB5:
            case 0xB6:
            case 0xB7:
            case 0xB8:
            case 0xB9:
            case 0xBA:
            case 0xBB:
            case 0xBC:
            case 0xBD:
            case 0xBE:
            case 0xBF:
            {
                return get_string(input_format_t::msgpack, static_cast<unsigned int>(current) & 0x1Fu, result);
            }

            case 0xD9: // str 8
            {
                std::uint8_t len{};
                return get_number(input_format_t::msgpack, len) && get_string(input_format_t::msgpack, len, result);
            }

            case 0xDA: // str 16
            {
                std::uint16_t len{};
                return get_number(input_format_t::msgpack, len) && get_string(input_format_t::msgpack, len, result);
            }

            case 0xDB: // str 32
            {
                std::uint32_t len{};
                return get_number(input_format_t::msgpack, len) && get_string(input_format_t::msgpack, len, result);
            }

            default:
            {
                auto last_token = get_token_string();
                return sax->parse_error(chars_read, last_token, parse_error::create(113, chars_read,
                                        exception_message(input_format_t::msgpack, concat("expected length specification (0xA0-0xBF, 0xD9-0xDB); last byte: 0x", last_token), "string"), nullptr));
            }
        }
    }

    /*!
    @brief reads a MessagePack byte array

    This function first reads starting bytes to determine the expected
    byte array length and then copies this number of bytes into a byte array.

    @param[out] result  created byte array

    @return whether byte array creation completed
    */
    bool get_msgpack_binary(binary_t& result)
    {
        // helper function to set the subtype
        auto assign_and_return_true = [&result](std::int8_t subtype)
        {
            result.set_subtype(static_cast<std::uint8_t>(subtype));
            return true;
        };

        switch (current)
        {
            case 0xC4: // bin 8
            {
                std::uint8_t len{};
                return get_number(input_format_t::msgpack, len) &&
                       get_binary(input_format_t::msgpack, len, result);
            }

            case 0xC5: // bin 16
            {
                std::uint16_t len{};
                return get_number(input_format_t::msgpack, len) &&
                       get_binary(input_format_t::msgpack, len, result);
            }

            case 0xC6: // bin 32
            {
                std::uint32_t len{};
                return get_number(input_format_t::msgpack, len) &&
                       get_binary(input_format_t::msgpack, len, result);
            }

            case 0xC7: // ext 8
            {
                std::uint8_t len{};
                std::int8_t subtype{};
                return get_number(input_format_t::msgpack, len) &&
                       get_number(input_format_t::msgpack, subtype) &&
                       get_binary(input_format_t::msgpack, len, result) &&
                       assign_and_return_true(subtype);
            }

            case 0xC8: // ext 16
            {
                std::uint16_t len{};
                std::int8_t subtype{};
                return get_number(input_format_t::msgpack, len) &&
                       get_number(input_format_t::msgpack, subtype) &&
                       get_binary(input_format_t::msgpack, len, result) &&
                       assign_and_return_true(subtype);
            }

            case 0xC9: // ext 32
            {
                std::uint32_t len{};
                std::int8_t subtype{};
                return get_number(input_format_t::msgpack, len) &&
                       get_number(input_format_t::msgpack, subtype) &&
                       get_binary(input_format_t::msgpack, len, result) &&
                       assign_and_return_true(subtype);
            }

            case 0xD4: // fixext 1
            {
                std::int8_t subtype{};
                return get_number(input_format_t::msgpack, subtype) &&
                       get_binary(input_format_t::msgpack, 1, result) &&
                       assign_and_return_true(subtype);
            }

            case 0xD5: // fixext 2
            {
                std::int8_t subtype{};
                return get_number(input_format_t::msgpack, subtype) &&
                       get_binary(input_format_t::msgpack, 2, result) &&
                       assign_and_return_true(subtype);
            }

            case 0xD6: // fixext 4
            {
                std::int8_t subtype{};
                return get_number(input_format_t::msgpack, subtype) &&
                       get_binary(input_format_t::msgpack, 4, result) &&
                       assign_and_return_true(subtype);
            }

            case 0xD7: // fixext 8
            {
                std::int8_t subtype{};
                return get_number(input_format_t::msgpack, subtype) &&
                       get_binary(input_format_t::msgpack, 8, result) &&
                       assign_and_return_true(subtype);
            }

            case 0xD8: // fixext 16
            {
                std::int8_t subtype{};
                return get_number(input_format_t::msgpack, subtype) &&
                       get_binary(input_format_t::msgpack, 16, result) &&
                       assign_and_return_true(subtype);
            }

            default:           // LCOV_EXCL_LINE
                return false;  // LCOV_EXCL_LINE
        }
    }

    /*!
    @param[in] len  the length of the array
    @return whether array creation completed
    */
    bool get_msgpack_array(const std::size_t len)
    {
        if (JSON_HEDLEY_UNLIKELY(!sax->start_array(len)))
        {
            return false;
        }

        for (std::size_t i = 0; i < len; ++i)
        {
            if (JSON_HEDLEY_UNLIKELY(!parse_msgpack_internal()))
            {
                return false;
            }
        }

        return sax->end_array();
    }

    /*!
    @param[in] len  the length of the object
    @return whether object creation completed
    */
    bool get_msgpack_object(const std::size_t len)
    {
        if (JSON_HEDLEY_UNLIKELY(!sax->start_object(len)))
        {
            return false;
        }

        string_t key;
        for (std::size_t i = 0; i < len; ++i)
        {
            get();
            if (JSON_HEDLEY_UNLIKELY(!get_msgpack_string(key) || !sax->key(key)))
            {
                return false;
            }

            if (JSON_HEDLEY_UNLIKELY(!parse_msgpack_internal()))
            {
                return false;
            }
            key.clear();
        }

        return sax->end_object();
    }

    ////////////
    // UBJSON //
    ////////////

    /*!
    @param[in] get_char  whether a new character should be retrieved from the
                         input (true, default) or whether the last read
                         character should be considered instead

    @return whether a valid UBJSON value was passed to the SAX parser
    */
    bool parse_ubjson_internal(const bool get_char = true)
    {
        return get_ubjson_value(get_char ? get_ignore_noop() : current);
    }

    /*!
    @brief reads a UBJSON string

    This function is either called after reading the 'S' byte explicitly
    indicating a string, or in case of an object key where the 'S' byte can be
    left out.

    @param[out] result   created string
    @param[in] get_char  whether a new character should be retrieved from the
                         input (true, default) or whether the last read
                         character should be considered instead

    @return whether string creation completed
    */
    bool get_ubjson_string(string_t& result, const bool get_char = true)
    {
        if (get_char)
        {
            get();  // TODO(niels): may we ignore N here?
        }

        if (JSON_HEDLEY_UNLIKELY(!unexpect_eof(input_format, "value")))
        {
            return false;
        }

        switch (current)
        {
            case 'U':
            {
                std::uint8_t len{};
                return get_number(input_format, len) && get_string(input_format, len, result);
            }

            case 'i':
            {
                std::int8_t len{};
                return get_number(input_format, len) && get_string(input_format, len, result);
            }

            case 'I':
            {
                std::int16_t len{};
                return get_number(input_format, len) && get_string(input_format, len, result);
            }

            case 'l':
            {
                std::int32_t len{};
                return get_number(input_format, len) && get_string(input_format, len, result);
            }

            case 'L':
            {
                std::int64_t len{};
                return get_number(input_format, len) && get_string(input_format, len, result);
            }

            case 'u':
            {
                if (input_format != input_format_t::bjdata)
                {
                    break;
                }
                std::uint16_t len{};
                return get_number(input_format, len) && get_string(input_format, len, result);
            }

            case 'm':
            {
                if (input_format != input_format_t::bjdata)
                {
                    break;
                }
                std::uint32_t len{};
                return get_number(input_format, len) && get_string(input_format, len, result);
            }

            case 'M':
            {
                if (input_format != input_format_t::bjdata)
                {
                    break;
                }
                std::uint64_t len{};
                return get_number(input_format, len) && get_string(input_format, len, result);
            }

            default:
                break;
        }
        auto last_token = get_token_string();
        std::string message;

        if (input_format != input_format_t::bjdata)
        {
            message = "expected length type specification (U, i, I, l, L); last byte: 0x" + last_token;
        }
        else
        {
            message = "expected length type specification (U, i, u, I, m, l, M, L); last byte: 0x" + last_token;
        }
        return sax->parse_error(chars_read, last_token, parse_error::create(113, chars_read, exception_message(input_format, message, "string"), nullptr));
    }

    /*!
    @param[out] dim  an integer vector storing the ND array dimensions
    @return whether reading ND array size vector is successful
    */
    bool get_ubjson_ndarray_size(std::vector<size_t>& dim)
    {
        std::pair<std::size_t, char_int_type> size_and_type;
        size_t dimlen = 0;
        bool no_ndarray = true;

        if (JSON_HEDLEY_UNLIKELY(!get_ubjson_size_type(size_and_type, no_ndarray)))
        {
            return false;
        }

        if (size_and_type.first != npos)
        {
            if (size_and_type.second != 0)
            {
                if (size_and_type.second != 'N')
                {
                    for (std::size_t i = 0; i < size_and_type.first; ++i)
                    {
                        if (JSON_HEDLEY_UNLIKELY(!get_ubjson_size_value(dimlen, no_ndarray, size_and_type.second)))
                        {
                            return false;
                        }
                        dim.push_back(dimlen);
                    }
                }
            }
            else
            {
                for (std::size_t i = 0; i < size_and_type.first; ++i)
                {
                    if (JSON_HEDLEY_UNLIKELY(!get_ubjson_size_value(dimlen, no_ndarray)))
                    {
                        return false;
                    }
                    dim.push_back(dimlen);
                }
            }
        }
        else
        {
            while (current != ']')
            {
                if (JSON_HEDLEY_UNLIKELY(!get_ubjson_size_value(dimlen, no_ndarray, current)))
                {
                    return false;
                }
                dim.push_back(dimlen);
                get_ignore_noop();
            }
        }
        return true;
    }

    /*!
    @param[out] result  determined size
    @param[in,out] is_ndarray  for input, `true` means already inside an ndarray vector
                               or ndarray dimension is not allowed; `false` means ndarray
                               is allowed; for output, `true` means an ndarray is found;
                               is_ndarray can only return `true` when its initial value
                               is `false`
    @param[in] prefix  type marker if already read, otherwise set to 0

    @return whether size determination completed
    */
    bool get_ubjson_size_value(std::size_t& result, bool& is_ndarray, char_int_type prefix = 0)
    {
        if (prefix == 0)
        {
            prefix = get_ignore_noop();
        }

        switch (prefix)
        {
            case 'U':
            {
                std::uint8_t number{};
                if (JSON_HEDLEY_UNLIKELY(!get_number(input_format, number)))
                {
                    return false;
                }
                result = static_cast<std::size_t>(number);
                return true;
            }

            case 'i':
            {
                std::int8_t number{};
                if (JSON_HEDLEY_UNLIKELY(!get_number(input_format, number)))
                {
                    return false;
                }
                if (number < 0)
                {
                    return sax->parse_error(chars_read, get_token_string(), parse_error::create(113, chars_read,
                                            exception_message(input_format, "count in an optimized container must be positive", "size"), nullptr));
                }
                result = static_cast<std::size_t>(number); // NOLINT(bugprone-signed-char-misuse,cert-str34-c): number is not a char
                return true;
            }

            case 'I':
            {
                std::int16_t number{};
                if (JSON_HEDLEY_UNLIKELY(!get_number(input_format, number)))
                {
                    return false;
                }
                if (number < 0)
                {
                    return sax->parse_error(chars_read, get_token_string(), parse_error::create(113, chars_read,
                                            exception_message(input_format, "count in an optimized container must be positive", "size"), nullptr));
                }
                result = static_cast<std::size_t>(number);
                return true;
            }

            case 'l':
            {
                std::int32_t number{};
                if (JSON_HEDLEY_UNLIKELY(!get_number(input_format, number)))
                {
                    return false;
                }
                if (number < 0)
                {
                    return sax->parse_error(chars_read, get_token_string(), parse_error::create(113, chars_read,
                                            exception_message(input_format, "count in an optimized container must be positive", "size"), nullptr));
                }
                result = static_cast<std::size_t>(number);
                return true;
            }

            case 'L':
            {
                std::int64_t number{};
                if (JSON_HEDLEY_UNLIKELY(!get_number(input_format, number)))
                {
                    return false;
                }
                if (number < 0)
                {
                    return sax->parse_error(chars_read, get_token_string(), parse_error::create(113, chars_read,
                                            exception_message(input_format, "count in an optimized container must be positive", "size"), nullptr));
                }
                if (!value_in_range_of<std::size_t>(number))
                {
                    return sax->parse_error(chars_read, get_token_string(), out_of_range::create(408,
                                            exception_message(input_format, "integer value overflow", "size"), nullptr));
                }
                result = static_cast<std::size_t>(number);
                return true;
            }

            case 'u':
            {
                if (input_format != input_format_t::bjdata)
                {
                    break;
                }
                std::uint16_t number{};
                if (JSON_HEDLEY_UNLIKELY(!get_number(input_format, number)))
                {
                    return false;
                }
                result = static_cast<std::size_t>(number);
                return true;
            }

            case 'm':
            {
                if (input_format != input_format_t::bjdata)
                {
                    break;
                }
                std::uint32_t number{};
                if (JSON_HEDLEY_UNLIKELY(!get_number(input_format, number)))
                {
                    return false;
                }
                result = conditional_static_cast<std::size_t>(number);
                return true;
            }

            case 'M':
            {
                if (input_format != input_format_t::bjdata)
                {
                    break;
                }
                std::uint64_t number{};
                if (JSON_HEDLEY_UNLIKELY(!get_number(input_format, number)))
                {
                    return false;
                }
                if (!value_in_range_of<std::size_t>(number))
                {
                    return sax->parse_error(chars_read, get_token_string(), out_of_range::create(408,
                                            exception_message(input_format, "integer value overflow", "size"), nullptr));
                }
                result = detail::conditional_static_cast<std::size_t>(number);
                return true;
            }

            case '[':
            {
                if (input_format != input_format_t::bjdata)
                {
                    break;
                }
                if (is_ndarray) // ndarray dimensional vector can only contain integers, and can not embed another array
                {
                    return sax->parse_error(chars_read, get_token_string(), parse_error::create(113, chars_read, exception_message(input_format, "ndarray dimensional vector is not allowed", "size"), nullptr));
                }
                std::vector<size_t> dim;
                if (JSON_HEDLEY_UNLIKELY(!get_ubjson_ndarray_size(dim)))
                {
                    return false;
                }
                if (dim.size() == 1 || (dim.size() == 2 && dim.at(0) == 1)) // return normal array size if 1D row vector
                {
                    result = dim.at(dim.size() - 1);
                    return true;
                }
                if (!dim.empty())  // if ndarray, convert to an object in JData annotated array format
                {
                    for (auto i : dim) // test if any dimension in an ndarray is 0, if so, return a 1D empty container
                    {
                        if ( i == 0 )
                        {
                            result = 0;
                            return true;
                        }
                    }

                    string_t key = "_ArraySize_";
                    if (JSON_HEDLEY_UNLIKELY(!sax->start_object(3) || !sax->key(key) || !sax->start_array(dim.size())))
                    {
                        return false;
                    }
                    result = 1;
                    for (auto i : dim)
                    {
                        result *= i;
                        if (result == 0 || result == npos) // because dim elements shall not have zeros, result = 0 means overflow happened; it also can't be npos as it is used to initialize size in get_ubjson_size_type()
                        {
                            return sax->parse_error(chars_read, get_token_string(), out_of_range::create(408, exception_message(input_format, "excessive ndarray size caused overflow", "size"), nullptr));
                        }
                        if (JSON_HEDLEY_UNLIKELY(!sax->number_unsigned(static_cast<number_unsigned_t>(i))))
                        {
                            return false;
                        }
                    }
                    is_ndarray = true;
                    return sax->end_array();
                }
                result = 0;
                return true;
            }

            default:
                break;
        }
        auto last_token = get_token_string();
        std::string message;

        if (input_format != input_format_t::bjdata)
        {
            message = "expected length type specification (U, i, I, l, L) after '#'; last byte: 0x" + last_token;
        }
        else
        {
            message = "expected length type specification (U, i, u, I, m, l, M, L) after '#'; last byte: 0x" + last_token;
        }
        return sax->parse_error(chars_read, last_token, parse_error::create(113, chars_read, exception_message(input_format, message, "size"), nullptr));
    }

    /*!
    @brief determine the type and size for a container

    In the optimized UBJSON format, a type and a size can be provided to allow
    for a more compact representation.

    @param[out] result  pair of the size and the type
    @param[in] inside_ndarray  whether the parser is parsing an ND array dimensional vector

    @return whether pair creation completed
    */
    bool get_ubjson_size_type(std::pair<std::size_t, char_int_type>& result, bool inside_ndarray = false)
    {
        result.first = npos; // size
        result.second = 0; // type
        bool is_ndarray = false;

        get_ignore_noop();

        if (current == '$')
        {
            result.second = get();  // must not ignore 'N', because 'N' maybe the type
            if (input_format == input_format_t::bjdata
                    && JSON_HEDLEY_UNLIKELY(std::binary_search(bjd_optimized_type_markers.begin(), bjd_optimized_type_markers.end(), result.second)))
            {
                auto last_token = get_token_string();
                return sax->parse_error(chars_read, last_token, parse_error::create(112, chars_read,
                                        exception_message(input_format, concat("marker 0x", last_token, " is not a permitted optimized array type"), "type"), nullptr));
            }

            if (JSON_HEDLEY_UNLIKELY(!unexpect_eof(input_format, "type")))
            {
                return false;
            }

            get_ignore_noop();
            if (JSON_HEDLEY_UNLIKELY(current != '#'))
            {
                if (JSON_HEDLEY_UNLIKELY(!unexpect_eof(input_format, "value")))
                {
                    return false;
                }
                auto last_token = get_token_string();
                return sax->parse_error(chars_read, last_token, parse_error::create(112, chars_read,
                                        exception_message(input_format, concat("expected '#' after type information; last byte: 0x", last_token), "size"), nullptr));
            }

            const bool is_error = get_ubjson_size_value(result.first, is_ndarray);
            if (input_format == input_format_t::bjdata && is_ndarray)
            {
                if (inside_ndarray)
                {
                    return sax->parse_error(chars_read, get_token_string(), parse_error::create(112, chars_read,
                                            exception_message(input_format, "ndarray can not be recursive", "size"), nullptr));
                }
                result.second |= (1 << 8); // use bit 8 to indicate ndarray, all UBJSON and BJData markers should be ASCII letters
            }
            return is_error;
        }

        if (current == '#')
        {
            const bool is_error = get_ubjson_size_value(result.first, is_ndarray);
            if (input_format == input_format_t::bjdata && is_ndarray)
            {
                return sax->parse_error(chars_read, get_token_string(), parse_error::create(112, chars_read,
                                        exception_message(input_format, "ndarray requires both type and size", "size"), nullptr));
            }
            return is_error;
        }

        return true;
    }

    /*!
    @param prefix  the previously read or set type prefix
    @return whether value creation completed
    */
    bool get_ubjson_value(const char_int_type prefix)
    {
        switch (prefix)
        {
            case char_traits<char_type>::eof():  // EOF
                return unexpect_eof(input_format, "value");

            case 'T':  // true
                return sax->boolean(true);
            case 'F':  // false
                return sax->boolean(false);

            case 'Z':  // null
                return sax->null();

            case 'B':  // byte
            {
                if (input_format != input_format_t::bjdata)
                {
                    break;
                }
                std::uint8_t number{};
                return get_number(input_format, number) && sax->number_unsigned(number);
            }

            case 'U':
            {
                std::uint8_t number{};
                return get_number(input_format, number) && sax->number_unsigned(number);
            }

            case 'i':
            {
                std::int8_t number{};
                return get_number(input_format, number) && sax->number_integer(number);
            }

            case 'I':
            {
                std::int16_t number{};
                return get_number(input_format, number) && sax->number_integer(number);
            }

            case 'l':
            {
                std::int32_t number{};
                return get_number(input_format, number) && sax->number_integer(number);
            }

            case 'L':
            {
                std::int64_t number{};
                return get_number(input_format, number) && sax->number_integer(number);
            }

            case 'u':
            {
                if (input_format != input_format_t::bjdata)
                {
                    break;
                }
                std::uint16_t number{};
                return get_number(input_format, number) && sax->number_unsigned(number);
            }

            case 'm':
            {
                if (input_format != input_format_t::bjdata)
                {
                    break;
                }
                std::uint32_t number{};
                return get_number(input_format, number) && sax->number_unsigned(number);
            }

            case 'M':
            {
                if (input_format != input_format_t::bjdata)
                {
                    break;
                }
                std::uint64_t number{};
                return get_number(input_format, number) && sax->number_unsigned(number);
            }

            case 'h':
            {
                if (input_format != input_format_t::bjdata)
                {
                    break;
                }
                const auto byte1_raw = get();
                if (JSON_HEDLEY_UNLIKELY(!unexpect_eof(input_format, "number")))
                {
                    return false;
                }
                const auto byte2_raw = get();
                if (JSON_HEDLEY_UNLIKELY(!unexpect_eof(input_format, "number")))
                {
                    return false;
                }

                const auto byte1 = static_cast<unsigned char>(byte1_raw);
                const auto byte2 = static_cast<unsigned char>(byte2_raw);

                // code from RFC 7049, Appendix D, Figure 3:
                // As half-precision floating-point numbers were only added
                // to IEEE 754 in 2008, today's programming platforms often
                // still only have limited support for them. It is very
                // easy to include at least decoding support for them even
                // without such support. An example of a small decoder for
                // half-precision floating-point numbers in the C language
                // is shown in Fig. 3.
                const auto half = static_cast<unsigned int>((byte2 << 8u) + byte1);
                const double val = [&half]
                {
                    const int exp = (half >> 10u) & 0x1Fu;
                    const unsigned int mant = half & 0x3FFu;
                    JSON_ASSERT(0 <= exp&& exp <= 32);
                    JSON_ASSERT(mant <= 1024);
                    switch (exp)
                    {
                        case 0:
                            return std::ldexp(mant, -24);
                        case 31:
                            return (mant == 0)
                            ? std::numeric_limits<double>::infinity()
                            : std::numeric_limits<double>::quiet_NaN();
                        default:
                            return std::ldexp(mant + 1024, exp - 25);
                    }
                }();
                return sax->number_float((half & 0x8000u) != 0
                                         ? static_cast<number_float_t>(-val)
                                         : static_cast<number_float_t>(val), "");
            }

            case 'd':
            {
                float number{};
                return get_number(input_format, number) && sax->number_float(static_cast<number_float_t>(number), "");
            }

            case 'D':
            {
                double number{};
                return get_number(input_format, number) && sax->number_float(static_cast<number_float_t>(number), "");
            }

            case 'H':
            {
                return get_ubjson_high_precision_number();
            }

            case 'C':  // char
            {
                get();
                if (JSON_HEDLEY_UNLIKELY(!unexpect_eof(input_format, "char")))
                {
                    return false;
                }
                if (JSON_HEDLEY_UNLIKELY(current > 127))
                {
                    auto last_token = get_token_string();
                    return sax->parse_error(chars_read, last_token, parse_error::create(113, chars_read,
                                            exception_message(input_format, concat("byte after 'C' must be in range 0x00..0x7F; last byte: 0x", last_token), "char"), nullptr));
                }
                string_t s(1, static_cast<typename string_t::value_type>(current));
                return sax->string(s);
            }

            case 'S':  // string
            {
                string_t s;
                return get_ubjson_string(s) && sax->string(s);
            }

            case '[':  // array
                return get_ubjson_array();

            case '{':  // object
                return get_ubjson_object();

            default: // anything else
                break;
        }
        auto last_token = get_token_string();
        return sax->parse_error(chars_read, last_token, parse_error::create(112, chars_read, exception_message(input_format, "invalid byte: 0x" + last_token, "value"), nullptr));
    }

    /*!
    @return whether array creation completed
    */
    bool get_ubjson_array()
    {
        std::pair<std::size_t, char_int_type> size_and_type;
        if (JSON_HEDLEY_UNLIKELY(!get_ubjson_size_type(size_and_type)))
        {
            return false;
        }

        // if bit-8 of size_and_type.second is set to 1, encode bjdata ndarray as an object in JData annotated array format (https://github.com/NeuroJSON/jdata):
        // {"_ArrayType_" : "typeid", "_ArraySize_" : [n1, n2, ...], "_ArrayData_" : [v1, v2, ...]}

        if (input_format == input_format_t::bjdata && size_and_type.first != npos && (size_and_type.second & (1 << 8)) != 0)
        {
            size_and_type.second &= ~(static_cast<char_int_type>(1) << 8);  // use bit 8 to indicate ndarray, here we remove the bit to restore the type marker
            auto it = std::lower_bound(bjd_types_map.begin(), bjd_types_map.end(), size_and_type.second, [](const bjd_type & p, char_int_type t)
            {
                return p.first < t;
            });
            string_t key = "_ArrayType_";
            if (JSON_HEDLEY_UNLIKELY(it == bjd_types_map.end() || it->first != size_and_type.second))
            {
                auto last_token = get_token_string();
                return sax->parse_error(chars_read, last_token, parse_error::create(112, chars_read,
                                        exception_message(input_format, "invalid byte: 0x" + last_token, "type"), nullptr));
            }

            string_t type = it->second; // sax->string() takes a reference
            if (JSON_HEDLEY_UNLIKELY(!sax->key(key) || !sax->string(type)))
            {
                return false;
            }

            if (size_and_type.second == 'C' || size_and_type.second == 'B')
            {
                size_and_type.second = 'U';
            }

            key = "_ArrayData_";
            if (JSON_HEDLEY_UNLIKELY(!sax->key(key) || !sax->start_array(size_and_type.first) ))
            {
                return false;
            }

            for (std::size_t i = 0; i < size_and_type.first; ++i)
            {
                if (JSON_HEDLEY_UNLIKELY(!get_ubjson_value(size_and_type.second)))
                {
                    return false;
                }
            }

            return (sax->end_array() && sax->end_object());
        }

        // If BJData type marker is 'B' decode as binary
        if (input_format == input_format_t::bjdata && size_and_type.first != npos && size_and_type.second == 'B')
        {
            binary_t result;
            return get_binary(input_format, size_and_type.first, result) && sax->binary(result);
        }

        if (size_and_type.first != npos)
        {
            if (JSON_HEDLEY_UNLIKELY(!sax->start_array(size_and_type.first)))
            {
                return false;
            }

            if (size_and_type.second != 0)
            {
                if (size_and_type.second != 'N')
                {
                    for (std::size_t i = 0; i < size_and_type.first; ++i)
                    {
                        if (JSON_HEDLEY_UNLIKELY(!get_ubjson_value(size_and_type.second)))
                        {
                            return false;
                        }
                    }
                }
            }
            else
            {
                for (std::size_t i = 0; i < size_and_type.first; ++i)
                {
                    if (JSON_HEDLEY_UNLIKELY(!parse_ubjson_internal()))
                    {
                        return false;
                    }
                }
            }
        }
        else
        {
            if (JSON_HEDLEY_UNLIKELY(!sax->start_array(detail::unknown_size())))
            {
                return false;
            }

            while (current != ']')
            {
                if (JSON_HEDLEY_UNLIKELY(!parse_ubjson_internal(false)))
                {
                    return false;
                }
                get_ignore_noop();
            }
        }

        return sax->end_array();
    }

    /*!
    @return whether object creation completed
    */
    bool get_ubjson_object()
    {
        std::pair<std::size_t, char_int_type> size_and_type;
        if (JSON_HEDLEY_UNLIKELY(!get_ubjson_size_type(size_and_type)))
        {
            return false;
        }

        // do not accept ND-array size in objects in BJData
        if (input_format == input_format_t::bjdata && size_and_type.first != npos && (size_and_type.second & (1 << 8)) != 0)
        {
            auto last_token = get_token_string();
            return sax->parse_error(chars_read, last_token, parse_error::create(112, chars_read,
                                    exception_message(input_format, "BJData object does not support ND-array size in optimized format", "object"), nullptr));
        }

        string_t key;
        if (size_and_type.first != npos)
        {
            if (JSON_HEDLEY_UNLIKELY(!sax->start_object(size_and_type.first)))
            {
                return false;
            }

            if (size_and_type.second != 0)
            {
                for (std::size_t i = 0; i < size_and_type.first; ++i)
                {
                    if (JSON_HEDLEY_UNLIKELY(!get_ubjson_string(key) || !sax->key(key)))
                    {
                        return false;
                    }
                    if (JSON_HEDLEY_UNLIKELY(!get_ubjson_value(size_and_type.second)))
                    {
                        return false;
                    }
                    key.clear();
                }
            }
            else
            {
                for (std::size_t i = 0; i < size_and_type.first; ++i)
                {
                    if (JSON_HEDLEY_UNLIKELY(!get_ubjson_string(key) || !sax->key(key)))
                    {
                        return false;
                    }
                    if (JSON_HEDLEY_UNLIKELY(!parse_ubjson_internal()))
                    {
                        return false;
                    }
                    key.clear();
                }
            }
        }
        else
        {
            if (JSON_HEDLEY_UNLIKELY(!sax->start_object(detail::unknown_size())))
            {
                return false;
            }

            while (current != '}')
            {
                if (JSON_HEDLEY_UNLIKELY(!get_ubjson_string(key, false) || !sax->key(key)))
                {
                    return false;
                }
                if (JSON_HEDLEY_UNLIKELY(!parse_ubjson_internal()))
                {
                    return false;
                }
                get_ignore_noop();
                key.clear();
            }
        }

        return sax->end_object();
    }

    // Note, no reader for UBJSON binary types is implemented because they do
    // not exist

    bool get_ubjson_high_precision_number()
    {
        // get size of following number string
        std::size_t size{};
        bool no_ndarray = true;
        auto res = get_ubjson_size_value(size, no_ndarray);
        if (JSON_HEDLEY_UNLIKELY(!res))
        {
            return res;
        }

        // get number string
        std::vector<char> number_vector;
        for (std::size_t i = 0; i < size; ++i)
        {
            get();
            if (JSON_HEDLEY_UNLIKELY(!unexpect_eof(input_format, "number")))
            {
                return false;
            }
            number_vector.push_back(static_cast<char>(current));
        }

        // parse number string
        using ia_type = decltype(detail::input_adapter(number_vector));
        auto number_lexer = detail::lexer<BasicJsonType, ia_type>(detail::input_adapter(number_vector), false);
        const auto result_number = number_lexer.scan();
        const auto number_string = number_lexer.get_token_string();
        const auto result_remainder = number_lexer.scan();

        using token_type = typename detail::lexer_base<BasicJsonType>::token_type;

        if (JSON_HEDLEY_UNLIKELY(result_remainder != token_type::end_of_input))
        {
            return sax->parse_error(chars_read, number_string, parse_error::create(115, chars_read,
                                    exception_message(input_format, concat("invalid number text: ", number_lexer.get_token_string()), "high-precision number"), nullptr));
        }

        switch (result_number)
        {
            case token_type::value_integer:
                return sax->number_integer(number_lexer.get_number_integer());
            case token_type::value_unsigned:
                return sax->number_unsigned(number_lexer.get_number_unsigned());
            case token_type::value_float:
                return sax->number_float(number_lexer.get_number_float(), std::move(number_string));
            case token_type::uninitialized:
            case token_type::literal_true:
            case token_type::literal_false:
            case token_type::literal_null:
            case token_type::value_string:
            case token_type::begin_array:
            case token_type::begin_object:
            case token_type::end_array:
            case token_type::end_object:
            case token_type::name_separator:
            case token_type::value_separator:
            case token_type::parse_error:
            case token_type::end_of_input:
            case token_type::literal_or_value:
            default:
                return sax->parse_error(chars_read, number_string, parse_error::create(115, chars_read,
                                        exception_message(input_format, concat("invalid number text: ", number_lexer.get_token_string()), "high-precision number"), nullptr));
        }
    }

    ///////////////////////
    // Utility functions //
    ///////////////////////

    /*!
    @brief get next character from the input

    This function provides the interface to the used input adapter. It does
    not throw in case the input reached EOF, but returns a -'ve valued
    `char_traits<char_type>::eof()` in that case.

    @return character read from the input
    */
    char_int_type get()
    {
        ++chars_read;
        return current = ia.get_character();
    }

    /*!
    @brief get_to read into a primitive type

    This function provides the interface to the used input adapter. It does
    not throw in case the input reached EOF, but returns false instead

    @return bool, whether the read was successful
    */
    template<class T>
    bool get_to(T& dest, const input_format_t format, const char* context)
    {
        auto new_chars_read = ia.get_elements(&dest);
        chars_read += new_chars_read;
        if (JSON_HEDLEY_UNLIKELY(new_chars_read < sizeof(T)))
        {
            // in case of failure, advance position by 1 to report failing location
            ++chars_read;
            sax->parse_error(chars_read, "<end of file>", parse_error::create(110, chars_read, exception_message(format, "unexpected end of input", context), nullptr));
            return false;
        }
        return true;
    }

    /*!
    @return character read from the input after ignoring all 'N' entries
    */
    char_int_type get_ignore_noop()
    {
        do
        {
            get();
        }
        while (current == 'N');

        return current;
    }

    template<class NumberType>
    static void byte_swap(NumberType& number)
    {
        constexpr std::size_t sz = sizeof(number);
#ifdef __cpp_lib_byteswap
        if constexpr (sz == 1)
        {
            return;
        }
        if constexpr(std::is_integral_v<NumberType>)
        {
            number = std::byteswap(number);
            return;
        }
#endif
        auto* ptr = reinterpret_cast<std::uint8_t*>(&number);
        for (std::size_t i = 0; i < sz / 2; ++i)
        {
            std::swap(ptr[i], ptr[sz - i - 1]);
        }
    }

    /*
    @brief read a number from the input

    @tparam NumberType the type of the number
    @param[in] format   the current format (for diagnostics)
    @param[out] result  number of type @a NumberType

    @return whether conversion completed

    @note This function needs to respect the system's endianness, because
          bytes in CBOR, MessagePack, and UBJSON are stored in network order
          (big endian) and therefore need reordering on little endian systems.
          On the other hand, BSON and BJData use little endian and should reorder
          on big endian systems.
    */
    template<typename NumberType, bool InputIsLittleEndian = false>
    bool get_number(const input_format_t format, NumberType& result)
    {
        // read in the original format

        if (JSON_HEDLEY_UNLIKELY(!get_to(result, format, "number")))
        {
            return false;
        }
        if (is_little_endian != (InputIsLittleEndian || format == input_format_t::bjdata))
        {
            byte_swap(result);
        }
        return true;
    }

    /*!
    @brief create a string by reading characters from the input

    @tparam NumberType the type of the number
    @param[in] format the current format (for diagnostics)
    @param[in] len number of characters to read
    @param[out] result string created by reading @a len bytes

    @return whether string creation completed

    @note We can not reserve @a len bytes for the result, because @a len
          may be too large. Usually, @ref unexpect_eof() detects the end of
          the input before we run out of string memory.
    */
    template<typename NumberType>
    bool get_string(const input_format_t format,
                    const NumberType len,
                    string_t& result)
    {
        bool success = true;
        for (NumberType i = 0; i < len; i++)
        {
            get();
            if (JSON_HEDLEY_UNLIKELY(!unexpect_eof(format, "string")))
            {
                success = false;
                break;
            }
            result.push_back(static_cast<typename string_t::value_type>(current));
        }
        return success;
    }

    /*!
    @brief create a byte array by reading bytes from the input

    @tparam NumberType the type of the number
    @param[in] format the current format (for diagnostics)
    @param[in] len number of bytes to read
    @param[out] result byte array created by reading @a len bytes

    @return whether byte array creation completed

    @note We can not reserve @a len bytes for the result, because @a len
          may be too large. Usually, @ref unexpect_eof() detects the end of
          the input before we run out of memory.
    */
    template<typename NumberType>
    bool get_binary(const input_format_t format,
                    const NumberType len,
                    binary_t& result)
    {
        bool success = true;
        for (NumberType i = 0; i < len; i++)
        {
            get();
            if (JSON_HEDLEY_UNLIKELY(!unexpect_eof(format, "binary")))
            {
                success = false;
                break;
            }
            result.push_back(static_cast<std::uint8_t>(current));
        }
        return success;
    }

    /*!
    @param[in] format   the current format (for diagnostics)
    @param[in] context  further context information (for diagnostics)
    @return whether the last read character is not EOF
    */
    JSON_HEDLEY_NON_NULL(3)
    bool unexpect_eof(const input_format_t format, const char* context) const
    {
        if (JSON_HEDLEY_UNLIKELY(current == char_traits<char_type>::eof()))
        {
            return sax->parse_error(chars_read, "<end of file>",
                                    parse_error::create(110, chars_read, exception_message(format, "unexpected end of input", context), nullptr));
        }
        return true;
    }

    /*!
    @return a string representation of the last read byte
    */
    std::string get_token_string() const
    {
        std::array<char, 3> cr{{}};
        static_cast<void>((std::snprintf)(cr.data(), cr.size(), "%.2hhX", static_cast<unsigned char>(current))); // NOLINT(cppcoreguidelines-pro-type-vararg,hicpp-vararg)
        return std::string{cr.data()};
    }

    /*!
    @param[in] format   the current format
    @param[in] detail   a detailed error message
    @param[in] context  further context information
    @return a message string to use in the parse_error exceptions
    */
    std::string exception_message(const input_format_t format,
                                  const std::string& detail,
                                  const std::string& context) const
    {
        std::string error_msg = "syntax error while parsing ";

        switch (format)
        {
            case input_format_t::cbor:
                error_msg += "CBOR";
                break;

            case input_format_t::msgpack:
                error_msg += "MessagePack";
                break;

            case input_format_t::ubjson:
                error_msg += "UBJSON";
                break;

            case input_format_t::bson:
                error_msg += "BSON";
                break;

            case input_format_t::bjdata:
                error_msg += "BJData";
                break;

            case input_format_t::json: // LCOV_EXCL_LINE
            default:            // LCOV_EXCL_LINE
                JSON_ASSERT(false); // NOLINT(cert-dcl03-c,hicpp-static-assert,misc-static-assert) LCOV_EXCL_LINE
        }

        return concat(error_msg, ' ', context, ": ", detail);
    }

  private:
    static JSON_INLINE_VARIABLE constexpr std::size_t npos = detail::unknown_size();

    /// input adapter
    InputAdapterType ia;

    /// the current character
    char_int_type current = char_traits<char_type>::eof();

    /// the number of characters read
    std::size_t chars_read = 0;

    /// whether we can assume little endianness
    const bool is_little_endian = little_endianness();

    /// input format
    const input_format_t input_format = input_format_t::json;

    /// the SAX parser
    json_sax_t* sax = nullptr;

    // excluded markers in bjdata optimized type
#define JSON_BINARY_READER_MAKE_BJD_OPTIMIZED_TYPE_MARKERS_ \
    make_array<char_int_type>('F', 'H', 'N', 'S', 'T', 'Z', '[', '{')

#define JSON_BINARY_READER_MAKE_BJD_TYPES_MAP_ \
    make_array<bjd_type>(                      \
    bjd_type{'B', "byte"},                     \
    bjd_type{'C', "char"},                     \
    bjd_type{'D', "double"},                   \
    bjd_type{'I', "int16"},                    \
    bjd_type{'L', "int64"},                    \
    bjd_type{'M', "uint64"},                   \
    bjd_type{'U', "uint8"},                    \
    bjd_type{'d', "single"},                   \
    bjd_type{'i', "int8"},                     \
    bjd_type{'l', "int32"},                    \
    bjd_type{'m', "uint32"},                   \
    bjd_type{'u', "uint16"})

  JSON_PRIVATE_UNLESS_TESTED:
    // lookup tables
    // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
    const decltype(JSON_BINARY_READER_MAKE_BJD_OPTIMIZED_TYPE_MARKERS_) bjd_optimized_type_markers =
        JSON_BINARY_READER_MAKE_BJD_OPTIMIZED_TYPE_MARKERS_;

    using bjd_type = std::pair<char_int_type, string_t>;
    // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
    const decltype(JSON_BINARY_READER_MAKE_BJD_TYPES_MAP_) bjd_types_map =
        JSON_BINARY_READER_MAKE_BJD_TYPES_MAP_;

#undef JSON_BINARY_READER_MAKE_BJD_OPTIMIZED_TYPE_MARKERS_
#undef JSON_BINARY_READER_MAKE_BJD_TYPES_MAP_
};

#ifndef JSON_HAS_CPP_17
    template<typename BasicJsonType, typename InputAdapterType, typename SAX>
    constexpr std::size_t binary_reader<BasicJsonType, InputAdapterType, SAX>::npos;
#endif

}  // namespace detail
NLOHMANN_JSON_NAMESPACE_END

// #include <nlohmann/detail/input/input_adapters.hpp>

// #include <nlohmann/detail/input/lexer.hpp>

// #include <nlohmann/detail/input/parser.hpp>
//     __ _____ _____ _____
//  __|  |   __|     |   | |  JSON for Modern C++
// |  |  |__   |  |  | | | |  version 3.12.0
// |_____|_____|_____|_|___|  https://github.com/nlohmann/json
//
// SPDX-FileCopyrightText: 2013 - 2025 Niels Lohmann <https://nlohmann.me>
// SPDX-License-Identifier: MIT



#include <cmath> // isfinite
#include <cstdint> // uint8_t
#include <functional> // function
#include <string> // string
#include <utility> // move
#include <vector> // vector

// #include <nlohmann/detail/exceptions.hpp>

// #include <nlohmann/detail/input/input_adapters.hpp>

// #include <nlohmann/detail/input/json_sax.hpp>

// #include <nlohmann/detail/input/lexer.hpp>

// #include <nlohmann/detail/macro_scope.hpp>

// #include <nlohmann/detail/meta/is_sax.hpp>

// #include <nlohmann/detail/string_concat.hpp>

// #include <nlohmann/detail/value_t.hpp>


NLOHMANN_JSON_NAMESPACE_BEGIN
namespace detail
{
////////////
// parser //
////////////

enum class parse_event_t : std::uint8_t
{
    /// the parser read `{` and started to process a JSON object
    object_start,
    /// the parser read `}` and finished processing a JSON object
    object_end,
    /// the parser read `[` and started to process a JSON array
    array_start,
    /// the parser read `]` and finished processing a JSON array
    array_end,
    /// the parser read a key of a value in an object
    key,
    /// the parser finished reading a JSON value
    value
};

template<typename BasicJsonType>
using parser_callback_t =
    std::function<bool(int /*depth*/, parse_event_t /*event*/, BasicJsonType& /*parsed*/)>;

/*!
@brief syntax analysis

This class implements a recursive descent parser.
*/
template<typename BasicJsonType, typename InputAdapterType>
class parser
{
    using number_integer_t = typename BasicJsonType::number_integer_t;
    using number_unsigned_t = typename BasicJsonType::number_unsigned_t;
    using number_float_t = typename BasicJsonType::number_float_t;
    using string_t = typename BasicJsonType::string_t;
    using lexer_t = lexer<BasicJsonType, InputAdapterType>;
    using token_type = typename lexer_t::token_type;

  public:
    /// a parser reading from an input adapter
    explicit parser(InputAdapterType&& adapter,
                    parser_callback_t<BasicJsonType> cb = nullptr,
                    const bool allow_exceptions_ = true,
                    const bool skip_comments = false)
        : callback(std::move(cb))
        , m_lexer(std::move(adapter), skip_comments)
        , allow_exceptions(allow_exceptions_)
    {
        // read first token
        get_token();
    }

    /*!
    @brief public parser interface

    @param[in] strict      whether to expect the last token to be EOF
    @param[in,out] result  parsed JSON value

    @throw parse_error.101 in case of an unexpected token
    @throw parse_error.102 if to_unicode fails or surrogate error
    @throw parse_error.103 if to_unicode fails
    */
    void parse(const bool strict, BasicJsonType& result)
    {
        if (callback)
        {
            json_sax_dom_callback_parser<BasicJsonType, InputAdapterType> sdp(result, callback, allow_exceptions, &m_lexer);
            sax_parse_internal(&sdp);

            // in strict mode, input must be completely read
            if (strict && (get_token() != token_type::end_of_input))
            {
                sdp.parse_error(m_lexer.get_position(),
                                m_lexer.get_token_string(),
                                parse_error::create(101, m_lexer.get_position(),
                                                    exception_message(token_type::end_of_input, "value"), nullptr));
            }

            // in case of an error, return discarded value
            if (sdp.is_errored())
            {
                result = value_t::discarded;
                return;
            }

            // set top-level value to null if it was discarded by the callback
            // function
            if (result.is_discarded())
            {
                result = nullptr;
            }
        }
        else
        {
            json_sax_dom_parser<BasicJsonType, InputAdapterType> sdp(result, allow_exceptions, &m_lexer);
            sax_parse_internal(&sdp);

            // in strict mode, input must be completely read
            if (strict && (get_token() != token_type::end_of_input))
            {
                sdp.parse_error(m_lexer.get_position(),
                                m_lexer.get_token_string(),
                                parse_error::create(101, m_lexer.get_position(), exception_message(token_type::end_of_input, "value"), nullptr));
            }

            // in case of an error, return discarded value
            if (sdp.is_errored())
            {
                result = value_t::discarded;
                return;
            }
        }

        result.assert_invariant();
    }

    /*!
    @brief public accept interface

    @param[in] strict  whether to expect the last token to be EOF
    @return whether the input is a proper JSON text
    */
    bool accept(const bool strict = true)
    {
        json_sax_acceptor<BasicJsonType> sax_acceptor;
        return sax_parse(&sax_acceptor, strict);
    }

    template<typename SAX>
    JSON_HEDLEY_NON_NULL(2)
    bool sax_parse(SAX* sax, const bool strict = true)
    {
        (void)detail::is_sax_static_asserts<SAX, BasicJsonType> {};
        const bool result = sax_parse_internal(sax);

        // strict mode: next byte must be EOF
        if (result && strict && (get_token() != token_type::end_of_input))
        {
            return sax->parse_error(m_lexer.get_position(),
                                    m_lexer.get_token_string(),
                                    parse_error::create(101, m_lexer.get_position(), exception_message(token_type::end_of_input, "value"), nullptr));
        }

        return result;
    }

  private:
    template<typename SAX>
    JSON_HEDLEY_NON_NULL(2)
    bool sax_parse_internal(SAX* sax)
    {
        // stack to remember the hierarchy of structured values we are parsing
        // true = array; false = object
        std::vector<bool> states;
        // value to avoid a goto (see comment where set to true)
        bool skip_to_state_evaluation = false;

        while (true)
        {
            if (!skip_to_state_evaluation)
            {
                // invariant: get_token() was called before each iteration
                switch (last_token)
                {
                    case token_type::begin_object:
                    {
                        if (JSON_HEDLEY_UNLIKELY(!sax->start_object(detail::unknown_size())))
                        {
                            return false;
                        }

                        // closing } -> we are done
                        if (get_token() == token_type::end_object)
                        {
                            if (JSON_HEDLEY_UNLIKELY(!sax->end_object()))
                            {
                                return false;
                            }
                            break;
                        }

                        // parse key
                        if (JSON_HEDLEY_UNLIKELY(last_token != token_type::value_string))
                        {
                            return sax->parse_error(m_lexer.get_position(),
                                                    m_lexer.get_token_string(),
                                                    parse_error::create(101, m_lexer.get_position(), exception_message(token_type::value_string, "object key"), nullptr));
                        }
                        if (JSON_HEDLEY_UNLIKELY(!sax->key(m_lexer.get_string())))
                        {
                            return false;
                        }

                        // parse separator (:)
                        if (JSON_HEDLEY_UNLIKELY(get_token() != token_type::name_separator))
                        {
                            return sax->parse_error(m_lexer.get_position(),
                                                    m_lexer.get_token_string(),
                                                    parse_error::create(101, m_lexer.get_position(), exception_message(token_type::name_separator, "object separator"), nullptr));
                        }

                        // remember we are now inside an object
                        states.push_back(false);

                        // parse values
                        get_token();
                        continue;
                    }

                    case token_type::begin_array:
                    {
                        if (JSON_HEDLEY_UNLIKELY(!sax->start_array(detail::unknown_size())))
                        {
                            return false;
                        }

                        // closing ] -> we are done
                        if (get_token() == token_type::end_array)
                        {
                            if (JSON_HEDLEY_UNLIKELY(!sax->end_array()))
                            {
                                return false;
                            }
                            break;
                        }

                        // remember we are now inside an array
                        states.push_back(true);

                        // parse values (no need to call get_token)
                        continue;
                    }

                    case token_type::value_float:
                    {
                        const auto res = m_lexer.get_number_float();

                        if (JSON_HEDLEY_UNLIKELY(!std::isfinite(res)))
                        {
                            return sax->parse_error(m_lexer.get_position(),
                                                    m_lexer.get_token_string(),
                                                    out_of_range::create(406, concat("number overflow parsing '", m_lexer.get_token_string(), '\''), nullptr));
                        }

                        if (JSON_HEDLEY_UNLIKELY(!sax->number_float(res, m_lexer.get_string())))
                        {
                            return false;
                        }

                        break;
                    }

                    case token_type::literal_false:
                    {
                        if (JSON_HEDLEY_UNLIKELY(!sax->boolean(false)))
                        {
                            return false;
                        }
                        break;
                    }

                    case token_type::literal_null:
                    {
                        if (JSON_HEDLEY_UNLIKELY(!sax->null()))
                        {
                            return false;
                        }
                        break;
                    }

                    case token_type::literal_true:
                    {
                        if (JSON_HEDLEY_UNLIKELY(!sax->boolean(true)))
                        {
                            return false;
                        }
                        break;
                    }

                    case token_type::value_integer:
                    {
                        if (JSON_HEDLEY_UNLIKELY(!sax->number_integer(m_lexer.get_number_integer())))
                        {
                            return false;
                        }
                        break;
                    }

                    case token_type::value_string:
                    {
                        if (JSON_HEDLEY_UNLIKELY(!sax->string(m_lexer.get_string())))
                        {
                            return false;
                        }
                        break;
                    }

                    case token_type::value_unsigned:
                    {
                        if (JSON_HEDLEY_UNLIKELY(!sax->number_unsigned(m_lexer.get_number_unsigned())))
                        {
                            return false;
                        }
                        break;
                    }

                    case token_type::parse_error:
                    {
                        // using "uninitialized" to avoid "expected" message
                        return sax->parse_error(m_lexer.get_position(),
                                                m_lexer.get_token_string(),
                                                parse_error::create(101, m_lexer.get_position(), exception_message(token_type::uninitialized, "value"), nullptr));
                    }
                    case token_type::end_of_input:
                    {
                        if (JSON_HEDLEY_UNLIKELY(m_lexer.get_position().chars_read_total == 1))
                        {
                            return sax->parse_error(m_lexer.get_position(),
                                                    m_lexer.get_token_string(),
                                                    parse_error::create(101, m_lexer.get_position(),
                                                            "attempting to parse an empty input; check that your input string or stream contains the expected JSON", nullptr));
                        }

                        return sax->parse_error(m_lexer.get_position(),
                                                m_lexer.get_token_string(),
                                                parse_error::create(101, m_lexer.get_position(), exception_message(token_type::literal_or_value, "value"), nullptr));
                    }
                    case token_type::uninitialized:
                    case token_type::end_array:
                    case token_type::end_object:
                    case token_type::name_separator:
                    case token_type::value_separator:
                    case token_type::literal_or_value:
                    default: // the last token was unexpected
                    {
                        return sax->parse_error(m_lexer.get_position(),
                                                m_lexer.get_token_string(),
                                                parse_error::create(101, m_lexer.get_position(), exception_message(token_type::literal_or_value, "value"), nullptr));
                    }
                }
            }
            else
            {
                skip_to_state_evaluation = false;
            }

            // we reached this line after we successfully parsed a value
            if (states.empty())
            {
                // empty stack: we reached the end of the hierarchy: done
                return true;
            }

            if (states.back())  // array
            {
                // comma -> next value
                if (get_token() == token_type::value_separator)
                {
                    // parse a new value
                    get_token();
                    continue;
                }

                // closing ]
                if (JSON_HEDLEY_LIKELY(last_token == token_type::end_array))
                {
                    if (JSON_HEDLEY_UNLIKELY(!sax->end_array()))
                    {
                        return false;
                    }

                    // We are done with this array. Before we can parse a
                    // new value, we need to evaluate the new state first.
                    // By setting skip_to_state_evaluation to false, we
                    // are effectively jumping to the beginning of this if.
                    JSON_ASSERT(!states.empty());
                    states.pop_back();
                    skip_to_state_evaluation = true;
                    continue;
                }

                return sax->parse_error(m_lexer.get_position(),
                                        m_lexer.get_token_string(),
                                        parse_error::create(101, m_lexer.get_position(), exception_message(token_type::end_array, "array"), nullptr));
            }

            // states.back() is false -> object

            // comma -> next value
            if (get_token() == token_type::value_separator)
            {
                // parse key
                if (JSON_HEDLEY_UNLIKELY(get_token() != token_type::value_string))
                {
                    return sax->parse_error(m_lexer.get_position(),
                                            m_lexer.get_token_string(),
                                            parse_error::create(101, m_lexer.get_position(), exception_message(token_type::value_string, "object key"), nullptr));
                }

                if (JSON_HEDLEY_UNLIKELY(!sax->key(m_lexer.get_string())))
                {
                    return false;
                }

                // parse separator (:)
                if (JSON_HEDLEY_UNLIKELY(get_token() != token_type::name_separator))
                {
                    return sax->parse_error(m_lexer.get_position(),
                                            m_lexer.get_token_string(),
                                            parse_error::create(101, m_lexer.get_position(), exception_message(token_type::name_separator, "object separator"), nullptr));
                }

                // parse values
                get_token();
                continue;
            }

            // closing }
            if (JSON_HEDLEY_LIKELY(last_token == token_type::end_object))
            {
                if (JSON_HEDLEY_UNLIKELY(!sax->end_object()))
                {
                    return false;
                }

                // We are done with this object. Before we can parse a
                // new value, we need to evaluate the new state first.
                // By setting skip_to_state_evaluation to false, we
                // are effectively jumping to the beginning of this if.
                JSON_ASSERT(!states.empty());
                states.pop_back();
                skip_to_state_evaluation = true;
                continue;
            }

            return sax->parse_error(m_lexer.get_position(),
                                    m_lexer.get_token_string(),
                                    parse_error::create(101, m_lexer.get_position(), exception_message(token_type::end_object, "object"), nullptr));
        }
    }

    /// get next token from lexer
    token_type get_token()
    {
        return last_token = m_lexer.scan();
    }

    std::string exception_message(const token_type expected, const std::string& context)
    {
        std::string error_msg = "syntax error ";

        if (!context.empty())
        {
            error_msg += concat("while parsing ", context, ' ');
        }

        error_msg += "- ";

        if (last_token == token_type::parse_error)
        {
            error_msg += concat(m_lexer.get_error_message(), "; last read: '",
                                m_lexer.get_token_string(), '\'');
        }
        else
        {
            error_msg += concat("unexpected ", lexer_t::token_type_name(last_token));
        }

        if (expected != token_type::uninitialized)
        {
            error_msg += concat("; expected ", lexer_t::token_type_name(expected));
        }

        return error_msg;
    }

  private:
    /// callback function
    const parser_callback_t<BasicJsonType> callback = nullptr;
    /// the type of the last read token
    token_type last_token = token_type::uninitialized;
    /// the lexer
    lexer_t m_lexer;
    /// whether to throw exceptions in case of errors
    const bool allow_exceptions = true;
};

}  // namespace detail
NLOHMANN_JSON_NAMESPACE_END

// #include <nlohmann/detail/iterators/internal_iterator.hpp>
//     __ _____ _____ _____
//  __|  |   __|     |   | |  JSON for Modern C++
// |  |  |__   |  |  | | | |  version 3.12.0
// |_____|_____|_____|_|___|  https://github.com/nlohmann/json
//
// SPDX-FileCopyrightText: 2013 - 2025 Niels Lohmann <https://nlohmann.me>
// SPDX-License-Identifier: MIT



// #include <nlohmann/detail/abi_macros.hpp>

// #include <nlohmann/detail/iterators/primitive_iterator.hpp>
//     __ _____ _____ _____
//  __|  |   __|     |   | |  JSON for Modern C++
// |  |  |__   |  |  | | | |  version 3.12.0
// |_____|_____|_____|_|___|  https://github.com/nlohmann/json
//
// SPDX-FileCopyrightText: 2013 - 2025 Niels Lohmann <https://nlohmann.me>
// SPDX-License-Identifier: MIT



#include <cstddef> // ptrdiff_t
#include <limits>  // numeric_limits

// #include <nlohmann/detail/macro_scope.hpp>


NLOHMANN_JSON_NAMESPACE_BEGIN
namespace detail
{

/*
@brief an iterator for primitive JSON types

This class models an iterator for primitive JSON types (boolean, number,
string). It's only purpose is to allow the iterator/const_iterator classes
to "iterate" over primitive values. Internally, the iterator is modeled by
a `difference_type` variable. Value begin_value (`0`) models the begin,
end_value (`1`) models past the end.
*/
class primitive_iterator_t
{
  private:
    using difference_type = std::ptrdiff_t;
    static constexpr difference_type begin_value = 0;
    static constexpr difference_type end_value = begin_value + 1;

  JSON_PRIVATE_UNLESS_TESTED:
    /// iterator as signed integer type
    difference_type m_it = (std::numeric_limits<std::ptrdiff_t>::min)();

  public:
    constexpr difference_type get_value() const noexcept
    {
        return m_it;
    }

    /// set iterator to a defined beginning
    void set_begin() noexcept
    {
        m_it = begin_value;
    }

    /// set iterator to a defined past the end
    void set_end() noexcept
    {
        m_it = end_value;
    }

    /// return whether the iterator can be dereferenced
    constexpr bool is_begin() const noexcept
    {
        return m_it == begin_value;
    }

    /// return whether the iterator is at end
    constexpr bool is_end() const noexcept
    {
        return m_it == end_value;
    }

    friend constexpr bool operator==(primitive_iterator_t lhs, primitive_iterator_t rhs) noexcept
    {
        return lhs.m_it == rhs.m_it;
    }

    friend constexpr bool operator<(primitive_iterator_t lhs, primitive_iterator_t rhs) noexcept
    {
        return lhs.m_it < rhs.m_it;
    }

    primitive_iterator_t operator+(difference_type n) noexcept
    {
        auto result = *this;
        result += n;
        return result;
    }

    friend constexpr difference_type operator-(primitive_iterator_t lhs, primitive_iterator_t rhs) noexcept
    {
        return lhs.m_it - rhs.m_it;
    }

    primitive_iterator_t& operator++() noexcept
    {
        ++m_it;
        return *this;
    }

    primitive_iterator_t operator++(int)& noexcept // NOLINT(cert-dcl21-cpp)
    {
        auto result = *this;
        ++m_it;
        return result;
    }

    primitive_iterator_t& operator--() noexcept
    {
        --m_it;
        return *this;
    }

    primitive_iterator_t operator--(int)& noexcept // NOLINT(cert-dcl21-cpp)
    {
        auto result = *this;
        --m_it;
        return result;
    }

    primitive_iterator_t& operator+=(difference_type n) noexcept
    {
        m_it += n;
        return *this;
    }

    primitive_iterator_t& operator-=(difference_type n) noexcept
    {
        m_it -= n;
        return *this;
    }
};

}  // namespace detail
NLOHMANN_JSON_NAMESPACE_END


NLOHMANN_JSON_NAMESPACE_BEGIN
namespace detail
{

/*!
@brief an iterator value

@note This structure could easily be a union, but MSVC currently does not allow
unions members with complex constructors, see https://github.com/nlohmann/json/pull/105.
*/
template<typename BasicJsonType> struct internal_iterator
{
    /// iterator for JSON objects
    typename BasicJsonType::object_t::iterator object_iterator {};
    /// iterator for JSON arrays
    typename BasicJsonType::array_t::iterator array_iterator {};
    /// generic iterator for all other types
    primitive_iterator_t primitive_iterator {};
};

}  // namespace detail
NLOHMANN_JSON_NAMESPACE_END

// #include <nlohmann/detail/iterators/iter_impl.hpp>
//     __ _____ _____ _____
//  __|  |   __|     |   | |  JSON for Modern C++
// |  |  |__   |  |  | | | |  version 3.12.0
// |_____|_____|_____|_|___|  https://github.com/nlohmann/json
//
// SPDX-FileCopyrightText: 2013 - 2025 Niels Lohmann <https://nlohmann.me>
// SPDX-License-Identifier: MIT



#include <iterator> // iterator, random_access_iterator_tag, bidirectional_iterator_tag, advance, next
#include <type_traits> // conditional, is_const, remove_const

// #include <nlohmann/detail/exceptions.hpp>

// #include <nlohmann/detail/iterators/internal_iterator.hpp>

// #include <nlohmann/detail/iterators/primitive_iterator.hpp>

// #include <nlohmann/detail/macro_scope.hpp>

// #include <nlohmann/detail/meta/cpp_future.hpp>

// #include <nlohmann/detail/meta/type_traits.hpp>

// #include <nlohmann/detail/value_t.hpp>


NLOHMANN_JSON_NAMESPACE_BEGIN
namespace detail
{

// forward declare, to be able to friend it later on
template<typename IteratorType> class iteration_proxy;
template<typename IteratorType> class iteration_proxy_value;

/*!
@brief a template for a bidirectional iterator for the @ref basic_json class
This class implements a both iterators (iterator and const_iterator) for the
@ref basic_json class.
@note An iterator is called *initialized* when a pointer to a JSON value has
      been set (e.g., by a constructor or a copy assignment). If the iterator is
      default-constructed, it is *uninitialized* and most methods are undefined.
      **The library uses assertions to detect calls on uninitialized iterators.**
@requirement The class satisfies the following concept requirements:
-
[BidirectionalIterator](https://en.cppreference.com/w/cpp/named_req/BidirectionalIterator):
  The iterator that can be moved can be moved in both directions (i.e.
  incremented and decremented).
@since version 1.0.0, simplified in version 2.0.9, change to bidirectional
       iterators in version 3.0.0 (see https://github.com/nlohmann/json/issues/593)
*/
template<typename BasicJsonType>
class iter_impl // NOLINT(cppcoreguidelines-special-member-functions,hicpp-special-member-functions)
{
    /// the iterator with BasicJsonType of different const-ness
    using other_iter_impl = iter_impl<typename std::conditional<std::is_const<BasicJsonType>::value, typename std::remove_const<BasicJsonType>::type, const BasicJsonType>::type>;
    /// allow basic_json to access private members
    friend other_iter_impl;
    friend BasicJsonType;
    friend iteration_proxy<iter_impl>;
    friend iteration_proxy_value<iter_impl>;

    using object_t = typename BasicJsonType::object_t;
    using array_t = typename BasicJsonType::array_t;
    // make sure BasicJsonType is basic_json or const basic_json
    static_assert(is_basic_json<typename std::remove_const<BasicJsonType>::type>::value,
                  "iter_impl only accepts (const) basic_json");
    // superficial check for the LegacyBidirectionalIterator named requirement
    static_assert(std::is_base_of<std::bidirectional_iterator_tag, std::bidirectional_iterator_tag>::value
                  &&  std::is_base_of<std::bidirectional_iterator_tag, typename std::iterator_traits<typename array_t::iterator>::iterator_category>::value,
                  "basic_json iterator assumes array and object type iterators satisfy the LegacyBidirectionalIterator named requirement.");

  public:
    /// The std::iterator class template (used as a base class to provide typedefs) is deprecated in C++17.
    /// The C++ Standard has never required user-defined iterators to derive from std::iterator.
    /// A user-defined iterator should provide publicly accessible typedefs named
    /// iterator_category, value_type, difference_type, pointer, and reference.
    /// Note that value_type is required to be non-const, even for constant iterators.
    using iterator_category = std::bidirectional_iterator_tag;

    /// the type of the values when the iterator is dereferenced
    using value_type = typename BasicJsonType::value_type;
    /// a type to represent differences between iterators
    using difference_type = typename BasicJsonType::difference_type;
    /// defines a pointer to the type iterated over (value_type)
    using pointer = typename std::conditional<std::is_const<BasicJsonType>::value,
          typename BasicJsonType::const_pointer,
          typename BasicJsonType::pointer>::type;
    /// defines a reference to the type iterated over (value_type)
    using reference =
        typename std::conditional<std::is_const<BasicJsonType>::value,
        typename BasicJsonType::const_reference,
        typename BasicJsonType::reference>::type;

    iter_impl() = default;
    ~iter_impl() = default;
    iter_impl(iter_impl&&) noexcept = default;
    iter_impl& operator=(iter_impl&&) noexcept = default;

    /*!
    @brief constructor for a given JSON instance
    @param[in] object  pointer to a JSON object for this iterator
    @pre object != nullptr
    @post The iterator is initialized; i.e. `m_object != nullptr`.
    */
    explicit iter_impl(pointer object) noexcept : m_object(object)
    {
        JSON_ASSERT(m_object != nullptr);

        switch (m_object->m_data.m_type)
        {
            case value_t::object:
            {
                m_it.object_iterator = typename object_t::iterator();
                break;
            }

            case value_t::array:
            {
                m_it.array_iterator = typename array_t::iterator();
                break;
            }

            case value_t::null:
            case value_t::string:
            case value_t::boolean:
            case value_t::number_integer:
            case value_t::number_unsigned:
            case value_t::number_float:
            case value_t::binary:
            case value_t::discarded:
            default:
            {
                m_it.primitive_iterator = primitive_iterator_t();
                break;
            }
        }
    }

    /*!
    @note The conventional copy constructor and copy assignment are implicitly
          defined. Combined with the following converting constructor and
          assignment, they support: (1) copy from iterator to iterator, (2)
          copy from const iterator to const iterator, and (3) conversion from
          iterator to const iterator. However conversion from const iterator
          to iterator is not defined.
    */

    /*!
    @brief const copy constructor
    @param[in] other const iterator to copy from
    @note This copy constructor had to be defined explicitly to circumvent a bug
          occurring on msvc v19.0 compiler (VS 2015) debug build. For more
          information refer to: https://github.com/nlohmann/json/issues/1608
    */
    iter_impl(const iter_impl<const BasicJsonType>& other) noexcept
        : m_object(other.m_object), m_it(other.m_it)
    {}

    /*!
    @brief converting assignment
    @param[in] other const iterator to copy from
    @return const/non-const iterator
    @note It is not checked whether @a other is initialized.
    */
    iter_impl& operator=(const iter_impl<const BasicJsonType>& other) noexcept
    {
        if (&other != this)
        {
            m_object = other.m_object;
            m_it = other.m_it;
        }
        return *this;
    }

    /*!
    @brief converting constructor
    @param[in] other  non-const iterator to copy from
    @note It is not checked whether @a other is initialized.
    */
    iter_impl(const iter_impl<typename std::remove_const<BasicJsonType>::type>& other) noexcept
        : m_object(other.m_object), m_it(other.m_it)
    {}

    /*!
    @brief converting assignment
    @param[in] other  non-const iterator to copy from
    @return const/non-const iterator
    @note It is not checked whether @a other is initialized.
    */
    iter_impl& operator=(const iter_impl<typename std::remove_const<BasicJsonType>::type>& other) noexcept // NOLINT(cert-oop54-cpp)
    {
        m_object = other.m_object;
        m_it = other.m_it;
        return *this;
    }

  JSON_PRIVATE_UNLESS_TESTED:
    /*!
    @brief set the iterator to the first value
    @pre The iterator is initialized; i.e. `m_object != nullptr`.
    */
    void set_begin() noexcept
    {
        JSON_ASSERT(m_object != nullptr);

        switch (m_object->m_data.m_type)
        {
            case value_t::object:
            {
                m_it.object_iterator = m_object->m_data.m_value.object->begin();
                break;
            }

            case value_t::array:
            {
                m_it.array_iterator = m_object->m_data.m_value.array->begin();
                break;
            }

            case value_t::null:
            {
                // set to end so begin()==end() is true: null is empty
                m_it.primitive_iterator.set_end();
                break;
            }

            case value_t::string:
            case value_t::boolean:
            case value_t::number_integer:
            case value_t::number_unsigned:
            case value_t::number_float:
            case value_t::binary:
            case value_t::discarded:
            default:
            {
                m_it.primitive_iterator.set_begin();
                break;
            }
        }
    }

    /*!
    @brief set the iterator past the last value
    @pre The iterator is initialized; i.e. `m_object != nullptr`.
    */
    void set_end() noexcept
    {
        JSON_ASSERT(m_object != nullptr);

        switch (m_object->m_data.m_type)
        {
            case value_t::object:
            {
                m_it.object_iterator = m_object->m_data.m_value.object->end();
                break;
            }

            case value_t::array:
            {
                m_it.array_iterator = m_object->m_data.m_value.array->end();
                break;
            }

            case value_t::null:
            case value_t::string:
            case value_t::boolean:
            case value_t::number_integer:
            case value_t::number_unsigned:
            case value_t::number_float:
            case value_t::binary:
            case value_t::discarded:
            default:
            {
                m_it.primitive_iterator.set_end();
                break;
            }
        }
    }

  public:
    /*!
    @brief return a reference to the value pointed to by the iterator
    @pre The iterator is initialized; i.e. `m_object != nullptr`.
    */
    reference operator*() const
    {
        JSON_ASSERT(m_object != nullptr);

        switch (m_object->m_data.m_type)
        {
            case value_t::object:
            {
                JSON_ASSERT(m_it.object_iterator != m_object->m_data.m_value.object->end());
                return m_it.object_iterator->second;
            }

            case value_t::array:
            {
                JSON_ASSERT(m_it.array_iterator != m_object->m_data.m_value.array->end());
                return *m_it.array_iterator;
            }

            case value_t::null:
                JSON_THROW(invalid_iterator::create(214, "cannot get value", m_object));

            case value_t::string:
            case value_t::boolean:
            case value_t::number_integer:
            case value_t::number_unsigned:
            case value_t::number_float:
            case value_t::binary:
            case value_t::discarded:
            default:
            {
                if (JSON_HEDLEY_LIKELY(m_it.primitive_iterator.is_begin()))
                {
                    return *m_object;
                }

                JSON_THROW(invalid_iterator::create(214, "cannot get value", m_object));
            }
        }
    }

    /*!
    @brief dereference the iterator
    @pre The iterator is initialized; i.e. `m_object != nullptr`.
    */
    pointer operator->() const
    {
        JSON_ASSERT(m_object != nullptr);

        switch (m_object->m_data.m_type)
        {
            case value_t::object:
            {
                JSON_ASSERT(m_it.object_iterator != m_object->m_data.m_value.object->end());
                return &(m_it.object_iterator->second);
            }

            case value_t::array:
            {
                JSON_ASSERT(m_it.array_iterator != m_object->m_data.m_value.array->end());
                return &*m_it.array_iterator;
            }

            case value_t::null:
            case value_t::string:
            case value_t::boolean:
            case value_t::number_integer:
            case value_t::number_unsigned:
            case value_t::number_float:
            case value_t::binary:
            case value_t::discarded:
            default:
            {
                if (JSON_HEDLEY_LIKELY(m_it.primitive_iterator.is_begin()))
                {
                    return m_object;
                }

                JSON_THROW(invalid_iterator::create(214, "cannot get value", m_object));
            }
        }
    }

    /*!
    @brief post-increment (it++)
    @pre The iterator is initialized; i.e. `m_object != nullptr`.
    */
    iter_impl operator++(int)& // NOLINT(cert-dcl21-cpp)
    {
        auto result = *this;
        ++(*this);
        return result;
    }

    /*!
    @brief pre-increment (++it)
    @pre The iterator is initialized; i.e. `m_object != nullptr`.
    */
    iter_impl& operator++()
    {
        JSON_ASSERT(m_object != nullptr);

        switch (m_object->m_data.m_type)
        {
            case value_t::object:
            {
                std::advance(m_it.object_iterator, 1);
                break;
            }

            case value_t::array:
            {
                std::advance(m_it.array_iterator, 1);
                break;
            }

            case value_t::null:
            case value_t::string:
            case value_t::boolean:
            case value_t::number_integer:
            case value_t::number_unsigned:
            case value_t::number_float:
            case value_t::binary:
            case value_t::discarded:
            default:
            {
                ++m_it.primitive_iterator;
                break;
            }
        }

        return *this;
    }

    /*!
    @brief post-decrement (it--)
    @pre The iterator is initialized; i.e. `m_object != nullptr`.
    */
    iter_impl operator--(int)& // NOLINT(cert-dcl21-cpp)
    {
        auto result = *this;
        --(*this);
        return result;
    }

    /*!
    @brief pre-decrement (--it)
    @pre The iterator is initialized; i.e. `m_object != nullptr`.
    */
    iter_impl& operator--()
    {
        JSON_ASSERT(m_object != nullptr);

        switch (m_object->m_data.m_type)
        {
            case value_t::object:
            {
                std::advance(m_it.object_iterator, -1);
                break;
            }

            case value_t::array:
            {
                std::advance(m_it.array_iterator, -1);
                break;
            }

            case value_t::null:
            case value_t::string:
            case value_t::boolean:
            case value_t::number_integer:
            case value_t::number_unsigned:
            case value_t::number_float:
            case value_t::binary:
            case value_t::discarded:
            default:
            {
                --m_it.primitive_iterator;
                break;
            }
        }

        return *this;
    }

    /*!
    @brief comparison: equal
    @pre (1) Both iterators are initialized to point to the same object, or (2) both iterators are value-initialized.
    */
    template < typename IterImpl, detail::enable_if_t < (std::is_same<IterImpl, iter_impl>::value || std::is_same<IterImpl, other_iter_impl>::value), std::nullptr_t > = nullptr >
    bool operator==(const IterImpl& other) const
    {
        // if objects are not the same, the comparison is undefined
        if (JSON_HEDLEY_UNLIKELY(m_object != other.m_object))
        {
            JSON_THROW(invalid_iterator::create(212, "cannot compare iterators of different containers", m_object));
        }

        // value-initialized forward iterators can be compared, and must compare equal to other value-initialized iterators of the same type #4493
        if (m_object == nullptr)
        {
            return true;
        }

        switch (m_object->m_data.m_type)
        {
            case value_t::object:
                return (m_it.object_iterator == other.m_it.object_iterator);

            case value_t::array:
                return (m_it.array_iterator == other.m_it.array_iterator);

            case value_t::null:
            case value_t::string:
            case value_t::boolean:
            case value_t::number_integer:
            case value_t::number_unsigned:
            case value_t::number_float:
            case value_t::binary:
            case value_t::discarded:
            default:
                return (m_it.primitive_iterator == other.m_it.primitive_iterator);
        }
    }

    /*!
    @brief comparison: not equal
    @pre (1) Both iterators are initialized to point to the same object, or (2) both iterators are value-initialized.
    */
    template < typename IterImpl, detail::enable_if_t < (std::is_same<IterImpl, iter_impl>::value || std::is_same<IterImpl, other_iter_impl>::value), std::nullptr_t > = nullptr >
    bool operator!=(const IterImpl& other) const
    {
        return !operator==(other);
    }

    /*!
    @brief comparison: smaller
    @pre (1) Both iterators are initialized to point to the same object, or (2) both iterators are value-initialized.
    */
    bool operator<(const iter_impl& other) const
    {
        // if objects are not the same, the comparison is undefined
        if (JSON_HEDLEY_UNLIKELY(m_object != other.m_object))
        {
            JSON_THROW(invalid_iterator::create(212, "cannot compare iterators of different containers", m_object));
        }

        // value-initialized forward iterators can be compared, and must compare equal to other value-initialized iterators of the same type #4493
        if (m_object == nullptr)
        {
            // the iterators are both value-initialized and are to be considered equal, but this function checks for smaller, so we return false
            return false;
        }

        switch (m_object->m_data.m_type)
        {
            case value_t::object:
                JSON_THROW(invalid_iterator::create(213, "cannot compare order of object iterators", m_object));

            case value_t::array:
                return (m_it.array_iterator < other.m_it.array_iterator);

            case value_t::null:
            case value_t::string:
            case value_t::boolean:
            case value_t::number_integer:
            case value_t::number_unsigned:
            case value_t::number_float:
            case value_t::binary:
            case value_t::discarded:
            default:
                return (m_it.primitive_iterator < other.m_it.primitive_iterator);
        }
    }

    /*!
    @brief comparison: less than or equal
    @pre (1) Both iterators are initialized to point to the same object, or (2) both iterators are value-initialized.
    */
    bool operator<=(const iter_impl& other) const
    {
        return !other.operator < (*this);
    }

    /*!
    @brief comparison: greater than
    @pre (1) Both iterators are initialized to point to the same object, or (2) both iterators are value-initialized.
    */
    bool operator>(const iter_impl& other) const
    {
        return !operator<=(other);
    }

    /*!
    @brief comparison: greater than or equal
    @pre (1) The iterator is initialized; i.e. `m_object != nullptr`, or (2) both iterators are value-initialized.
    */
    bool operator>=(const iter_impl& other) const
    {
        return !operator<(other);
    }

    /*!
    @brief add to iterator
    @pre The iterator is initialized; i.e. `m_object != nullptr`.
    */
    iter_impl& operator+=(difference_type i)
    {
        JSON_ASSERT(m_object != nullptr);

        switch (m_object->m_data.m_type)
        {
            case value_t::object:
                JSON_THROW(invalid_iterator::create(209, "cannot use offsets with object iterators", m_object));

            case value_t::array:
            {
                std::advance(m_it.array_iterator, i);
                break;
            }

            case value_t::null:
            case value_t::string:
            case value_t::boolean:
            case value_t::number_integer:
            case value_t::number_unsigned:
            case value_t::number_float:
            case value_t::binary:
            case value_t::discarded:
            default:
            {
                m_it.primitive_iterator += i;
                break;
            }
        }

        return *this;
    }

    /*!
    @brief subtract from iterator
    @pre The iterator is initialized; i.e. `m_object != nullptr`.
    */
    iter_impl& operator-=(difference_type i)
    {
        return operator+=(-i);
    }

    /*!
    @brief add to iterator
    @pre The iterator is initialized; i.e. `m_object != nullptr`.
    */
    iter_impl operator+(difference_type i) const
    {
        auto result = *this;
        result += i;
        return result;
    }

    /*!
    @brief addition of distance and iterator
    @pre The iterator is initialized; i.e. `m_object != nullptr`.
    */
    friend iter_impl operator+(difference_type i, const iter_impl& it)
    {
        auto result = it;
        result += i;
        return result;
    }

    /*!
    @brief subtract from iterator
    @pre The iterator is initialized; i.e. `m_object != nullptr`.
    */
    iter_impl operator-(difference_type i) const
    {
        auto result = *this;
        result -= i;
        return result;
    }

    /*!
    @brief return difference
    @pre The iterator is initialized; i.e. `m_object != nullptr`.
    */
    difference_type operator-(const iter_impl& other) const
    {
        JSON_ASSERT(m_object != nullptr);

        switch (m_object->m_data.m_type)
        {
            case value_t::object:
                JSON_THROW(invalid_iterator::create(209, "cannot use offsets with object iterators", m_object));

            case value_t::array:
                return m_it.array_iterator - other.m_it.array_iterator;

            case value_t::null:
            case value_t::string:
            case value_t::boolean:
            case value_t::number_integer:
            case value_t::number_unsigned:
            case value_t::number_float:
            case value_t::binary:
            case value_t::discarded:
            default:
                return m_it.primitive_iterator - other.m_it.primitive_iterator;
        }
    }

    /*!
    @brief access to successor
    @pre The iterator is initialized; i.e. `m_object != nullptr`.
    */
    reference operator[](difference_type n) const
    {
        JSON_ASSERT(m_object != nullptr);

        switch (m_object->m_data.m_type)
        {
            case value_t::object:
                JSON_THROW(invalid_iterator::create(208, "cannot use operator[] for object iterators", m_object));

            case value_t::array:
                return *std::next(m_it.array_iterator, n);

            case value_t::null:
                JSON_THROW(invalid_iterator::create(214, "cannot get value", m_object));

            case value_t::string:
            case value_t::boolean:
            case value_t::number_integer:
            case value_t::number_unsigned:
            case value_t::number_float:
            case value_t::binary:
            case value_t::discarded:
            default:
            {
                if (JSON_HEDLEY_LIKELY(m_it.primitive_iterator.get_value() == -n))
                {
                    return *m_object;
                }

                JSON_THROW(invalid_iterator::create(214, "cannot get value", m_object));
            }
        }
    }

    /*!
    @brief return the key of an object iterator
    @pre The iterator is initialized; i.e. `m_object != nullptr`.
    */
    const typename object_t::key_type& key() const
    {
        JSON_ASSERT(m_object != nullptr);

        if (JSON_HEDLEY_LIKELY(m_object->is_object()))
        {
            return m_it.object_iterator->first;
        }

        JSON_THROW(invalid_iterator::create(207, "cannot use key() for non-object iterators", m_object));
    }

    /*!
    @brief return the value of an iterator
    @pre The iterator is initialized; i.e. `m_object != nullptr`.
    */
    reference value() const
    {
        return operator*();
    }

  JSON_PRIVATE_UNLESS_TESTED:
    /// associated JSON instance
    pointer m_object = nullptr;
    /// the actual iterator of the associated instance
    internal_iterator<typename std::remove_const<BasicJsonType>::type> m_it {};
};

}  // namespace detail
NLOHMANN_JSON_NAMESPACE_END

// #include <nlohmann/detail/iterators/iteration_proxy.hpp>

// #include <nlohmann/detail/iterators/json_reverse_iterator.hpp>
//     __ _____ _____ _____
//  __|  |   __|     |   | |  JSON for Modern C++
// |  |  |__   |  |  | | | |  version 3.12.0
// |_____|_____|_____|_|___|  https://github.com/nlohmann/json
//
// SPDX-FileCopyrightText: 2013 - 2025 Niels Lohmann <https://nlohmann.me>
// SPDX-License-Identifier: MIT



#include <cstddef> // ptrdiff_t
#include <iterator> // reverse_iterator
#include <utility> // declval

// #include <nlohmann/detail/abi_macros.hpp>


NLOHMANN_JSON_NAMESPACE_BEGIN
namespace detail
{

//////////////////////
// reverse_iterator //
//////////////////////

/*!
@brief a template for a reverse iterator class

@tparam Base the base iterator type to reverse. Valid types are @ref
iterator (to create @ref reverse_iterator) and @ref const_iterator (to
create @ref const_reverse_iterator).

@requirement The class satisfies the following concept requirements:
-
[BidirectionalIterator](https://en.cppreference.com/w/cpp/named_req/BidirectionalIterator):
  The iterator that can be moved can be moved in both directions (i.e.
  incremented and decremented).
- [OutputIterator](https://en.cppreference.com/w/cpp/named_req/OutputIterator):
  It is possible to write to the pointed-to element (only if @a Base is
  @ref iterator).

@since version 1.0.0
*/
template<typename Base>
class json_reverse_iterator : public std::reverse_iterator<Base>
{
  public:
    using difference_type = std::ptrdiff_t;
    /// shortcut to the reverse iterator adapter
    using base_iterator = std::reverse_iterator<Base>;
    /// the reference type for the pointed-to element
    using reference = typename Base::reference;

    /// create reverse iterator from iterator
    explicit json_reverse_iterator(const typename base_iterator::iterator_type& it) noexcept
        : base_iterator(it) {}

    /// create reverse iterator from base class
    explicit json_reverse_iterator(const base_iterator& it) noexcept : base_iterator(it) {}

    /// post-increment (it++)
    json_reverse_iterator operator++(int)& // NOLINT(cert-dcl21-cpp)
    {
        return static_cast<json_reverse_iterator>(base_iterator::operator++(1));
    }

    /// pre-increment (++it)
    json_reverse_iterator& operator++()
    {
        return static_cast<json_reverse_iterator&>(base_iterator::operator++());
    }

    /// post-decrement (it--)
    json_reverse_iterator operator--(int)& // NOLINT(cert-dcl21-cpp)
    {
        return static_cast<json_reverse_iterator>(base_iterator::operator--(1));
    }

    /// pre-decrement (--it)
    json_reverse_iterator& operator--()
    {
        return static_cast<json_reverse_iterator&>(base_iterator::operator--());
    }

    /// add to iterator
    json_reverse_iterator& operator+=(difference_type i)
    {
        return static_cast<json_reverse_iterator&>(base_iterator::operator+=(i));
    }

    /// add to iterator
    json_reverse_iterator operator+(difference_type i) const
    {
        return static_cast<json_reverse_iterator>(base_iterator::operator+(i));
    }

    /// subtract from iterator
    json_reverse_iterator operator-(difference_type i) const
    {
        return static_cast<json_reverse_iterator>(base_iterator::operator-(i));
    }

    /// return difference
    difference_type operator-(const json_reverse_iterator& other) const
    {
        return base_iterator(*this) - base_iterator(other);
    }

    /// access to successor
    reference operator[](difference_type n) const
    {
        return *(this->operator+(n));
    }

    /// return the key of an object iterator
    auto key() const -> decltype(std::declval<Base>().key())
    {
        auto it = --this->base();
        return it.key();
    }

    /// return the value of an iterator
    reference value() const
    {
        auto it = --this->base();
        return it.operator * ();
    }
};

}  // namespace detail
NLOHMANN_JSON_NAMESPACE_END

// #include <nlohmann/detail/iterators/primitive_iterator.hpp>

// #include <nlohmann/detail/json_custom_base_class.hpp>
//     __ _____ _____ _____
//  __|  |   __|     |   | |  JSON for Modern C++
// |  |  |__   |  |  | | | |  version 3.12.0
// |_____|_____|_____|_|___|  https://github.com/nlohmann/json
//
// SPDX-FileCopyrightText: 2013 - 2025 Niels Lohmann <https://nlohmann.me>
// SPDX-License-Identifier: MIT



#include <type_traits> // conditional, is_same

// #include <nlohmann/detail/abi_macros.hpp>


NLOHMANN_JSON_NAMESPACE_BEGIN
namespace detail
{

/*!
@brief Default base class of the @ref basic_json class.

So that the correct implementations of the copy / move ctors / assign operators
of @ref basic_json do not require complex case distinctions
(no base class / custom base class used as customization point),
@ref basic_json always has a base class.
By default, this class is used because it is empty and thus has no effect
on the behavior of @ref basic_json.
*/
struct json_default_base {};

template<class T>
using json_base_class = typename std::conditional <
                        std::is_same<T, void>::value,
                        json_default_base,
                        T
                        >::type;

}  // namespace detail
NLOHMANN_JSON_NAMESPACE_END

// #include <nlohmann/detail/json_pointer.hpp>
//     __ _____ _____ _____
//  __|  |   __|     |   | |  JSON for Modern C++
// |  |  |__   |  |  | | | |  version 3.12.0
// |_____|_____|_____|_|___|  https://github.com/nlohmann/json
//
// SPDX-FileCopyrightText: 2013 - 2025 Niels Lohmann <https://nlohmann.me>
// SPDX-License-Identifier: MIT



#include <algorithm> // all_of
#include <cctype> // isdigit
#include <cerrno> // errno, ERANGE
#include <cstdlib> // strtoull
#ifndef JSON_NO_IO
    #include <iosfwd> // ostream
#endif  // JSON_NO_IO
#include <limits> // max
#include <numeric> // accumulate
#include <string> // string
#include <utility> // move
#include <vector> // vector

// #include <nlohmann/detail/exceptions.hpp>

// #include <nlohmann/detail/macro_scope.hpp>

// #include <nlohmann/detail/string_concat.hpp>

// #include <nlohmann/detail/string_escape.hpp>

// #include <nlohmann/detail/value_t.hpp>


NLOHMANN_JSON_NAMESPACE_BEGIN

/// @brief JSON Pointer defines a string syntax for identifying a specific value within a JSON document
/// @sa https://json.nlohmann.me/api/json_pointer/
template<typename RefStringType>
class json_pointer
{
    // allow basic_json to access private members
    NLOHMANN_BASIC_JSON_TPL_DECLARATION
    friend class basic_json;

    template<typename>
    friend class json_pointer;

    template<typename T>
    struct string_t_helper
    {
        using type = T;
    };

    NLOHMANN_BASIC_JSON_TPL_DECLARATION
    struct string_t_helper<NLOHMANN_BASIC_JSON_TPL>
    {
        using type = StringType;
    };

  public:
    // for backwards compatibility accept BasicJsonType
    using string_t = typename string_t_helper<RefStringType>::type;

    /// @brief create JSON pointer
    /// @sa https://json.nlohmann.me/api/json_pointer/json_pointer/
    explicit json_pointer(const string_t& s = "")
        : reference_tokens(split(s))
    {}

    /// @brief return a string representation of the JSON pointer
    /// @sa https://json.nlohmann.me/api/json_pointer/to_string/
    string_t to_string() const
    {
        return std::accumulate(reference_tokens.begin(), reference_tokens.end(),
                               string_t{},
                               [](const string_t& a, const string_t& b)
        {
            return detail::concat(a, '/', detail::escape(b));
        });
    }

    /// @brief return a string representation of the JSON pointer
    /// @sa https://json.nlohmann.me/api/json_pointer/operator_string/
    JSON_HEDLEY_DEPRECATED_FOR(3.11.0, to_string())
    operator string_t() const
    {
        return to_string();
    }

#ifndef JSON_NO_IO
    /// @brief write string representation of the JSON pointer to stream
    /// @sa https://json.nlohmann.me/api/basic_json/operator_ltlt/
    friend std::ostream& operator<<(std::ostream& o, const json_pointer& ptr)
    {
        o << ptr.to_string();
        return o;
    }
#endif

    /// @brief append another JSON pointer at the end of this JSON pointer
    /// @sa https://json.nlohmann.me/api/json_pointer/operator_slasheq/
    json_pointer& operator/=(const json_pointer& ptr)
    {
        reference_tokens.insert(reference_tokens.end(),
                                ptr.reference_tokens.begin(),
                                ptr.reference_tokens.end());
        return *this;
    }

    /// @brief append an unescaped reference token at the end of this JSON pointer
    /// @sa https://json.nlohmann.me/api/json_pointer/operator_slasheq/
    json_pointer& operator/=(string_t token)
    {
        push_back(std::move(token));
        return *this;
    }

    /// @brief append an array index at the end of this JSON pointer
    /// @sa https://json.nlohmann.me/api/json_pointer/operator_slasheq/
    json_pointer& operator/=(std::size_t array_idx)
    {
        return *this /= std::to_string(array_idx);
    }

    /// @brief create a new JSON pointer by appending the right JSON pointer at the end of the left JSON pointer
    /// @sa https://json.nlohmann.me/api/json_pointer/operator_slash/
    friend json_pointer operator/(const json_pointer& lhs,
                                  const json_pointer& rhs)
    {
        return json_pointer(lhs) /= rhs;
    }

    /// @brief create a new JSON pointer by appending the unescaped token at the end of the JSON pointer
    /// @sa https://json.nlohmann.me/api/json_pointer/operator_slash/
    friend json_pointer operator/(const json_pointer& lhs, string_t token) // NOLINT(performance-unnecessary-value-param)
    {
        return json_pointer(lhs) /= std::move(token);
    }

    /// @brief create a new JSON pointer by appending the array-index-token at the end of the JSON pointer
    /// @sa https://json.nlohmann.me/api/json_pointer/operator_slash/
    friend json_pointer operator/(const json_pointer& lhs, std::size_t array_idx)
    {
        return json_pointer(lhs) /= array_idx;
    }

    /// @brief returns the parent of this JSON pointer
    /// @sa https://json.nlohmann.me/api/json_pointer/parent_pointer/
    json_pointer parent_pointer() const
    {
        if (empty())
        {
            return *this;
        }

        json_pointer res = *this;
        res.pop_back();
        return res;
    }

    /// @brief remove last reference token
    /// @sa https://json.nlohmann.me/api/json_pointer/pop_back/
    void pop_back()
    {
        if (JSON_HEDLEY_UNLIKELY(empty()))
        {
            JSON_THROW(detail::out_of_range::create(405, "JSON pointer has no parent", nullptr));
        }

        reference_tokens.pop_back();
    }

    /// @brief return last reference token
    /// @sa https://json.nlohmann.me/api/json_pointer/back/
    const string_t& back() const
    {
        if (JSON_HEDLEY_UNLIKELY(empty()))
        {
            JSON_THROW(detail::out_of_range::create(405, "JSON pointer has no parent", nullptr));
        }

        return reference_tokens.back();
    }

    /// @brief append an unescaped token at the end of the reference pointer
    /// @sa https://json.nlohmann.me/api/json_pointer/push_back/
    void push_back(const string_t& token)
    {
        reference_tokens.push_back(token);
    }

    /// @brief append an unescaped token at the end of the reference pointer
    /// @sa https://json.nlohmann.me/api/json_pointer/push_back/
    void push_back(string_t&& token)
    {
        reference_tokens.push_back(std::move(token));
    }

    /// @brief return whether pointer points to the root document
    /// @sa https://json.nlohmann.me/api/json_pointer/empty/
    bool empty() const noexcept
    {
        return reference_tokens.empty();
    }

  private:
    /*!
    @param[in] s  reference token to be converted into an array index

    @return integer representation of @a s

    @throw parse_error.106  if an array index begins with '0'
    @throw parse_error.109  if an array index begins not with a digit
    @throw out_of_range.404 if string @a s could not be converted to an integer
    @throw out_of_range.410 if an array index exceeds size_type
    */
    template<typename BasicJsonType>
    static typename BasicJsonType::size_type array_index(const string_t& s)
    {
        using size_type = typename BasicJsonType::size_type;

        // error condition (cf. RFC 6901, Sect. 4)
        if (JSON_HEDLEY_UNLIKELY(s.size() > 1 && s[0] == '0'))
        {
            JSON_THROW(detail::parse_error::create(106, 0, detail::concat("array index '", s, "' must not begin with '0'"), nullptr));
        }

        // error condition (cf. RFC 6901, Sect. 4)
        if (JSON_HEDLEY_UNLIKELY(s.size() > 1 && !(s[0] >= '1' && s[0] <= '9')))
        {
            JSON_THROW(detail::parse_error::create(109, 0, detail::concat("array index '", s, "' is not a number"), nullptr));
        }

        const char* p = s.c_str();
        char* p_end = nullptr; // NOLINT(misc-const-correctness)
        errno = 0; // strtoull doesn't reset errno
        const unsigned long long res = std::strtoull(p, &p_end, 10); // NOLINT(runtime/int)
        if (p == p_end // invalid input or empty string
                || errno == ERANGE // out of range
                || JSON_HEDLEY_UNLIKELY(static_cast<std::size_t>(p_end - p) != s.size())) // incomplete read
        {
            JSON_THROW(detail::out_of_range::create(404, detail::concat("unresolved reference token '", s, "'"), nullptr));
        }

        // only triggered on special platforms (like 32bit), see also
        // https://github.com/nlohmann/json/pull/2203
        if (res >= static_cast<unsigned long long>((std::numeric_limits<size_type>::max)()))  // NOLINT(runtime/int)
        {
            JSON_THROW(detail::out_of_range::create(410, detail::concat("array index ", s, " exceeds size_type"), nullptr));   // LCOV_EXCL_LINE
        }

        return static_cast<size_type>(res);
    }

  JSON_PRIVATE_UNLESS_TESTED:
    json_pointer top() const
    {
        if (JSON_HEDLEY_UNLIKELY(empty()))
        {
            JSON_THROW(detail::out_of_range::create(405, "JSON pointer has no parent", nullptr));
        }

        json_pointer result = *this;
        result.reference_tokens = {reference_tokens[0]};
        return result;
    }

  private:
    /*!
    @brief create and return a reference to the pointed to value

    @complexity Linear in the number of reference tokens.

    @throw parse_error.109 if array index is not a number
    @throw type_error.313 if value cannot be unflattened
    */
    template<typename BasicJsonType>
    BasicJsonType& get_and_create(BasicJsonType& j) const
    {
        auto* result = &j;

        // in case no reference tokens exist, return a reference to the JSON value
        // j which will be overwritten by a primitive value
        for (const auto& reference_token : reference_tokens)
        {
            switch (result->type())
            {
                case detail::value_t::null:
                {
                    if (reference_token == "0")
                    {
                        // start a new array if reference token is 0
                        result = &result->operator[](0);
                    }
                    else
                    {
                        // start a new object otherwise
                        result = &result->operator[](reference_token);
                    }
                    break;
                }

                case detail::value_t::object:
                {
                    // create an entry in the object
                    result = &result->operator[](reference_token);
                    break;
                }

                case detail::value_t::array:
                {
                    // create an entry in the array
                    result = &result->operator[](array_index<BasicJsonType>(reference_token));
                    break;
                }

                /*
                The following code is only reached if there exists a reference
                token _and_ the current value is primitive. In this case, we have
                an error situation, because primitive values may only occur as
                single value; that is, with an empty list of reference tokens.
                */
                case detail::value_t::string:
                case detail::value_t::boolean:
                case detail::value_t::number_integer:
                case detail::value_t::number_unsigned:
                case detail::value_t::number_float:
                case detail::value_t::binary:
                case detail::value_t::discarded:
                default:
                    JSON_THROW(detail::type_error::create(313, "invalid value to unflatten", &j));
            }
        }

        return *result;
    }

    /*!
    @brief return a reference to the pointed to value

    @note This version does not throw if a value is not present, but tries to
          create nested values instead. For instance, calling this function
          with pointer `"/this/that"` on a null value is equivalent to calling
          `operator[]("this").operator[]("that")` on that value, effectively
          changing the null value to an object.

    @param[in] ptr  a JSON value

    @return reference to the JSON value pointed to by the JSON pointer

    @complexity Linear in the length of the JSON pointer.

    @throw parse_error.106   if an array index begins with '0'
    @throw parse_error.109   if an array index was not a number
    @throw out_of_range.404  if the JSON pointer can not be resolved
    */
    template<typename BasicJsonType>
    BasicJsonType& get_unchecked(BasicJsonType* ptr) const
    {
        for (const auto& reference_token : reference_tokens)
        {
            // convert null values to arrays or objects before continuing
            if (ptr->is_null())
            {
                // check if reference token is a number
                const bool nums =
                    std::all_of(reference_token.begin(), reference_token.end(),
                                [](const unsigned char x)
                {
                    return std::isdigit(x);
                });

                // change value to array for numbers or "-" or to object otherwise
                *ptr = (nums || reference_token == "-")
                       ? detail::value_t::array
                       : detail::value_t::object;
            }

            switch (ptr->type())
            {
                case detail::value_t::object:
                {
                    // use unchecked object access
                    ptr = &ptr->operator[](reference_token);
                    break;
                }

                case detail::value_t::array:
                {
                    if (reference_token == "-")
                    {
                        // explicitly treat "-" as index beyond the end
                        ptr = &ptr->operator[](ptr->m_data.m_value.array->size());
                    }
                    else
                    {
                        // convert array index to number; unchecked access
                        ptr = &ptr->operator[](array_index<BasicJsonType>(reference_token));
                    }
                    break;
                }

                case detail::value_t::null:
                case detail::value_t::string:
                case detail::value_t::boolean:
                case detail::value_t::number_integer:
                case detail::value_t::number_unsigned:
                case detail::value_t::number_float:
                case detail::value_t::binary:
                case detail::value_t::discarded:
                default:
                    JSON_THROW(detail::out_of_range::create(404, detail::concat("unresolved reference token '", reference_token, "'"), ptr));
            }
        }

        return *ptr;
    }

    /*!
    @throw parse_error.106   if an array index begins with '0'
    @throw parse_error.109   if an array index was not a number
    @throw out_of_range.402  if the array index '-' is used
    @throw out_of_range.404  if the JSON pointer can not be resolved
    */
    template<typename BasicJsonType>
    BasicJsonType& get_checked(BasicJsonType* ptr) const
    {
        for (const auto& reference_token : reference_tokens)
        {
            switch (ptr->type())
            {
                case detail::value_t::object:
                {
                    // note: at performs range check
                    ptr = &ptr->at(reference_token);
                    break;
                }

                case detail::value_t::array:
                {
                    if (JSON_HEDLEY_UNLIKELY(reference_token == "-"))
                    {
                        // "-" always fails the range check
                        JSON_THROW(detail::out_of_range::create(402, detail::concat(
                                "array index '-' (", std::to_string(ptr->m_data.m_value.array->size()),
                                ") is out of range"), ptr));
                    }

                    // note: at performs range check
                    ptr = &ptr->at(array_index<BasicJsonType>(reference_token));
                    break;
                }

                case detail::value_t::null:
                case detail::value_t::string:
                case detail::value_t::boolean:
                case detail::value_t::number_integer:
                case detail::value_t::number_unsigned:
                case detail::value_t::number_float:
                case detail::value_t::binary:
                case detail::value_t::discarded:
                default:
                    JSON_THROW(detail::out_of_range::create(404, detail::concat("unresolved reference token '", reference_token, "'"), ptr));
            }
        }

        return *ptr;
    }

    /*!
    @brief return a const reference to the pointed to value

    @param[in] ptr  a JSON value

    @return const reference to the JSON value pointed to by the JSON
    pointer

    @throw parse_error.106   if an array index begins with '0'
    @throw parse_error.109   if an array index was not a number
    @throw out_of_range.402  if the array index '-' is used
    @throw out_of_range.404  if the JSON pointer can not be resolved
    */
    template<typename BasicJsonType>
    const BasicJsonType& get_unchecked(const BasicJsonType* ptr) const
    {
        for (const auto& reference_token : reference_tokens)
        {
            switch (ptr->type())
            {
                case detail::value_t::object:
                {
                    // use unchecked object access
                    ptr = &ptr->operator[](reference_token);
                    break;
                }

                case detail::value_t::array:
                {
                    if (JSON_HEDLEY_UNLIKELY(reference_token == "-"))
                    {
                        // "-" cannot be used for const access
                        JSON_THROW(detail::out_of_range::create(402, detail::concat("array index '-' (", std::to_string(ptr->m_data.m_value.array->size()), ") is out of range"), ptr));
                    }

                    // use unchecked array access
                    ptr = &ptr->operator[](array_index<BasicJsonType>(reference_token));
                    break;
                }

                case detail::value_t::null:
                case detail::value_t::string:
                case detail::value_t::boolean:
                case detail::value_t::number_integer:
                case detail::value_t::number_unsigned:
                case detail::value_t::number_float:
                case detail::value_t::binary:
                case detail::value_t::discarded:
                default:
                    JSON_THROW(detail::out_of_range::create(404, detail::concat("unresolved reference token '", reference_token, "'"), ptr));
            }
        }

        return *ptr;
    }

    /*!
    @throw parse_error.106   if an array index begins with '0'
    @throw parse_error.109   if an array index was not a number
    @throw out_of_range.402  if the array index '-' is used
    @throw out_of_range.404  if the JSON pointer can not be resolved
    */
    template<typename BasicJsonType>
    const BasicJsonType& get_checked(const BasicJsonType* ptr) const
    {
        for (const auto& reference_token : reference_tokens)
        {
            switch (ptr->type())
            {
                case detail::value_t::object:
                {
                    // note: at performs range check
                    ptr = &ptr->at(reference_token);
                    break;
                }

                case detail::value_t::array:
                {
                    if (JSON_HEDLEY_UNLIKELY(reference_token == "-"))
                    {
                        // "-" always fails the range check
                        JSON_THROW(detail::out_of_range::create(402, detail::concat(
                                "array index '-' (", std::to_string(ptr->m_data.m_value.array->size()),
                                ") is out of range"), ptr));
                    }

                    // note: at performs range check
                    ptr = &ptr->at(array_index<BasicJsonType>(reference_token));
                    break;
                }

                case detail::value_t::null:
                case detail::value_t::string:
                case detail::value_t::boolean:
                case detail::value_t::number_integer:
                case detail::value_t::number_unsigned:
                case detail::value_t::number_float:
                case detail::value_t::binary:
                case detail::value_t::discarded:
                default:
                    JSON_THROW(detail::out_of_range::create(404, detail::concat("unresolved reference token '", reference_token, "'"), ptr));
            }
        }

        return *ptr;
    }

    /*!
    @throw parse_error.106   if an array index begins with '0'
    @throw parse_error.109   if an array index was not a number
    */
    template<typename BasicJsonType>
    bool contains(const BasicJsonType* ptr) const
    {
        for (const auto& reference_token : reference_tokens)
        {
            switch (ptr->type())
            {
                case detail::value_t::object:
                {
                    if (!ptr->contains(reference_token))
                    {
                        // we did not find the key in the object
                        return false;
                    }

                    ptr = &ptr->operator[](reference_token);
                    break;
                }

                case detail::value_t::array:
                {
                    if (JSON_HEDLEY_UNLIKELY(reference_token == "-"))
                    {
                        // "-" always fails the range check
                        return false;
                    }
                    if (JSON_HEDLEY_UNLIKELY(reference_token.size() == 1 && !("0" <= reference_token && reference_token <= "9")))
                    {
                        // invalid char
                        return false;
                    }
                    if (JSON_HEDLEY_UNLIKELY(reference_token.size() > 1))
                    {
                        if (JSON_HEDLEY_UNLIKELY(!('1' <= reference_token[0] && reference_token[0] <= '9')))
                        {
                            // first char should be between '1' and '9'
                            return false;
                        }
                        for (std::size_t i = 1; i < reference_token.size(); i++)
                        {
                            if (JSON_HEDLEY_UNLIKELY(!('0' <= reference_token[i] && reference_token[i] <= '9')))
                            {
                                // other char should be between '0' and '9'
                                return false;
                            }
                        }
                    }

                    const auto idx = array_index<BasicJsonType>(reference_token);
                    if (idx >= ptr->size())
                    {
                        // index out of range
                        return false;
                    }

                    ptr = &ptr->operator[](idx);
                    break;
                }

                case detail::value_t::null:
                case detail::value_t::string:
                case detail::value_t::boolean:
                case detail::value_t::number_integer:
                case detail::value_t::number_unsigned:
                case detail::value_t::number_float:
                case detail::value_t::binary:
                case detail::value_t::discarded:
                default:
                {
                    // we do not expect primitive values if there is still a
                    // reference token to process
                    return false;
                }
            }
        }

        // no reference token left means we found a primitive value
        return true;
    }

    /*!
    @brief split the string input to reference tokens

    @note This function is only called by the json_pointer constructor.
          All exceptions below are documented there.

    @throw parse_error.107  if the pointer is not empty or begins with '/'
    @throw parse_error.108  if character '~' is not followed by '0' or '1'
    */
    static std::vector<string_t> split(const string_t& reference_string)
    {
        std::vector<string_t> result;

        // special case: empty reference string -> no reference tokens
        if (reference_string.empty())
        {
            return result;
        }

        // check if nonempty reference string begins with slash
        if (JSON_HEDLEY_UNLIKELY(reference_string[0] != '/'))
        {
            JSON_THROW(detail::parse_error::create(107, 1, detail::concat("JSON pointer must be empty or begin with '/' - was: '", reference_string, "'"), nullptr));
        }

        // extract the reference tokens:
        // - slash: position of the last read slash (or end of string)
        // - start: position after the previous slash
        for (
            // search for the first slash after the first character
            std::size_t slash = reference_string.find_first_of('/', 1),
            // set the beginning of the first reference token
            start = 1;
            // we can stop if start == 0 (if slash == string_t::npos)
            start != 0;
            // set the beginning of the next reference token
            // (will eventually be 0 if slash == string_t::npos)
            start = (slash == string_t::npos) ? 0 : slash + 1,
            // find next slash
            slash = reference_string.find_first_of('/', start))
        {
            // use the text between the beginning of the reference token
            // (start) and the last slash (slash).
            auto reference_token = reference_string.substr(start, slash - start);

            // check reference tokens are properly escaped
            for (std::size_t pos = reference_token.find_first_of('~');
                    pos != string_t::npos;
                    pos = reference_token.find_first_of('~', pos + 1))
            {
                JSON_ASSERT(reference_token[pos] == '~');

                // ~ must be followed by 0 or 1
                if (JSON_HEDLEY_UNLIKELY(pos == reference_token.size() - 1 ||
                                         (reference_token[pos + 1] != '0' &&
                                          reference_token[pos + 1] != '1')))
                {
                    JSON_THROW(detail::parse_error::create(108, 0, "escape character '~' must be followed with '0' or '1'", nullptr));
                }
            }

            // finally, store the reference token
            detail::unescape(reference_token);
            result.push_back(reference_token);
        }

        return result;
    }

  private:
    /*!
    @param[in] reference_string  the reference string to the current value
    @param[in] value             the value to consider
    @param[in,out] result        the result object to insert values to

    @note Empty objects or arrays are flattened to `null`.
    */
    template<typename BasicJsonType>
    static void flatten(const string_t& reference_string,
                        const BasicJsonType& value,
                        BasicJsonType& result)
    {
        switch (value.type())
        {
            case detail::value_t::array:
            {
                if (value.m_data.m_value.array->empty())
                {
                    // flatten empty array as null
                    result[reference_string] = nullptr;
                }
                else
                {
                    // iterate array and use index as reference string
                    for (std::size_t i = 0; i < value.m_data.m_value.array->size(); ++i)
                    {
                        flatten(detail::concat<string_t>(reference_string, '/', std::to_string(i)),
                                value.m_data.m_value.array->operator[](i), result);
                    }
                }
                break;
            }

            case detail::value_t::object:
            {
                if (value.m_data.m_value.object->empty())
                {
                    // flatten empty object as null
                    result[reference_string] = nullptr;
                }
                else
                {
                    // iterate object and use keys as reference string
                    for (const auto& element : *value.m_data.m_value.object)
                    {
                        flatten(detail::concat<string_t>(reference_string, '/', detail::escape(element.first)), element.second, result);
                    }
                }
                break;
            }

            case detail::value_t::null:
            case detail::value_t::string:
            case detail::value_t::boolean:
            case detail::value_t::number_integer:
            case detail::value_t::number_unsigned:
            case detail::value_t::number_float:
            case detail::value_t::binary:
            case detail::value_t::discarded:
            default:
            {
                // add primitive value with its reference string
                result[reference_string] = value;
                break;
            }
        }
    }

    /*!
    @param[in] value  flattened JSON

    @return unflattened JSON

    @throw parse_error.109 if array index is not a number
    @throw type_error.314  if value is not an object
    @throw type_error.315  if object values are not primitive
    @throw type_error.313  if value cannot be unflattened
    */
    template<typename BasicJsonType>
    static BasicJsonType
    unflatten(const BasicJsonType& value)
    {
        if (JSON_HEDLEY_UNLIKELY(!value.is_object()))
        {
            JSON_THROW(detail::type_error::create(314, "only objects can be unflattened", &value));
        }

        BasicJsonType result;

        // iterate the JSON object values
        for (const auto& element : *value.m_data.m_value.object)
        {
            if (JSON_HEDLEY_UNLIKELY(!element.second.is_primitive()))
            {
                JSON_THROW(detail::type_error::create(315, "values in object must be primitive", &element.second));
            }

            // assign value to reference pointed to by JSON pointer; Note that if
            // the JSON pointer is "" (i.e., points to the whole value), function
            // get_and_create returns a reference to result itself. An assignment
            // will then create a primitive value.
            json_pointer(element.first).get_and_create(result) = element.second;
        }

        return result;
    }

    // can't use conversion operator because of ambiguity
    json_pointer<string_t> convert() const&
    {
        json_pointer<string_t> result;
        result.reference_tokens = reference_tokens;
        return result;
    }

    json_pointer<string_t> convert()&&
    {
        json_pointer<string_t> result;
        result.reference_tokens = std::move(reference_tokens);
        return result;
    }

  public:
#if JSON_HAS_THREE_WAY_COMPARISON
    /// @brief compares two JSON pointers for equality
    /// @sa https://json.nlohmann.me/api/json_pointer/operator_eq/
    template<typename RefStringTypeRhs>
    bool operator==(const json_pointer<RefStringTypeRhs>& rhs) const noexcept
    {
        return reference_tokens == rhs.reference_tokens;
    }

    /// @brief compares JSON pointer and string for equality
    /// @sa https://json.nlohmann.me/api/json_pointer/operator_eq/
    JSON_HEDLEY_DEPRECATED_FOR(3.11.2, operator==(json_pointer))
    bool operator==(const string_t& rhs) const
    {
        return *this == json_pointer(rhs);
    }

    /// @brief 3-way compares two JSON pointers
    template<typename RefStringTypeRhs>
    std::strong_ordering operator<=>(const json_pointer<RefStringTypeRhs>& rhs) const noexcept // *NOPAD*
    {
        return  reference_tokens <=> rhs.reference_tokens; // *NOPAD*
    }
#else
    /// @brief compares two JSON pointers for equality
    /// @sa https://json.nlohmann.me/api/json_pointer/operator_eq/
    template<typename RefStringTypeLhs, typename RefStringTypeRhs>
    // NOLINTNEXTLINE(readability-redundant-declaration)
    friend bool operator==(const json_pointer<RefStringTypeLhs>& lhs,
                           const json_pointer<RefStringTypeRhs>& rhs) noexcept;

    /// @brief compares JSON pointer and string for equality
    /// @sa https://json.nlohmann.me/api/json_pointer/operator_eq/
    template<typename RefStringTypeLhs, typename StringType>
    // NOLINTNEXTLINE(readability-redundant-declaration)
    friend bool operator==(const json_pointer<RefStringTypeLhs>& lhs,
                           const StringType& rhs);

    /// @brief compares string and JSON pointer for equality
    /// @sa https://json.nlohmann.me/api/json_pointer/operator_eq/
    template<typename RefStringTypeRhs, typename StringType>
    // NOLINTNEXTLINE(readability-redundant-declaration)
    friend bool operator==(const StringType& lhs,
                           const json_pointer<RefStringTypeRhs>& rhs);

    /// @brief compares two JSON pointers for inequality
    /// @sa https://json.nlohmann.me/api/json_pointer/operator_ne/
    template<typename RefStringTypeLhs, typename RefStringTypeRhs>
    // NOLINTNEXTLINE(readability-redundant-declaration)
    friend bool operator!=(const json_pointer<RefStringTypeLhs>& lhs,
                           const json_pointer<RefStringTypeRhs>& rhs) noexcept;

    /// @brief compares JSON pointer and string for inequality
    /// @sa https://json.nlohmann.me/api/json_pointer/operator_ne/
    template<typename RefStringTypeLhs, typename StringType>
    // NOLINTNEXTLINE(readability-redundant-declaration)
    friend bool operator!=(const json_pointer<RefStringTypeLhs>& lhs,
                           const StringType& rhs);

    /// @brief compares string and JSON pointer for inequality
    /// @sa https://json.nlohmann.me/api/json_pointer/operator_ne/
    template<typename RefStringTypeRhs, typename StringType>
    // NOLINTNEXTLINE(readability-redundant-declaration)
    friend bool operator!=(const StringType& lhs,
                           const json_pointer<RefStringTypeRhs>& rhs);

    /// @brief compares two JSON pointer for less-than
    template<typename RefStringTypeLhs, typename RefStringTypeRhs>
    // NOLINTNEXTLINE(readability-redundant-declaration)
    friend bool operator<(const json_pointer<RefStringTypeLhs>& lhs,
                          const json_pointer<RefStringTypeRhs>& rhs) noexcept;
#endif

  private:
    /// the reference tokens
    std::vector<string_t> reference_tokens;
};

#if !JSON_HAS_THREE_WAY_COMPARISON
// functions cannot be defined inside class due to ODR violations
template<typename RefStringTypeLhs, typename RefStringTypeRhs>
inline bool operator==(const json_pointer<RefStringTypeLhs>& lhs,
                       const json_pointer<RefStringTypeRhs>& rhs) noexcept
{
    return lhs.reference_tokens == rhs.reference_tokens;
}

template<typename RefStringTypeLhs,
         typename StringType = typename json_pointer<RefStringTypeLhs>::string_t>
JSON_HEDLEY_DEPRECATED_FOR(3.11.2, operator==(json_pointer, json_pointer))
inline bool operator==(const json_pointer<RefStringTypeLhs>& lhs,
                       const StringType& rhs)
{
    return lhs == json_pointer<RefStringTypeLhs>(rhs);
}

template<typename RefStringTypeRhs,
         typename StringType = typename json_pointer<RefStringTypeRhs>::string_t>
JSON_HEDLEY_DEPRECATED_FOR(3.11.2, operator==(json_pointer, json_pointer))
inline bool operator==(const StringType& lhs,
                       const json_pointer<RefStringTypeRhs>& rhs)
{
    return json_pointer<RefStringTypeRhs>(lhs) == rhs;
}

template<typename RefStringTypeLhs, typename RefStringTypeRhs>
inline bool operator!=(const json_pointer<RefStringTypeLhs>& lhs,
                       const json_pointer<RefStringTypeRhs>& rhs) noexcept
{
    return !(lhs == rhs);
}

template<typename RefStringTypeLhs,
         typename StringType = typename json_pointer<RefStringTypeLhs>::string_t>
JSON_HEDLEY_DEPRECATED_FOR(3.11.2, operator!=(json_pointer, json_pointer))
inline bool operator!=(const json_pointer<RefStringTypeLhs>& lhs,
                       const StringType& rhs)
{
    return !(lhs == rhs);
}

template<typename RefStringTypeRhs,
         typename StringType = typename json_pointer<RefStringTypeRhs>::string_t>
JSON_HEDLEY_DEPRECATED_FOR(3.11.2, operator!=(json_pointer, json_pointer))
inline bool operator!=(const StringType& lhs,
                       const json_pointer<RefStringTypeRhs>& rhs)
{
    return !(lhs == rhs);
}

template<typename RefStringTypeLhs, typename RefStringTypeRhs>
inline bool operator<(const json_pointer<RefStringTypeLhs>& lhs,
                      const json_pointer<RefStringTypeRhs>& rhs) noexcept
{
    return lhs.reference_tokens < rhs.reference_tokens;
}
#endif

NLOHMANN_JSON_NAMESPACE_END

// #include <nlohmann/detail/json_ref.hpp>
//     __ _____ _____ _____
//  __|  |   __|     |   | |  JSON for Modern C++
// |  |  |__   |  |  | | | |  version 3.12.0
// |_____|_____|_____|_|___|  https://github.com/nlohmann/json
//
// SPDX-FileCopyrightText: 2013 - 2025 Niels Lohmann <https://nlohmann.me>
// SPDX-License-Identifier: MIT



#include <initializer_list>
#include <utility>

// #include <nlohmann/detail/abi_macros.hpp>

// #include <nlohmann/detail/meta/type_traits.hpp>


NLOHMANN_JSON_NAMESPACE_BEGIN
namespace detail
{

template<typename BasicJsonType>
class json_ref
{
  public:
    using value_type = BasicJsonType;

    json_ref(value_type&& value)
        : owned_value(std::move(value))
    {}

    json_ref(const value_type& value)
        : value_ref(&value)
    {}

    json_ref(std::initializer_list<json_ref> init)
        : owned_value(init)
    {}

    template <
        class... Args,
        enable_if_t<std::is_constructible<value_type, Args...>::value, int> = 0 >
    json_ref(Args && ... args)
        : owned_value(std::forward<Args>(args)...)
    {}

    // class should be movable only
    json_ref(json_ref&&) noexcept = default;
    json_ref(const json_ref&) = delete;
    json_ref& operator=(const json_ref&) = delete;
    json_ref& operator=(json_ref&&) = delete;
    ~json_ref() = default;

    value_type moved_or_copied() const
    {
        if (value_ref == nullptr)
        {
            return std::move(owned_value);
        }
        return *value_ref;
    }

    value_type const& operator*() const
    {
        return value_ref ? *value_ref : owned_value;
    }

    value_type const* operator->() const
    {
        return &** this;
    }

  private:
    mutable value_type owned_value = nullptr;
    value_type const* value_ref = nullptr;
};

}  // namespace detail
NLOHMANN_JSON_NAMESPACE_END

// #include <nlohmann/detail/macro_scope.hpp>

// #include <nlohmann/detail/string_concat.hpp>

// #include <nlohmann/detail/string_escape.hpp>

// #include <nlohmann/detail/string_utils.hpp>

// #include <nlohmann/detail/meta/cpp_future.hpp>

// #include <nlohmann/detail/meta/type_traits.hpp>

// #include <nlohmann/detail/output/binary_writer.hpp>
//     __ _____ _____ _____
//  __|  |   __|     |   | |  JSON for Modern C++
// |  |  |__   |  |  | | | |  version 3.12.0
// |_____|_____|_____|_|___|  https://github.com/nlohmann/json
//
// SPDX-FileCopyrightText: 2013 - 2025 Niels Lohmann <https://nlohmann.me>
// SPDX-License-Identifier: MIT



#include <algorithm> // reverse
#include <array> // array
#include <map> // map
#include <cmath> // isnan, isinf
#include <cstdint> // uint8_t, uint16_t, uint32_t, uint64_t
#include <cstring> // memcpy
#include <limits> // numeric_limits
#include <string> // string
#include <utility> // move
#include <vector> // vector

// #include <nlohmann/detail/input/binary_reader.hpp>

// #include <nlohmann/detail/macro_scope.hpp>

// #include <nlohmann/detail/output/output_adapters.hpp>
//     __ _____ _____ _____
//  __|  |   __|     |   | |  JSON for Modern C++
// |  |  |__   |  |  | | | |  version 3.12.0
// |_____|_____|_____|_|___|  https://github.com/nlohmann/json
//
// SPDX-FileCopyrightText: 2013 - 2025 Niels Lohmann <https://nlohmann.me>
// SPDX-License-Identifier: MIT



#include <algorithm> // copy
#include <cstddef> // size_t
#include <iterator> // back_inserter
#include <memory> // shared_ptr, make_shared
#include <string> // basic_string
#include <vector> // vector

#ifndef JSON_NO_IO
    #include <ios>      // streamsize
    #include <ostream>  // basic_ostream
#endif  // JSON_NO_IO

// #include <nlohmann/detail/macro_scope.hpp>


NLOHMANN_JSON_NAMESPACE_BEGIN
namespace detail
{

/// abstract output adapter interface
template<typename CharType> struct output_adapter_protocol
{
    virtual void write_character(CharType c) = 0;
    virtual void write_characters(const CharType* s, std::size_t length) = 0;
    virtual ~output_adapter_protocol() = default;

    output_adapter_protocol() = default;
    output_adapter_protocol(const output_adapter_protocol&) = default;
    output_adapter_protocol(output_adapter_protocol&&) noexcept = default;
    output_adapter_protocol& operator=(const output_adapter_protocol&) = default;
    output_adapter_protocol& operator=(output_adapter_protocol&&) noexcept = default;
};

/// a type to simplify interfaces
template<typename CharType>
using output_adapter_t = std::shared_ptr<output_adapter_protocol<CharType>>;

/// output adapter for byte vectors
template<typename CharType, typename AllocatorType = std::allocator<CharType>>
class output_vector_adapter : public output_adapter_protocol<CharType>
{
  public:
    explicit output_vector_adapter(std::vector<CharType, AllocatorType>& vec) noexcept
        : v(vec)
    {}

    void write_character(CharType c) override
    {
        v.push_back(c);
    }

    JSON_HEDLEY_NON_NULL(2)
    void write_characters(const CharType* s, std::size_t length) override
    {
        v.insert(v.end(), s, s + length);
    }

  private:
    std::vector<CharType, AllocatorType>& v;
};

#ifndef JSON_NO_IO
/// output adapter for output streams
template<typename CharType>
class output_stream_adapter : public output_adapter_protocol<CharType>
{
  public:
    explicit output_stream_adapter(std::basic_ostream<CharType>& s) noexcept
        : stream(s)
    {}

    void write_character(CharType c) override
    {
        stream.put(c);
    }

    JSON_HEDLEY_NON_NULL(2)
    void write_characters(const CharType* s, std::size_t length) override
    {
        stream.write(s, static_cast<std::streamsize>(length));
    }

  private:
    std::basic_ostream<CharType>& stream;
};
#endif  // JSON_NO_IO

/// output adapter for basic_string
template<typename CharType, typename StringType = std::basic_string<CharType>>
class output_string_adapter : public output_adapter_protocol<CharType>
{
  public:
    explicit output_string_adapter(StringType& s) noexcept
        : str(s)
    {}

    void write_character(CharType c) override
    {
        str.push_back(c);
    }

    JSON_HEDLEY_NON_NULL(2)
    void write_characters(const CharType* s, std::size_t length) override
    {
        str.append(s, length);
    }

  private:
    StringType& str;
};

template<typename CharType, typename StringType = std::basic_string<CharType>>
class output_adapter
{
  public:
    template<typename AllocatorType = std::allocator<CharType>>
    output_adapter(std::vector<CharType, AllocatorType>& vec)
        : oa(std::make_shared<output_vector_adapter<CharType, AllocatorType>>(vec)) {}

#ifndef JSON_NO_IO
    output_adapter(std::basic_ostream<CharType>& s)
        : oa(std::make_shared<output_stream_adapter<CharType>>(s)) {}
#endif  // JSON_NO_IO

    output_adapter(StringType& s)
        : oa(std::make_shared<output_string_adapter<CharType, StringType>>(s)) {}

    operator output_adapter_t<CharType>()
    {
        return oa;
    }

  private:
    output_adapter_t<CharType> oa = nullptr;
};

}  // namespace detail
NLOHMANN_JSON_NAMESPACE_END

// #include <nlohmann/detail/string_concat.hpp>


NLOHMANN_JSON_NAMESPACE_BEGIN
namespace detail
{

/// how to encode BJData
enum class bjdata_version_t
{
    draft2,
    draft3,
};

///////////////////
// binary writer //
///////////////////

/*!
@brief serialization to CBOR and MessagePack values
*/
template<typename BasicJsonType, typename CharType>
class binary_writer
{
    using string_t = typename BasicJsonType::string_t;
    using binary_t = typename BasicJsonType::binary_t;
    using number_float_t = typename BasicJsonType::number_float_t;

  public:
    /*!
    @brief create a binary writer

    @param[in] adapter  output adapter to write to
    */
    explicit binary_writer(output_adapter_t<CharType> adapter) : oa(std::move(adapter))
    {
        JSON_ASSERT(oa);
    }

    /*!
    @param[in] j  JSON value to serialize
    @pre       j.type() == value_t::object
    */
    void write_bson(const BasicJsonType& j)
    {
        switch (j.type())
        {
            case value_t::object:
            {
                write_bson_object(*j.m_data.m_value.object);
                break;
            }

            case value_t::null:
            case value_t::array:
            case value_t::string:
            case value_t::boolean:
            case value_t::number_integer:
            case value_t::number_unsigned:
            case value_t::number_float:
            case value_t::binary:
            case value_t::discarded:
            default:
            {
                JSON_THROW(type_error::create(317, concat("to serialize to BSON, top-level type must be object, but is ", j.type_name()), &j));
            }
        }
    }

    /*!
    @param[in] j  JSON value to serialize
    */
    void write_cbor(const BasicJsonType& j)
    {
        switch (j.type())
        {
            case value_t::null:
            {
                oa->write_character(to_char_type(0xF6));
                break;
            }

            case value_t::boolean:
            {
                oa->write_character(j.m_data.m_value.boolean
                                    ? to_char_type(0xF5)
                                    : to_char_type(0xF4));
                break;
            }

            case value_t::number_integer:
            {
                if (j.m_data.m_value.number_integer >= 0)
                {
                    // CBOR does not differentiate between positive signed
                    // integers and unsigned integers. Therefore, we used the
                    // code from the value_t::number_unsigned case here.
                    if (j.m_data.m_value.number_integer <= 0x17)
                    {
                        write_number(static_cast<std::uint8_t>(j.m_data.m_value.number_integer));
                    }
                    else if (j.m_data.m_value.number_integer <= (std::numeric_limits<std::uint8_t>::max)())
                    {
                        oa->write_character(to_char_type(0x18));
                        write_number(static_cast<std::uint8_t>(j.m_data.m_value.number_integer));
                    }
                    else if (j.m_data.m_value.number_integer <= (std::numeric_limits<std::uint16_t>::max)())
                    {
                        oa->write_character(to_char_type(0x19));
                        write_number(static_cast<std::uint16_t>(j.m_data.m_value.number_integer));
                    }
                    else if (j.m_data.m_value.number_integer <= (std::numeric_limits<std::uint32_t>::max)())
                    {
                        oa->write_character(to_char_type(0x1A));
                        write_number(static_cast<std::uint32_t>(j.m_data.m_value.number_integer));
                    }
                    else
                    {
                        oa->write_character(to_char_type(0x1B));
                        write_number(static_cast<std::uint64_t>(j.m_data.m_value.number_integer));
                    }
                }
                else
                {
                    // The conversions below encode the sign in the first
                    // byte, and the value is converted to a positive number.
                    const auto positive_number = -1 - j.m_data.m_value.number_integer;
                    if (j.m_data.m_value.number_integer >= -24)
                    {
                        write_number(static_cast<std::uint8_t>(0x20 + positive_number));
                    }
                    else if (positive_number <= (std::numeric_limits<std::uint8_t>::max)())
                    {
                        oa->write_character(to_char_type(0x38));
                        write_number(static_cast<std::uint8_t>(positive_number));
                    }
                    else if (positive_number <= (std::numeric_limits<std::uint16_t>::max)())
                    {
                        oa->write_character(to_char_type(0x39));
                        write_number(static_cast<std::uint16_t>(positive_number));
                    }
                    else if (positive_number <= (std::numeric_limits<std::uint32_t>::max)())
                    {
                        oa->write_character(to_char_type(0x3A));
                        write_number(static_cast<std::uint32_t>(positive_number));
                    }
                    else
                    {
                        oa->write_character(to_char_type(0x3B));
                        write_number(static_cast<std::uint64_t>(positive_number));
                    }
                }
                break;
            }

            case value_t::number_unsigned:
            {
                if (j.m_data.m_value.number_unsigned <= 0x17)
                {
                    write_number(static_cast<std::uint8_t>(j.m_data.m_value.number_unsigned));
                }
                else if (j.m_data.m_value.number_unsigned <= (std::numeric_limits<std::uint8_t>::max)())
                {
                    oa->write_character(to_char_type(0x18));
                    write_number(static_cast<std::uint8_t>(j.m_data.m_value.number_unsigned));
                }
                else if (j.m_data.m_value.number_unsigned <= (std::numeric_limits<std::uint16_t>::max)())
                {
                    oa->write_character(to_char_type(0x19));
                    write_number(static_cast<std::uint16_t>(j.m_data.m_value.number_unsigned));
                }
                else if (j.m_data.m_value.number_unsigned <= (std::numeric_limits<std::uint32_t>::max)())
                {
                    oa->write_character(to_char_type(0x1A));
                    write_number(static_cast<std::uint32_t>(j.m_data.m_value.number_unsigned));
                }
                else
                {
                    oa->write_character(to_char_type(0x1B));
                    write_number(static_cast<std::uint64_t>(j.m_data.m_value.number_unsigned));
                }
                break;
            }

            case value_t::number_float:
            {
                if (std::isnan(j.m_data.m_value.number_float))
                {
                    // NaN is 0xf97e00 in CBOR
                    oa->write_character(to_char_type(0xF9));
                    oa->write_character(to_char_type(0x7E));
                    oa->write_character(to_char_type(0x00));
                }
                else if (std::isinf(j.m_data.m_value.number_float))
                {
                    // Infinity is 0xf97c00, -Infinity is 0xf9fc00
                    oa->write_character(to_char_type(0xf9));
                    oa->write_character(j.m_data.m_value.number_float > 0 ? to_char_type(0x7C) : to_char_type(0xFC));
                    oa->write_character(to_char_type(0x00));
                }
                else
                {
                    write_compact_float(j.m_data.m_value.number_float, detail::input_format_t::cbor);
                }
                break;
            }

            case value_t::string:
            {
                // step 1: write control byte and the string length
                const auto N = j.m_data.m_value.string->size();
                if (N <= 0x17)
                {
                    write_number(static_cast<std::uint8_t>(0x60 + N));
                }
                else if (N <= (std::numeric_limits<std::uint8_t>::max)())
                {
                    oa->write_character(to_char_type(0x78));
                    write_number(static_cast<std::uint8_t>(N));
                }
                else if (N <= (std::numeric_limits<std::uint16_t>::max)())
                {
                    oa->write_character(to_char_type(0x79));
                    write_number(static_cast<std::uint16_t>(N));
                }
                else if (N <= (std::numeric_limits<std::uint32_t>::max)())
                {
                    oa->write_character(to_char_type(0x7A));
                    write_number(static_cast<std::uint32_t>(N));
                }
                // LCOV_EXCL_START
                else if (N <= (std::numeric_limits<std::uint64_t>::max)())
                {
                    oa->write_character(to_char_type(0x7B));
                    write_number(static_cast<std::uint64_t>(N));
                }
                // LCOV_EXCL_STOP

                // step 2: write the string
                oa->write_characters(
                    reinterpret_cast<const CharType*>(j.m_data.m_value.string->c_str()),
                    j.m_data.m_value.string->size());
                break;
            }

            case value_t::array:
            {
                // step 1: write control byte and the array size
                const auto N = j.m_data.m_value.array->size();
                if (N <= 0x17)
                {
                    write_number(static_cast<std::uint8_t>(0x80 + N));
                }
                else if (N <= (std::numeric_limits<std::uint8_t>::max)())
                {
                    oa->write_character(to_char_type(0x98));
                    write_number(static_cast<std::uint8_t>(N));
                }
                else if (N <= (std::numeric_limits<std::uint16_t>::max)())
                {
                    oa->write_character(to_char_type(0x99));
                    write_number(static_cast<std::uint16_t>(N));
                }
                else if (N <= (std::numeric_limits<std::uint32_t>::max)())
                {
                    oa->write_character(to_char_type(0x9A));
                    write_number(static_cast<std::uint32_t>(N));
                }
                // LCOV_EXCL_START
                else if (N <= (std::numeric_limits<std::uint64_t>::max)())
                {
                    oa->write_character(to_char_type(0x9B));
                    write_number(static_cast<std::uint64_t>(N));
                }
                // LCOV_EXCL_STOP

                // step 2: write each element
                for (const auto& el : *j.m_data.m_value.array)
                {
                    write_cbor(el);
                }
                break;
            }

            case value_t::binary:
            {
                if (j.m_data.m_value.binary->has_subtype())
                {
                    if (j.m_data.m_value.binary->subtype() <= (std::numeric_limits<std::uint8_t>::max)())
                    {
                        write_number(static_cast<std::uint8_t>(0xd8));
                        write_number(static_cast<std::uint8_t>(j.m_data.m_value.binary->subtype()));
                    }
                    else if (j.m_data.m_value.binary->subtype() <= (std::numeric_limits<std::uint16_t>::max)())
                    {
                        write_number(static_cast<std::uint8_t>(0xd9));
                        write_number(static_cast<std::uint16_t>(j.m_data.m_value.binary->subtype()));
                    }
                    else if (j.m_data.m_value.binary->subtype() <= (std::numeric_limits<std::uint32_t>::max)())
                    {
                        write_number(static_cast<std::uint8_t>(0xda));
                        write_number(static_cast<std::uint32_t>(j.m_data.m_value.binary->subtype()));
                    }
                    else if (j.m_data.m_value.binary->subtype() <= (std::numeric_limits<std::uint64_t>::max)())
                    {
                        write_number(static_cast<std::uint8_t>(0xdb));
                        write_number(static_cast<std::uint64_t>(j.m_data.m_value.binary->subtype()));
                    }
                }

                // step 1: write control byte and the binary array size
                const auto N = j.m_data.m_value.binary->size();
                if (N <= 0x17)
                {
                    write_number(static_cast<std::uint8_t>(0x40 + N));
                }
                else if (N <= (std::numeric_limits<std::uint8_t>::max)())
                {
                    oa->write_character(to_char_type(0x58));
                    write_number(static_cast<std::uint8_t>(N));
                }
                else if (N <= (std::numeric_limits<std::uint16_t>::max)())
                {
                    oa->write_character(to_char_type(0x59));
                    write_number(static_cast<std::uint16_t>(N));
                }
                else if (N <= (std::numeric_limits<std::uint32_t>::max)())
                {
                    oa->write_character(to_char_type(0x5A));
                    write_number(static_cast<std::uint32_t>(N));
                }
                // LCOV_EXCL_START
                else if (N <= (std::numeric_limits<std::uint64_t>::max)())
                {
                    oa->write_character(to_char_type(0x5B));
                    write_number(static_cast<std::uint64_t>(N));
                }
                // LCOV_EXCL_STOP

                // step 2: write each element
                oa->write_characters(
                    reinterpret_cast<const CharType*>(j.m_data.m_value.binary->data()),
                    N);

                break;
            }

            case value_t::object:
            {
                // step 1: write control byte and the object size
                const auto N = j.m_data.m_value.object->size();
                if (N <= 0x17)
                {
                    write_number(static_cast<std::uint8_t>(0xA0 + N));
                }
                else if (N <= (std::numeric_limits<std::uint8_t>::max)())
                {
                    oa->write_character(to_char_type(0xB8));
                    write_number(static_cast<std::uint8_t>(N));
                }
                else if (N <= (std::numeric_limits<std::uint16_t>::max)())
                {
                    oa->write_character(to_char_type(0xB9));
                    write_number(static_cast<std::uint16_t>(N));
                }
                else if (N <= (std::numeric_limits<std::uint32_t>::max)())
                {
                    oa->write_character(to_char_type(0xBA));
                    write_number(static_cast<std::uint32_t>(N));
                }
                // LCOV_EXCL_START
                else if (N <= (std::numeric_limits<std::uint64_t>::max)())
                {
                    oa->write_character(to_char_type(0xBB));
                    write_number(static_cast<std::uint64_t>(N));
                }
                // LCOV_EXCL_STOP

                // step 2: write each element
                for (const auto& el : *j.m_data.m_value.object)
                {
                    write_cbor(el.first);
                    write_cbor(el.second);
                }
                break;
            }

            case value_t::discarded:
            default:
                break;
        }
    }

    /*!
    @param[in] j  JSON value to serialize
    */
    void write_msgpack(const BasicJsonType& j)
    {
        switch (j.type())
        {
            case value_t::null: // nil
            {
                oa->write_character(to_char_type(0xC0));
                break;
            }

            case value_t::boolean: // true and false
            {
                oa->write_character(j.m_data.m_value.boolean
                                    ? to_char_type(0xC3)
                                    : to_char_type(0xC2));
                break;
            }

            case value_t::number_integer:
            {
                if (j.m_data.m_value.number_integer >= 0)
                {
                    // MessagePack does not differentiate between positive
                    // signed integers and unsigned integers. Therefore, we used
                    // the code from the value_t::number_unsigned case here.
                    if (j.m_data.m_value.number_unsigned < 128)
                    {
                        // positive fixnum
                        write_number(static_cast<std::uint8_t>(j.m_data.m_value.number_integer));
                    }
                    else if (j.m_data.m_value.number_unsigned <= (std::numeric_limits<std::uint8_t>::max)())
                    {
                        // uint 8
                        oa->write_character(to_char_type(0xCC));
                        write_number(static_cast<std::uint8_t>(j.m_data.m_value.number_integer));
                    }
                    else if (j.m_data.m_value.number_unsigned <= (std::numeric_limits<std::uint16_t>::max)())
                    {
                        // uint 16
                        oa->write_character(to_char_type(0xCD));
                        write_number(static_cast<std::uint16_t>(j.m_data.m_value.number_integer));
                    }
                    else if (j.m_data.m_value.number_unsigned <= (std::numeric_limits<std::uint32_t>::max)())
                    {
                        // uint 32
                        oa->write_character(to_char_type(0xCE));
                        write_number(static_cast<std::uint32_t>(j.m_data.m_value.number_integer));
                    }
                    else if (j.m_data.m_value.number_unsigned <= (std::numeric_limits<std::uint64_t>::max)())
                    {
                        // uint 64
                        oa->write_character(to_char_type(0xCF));
                        write_number(static_cast<std::uint64_t>(j.m_data.m_value.number_integer));
                    }
                }
                else
                {
                    if (j.m_data.m_value.number_integer >= -32)
                    {
                        // negative fixnum
                        write_number(static_cast<std::int8_t>(j.m_data.m_value.number_integer));
                    }
                    else if (j.m_data.m_value.number_integer >= (std::numeric_limits<std::int8_t>::min)() &&
                             j.m_data.m_value.number_integer <= (std::numeric_limits<std::int8_t>::max)())
                    {
                        // int 8
                        oa->write_character(to_char_type(0xD0));
                        write_number(static_cast<std::int8_t>(j.m_data.m_value.number_integer));
                    }
                    else if (j.m_data.m_value.number_integer >= (std::numeric_limits<std::int16_t>::min)() &&
                             j.m_data.m_value.number_integer <= (std::numeric_limits<std::int16_t>::max)())
                    {
                        // int 16
                        oa->write_character(to_char_type(0xD1));
                        write_number(static_cast<std::int16_t>(j.m_data.m_value.number_integer));
                    }
                    else if (j.m_data.m_value.number_integer >= (std::numeric_limits<std::int32_t>::min)() &&
                             j.m_data.m_value.number_integer <= (std::numeric_limits<std::int32_t>::max)())
                    {
                        // int 32
                        oa->write_character(to_char_type(0xD2));
                        write_number(static_cast<std::int32_t>(j.m_data.m_value.number_integer));
                    }
                    else if (j.m_data.m_value.number_integer >= (std::numeric_limits<std::int64_t>::min)() &&
                             j.m_data.m_value.number_integer <= (std::numeric_limits<std::int64_t>::max)())
                    {
                        // int 64
                        oa->write_character(to_char_type(0xD3));
                        write_number(static_cast<std::int64_t>(j.m_data.m_value.number_integer));
                    }
                }
                break;
            }

            case value_t::number_unsigned:
            {
                if (j.m_data.m_value.number_unsigned < 128)
                {
                    // positive fixnum
                    write_number(static_cast<std::uint8_t>(j.m_data.m_value.number_integer));
                }
                else if (j.m_data.m_value.number_unsigned <= (std::numeric_limits<std::uint8_t>::max)())
                {
                    // uint 8
                    oa->write_character(to_char_type(0xCC));
                    write_number(static_cast<std::uint8_t>(j.m_data.m_value.number_integer));
                }
                else if (j.m_data.m_value.number_unsigned <= (std::numeric_limits<std::uint16_t>::max)())
                {
                    // uint 16
                    oa->write_character(to_char_type(0xCD));
                    write_number(static_cast<std::uint16_t>(j.m_data.m_value.number_integer));
                }
                else if (j.m_data.m_value.number_unsigned <= (std::numeric_limits<std::uint32_t>::max)())
                {
                    // uint 32
                    oa->write_character(to_char_type(0xCE));
                    write_number(static_cast<std::uint32_t>(j.m_data.m_value.number_integer));
                }
                else if (j.m_data.m_value.number_unsigned <= (std::numeric_limits<std::uint64_t>::max)())
                {
                    // uint 64
                    oa->write_character(to_char_type(0xCF));
                    write_number(static_cast<std::uint64_t>(j.m_data.m_value.number_integer));
                }
                break;
            }

            case value_t::number_float:
            {
                write_compact_float(j.m_data.m_value.number_float, detail::input_format_t::msgpack);
                break;
            }

            case value_t::string:
            {
                // step 1: write control byte and the string length
                const auto N = j.m_data.m_value.string->size();
                if (N <= 31)
                {
                    // fixstr
                    write_number(static_cast<std::uint8_t>(0xA0 | N));
                }
                else if (N <= (std::numeric_limits<std::uint8_t>::max)())
                {
                    // str 8
                    oa->write_character(to_char_type(0xD9));
                    write_number(static_cast<std::uint8_t>(N));
                }
                else if (N <= (std::numeric_limits<std::uint16_t>::max)())
                {
                    // str 16
                    oa->write_character(to_char_type(0xDA));
                    write_number(static_cast<std::uint16_t>(N));
                }
                else if (N <= (std::numeric_limits<std::uint32_t>::max)())
                {
                    // str 32
                    oa->write_character(to_char_type(0xDB));
                    write_number(static_cast<std::uint32_t>(N));
                }

                // step 2: write the string
                oa->write_characters(
                    reinterpret_cast<const CharType*>(j.m_data.m_value.string->c_str()),
                    j.m_data.m_value.string->size());
                break;
            }

            case value_t::array:
            {
                // step 1: write control byte and the array size
                const auto N = j.m_data.m_value.array->size();
                if (N <= 15)
                {
                    // fixarray
                    write_number(static_cast<std::uint8_t>(0x90 | N));
                }
                else if (N <= (std::numeric_limits<std::uint16_t>::max)())
                {
                    // array 16
                    oa->write_character(to_char_type(0xDC));
                    write_number(static_cast<std::uint16_t>(N));
                }
                else if (N <= (std::numeric_limits<std::uint32_t>::max)())
                {
                    // array 32
                    oa->write_character(to_char_type(0xDD));
                    write_number(static_cast<std::uint32_t>(N));
                }

                // step 2: write each element
                for (const auto& el : *j.m_data.m_value.array)
                {
                    write_msgpack(el);
                }
                break;
            }

            case value_t::binary:
            {
                // step 0: determine if the binary type has a set subtype to
                // determine whether to use the ext or fixext types
                const bool use_ext = j.m_data.m_value.binary->has_subtype();

                // step 1: write control byte and the byte string length
                const auto N = j.m_data.m_value.binary->size();
                if (N <= (std::numeric_limits<std::uint8_t>::max)())
                {
                    std::uint8_t output_type{};
                    bool fixed = true;
                    if (use_ext)
                    {
                        switch (N)
                        {
                            case 1:
                                output_type = 0xD4; // fixext 1
                                break;
                            case 2:
                                output_type = 0xD5; // fixext 2
                                break;
                            case 4:
                                output_type = 0xD6; // fixext 4
                                break;
                            case 8:
                                output_type = 0xD7; // fixext 8
                                break;
                            case 16:
                                output_type = 0xD8; // fixext 16
                                break;
                            default:
                                output_type = 0xC7; // ext 8
                                fixed = false;
                                break;
                        }

                    }
                    else
                    {
                        output_type = 0xC4; // bin 8
                        fixed = false;
                    }

                    oa->write_character(to_char_type(output_type));
                    if (!fixed)
                    {
                        write_number(static_cast<std::uint8_t>(N));
                    }
                }
                else if (N <= (std::numeric_limits<std::uint16_t>::max)())
                {
                    const std::uint8_t output_type = use_ext
                                                     ? 0xC8 // ext 16
                                                     : 0xC5; // bin 16

                    oa->write_character(to_char_type(output_type));
                    write_number(static_cast<std::uint16_t>(N));
                }
                else if (N <= (std::numeric_limits<std::uint32_t>::max)())
                {
                    const std::uint8_t output_type = use_ext
                                                     ? 0xC9 // ext 32
                                                     : 0xC6; // bin 32

                    oa->write_character(to_char_type(output_type));
                    write_number(static_cast<std::uint32_t>(N));
                }

                // step 1.5: if this is an ext type, write the subtype
                if (use_ext)
                {
                    write_number(static_cast<std::int8_t>(j.m_data.m_value.binary->subtype()));
                }

                // step 2: write the byte string
                oa->write_characters(
                    reinterpret_cast<const CharType*>(j.m_data.m_value.binary->data()),
                    N);

                break;
            }

            case value_t::object:
            {
                // step 1: write control byte and the object size
                const auto N = j.m_data.m_value.object->size();
                if (N <= 15)
                {
                    // fixmap
                    write_number(static_cast<std::uint8_t>(0x80 | (N & 0xF)));
                }
                else if (N <= (std::numeric_limits<std::uint16_t>::max)())
                {
                    // map 16
                    oa->write_character(to_char_type(0xDE));
                    write_number(static_cast<std::uint16_t>(N));
                }
                else if (N <= (std::numeric_limits<std::uint32_t>::max)())
                {
                    // map 32
                    oa->write_character(to_char_type(0xDF));
                    write_number(static_cast<std::uint32_t>(N));
                }

                // step 2: write each element
                for (const auto& el : *j.m_data.m_value.object)
                {
                    write_msgpack(el.first);
                    write_msgpack(el.second);
                }
                break;
            }

            case value_t::discarded:
            default:
                break;
        }
    }

    /*!
    @param[in] j  JSON value to serialize
    @param[in] use_count   whether to use '#' prefixes (optimized format)
    @param[in] use_type    whether to use '$' prefixes (optimized format)
    @param[in] add_prefix  whether prefixes need to be used for this value
    @param[in] use_bjdata  whether write in BJData format, default is false
    @param[in] bjdata_version  which BJData version to use, default is draft2
    */
    void write_ubjson(const BasicJsonType& j, const bool use_count,
                      const bool use_type, const bool add_prefix = true,
                      const bool use_bjdata = false, const bjdata_version_t bjdata_version = bjdata_version_t::draft2)
    {
        const bool bjdata_draft3 = use_bjdata && bjdata_version == bjdata_version_t::draft3;

        switch (j.type())
        {
            case value_t::null:
            {
                if (add_prefix)
                {
                    oa->write_character(to_char_type('Z'));
                }
                break;
            }

            case value_t::boolean:
            {
                if (add_prefix)
                {
                    oa->write_character(j.m_data.m_value.boolean
                                        ? to_char_type('T')
                                        : to_char_type('F'));
                }
                break;
            }

            case value_t::number_integer:
            {
                write_number_with_ubjson_prefix(j.m_data.m_value.number_integer, add_prefix, use_bjdata);
                break;
            }

            case value_t::number_unsigned:
            {
                write_number_with_ubjson_prefix(j.m_data.m_value.number_unsigned, add_prefix, use_bjdata);
                break;
            }

            case value_t::number_float:
            {
                write_number_with_ubjson_prefix(j.m_data.m_value.number_float, add_prefix, use_bjdata);
                break;
            }

            case value_t::string:
            {
                if (add_prefix)
                {
                    oa->write_character(to_char_type('S'));
                }
                write_number_with_ubjson_prefix(j.m_data.m_value.string->size(), true, use_bjdata);
                oa->write_characters(
                    reinterpret_cast<const CharType*>(j.m_data.m_value.string->c_str()),
                    j.m_data.m_value.string->size());
                break;
            }

            case value_t::array:
            {
                if (add_prefix)
                {
                    oa->write_character(to_char_type('['));
                }

                bool prefix_required = true;
                if (use_type && !j.m_data.m_value.array->empty())
                {
                    JSON_ASSERT(use_count);
                    const CharType first_prefix = ubjson_prefix(j.front(), use_bjdata);
                    const bool same_prefix = std::all_of(j.begin() + 1, j.end(),
                                                         [this, first_prefix, use_bjdata](const BasicJsonType & v)
                    {
                        return ubjson_prefix(v, use_bjdata) == first_prefix;
                    });

                    std::vector<CharType> bjdx = {'[', '{', 'S', 'H', 'T', 'F', 'N', 'Z'}; // excluded markers in bjdata optimized type

                    if (same_prefix && !(use_bjdata && std::find(bjdx.begin(), bjdx.end(), first_prefix) != bjdx.end()))
                    {
                        prefix_required = false;
                        oa->write_character(to_char_type('$'));
                        oa->write_character(first_prefix);
                    }
                }

                if (use_count)
                {
                    oa->write_character(to_char_type('#'));
                    write_number_with_ubjson_prefix(j.m_data.m_value.array->size(), true, use_bjdata);
                }

                for (const auto& el : *j.m_data.m_value.array)
                {
                    write_ubjson(el, use_count, use_type, prefix_required, use_bjdata, bjdata_version);
                }

                if (!use_count)
                {
                    oa->write_character(to_char_type(']'));
                }

                break;
            }

            case value_t::binary:
            {
                if (add_prefix)
                {
                    oa->write_character(to_char_type('['));
                }

                if (use_type && (bjdata_draft3 || !j.m_data.m_value.binary->empty()))
                {
                    JSON_ASSERT(use_count);
                    oa->write_character(to_char_type('$'));
                    oa->write_character(bjdata_draft3 ? 'B' : 'U');
                }

                if (use_count)
                {
                    oa->write_character(to_char_type('#'));
                    write_number_with_ubjson_prefix(j.m_data.m_value.binary->size(), true, use_bjdata);
                }

                if (use_type)
                {
                    oa->write_characters(
                        reinterpret_cast<const CharType*>(j.m_data.m_value.binary->data()),
                        j.m_data.m_value.binary->size());
                }
                else
                {
                    for (size_t i = 0; i < j.m_data.m_value.binary->size(); ++i)
                    {
                        oa->write_character(to_char_type(bjdata_draft3 ? 'B' : 'U'));
                        oa->write_character(j.m_data.m_value.binary->data()[i]);
                    }
                }

                if (!use_count)
                {
                    oa->write_character(to_char_type(']'));
                }

                break;
            }

            case value_t::object:
            {
                if (use_bjdata && j.m_data.m_value.object->size() == 3 && j.m_data.m_value.object->find("_ArrayType_") != j.m_data.m_value.object->end() && j.m_data.m_value.object->find("_ArraySize_") != j.m_data.m_value.object->end() && j.m_data.m_value.object->find("_ArrayData_") != j.m_data.m_value.object->end())
                {
                    if (!write_bjdata_ndarray(*j.m_data.m_value.object, use_count, use_type, bjdata_version))  // decode bjdata ndarray in the JData format (https://github.com/NeuroJSON/jdata)
                    {
                        break;
                    }
                }

                if (add_prefix)
                {
                    oa->write_character(to_char_type('{'));
                }

                bool prefix_required = true;
                if (use_type && !j.m_data.m_value.object->empty())
                {
                    JSON_ASSERT(use_count);
                    const CharType first_prefix = ubjson_prefix(j.front(), use_bjdata);
                    const bool same_prefix = std::all_of(j.begin(), j.end(),
                                                         [this, first_prefix, use_bjdata](const BasicJsonType & v)
                    {
                        return ubjson_prefix(v, use_bjdata) == first_prefix;
                    });

                    std::vector<CharType> bjdx = {'[', '{', 'S', 'H', 'T', 'F', 'N', 'Z'}; // excluded markers in bjdata optimized type

                    if (same_prefix && !(use_bjdata && std::find(bjdx.begin(), bjdx.end(), first_prefix) != bjdx.end()))
                    {
                        prefix_required = false;
                        oa->write_character(to_char_type('$'));
                        oa->write_character(first_prefix);
                    }
                }

                if (use_count)
                {
                    oa->write_character(to_char_type('#'));
                    write_number_with_ubjson_prefix(j.m_data.m_value.object->size(), true, use_bjdata);
                }

                for (const auto& el : *j.m_data.m_value.object)
                {
                    write_number_with_ubjson_prefix(el.first.size(), true, use_bjdata);
                    oa->write_characters(
                        reinterpret_cast<const CharType*>(el.first.c_str()),
                        el.first.size());
                    write_ubjson(el.second, use_count, use_type, prefix_required, use_bjdata, bjdata_version);
                }

                if (!use_count)
                {
                    oa->write_character(to_char_type('}'));
                }

                break;
            }

            case value_t::discarded:
            default:
                break;
        }
    }

  private:
    //////////
    // BSON //
    //////////

    /*!
    @return The size of a BSON document entry header, including the id marker
            and the entry name size (and its null-terminator).
    */
    static std::size_t calc_bson_entry_header_size(const string_t& name, const BasicJsonType& j)
    {
        const auto it = name.find(static_cast<typename string_t::value_type>(0));
        if (JSON_HEDLEY_UNLIKELY(it != BasicJsonType::string_t::npos))
        {
            JSON_THROW(out_of_range::create(409, concat("BSON key cannot contain code point U+0000 (at byte ", std::to_string(it), ")"), &j));
            static_cast<void>(j);
        }

        return /*id*/ 1ul + name.size() + /*zero-terminator*/1u;
    }

    /*!
    @brief Writes the given @a element_type and @a name to the output adapter
    */
    void write_bson_entry_header(const string_t& name,
                                 const std::uint8_t element_type)
    {
        oa->write_character(to_char_type(element_type)); // boolean
        oa->write_characters(
            reinterpret_cast<const CharType*>(name.c_str()),
            name.size() + 1u);
    }

    /*!
    @brief Writes a BSON element with key @a name and boolean value @a value
    */
    void write_bson_boolean(const string_t& name,
                            const bool value)
    {
        write_bson_entry_header(name, 0x08);
        oa->write_character(value ? to_char_type(0x01) : to_char_type(0x00));
    }

    /*!
    @brief Writes a BSON element with key @a name and double value @a value
    */
    void write_bson_double(const string_t& name,
                           const double value)
    {
        write_bson_entry_header(name, 0x01);
        write_number<double>(value, true);
    }

    /*!
    @return The size of the BSON-encoded string in @a value
    */
    static std::size_t calc_bson_string_size(const string_t& value)
    {
        return sizeof(std::int32_t) + value.size() + 1ul;
    }

    /*!
    @brief Writes a BSON element with key @a name and string value @a value
    */
    void write_bson_string(const string_t& name,
                           const string_t& value)
    {
        write_bson_entry_header(name, 0x02);

        write_number<std::int32_t>(static_cast<std::int32_t>(value.size() + 1ul), true);
        oa->write_characters(
            reinterpret_cast<const CharType*>(value.c_str()),
            value.size() + 1);
    }

    /*!
    @brief Writes a BSON element with key @a name and null value
    */
    void write_bson_null(const string_t& name)
    {
        write_bson_entry_header(name, 0x0A);
    }

    /*!
    @return The size of the BSON-encoded integer @a value
    */
    static std::size_t calc_bson_integer_size(const std::int64_t value)
    {
        return (std::numeric_limits<std::int32_t>::min)() <= value && value <= (std::numeric_limits<std::int32_t>::max)()
               ? sizeof(std::int32_t)
               : sizeof(std::int64_t);
    }

    /*!
    @brief Writes a BSON element with key @a name and integer @a value
    */
    void write_bson_integer(const string_t& name,
                            const std::int64_t value)
    {
        if ((std::numeric_limits<std::int32_t>::min)() <= value && value <= (std::numeric_limits<std::int32_t>::max)())
        {
            write_bson_entry_header(name, 0x10); // int32
            write_number<std::int32_t>(static_cast<std::int32_t>(value), true);
        }
        else
        {
            write_bson_entry_header(name, 0x12); // int64
            write_number<std::int64_t>(static_cast<std::int64_t>(value), true);
        }
    }

    /*!
    @return The size of the BSON-encoded unsigned integer in @a j
    */
    static constexpr std::size_t calc_bson_unsigned_size(const std::uint64_t value) noexcept
    {
        return (value <= static_cast<std::uint64_t>((std::numeric_limits<std::int32_t>::max)()))
               ? sizeof(std::int32_t)
               : sizeof(std::int64_t);
    }

    /*!
    @brief Writes a BSON element with key @a name and unsigned @a value
    */
    void write_bson_unsigned(const string_t& name,
                             const BasicJsonType& j)
    {
        if (j.m_data.m_value.number_unsigned <= static_cast<std::uint64_t>((std::numeric_limits<std::int32_t>::max)()))
        {
            write_bson_entry_header(name, 0x10 /* int32 */);
            write_number<std::int32_t>(static_cast<std::int32_t>(j.m_data.m_value.number_unsigned), true);
        }
        else if (j.m_data.m_value.number_unsigned <= static_cast<std::uint64_t>((std::numeric_limits<std::int64_t>::max)()))
        {
            write_bson_entry_header(name, 0x12 /* int64 */);
            write_number<std::int64_t>(static_cast<std::int64_t>(j.m_data.m_value.number_unsigned), true);
        }
        else
        {
            write_bson_entry_header(name, 0x11 /* uint64 */);
            write_number<std::uint64_t>(static_cast<std::uint64_t>(j.m_data.m_value.number_unsigned), true);
        }
    }

    /*!
    @brief Writes a BSON element with key @a name and object @a value
    */
    void write_bson_object_entry(const string_t& name,
                                 const typename BasicJsonType::object_t& value)
    {
        write_bson_entry_header(name, 0x03); // object
        write_bson_object(value);
    }

    /*!
    @return The size of the BSON-encoded array @a value
    */
    static std::size_t calc_bson_array_size(const typename BasicJsonType::array_t& value)
    {
        std::size_t array_index = 0ul;

        const std::size_t embedded_document_size = std::accumulate(std::begin(value), std::end(value), static_cast<std::size_t>(0), [&array_index](std::size_t result, const typename BasicJsonType::array_t::value_type & el)
        {
            return result + calc_bson_element_size(std::to_string(array_index++), el);
        });

        return sizeof(std::int32_t) + embedded_document_size + 1ul;
    }

    /*!
    @return The size of the BSON-encoded binary array @a value
    */
    static std::size_t calc_bson_binary_size(const typename BasicJsonType::binary_t& value)
    {
        return sizeof(std::int32_t) + value.size() + 1ul;
    }

    /*!
    @brief Writes a BSON element with key @a name and array @a value
    */
    void write_bson_array(const string_t& name,
                          const typename BasicJsonType::array_t& value)
    {
        write_bson_entry_header(name, 0x04); // array
        write_number<std::int32_t>(static_cast<std::int32_t>(calc_bson_array_size(value)), true);

        std::size_t array_index = 0ul;

        for (const auto& el : value)
        {
            write_bson_element(std::to_string(array_index++), el);
        }

        oa->write_character(to_char_type(0x00));
    }

    /*!
    @brief Writes a BSON element with key @a name and binary value @a value
    */
    void write_bson_binary(const string_t& name,
                           const binary_t& value)
    {
        write_bson_entry_header(name, 0x05);

        write_number<std::int32_t>(static_cast<std::int32_t>(value.size()), true);
        write_number(value.has_subtype() ? static_cast<std::uint8_t>(value.subtype()) : static_cast<std::uint8_t>(0x00));

        oa->write_characters(reinterpret_cast<const CharType*>(value.data()), value.size());
    }

    /*!
    @brief Calculates the size necessary to serialize the JSON value @a j with its @a name
    @return The calculated size for the BSON document entry for @a j with the given @a name.
    */
    static std::size_t calc_bson_element_size(const string_t& name,
            const BasicJsonType& j)
    {
        const auto header_size = calc_bson_entry_header_size(name, j);
        switch (j.type())
        {
            case value_t::object:
                return header_size + calc_bson_object_size(*j.m_data.m_value.object);

            case value_t::array:
                return header_size + calc_bson_array_size(*j.m_data.m_value.array);

            case value_t::binary:
                return header_size + calc_bson_binary_size(*j.m_data.m_value.binary);

            case value_t::boolean:
                return header_size + 1ul;

            case value_t::number_float:
                return header_size + 8ul;

            case value_t::number_integer:
                return header_size + calc_bson_integer_size(j.m_data.m_value.number_integer);

            case value_t::number_unsigned:
                return header_size + calc_bson_unsigned_size(j.m_data.m_value.number_unsigned);

            case value_t::string:
                return header_size + calc_bson_string_size(*j.m_data.m_value.string);

            case value_t::null:
                return header_size + 0ul;

            // LCOV_EXCL_START
            case value_t::discarded:
            default:
                JSON_ASSERT(false); // NOLINT(cert-dcl03-c,hicpp-static-assert,misc-static-assert)
                return 0ul;
                // LCOV_EXCL_STOP
        }
    }

    /*!
    @brief Serializes the JSON value @a j to BSON and associates it with the
           key @a name.
    @param name The name to associate with the JSON entity @a j within the
                current BSON document
    */
    void write_bson_element(const string_t& name,
                            const BasicJsonType& j)
    {
        switch (j.type())
        {
            case value_t::object:
                return write_bson_object_entry(name, *j.m_data.m_value.object);

            case value_t::array:
                return write_bson_array(name, *j.m_data.m_value.array);

            case value_t::binary:
                return write_bson_binary(name, *j.m_data.m_value.binary);

            case value_t::boolean:
                return write_bson_boolean(name, j.m_data.m_value.boolean);

            case value_t::number_float:
                return write_bson_double(name, j.m_data.m_value.number_float);

            case value_t::number_integer:
                return write_bson_integer(name, j.m_data.m_value.number_integer);

            case value_t::number_unsigned:
                return write_bson_unsigned(name, j);

            case value_t::string:
                return write_bson_string(name, *j.m_data.m_value.string);

            case value_t::null:
                return write_bson_null(name);

            // LCOV_EXCL_START
            case value_t::discarded:
            default:
                JSON_ASSERT(false); // NOLINT(cert-dcl03-c,hicpp-static-assert,misc-static-assert)
                return;
                // LCOV_EXCL_STOP
        }
    }

    /*!
    @brief Calculates the size of the BSON serialization of the given
           JSON-object @a j.
    @param[in] value  JSON value to serialize
    @pre       value.type() == value_t::object
    */
    static std::size_t calc_bson_object_size(const typename BasicJsonType::object_t& value)
    {
        const std::size_t document_size = std::accumulate(value.begin(), value.end(), static_cast<std::size_t>(0),
                                          [](size_t result, const typename BasicJsonType::object_t::value_type & el)
        {
            return result += calc_bson_element_size(el.first, el.second);
        });

        return sizeof(std::int32_t) + document_size + 1ul;
    }

    /*!
    @param[in] value  JSON value to serialize
    @pre       value.type() == value_t::object
    */
    void write_bson_object(const typename BasicJsonType::object_t& value)
    {
        write_number<std::int32_t>(static_cast<std::int32_t>(calc_bson_object_size(value)), true);

        for (const auto& el : value)
        {
            write_bson_element(el.first, el.second);
        }

        oa->write_character(to_char_type(0x00));
    }

    //////////
    // CBOR //
    //////////

    static constexpr CharType get_cbor_float_prefix(float /*unused*/)
    {
        return to_char_type(0xFA);  // Single-Precision Float
    }

    static constexpr CharType get_cbor_float_prefix(double /*unused*/)
    {
        return to_char_type(0xFB);  // Double-Precision Float
    }

    /////////////
    // MsgPack //
    /////////////

    static constexpr CharType get_msgpack_float_prefix(float /*unused*/)
    {
        return to_char_type(0xCA);  // float 32
    }

    static constexpr CharType get_msgpack_float_prefix(double /*unused*/)
    {
        return to_char_type(0xCB);  // float 64
    }

    ////////////
    // UBJSON //
    ////////////

    // UBJSON: write number (floating point)
    template<typename NumberType, typename std::enable_if<
                 std::is_floating_point<NumberType>::value, int>::type = 0>
    void write_number_with_ubjson_prefix(const NumberType n,
                                         const bool add_prefix,
                                         const bool use_bjdata)
    {
        if (add_prefix)
        {
            oa->write_character(get_ubjson_float_prefix(n));
        }
        write_number(n, use_bjdata);
    }

    // UBJSON: write number (unsigned integer)
    template<typename NumberType, typename std::enable_if<
                 std::is_unsigned<NumberType>::value, int>::type = 0>
    void write_number_with_ubjson_prefix(const NumberType n,
                                         const bool add_prefix,
                                         const bool use_bjdata)
    {
        if (n <= static_cast<std::uint64_t>((std::numeric_limits<std::int8_t>::max)()))
        {
            if (add_prefix)
            {
                oa->write_character(to_char_type('i'));  // int8
            }
            write_number(static_cast<std::uint8_t>(n), use_bjdata);
        }
        else if (n <= (std::numeric_limits<std::uint8_t>::max)())
        {
            if (add_prefix)
            {
                oa->write_character(to_char_type('U'));  // uint8
            }
            write_number(static_cast<std::uint8_t>(n), use_bjdata);
        }
        else if (n <= static_cast<std::uint64_t>((std::numeric_limits<std::int16_t>::max)()))
        {
            if (add_prefix)
            {
                oa->write_character(to_char_type('I'));  // int16
            }
            write_number(static_cast<std::int16_t>(n), use_bjdata);
        }
        else if (use_bjdata && n <= static_cast<uint64_t>((std::numeric_limits<uint16_t>::max)()))
        {
            if (add_prefix)
            {
                oa->write_character(to_char_type('u'));  // uint16 - bjdata only
            }
            write_number(static_cast<std::uint16_t>(n), use_bjdata);
        }
        else if (n <= static_cast<std::uint64_t>((std::numeric_limits<std::int32_t>::max)()))
        {
            if (add_prefix)
            {
                oa->write_character(to_char_type('l'));  // int32
            }
            write_number(static_cast<std::int32_t>(n), use_bjdata);
        }
        else if (use_bjdata && n <= static_cast<uint64_t>((std::numeric_limits<uint32_t>::max)()))
        {
            if (add_prefix)
            {
                oa->write_character(to_char_type('m'));  // uint32 - bjdata only
            }
            write_number(static_cast<std::uint32_t>(n), use_bjdata);
        }
        else if (n <= static_cast<std::uint64_t>((std::numeric_limits<std::int64_t>::max)()))
        {
            if (add_prefix)
            {
                oa->write_character(to_char_type('L'));  // int64
            }
            write_number(static_cast<std::int64_t>(n), use_bjdata);
        }
        else if (use_bjdata && n <= (std::numeric_limits<uint64_t>::max)())
        {
            if (add_prefix)
            {
                oa->write_character(to_char_type('M'));  // uint64 - bjdata only
            }
            write_number(static_cast<std::uint64_t>(n), use_bjdata);
        }
        else
        {
            if (add_prefix)
            {
                oa->write_character(to_char_type('H'));  // high-precision number
            }

            const auto number = BasicJsonType(n).dump();
            write_number_with_ubjson_prefix(number.size(), true, use_bjdata);
            for (std::size_t i = 0; i < number.size(); ++i)
            {
                oa->write_character(to_char_type(static_cast<std::uint8_t>(number[i])));
            }
        }
    }

    // UBJSON: write number (signed integer)
    template < typename NumberType, typename std::enable_if <
                   std::is_signed<NumberType>::value&&
                   !std::is_floating_point<NumberType>::value, int >::type = 0 >
    void write_number_with_ubjson_prefix(const NumberType n,
                                         const bool add_prefix,
                                         const bool use_bjdata)
    {
        if ((std::numeric_limits<std::int8_t>::min)() <= n && n <= (std::numeric_limits<std::int8_t>::max)())
        {
            if (add_prefix)
            {
                oa->write_character(to_char_type('i'));  // int8
            }
            write_number(static_cast<std::int8_t>(n), use_bjdata);
        }
        else if (static_cast<std::int64_t>((std::numeric_limits<std::uint8_t>::min)()) <= n && n <= static_cast<std::int64_t>((std::numeric_limits<std::uint8_t>::max)()))
        {
            if (add_prefix)
            {
                oa->write_character(to_char_type('U'));  // uint8
            }
            write_number(static_cast<std::uint8_t>(n), use_bjdata);
        }
        else if ((std::numeric_limits<std::int16_t>::min)() <= n && n <= (std::numeric_limits<std::int16_t>::max)())
        {
            if (add_prefix)
            {
                oa->write_character(to_char_type('I'));  // int16
            }
            write_number(static_cast<std::int16_t>(n), use_bjdata);
        }
        else if (use_bjdata && (static_cast<std::int64_t>((std::numeric_limits<std::uint16_t>::min)()) <= n && n <= static_cast<std::int64_t>((std::numeric_limits<std::uint16_t>::max)())))
        {
            if (add_prefix)
            {
                oa->write_character(to_char_type('u'));  // uint16 - bjdata only
            }
            write_number(static_cast<uint16_t>(n), use_bjdata);
        }
        else if ((std::numeric_limits<std::int32_t>::min)() <= n && n <= (std::numeric_limits<std::int32_t>::max)())
        {
            if (add_prefix)
            {
                oa->write_character(to_char_type('l'));  // int32
            }
            write_number(static_cast<std::int32_t>(n), use_bjdata);
        }
        else if (use_bjdata && (static_cast<std::int64_t>((std::numeric_limits<std::uint32_t>::min)()) <= n && n <= static_cast<std::int64_t>((std::numeric_limits<std::uint32_t>::max)())))
        {
            if (add_prefix)
            {
                oa->write_character(to_char_type('m'));  // uint32 - bjdata only
            }
            write_number(static_cast<uint32_t>(n), use_bjdata);
        }
        else if ((std::numeric_limits<std::int64_t>::min)() <= n && n <= (std::numeric_limits<std::int64_t>::max)())
        {
            if (add_prefix)
            {
                oa->write_character(to_char_type('L'));  // int64
            }
            write_number(static_cast<std::int64_t>(n), use_bjdata);
        }
        // LCOV_EXCL_START
        else
        {
            if (add_prefix)
            {
                oa->write_character(to_char_type('H'));  // high-precision number
            }

            const auto number = BasicJsonType(n).dump();
            write_number_with_ubjson_prefix(number.size(), true, use_bjdata);
            for (std::size_t i = 0; i < number.size(); ++i)
            {
                oa->write_character(to_char_type(static_cast<std::uint8_t>(number[i])));
            }
        }
        // LCOV_EXCL_STOP
    }

    /*!
    @brief determine the type prefix of container values
    */
    CharType ubjson_prefix(const BasicJsonType& j, const bool use_bjdata) const noexcept
    {
        switch (j.type())
        {
            case value_t::null:
                return 'Z';

            case value_t::boolean:
                return j.m_data.m_value.boolean ? 'T' : 'F';

            case value_t::number_integer:
            {
                if ((std::numeric_limits<std::int8_t>::min)() <= j.m_data.m_value.number_integer && j.m_data.m_value.number_integer <= (std::numeric_limits<std::int8_t>::max)())
                {
                    return 'i';
                }
                if ((std::numeric_limits<std::uint8_t>::min)() <= j.m_data.m_value.number_integer && j.m_data.m_value.number_integer <= (std::numeric_limits<std::uint8_t>::max)())
                {
                    return 'U';
                }
                if ((std::numeric_limits<std::int16_t>::min)() <= j.m_data.m_value.number_integer && j.m_data.m_value.number_integer <= (std::numeric_limits<std::int16_t>::max)())
                {
                    return 'I';
                }
                if (use_bjdata && ((std::numeric_limits<std::uint16_t>::min)() <= j.m_data.m_value.number_integer && j.m_data.m_value.number_integer <= (std::numeric_limits<std::uint16_t>::max)()))
                {
                    return 'u';
                }
                if ((std::numeric_limits<std::int32_t>::min)() <= j.m_data.m_value.number_integer && j.m_data.m_value.number_integer <= (std::numeric_limits<std::int32_t>::max)())
                {
                    return 'l';
                }
                if (use_bjdata && ((std::numeric_limits<std::uint32_t>::min)() <= j.m_data.m_value.number_integer && j.m_data.m_value.number_integer <= (std::numeric_limits<std::uint32_t>::max)()))
                {
                    return 'm';
                }
                if ((std::numeric_limits<std::int64_t>::min)() <= j.m_data.m_value.number_integer && j.m_data.m_value.number_integer <= (std::numeric_limits<std::int64_t>::max)())
                {
                    return 'L';
                }
                // anything else is treated as high-precision number
                return 'H'; // LCOV_EXCL_LINE
            }

            case value_t::number_unsigned:
            {
                if (j.m_data.m_value.number_unsigned <= static_cast<std::uint64_t>((std::numeric_limits<std::int8_t>::max)()))
                {
                    return 'i';
                }
                if (j.m_data.m_value.number_unsigned <= static_cast<std::uint64_t>((std::numeric_limits<std::uint8_t>::max)()))
                {
                    return 'U';
                }
                if (j.m_data.m_value.number_unsigned <= static_cast<std::uint64_t>((std::numeric_limits<std::int16_t>::max)()))
                {
                    return 'I';
                }
                if (use_bjdata && j.m_data.m_value.number_unsigned <= static_cast<std::uint64_t>((std::numeric_limits<std::uint16_t>::max)()))
                {
                    return 'u';
                }
                if (j.m_data.m_value.number_unsigned <= static_cast<std::uint64_t>((std::numeric_limits<std::int32_t>::max)()))
                {
                    return 'l';
                }
                if (use_bjdata && j.m_data.m_value.number_unsigned <= static_cast<std::uint64_t>((std::numeric_limits<std::uint32_t>::max)()))
                {
                    return 'm';
                }
                if (j.m_data.m_value.number_unsigned <= static_cast<std::uint64_t>((std::numeric_limits<std::int64_t>::max)()))
                {
                    return 'L';
                }
                if (use_bjdata && j.m_data.m_value.number_unsigned <= (std::numeric_limits<std::uint64_t>::max)())
                {
                    return 'M';
                }
                // anything else is treated as high-precision number
                return 'H'; // LCOV_EXCL_LINE
            }

            case value_t::number_float:
                return get_ubjson_float_prefix(j.m_data.m_value.number_float);

            case value_t::string:
                return 'S';

            case value_t::array: // fallthrough
            case value_t::binary:
                return '[';

            case value_t::object:
                return '{';

            case value_t::discarded:
            default:  // discarded values
                return 'N';
        }
    }

    static constexpr CharType get_ubjson_float_prefix(float /*unused*/)
    {
        return 'd';  // float 32
    }

    static constexpr CharType get_ubjson_float_prefix(double /*unused*/)
    {
        return 'D';  // float 64
    }

    /*!
    @return false if the object is successfully converted to a bjdata ndarray, true if the type or size is invalid
    */
    bool write_bjdata_ndarray(const typename BasicJsonType::object_t& value, const bool use_count, const bool use_type, const bjdata_version_t bjdata_version)
    {
        std::map<string_t, CharType> bjdtype = {{"uint8", 'U'},  {"int8", 'i'},  {"uint16", 'u'}, {"int16", 'I'},
            {"uint32", 'm'}, {"int32", 'l'}, {"uint64", 'M'}, {"int64", 'L'}, {"single", 'd'}, {"double", 'D'},
            {"char", 'C'}, {"byte", 'B'}
        };

        string_t key = "_ArrayType_";
        auto it = bjdtype.find(static_cast<string_t>(value.at(key)));
        if (it == bjdtype.end())
        {
            return true;
        }
        CharType dtype = it->second;

        key = "_ArraySize_";
        std::size_t len = (value.at(key).empty() ? 0 : 1);
        for (const auto& el : value.at(key))
        {
            len *= static_cast<std::size_t>(el.m_data.m_value.number_unsigned);
        }

        key = "_ArrayData_";
        if (value.at(key).size() != len)
        {
            return true;
        }

        oa->write_character('[');
        oa->write_character('$');
        oa->write_character(dtype);
        oa->write_character('#');

        key = "_ArraySize_";
        write_ubjson(value.at(key), use_count, use_type, true,  true, bjdata_version);

        key = "_ArrayData_";
        if (dtype == 'U' || dtype == 'C' || dtype == 'B')
        {
            for (const auto& el : value.at(key))
            {
                write_number(static_cast<std::uint8_t>(el.m_data.m_value.number_unsigned), true);
            }
        }
        else if (dtype == 'i')
        {
            for (const auto& el : value.at(key))
            {
                write_number(static_cast<std::int8_t>(el.m_data.m_value.number_integer), true);
            }
        }
        else if (dtype == 'u')
        {
            for (const auto& el : value.at(key))
            {
                write_number(static_cast<std::uint16_t>(el.m_data.m_value.number_unsigned), true);
            }
        }
        else if (dtype == 'I')
        {
            for (const auto& el : value.at(key))
            {
                write_number(static_cast<std::int16_t>(el.m_data.m_value.number_integer), true);
            }
        }
        else if (dtype == 'm')
        {
            for (const auto& el : value.at(key))
            {
                write_number(static_cast<std::uint32_t>(el.m_data.m_value.number_unsigned), true);
            }
        }
        else if (dtype == 'l')
        {
            for (const auto& el : value.at(key))
            {
                write_number(static_cast<std::int32_t>(el.m_data.m_value.number_integer), true);
            }
        }
        else if (dtype == 'M')
        {
            for (const auto& el : value.at(key))
            {
                write_number(static_cast<std::uint64_t>(el.m_data.m_value.number_unsigned), true);
            }
        }
        else if (dtype == 'L')
        {
            for (const auto& el : value.at(key))
            {
                write_number(static_cast<std::int64_t>(el.m_data.m_value.number_integer), true);
            }
        }
        else if (dtype == 'd')
        {
            for (const auto& el : value.at(key))
            {
                write_number(static_cast<float>(el.m_data.m_value.number_float), true);
            }
        }
        else if (dtype == 'D')
        {
            for (const auto& el : value.at(key))
            {
                write_number(static_cast<double>(el.m_data.m_value.number_float), true);
            }
        }
        return false;
    }

    ///////////////////////
    // Utility functions //
    ///////////////////////

    /*
    @brief write a number to output input
    @param[in] n number of type @a NumberType
    @param[in] OutputIsLittleEndian Set to true if output data is
                                 required to be little endian
    @tparam NumberType the type of the number

    @note This function needs to respect the system's endianness, because bytes
          in CBOR, MessagePack, and UBJSON are stored in network order (big
          endian) and therefore need reordering on little endian systems.
          On the other hand, BSON and BJData use little endian and should reorder
          on big endian systems.
    */
    template<typename NumberType>
    void write_number(const NumberType n, const bool OutputIsLittleEndian = false)
    {
        // step 1: write number to array of length NumberType
        std::array<CharType, sizeof(NumberType)> vec{};
        std::memcpy(vec.data(), &n, sizeof(NumberType));

        // step 2: write array to output (with possible reordering)
        if (is_little_endian != OutputIsLittleEndian)
        {
            // reverse byte order prior to conversion if necessary
            std::reverse(vec.begin(), vec.end());
        }

        oa->write_characters(vec.data(), sizeof(NumberType));
    }

    void write_compact_float(const number_float_t n, detail::input_format_t format)
    {
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"
#endif
        if (static_cast<double>(n) >= static_cast<double>(std::numeric_limits<float>::lowest()) &&
                static_cast<double>(n) <= static_cast<double>((std::numeric_limits<float>::max)()) &&
                static_cast<double>(static_cast<float>(n)) == static_cast<double>(n))
        {
            oa->write_character(format == detail::input_format_t::cbor
                                ? get_cbor_float_prefix(static_cast<float>(n))
                                : get_msgpack_float_prefix(static_cast<float>(n)));
            write_number(static_cast<float>(n));
        }
        else
        {
            oa->write_character(format == detail::input_format_t::cbor
                                ? get_cbor_float_prefix(n)
                                : get_msgpack_float_prefix(n));
            write_number(n);
        }
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
    }

  public:
    // The following to_char_type functions are implement the conversion
    // between uint8_t and CharType. In case CharType is not unsigned,
    // such a conversion is required to allow values greater than 128.
    // See <https://github.com/nlohmann/json/issues/1286> for a discussion.
    template < typename C = CharType,
               enable_if_t < std::is_signed<C>::value && std::is_signed<char>::value > * = nullptr >
    static constexpr CharType to_char_type(std::uint8_t x) noexcept
    {
        return *reinterpret_cast<char*>(&x);
    }

    template < typename C = CharType,
               enable_if_t < std::is_signed<C>::value && std::is_unsigned<char>::value > * = nullptr >
    static CharType to_char_type(std::uint8_t x) noexcept
    {
        static_assert(sizeof(std::uint8_t) == sizeof(CharType), "size of CharType must be equal to std::uint8_t");
        static_assert(std::is_trivial<CharType>::value, "CharType must be trivial");
        CharType result;
        std::memcpy(&result, &x, sizeof(x));
        return result;
    }

    template<typename C = CharType,
             enable_if_t<std::is_unsigned<C>::value>* = nullptr>
    static constexpr CharType to_char_type(std::uint8_t x) noexcept
    {
        return x;
    }

    template < typename InputCharType, typename C = CharType,
               enable_if_t <
                   std::is_signed<C>::value &&
                   std::is_signed<char>::value &&
                   std::is_same<char, typename std::remove_cv<InputCharType>::type>::value
                   > * = nullptr >
    static constexpr CharType to_char_type(InputCharType x) noexcept
    {
        return x;
    }

  private:
    /// whether we can assume little endianness
    const bool is_little_endian = little_endianness();

    /// the output
    output_adapter_t<CharType> oa = nullptr;
};

}  // namespace detail
NLOHMANN_JSON_NAMESPACE_END

// #include <nlohmann/detail/output/output_adapters.hpp>

// #include <nlohmann/detail/output/serializer.hpp>
//     __ _____ _____ _____
//  __|  |   __|     |   | |  JSON for Modern C++
// |  |  |__   |  |  | | | |  version 3.12.0
// |_____|_____|_____|_|___|  https://github.com/nlohmann/json
//
// SPDX-FileCopyrightText: 2008 - 2009 Björn Hoehrmann <bjoern@hoehrmann.de>
// SPDX-FileCopyrightText: 2013 - 2025 Niels Lohmann <https://nlohmann.me>
// SPDX-License-Identifier: MIT



#include <algorithm> // reverse, remove, fill, find, none_of
#include <array> // array
#include <clocale> // localeconv, lconv
#include <cmath> // labs, isfinite, isnan, signbit
#include <cstddef> // size_t, ptrdiff_t
#include <cstdint> // uint8_t
#include <cstdio> // snprintf
#include <limits> // numeric_limits
#include <string> // string, char_traits
#include <iomanip> // setfill, setw
#include <type_traits> // is_same
#include <utility> // move

// #include <nlohmann/detail/conversions/to_chars.hpp>
//     __ _____ _____ _____
//  __|  |   __|     |   | |  JSON for Modern C++
// |  |  |__   |  |  | | | |  version 3.12.0
// |_____|_____|_____|_|___|  https://github.com/nlohmann/json
//
// SPDX-FileCopyrightText: 2009 Florian Loitsch <https://florian.loitsch.com/>
// SPDX-FileCopyrightText: 2013 - 2025 Niels Lohmann <https://nlohmann.me>
// SPDX-License-Identifier: MIT



#include <array> // array
#include <cmath>   // signbit, isfinite
#include <cstdint> // intN_t, uintN_t
#include <cstring> // memcpy, memmove
#include <limits> // numeric_limits
#include <type_traits> // conditional

// #include <nlohmann/detail/macro_scope.hpp>


NLOHMANN_JSON_NAMESPACE_BEGIN
namespace detail
{

/*!
@brief implements the Grisu2 algorithm for binary to decimal floating-point
conversion.

This implementation is a slightly modified version of the reference
implementation which may be obtained from
http://florian.loitsch.com/publications (bench.tar.gz).

The code is distributed under the MIT license, Copyright (c) 2009 Florian Loitsch.

For a detailed description of the algorithm see:

[1] Loitsch, "Printing Floating-Point Numbers Quickly and Accurately with
    Integers", Proceedings of the ACM SIGPLAN 2010 Conference on Programming
    Language Design and Implementation, PLDI 2010
[2] Burger, Dybvig, "Printing Floating-Point Numbers Quickly and Accurately",
    Proceedings of the ACM SIGPLAN 1996 Conference on Programming Language
    Design and Implementation, PLDI 1996
*/
namespace dtoa_impl
{

template<typename Target, typename Source>
Target reinterpret_bits(const Source source)
{
    static_assert(sizeof(Target) == sizeof(Source), "size mismatch");

    Target target;
    std::memcpy(&target, &source, sizeof(Source));
    return target;
}

struct diyfp // f * 2^e
{
    static constexpr int kPrecision = 64; // = q

    std::uint64_t f = 0;
    int e = 0;

    constexpr diyfp(std::uint64_t f_, int e_) noexcept : f(f_), e(e_) {}

    /*!
    @brief returns x - y
    @pre x.e == y.e and x.f >= y.f
    */
    static diyfp sub(const diyfp& x, const diyfp& y) noexcept
    {
        JSON_ASSERT(x.e == y.e);
        JSON_ASSERT(x.f >= y.f);

        return {x.f - y.f, x.e};
    }

    /*!
    @brief returns x * y
    @note The result is rounded. (Only the upper q bits are returned.)
    */
    static diyfp mul(const diyfp& x, const diyfp& y) noexcept
    {
        static_assert(kPrecision == 64, "internal error");

        // Computes:
        //  f = round((x.f * y.f) / 2^q)
        //  e = x.e + y.e + q

        // Emulate the 64-bit * 64-bit multiplication:
        //
        // p = u * v
        //   = (u_lo + 2^32 u_hi) (v_lo + 2^32 v_hi)
        //   = (u_lo v_lo         ) + 2^32 ((u_lo v_hi         ) + (u_hi v_lo         )) + 2^64 (u_hi v_hi         )
        //   = (p0                ) + 2^32 ((p1                ) + (p2                )) + 2^64 (p3                )
        //   = (p0_lo + 2^32 p0_hi) + 2^32 ((p1_lo + 2^32 p1_hi) + (p2_lo + 2^32 p2_hi)) + 2^64 (p3                )
        //   = (p0_lo             ) + 2^32 (p0_hi + p1_lo + p2_lo                      ) + 2^64 (p1_hi + p2_hi + p3)
        //   = (p0_lo             ) + 2^32 (Q                                          ) + 2^64 (H                 )
        //   = (p0_lo             ) + 2^32 (Q_lo + 2^32 Q_hi                           ) + 2^64 (H                 )
        //
        // (Since Q might be larger than 2^32 - 1)
        //
        //   = (p0_lo + 2^32 Q_lo) + 2^64 (Q_hi + H)
        //
        // (Q_hi + H does not overflow a 64-bit int)
        //
        //   = p_lo + 2^64 p_hi

        const std::uint64_t u_lo = x.f & 0xFFFFFFFFu;
        const std::uint64_t u_hi = x.f >> 32u;
        const std::uint64_t v_lo = y.f & 0xFFFFFFFFu;
        const std::uint64_t v_hi = y.f >> 32u;

        const std::uint64_t p0 = u_lo * v_lo;
        const std::uint64_t p1 = u_lo * v_hi;
        const std::uint64_t p2 = u_hi * v_lo;
        const std::uint64_t p3 = u_hi * v_hi;

        const std::uint64_t p0_hi = p0 >> 32u;
        const std::uint64_t p1_lo = p1 & 0xFFFFFFFFu;
        const std::uint64_t p1_hi = p1 >> 32u;
        const std::uint64_t p2_lo = p2 & 0xFFFFFFFFu;
        const std::uint64_t p2_hi = p2 >> 32u;

        std::uint64_t Q = p0_hi + p1_lo + p2_lo;

        // The full product might now be computed as
        //
        // p_hi = p3 + p2_hi + p1_hi + (Q >> 32)
        // p_lo = p0_lo + (Q << 32)
        //
        // But in this particular case here, the full p_lo is not required.
        // Effectively we only need to add the highest bit in p_lo to p_hi (and
        // Q_hi + 1 does not overflow).

        Q += std::uint64_t{1} << (64u - 32u - 1u); // round, ties up

        const std::uint64_t h = p3 + p2_hi + p1_hi + (Q >> 32u);

        return {h, x.e + y.e + 64};
    }

    /*!
    @brief normalize x such that the significand is >= 2^(q-1)
    @pre x.f != 0
    */
    static diyfp normalize(diyfp x) noexcept
    {
        JSON_ASSERT(x.f != 0);

        while ((x.f >> 63u) == 0)
        {
            x.f <<= 1u;
            x.e--;
        }

        return x;
    }

    /*!
    @brief normalize x such that the result has the exponent E
    @pre e >= x.e and the upper e - x.e bits of x.f must be zero.
    */
    static diyfp normalize_to(const diyfp& x, const int target_exponent) noexcept
    {
        const int delta = x.e - target_exponent;

        JSON_ASSERT(delta >= 0);
        JSON_ASSERT(((x.f << delta) >> delta) == x.f);

        return {x.f << delta, target_exponent};
    }
};

struct boundaries
{
    diyfp w;
    diyfp minus;
    diyfp plus;
};

/*!
Compute the (normalized) diyfp representing the input number 'value' and its
boundaries.

@pre value must be finite and positive
*/
template<typename FloatType>
boundaries compute_boundaries(FloatType value)
{
    JSON_ASSERT(std::isfinite(value));
    JSON_ASSERT(value > 0);

    // Convert the IEEE representation into a diyfp.
    //
    // If v is denormal:
    //      value = 0.F * 2^(1 - bias) = (          F) * 2^(1 - bias - (p-1))
    // If v is normalized:
    //      value = 1.F * 2^(E - bias) = (2^(p-1) + F) * 2^(E - bias - (p-1))

    static_assert(std::numeric_limits<FloatType>::is_iec559,
                  "internal error: dtoa_short requires an IEEE-754 floating-point implementation");

    constexpr int      kPrecision = std::numeric_limits<FloatType>::digits; // = p (includes the hidden bit)
    constexpr int      kBias      = std::numeric_limits<FloatType>::max_exponent - 1 + (kPrecision - 1);
    constexpr int      kMinExp    = 1 - kBias;
    constexpr std::uint64_t kHiddenBit = std::uint64_t{1} << (kPrecision - 1); // = 2^(p-1)

    using bits_type = typename std::conditional<kPrecision == 24, std::uint32_t, std::uint64_t >::type;

    const auto bits = static_cast<std::uint64_t>(reinterpret_bits<bits_type>(value));
    const std::uint64_t E = bits >> (kPrecision - 1);
    const std::uint64_t F = bits & (kHiddenBit - 1);

    const bool is_denormal = E == 0;
    const diyfp v = is_denormal
                    ? diyfp(F, kMinExp)
                    : diyfp(F + kHiddenBit, static_cast<int>(E) - kBias);

    // Compute the boundaries m- and m+ of the floating-point value
    // v = f * 2^e.
    //
    // Determine v- and v+, the floating-point predecessor and successor if v,
    // respectively.
    //
    //      v- = v - 2^e        if f != 2^(p-1) or e == e_min                (A)
    //         = v - 2^(e-1)    if f == 2^(p-1) and e > e_min                (B)
    //
    //      v+ = v + 2^e
    //
    // Let m- = (v- + v) / 2 and m+ = (v + v+) / 2. All real numbers _strictly_
    // between m- and m+ round to v, regardless of how the input rounding
    // algorithm breaks ties.
    //
    //      ---+-------------+-------------+-------------+-------------+---  (A)
    //         v-            m-            v             m+            v+
    //
    //      -----------------+------+------+-------------+-------------+---  (B)
    //                       v-     m-     v             m+            v+

    const bool lower_boundary_is_closer = F == 0 && E > 1;
    const diyfp m_plus = diyfp((2 * v.f) + 1, v.e - 1);
    const diyfp m_minus = lower_boundary_is_closer
                          ? diyfp((4 * v.f) - 1, v.e - 2)  // (B)
                          : diyfp((2 * v.f) - 1, v.e - 1); // (A)

    // Determine the normalized w+ = m+.
    const diyfp w_plus = diyfp::normalize(m_plus);

    // Determine w- = m- such that e_(w-) = e_(w+).
    const diyfp w_minus = diyfp::normalize_to(m_minus, w_plus.e);

    return {diyfp::normalize(v), w_minus, w_plus};
}

// Given normalized diyfp w, Grisu needs to find a (normalized) cached
// power-of-ten c, such that the exponent of the product c * w = f * 2^e lies
// within a certain range [alpha, gamma] (Definition 3.2 from [1])
//
//      alpha <= e = e_c + e_w + q <= gamma
//
// or
//
//      f_c * f_w * 2^alpha <= f_c 2^(e_c) * f_w 2^(e_w) * 2^q
//                          <= f_c * f_w * 2^gamma
//
// Since c and w are normalized, i.e. 2^(q-1) <= f < 2^q, this implies
//
//      2^(q-1) * 2^(q-1) * 2^alpha <= c * w * 2^q < 2^q * 2^q * 2^gamma
//
// or
//
//      2^(q - 2 + alpha) <= c * w < 2^(q + gamma)
//
// The choice of (alpha,gamma) determines the size of the table and the form of
// the digit generation procedure. Using (alpha,gamma)=(-60,-32) works out well
// in practice:
//
// The idea is to cut the number c * w = f * 2^e into two parts, which can be
// processed independently: An integral part p1, and a fractional part p2:
//
//      f * 2^e = ( (f div 2^-e) * 2^-e + (f mod 2^-e) ) * 2^e
//              = (f div 2^-e) + (f mod 2^-e) * 2^e
//              = p1 + p2 * 2^e
//
// The conversion of p1 into decimal form requires a series of divisions and
// modulos by (a power of) 10. These operations are faster for 32-bit than for
// 64-bit integers, so p1 should ideally fit into a 32-bit integer. This can be
// achieved by choosing
//
//      -e >= 32   or   e <= -32 := gamma
//
// In order to convert the fractional part
//
//      p2 * 2^e = p2 / 2^-e = d[-1] / 10^1 + d[-2] / 10^2 + ...
//
// into decimal form, the fraction is repeatedly multiplied by 10 and the digits
// d[-i] are extracted in order:
//
//      (10 * p2) div 2^-e = d[-1]
//      (10 * p2) mod 2^-e = d[-2] / 10^1 + ...
//
// The multiplication by 10 must not overflow. It is sufficient to choose
//
//      10 * p2 < 16 * p2 = 2^4 * p2 <= 2^64.
//
// Since p2 = f mod 2^-e < 2^-e,
//
//      -e <= 60   or   e >= -60 := alpha

constexpr int kAlpha = -60;
constexpr int kGamma = -32;

struct cached_power // c = f * 2^e ~= 10^k
{
    std::uint64_t f;
    int e;
    int k;
};

/*!
For a normalized diyfp w = f * 2^e, this function returns a (normalized) cached
power-of-ten c = f_c * 2^e_c, such that the exponent of the product w * c
satisfies (Definition 3.2 from [1])

     alpha <= e_c + e + q <= gamma.
*/
inline cached_power get_cached_power_for_binary_exponent(int e)
{
    // Now
    //
    //      alpha <= e_c + e + q <= gamma                                    (1)
    //      ==> f_c * 2^alpha <= c * 2^e * 2^q
    //
    // and since the c's are normalized, 2^(q-1) <= f_c,
    //
    //      ==> 2^(q - 1 + alpha) <= c * 2^(e + q)
    //      ==> 2^(alpha - e - 1) <= c
    //
    // If c were an exact power of ten, i.e. c = 10^k, one may determine k as
    //
    //      k = ceil( log_10( 2^(alpha - e - 1) ) )
    //        = ceil( (alpha - e - 1) * log_10(2) )
    //
    // From the paper:
    // "In theory the result of the procedure could be wrong since c is rounded,
    //  and the computation itself is approximated [...]. In practice, however,
    //  this simple function is sufficient."
    //
    // For IEEE double precision floating-point numbers converted into
    // normalized diyfp's w = f * 2^e, with q = 64,
    //
    //      e >= -1022      (min IEEE exponent)
    //           -52        (p - 1)
    //           -52        (p - 1, possibly normalize denormal IEEE numbers)
    //           -11        (normalize the diyfp)
    //         = -1137
    //
    // and
    //
    //      e <= +1023      (max IEEE exponent)
    //           -52        (p - 1)
    //           -11        (normalize the diyfp)
    //         = 960
    //
    // This binary exponent range [-1137,960] results in a decimal exponent
    // range [-307,324]. One does not need to store a cached power for each
    // k in this range. For each such k it suffices to find a cached power
    // such that the exponent of the product lies in [alpha,gamma].
    // This implies that the difference of the decimal exponents of adjacent
    // table entries must be less than or equal to
    //
    //      floor( (gamma - alpha) * log_10(2) ) = 8.
    //
    // (A smaller distance gamma-alpha would require a larger table.)

    // NB:
    // Actually this function returns c, such that -60 <= e_c + e + 64 <= -34.

    constexpr int kCachedPowersMinDecExp = -300;
    constexpr int kCachedPowersDecStep = 8;

    static constexpr std::array<cached_power, 79> kCachedPowers =
    {
        {
            { 0xAB70FE17C79AC6CA, -1060, -300 },
            { 0xFF77B1FCBEBCDC4F, -1034, -292 },
            { 0xBE5691EF416BD60C, -1007, -284 },
            { 0x8DD01FAD907FFC3C,  -980, -276 },
            { 0xD3515C2831559A83,  -954, -268 },
            { 0x9D71AC8FADA6C9B5,  -927, -260 },
            { 0xEA9C227723EE8BCB,  -901, -252 },
            { 0xAECC49914078536D,  -874, -244 },
            { 0x823C12795DB6CE57,  -847, -236 },
            { 0xC21094364DFB5637,  -821, -228 },
            { 0x9096EA6F3848984F,  -794, -220 },
            { 0xD77485CB25823AC7,  -768, -212 },
            { 0xA086CFCD97BF97F4,  -741, -204 },
            { 0xEF340A98172AACE5,  -715, -196 },
            { 0xB23867FB2A35B28E,  -688, -188 },
            { 0x84C8D4DFD2C63F3B,  -661, -180 },
            { 0xC5DD44271AD3CDBA,  -635, -172 },
            { 0x936B9FCEBB25C996,  -608, -164 },
            { 0xDBAC6C247D62A584,  -582, -156 },
            { 0xA3AB66580D5FDAF6,  -555, -148 },
            { 0xF3E2F893DEC3F126,  -529, -140 },
            { 0xB5B5ADA8AAFF80B8,  -502, -132 },
            { 0x87625F056C7C4A8B,  -475, -124 },
            { 0xC9BCFF6034C13053,  -449, -116 },
            { 0x964E858C91BA2655,  -422, -108 },
            { 0xDFF9772470297EBD,  -396, -100 },
            { 0xA6DFBD9FB8E5B88F,  -369,  -92 },
            { 0xF8A95FCF88747D94,  -343,  -84 },
            { 0xB94470938FA89BCF,  -316,  -76 },
            { 0x8A08F0F8BF0F156B,  -289,  -68 },
            { 0xCDB02555653131B6,  -263,  -60 },
            { 0x993FE2C6D07B7FAC,  -236,  -52 },
            { 0xE45C10C42A2B3B06,  -210,  -44 },
            { 0xAA242499697392D3,  -183,  -36 },
            { 0xFD87B5F28300CA0E,  -157,  -28 },
            { 0xBCE5086492111AEB,  -130,  -20 },
            { 0x8CBCCC096F5088CC,  -103,  -12 },
            { 0xD1B71758E219652C,   -77,   -4 },
            { 0x9C40000000000000,   -50,    4 },
            { 0xE8D4A51000000000,   -24,   12 },
            { 0xAD78EBC5AC620000,     3,   20 },
            { 0x813F3978F8940984,    30,   28 },
            { 0xC097CE7BC90715B3,    56,   36 },
            { 0x8F7E32CE7BEA5C70,    83,   44 },
            { 0xD5D238A4ABE98068,   109,   52 },
            { 0x9F4F2726179A2245,   136,   60 },
            { 0xED63A231D4C4FB27,   162,   68 },
            { 0xB0DE65388CC8ADA8,   189,   76 },
            { 0x83C7088E1AAB65DB,   216,   84 },
            { 0xC45D1DF942711D9A,   242,   92 },
            { 0x924D692CA61BE758,   269,  100 },
            { 0xDA01EE641A708DEA,   295,  108 },
            { 0xA26DA3999AEF774A,   322,  116 },
            { 0xF209787BB47D6B85,   348,  124 },
            { 0xB454E4A179DD1877,   375,  132 },
            { 0x865B86925B9BC5C2,   402,  140 },
            { 0xC83553C5C8965D3D,   428,  148 },
            { 0x952AB45CFA97A0B3,   455,  156 },
            { 0xDE469FBD99A05FE3,   481,  164 },
            { 0xA59BC234DB398C25,   508,  172 },
            { 0xF6C69A72A3989F5C,   534,  180 },
            { 0xB7DCBF5354E9BECE,   561,  188 },
            { 0x88FCF317F22241E2,   588,  196 },
            { 0xCC20CE9BD35C78A5,   614,  204 },
            { 0x98165AF37B2153DF,   641,  212 },
            { 0xE2A0B5DC971F303A,   667,  220 },
            { 0xA8D9D1535CE3B396,   694,  228 },
            { 0xFB9B7CD9A4A7443C,   720,  236 },
            { 0xBB764C4CA7A44410,   747,  244 },
            { 0x8BAB8EEFB6409C1A,   774,  252 },
            { 0xD01FEF10A657842C,   800,  260 },
            { 0x9B10A4E5E9913129,   827,  268 },
            { 0xE7109BFBA19C0C9D,   853,  276 },
            { 0xAC2820D9623BF429,   880,  284 },
            { 0x80444B5E7AA7CF85,   907,  292 },
            { 0xBF21E44003ACDD2D,   933,  300 },
            { 0x8E679C2F5E44FF8F,   960,  308 },
            { 0xD433179D9C8CB841,   986,  316 },
            { 0x9E19DB92B4E31BA9,  1013,  324 },
        }
    };

    // This computation gives exactly the same results for k as
    //      k = ceil((kAlpha - e - 1) * 0.30102999566398114)
    // for |e| <= 1500, but doesn't require floating-point operations.
    // NB: log_10(2) ~= 78913 / 2^18
    JSON_ASSERT(e >= -1500);
    JSON_ASSERT(e <=  1500);
    const int f = kAlpha - e - 1;
    const int k = ((f * 78913) / (1 << 18)) + static_cast<int>(f > 0);

    const int index = (-kCachedPowersMinDecExp + k + (kCachedPowersDecStep - 1)) / kCachedPowersDecStep;
    JSON_ASSERT(index >= 0);
    JSON_ASSERT(static_cast<std::size_t>(index) < kCachedPowers.size());

    const cached_power cached = kCachedPowers[static_cast<std::size_t>(index)];
    JSON_ASSERT(kAlpha <= cached.e + e + 64);
    JSON_ASSERT(kGamma >= cached.e + e + 64);

    return cached;
}

/*!
For n != 0, returns k, such that pow10 := 10^(k-1) <= n < 10^k.
For n == 0, returns 1 and sets pow10 := 1.
*/
inline int find_largest_pow10(const std::uint32_t n, std::uint32_t& pow10)
{
    // LCOV_EXCL_START
    if (n >= 1000000000)
    {
        pow10 = 1000000000;
        return 10;
    }
    // LCOV_EXCL_STOP
    if (n >= 100000000)
    {
        pow10 = 100000000;
        return  9;
    }
    if (n >= 10000000)
    {
        pow10 = 10000000;
        return  8;
    }
    if (n >= 1000000)
    {
        pow10 = 1000000;
        return  7;
    }
    if (n >= 100000)
    {
        pow10 = 100000;
        return  6;
    }
    if (n >= 10000)
    {
        pow10 = 10000;
        return  5;
    }
    if (n >= 1000)
    {
        pow10 = 1000;
        return  4;
    }
    if (n >= 100)
    {
        pow10 = 100;
        return  3;
    }
    if (n >= 10)
    {
        pow10 = 10;
        return  2;
    }

    pow10 = 1;
    return 1;
}

inline void grisu2_round(char* buf, int len, std::uint64_t dist, std::uint64_t delta,
                         std::uint64_t rest, std::uint64_t ten_k)
{
    JSON_ASSERT(len >= 1);
    JSON_ASSERT(dist <= delta);
    JSON_ASSERT(rest <= delta);
    JSON_ASSERT(ten_k > 0);

    //               <--------------------------- delta ---->
    //                                  <---- dist --------->
    // --------------[------------------+-------------------]--------------
    //               M-                 w                   M+
    //
    //                                  ten_k
    //                                <------>
    //                                       <---- rest ---->
    // --------------[------------------+----+--------------]--------------
    //                                  w    V
    //                                       = buf * 10^k
    //
    // ten_k represents a unit-in-the-last-place in the decimal representation
    // stored in buf.
    // Decrement buf by ten_k while this takes buf closer to w.

    // The tests are written in this order to avoid overflow in unsigned
    // integer arithmetic.

    while (rest < dist
            && delta - rest >= ten_k
            && (rest + ten_k < dist || dist - rest > rest + ten_k - dist))
    {
        JSON_ASSERT(buf[len - 1] != '0');
        buf[len - 1]--;
        rest += ten_k;
    }
}

/*!
Generates V = buffer * 10^decimal_exponent, such that M- <= V <= M+.
M- and M+ must be normalized and share the same exponent -60 <= e <= -32.
*/
inline void grisu2_digit_gen(char* buffer, int& length, int& decimal_exponent,
                             diyfp M_minus, diyfp w, diyfp M_plus)
{
    static_assert(kAlpha >= -60, "internal error");
    static_assert(kGamma <= -32, "internal error");

    // Generates the digits (and the exponent) of a decimal floating-point
    // number V = buffer * 10^decimal_exponent in the range [M-, M+]. The diyfp's
    // w, M- and M+ share the same exponent e, which satisfies alpha <= e <= gamma.
    //
    //               <--------------------------- delta ---->
    //                                  <---- dist --------->
    // --------------[------------------+-------------------]--------------
    //               M-                 w                   M+
    //
    // Grisu2 generates the digits of M+ from left to right and stops as soon as
    // V is in [M-,M+].

    JSON_ASSERT(M_plus.e >= kAlpha);
    JSON_ASSERT(M_plus.e <= kGamma);

    std::uint64_t delta = diyfp::sub(M_plus, M_minus).f; // (significand of (M+ - M-), implicit exponent is e)
    std::uint64_t dist  = diyfp::sub(M_plus, w      ).f; // (significand of (M+ - w ), implicit exponent is e)

    // Split M+ = f * 2^e into two parts p1 and p2 (note: e < 0):
    //
    //      M+ = f * 2^e
    //         = ((f div 2^-e) * 2^-e + (f mod 2^-e)) * 2^e
    //         = ((p1        ) * 2^-e + (p2        )) * 2^e
    //         = p1 + p2 * 2^e

    const diyfp one(std::uint64_t{1} << -M_plus.e, M_plus.e);

    auto p1 = static_cast<std::uint32_t>(M_plus.f >> -one.e); // p1 = f div 2^-e (Since -e >= 32, p1 fits into a 32-bit int.)
    std::uint64_t p2 = M_plus.f & (one.f - 1);                    // p2 = f mod 2^-e

    // 1)
    //
    // Generate the digits of the integral part p1 = d[n-1]...d[1]d[0]

    JSON_ASSERT(p1 > 0);

    std::uint32_t pow10{};
    const int k = find_largest_pow10(p1, pow10);

    //      10^(k-1) <= p1 < 10^k, pow10 = 10^(k-1)
    //
    //      p1 = (p1 div 10^(k-1)) * 10^(k-1) + (p1 mod 10^(k-1))
    //         = (d[k-1]         ) * 10^(k-1) + (p1 mod 10^(k-1))
    //
    //      M+ = p1                                             + p2 * 2^e
    //         = d[k-1] * 10^(k-1) + (p1 mod 10^(k-1))          + p2 * 2^e
    //         = d[k-1] * 10^(k-1) + ((p1 mod 10^(k-1)) * 2^-e + p2) * 2^e
    //         = d[k-1] * 10^(k-1) + (                         rest) * 2^e
    //
    // Now generate the digits d[n] of p1 from left to right (n = k-1,...,0)
    //
    //      p1 = d[k-1]...d[n] * 10^n + d[n-1]...d[0]
    //
    // but stop as soon as
    //
    //      rest * 2^e = (d[n-1]...d[0] * 2^-e + p2) * 2^e <= delta * 2^e

    int n = k;
    while (n > 0)
    {
        // Invariants:
        //      M+ = buffer * 10^n + (p1 + p2 * 2^e)    (buffer = 0 for n = k)
        //      pow10 = 10^(n-1) <= p1 < 10^n
        //
        const std::uint32_t d = p1 / pow10;  // d = p1 div 10^(n-1)
        const std::uint32_t r = p1 % pow10;  // r = p1 mod 10^(n-1)
        //
        //      M+ = buffer * 10^n + (d * 10^(n-1) + r) + p2 * 2^e
        //         = (buffer * 10 + d) * 10^(n-1) + (r + p2 * 2^e)
        //
        JSON_ASSERT(d <= 9);
        buffer[length++] = static_cast<char>('0' + d); // buffer := buffer * 10 + d
        //
        //      M+ = buffer * 10^(n-1) + (r + p2 * 2^e)
        //
        p1 = r;
        n--;
        //
        //      M+ = buffer * 10^n + (p1 + p2 * 2^e)
        //      pow10 = 10^n
        //

        // Now check if enough digits have been generated.
        // Compute
        //
        //      p1 + p2 * 2^e = (p1 * 2^-e + p2) * 2^e = rest * 2^e
        //
        // Note:
        // Since rest and delta share the same exponent e, it suffices to
        // compare the significands.
        const std::uint64_t rest = (std::uint64_t{p1} << -one.e) + p2;
        if (rest <= delta)
        {
            // V = buffer * 10^n, with M- <= V <= M+.

            decimal_exponent += n;

            // We may now just stop. But instead look if the buffer could be
            // decremented to bring V closer to w.
            //
            // pow10 = 10^n is now 1 ulp in the decimal representation V.
            // The rounding procedure works with diyfp's with an implicit
            // exponent of e.
            //
            //      10^n = (10^n * 2^-e) * 2^e = ulp * 2^e
            //
            const std::uint64_t ten_n = std::uint64_t{pow10} << -one.e;
            grisu2_round(buffer, length, dist, delta, rest, ten_n);

            return;
        }

        pow10 /= 10;
        //
        //      pow10 = 10^(n-1) <= p1 < 10^n
        // Invariants restored.
    }

    // 2)
    //
    // The digits of the integral part have been generated:
    //
    //      M+ = d[k-1]...d[1]d[0] + p2 * 2^e
    //         = buffer            + p2 * 2^e
    //
    // Now generate the digits of the fractional part p2 * 2^e.
    //
    // Note:
    // No decimal point is generated: the exponent is adjusted instead.
    //
    // p2 actually represents the fraction
    //
    //      p2 * 2^e
    //          = p2 / 2^-e
    //          = d[-1] / 10^1 + d[-2] / 10^2 + ...
    //
    // Now generate the digits d[-m] of p1 from left to right (m = 1,2,...)
    //
    //      p2 * 2^e = d[-1]d[-2]...d[-m] * 10^-m
    //                      + 10^-m * (d[-m-1] / 10^1 + d[-m-2] / 10^2 + ...)
    //
    // using
    //
    //      10^m * p2 = ((10^m * p2) div 2^-e) * 2^-e + ((10^m * p2) mod 2^-e)
    //                = (                   d) * 2^-e + (                   r)
    //
    // or
    //      10^m * p2 * 2^e = d + r * 2^e
    //
    // i.e.
    //
    //      M+ = buffer + p2 * 2^e
    //         = buffer + 10^-m * (d + r * 2^e)
    //         = (buffer * 10^m + d) * 10^-m + 10^-m * r * 2^e
    //
    // and stop as soon as 10^-m * r * 2^e <= delta * 2^e

    JSON_ASSERT(p2 > delta);

    int m = 0;
    for (;;)
    {
        // Invariant:
        //      M+ = buffer * 10^-m + 10^-m * (d[-m-1] / 10 + d[-m-2] / 10^2 + ...) * 2^e
        //         = buffer * 10^-m + 10^-m * (p2                                 ) * 2^e
        //         = buffer * 10^-m + 10^-m * (1/10 * (10 * p2)                   ) * 2^e
        //         = buffer * 10^-m + 10^-m * (1/10 * ((10*p2 div 2^-e) * 2^-e + (10*p2 mod 2^-e)) * 2^e
        //
        JSON_ASSERT(p2 <= (std::numeric_limits<std::uint64_t>::max)() / 10);
        p2 *= 10;
        const std::uint64_t d = p2 >> -one.e;     // d = (10 * p2) div 2^-e
        const std::uint64_t r = p2 & (one.f - 1); // r = (10 * p2) mod 2^-e
        //
        //      M+ = buffer * 10^-m + 10^-m * (1/10 * (d * 2^-e + r) * 2^e
        //         = buffer * 10^-m + 10^-m * (1/10 * (d + r * 2^e))
        //         = (buffer * 10 + d) * 10^(-m-1) + 10^(-m-1) * r * 2^e
        //
        JSON_ASSERT(d <= 9);
        buffer[length++] = static_cast<char>('0' + d); // buffer := buffer * 10 + d
        //
        //      M+ = buffer * 10^(-m-1) + 10^(-m-1) * r * 2^e
        //
        p2 = r;
        m++;
        //
        //      M+ = buffer * 10^-m + 10^-m * p2 * 2^e
        // Invariant restored.

        // Check if enough digits have been generated.
        //
        //      10^-m * p2 * 2^e <= delta * 2^e
        //              p2 * 2^e <= 10^m * delta * 2^e
        //                    p2 <= 10^m * delta
        delta *= 10;
        dist  *= 10;
        if (p2 <= delta)
        {
            break;
        }
    }

    // V = buffer * 10^-m, with M- <= V <= M+.

    decimal_exponent -= m;

    // 1 ulp in the decimal representation is now 10^-m.
    // Since delta and dist are now scaled by 10^m, we need to do the
    // same with ulp in order to keep the units in sync.
    //
    //      10^m * 10^-m = 1 = 2^-e * 2^e = ten_m * 2^e
    //
    const std::uint64_t ten_m = one.f;
    grisu2_round(buffer, length, dist, delta, p2, ten_m);

    // By construction this algorithm generates the shortest possible decimal
    // number (Loitsch, Theorem 6.2) which rounds back to w.
    // For an input number of precision p, at least
    //
    //      N = 1 + ceil(p * log_10(2))
    //
    // decimal digits are sufficient to identify all binary floating-point
    // numbers (Matula, "In-and-Out conversions").
    // This implies that the algorithm does not produce more than N decimal
    // digits.
    //
    //      N = 17 for p = 53 (IEEE double precision)
    //      N = 9  for p = 24 (IEEE single precision)
}

/*!
v = buf * 10^decimal_exponent
len is the length of the buffer (number of decimal digits)
The buffer must be large enough, i.e. >= max_digits10.
*/
JSON_HEDLEY_NON_NULL(1)
inline void grisu2(char* buf, int& len, int& decimal_exponent,
                   diyfp m_minus, diyfp v, diyfp m_plus)
{
    JSON_ASSERT(m_plus.e == m_minus.e);
    JSON_ASSERT(m_plus.e == v.e);

    //  --------(-----------------------+-----------------------)--------    (A)
    //          m-                      v                       m+
    //
    //  --------------------(-----------+-----------------------)--------    (B)
    //                      m-          v                       m+
    //
    // First scale v (and m- and m+) such that the exponent is in the range
    // [alpha, gamma].

    const cached_power cached = get_cached_power_for_binary_exponent(m_plus.e);

    const diyfp c_minus_k(cached.f, cached.e); // = c ~= 10^-k

    // The exponent of the products is = v.e + c_minus_k.e + q and is in the range [alpha,gamma]
    const diyfp w       = diyfp::mul(v,       c_minus_k);
    const diyfp w_minus = diyfp::mul(m_minus, c_minus_k);
    const diyfp w_plus  = diyfp::mul(m_plus,  c_minus_k);

    //  ----(---+---)---------------(---+---)---------------(---+---)----
    //          w-                      w                       w+
    //          = c*m-                  = c*v                   = c*m+
    //
    // diyfp::mul rounds its result and c_minus_k is approximated too. w, w- and
    // w+ are now off by a small amount.
    // In fact:
    //
    //      w - v * 10^k < 1 ulp
    //
    // To account for this inaccuracy, add resp. subtract 1 ulp.
    //
    //  --------+---[---------------(---+---)---------------]---+--------
    //          w-  M-                  w                   M+  w+
    //
    // Now any number in [M-, M+] (bounds included) will round to w when input,
    // regardless of how the input rounding algorithm breaks ties.
    //
    // And digit_gen generates the shortest possible such number in [M-, M+].
    // Note that this does not mean that Grisu2 always generates the shortest
    // possible number in the interval (m-, m+).
    const diyfp M_minus(w_minus.f + 1, w_minus.e);
    const diyfp M_plus (w_plus.f  - 1, w_plus.e );

    decimal_exponent = -cached.k; // = -(-k) = k

    grisu2_digit_gen(buf, len, decimal_exponent, M_minus, w, M_plus);
}

/*!
v = buf * 10^decimal_exponent
len is the length of the buffer (number of decimal digits)
The buffer must be large enough, i.e. >= max_digits10.
*/
template<typename FloatType>
JSON_HEDLEY_NON_NULL(1)
void grisu2(char* buf, int& len, int& decimal_exponent, FloatType value)
{
    static_assert(diyfp::kPrecision >= std::numeric_limits<FloatType>::digits + 3,
                  "internal error: not enough precision");

    JSON_ASSERT(std::isfinite(value));
    JSON_ASSERT(value > 0);

    // If the neighbors (and boundaries) of 'value' are always computed for double-precision
    // numbers, all float's can be recovered using strtod (and strtof). However, the resulting
    // decimal representations are not exactly "short".
    //
    // The documentation for 'std::to_chars' (https://en.cppreference.com/w/cpp/utility/to_chars)
    // says "value is converted to a string as if by std::sprintf in the default ("C") locale"
    // and since sprintf promotes floats to doubles, I think this is exactly what 'std::to_chars'
    // does.
    // On the other hand, the documentation for 'std::to_chars' requires that "parsing the
    // representation using the corresponding std::from_chars function recovers value exactly". That
    // indicates that single precision floating-point numbers should be recovered using
    // 'std::strtof'.
    //
    // NB: If the neighbors are computed for single-precision numbers, there is a single float
    //     (7.0385307e-26f) which can't be recovered using strtod. The resulting double precision
    //     value is off by 1 ulp.
#if 0 // NOLINT(readability-avoid-unconditional-preprocessor-if)
    const boundaries w = compute_boundaries(static_cast<double>(value));
#else
    const boundaries w = compute_boundaries(value);
#endif

    grisu2(buf, len, decimal_exponent, w.minus, w.w, w.plus);
}

/*!
@brief appends a decimal representation of e to buf
@return a pointer to the element following the exponent.
@pre -1000 < e < 1000
*/
JSON_HEDLEY_NON_NULL(1)
JSON_HEDLEY_RETURNS_NON_NULL
inline char* append_exponent(char* buf, int e)
{
    JSON_ASSERT(e > -1000);
    JSON_ASSERT(e <  1000);

    if (e < 0)
    {
        e = -e;
        *buf++ = '-';
    }
    else
    {
        *buf++ = '+';
    }

    auto k = static_cast<std::uint32_t>(e);
    if (k < 10)
    {
        // Always print at least two digits in the exponent.
        // This is for compatibility with printf("%g").
        *buf++ = '0';
        *buf++ = static_cast<char>('0' + k);
    }
    else if (k < 100)
    {
        *buf++ = static_cast<char>('0' + (k / 10));
        k %= 10;
        *buf++ = static_cast<char>('0' + k);
    }
    else
    {
        *buf++ = static_cast<char>('0' + (k / 100));
        k %= 100;
        *buf++ = static_cast<char>('0' + (k / 10));
        k %= 10;
        *buf++ = static_cast<char>('0' + k);
    }

    return buf;
}

/*!
@brief prettify v = buf * 10^decimal_exponent

If v is in the range [10^min_exp, 10^max_exp) it will be printed in fixed-point
notation. Otherwise it will be printed in exponential notation.

@pre min_exp < 0
@pre max_exp > 0
*/
JSON_HEDLEY_NON_NULL(1)
JSON_HEDLEY_RETURNS_NON_NULL
inline char* format_buffer(char* buf, int len, int decimal_exponent,
                           int min_exp, int max_exp)
{
    JSON_ASSERT(min_exp < 0);
    JSON_ASSERT(max_exp > 0);

    const int k = len;
    const int n = len + decimal_exponent;

    // v = buf * 10^(n-k)
    // k is the length of the buffer (number of decimal digits)
    // n is the position of the decimal point relative to the start of the buffer.

    if (k <= n && n <= max_exp)
    {
        // digits[000]
        // len <= max_exp + 2

        std::memset(buf + k, '0', static_cast<size_t>(n) - static_cast<size_t>(k));
        // Make it look like a floating-point number (#362, #378)
        buf[n + 0] = '.';
        buf[n + 1] = '0';
        return buf + (static_cast<size_t>(n) + 2);
    }

    if (0 < n && n <= max_exp)
    {
        // dig.its
        // len <= max_digits10 + 1

        JSON_ASSERT(k > n);

        std::memmove(buf + (static_cast<size_t>(n) + 1), buf + n, static_cast<size_t>(k) - static_cast<size_t>(n));
        buf[n] = '.';
        return buf + (static_cast<size_t>(k) + 1U);
    }

    if (min_exp < n && n <= 0)
    {
        // 0.[000]digits
        // len <= 2 + (-min_exp - 1) + max_digits10

        std::memmove(buf + (2 + static_cast<size_t>(-n)), buf, static_cast<size_t>(k));
        buf[0] = '0';
        buf[1] = '.';
        std::memset(buf + 2, '0', static_cast<size_t>(-n));
        return buf + (2U + static_cast<size_t>(-n) + static_cast<size_t>(k));
    }

    if (k == 1)
    {
        // dE+123
        // len <= 1 + 5

        buf += 1;
    }
    else
    {
        // d.igitsE+123
        // len <= max_digits10 + 1 + 5

        std::memmove(buf + 2, buf + 1, static_cast<size_t>(k) - 1);
        buf[1] = '.';
        buf += 1 + static_cast<size_t>(k);
    }

    *buf++ = 'e';
    return append_exponent(buf, n - 1);
}

}  // namespace dtoa_impl

/*!
@brief generates a decimal representation of the floating-point number value in [first, last).

The format of the resulting decimal representation is similar to printf's %g
format. Returns an iterator pointing past-the-end of the decimal representation.

@note The input number must be finite, i.e. NaN's and Inf's are not supported.
@note The buffer must be large enough.
@note The result is NOT null-terminated.
*/
template<typename FloatType>
JSON_HEDLEY_NON_NULL(1, 2)
JSON_HEDLEY_RETURNS_NON_NULL
char* to_chars(char* first, const char* last, FloatType value)
{
    static_cast<void>(last); // maybe unused - fix warning
    JSON_ASSERT(std::isfinite(value));

    // Use signbit(value) instead of (value < 0) since signbit works for -0.
    if (std::signbit(value))
    {
        value = -value;
        *first++ = '-';
    }

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"
#endif
    if (value == 0) // +-0
    {
        *first++ = '0';
        // Make it look like a floating-point number (#362, #378)
        *first++ = '.';
        *first++ = '0';
        return first;
    }
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

    JSON_ASSERT(last - first >= std::numeric_limits<FloatType>::max_digits10);

    // Compute v = buffer * 10^decimal_exponent.
    // The decimal digits are stored in the buffer, which needs to be interpreted
    // as an unsigned decimal integer.
    // len is the length of the buffer, i.e. the number of decimal digits.
    int len = 0;
    int decimal_exponent = 0;
    dtoa_impl::grisu2(first, len, decimal_exponent, value);

    JSON_ASSERT(len <= std::numeric_limits<FloatType>::max_digits10);

    // Format the buffer like printf("%.*g", prec, value)
    constexpr int kMinExp = -4;
    // Use digits10 here to increase compatibility with version 2.
    constexpr int kMaxExp = std::numeric_limits<FloatType>::digits10;

    JSON_ASSERT(last - first >= kMaxExp + 2);
    JSON_ASSERT(last - first >= 2 + (-kMinExp - 1) + std::numeric_limits<FloatType>::max_digits10);
    JSON_ASSERT(last - first >= std::numeric_limits<FloatType>::max_digits10 + 6);

    return dtoa_impl::format_buffer(first, len, decimal_exponent, kMinExp, kMaxExp);
}

}  // namespace detail
NLOHMANN_JSON_NAMESPACE_END

// #include <nlohmann/detail/exceptions.hpp>

// #include <nlohmann/detail/macro_scope.hpp>

// #include <nlohmann/detail/meta/cpp_future.hpp>

// #include <nlohmann/detail/output/binary_writer.hpp>

// #include <nlohmann/detail/output/output_adapters.hpp>

// #include <nlohmann/detail/string_concat.hpp>

// #include <nlohmann/detail/value_t.hpp>


NLOHMANN_JSON_NAMESPACE_BEGIN
namespace detail
{

///////////////////
// serialization //
///////////////////

/// how to treat decoding errors
enum class error_handler_t
{
    strict,  ///< throw a type_error exception in case of invalid UTF-8
    replace, ///< replace invalid UTF-8 sequences with U+FFFD
    ignore   ///< ignore invalid UTF-8 sequences
};

template<typename BasicJsonType>
class serializer
{
    using string_t = typename BasicJsonType::string_t;
    using number_float_t = typename BasicJsonType::number_float_t;
    using number_integer_t = typename BasicJsonType::number_integer_t;
    using number_unsigned_t = typename BasicJsonType::number_unsigned_t;
    using binary_char_t = typename BasicJsonType::binary_t::value_type;
    static constexpr std::uint8_t UTF8_ACCEPT = 0;
    static constexpr std::uint8_t UTF8_REJECT = 1;

  public:
    /*!
    @param[in] s  output stream to serialize to
    @param[in] ichar  indentation character to use
    @param[in] error_handler_  how to react on decoding errors
    */
    serializer(output_adapter_t<char> s, const char ichar,
               error_handler_t error_handler_ = error_handler_t::strict)
        : o(std::move(s))
        , loc(std::localeconv())
        , thousands_sep(loc->thousands_sep == nullptr ? '\0' : std::char_traits<char>::to_char_type(* (loc->thousands_sep)))
        , decimal_point(loc->decimal_point == nullptr ? '\0' : std::char_traits<char>::to_char_type(* (loc->decimal_point)))
        , indent_char(ichar)
        , indent_string(512, indent_char)
        , error_handler(error_handler_)
    {}

    // delete because of pointer members
    serializer(const serializer&) = delete;
    serializer& operator=(const serializer&) = delete;
    serializer(serializer&&) = delete;
    serializer& operator=(serializer&&) = delete;
    ~serializer() = default;

    /*!
    @brief internal implementation of the serialization function

    This function is called by the public member function dump and organizes
    the serialization internally. The indentation level is propagated as
    additional parameter. In case of arrays and objects, the function is
    called recursively.

    - strings and object keys are escaped using `escape_string()`
    - integer numbers are converted implicitly via `operator<<`
    - floating-point numbers are converted to a string using `"%g"` format
    - binary values are serialized as objects containing the subtype and the
      byte array

    @param[in] val               value to serialize
    @param[in] pretty_print      whether the output shall be pretty-printed
    @param[in] ensure_ascii If @a ensure_ascii is true, all non-ASCII characters
    in the output are escaped with `\uXXXX` sequences, and the result consists
    of ASCII characters only.
    @param[in] indent_step       the indent level
    @param[in] current_indent    the current indent level (only used internally)
    */
    void dump(const BasicJsonType& val,
              const bool pretty_print,
              const bool ensure_ascii,
              const unsigned int indent_step,
              const unsigned int current_indent = 0)
    {
        switch (val.m_data.m_type)
        {
            case value_t::object:
            {
                if (val.m_data.m_value.object->empty())
                {
                    o->write_characters("{}", 2);
                    return;
                }

                if (pretty_print)
                {
                    o->write_characters("{\n", 2);

                    // variable to hold indentation for recursive calls
                    const auto new_indent = current_indent + indent_step;
                    if (JSON_HEDLEY_UNLIKELY(indent_string.size() < new_indent))
                    {
                        indent_string.resize(indent_string.size() * 2, ' ');
                    }

                    // first n-1 elements
                    auto i = val.m_data.m_value.object->cbegin();
                    for (std::size_t cnt = 0; cnt < val.m_data.m_value.object->size() - 1; ++cnt, ++i)
                    {
                        o->write_characters(indent_string.c_str(), new_indent);
                        o->write_character('\"');
                        dump_escaped(i->first, ensure_ascii);
                        o->write_characters("\": ", 3);
                        dump(i->second, true, ensure_ascii, indent_step, new_indent);
                        o->write_characters(",\n", 2);
                    }

                    // last element
                    JSON_ASSERT(i != val.m_data.m_value.object->cend());
                    JSON_ASSERT(std::next(i) == val.m_data.m_value.object->cend());
                    o->write_characters(indent_string.c_str(), new_indent);
                    o->write_character('\"');
                    dump_escaped(i->first, ensure_ascii);
                    o->write_characters("\": ", 3);
                    dump(i->second, true, ensure_ascii, indent_step, new_indent);

                    o->write_character('\n');
                    o->write_characters(indent_string.c_str(), current_indent);
                    o->write_character('}');
                }
                else
                {
                    o->write_character('{');

                    // first n-1 elements
                    auto i = val.m_data.m_value.object->cbegin();
                    for (std::size_t cnt = 0; cnt < val.m_data.m_value.object->size() - 1; ++cnt, ++i)
                    {
                        o->write_character('\"');
                        dump_escaped(i->first, ensure_ascii);
                        o->write_characters("\":", 2);
                        dump(i->second, false, ensure_ascii, indent_step, current_indent);
                        o->write_character(',');
                    }

                    // last element
                    JSON_ASSERT(i != val.m_data.m_value.object->cend());
                    JSON_ASSERT(std::next(i) == val.m_data.m_value.object->cend());
                    o->write_character('\"');
                    dump_escaped(i->first, ensure_ascii);
                    o->write_characters("\":", 2);
                    dump(i->second, false, ensure_ascii, indent_step, current_indent);

                    o->write_character('}');
                }

                return;
            }

            case value_t::array:
            {
                if (val.m_data.m_value.array->empty())
                {
                    o->write_characters("[]", 2);
                    return;
                }

                if (pretty_print)
                {
                    o->write_characters("[\n", 2);

                    // variable to hold indentation for recursive calls
                    const auto new_indent = current_indent + indent_step;
                    if (JSON_HEDLEY_UNLIKELY(indent_string.size() < new_indent))
                    {
                        indent_string.resize(indent_string.size() * 2, ' ');
                    }

                    // first n-1 elements
                    for (auto i = val.m_data.m_value.array->cbegin();
                            i != val.m_data.m_value.array->cend() - 1; ++i)
                    {
                        o->write_characters(indent_string.c_str(), new_indent);
                        dump(*i, true, ensure_ascii, indent_step, new_indent);
                        o->write_characters(",\n", 2);
                    }

                    // last element
                    JSON_ASSERT(!val.m_data.m_value.array->empty());
                    o->write_characters(indent_string.c_str(), new_indent);
                    dump(val.m_data.m_value.array->back(), true, ensure_ascii, indent_step, new_indent);

                    o->write_character('\n');
                    o->write_characters(indent_string.c_str(), current_indent);
                    o->write_character(']');
                }
                else
                {
                    o->write_character('[');

                    // first n-1 elements
                    for (auto i = val.m_data.m_value.array->cbegin();
                            i != val.m_data.m_value.array->cend() - 1; ++i)
                    {
                        dump(*i, false, ensure_ascii, indent_step, current_indent);
                        o->write_character(',');
                    }

                    // last element
                    JSON_ASSERT(!val.m_data.m_value.array->empty());
                    dump(val.m_data.m_value.array->back(), false, ensure_ascii, indent_step, current_indent);

                    o->write_character(']');
                }

                return;
            }

            case value_t::string:
            {
                o->write_character('\"');
                dump_escaped(*val.m_data.m_value.string, ensure_ascii);
                o->write_character('\"');
                return;
            }

            case value_t::binary:
            {
                if (pretty_print)
                {
                    o->write_characters("{\n", 2);

                    // variable to hold indentation for recursive calls
                    const auto new_indent = current_indent + indent_step;
                    if (JSON_HEDLEY_UNLIKELY(indent_string.size() < new_indent))
                    {
                        indent_string.resize(indent_string.size() * 2, ' ');
                    }

                    o->write_characters(indent_string.c_str(), new_indent);

                    o->write_characters("\"bytes\": [", 10);

                    if (!val.m_data.m_value.binary->empty())
                    {
                        for (auto i = val.m_data.m_value.binary->cbegin();
                                i != val.m_data.m_value.binary->cend() - 1; ++i)
                        {
                            dump_integer(*i);
                            o->write_characters(", ", 2);
                        }
                        dump_integer(val.m_data.m_value.binary->back());
                    }

                    o->write_characters("],\n", 3);
                    o->write_characters(indent_string.c_str(), new_indent);

                    o->write_characters("\"subtype\": ", 11);
                    if (val.m_data.m_value.binary->has_subtype())
                    {
                        dump_integer(val.m_data.m_value.binary->subtype());
                    }
                    else
                    {
                        o->write_characters("null", 4);
                    }
                    o->write_character('\n');
                    o->write_characters(indent_string.c_str(), current_indent);
                    o->write_character('}');
                }
                else
                {
                    o->write_characters("{\"bytes\":[", 10);

                    if (!val.m_data.m_value.binary->empty())
                    {
                        for (auto i = val.m_data.m_value.binary->cbegin();
                                i != val.m_data.m_value.binary->cend() - 1; ++i)
                        {
                            dump_integer(*i);
                            o->write_character(',');
                        }
                        dump_integer(val.m_data.m_value.binary->back());
                    }

                    o->write_characters("],\"subtype\":", 12);
                    if (val.m_data.m_value.binary->has_subtype())
                    {
                        dump_integer(val.m_data.m_value.binary->subtype());
                        o->write_character('}');
                    }
                    else
                    {
                        o->write_characters("null}", 5);
                    }
                }
                return;
            }

            case value_t::boolean:
            {
                if (val.m_data.m_value.boolean)
                {
                    o->write_characters("true", 4);
                }
                else
                {
                    o->write_characters("false", 5);
                }
                return;
            }

            case value_t::number_integer:
            {
                dump_integer(val.m_data.m_value.number_integer);
                return;
            }

            case value_t::number_unsigned:
            {
                dump_integer(val.m_data.m_value.number_unsigned);
                return;
            }

            case value_t::number_float:
            {
                dump_float(val.m_data.m_value.number_float);
                return;
            }

            case value_t::discarded:
            {
                o->write_characters("<discarded>", 11);
                return;
            }

            case value_t::null:
            {
                o->write_characters("null", 4);
                return;
            }

            default:            // LCOV_EXCL_LINE
                JSON_ASSERT(false); // NOLINT(cert-dcl03-c,hicpp-static-assert,misc-static-assert) LCOV_EXCL_LINE
        }
    }

  JSON_PRIVATE_UNLESS_TESTED:
    /*!
    @brief dump escaped string

    Escape a string by replacing certain special characters by a sequence of an
    escape character (backslash) and another character and other control
    characters by a sequence of "\u" followed by a four-digit hex
    representation. The escaped string is written to output stream @a o.

    @param[in] s  the string to escape
    @param[in] ensure_ascii  whether to escape non-ASCII characters with
                             \uXXXX sequences

    @complexity Linear in the length of string @a s.
    */
    void dump_escaped(const string_t& s, const bool ensure_ascii)
    {
        std::uint32_t codepoint{};
        std::uint8_t state = UTF8_ACCEPT;
        std::size_t bytes = 0;  // number of bytes written to string_buffer

        // number of bytes written at the point of the last valid byte
        std::size_t bytes_after_last_accept = 0;
        std::size_t undumped_chars = 0;

        for (std::size_t i = 0; i < s.size(); ++i)
        {
            const auto byte = static_cast<std::uint8_t>(s[i]);

            switch (decode(state, codepoint, byte))
            {
                case UTF8_ACCEPT:  // decode found a new code point
                {
                    switch (codepoint)
                    {
                        case 0x08: // backspace
                        {
                            string_buffer[bytes++] = '\\';
                            string_buffer[bytes++] = 'b';
                            break;
                        }

                        case 0x09: // horizontal tab
                        {
                            string_buffer[bytes++] = '\\';
                            string_buffer[bytes++] = 't';
                            break;
                        }

                        case 0x0A: // newline
                        {
                            string_buffer[bytes++] = '\\';
                            string_buffer[bytes++] = 'n';
                            break;
                        }

                        case 0x0C: // formfeed
                        {
                            string_buffer[bytes++] = '\\';
                            string_buffer[bytes++] = 'f';
                            break;
                        }

                        case 0x0D: // carriage return
                        {
                            string_buffer[bytes++] = '\\';
                            string_buffer[bytes++] = 'r';
                            break;
                        }

                        case 0x22: // quotation mark
                        {
                            string_buffer[bytes++] = '\\';
                            string_buffer[bytes++] = '\"';
                            break;
                        }

                        case 0x5C: // reverse solidus
                        {
                            string_buffer[bytes++] = '\\';
                            string_buffer[bytes++] = '\\';
                            break;
                        }

                        default:
                        {
                            // escape control characters (0x00..0x1F) or, if
                            // ensure_ascii parameter is used, non-ASCII characters
                            if ((codepoint <= 0x1F) || (ensure_ascii && (codepoint >= 0x7F)))
                            {
                                if (codepoint <= 0xFFFF)
                                {
                                    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg,hicpp-vararg)
                                    static_cast<void>((std::snprintf)(string_buffer.data() + bytes, 7, "\\u%04x",
                                                                      static_cast<std::uint16_t>(codepoint)));
                                    bytes += 6;
                                }
                                else
                                {
                                    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg,hicpp-vararg)
                                    static_cast<void>((std::snprintf)(string_buffer.data() + bytes, 13, "\\u%04x\\u%04x",
                                                                      static_cast<std::uint16_t>(0xD7C0u + (codepoint >> 10u)),
                                                                      static_cast<std::uint16_t>(0xDC00u + (codepoint & 0x3FFu))));
                                    bytes += 12;
                                }
                            }
                            else
                            {
                                // copy byte to buffer (all previous bytes
                                // been copied have in default case above)
                                string_buffer[bytes++] = s[i];
                            }
                            break;
                        }
                    }

                    // write buffer and reset index; there must be 13 bytes
                    // left, as this is the maximal number of bytes to be
                    // written ("\uxxxx\uxxxx\0") for one code point
                    if (string_buffer.size() - bytes < 13)
                    {
                        o->write_characters(string_buffer.data(), bytes);
                        bytes = 0;
                    }

                    // remember the byte position of this accept
                    bytes_after_last_accept = bytes;
                    undumped_chars = 0;
                    break;
                }

                case UTF8_REJECT:  // decode found invalid UTF-8 byte
                {
                    switch (error_handler)
                    {
                        case error_handler_t::strict:
                        {
                            JSON_THROW(type_error::create(316, concat("invalid UTF-8 byte at index ", std::to_string(i), ": 0x", hex_bytes(byte | 0)), nullptr));
                        }

                        case error_handler_t::ignore:
                        case error_handler_t::replace:
                        {
                            // in case we saw this character the first time, we
                            // would like to read it again, because the byte
                            // may be OK for itself, but just not OK for the
                            // previous sequence
                            if (undumped_chars > 0)
                            {
                                --i;
                            }

                            // reset length buffer to the last accepted index;
                            // thus removing/ignoring the invalid characters
                            bytes = bytes_after_last_accept;

                            if (error_handler == error_handler_t::replace)
                            {
                                // add a replacement character
                                if (ensure_ascii)
                                {
                                    string_buffer[bytes++] = '\\';
                                    string_buffer[bytes++] = 'u';
                                    string_buffer[bytes++] = 'f';
                                    string_buffer[bytes++] = 'f';
                                    string_buffer[bytes++] = 'f';
                                    string_buffer[bytes++] = 'd';
                                }
                                else
                                {
                                    string_buffer[bytes++] = detail::binary_writer<BasicJsonType, char>::to_char_type('\xEF');
                                    string_buffer[bytes++] = detail::binary_writer<BasicJsonType, char>::to_char_type('\xBF');
                                    string_buffer[bytes++] = detail::binary_writer<BasicJsonType, char>::to_char_type('\xBD');
                                }

                                // write buffer and reset index; there must be 13 bytes
                                // left, as this is the maximal number of bytes to be
                                // written ("\uxxxx\uxxxx\0") for one code point
                                if (string_buffer.size() - bytes < 13)
                                {
                                    o->write_characters(string_buffer.data(), bytes);
                                    bytes = 0;
                                }

                                bytes_after_last_accept = bytes;
                            }

                            undumped_chars = 0;

                            // continue processing the string
                            state = UTF8_ACCEPT;
                            break;
                        }

                        default:            // LCOV_EXCL_LINE
                            JSON_ASSERT(false); // NOLINT(cert-dcl03-c,hicpp-static-assert,misc-static-assert) LCOV_EXCL_LINE
                    }
                    break;
                }

                default:  // decode found yet incomplete multi-byte code point
                {
                    if (!ensure_ascii)
                    {
                        // code point will not be escaped - copy byte to buffer
                        string_buffer[bytes++] = s[i];
                    }
                    ++undumped_chars;
                    break;
                }
            }
        }

        // we finished processing the string
        if (JSON_HEDLEY_LIKELY(state == UTF8_ACCEPT))
        {
            // write buffer
            if (bytes > 0)
            {
                o->write_characters(string_buffer.data(), bytes);
            }
        }
        else
        {
            // we finish reading, but do not accept: string was incomplete
            switch (error_handler)
            {
                case error_handler_t::strict:
                {
                    JSON_THROW(type_error::create(316, concat("incomplete UTF-8 string; last byte: 0x", hex_bytes(static_cast<std::uint8_t>(s.back() | 0))), nullptr));
                }

                case error_handler_t::ignore:
                {
                    // write all accepted bytes
                    o->write_characters(string_buffer.data(), bytes_after_last_accept);
                    break;
                }

                case error_handler_t::replace:
                {
                    // write all accepted bytes
                    o->write_characters(string_buffer.data(), bytes_after_last_accept);
                    // add a replacement character
                    if (ensure_ascii)
                    {
                        o->write_characters("\\ufffd", 6);
                    }
                    else
                    {
                        o->write_characters("\xEF\xBF\xBD", 3);
                    }
                    break;
                }

                default:            // LCOV_EXCL_LINE
                    JSON_ASSERT(false); // NOLINT(cert-dcl03-c,hicpp-static-assert,misc-static-assert) LCOV_EXCL_LINE
            }
        }
    }

  private:
    /*!
    @brief count digits

    Count the number of decimal (base 10) digits for an input unsigned integer.

    @param[in] x  unsigned integer number to count its digits
    @return    number of decimal digits
    */
    unsigned int count_digits(number_unsigned_t x) noexcept
    {
        unsigned int n_digits = 1;
        for (;;)
        {
            if (x < 10)
            {
                return n_digits;
            }
            if (x < 100)
            {
                return n_digits + 1;
            }
            if (x < 1000)
            {
                return n_digits + 2;
            }
            if (x < 10000)
            {
                return n_digits + 3;
            }
            x = x / 10000u;
            n_digits += 4;
        }
    }

    /*!
     * @brief convert a byte to a uppercase hex representation
     * @param[in] byte byte to represent
     * @return representation ("00".."FF")
     */
    static std::string hex_bytes(std::uint8_t byte)
    {
        std::string result = "FF";
        constexpr const char* nibble_to_hex = "0123456789ABCDEF";
        result[0] = nibble_to_hex[byte / 16];
        result[1] = nibble_to_hex[byte % 16];
        return result;
    }

    // templates to avoid warnings about useless casts
    template <typename NumberType, enable_if_t<std::is_signed<NumberType>::value, int> = 0>
    bool is_negative_number(NumberType x)
    {
        return x < 0;
    }

    template < typename NumberType, enable_if_t <std::is_unsigned<NumberType>::value, int > = 0 >
    bool is_negative_number(NumberType /*unused*/)
    {
        return false;
    }

    /*!
    @brief dump an integer

    Dump a given integer to output stream @a o. Works internally with
    @a number_buffer.

    @param[in] x  integer number (signed or unsigned) to dump
    @tparam NumberType either @a number_integer_t or @a number_unsigned_t
    */
    template < typename NumberType, detail::enable_if_t <
                   std::is_integral<NumberType>::value ||
                   std::is_same<NumberType, number_unsigned_t>::value ||
                   std::is_same<NumberType, number_integer_t>::value ||
                   std::is_same<NumberType, binary_char_t>::value,
                   int > = 0 >
    void dump_integer(NumberType x)
    {
        static constexpr std::array<std::array<char, 2>, 100> digits_to_99
        {
            {
                {{'0', '0'}}, {{'0', '1'}}, {{'0', '2'}}, {{'0', '3'}}, {{'0', '4'}}, {{'0', '5'}}, {{'0', '6'}}, {{'0', '7'}}, {{'0', '8'}}, {{'0', '9'}},
                {{'1', '0'}}, {{'1', '1'}}, {{'1', '2'}}, {{'1', '3'}}, {{'1', '4'}}, {{'1', '5'}}, {{'1', '6'}}, {{'1', '7'}}, {{'1', '8'}}, {{'1', '9'}},
                {{'2', '0'}}, {{'2', '1'}}, {{'2', '2'}}, {{'2', '3'}}, {{'2', '4'}}, {{'2', '5'}}, {{'2', '6'}}, {{'2', '7'}}, {{'2', '8'}}, {{'2', '9'}},
                {{'3', '0'}}, {{'3', '1'}}, {{'3', '2'}}, {{'3', '3'}}, {{'3', '4'}}, {{'3', '5'}}, {{'3', '6'}}, {{'3', '7'}}, {{'3', '8'}}, {{'3', '9'}},
                {{'4', '0'}}, {{'4', '1'}}, {{'4', '2'}}, {{'4', '3'}}, {{'4', '4'}}, {{'4', '5'}}, {{'4', '6'}}, {{'4', '7'}}, {{'4', '8'}}, {{'4', '9'}},
                {{'5', '0'}}, {{'5', '1'}}, {{'5', '2'}}, {{'5', '3'}}, {{'5', '4'}}, {{'5', '5'}}, {{'5', '6'}}, {{'5', '7'}}, {{'5', '8'}}, {{'5', '9'}},
                {{'6', '0'}}, {{'6', '1'}}, {{'6', '2'}}, {{'6', '3'}}, {{'6', '4'}}, {{'6', '5'}}, {{'6', '6'}}, {{'6', '7'}}, {{'6', '8'}}, {{'6', '9'}},
                {{'7', '0'}}, {{'7', '1'}}, {{'7', '2'}}, {{'7', '3'}}, {{'7', '4'}}, {{'7', '5'}}, {{'7', '6'}}, {{'7', '7'}}, {{'7', '8'}}, {{'7', '9'}},
                {{'8', '0'}}, {{'8', '1'}}, {{'8', '2'}}, {{'8', '3'}}, {{'8', '4'}}, {{'8', '5'}}, {{'8', '6'}}, {{'8', '7'}}, {{'8', '8'}}, {{'8', '9'}},
                {{'9', '0'}}, {{'9', '1'}}, {{'9', '2'}}, {{'9', '3'}}, {{'9', '4'}}, {{'9', '5'}}, {{'9', '6'}}, {{'9', '7'}}, {{'9', '8'}}, {{'9', '9'}},
            }
        };

        // special case for "0"
        if (x == 0)
        {
            o->write_character('0');
            return;
        }

        // use a pointer to fill the buffer
        auto buffer_ptr = number_buffer.begin(); // NOLINT(llvm-qualified-auto,readability-qualified-auto,cppcoreguidelines-pro-type-vararg,hicpp-vararg)

        number_unsigned_t abs_value;

        unsigned int n_chars{};

        if (is_negative_number(x))
        {
            *buffer_ptr = '-';
            abs_value = remove_sign(static_cast<number_integer_t>(x));

            // account one more byte for the minus sign
            n_chars = 1 + count_digits(abs_value);
        }
        else
        {
            abs_value = static_cast<number_unsigned_t>(x);
            n_chars = count_digits(abs_value);
        }

        // spare 1 byte for '\0'
        JSON_ASSERT(n_chars < number_buffer.size() - 1);

        // jump to the end to generate the string from backward,
        // so we later avoid reversing the result
        buffer_ptr += n_chars;

        // Fast int2ascii implementation inspired by "Fastware" talk by Andrei Alexandrescu
        // See: https://www.youtube.com/watch?v=o4-CwDo2zpg
        while (abs_value >= 100)
        {
            const auto digits_index = static_cast<unsigned>((abs_value % 100));
            abs_value /= 100;
            *(--buffer_ptr) = digits_to_99[digits_index][1];
            *(--buffer_ptr) = digits_to_99[digits_index][0];
        }

        if (abs_value >= 10)
        {
            const auto digits_index = static_cast<unsigned>(abs_value);
            *(--buffer_ptr) = digits_to_99[digits_index][1];
            *(--buffer_ptr) = digits_to_99[digits_index][0];
        }
        else
        {
            *(--buffer_ptr) = static_cast<char>('0' + abs_value);
        }

        o->write_characters(number_buffer.data(), n_chars);
    }

    /*!
    @brief dump a floating-point number

    Dump a given floating-point number to output stream @a o. Works internally
    with @a number_buffer.

    @param[in] x  floating-point number to dump
    */
    void dump_float(number_float_t x)
    {
        // NaN / inf
        if (!std::isfinite(x))
        {
            o->write_characters("null", 4);
            return;
        }

        // If number_float_t is an IEEE-754 single or double precision number,
        // use the Grisu2 algorithm to produce short numbers which are
        // guaranteed to round-trip, using strtof and strtod, resp.
        //
        // NB: The test below works if <long double> == <double>.
        static constexpr bool is_ieee_single_or_double
            = (std::numeric_limits<number_float_t>::is_iec559 && std::numeric_limits<number_float_t>::digits == 24 && std::numeric_limits<number_float_t>::max_exponent == 128) ||
              (std::numeric_limits<number_float_t>::is_iec559 && std::numeric_limits<number_float_t>::digits == 53 && std::numeric_limits<number_float_t>::max_exponent == 1024);

        dump_float(x, std::integral_constant<bool, is_ieee_single_or_double>());
    }

    void dump_float(number_float_t x, std::true_type /*is_ieee_single_or_double*/)
    {
        auto* begin = number_buffer.data();
        auto* end = ::nlohmann::detail::to_chars(begin, begin + number_buffer.size(), x);

        o->write_characters(begin, static_cast<size_t>(end - begin));
    }

    void dump_float(number_float_t x, std::false_type /*is_ieee_single_or_double*/)
    {
        // get number of digits for a float -> text -> float round-trip
        static constexpr auto d = std::numeric_limits<number_float_t>::max_digits10;

        // the actual conversion
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg,hicpp-vararg)
        std::ptrdiff_t len = (std::snprintf)(number_buffer.data(), number_buffer.size(), "%.*g", d, x);

        // negative value indicates an error
        JSON_ASSERT(len > 0);
        // check if buffer was large enough
        JSON_ASSERT(static_cast<std::size_t>(len) < number_buffer.size());

        // erase thousands separator
        if (thousands_sep != '\0')
        {
            // NOLINTNEXTLINE(readability-qualified-auto,llvm-qualified-auto): std::remove returns an iterator, see https://github.com/nlohmann/json/issues/3081
            const auto end = std::remove(number_buffer.begin(), number_buffer.begin() + len, thousands_sep);
            std::fill(end, number_buffer.end(), '\0');
            JSON_ASSERT((end - number_buffer.begin()) <= len);
            len = (end - number_buffer.begin());
        }

        // convert decimal point to '.'
        if (decimal_point != '\0' && decimal_point != '.')
        {
            // NOLINTNEXTLINE(readability-qualified-auto,llvm-qualified-auto): std::find returns an iterator, see https://github.com/nlohmann/json/issues/3081
            const auto dec_pos = std::find(number_buffer.begin(), number_buffer.end(), decimal_point);
            if (dec_pos != number_buffer.end())
            {
                *dec_pos = '.';
            }
        }

        o->write_characters(number_buffer.data(), static_cast<std::size_t>(len));

        // determine if we need to append ".0"
        const bool value_is_int_like =
            std::none_of(number_buffer.begin(), number_buffer.begin() + len + 1,
                         [](char c)
        {
            return c == '.' || c == 'e';
        });

        if (value_is_int_like)
        {
            o->write_characters(".0", 2);
        }
    }

    /*!
    @brief check whether a string is UTF-8 encoded

    The function checks each byte of a string whether it is UTF-8 encoded. The
    result of the check is stored in the @a state parameter. The function must
    be called initially with state 0 (accept). State 1 means the string must
    be rejected, because the current byte is not allowed. If the string is
    completely processed, but the state is non-zero, the string ended
    prematurely; that is, the last byte indicated more bytes should have
    followed.

    @param[in,out] state  the state of the decoding
    @param[in,out] codep  codepoint (valid only if resulting state is UTF8_ACCEPT)
    @param[in] byte       next byte to decode
    @return               new state

    @note The function has been edited: a std::array is used.

    @copyright Copyright (c) 2008-2009 Bjoern Hoehrmann <bjoern@hoehrmann.de>
    @sa http://bjoern.hoehrmann.de/utf-8/decoder/dfa/
    */
    static std::uint8_t decode(std::uint8_t& state, std::uint32_t& codep, const std::uint8_t byte) noexcept
    {
        static const std::array<std::uint8_t, 400> utf8d =
        {
            {
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 00..1F
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 20..3F
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 40..5F
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 60..7F
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, // 80..9F
                7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, // A0..BF
                8, 8, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, // C0..DF
                0xA, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x4, 0x3, 0x3, // E0..EF
                0xB, 0x6, 0x6, 0x6, 0x5, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, // F0..FF
                0x0, 0x1, 0x2, 0x3, 0x5, 0x8, 0x7, 0x1, 0x1, 0x1, 0x4, 0x6, 0x1, 0x1, 0x1, 0x1, // s0..s0
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, // s1..s2
                1, 2, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, // s3..s4
                1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 3, 1, 1, 1, 1, 1, 1, // s5..s6
                1, 3, 1, 1, 1, 1, 1, 3, 1, 3, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 // s7..s8
            }
        };

        JSON_ASSERT(byte < utf8d.size());
        const std::uint8_t type = utf8d[byte];

        codep = (state != UTF8_ACCEPT)
                ? (byte & 0x3fu) | (codep << 6u)
                : (0xFFu >> type) & (byte);

        const std::size_t index = 256u + (static_cast<size_t>(state) * 16u) + static_cast<size_t>(type);
        JSON_ASSERT(index < utf8d.size());
        state = utf8d[index];
        return state;
    }

    /*
     * Overload to make the compiler happy while it is instantiating
     * dump_integer for number_unsigned_t.
     * Must never be called.
     */
    number_unsigned_t remove_sign(number_unsigned_t x)
    {
        JSON_ASSERT(false); // NOLINT(cert-dcl03-c,hicpp-static-assert,misc-static-assert) LCOV_EXCL_LINE
        return x; // LCOV_EXCL_LINE
    }

    /*
     * Helper function for dump_integer
     *
     * This function takes a negative signed integer and returns its absolute
     * value as unsigned integer. The plus/minus shuffling is necessary as we can
     * not directly remove the sign of an arbitrary signed integer as the
     * absolute values of INT_MIN and INT_MAX are usually not the same. See
     * #1708 for details.
     */
    number_unsigned_t remove_sign(number_integer_t x) noexcept
    {
        JSON_ASSERT(x < 0 && x < (std::numeric_limits<number_integer_t>::max)()); // NOLINT(misc-redundant-expression)
        return static_cast<number_unsigned_t>(-(x + 1)) + 1;
    }

  private:
    /// the output of the serializer
    output_adapter_t<char> o = nullptr;

    /// a (hopefully) large enough character buffer
    std::array<char, 64> number_buffer{{}};

    /// the locale
    const std::lconv* loc = nullptr;
    /// the locale's thousand separator character
    const char thousands_sep = '\0';
    /// the locale's decimal point character
    const char decimal_point = '\0';

    /// string buffer
    std::array<char, 512> string_buffer{{}};

    /// the indentation character
    const char indent_char;
    /// the indentation string
    string_t indent_string;

    /// error_handler how to react on decoding errors
    const error_handler_t error_handler;
};

}  // namespace detail
NLOHMANN_JSON_NAMESPACE_END

// #include <nlohmann/detail/value_t.hpp>

// #include <nlohmann/json_fwd.hpp>

// #include <nlohmann/ordered_map.hpp>
//     __ _____ _____ _____
//  __|  |   __|     |   | |  JSON for Modern C++
// |  |  |__   |  |  | | | |  version 3.12.0
// |_____|_____|_____|_|___|  https://github.com/nlohmann/json
//
// SPDX-FileCopyrightText: 2013 - 2025 Niels Lohmann <https://nlohmann.me>
// SPDX-License-Identifier: MIT



#include <functional> // equal_to, less
#include <initializer_list> // initializer_list
#include <iterator> // input_iterator_tag, iterator_traits
#include <memory> // allocator
#include <stdexcept> // for out_of_range
#include <type_traits> // enable_if, is_convertible
#include <utility> // pair
#include <vector> // vector

// #include <nlohmann/detail/macro_scope.hpp>

// #include <nlohmann/detail/meta/type_traits.hpp>


NLOHMANN_JSON_NAMESPACE_BEGIN

/// ordered_map: a minimal map-like container that preserves insertion order
/// for use within nlohmann::basic_json<ordered_map>
template <class Key, class T, class IgnoredLess = std::less<Key>,
          class Allocator = std::allocator<std::pair<const Key, T>>>
              struct ordered_map : std::vector<std::pair<const Key, T>, Allocator>
{
    using key_type = Key;
    using mapped_type = T;
    using Container = std::vector<std::pair<const Key, T>, Allocator>;
    using iterator = typename Container::iterator;
    using const_iterator = typename Container::const_iterator;
    using size_type = typename Container::size_type;
    using value_type = typename Container::value_type;
#ifdef JSON_HAS_CPP_14
    using key_compare = std::equal_to<>;
#else
    using key_compare = std::equal_to<Key>;
#endif

    // Explicit constructors instead of `using Container::Container`
    // otherwise older compilers choke on it (GCC <= 5.5, xcode <= 9.4)
    ordered_map() noexcept(noexcept(Container())) : Container{} {}
    explicit ordered_map(const Allocator& alloc) noexcept(noexcept(Container(alloc))) : Container{alloc} {}
    template <class It>
    ordered_map(It first, It last, const Allocator& alloc = Allocator())
        : Container{first, last, alloc} {}
    ordered_map(std::initializer_list<value_type> init, const Allocator& alloc = Allocator() )
        : Container{init, alloc} {}

    std::pair<iterator, bool> emplace(const key_type& key, T&& t)
    {
        for (auto it = this->begin(); it != this->end(); ++it)
        {
            if (m_compare(it->first, key))
            {
                return {it, false};
            }
        }
        Container::emplace_back(key, std::forward<T>(t));
        return {std::prev(this->end()), true};
    }

    template<class KeyType, detail::enable_if_t<
                 detail::is_usable_as_key_type<key_compare, key_type, KeyType>::value, int> = 0>
    std::pair<iterator, bool> emplace(KeyType && key, T && t)
    {
        for (auto it = this->begin(); it != this->end(); ++it)
        {
            if (m_compare(it->first, key))
            {
                return {it, false};
            }
        }
        Container::emplace_back(std::forward<KeyType>(key), std::forward<T>(t));
        return {std::prev(this->end()), true};
    }

    T& operator[](const key_type& key)
    {
        return emplace(key, T{}).first->second;
    }

    template<class KeyType, detail::enable_if_t<
                 detail::is_usable_as_key_type<key_compare, key_type, KeyType>::value, int> = 0>
    T & operator[](KeyType && key)
    {
        return emplace(std::forward<KeyType>(key), T{}).first->second;
    }

    const T& operator[](const key_type& key) const
    {
        return at(key);
    }

    template<class KeyType, detail::enable_if_t<
                 detail::is_usable_as_key_type<key_compare, key_type, KeyType>::value, int> = 0>
    const T & operator[](KeyType && key) const
    {
        return at(std::forward<KeyType>(key));
    }

    T& at(const key_type& key)
    {
        for (auto it = this->begin(); it != this->end(); ++it)
        {
            if (m_compare(it->first, key))
            {
                return it->second;
            }
        }

        JSON_THROW(std::out_of_range("key not found"));
    }

    template<class KeyType, detail::enable_if_t<
                 detail::is_usable_as_key_type<key_compare, key_type, KeyType>::value, int> = 0>
    T & at(KeyType && key) // NOLINT(cppcoreguidelines-missing-std-forward)
    {
        for (auto it = this->begin(); it != this->end(); ++it)
        {
            if (m_compare(it->first, key))
            {
                return it->second;
            }
        }

        JSON_THROW(std::out_of_range("key not found"));
    }

    const T& at(const key_type& key) const
    {
        for (auto it = this->begin(); it != this->end(); ++it)
        {
            if (m_compare(it->first, key))
            {
                return it->second;
            }
        }

        JSON_THROW(std::out_of_range("key not found"));
    }

    template<class KeyType, detail::enable_if_t<
                 detail::is_usable_as_key_type<key_compare, key_type, KeyType>::value, int> = 0>
    const T & at(KeyType && key) const // NOLINT(cppcoreguidelines-missing-std-forward)
    {
        for (auto it = this->begin(); it != this->end(); ++it)
        {
            if (m_compare(it->first, key))
            {
                return it->second;
            }
        }

        JSON_THROW(std::out_of_range("key not found"));
    }

    size_type erase(const key_type& key)
    {
        for (auto it = this->begin(); it != this->end(); ++it)
        {
            if (m_compare(it->first, key))
            {
                // Since we cannot move const Keys, re-construct them in place
                for (auto next = it; ++next != this->end(); ++it)
                {
                    it->~value_type(); // Destroy but keep allocation
                    new (&*it) value_type{std::move(*next)};
                }
                Container::pop_back();
                return 1;
            }
        }
        return 0;
    }

    template<class KeyType, detail::enable_if_t<
                 detail::is_usable_as_key_type<key_compare, key_type, KeyType>::value, int> = 0>
    size_type erase(KeyType && key) // NOLINT(cppcoreguidelines-missing-std-forward)
    {
        for (auto it = this->begin(); it != this->end(); ++it)
        {
            if (m_compare(it->first, key))
            {
                // Since we cannot move const Keys, re-construct them in place
                for (auto next = it; ++next != this->end(); ++it)
                {
                    it->~value_type(); // Destroy but keep allocation
                    new (&*it) value_type{std::move(*next)};
                }
                Container::pop_back();
                return 1;
            }
        }
        return 0;
    }

    iterator erase(iterator pos)
    {
        return erase(pos, std::next(pos));
    }

    iterator erase(iterator first, iterator last)
    {
        if (first == last)
        {
            return first;
        }

        const auto elements_affected = std::distance(first, last);
        const auto offset = std::distance(Container::begin(), first);

        // This is the start situation. We need to delete elements_affected
        // elements (3 in this example: e, f, g), and need to return an
        // iterator past the last deleted element (h in this example).
        // Note that offset is the distance from the start of the vector
        // to first. We will need this later.

        // [ a, b, c, d, e, f, g, h, i, j ]
        //               ^        ^
        //             first    last

        // Since we cannot move const Keys, we re-construct them in place.
        // We start at first and re-construct (viz. copy) the elements from
        // the back of the vector. Example for first iteration:

        //               ,--------.
        //               v        |   destroy e and re-construct with h
        // [ a, b, c, d, e, f, g, h, i, j ]
        //               ^        ^
        //               it       it + elements_affected

        for (auto it = first; std::next(it, elements_affected) != Container::end(); ++it)
        {
            it->~value_type(); // destroy but keep allocation
            new (&*it) value_type{std::move(*std::next(it, elements_affected))}; // "move" next element to it
        }

        // [ a, b, c, d, h, i, j, h, i, j ]
        //               ^        ^
        //             first    last

        // remove the unneeded elements at the end of the vector
        Container::resize(this->size() - static_cast<size_type>(elements_affected));

        // [ a, b, c, d, h, i, j ]
        //               ^        ^
        //             first    last

        // first is now pointing past the last deleted element, but we cannot
        // use this iterator, because it may have been invalidated by the
        // resize call. Instead, we can return begin() + offset.
        return Container::begin() + offset;
    }

    size_type count(const key_type& key) const
    {
        for (auto it = this->begin(); it != this->end(); ++it)
        {
            if (m_compare(it->first, key))
            {
                return 1;
            }
        }
        return 0;
    }

    template<class KeyType, detail::enable_if_t<
                 detail::is_usable_as_key_type<key_compare, key_type, KeyType>::value, int> = 0>
    size_type count(KeyType && key) const // NOLINT(cppcoreguidelines-missing-std-forward)
    {
        for (auto it = this->begin(); it != this->end(); ++it)
        {
            if (m_compare(it->first, key))
            {
                return 1;
            }
        }
        return 0;
    }

    iterator find(const key_type& key)
    {
        for (auto it = this->begin(); it != this->end(); ++it)
        {
            if (m_compare(it->first, key))
            {
                return it;
            }
        }
        return Container::end();
    }

    template<class KeyType, detail::enable_if_t<
                 detail::is_usable_as_key_type<key_compare, key_type, KeyType>::value, int> = 0>
    iterator find(KeyType && key) // NOLINT(cppcoreguidelines-missing-std-forward)
    {
        for (auto it = this->begin(); it != this->end(); ++it)
        {
            if (m_compare(it->first, key))
            {
                return it;
            }
        }
        return Container::end();
    }

    const_iterator find(const key_type& key) const
    {
        for (auto it = this->begin(); it != this->end(); ++it)
        {
            if (m_compare(it->first, key))
            {
                return it;
            }
        }
        return Container::end();
    }

    std::pair<iterator, bool> insert( value_type&& value )
    {
        return emplace(value.first, std::move(value.second));
    }

    std::pair<iterator, bool> insert( const value_type& value )
    {
        for (auto it = this->begin(); it != this->end(); ++it)
        {
            if (m_compare(it->first, value.first))
            {
                return {it, false};
            }
        }
        Container::push_back(value);
        return {--this->end(), true};
    }

    template<typename InputIt>
    using require_input_iter = typename std::enable_if<std::is_convertible<typename std::iterator_traits<InputIt>::iterator_category,
        std::input_iterator_tag>::value>::type;

    template<typename InputIt, typename = require_input_iter<InputIt>>
    void insert(InputIt first, InputIt last)
    {
        for (auto it = first; it != last; ++it)
        {
            insert(*it);
        }
    }

private:
    JSON_NO_UNIQUE_ADDRESS key_compare m_compare = key_compare();
};

NLOHMANN_JSON_NAMESPACE_END


#if defined(JSON_HAS_CPP_17)
    #if JSON_HAS_STATIC_RTTI
        #include <any>
    #endif
    #include <string_view>
#endif

/*!
@brief namespace for Niels Lohmann
@see https://github.com/nlohmann
@since version 1.0.0
*/
NLOHMANN_JSON_NAMESPACE_BEGIN

/*!
@brief a class to store JSON values

@internal
@invariant The member variables @a m_value and @a m_type have the following
relationship:
- If `m_type == value_t::object`, then `m_value.object != nullptr`.
- If `m_type == value_t::array`, then `m_value.array != nullptr`.
- If `m_type == value_t::string`, then `m_value.string != nullptr`.
The invariants are checked by member function assert_invariant().

@note ObjectType trick from https://stackoverflow.com/a/9860911
@endinternal

@since version 1.0.0

@nosubgrouping
*/
NLOHMANN_BASIC_JSON_TPL_DECLARATION
class basic_json // NOLINT(cppcoreguidelines-special-member-functions,hicpp-special-member-functions)
    : public ::nlohmann::detail::json_base_class<CustomBaseClass>
{
  private:
    template<detail::value_t> friend struct detail::external_constructor;

    template<typename>
    friend class ::nlohmann::json_pointer;
    // can be restored when json_pointer backwards compatibility is removed
    // friend ::nlohmann::json_pointer<StringType>;

    template<typename BasicJsonType, typename InputType>
    friend class ::nlohmann::detail::parser;
    friend ::nlohmann::detail::serializer<basic_json>;
    template<typename BasicJsonType>
    friend class ::nlohmann::detail::iter_impl;
    template<typename BasicJsonType, typename CharType>
    friend class ::nlohmann::detail::binary_writer;
    template<typename BasicJsonType, typename InputType, typename SAX>
    friend class ::nlohmann::detail::binary_reader;
    template<typename BasicJsonType, typename InputAdapterType>
    friend class ::nlohmann::detail::json_sax_dom_parser;
    template<typename BasicJsonType, typename InputAdapterType>
    friend class ::nlohmann::detail::json_sax_dom_callback_parser;
    friend class ::nlohmann::detail::exception;

    /// workaround type for MSVC
    using basic_json_t = NLOHMANN_BASIC_JSON_TPL;
    using json_base_class_t = ::nlohmann::detail::json_base_class<CustomBaseClass>;

  JSON_PRIVATE_UNLESS_TESTED:
    // convenience aliases for types residing in namespace detail;
    using lexer = ::nlohmann::detail::lexer_base<basic_json>;

    template<typename InputAdapterType>
    static ::nlohmann::detail::parser<basic_json, InputAdapterType> parser(
        InputAdapterType adapter,
        detail::parser_callback_t<basic_json>cb = nullptr,
        const bool allow_exceptions = true,
        const bool ignore_comments = false
                                 )
    {
        return ::nlohmann::detail::parser<basic_json, InputAdapterType>(std::move(adapter),
            std::move(cb), allow_exceptions, ignore_comments);
    }

  private:
    using primitive_iterator_t = ::nlohmann::detail::primitive_iterator_t;
    template<typename BasicJsonType>
    using internal_iterator = ::nlohmann::detail::internal_iterator<BasicJsonType>;
    template<typename BasicJsonType>
    using iter_impl = ::nlohmann::detail::iter_impl<BasicJsonType>;
    template<typename Iterator>
    using iteration_proxy = ::nlohmann::detail::iteration_proxy<Iterator>;
    template<typename Base> using json_reverse_iterator = ::nlohmann::detail::json_reverse_iterator<Base>;

    template<typename CharType>
    using output_adapter_t = ::nlohmann::detail::output_adapter_t<CharType>;

    template<typename InputType>
    using binary_reader = ::nlohmann::detail::binary_reader<basic_json, InputType>;
    template<typename CharType> using binary_writer = ::nlohmann::detail::binary_writer<basic_json, CharType>;

  JSON_PRIVATE_UNLESS_TESTED:
    using serializer = ::nlohmann::detail::serializer<basic_json>;

  public:
    using value_t = detail::value_t;
    /// JSON Pointer, see @ref nlohmann::json_pointer
    using json_pointer = ::nlohmann::json_pointer<StringType>;
    template<typename T, typename SFINAE>
    using json_serializer = JSONSerializer<T, SFINAE>;
    /// how to treat decoding errors
    using error_handler_t = detail::error_handler_t;
    /// how to treat CBOR tags
    using cbor_tag_handler_t = detail::cbor_tag_handler_t;
    /// how to encode BJData
    using bjdata_version_t = detail::bjdata_version_t;
    /// helper type for initializer lists of basic_json values
    using initializer_list_t = std::initializer_list<detail::json_ref<basic_json>>;

    using input_format_t = detail::input_format_t;
    /// SAX interface type, see @ref nlohmann::json_sax
    using json_sax_t = json_sax<basic_json>;

    ////////////////
    // exceptions //
    ////////////////

    /// @name exceptions
    /// Classes to implement user-defined exceptions.
    /// @{

    using exception = detail::exception;
    using parse_error = detail::parse_error;
    using invalid_iterator = detail::invalid_iterator;
    using type_error = detail::type_error;
    using out_of_range = detail::out_of_range;
    using other_error = detail::other_error;

    /// @}

    /////////////////////
    // container types //
    /////////////////////

    /// @name container types
    /// The canonic container types to use @ref basic_json like any other STL
    /// container.
    /// @{

    /// the type of elements in a basic_json container
    using value_type = basic_json;

    /// the type of an element reference
    using reference = value_type&;
    /// the type of an element const reference
    using const_reference = const value_type&;

    /// a type to represent differences between iterators
    using difference_type = std::ptrdiff_t;
    /// a type to represent container sizes
    using size_type = std::size_t;

    /// the allocator type
    using allocator_type = AllocatorType<basic_json>;

    /// the type of an element pointer
    using pointer = typename std::allocator_traits<allocator_type>::pointer;
    /// the type of an element const pointer
    using const_pointer = typename std::allocator_traits<allocator_type>::const_pointer;

    /// an iterator for a basic_json container
    using iterator = iter_impl<basic_json>;
    /// a const iterator for a basic_json container
    using const_iterator = iter_impl<const basic_json>;
    /// a reverse iterator for a basic_json container
    using reverse_iterator = json_reverse_iterator<typename basic_json::iterator>;
    /// a const reverse iterator for a basic_json container
    using const_reverse_iterator = json_reverse_iterator<typename basic_json::const_iterator>;

    /// @}

    /// @brief returns the allocator associated with the container
    /// @sa https://json.nlohmann.me/api/basic_json/get_allocator/
    static allocator_type get_allocator()
    {
        return allocator_type();
    }

    /// @brief returns version information on the library
    /// @sa https://json.nlohmann.me/api/basic_json/meta/
    JSON_HEDLEY_WARN_UNUSED_RESULT
    static basic_json meta()
    {
        basic_json result;

        result["copyright"] = "(C) 2013-2025 Niels Lohmann";
        result["name"] = "JSON for Modern C++";
        result["url"] = "https://github.com/nlohmann/json";
        result["version"]["string"] =
            detail::concat(std::to_string(NLOHMANN_JSON_VERSION_MAJOR), '.',
                           std::to_string(NLOHMANN_JSON_VERSION_MINOR), '.',
                           std::to_string(NLOHMANN_JSON_VERSION_PATCH));
        result["version"]["major"] = NLOHMANN_JSON_VERSION_MAJOR;
        result["version"]["minor"] = NLOHMANN_JSON_VERSION_MINOR;
        result["version"]["patch"] = NLOHMANN_JSON_VERSION_PATCH;

#ifdef _WIN32
        result["platform"] = "win32";
#elif defined __linux__
        result["platform"] = "linux";
#elif defined __APPLE__
        result["platform"] = "apple";
#elif defined __unix__
        result["platform"] = "unix";
#else
        result["platform"] = "unknown";
#endif

#if defined(__ICC) || defined(__INTEL_COMPILER)
        result["compiler"] = {{"family", "icc"}, {"version", __INTEL_COMPILER}};
#elif defined(__clang__)
        result["compiler"] = {{"family", "clang"}, {"version", __clang_version__}};
#elif defined(__GNUC__) || defined(__GNUG__)
        result["compiler"] = {{"family", "gcc"}, {"version", detail::concat(
                    std::to_string(__GNUC__), '.',
                    std::to_string(__GNUC_MINOR__), '.',
                    std::to_string(__GNUC_PATCHLEVEL__))
            }
        };
#elif defined(__HP_cc) || defined(__HP_aCC)
        result["compiler"] = "hp"
#elif defined(__IBMCPP__)
        result["compiler"] = {{"family", "ilecpp"}, {"version", __IBMCPP__}};
#elif defined(_MSC_VER)
        result["compiler"] = {{"family", "msvc"}, {"version", _MSC_VER}};
#elif defined(__PGI)
        result["compiler"] = {{"family", "pgcpp"}, {"version", __PGI}};
#elif defined(__SUNPRO_CC)
        result["compiler"] = {{"family", "sunpro"}, {"version", __SUNPRO_CC}};
#else
        result["compiler"] = {{"family", "unknown"}, {"version", "unknown"}};
#endif

#if defined(_MSVC_LANG)
        result["compiler"]["c++"] = std::to_string(_MSVC_LANG);
#elif defined(__cplusplus)
        result["compiler"]["c++"] = std::to_string(__cplusplus);
#else
        result["compiler"]["c++"] = "unknown";
#endif
        return result;
    }

    ///////////////////////////
    // JSON value data types //
    ///////////////////////////

    /// @name JSON value data types
    /// The data types to store a JSON value. These types are derived from
    /// the template arguments passed to class @ref basic_json.
    /// @{

    /// @brief default object key comparator type
    /// The actual object key comparator type (@ref object_comparator_t) may be
    /// different.
    /// @sa https://json.nlohmann.me/api/basic_json/default_object_comparator_t/
#if defined(JSON_HAS_CPP_14)
    // use of transparent comparator avoids unnecessary repeated construction of temporaries
    // in functions involving lookup by key with types other than object_t::key_type (aka. StringType)
    using default_object_comparator_t = std::less<>;
#else
    using default_object_comparator_t = std::less<StringType>;
#endif

    /// @brief a type for an object
    /// @sa https://json.nlohmann.me/api/basic_json/object_t/
    using object_t = ObjectType<StringType,
          basic_json,
          default_object_comparator_t,
          AllocatorType<std::pair<const StringType,
          basic_json>>>;

    /// @brief a type for an array
    /// @sa https://json.nlohmann.me/api/basic_json/array_t/
    using array_t = ArrayType<basic_json, AllocatorType<basic_json>>;

    /// @brief a type for a string
    /// @sa https://json.nlohmann.me/api/basic_json/string_t/
    using string_t = StringType;

    /// @brief a type for a boolean
    /// @sa https://json.nlohmann.me/api/basic_json/boolean_t/
    using boolean_t = BooleanType;

    /// @brief a type for a number (integer)
    /// @sa https://json.nlohmann.me/api/basic_json/number_integer_t/
    using number_integer_t = NumberIntegerType;

    /// @brief a type for a number (unsigned)
    /// @sa https://json.nlohmann.me/api/basic_json/number_unsigned_t/
    using number_unsigned_t = NumberUnsignedType;

    /// @brief a type for a number (floating-point)
    /// @sa https://json.nlohmann.me/api/basic_json/number_float_t/
    using number_float_t = NumberFloatType;

    /// @brief a type for a packed binary type
    /// @sa https://json.nlohmann.me/api/basic_json/binary_t/
    using binary_t = nlohmann::byte_container_with_subtype<BinaryType>;

    /// @brief object key comparator type
    /// @sa https://json.nlohmann.me/api/basic_json/object_comparator_t/
    using object_comparator_t = detail::actual_object_comparator_t<basic_json>;

    /// @}

  private:

    /// helper for exception-safe object creation
    template<typename T, typename... Args>
    JSON_HEDLEY_RETURNS_NON_NULL
    static T* create(Args&& ... args)
    {
        AllocatorType<T> alloc;
        using AllocatorTraits = std::allocator_traits<AllocatorType<T>>;

        auto deleter = [&](T * obj)
        {
            AllocatorTraits::deallocate(alloc, obj, 1);
        };
        std::unique_ptr<T, decltype(deleter)> obj(AllocatorTraits::allocate(alloc, 1), deleter);
        AllocatorTraits::construct(alloc, obj.get(), std::forward<Args>(args)...);
        JSON_ASSERT(obj != nullptr);
        return obj.release();
    }

    ////////////////////////
    // JSON value storage //
    ////////////////////////

  JSON_PRIVATE_UNLESS_TESTED:
    /*!
    @brief a JSON value

    The actual storage for a JSON value of the @ref basic_json class. This
    union combines the different storage types for the JSON value types
    defined in @ref value_t.

    JSON type | value_t type    | used type
    --------- | --------------- | ------------------------
    object    | object          | pointer to @ref object_t
    array     | array           | pointer to @ref array_t
    string    | string          | pointer to @ref string_t
    boolean   | boolean         | @ref boolean_t
    number    | number_integer  | @ref number_integer_t
    number    | number_unsigned | @ref number_unsigned_t
    number    | number_float    | @ref number_float_t
    binary    | binary          | pointer to @ref binary_t
    null      | null            | *no value is stored*

    @note Variable-length types (objects, arrays, and strings) are stored as
    pointers. The size of the union should not exceed 64 bits if the default
    value types are used.

    @since version 1.0.0
    */
    union json_value
    {
        /// object (stored with pointer to save storage)
        object_t* object;
        /// array (stored with pointer to save storage)
        array_t* array;
        /// string (stored with pointer to save storage)
        string_t* string;
        /// binary (stored with pointer to save storage)
        binary_t* binary;
        /// boolean
        boolean_t boolean;
        /// number (integer)
        number_integer_t number_integer;
        /// number (unsigned integer)
        number_unsigned_t number_unsigned;
        /// number (floating-point)
        number_float_t number_float;

        /// default constructor (for null values)
        json_value() = default;
        /// constructor for booleans
        json_value(boolean_t v) noexcept : boolean(v) {}
        /// constructor for numbers (integer)
        json_value(number_integer_t v) noexcept : number_integer(v) {}
        /// constructor for numbers (unsigned)
        json_value(number_unsigned_t v) noexcept : number_unsigned(v) {}
        /// constructor for numbers (floating-point)
        json_value(number_float_t v) noexcept : number_float(v) {}
        /// constructor for empty values of a given type
        json_value(value_t t)
        {
            switch (t)
            {
                case value_t::object:
                {
                    object = create<object_t>();
                    break;
                }

                case value_t::array:
                {
                    array = create<array_t>();
                    break;
                }

                case value_t::string:
                {
                    string = create<string_t>("");
                    break;
                }

                case value_t::binary:
                {
                    binary = create<binary_t>();
                    break;
                }

                case value_t::boolean:
                {
                    boolean = static_cast<boolean_t>(false);
                    break;
                }

                case value_t::number_integer:
                {
                    number_integer = static_cast<number_integer_t>(0);
                    break;
                }

                case value_t::number_unsigned:
                {
                    number_unsigned = static_cast<number_unsigned_t>(0);
                    break;
                }

                case value_t::number_float:
                {
                    number_float = static_cast<number_float_t>(0.0);
                    break;
                }

                case value_t::null:
                {
                    object = nullptr;  // silence warning, see #821
                    break;
                }

                case value_t::discarded:
                default:
                {
                    object = nullptr;  // silence warning, see #821
                    if (JSON_HEDLEY_UNLIKELY(t == value_t::null))
                    {
                        JSON_THROW(other_error::create(500, "961c151d2e87f2686a955a9be24d316f1362bf21 3.12.0", nullptr)); // LCOV_EXCL_LINE
                    }
                    break;
                }
            }
        }

        /// constructor for strings
        json_value(const string_t& value) : string(create<string_t>(value)) {}

        /// constructor for rvalue strings
        json_value(string_t&& value) : string(create<string_t>(std::move(value))) {}

        /// constructor for objects
        json_value(const object_t& value) : object(create<object_t>(value)) {}

        /// constructor for rvalue objects
        json_value(object_t&& value) : object(create<object_t>(std::move(value))) {}

        /// constructor for arrays
        json_value(const array_t& value) : array(create<array_t>(value)) {}

        /// constructor for rvalue arrays
        json_value(array_t&& value) : array(create<array_t>(std::move(value))) {}

        /// constructor for binary arrays
        json_value(const typename binary_t::container_type& value) : binary(create<binary_t>(value)) {}

        /// constructor for rvalue binary arrays
        json_value(typename binary_t::container_type&& value) : binary(create<binary_t>(std::move(value))) {}

        /// constructor for binary arrays (internal type)
        json_value(const binary_t& value) : binary(create<binary_t>(value)) {}

        /// constructor for rvalue binary arrays (internal type)
        json_value(binary_t&& value) : binary(create<binary_t>(std::move(value))) {}

        void destroy(value_t t)
        {
            if (
                (t == value_t::object && object == nullptr) ||
                (t == value_t::array && array == nullptr) ||
                (t == value_t::string && string == nullptr) ||
                (t == value_t::binary && binary == nullptr)
            )
            {
                //not initialized (e.g. due to exception in the ctor)
                return;
            }
            if (t == value_t::array || t == value_t::object)
            {
                // flatten the current json_value to a heap-allocated stack
                std::vector<basic_json> stack;

                // move the top-level items to stack
                if (t == value_t::array)
                {
                    stack.reserve(array->size());
                    std::move(array->begin(), array->end(), std::back_inserter(stack));
                }
                else
                {
                    stack.reserve(object->size());
                    for (auto&& it : *object)
                    {
                        stack.push_back(std::move(it.second));
                    }
                }

                while (!stack.empty())
                {
                    // move the last item to local variable to be processed
                    basic_json current_item(std::move(stack.back()));
                    stack.pop_back();

                    // if current_item is array/object, move
                    // its children to the stack to be processed later
                    if (current_item.is_array())
                    {
                        std::move(current_item.m_data.m_value.array->begin(), current_item.m_data.m_value.array->end(), std::back_inserter(stack));

                        current_item.m_data.m_value.array->clear();
                    }
                    else if (current_item.is_object())
                    {
                        for (auto&& it : *current_item.m_data.m_value.object)
                        {
                            stack.push_back(std::move(it.second));
                        }

                        current_item.m_data.m_value.object->clear();
                    }

                    // it's now safe that current_item get destructed
                    // since it doesn't have any children
                }
            }

            switch (t)
            {
                case value_t::object:
                {
                    AllocatorType<object_t> alloc;
                    std::allocator_traits<decltype(alloc)>::destroy(alloc, object);
                    std::allocator_traits<decltype(alloc)>::deallocate(alloc, object, 1);
                    break;
                }

                case value_t::array:
                {
                    AllocatorType<array_t> alloc;
                    std::allocator_traits<decltype(alloc)>::destroy(alloc, array);
                    std::allocator_traits<decltype(alloc)>::deallocate(alloc, array, 1);
                    break;
                }

                case value_t::string:
                {
                    AllocatorType<string_t> alloc;
                    std::allocator_traits<decltype(alloc)>::destroy(alloc, string);
                    std::allocator_traits<decltype(alloc)>::deallocate(alloc, string, 1);
                    break;
                }

                case value_t::binary:
                {
                    AllocatorType<binary_t> alloc;
                    std::allocator_traits<decltype(alloc)>::destroy(alloc, binary);
                    std::allocator_traits<decltype(alloc)>::deallocate(alloc, binary, 1);
                    break;
                }

                case value_t::null:
                case value_t::boolean:
                case value_t::number_integer:
                case value_t::number_unsigned:
                case value_t::number_float:
                case value_t::discarded:
                default:
                {
                    break;
                }
            }
        }
    };

  private:
    /*!
    @brief checks the class invariants

    This function asserts the class invariants. It needs to be called at the
    end of every constructor to make sure that created objects respect the
    invariant. Furthermore, it has to be called each time the type of a JSON
    value is changed, because the invariant expresses a relationship between
    @a m_type and @a m_value.

    Furthermore, the parent relation is checked for arrays and objects: If
    @a check_parents true and the value is an array or object, then the
    container's elements must have the current value as parent.

    @param[in] check_parents  whether the parent relation should be checked.
               The value is true by default and should only be set to false
               during destruction of objects when the invariant does not
               need to hold.
    */
    void assert_invariant(bool check_parents = true) const noexcept
    {
        JSON_ASSERT(m_data.m_type != value_t::object || m_data.m_value.object != nullptr);
        JSON_ASSERT(m_data.m_type != value_t::array || m_data.m_value.array != nullptr);
        JSON_ASSERT(m_data.m_type != value_t::string || m_data.m_value.string != nullptr);
        JSON_ASSERT(m_data.m_type != value_t::binary || m_data.m_value.binary != nullptr);

#if JSON_DIAGNOSTICS
        JSON_TRY
        {
            // cppcheck-suppress assertWithSideEffect
            JSON_ASSERT(!check_parents || !is_structured() || std::all_of(begin(), end(), [this](const basic_json & j)
            {
                return j.m_parent == this;
            }));
        }
        JSON_CATCH(...) {} // LCOV_EXCL_LINE
#endif
        static_cast<void>(check_parents);
    }

    void set_parents()
    {
#if JSON_DIAGNOSTICS
        switch (m_data.m_type)
        {
            case value_t::array:
            {
                for (auto& element : *m_data.m_value.array)
                {
                    element.m_parent = this;
                }
                break;
            }

            case value_t::object:
            {
                for (auto& element : *m_data.m_value.object)
                {
                    element.second.m_parent = this;
                }
                break;
            }

            case value_t::null:
            case value_t::string:
            case value_t::boolean:
            case value_t::number_integer:
            case value_t::number_unsigned:
            case value_t::number_float:
            case value_t::binary:
            case value_t::discarded:
            default:
                break;
        }
#endif
    }

    iterator set_parents(iterator it, typename iterator::difference_type count_set_parents)
    {
#if JSON_DIAGNOSTICS
        for (typename iterator::difference_type i = 0; i < count_set_parents; ++i)
        {
            (it + i)->m_parent = this;
        }
#else
        static_cast<void>(count_set_parents);
#endif
        return it;
    }

    reference set_parent(reference j, std::size_t old_capacity = detail::unknown_size())
    {
#if JSON_DIAGNOSTICS
        if (old_capacity != detail::unknown_size())
        {
            // see https://github.com/nlohmann/json/issues/2838
            JSON_ASSERT(type() == value_t::array);
            if (JSON_HEDLEY_UNLIKELY(m_data.m_value.array->capacity() != old_capacity))
            {
                // capacity has changed: update all parents
                set_parents();
                return j;
            }
        }

        // ordered_json uses a vector internally, so pointers could have
        // been invalidated; see https://github.com/nlohmann/json/issues/2962
#ifdef JSON_HEDLEY_MSVC_VERSION
#pragma warning(push )
#pragma warning(disable : 4127) // ignore warning to replace if with if constexpr
#endif
        if (detail::is_ordered_map<object_t>::value)
        {
            set_parents();
            return j;
        }
#ifdef JSON_HEDLEY_MSVC_VERSION
#pragma warning( pop )
#endif

        j.m_parent = this;
#else
        static_cast<void>(j);
        static_cast<void>(old_capacity);
#endif
        return j;
    }

  public:
    //////////////////////////
    // JSON parser callback //
    //////////////////////////

    /// @brief parser event types
    /// @sa https://json.nlohmann.me/api/basic_json/parse_event_t/
    using parse_event_t = detail::parse_event_t;

    /// @brief per-element parser callback type
    /// @sa https://json.nlohmann.me/api/basic_json/parser_callback_t/
    using parser_callback_t = detail::parser_callback_t<basic_json>;

    //////////////////
    // constructors //
    //////////////////

    /// @name constructors and destructors
    /// Constructors of class @ref basic_json, copy/move constructor, copy
    /// assignment, static functions creating objects, and the destructor.
    /// @{

    /// @brief create an empty value with a given type
    /// @sa https://json.nlohmann.me/api/basic_json/basic_json/
    basic_json(const value_t v)
        : m_data(v)
    {
        assert_invariant();
    }

    /// @brief create a null object
    /// @sa https://json.nlohmann.me/api/basic_json/basic_json/
    basic_json(std::nullptr_t = nullptr) noexcept // NOLINT(bugprone-exception-escape)
        : basic_json(value_t::null)
    {
        assert_invariant();
    }

    /// @brief create a JSON value from compatible types
    /// @sa https://json.nlohmann.me/api/basic_json/basic_json/
    template < typename CompatibleType,
               typename U = detail::uncvref_t<CompatibleType>,
               detail::enable_if_t <
                   !detail::is_basic_json<U>::value && detail::is_compatible_type<basic_json_t, U>::value, int > = 0 >
    basic_json(CompatibleType && val) noexcept(noexcept( // NOLINT(bugprone-forwarding-reference-overload,bugprone-exception-escape)
            JSONSerializer<U>::to_json(std::declval<basic_json_t&>(),
                                       std::forward<CompatibleType>(val))))
    {
        JSONSerializer<U>::to_json(*this, std::forward<CompatibleType>(val));
        set_parents();
        assert_invariant();
    }

    /// @brief create a JSON value from an existing one
    /// @sa https://json.nlohmann.me/api/basic_json/basic_json/
    template < typename BasicJsonType,
               detail::enable_if_t <
                   detail::is_basic_json<BasicJsonType>::value&& !std::is_same<basic_json, BasicJsonType>::value, int > = 0 >
    basic_json(const BasicJsonType& val)
#if JSON_DIAGNOSTIC_POSITIONS
        : start_position(val.start_pos()),
          end_position(val.end_pos())
#endif
    {
        using other_boolean_t = typename BasicJsonType::boolean_t;
        using other_number_float_t = typename BasicJsonType::number_float_t;
        using other_number_integer_t = typename BasicJsonType::number_integer_t;
        using other_number_unsigned_t = typename BasicJsonType::number_unsigned_t;
        using other_string_t = typename BasicJsonType::string_t;
        using other_object_t = typename BasicJsonType::object_t;
        using other_array_t = typename BasicJsonType::array_t;
        using other_binary_t = typename BasicJsonType::binary_t;

        switch (val.type())
        {
            case value_t::boolean:
                JSONSerializer<other_boolean_t>::to_json(*this, val.template get<other_boolean_t>());
                break;
            case value_t::number_float:
                JSONSerializer<other_number_float_t>::to_json(*this, val.template get<other_number_float_t>());
                break;
            case value_t::number_integer:
                JSONSerializer<other_number_integer_t>::to_json(*this, val.template get<other_number_integer_t>());
                break;
            case value_t::number_unsigned:
                JSONSerializer<other_number_unsigned_t>::to_json(*this, val.template get<other_number_unsigned_t>());
                break;
            case value_t::string:
                JSONSerializer<other_string_t>::to_json(*this, val.template get_ref<const other_string_t&>());
                break;
            case value_t::object:
                JSONSerializer<other_object_t>::to_json(*this, val.template get_ref<const other_object_t&>());
                break;
            case value_t::array:
                JSONSerializer<other_array_t>::to_json(*this, val.template get_ref<const other_array_t&>());
                break;
            case value_t::binary:
                JSONSerializer<other_binary_t>::to_json(*this, val.template get_ref<const other_binary_t&>());
                break;
            case value_t::null:
                *this = nullptr;
                break;
            case value_t::discarded:
                m_data.m_type = value_t::discarded;
                break;
            default:            // LCOV_EXCL_LINE
                JSON_ASSERT(false); // NOLINT(cert-dcl03-c,hicpp-static-assert,misc-static-assert) LCOV_EXCL_LINE
        }
        JSON_ASSERT(m_data.m_type == val.type());

        set_parents();
        assert_invariant();
    }

    /// @brief create a container (array or object) from an initializer list
    /// @sa https://json.nlohmann.me/api/basic_json/basic_json/
    basic_json(initializer_list_t init,
               bool type_deduction = true,
               value_t manual_type = value_t::array)
    {
        // check if each element is an array with two elements whose first
        // element is a string
        bool is_an_object = std::all_of(init.begin(), init.end(),
                                        [](const detail::json_ref<basic_json>& element_ref)
        {
            // The cast is to ensure op[size_type] is called, bearing in mind size_type may not be int;
            // (many string types can be constructed from 0 via its null-pointer guise, so we get a
            // broken call to op[key_type], the wrong semantics and a 4804 warning on Windows)
            return element_ref->is_array() && element_ref->size() == 2 && (*element_ref)[static_cast<size_type>(0)].is_string();
        });

        // adjust type if type deduction is not wanted
        if (!type_deduction)
        {
            // if array is wanted, do not create an object though possible
            if (manual_type == value_t::array)
            {
                is_an_object = false;
            }

            // if object is wanted but impossible, throw an exception
            if (JSON_HEDLEY_UNLIKELY(manual_type == value_t::object && !is_an_object))
            {
                JSON_THROW(type_error::create(301, "cannot create object from initializer list", nullptr));
            }
        }

        if (is_an_object)
        {
            // the initializer list is a list of pairs -> create object
            m_data.m_type = value_t::object;
            m_data.m_value = value_t::object;

            for (auto& element_ref : init)
            {
                auto element = element_ref.moved_or_copied();
                m_data.m_value.object->emplace(
                    std::move(*((*element.m_data.m_value.array)[0].m_data.m_value.string)),
                    std::move((*element.m_data.m_value.array)[1]));
            }
        }
        else
        {
            // the initializer list describes an array -> create array
            m_data.m_type = value_t::array;
            m_data.m_value.array = create<array_t>(init.begin(), init.end());
        }

        set_parents();
        assert_invariant();
    }

    /// @brief explicitly create a binary array (without subtype)
    /// @sa https://json.nlohmann.me/api/basic_json/binary/
    JSON_HEDLEY_WARN_UNUSED_RESULT
    static basic_json binary(const typename binary_t::container_type& init)
    {
        auto res = basic_json();
        res.m_data.m_type = value_t::binary;
        res.m_data.m_value = init;
        return res;
    }

    /// @brief explicitly create a binary array (with subtype)
    /// @sa https://json.nlohmann.me/api/basic_json/binary/
    JSON_HEDLEY_WARN_UNUSED_RESULT
    static basic_json binary(const typename binary_t::container_type& init, typename binary_t::subtype_type subtype)
    {
        auto res = basic_json();
        res.m_data.m_type = value_t::binary;
        res.m_data.m_value = binary_t(init, subtype);
        return res;
    }

    /// @brief explicitly create a binary array
    /// @sa https://json.nlohmann.me/api/basic_json/binary/
    JSON_HEDLEY_WARN_UNUSED_RESULT
    static basic_json binary(typename binary_t::container_type&& init)
    {
        auto res = basic_json();
        res.m_data.m_type = value_t::binary;
        res.m_data.m_value = std::move(init);
        return res;
    }

    /// @brief explicitly create a binary array (with subtype)
    /// @sa https://json.nlohmann.me/api/basic_json/binary/
    JSON_HEDLEY_WARN_UNUSED_RESULT
    static basic_json binary(typename binary_t::container_type&& init, typename binary_t::subtype_type subtype)
    {
        auto res = basic_json();
        res.m_data.m_type = value_t::binary;
        res.m_data.m_value = binary_t(std::move(init), subtype);
        return res;
    }

    /// @brief explicitly create an array from an initializer list
    /// @sa https://json.nlohmann.me/api/basic_json/array/
    JSON_HEDLEY_WARN_UNUSED_RESULT
    static basic_json array(initializer_list_t init = {})
    {
        return basic_json(init, false, value_t::array);
    }

    /// @brief explicitly create an object from an initializer list
    /// @sa https://json.nlohmann.me/api/basic_json/object/
    JSON_HEDLEY_WARN_UNUSED_RESULT
    static basic_json object(initializer_list_t init = {})
    {
        return basic_json(init, false, value_t::object);
    }

    /// @brief construct an array with count copies of given value
    /// @sa https://json.nlohmann.me/api/basic_json/basic_json/
    basic_json(size_type cnt, const basic_json& val):
        m_data{cnt, val}
    {
        set_parents();
        assert_invariant();
    }

    /// @brief construct a JSON container given an iterator range
    /// @sa https://json.nlohmann.me/api/basic_json/basic_json/
    template < class InputIT, typename std::enable_if <
                   std::is_same<InputIT, typename basic_json_t::iterator>::value ||
                   std::is_same<InputIT, typename basic_json_t::const_iterator>::value, int >::type = 0 >
    basic_json(InputIT first, InputIT last) // NOLINT(performance-unnecessary-value-param)
    {
        JSON_ASSERT(first.m_object != nullptr);
        JSON_ASSERT(last.m_object != nullptr);

        // make sure iterator fits the current value
        if (JSON_HEDLEY_UNLIKELY(first.m_object != last.m_object))
        {
            JSON_THROW(invalid_iterator::create(201, "iterators are not compatible", nullptr));
        }

        // copy type from first iterator
        m_data.m_type = first.m_object->m_data.m_type;

        // check if iterator range is complete for primitive values
        switch (m_data.m_type)
        {
            case value_t::boolean:
            case value_t::number_float:
            case value_t::number_integer:
            case value_t::number_unsigned:
            case value_t::string:
            {
                if (JSON_HEDLEY_UNLIKELY(!first.m_it.primitive_iterator.is_begin()
                                         || !last.m_it.primitive_iterator.is_end()))
                {
                    JSON_THROW(invalid_iterator::create(204, "iterators out of range", first.m_object));
                }
                break;
            }

            case value_t::null:
            case value_t::object:
            case value_t::array:
            case value_t::binary:
            case value_t::discarded:
            default:
                break;
        }

        switch (m_data.m_type)
        {
            case value_t::number_integer:
            {
                m_data.m_value.number_integer = first.m_object->m_data.m_value.number_integer;
                break;
            }

            case value_t::number_unsigned:
            {
                m_data.m_value.number_unsigned = first.m_object->m_data.m_value.number_unsigned;
                break;
            }

            case value_t::number_float:
            {
                m_data.m_value.number_float = first.m_object->m_data.m_value.number_float;
                break;
            }

            case value_t::boolean:
            {
                m_data.m_value.boolean = first.m_object->m_data.m_value.boolean;
                break;
            }

            case value_t::string:
            {
                m_data.m_value = *first.m_object->m_data.m_value.string;
                break;
            }

            case value_t::object:
            {
                m_data.m_value.object = create<object_t>(first.m_it.object_iterator,
                                        last.m_it.object_iterator);
                break;
            }

            case value_t::array:
            {
                m_data.m_value.array = create<array_t>(first.m_it.array_iterator,
                                                       last.m_it.array_iterator);
                break;
            }

            case value_t::binary:
            {
                m_data.m_value = *first.m_object->m_data.m_value.binary;
                break;
            }

            case value_t::null:
            case value_t::discarded:
            default:
                JSON_THROW(invalid_iterator::create(206, detail::concat("cannot construct with iterators from ", first.m_object->type_name()), first.m_object));
        }

        set_parents();
        assert_invariant();
    }

    ///////////////////////////////////////
    // other constructors and destructor //
    ///////////////////////////////////////

    template<typename JsonRef,
             detail::enable_if_t<detail::conjunction<detail::is_json_ref<JsonRef>,
                                 std::is_same<typename JsonRef::value_type, basic_json>>::value, int> = 0 >
    basic_json(const JsonRef& ref) : basic_json(ref.moved_or_copied()) {}

    /// @brief copy constructor
    /// @sa https://json.nlohmann.me/api/basic_json/basic_json/
    basic_json(const basic_json& other)
        : json_base_class_t(other)
#if JSON_DIAGNOSTIC_POSITIONS
        , start_position(other.start_position)
        , end_position(other.end_position)
#endif
    {
        m_data.m_type = other.m_data.m_type;
        // check of passed value is valid
        other.assert_invariant();

        switch (m_data.m_type)
        {
            case value_t::object:
            {
                m_data.m_value = *other.m_data.m_value.object;
                break;
            }

            case value_t::array:
            {
                m_data.m_value = *other.m_data.m_value.array;
                break;
            }

            case value_t::string:
            {
                m_data.m_value = *other.m_data.m_value.string;
                break;
            }

            case value_t::boolean:
            {
                m_data.m_value = other.m_data.m_value.boolean;
                break;
            }

            case value_t::number_integer:
            {
                m_data.m_value = other.m_data.m_value.number_integer;
                break;
            }

            case value_t::number_unsigned:
            {
                m_data.m_value = other.m_data.m_value.number_unsigned;
                break;
            }

            case value_t::number_float:
            {
                m_data.m_value = other.m_data.m_value.number_float;
                break;
            }

            case value_t::binary:
            {
                m_data.m_value = *other.m_data.m_value.binary;
                break;
            }

            case value_t::null:
            case value_t::discarded:
            default:
                break;
        }

        set_parents();
        assert_invariant();
    }

    /// @brief move constructor
    /// @sa https://json.nlohmann.me/api/basic_json/basic_json/
    basic_json(basic_json&& other) noexcept
        : json_base_class_t(std::forward<json_base_class_t>(other)),
          m_data(std::move(other.m_data)) // cppcheck-suppress[accessForwarded] TODO check
#if JSON_DIAGNOSTIC_POSITIONS
        , start_position(other.start_position) // cppcheck-suppress[accessForwarded] TODO check
        , end_position(other.end_position) // cppcheck-suppress[accessForwarded] TODO check
#endif
    {
        // check that passed value is valid
        other.assert_invariant(false); // cppcheck-suppress[accessForwarded]

        // invalidate payload
        other.m_data.m_type = value_t::null;
        other.m_data.m_value = {};

#if JSON_DIAGNOSTIC_POSITIONS
        other.start_position = std::string::npos;
        other.end_position = std::string::npos;
#endif

        set_parents();
        assert_invariant();
    }

    /// @brief copy assignment
    /// @sa https://json.nlohmann.me/api/basic_json/operator=/
    basic_json& operator=(basic_json other) noexcept (
        std::is_nothrow_move_constructible<value_t>::value&&
        std::is_nothrow_move_assignable<value_t>::value&&
        std::is_nothrow_move_constructible<json_value>::value&&
        std::is_nothrow_move_assignable<json_value>::value&&
        std::is_nothrow_move_assignable<json_base_class_t>::value
    )
    {
        // check that passed value is valid
        other.assert_invariant();

        using std::swap;
        swap(m_data.m_type, other.m_data.m_type);
        swap(m_data.m_value, other.m_data.m_value);

#if JSON_DIAGNOSTIC_POSITIONS
        swap(start_position, other.start_position);
        swap(end_position, other.end_position);
#endif

        json_base_class_t::operator=(std::move(other));

        set_parents();
        assert_invariant();
        return *this;
    }

    /// @brief destructor
    /// @sa https://json.nlohmann.me/api/basic_json/~basic_json/
    ~basic_json() noexcept
    {
        assert_invariant(false);
    }

    /// @}

  public:
    ///////////////////////
    // object inspection //
    ///////////////////////

    /// @name object inspection
    /// Functions to inspect the type of a JSON value.
    /// @{

    /// @brief serialization
    /// @sa https://json.nlohmann.me/api/basic_json/dump/
    string_t dump(const int indent = -1,
                  const char indent_char = ' ',
                  const bool ensure_ascii = false,
                  const error_handler_t error_handler = error_handler_t::strict) const
    {
        string_t result;
        serializer s(detail::output_adapter<char, string_t>(result), indent_char, error_handler);

        if (indent >= 0)
        {
            s.dump(*this, true, ensure_ascii, static_cast<unsigned int>(indent));
        }
        else
        {
            s.dump(*this, false, ensure_ascii, 0);
        }

        return result;
    }

    /// @brief return the type of the JSON value (explicit)
    /// @sa https://json.nlohmann.me/api/basic_json/type/
    constexpr value_t type() const noexcept
    {
        return m_data.m_type;
    }

    /// @brief return whether type is primitive
    /// @sa https://json.nlohmann.me/api/basic_json/is_primitive/
    constexpr bool is_primitive() const noexcept
    {
        return is_null() || is_string() || is_boolean() || is_number() || is_binary();
    }

    /// @brief return whether type is structured
    /// @sa https://json.nlohmann.me/api/basic_json/is_structured/
    constexpr bool is_structured() const noexcept
    {
        return is_array() || is_object();
    }

    /// @brief return whether value is null
    /// @sa https://json.nlohmann.me/api/basic_json/is_null/
    constexpr bool is_null() const noexcept
    {
        return m_data.m_type == value_t::null;
    }

    /// @brief return whether value is a boolean
    /// @sa https://json.nlohmann.me/api/basic_json/is_boolean/
    constexpr bool is_boolean() const noexcept
    {
        return m_data.m_type == value_t::boolean;
    }

    /// @brief return whether value is a number
    /// @sa https://json.nlohmann.me/api/basic_json/is_number/
    constexpr bool is_number() const noexcept
    {
        return is_number_integer() || is_number_float();
    }

    /// @brief return whether value is an integer number
    /// @sa https://json.nlohmann.me/api/basic_json/is_number_integer/
    constexpr bool is_number_integer() const noexcept
    {
        return m_data.m_type == value_t::number_integer || m_data.m_type == value_t::number_unsigned;
    }

    /// @brief return whether value is an unsigned integer number
    /// @sa https://json.nlohmann.me/api/basic_json/is_number_unsigned/
    constexpr bool is_number_unsigned() const noexcept
    {
        return m_data.m_type == value_t::number_unsigned;
    }

    /// @brief return whether value is a floating-point number
    /// @sa https://json.nlohmann.me/api/basic_json/is_number_float/
    constexpr bool is_number_float() const noexcept
    {
        return m_data.m_type == value_t::number_float;
    }

    /// @brief return whether value is an object
    /// @sa https://json.nlohmann.me/api/basic_json/is_object/
    constexpr bool is_object() const noexcept
    {
        return m_data.m_type == value_t::object;
    }

    /// @brief return whether value is an array
    /// @sa https://json.nlohmann.me/api/basic_json/is_array/
    constexpr bool is_array() const noexcept
    {
        return m_data.m_type == value_t::array;
    }

    /// @brief return whether value is a string
    /// @sa https://json.nlohmann.me/api/basic_json/is_string/
    constexpr bool is_string() const noexcept
    {
        return m_data.m_type == value_t::string;
    }

    /// @brief return whether value is a binary array
    /// @sa https://json.nlohmann.me/api/basic_json/is_binary/
    constexpr bool is_binary() const noexcept
    {
        return m_data.m_type == value_t::binary;
    }

    /// @brief return whether value is discarded
    /// @sa https://json.nlohmann.me/api/basic_json/is_discarded/
    constexpr bool is_discarded() const noexcept
    {
        return m_data.m_type == value_t::discarded;
    }

    /// @brief return the type of the JSON value (implicit)
    /// @sa https://json.nlohmann.me/api/basic_json/operator_value_t/
    constexpr operator value_t() const noexcept
    {
        return m_data.m_type;
    }

    /// @}

  private:
    //////////////////
    // value access //
    //////////////////

    /// get a boolean (explicit)
    boolean_t get_impl(boolean_t* /*unused*/) const
    {
        if (JSON_HEDLEY_LIKELY(is_boolean()))
        {
            return m_data.m_value.boolean;
        }

        JSON_THROW(type_error::create(302, detail::concat("type must be boolean, but is ", type_name()), this));
    }

    /// get a pointer to the value (object)
    object_t* get_impl_ptr(object_t* /*unused*/) noexcept
    {
        return is_object() ? m_data.m_value.object : nullptr;
    }

    /// get a pointer to the value (object)
    constexpr const object_t* get_impl_ptr(const object_t* /*unused*/) const noexcept
    {
        return is_object() ? m_data.m_value.object : nullptr;
    }

    /// get a pointer to the value (array)
    array_t* get_impl_ptr(array_t* /*unused*/) noexcept
    {
        return is_array() ? m_data.m_value.array : nullptr;
    }

    /// get a pointer to the value (array)
    constexpr const array_t* get_impl_ptr(const array_t* /*unused*/) const noexcept
    {
        return is_array() ? m_data.m_value.array : nullptr;
    }

    /// get a pointer to the value (string)
    string_t* get_impl_ptr(string_t* /*unused*/) noexcept
    {
        return is_string() ? m_data.m_value.string : nullptr;
    }

    /// get a pointer to the value (string)
    constexpr const string_t* get_impl_ptr(const string_t* /*unused*/) const noexcept
    {
        return is_string() ? m_data.m_value.string : nullptr;
    }

    /// get a pointer to the value (boolean)
    boolean_t* get_impl_ptr(boolean_t* /*unused*/) noexcept
    {
        return is_boolean() ? &m_data.m_value.boolean : nullptr;
    }

    /// get a pointer to the value (boolean)
    constexpr const boolean_t* get_impl_ptr(const boolean_t* /*unused*/) const noexcept
    {
        return is_boolean() ? &m_data.m_value.boolean : nullptr;
    }

    /// get a pointer to the value (integer number)
    number_integer_t* get_impl_ptr(number_integer_t* /*unused*/) noexcept
    {
        return m_data.m_type == value_t::number_integer ? &m_data.m_value.number_integer : nullptr;
    }

    /// get a pointer to the value (integer number)
    constexpr const number_integer_t* get_impl_ptr(const number_integer_t* /*unused*/) const noexcept
    {
        return m_data.m_type == value_t::number_integer ? &m_data.m_value.number_integer : nullptr;
    }

    /// get a pointer to the value (unsigned number)
    number_unsigned_t* get_impl_ptr(number_unsigned_t* /*unused*/) noexcept
    {
        return is_number_unsigned() ? &m_data.m_value.number_unsigned : nullptr;
    }

    /// get a pointer to the value (unsigned number)
    constexpr const number_unsigned_t* get_impl_ptr(const number_unsigned_t* /*unused*/) const noexcept
    {
        return is_number_unsigned() ? &m_data.m_value.number_unsigned : nullptr;
    }

    /// get a pointer to the value (floating-point number)
    number_float_t* get_impl_ptr(number_float_t* /*unused*/) noexcept
    {
        return is_number_float() ? &m_data.m_value.number_float : nullptr;
    }

    /// get a pointer to the value (floating-point number)
    constexpr const number_float_t* get_impl_ptr(const number_float_t* /*unused*/) const noexcept
    {
        return is_number_float() ? &m_data.m_value.number_float : nullptr;
    }

    /// get a pointer to the value (binary)
    binary_t* get_impl_ptr(binary_t* /*unused*/) noexcept
    {
        return is_binary() ? m_data.m_value.binary : nullptr;
    }

    /// get a pointer to the value (binary)
    constexpr const binary_t* get_impl_ptr(const binary_t* /*unused*/) const noexcept
    {
        return is_binary() ? m_data.m_value.binary : nullptr;
    }

    /*!
    @brief helper function to implement get_ref()

    This function helps to implement get_ref() without code duplication for
    const and non-const overloads

    @tparam ThisType will be deduced as `basic_json` or `const basic_json`

    @throw type_error.303 if ReferenceType does not match underlying value
    type of the current JSON
    */
    template<typename ReferenceType, typename ThisType>
    static ReferenceType get_ref_impl(ThisType& obj)
    {
        // delegate the call to get_ptr<>()
        auto* ptr = obj.template get_ptr<typename std::add_pointer<ReferenceType>::type>();

        if (JSON_HEDLEY_LIKELY(ptr != nullptr))
        {
            return *ptr;
        }

        JSON_THROW(type_error::create(303, detail::concat("incompatible ReferenceType for get_ref, actual type is ", obj.type_name()), &obj));
    }

  public:
    /// @name value access
    /// Direct access to the stored value of a JSON value.
    /// @{

    /// @brief get a pointer value (implicit)
    /// @sa https://json.nlohmann.me/api/basic_json/get_ptr/
    template<typename PointerType, typename std::enable_if<
                 std::is_pointer<PointerType>::value, int>::type = 0>
    auto get_ptr() noexcept -> decltype(std::declval<basic_json_t&>().get_impl_ptr(std::declval<PointerType>()))
    {
        // delegate the call to get_impl_ptr<>()
        return get_impl_ptr(static_cast<PointerType>(nullptr));
    }

    /// @brief get a pointer value (implicit)
    /// @sa https://json.nlohmann.me/api/basic_json/get_ptr/
    template < typename PointerType, typename std::enable_if <
                   std::is_pointer<PointerType>::value&&
                   std::is_const<typename std::remove_pointer<PointerType>::type>::value, int >::type = 0 >
    constexpr auto get_ptr() const noexcept -> decltype(std::declval<const basic_json_t&>().get_impl_ptr(std::declval<PointerType>()))
    {
        // delegate the call to get_impl_ptr<>() const
        return get_impl_ptr(static_cast<PointerType>(nullptr));
    }

  private:
    /*!
    @brief get a value (explicit)

    Explicit type conversion between the JSON value and a compatible value
    which is [CopyConstructible](https://en.cppreference.com/w/cpp/named_req/CopyConstructible)
    and [DefaultConstructible](https://en.cppreference.com/w/cpp/named_req/DefaultConstructible).
    The value is converted by calling the @ref json_serializer<ValueType>
    `from_json()` method.

    The function is equivalent to executing
    @code {.cpp}
    ValueType ret;
    JSONSerializer<ValueType>::from_json(*this, ret);
    return ret;
    @endcode

    This overloads is chosen if:
    - @a ValueType is not @ref basic_json,
    - @ref json_serializer<ValueType> has a `from_json()` method of the form
      `void from_json(const basic_json&, ValueType&)`, and
    - @ref json_serializer<ValueType> does not have a `from_json()` method of
      the form `ValueType from_json(const basic_json&)`

    @tparam ValueType the returned value type

    @return copy of the JSON value, converted to @a ValueType

    @throw what @ref json_serializer<ValueType> `from_json()` method throws

    @liveexample{The example below shows several conversions from JSON values
    to other types. There a few things to note: (1) Floating-point numbers can
    be converted to integers\, (2) A JSON array can be converted to a standard
    `std::vector<short>`\, (3) A JSON object can be converted to C++
    associative containers such as `std::unordered_map<std::string\,
    json>`.,get__ValueType_const}

    @since version 2.1.0
    */
    template < typename ValueType,
               detail::enable_if_t <
                   detail::is_default_constructible<ValueType>::value&&
                   detail::has_from_json<basic_json_t, ValueType>::value,
                   int > = 0 >
    ValueType get_impl(detail::priority_tag<0> /*unused*/) const noexcept(noexcept(
            JSONSerializer<ValueType>::from_json(std::declval<const basic_json_t&>(), std::declval<ValueType&>())))
    {
        auto ret = ValueType();
        JSONSerializer<ValueType>::from_json(*this, ret);
        return ret;
    }

    /*!
    @brief get a value (explicit); special case

    Explicit type conversion between the JSON value and a compatible value
    which is **not** [CopyConstructible](https://en.cppreference.com/w/cpp/named_req/CopyConstructible)
    and **not** [DefaultConstructible](https://en.cppreference.com/w/cpp/named_req/DefaultConstructible).
    The value is converted by calling the @ref json_serializer<ValueType>
    `from_json()` method.

    The function is equivalent to executing
    @code {.cpp}
    return JSONSerializer<ValueType>::from_json(*this);
    @endcode

    This overloads is chosen if:
    - @a ValueType is not @ref basic_json and
    - @ref json_serializer<ValueType> has a `from_json()` method of the form
      `ValueType from_json(const basic_json&)`

    @note If @ref json_serializer<ValueType> has both overloads of
    `from_json()`, this one is chosen.

    @tparam ValueType the returned value type

    @return copy of the JSON value, converted to @a ValueType

    @throw what @ref json_serializer<ValueType> `from_json()` method throws

    @since version 2.1.0
    */
    template < typename ValueType,
               detail::enable_if_t <
                   detail::has_non_default_from_json<basic_json_t, ValueType>::value,
                   int > = 0 >
    ValueType get_impl(detail::priority_tag<1> /*unused*/) const noexcept(noexcept(
            JSONSerializer<ValueType>::from_json(std::declval<const basic_json_t&>())))
    {
        return JSONSerializer<ValueType>::from_json(*this);
    }

    /*!
    @brief get special-case overload

    This overloads converts the current @ref basic_json in a different
    @ref basic_json type

    @tparam BasicJsonType == @ref basic_json

    @return a copy of *this, converted into @a BasicJsonType

    @complexity Depending on the implementation of the called `from_json()`
                method.

    @since version 3.2.0
    */
    template < typename BasicJsonType,
               detail::enable_if_t <
                   detail::is_basic_json<BasicJsonType>::value,
                   int > = 0 >
    BasicJsonType get_impl(detail::priority_tag<2> /*unused*/) const
    {
        return *this;
    }

    /*!
    @brief get special-case overload

    This overloads avoids a lot of template boilerplate, it can be seen as the
    identity method

    @tparam BasicJsonType == @ref basic_json

    @return a copy of *this

    @complexity Constant.

    @since version 2.1.0
    */
    template<typename BasicJsonType,
             detail::enable_if_t<
                 std::is_same<BasicJsonType, basic_json_t>::value,
                 int> = 0>
    basic_json get_impl(detail::priority_tag<3> /*unused*/) const
    {
        return *this;
    }

    /*!
    @brief get a pointer value (explicit)
    @copydoc get()
    */
    template<typename PointerType,
             detail::enable_if_t<
                 std::is_pointer<PointerType>::value,
                 int> = 0>
    constexpr auto get_impl(detail::priority_tag<4> /*unused*/) const noexcept
    -> decltype(std::declval<const basic_json_t&>().template get_ptr<PointerType>())
    {
        // delegate the call to get_ptr
        return get_ptr<PointerType>();
    }

  public:
    /*!
    @brief get a (pointer) value (explicit)

    Performs explicit type conversion between the JSON value and a compatible value if required.

    - If the requested type is a pointer to the internally stored JSON value that pointer is returned.
    No copies are made.

    - If the requested type is the current @ref basic_json, or a different @ref basic_json convertible
    from the current @ref basic_json.

    - Otherwise the value is converted by calling the @ref json_serializer<ValueType> `from_json()`
    method.

    @tparam ValueTypeCV the provided value type
    @tparam ValueType the returned value type

    @return copy of the JSON value, converted to @tparam ValueType if necessary

    @throw what @ref json_serializer<ValueType> `from_json()` method throws if conversion is required

    @since version 2.1.0
    */
    template < typename ValueTypeCV, typename ValueType = detail::uncvref_t<ValueTypeCV>>
#if defined(JSON_HAS_CPP_14)
    constexpr
#endif
    auto get() const noexcept(
    noexcept(std::declval<const basic_json_t&>().template get_impl<ValueType>(detail::priority_tag<4> {})))
    -> decltype(std::declval<const basic_json_t&>().template get_impl<ValueType>(detail::priority_tag<4> {}))
    {
        // we cannot static_assert on ValueTypeCV being non-const, because
        // there is support for get<const basic_json_t>(), which is why we
        // still need the uncvref
        static_assert(!std::is_reference<ValueTypeCV>::value,
                      "get() cannot be used with reference types, you might want to use get_ref()");
        return get_impl<ValueType>(detail::priority_tag<4> {});
    }

    /*!
    @brief get a pointer value (explicit)

    Explicit pointer access to the internally stored JSON value. No copies are
    made.

    @warning The pointer becomes invalid if the underlying JSON object
    changes.

    @tparam PointerType pointer type; must be a pointer to @ref array_t, @ref
    object_t, @ref string_t, @ref boolean_t, @ref number_integer_t,
    @ref number_unsigned_t, or @ref number_float_t.

    @return pointer to the internally stored JSON value if the requested
    pointer type @a PointerType fits to the JSON value; `nullptr` otherwise

    @complexity Constant.

    @liveexample{The example below shows how pointers to internal values of a
    JSON value can be requested. Note that no type conversions are made and a
    `nullptr` is returned if the value and the requested pointer type does not
    match.,get__PointerType}

    @sa see @ref get_ptr() for explicit pointer-member access

    @since version 1.0.0
    */
    template<typename PointerType, typename std::enable_if<
                 std::is_pointer<PointerType>::value, int>::type = 0>
    auto get() noexcept -> decltype(std::declval<basic_json_t&>().template get_ptr<PointerType>())
    {
        // delegate the call to get_ptr
        return get_ptr<PointerType>();
    }

    /// @brief get a value (explicit)
    /// @sa https://json.nlohmann.me/api/basic_json/get_to/
    template < typename ValueType,
               detail::enable_if_t <
                   !detail::is_basic_json<ValueType>::value&&
                   detail::has_from_json<basic_json_t, ValueType>::value,
                   int > = 0 >
    ValueType & get_to(ValueType& v) const noexcept(noexcept(
            JSONSerializer<ValueType>::from_json(std::declval<const basic_json_t&>(), v)))
    {
        JSONSerializer<ValueType>::from_json(*this, v);
        return v;
    }

    // specialization to allow calling get_to with a basic_json value
    // see https://github.com/nlohmann/json/issues/2175
    template<typename ValueType,
             detail::enable_if_t <
                 detail::is_basic_json<ValueType>::value,
                 int> = 0>
    ValueType & get_to(ValueType& v) const
    {
        v = *this;
        return v;
    }

    template <
        typename T, std::size_t N,
        typename Array = T (&)[N], // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays)
        detail::enable_if_t <
            detail::has_from_json<basic_json_t, Array>::value, int > = 0 >
    Array get_to(T (&v)[N]) const // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays)
    noexcept(noexcept(JSONSerializer<Array>::from_json(
                          std::declval<const basic_json_t&>(), v)))
    {
        JSONSerializer<Array>::from_json(*this, v);
        return v;
    }

    /// @brief get a reference value (implicit)
    /// @sa https://json.nlohmann.me/api/basic_json/get_ref/
    template<typename ReferenceType, typename std::enable_if<
                 std::is_reference<ReferenceType>::value, int>::type = 0>
    ReferenceType get_ref()
    {
        // delegate call to get_ref_impl
        return get_ref_impl<ReferenceType>(*this);
    }

    /// @brief get a reference value (implicit)
    /// @sa https://json.nlohmann.me/api/basic_json/get_ref/
    template < typename ReferenceType, typename std::enable_if <
                   std::is_reference<ReferenceType>::value&&
                   std::is_const<typename std::remove_reference<ReferenceType>::type>::value, int >::type = 0 >
    ReferenceType get_ref() const
    {
        // delegate call to get_ref_impl
        return get_ref_impl<ReferenceType>(*this);
    }

    /*!
    @brief get a value (implicit)

    Implicit type conversion between the JSON value and a compatible value.
    The call is realized by calling @ref get() const.

    @tparam ValueType non-pointer type compatible to the JSON value, for
    instance `int` for JSON integer numbers, `bool` for JSON booleans, or
    `std::vector` types for JSON arrays. The character type of @ref string_t
    as well as an initializer list of this type is excluded to avoid
    ambiguities as these types implicitly convert to `std::string`.

    @return copy of the JSON value, converted to type @a ValueType

    @throw type_error.302 in case passed type @a ValueType is incompatible
    to the JSON value type (e.g., the JSON value is of type boolean, but a
    string is requested); see example below

    @complexity Linear in the size of the JSON value.

    @liveexample{The example below shows several conversions from JSON values
    to other types. There a few things to note: (1) Floating-point numbers can
    be converted to integers\, (2) A JSON array can be converted to a standard
    `std::vector<short>`\, (3) A JSON object can be converted to C++
    associative containers such as `std::unordered_map<std::string\,
    json>`.,operator__ValueType}

    @since version 1.0.0
    */
    template < typename ValueType, typename std::enable_if <
                   detail::conjunction <
                       detail::negation<std::is_pointer<ValueType>>,
                       detail::negation<std::is_same<ValueType, std::nullptr_t>>,
                       detail::negation<std::is_same<ValueType, detail::json_ref<basic_json>>>,
                                        detail::negation<std::is_same<ValueType, typename string_t::value_type>>,
                                        detail::negation<detail::is_basic_json<ValueType>>,
                                        detail::negation<std::is_same<ValueType, std::initializer_list<typename string_t::value_type>>>,
#if defined(JSON_HAS_CPP_17) && (defined(__GNUC__) || (defined(_MSC_VER) && _MSC_VER >= 1910 && _MSC_VER <= 1914))
                                                detail::negation<std::is_same<ValueType, std::string_view>>,
#endif
#if defined(JSON_HAS_CPP_17) && JSON_HAS_STATIC_RTTI
                                                detail::negation<std::is_same<ValueType, std::any>>,
#endif
                                                detail::is_detected_lazy<detail::get_template_function, const basic_json_t&, ValueType>
                                                >::value, int >::type = 0 >
                                        JSON_EXPLICIT operator ValueType() const
    {
        // delegate the call to get<>() const
        return get<ValueType>();
    }

    /// @brief get a binary value
    /// @sa https://json.nlohmann.me/api/basic_json/get_binary/
    binary_t& get_binary()
    {
        if (!is_binary())
        {
            JSON_THROW(type_error::create(302, detail::concat("type must be binary, but is ", type_name()), this));
        }

        return *get_ptr<binary_t*>();
    }

    /// @brief get a binary value
    /// @sa https://json.nlohmann.me/api/basic_json/get_binary/
    const binary_t& get_binary() const
    {
        if (!is_binary())
        {
            JSON_THROW(type_error::create(302, detail::concat("type must be binary, but is ", type_name()), this));
        }

        return *get_ptr<const binary_t*>();
    }

    /// @}

    ////////////////////
    // element access //
    ////////////////////

    /// @name element access
    /// Access to the JSON value.
    /// @{

    /// @brief access specified array element with bounds checking
    /// @sa https://json.nlohmann.me/api/basic_json/at/
    reference at(size_type idx)
    {
        // at only works for arrays
        if (JSON_HEDLEY_LIKELY(is_array()))
        {
            JSON_TRY
            {
                return set_parent(m_data.m_value.array->at(idx));
            }
            JSON_CATCH (std::out_of_range&)
            {
                // create better exception explanation
                JSON_THROW(out_of_range::create(401, detail::concat("array index ", std::to_string(idx), " is out of range"), this));
            } // cppcheck-suppress[missingReturn]
        }
        else
        {
            JSON_THROW(type_error::create(304, detail::concat("cannot use at() with ", type_name()), this));
        }
    }

    /// @brief access specified array element with bounds checking
    /// @sa https://json.nlohmann.me/api/basic_json/at/
    const_reference at(size_type idx) const
    {
        // at only works for arrays
        if (JSON_HEDLEY_LIKELY(is_array()))
        {
            JSON_TRY
            {
                return m_data.m_value.array->at(idx);
            }
            JSON_CATCH (std::out_of_range&)
            {
                // create better exception explanation
                JSON_THROW(out_of_range::create(401, detail::concat("array index ", std::to_string(idx), " is out of range"), this));
            } // cppcheck-suppress[missingReturn]
        }
        else
        {
            JSON_THROW(type_error::create(304, detail::concat("cannot use at() with ", type_name()), this));
        }
    }

    /// @brief access specified object element with bounds checking
    /// @sa https://json.nlohmann.me/api/basic_json/at/
    reference at(const typename object_t::key_type& key)
    {
        // at only works for objects
        if (JSON_HEDLEY_UNLIKELY(!is_object()))
        {
            JSON_THROW(type_error::create(304, detail::concat("cannot use at() with ", type_name()), this));
        }

        auto it = m_data.m_value.object->find(key);
        if (it == m_data.m_value.object->end())
        {
            JSON_THROW(out_of_range::create(403, detail::concat("key '", key, "' not found"), this));
        }
        return set_parent(it->second);
    }

    /// @brief access specified object element with bounds checking
    /// @sa https://json.nlohmann.me/api/basic_json/at/
    template<class KeyType, detail::enable_if_t<
                 detail::is_usable_as_basic_json_key_type<basic_json_t, KeyType>::value, int> = 0>
    reference at(KeyType && key)
    {
        // at only works for objects
        if (JSON_HEDLEY_UNLIKELY(!is_object()))
        {
            JSON_THROW(type_error::create(304, detail::concat("cannot use at() with ", type_name()), this));
        }

        auto it = m_data.m_value.object->find(std::forward<KeyType>(key));
        if (it == m_data.m_value.object->end())
        {
            JSON_THROW(out_of_range::create(403, detail::concat("key '", string_t(std::forward<KeyType>(key)), "' not found"), this));
        }
        return set_parent(it->second);
    }

    /// @brief access specified object element with bounds checking
    /// @sa https://json.nlohmann.me/api/basic_json/at/
    const_reference at(const typename object_t::key_type& key) const
    {
        // at only works for objects
        if (JSON_HEDLEY_UNLIKELY(!is_object()))
        {
            JSON_THROW(type_error::create(304, detail::concat("cannot use at() with ", type_name()), this));
        }

        auto it = m_data.m_value.object->find(key);
        if (it == m_data.m_value.object->end())
        {
            JSON_THROW(out_of_range::create(403, detail::concat("key '", key, "' not found"), this));
        }
        return it->second;
    }

    /// @brief access specified object element with bounds checking
    /// @sa https://json.nlohmann.me/api/basic_json/at/
    template<class KeyType, detail::enable_if_t<
                 detail::is_usable_as_basic_json_key_type<basic_json_t, KeyType>::value, int> = 0>
    const_reference at(KeyType && key) const
    {
        // at only works for objects
        if (JSON_HEDLEY_UNLIKELY(!is_object()))
        {
            JSON_THROW(type_error::create(304, detail::concat("cannot use at() with ", type_name()), this));
        }

        auto it = m_data.m_value.object->find(std::forward<KeyType>(key));
        if (it == m_data.m_value.object->end())
        {
            JSON_THROW(out_of_range::create(403, detail::concat("key '", string_t(std::forward<KeyType>(key)), "' not found"), this));
        }
        return it->second;
    }

    /// @brief access specified array element
    /// @sa https://json.nlohmann.me/api/basic_json/operator%5B%5D/
    reference operator[](size_type idx)
    {
        // implicitly convert null value to an empty array
        if (is_null())
        {
            m_data.m_type = value_t::array;
            m_data.m_value.array = create<array_t>();
            assert_invariant();
        }

        // operator[] only works for arrays
        if (JSON_HEDLEY_LIKELY(is_array()))
        {
            // fill up array with null values if given idx is outside range
            if (idx >= m_data.m_value.array->size())
            {
#if JSON_DIAGNOSTICS
                // remember array size & capacity before resizing
                const auto old_size = m_data.m_value.array->size();
                const auto old_capacity = m_data.m_value.array->capacity();
#endif
                m_data.m_value.array->resize(idx + 1);

#if JSON_DIAGNOSTICS
                if (JSON_HEDLEY_UNLIKELY(m_data.m_value.array->capacity() != old_capacity))
                {
                    // capacity has changed: update all parents
                    set_parents();
                }
                else
                {
                    // set parent for values added above
                    set_parents(begin() + static_cast<typename iterator::difference_type>(old_size), static_cast<typename iterator::difference_type>(idx + 1 - old_size));
                }
#endif
                assert_invariant();
            }

            return m_data.m_value.array->operator[](idx);
        }

        JSON_THROW(type_error::create(305, detail::concat("cannot use operator[] with a numeric argument with ", type_name()), this));
    }

    /// @brief access specified array element
    /// @sa https://json.nlohmann.me/api/basic_json/operator%5B%5D/
    const_reference operator[](size_type idx) const
    {
        // const operator[] only works for arrays
        if (JSON_HEDLEY_LIKELY(is_array()))
        {
            return m_data.m_value.array->operator[](idx);
        }

        JSON_THROW(type_error::create(305, detail::concat("cannot use operator[] with a numeric argument with ", type_name()), this));
    }

    /// @brief access specified object element
    /// @sa https://json.nlohmann.me/api/basic_json/operator%5B%5D/
    reference operator[](typename object_t::key_type key) // NOLINT(performance-unnecessary-value-param)
    {
        // implicitly convert null value to an empty object
        if (is_null())
        {
            m_data.m_type = value_t::object;
            m_data.m_value.object = create<object_t>();
            assert_invariant();
        }

        // operator[] only works for objects
        if (JSON_HEDLEY_LIKELY(is_object()))
        {
            auto result = m_data.m_value.object->emplace(std::move(key), nullptr);
            return set_parent(result.first->second);
        }

        JSON_THROW(type_error::create(305, detail::concat("cannot use operator[] with a string argument with ", type_name()), this));
    }

    /// @brief access specified object element
    /// @sa https://json.nlohmann.me/api/basic_json/operator%5B%5D/
    const_reference operator[](const typename object_t::key_type& key) const
    {
        // const operator[] only works for objects
        if (JSON_HEDLEY_LIKELY(is_object()))
        {
            auto it = m_data.m_value.object->find(key);
            JSON_ASSERT(it != m_data.m_value.object->end());
            return it->second;
        }

        JSON_THROW(type_error::create(305, detail::concat("cannot use operator[] with a string argument with ", type_name()), this));
    }

    // these two functions resolve a (const) char * ambiguity affecting Clang and MSVC
    // (they seemingly cannot be constrained to resolve the ambiguity)
    template<typename T>
    reference operator[](T* key)
    {
        return operator[](typename object_t::key_type(key));
    }

    template<typename T>
    const_reference operator[](T* key) const
    {
        return operator[](typename object_t::key_type(key));
    }

    /// @brief access specified object element
    /// @sa https://json.nlohmann.me/api/basic_json/operator%5B%5D/
    template<class KeyType, detail::enable_if_t<
                 detail::is_usable_as_basic_json_key_type<basic_json_t, KeyType>::value, int > = 0 >
    reference operator[](KeyType && key)
    {
        // implicitly convert null value to an empty object
        if (is_null())
        {
            m_data.m_type = value_t::object;
            m_data.m_value.object = create<object_t>();
            assert_invariant();
        }

        // operator[] only works for objects
        if (JSON_HEDLEY_LIKELY(is_object()))
        {
            auto result = m_data.m_value.object->emplace(std::forward<KeyType>(key), nullptr);
            return set_parent(result.first->second);
        }

        JSON_THROW(type_error::create(305, detail::concat("cannot use operator[] with a string argument with ", type_name()), this));
    }

    /// @brief access specified object element
    /// @sa https://json.nlohmann.me/api/basic_json/operator%5B%5D/
    template<class KeyType, detail::enable_if_t<
                 detail::is_usable_as_basic_json_key_type<basic_json_t, KeyType>::value, int > = 0 >
    const_reference operator[](KeyType && key) const
    {
        // const operator[] only works for objects
        if (JSON_HEDLEY_LIKELY(is_object()))
        {
            auto it = m_data.m_value.object->find(std::forward<KeyType>(key));
            JSON_ASSERT(it != m_data.m_value.object->end());
            return it->second;
        }

        JSON_THROW(type_error::create(305, detail::concat("cannot use operator[] with a string argument with ", type_name()), this));
    }

  private:
    template<typename KeyType>
    using is_comparable_with_object_key = detail::is_comparable <
        object_comparator_t, const typename object_t::key_type&, KeyType >;

    template<typename ValueType>
    using value_return_type = std::conditional <
        detail::is_c_string_uncvref<ValueType>::value,
        string_t, typename std::decay<ValueType>::type >;

  public:
    /// @brief access specified object element with default value
    /// @sa https://json.nlohmann.me/api/basic_json/value/
    template < class ValueType, detail::enable_if_t <
                   !detail::is_transparent<object_comparator_t>::value
                   && detail::is_getable<basic_json_t, ValueType>::value
                   && !std::is_same<value_t, detail::uncvref_t<ValueType>>::value, int > = 0 >
    ValueType value(const typename object_t::key_type& key, const ValueType& default_value) const
    {
        // value only works for objects
        if (JSON_HEDLEY_LIKELY(is_object()))
        {
            // if key is found, return value and given default value otherwise
            const auto it = find(key);
            if (it != end())
            {
                return it->template get<ValueType>();
            }

            return default_value;
        }

        JSON_THROW(type_error::create(306, detail::concat("cannot use value() with ", type_name()), this));
    }

    /// @brief access specified object element with default value
    /// @sa https://json.nlohmann.me/api/basic_json/value/
    template < class ValueType, class ReturnType = typename value_return_type<ValueType>::type,
               detail::enable_if_t <
                   !detail::is_transparent<object_comparator_t>::value
                   && detail::is_getable<basic_json_t, ReturnType>::value
                   && !std::is_same<value_t, detail::uncvref_t<ValueType>>::value, int > = 0 >
    ReturnType value(const typename object_t::key_type& key, ValueType && default_value) const
    {
        // value only works for objects
        if (JSON_HEDLEY_LIKELY(is_object()))
        {
            // if key is found, return value and given default value otherwise
            const auto it = find(key);
            if (it != end())
            {
                return it->template get<ReturnType>();
            }

            return std::forward<ValueType>(default_value);
        }

        JSON_THROW(type_error::create(306, detail::concat("cannot use value() with ", type_name()), this));
    }

    /// @brief access specified object element with default value
    /// @sa https://json.nlohmann.me/api/basic_json/value/
    template < class ValueType, class KeyType, detail::enable_if_t <
                   detail::is_transparent<object_comparator_t>::value
                   && !detail::is_json_pointer<KeyType>::value
                   && is_comparable_with_object_key<KeyType>::value
                   && detail::is_getable<basic_json_t, ValueType>::value
                   && !std::is_same<value_t, detail::uncvref_t<ValueType>>::value, int > = 0 >
    ValueType value(KeyType && key, const ValueType& default_value) const
    {
        // value only works for objects
        if (JSON_HEDLEY_LIKELY(is_object()))
        {
            // if key is found, return value and given default value otherwise
            const auto it = find(std::forward<KeyType>(key));
            if (it != end())
            {
                return it->template get<ValueType>();
            }

            return default_value;
        }

        JSON_THROW(type_error::create(306, detail::concat("cannot use value() with ", type_name()), this));
    }

    /// @brief access specified object element via JSON Pointer with default value
    /// @sa https://json.nlohmann.me/api/basic_json/value/
    template < class ValueType, class KeyType, class ReturnType = typename value_return_type<ValueType>::type,
               detail::enable_if_t <
                   detail::is_transparent<object_comparator_t>::value
                   && !detail::is_json_pointer<KeyType>::value
                   && is_comparable_with_object_key<KeyType>::value
                   && detail::is_getable<basic_json_t, ReturnType>::value
                   && !std::is_same<value_t, detail::uncvref_t<ValueType>>::value, int > = 0 >
    ReturnType value(KeyType && key, ValueType && default_value) const
    {
        // value only works for objects
        if (JSON_HEDLEY_LIKELY(is_object()))
        {
            // if key is found, return value and given default value otherwise
            const auto it = find(std::forward<KeyType>(key));
            if (it != end())
            {
                return it->template get<ReturnType>();
            }

            return std::forward<ValueType>(default_value);
        }

        JSON_THROW(type_error::create(306, detail::concat("cannot use value() with ", type_name()), this));
    }

    /// @brief access specified object element via JSON Pointer with default value
    /// @sa https://json.nlohmann.me/api/basic_json/value/
    template < class ValueType, detail::enable_if_t <
                   detail::is_getable<basic_json_t, ValueType>::value
                   && !std::is_same<value_t, detail::uncvref_t<ValueType>>::value, int > = 0 >
    ValueType value(const json_pointer& ptr, const ValueType& default_value) const
    {
        // value only works for objects
        if (JSON_HEDLEY_LIKELY(is_object()))
        {
            // if pointer resolves a value, return it or use default value
            JSON_TRY
            {
                return ptr.get_checked(this).template get<ValueType>();
            }
            JSON_INTERNAL_CATCH (out_of_range&)
            {
                return default_value;
            }
        }

        JSON_THROW(type_error::create(306, detail::concat("cannot use value() with ", type_name()), this));
    }

    /// @brief access specified object element via JSON Pointer with default value
    /// @sa https://json.nlohmann.me/api/basic_json/value/
    template < class ValueType, class ReturnType = typename value_return_type<ValueType>::type,
               detail::enable_if_t <
                   detail::is_getable<basic_json_t, ReturnType>::value
                   && !std::is_same<value_t, detail::uncvref_t<ValueType>>::value, int > = 0 >
    ReturnType value(const json_pointer& ptr, ValueType && default_value) const
    {
        // value only works for objects
        if (JSON_HEDLEY_LIKELY(is_object()))
        {
            // if pointer resolves a value, return it or use default value
            JSON_TRY
            {
                return ptr.get_checked(this).template get<ReturnType>();
            }
            JSON_INTERNAL_CATCH (out_of_range&)
            {
                return std::forward<ValueType>(default_value);
            }
        }

        JSON_THROW(type_error::create(306, detail::concat("cannot use value() with ", type_name()), this));
    }

    template < class ValueType, class BasicJsonType, detail::enable_if_t <
                   detail::is_basic_json<BasicJsonType>::value
                   && detail::is_getable<basic_json_t, ValueType>::value
                   && !std::is_same<value_t, detail::uncvref_t<ValueType>>::value, int > = 0 >
    JSON_HEDLEY_DEPRECATED_FOR(3.11.0, basic_json::json_pointer or nlohmann::json_pointer<basic_json::string_t>) // NOLINT(readability/alt_tokens)
    ValueType value(const ::nlohmann::json_pointer<BasicJsonType>& ptr, const ValueType& default_value) const
    {
        return value(ptr.convert(), default_value);
    }

    template < class ValueType, class BasicJsonType, class ReturnType = typename value_return_type<ValueType>::type,
               detail::enable_if_t <
                   detail::is_basic_json<BasicJsonType>::value
                   && detail::is_getable<basic_json_t, ReturnType>::value
                   && !std::is_same<value_t, detail::uncvref_t<ValueType>>::value, int > = 0 >
    JSON_HEDLEY_DEPRECATED_FOR(3.11.0, basic_json::json_pointer or nlohmann::json_pointer<basic_json::string_t>) // NOLINT(readability/alt_tokens)
    ReturnType value(const ::nlohmann::json_pointer<BasicJsonType>& ptr, ValueType && default_value) const
    {
        return value(ptr.convert(), std::forward<ValueType>(default_value));
    }

    /// @brief access the first element
    /// @sa https://json.nlohmann.me/api/basic_json/front/
    reference front()
    {
        return *begin();
    }

    /// @brief access the first element
    /// @sa https://json.nlohmann.me/api/basic_json/front/
    const_reference front() const
    {
        return *cbegin();
    }

    /// @brief access the last element
    /// @sa https://json.nlohmann.me/api/basic_json/back/
    reference back()
    {
        auto tmp = end();
        --tmp;
        return *tmp;
    }

    /// @brief access the last element
    /// @sa https://json.nlohmann.me/api/basic_json/back/
    const_reference back() const
    {
        auto tmp = cend();
        --tmp;
        return *tmp;
    }

    /// @brief remove element given an iterator
    /// @sa https://json.nlohmann.me/api/basic_json/erase/
    template < class IteratorType, detail::enable_if_t <
                   std::is_same<IteratorType, typename basic_json_t::iterator>::value ||
                   std::is_same<IteratorType, typename basic_json_t::const_iterator>::value, int > = 0 >
    IteratorType erase(IteratorType pos) // NOLINT(performance-unnecessary-value-param)
    {
        // make sure iterator fits the current value
        if (JSON_HEDLEY_UNLIKELY(this != pos.m_object))
        {
            JSON_THROW(invalid_iterator::create(202, "iterator does not fit current value", this));
        }

        IteratorType result = end();

        switch (m_data.m_type)
        {
            case value_t::boolean:
            case value_t::number_float:
            case value_t::number_integer:
            case value_t::number_unsigned:
            case value_t::string:
            case value_t::binary:
            {
                if (JSON_HEDLEY_UNLIKELY(!pos.m_it.primitive_iterator.is_begin()))
                {
                    JSON_THROW(invalid_iterator::create(205, "iterator out of range", this));
                }

                if (is_string())
                {
                    AllocatorType<string_t> alloc;
                    std::allocator_traits<decltype(alloc)>::destroy(alloc, m_data.m_value.string);
                    std::allocator_traits<decltype(alloc)>::deallocate(alloc, m_data.m_value.string, 1);
                    m_data.m_value.string = nullptr;
                }
                else if (is_binary())
                {
                    AllocatorType<binary_t> alloc;
                    std::allocator_traits<decltype(alloc)>::destroy(alloc, m_data.m_value.binary);
                    std::allocator_traits<decltype(alloc)>::deallocate(alloc, m_data.m_value.binary, 1);
                    m_data.m_value.binary = nullptr;
                }

                m_data.m_type = value_t::null;
                assert_invariant();
                break;
            }

            case value_t::object:
            {
                result.m_it.object_iterator = m_data.m_value.object->erase(pos.m_it.object_iterator);
                break;
            }

            case value_t::array:
            {
                result.m_it.array_iterator = m_data.m_value.array->erase(pos.m_it.array_iterator);
                break;
            }

            case value_t::null:
            case value_t::discarded:
            default:
                JSON_THROW(type_error::create(307, detail::concat("cannot use erase() with ", type_name()), this));
        }

        return result;
    }

    /// @brief remove elements given an iterator range
    /// @sa https://json.nlohmann.me/api/basic_json/erase/
    template < class IteratorType, detail::enable_if_t <
                   std::is_same<IteratorType, typename basic_json_t::iterator>::value ||
                   std::is_same<IteratorType, typename basic_json_t::const_iterator>::value, int > = 0 >
    IteratorType erase(IteratorType first, IteratorType last) // NOLINT(performance-unnecessary-value-param)
    {
        // make sure iterator fits the current value
        if (JSON_HEDLEY_UNLIKELY(this != first.m_object || this != last.m_object))
        {
            JSON_THROW(invalid_iterator::create(203, "iterators do not fit current value", this));
        }

        IteratorType result = end();

        switch (m_data.m_type)
        {
            case value_t::boolean:
            case value_t::number_float:
            case value_t::number_integer:
            case value_t::number_unsigned:
            case value_t::string:
            case value_t::binary:
            {
                if (JSON_HEDLEY_LIKELY(!first.m_it.primitive_iterator.is_begin()
                                       || !last.m_it.primitive_iterator.is_end()))
                {
                    JSON_THROW(invalid_iterator::create(204, "iterators out of range", this));
                }

                if (is_string())
                {
                    AllocatorType<string_t> alloc;
                    std::allocator_traits<decltype(alloc)>::destroy(alloc, m_data.m_value.string);
                    std::allocator_traits<decltype(alloc)>::deallocate(alloc, m_data.m_value.string, 1);
                    m_data.m_value.string = nullptr;
                }
                else if (is_binary())
                {
                    AllocatorType<binary_t> alloc;
                    std::allocator_traits<decltype(alloc)>::destroy(alloc, m_data.m_value.binary);
                    std::allocator_traits<decltype(alloc)>::deallocate(alloc, m_data.m_value.binary, 1);
                    m_data.m_value.binary = nullptr;
                }

                m_data.m_type = value_t::null;
                assert_invariant();
                break;
            }

            case value_t::object:
            {
                result.m_it.object_iterator = m_data.m_value.object->erase(first.m_it.object_iterator,
                                              last.m_it.object_iterator);
                break;
            }

            case value_t::array:
            {
                result.m_it.array_iterator = m_data.m_value.array->erase(first.m_it.array_iterator,
                                             last.m_it.array_iterator);
                break;
            }

            case value_t::null:
            case value_t::discarded:
            default:
                JSON_THROW(type_error::create(307, detail::concat("cannot use erase() with ", type_name()), this));
        }

        return result;
    }

  private:
    template < typename KeyType, detail::enable_if_t <
                   detail::has_erase_with_key_type<basic_json_t, KeyType>::value, int > = 0 >
    size_type erase_internal(KeyType && key)
    {
        // this erase only works for objects
        if (JSON_HEDLEY_UNLIKELY(!is_object()))
        {
            JSON_THROW(type_error::create(307, detail::concat("cannot use erase() with ", type_name()), this));
        }

        return m_data.m_value.object->erase(std::forward<KeyType>(key));
    }

    template < typename KeyType, detail::enable_if_t <
                   !detail::has_erase_with_key_type<basic_json_t, KeyType>::value, int > = 0 >
    size_type erase_internal(KeyType && key)
    {
        // this erase only works for objects
        if (JSON_HEDLEY_UNLIKELY(!is_object()))
        {
            JSON_THROW(type_error::create(307, detail::concat("cannot use erase() with ", type_name()), this));
        }

        const auto it = m_data.m_value.object->find(std::forward<KeyType>(key));
        if (it != m_data.m_value.object->end())
        {
            m_data.m_value.object->erase(it);
            return 1;
        }
        return 0;
    }

  public:

    /// @brief remove element from a JSON object given a key
    /// @sa https://json.nlohmann.me/api/basic_json/erase/
    size_type erase(const typename object_t::key_type& key)
    {
        // the indirection via erase_internal() is added to avoid making this
        // function a template and thus de-rank it during overload resolution
        return erase_internal(key);
    }

    /// @brief remove element from a JSON object given a key
    /// @sa https://json.nlohmann.me/api/basic_json/erase/
    template<class KeyType, detail::enable_if_t<
                 detail::is_usable_as_basic_json_key_type<basic_json_t, KeyType>::value, int> = 0>
    size_type erase(KeyType && key)
    {
        return erase_internal(std::forward<KeyType>(key));
    }

    /// @brief remove element from a JSON array given an index
    /// @sa https://json.nlohmann.me/api/basic_json/erase/
    void erase(const size_type idx)
    {
        // this erase only works for arrays
        if (JSON_HEDLEY_LIKELY(is_array()))
        {
            if (JSON_HEDLEY_UNLIKELY(idx >= size()))
            {
                JSON_THROW(out_of_range::create(401, detail::concat("array index ", std::to_string(idx), " is out of range"), this));
            }

            m_data.m_value.array->erase(m_data.m_value.array->begin() + static_cast<difference_type>(idx));
        }
        else
        {
            JSON_THROW(type_error::create(307, detail::concat("cannot use erase() with ", type_name()), this));
        }
    }

    /// @}

    ////////////
    // lookup //
    ////////////

    /// @name lookup
    /// @{

    /// @brief find an element in a JSON object
    /// @sa https://json.nlohmann.me/api/basic_json/find/
    iterator find(const typename object_t::key_type& key)
    {
        auto result = end();

        if (is_object())
        {
            result.m_it.object_iterator = m_data.m_value.object->find(key);
        }

        return result;
    }

    /// @brief find an element in a JSON object
    /// @sa https://json.nlohmann.me/api/basic_json/find/
    const_iterator find(const typename object_t::key_type& key) const
    {
        auto result = cend();

        if (is_object())
        {
            result.m_it.object_iterator = m_data.m_value.object->find(key);
        }

        return result;
    }

    /// @brief find an element in a JSON object
    /// @sa https://json.nlohmann.me/api/basic_json/find/
    template<class KeyType, detail::enable_if_t<
                 detail::is_usable_as_basic_json_key_type<basic_json_t, KeyType>::value, int> = 0>
    iterator find(KeyType && key)
    {
        auto result = end();

        if (is_object())
        {
            result.m_it.object_iterator = m_data.m_value.object->find(std::forward<KeyType>(key));
        }

        return result;
    }

    /// @brief find an element in a JSON object
    /// @sa https://json.nlohmann.me/api/basic_json/find/
    template<class KeyType, detail::enable_if_t<
                 detail::is_usable_as_basic_json_key_type<basic_json_t, KeyType>::value, int> = 0>
    const_iterator find(KeyType && key) const
    {
        auto result = cend();

        if (is_object())
        {
            result.m_it.object_iterator = m_data.m_value.object->find(std::forward<KeyType>(key));
        }

        return result;
    }

    /// @brief returns the number of occurrences of a key in a JSON object
    /// @sa https://json.nlohmann.me/api/basic_json/count/
    size_type count(const typename object_t::key_type& key) const
    {
        // return 0 for all nonobject types
        return is_object() ? m_data.m_value.object->count(key) : 0;
    }

    /// @brief returns the number of occurrences of a key in a JSON object
    /// @sa https://json.nlohmann.me/api/basic_json/count/
    template<class KeyType, detail::enable_if_t<
                 detail::is_usable_as_basic_json_key_type<basic_json_t, KeyType>::value, int> = 0>
    size_type count(KeyType && key) const
    {
        // return 0 for all nonobject types
        return is_object() ? m_data.m_value.object->count(std::forward<KeyType>(key)) : 0;
    }

    /// @brief check the existence of an element in a JSON object
    /// @sa https://json.nlohmann.me/api/basic_json/contains/
    bool contains(const typename object_t::key_type& key) const
    {
        return is_object() && m_data.m_value.object->find(key) != m_data.m_value.object->end();
    }

    /// @brief check the existence of an element in a JSON object
    /// @sa https://json.nlohmann.me/api/basic_json/contains/
    template<class KeyType, detail::enable_if_t<
                 detail::is_usable_as_basic_json_key_type<basic_json_t, KeyType>::value, int> = 0>
    bool contains(KeyType && key) const
    {
        return is_object() && m_data.m_value.object->find(std::forward<KeyType>(key)) != m_data.m_value.object->end();
    }

    /// @brief check the existence of an element in a JSON object given a JSON pointer
    /// @sa https://json.nlohmann.me/api/basic_json/contains/
    bool contains(const json_pointer& ptr) const
    {
        return ptr.contains(this);
    }

    template<typename BasicJsonType, detail::enable_if_t<detail::is_basic_json<BasicJsonType>::value, int> = 0>
    JSON_HEDLEY_DEPRECATED_FOR(3.11.0, basic_json::json_pointer or nlohmann::json_pointer<basic_json::string_t>) // NOLINT(readability/alt_tokens)
    bool contains(const typename ::nlohmann::json_pointer<BasicJsonType>& ptr) const
    {
        return ptr.contains(this);
    }

    /// @}

    ///////////////
    // iterators //
    ///////////////

    /// @name iterators
    /// @{

    /// @brief returns an iterator to the first element
    /// @sa https://json.nlohmann.me/api/basic_json/begin/
    iterator begin() noexcept
    {
        iterator result(this);
        result.set_begin();
        return result;
    }

    /// @brief returns an iterator to the first element
    /// @sa https://json.nlohmann.me/api/basic_json/begin/
    const_iterator begin() const noexcept
    {
        return cbegin();
    }

    /// @brief returns a const iterator to the first element
    /// @sa https://json.nlohmann.me/api/basic_json/cbegin/
    const_iterator cbegin() const noexcept
    {
        const_iterator result(this);
        result.set_begin();
        return result;
    }

    /// @brief returns an iterator to one past the last element
    /// @sa https://json.nlohmann.me/api/basic_json/end/
    iterator end() noexcept
    {
        iterator result(this);
        result.set_end();
        return result;
    }

    /// @brief returns an iterator to one past the last element
    /// @sa https://json.nlohmann.me/api/basic_json/end/
    const_iterator end() const noexcept
    {
        return cend();
    }

    /// @brief returns an iterator to one past the last element
    /// @sa https://json.nlohmann.me/api/basic_json/cend/
    const_iterator cend() const noexcept
    {
        const_iterator result(this);
        result.set_end();
        return result;
    }

    /// @brief returns an iterator to the reverse-beginning
    /// @sa https://json.nlohmann.me/api/basic_json/rbegin/
    reverse_iterator rbegin() noexcept
    {
        return reverse_iterator(end());
    }

    /// @brief returns an iterator to the reverse-beginning
    /// @sa https://json.nlohmann.me/api/basic_json/rbegin/
    const_reverse_iterator rbegin() const noexcept
    {
        return crbegin();
    }

    /// @brief returns an iterator to the reverse-end
    /// @sa https://json.nlohmann.me/api/basic_json/rend/
    reverse_iterator rend() noexcept
    {
        return reverse_iterator(begin());
    }

    /// @brief returns an iterator to the reverse-end
    /// @sa https://json.nlohmann.me/api/basic_json/rend/
    const_reverse_iterator rend() const noexcept
    {
        return crend();
    }

    /// @brief returns a const reverse iterator to the last element
    /// @sa https://json.nlohmann.me/api/basic_json/crbegin/
    const_reverse_iterator crbegin() const noexcept
    {
        return const_reverse_iterator(cend());
    }

    /// @brief returns a const reverse iterator to one before the first
    /// @sa https://json.nlohmann.me/api/basic_json/crend/
    const_reverse_iterator crend() const noexcept
    {
        return const_reverse_iterator(cbegin());
    }

  public:
    /// @brief wrapper to access iterator member functions in range-based for
    /// @sa https://json.nlohmann.me/api/basic_json/items/
    /// @deprecated This function is deprecated since 3.1.0 and will be removed in
    ///             version 4.0.0 of the library. Please use @ref items() instead;
    ///             that is, replace `json::iterator_wrapper(j)` with `j.items()`.
    JSON_HEDLEY_DEPRECATED_FOR(3.1.0, items())
    static iteration_proxy<iterator> iterator_wrapper(reference ref) noexcept
    {
        return ref.items();
    }

    /// @brief wrapper to access iterator member functions in range-based for
    /// @sa https://json.nlohmann.me/api/basic_json/items/
    /// @deprecated This function is deprecated since 3.1.0 and will be removed in
    ///         version 4.0.0 of the library. Please use @ref items() instead;
    ///         that is, replace `json::iterator_wrapper(j)` with `j.items()`.
    JSON_HEDLEY_DEPRECATED_FOR(3.1.0, items())
    static iteration_proxy<const_iterator> iterator_wrapper(const_reference ref) noexcept
    {
        return ref.items();
    }

    /// @brief helper to access iterator member functions in range-based for
    /// @sa https://json.nlohmann.me/api/basic_json/items/
    iteration_proxy<iterator> items() noexcept
    {
        return iteration_proxy<iterator>(*this);
    }

    /// @brief helper to access iterator member functions in range-based for
    /// @sa https://json.nlohmann.me/api/basic_json/items/
    iteration_proxy<const_iterator> items() const noexcept
    {
        return iteration_proxy<const_iterator>(*this);
    }

    /// @}

    //////////////
    // capacity //
    //////////////

    /// @name capacity
    /// @{

    /// @brief checks whether the container is empty.
    /// @sa https://json.nlohmann.me/api/basic_json/empty/
    bool empty() const noexcept
    {
        switch (m_data.m_type)
        {
            case value_t::null:
            {
                // null values are empty
                return true;
            }

            case value_t::array:
            {
                // delegate call to array_t::empty()
                return m_data.m_value.array->empty();
            }

            case value_t::object:
            {
                // delegate call to object_t::empty()
                return m_data.m_value.object->empty();
            }

            case value_t::string:
            case value_t::boolean:
            case value_t::number_integer:
            case value_t::number_unsigned:
            case value_t::number_float:
            case value_t::binary:
            case value_t::discarded:
            default:
            {
                // all other types are nonempty
                return false;
            }
        }
    }

    /// @brief returns the number of elements
    /// @sa https://json.nlohmann.me/api/basic_json/size/
    size_type size() const noexcept
    {
        switch (m_data.m_type)
        {
            case value_t::null:
            {
                // null values are empty
                return 0;
            }

            case value_t::array:
            {
                // delegate call to array_t::size()
                return m_data.m_value.array->size();
            }

            case value_t::object:
            {
                // delegate call to object_t::size()
                return m_data.m_value.object->size();
            }

            case value_t::string:
            case value_t::boolean:
            case value_t::number_integer:
            case value_t::number_unsigned:
            case value_t::number_float:
            case value_t::binary:
            case value_t::discarded:
            default:
            {
                // all other types have size 1
                return 1;
            }
        }
    }

    /// @brief returns the maximum possible number of elements
    /// @sa https://json.nlohmann.me/api/basic_json/max_size/
    size_type max_size() const noexcept
    {
        switch (m_data.m_type)
        {
            case value_t::array:
            {
                // delegate call to array_t::max_size()
                return m_data.m_value.array->max_size();
            }

            case value_t::object:
            {
                // delegate call to object_t::max_size()
                return m_data.m_value.object->max_size();
            }

            case value_t::null:
            case value_t::string:
            case value_t::boolean:
            case value_t::number_integer:
            case value_t::number_unsigned:
            case value_t::number_float:
            case value_t::binary:
            case value_t::discarded:
            default:
            {
                // all other types have max_size() == size()
                return size();
            }
        }
    }

    /// @}

    ///////////////
    // modifiers //
    ///////////////

    /// @name modifiers
    /// @{

    /// @brief clears the contents
    /// @sa https://json.nlohmann.me/api/basic_json/clear/
    void clear() noexcept
    {
        switch (m_data.m_type)
        {
            case value_t::number_integer:
            {
                m_data.m_value.number_integer = 0;
                break;
            }

            case value_t::number_unsigned:
            {
                m_data.m_value.number_unsigned = 0;
                break;
            }

            case value_t::number_float:
            {
                m_data.m_value.number_float = 0.0;
                break;
            }

            case value_t::boolean:
            {
                m_data.m_value.boolean = false;
                break;
            }

            case value_t::string:
            {
                m_data.m_value.string->clear();
                break;
            }

            case value_t::binary:
            {
                m_data.m_value.binary->clear();
                break;
            }

            case value_t::array:
            {
                m_data.m_value.array->clear();
                break;
            }

            case value_t::object:
            {
                m_data.m_value.object->clear();
                break;
            }

            case value_t::null:
            case value_t::discarded:
            default:
                break;
        }
    }

    /// @brief add an object to an array
    /// @sa https://json.nlohmann.me/api/basic_json/push_back/
    void push_back(basic_json&& val)
    {
        // push_back only works for null objects or arrays
        if (JSON_HEDLEY_UNLIKELY(!(is_null() || is_array())))
        {
            JSON_THROW(type_error::create(308, detail::concat("cannot use push_back() with ", type_name()), this));
        }

        // transform null object into an array
        if (is_null())
        {
            m_data.m_type = value_t::array;
            m_data.m_value = value_t::array;
            assert_invariant();
        }

        // add element to array (move semantics)
        const auto old_capacity = m_data.m_value.array->capacity();
        m_data.m_value.array->push_back(std::move(val));
        set_parent(m_data.m_value.array->back(), old_capacity);
        // if val is moved from, basic_json move constructor marks it null, so we do not call the destructor
    }

    /// @brief add an object to an array
    /// @sa https://json.nlohmann.me/api/basic_json/operator+=/
    reference operator+=(basic_json&& val)
    {
        push_back(std::move(val));
        return *this;
    }

    /// @brief add an object to an array
    /// @sa https://json.nlohmann.me/api/basic_json/push_back/
    void push_back(const basic_json& val)
    {
        // push_back only works for null objects or arrays
        if (JSON_HEDLEY_UNLIKELY(!(is_null() || is_array())))
        {
            JSON_THROW(type_error::create(308, detail::concat("cannot use push_back() with ", type_name()), this));
        }

        // transform null object into an array
        if (is_null())
        {
            m_data.m_type = value_t::array;
            m_data.m_value = value_t::array;
            assert_invariant();
        }

        // add element to array
        const auto old_capacity = m_data.m_value.array->capacity();
        m_data.m_value.array->push_back(val);
        set_parent(m_data.m_value.array->back(), old_capacity);
    }

    /// @brief add an object to an array
    /// @sa https://json.nlohmann.me/api/basic_json/operator+=/
    reference operator+=(const basic_json& val)
    {
        push_back(val);
        return *this;
    }

    /// @brief add an object to an object
    /// @sa https://json.nlohmann.me/api/basic_json/push_back/
    void push_back(const typename object_t::value_type& val)
    {
        // push_back only works for null objects or objects
        if (JSON_HEDLEY_UNLIKELY(!(is_null() || is_object())))
        {
            JSON_THROW(type_error::create(308, detail::concat("cannot use push_back() with ", type_name()), this));
        }

        // transform null object into an object
        if (is_null())
        {
            m_data.m_type = value_t::object;
            m_data.m_value = value_t::object;
            assert_invariant();
        }

        // add element to object
        auto res = m_data.m_value.object->insert(val);
        set_parent(res.first->second);
    }

    /// @brief add an object to an object
    /// @sa https://json.nlohmann.me/api/basic_json/operator+=/
    reference operator+=(const typename object_t::value_type& val)
    {
        push_back(val);
        return *this;
    }

    /// @brief add an object to an object
    /// @sa https://json.nlohmann.me/api/basic_json/push_back/
    void push_back(initializer_list_t init)
    {
        if (is_object() && init.size() == 2 && (*init.begin())->is_string())
        {
            basic_json&& key = init.begin()->moved_or_copied();
            push_back(typename object_t::value_type(
                          std::move(key.get_ref<string_t&>()), (init.begin() + 1)->moved_or_copied()));
        }
        else
        {
            push_back(basic_json(init));
        }
    }

    /// @brief add an object to an object
    /// @sa https://json.nlohmann.me/api/basic_json/operator+=/
    reference operator+=(initializer_list_t init)
    {
        push_back(init);
        return *this;
    }

    /// @brief add an object to an array
    /// @sa https://json.nlohmann.me/api/basic_json/emplace_back/
    template<class... Args>
    reference emplace_back(Args&& ... args)
    {
        // emplace_back only works for null objects or arrays
        if (JSON_HEDLEY_UNLIKELY(!(is_null() || is_array())))
        {
            JSON_THROW(type_error::create(311, detail::concat("cannot use emplace_back() with ", type_name()), this));
        }

        // transform null object into an array
        if (is_null())
        {
            m_data.m_type = value_t::array;
            m_data.m_value = value_t::array;
            assert_invariant();
        }

        // add element to array (perfect forwarding)
        const auto old_capacity = m_data.m_value.array->capacity();
        m_data.m_value.array->emplace_back(std::forward<Args>(args)...);
        return set_parent(m_data.m_value.array->back(), old_capacity);
    }

    /// @brief add an object to an object if key does not exist
    /// @sa https://json.nlohmann.me/api/basic_json/emplace/
    template<class... Args>
    std::pair<iterator, bool> emplace(Args&& ... args)
    {
        // emplace only works for null objects or arrays
        if (JSON_HEDLEY_UNLIKELY(!(is_null() || is_object())))
        {
            JSON_THROW(type_error::create(311, detail::concat("cannot use emplace() with ", type_name()), this));
        }

        // transform null object into an object
        if (is_null())
        {
            m_data.m_type = value_t::object;
            m_data.m_value = value_t::object;
            assert_invariant();
        }

        // add element to array (perfect forwarding)
        auto res = m_data.m_value.object->emplace(std::forward<Args>(args)...);
        set_parent(res.first->second);

        // create result iterator and set iterator to the result of emplace
        auto it = begin();
        it.m_it.object_iterator = res.first;

        // return pair of iterator and boolean
        return {it, res.second};
    }

    /// Helper for insertion of an iterator
    /// @note: This uses std::distance to support GCC 4.8,
    ///        see https://github.com/nlohmann/json/pull/1257
    template<typename... Args>
    iterator insert_iterator(const_iterator pos, Args&& ... args) // NOLINT(performance-unnecessary-value-param)
    {
        iterator result(this);
        JSON_ASSERT(m_data.m_value.array != nullptr);

        auto insert_pos = std::distance(m_data.m_value.array->begin(), pos.m_it.array_iterator);
        m_data.m_value.array->insert(pos.m_it.array_iterator, std::forward<Args>(args)...);
        result.m_it.array_iterator = m_data.m_value.array->begin() + insert_pos;

        // This could have been written as:
        // result.m_it.array_iterator = m_data.m_value.array->insert(pos.m_it.array_iterator, cnt, val);
        // but the return value of insert is missing in GCC 4.8, so it is written this way instead.

        set_parents();
        return result;
    }

    /// @brief inserts element into array
    /// @sa https://json.nlohmann.me/api/basic_json/insert/
    iterator insert(const_iterator pos, const basic_json& val) // NOLINT(performance-unnecessary-value-param)
    {
        // insert only works for arrays
        if (JSON_HEDLEY_LIKELY(is_array()))
        {
            // check if iterator pos fits to this JSON value
            if (JSON_HEDLEY_UNLIKELY(pos.m_object != this))
            {
                JSON_THROW(invalid_iterator::create(202, "iterator does not fit current value", this));
            }

            // insert to array and return iterator
            return insert_iterator(pos, val);
        }

        JSON_THROW(type_error::create(309, detail::concat("cannot use insert() with ", type_name()), this));
    }

    /// @brief inserts element into array
    /// @sa https://json.nlohmann.me/api/basic_json/insert/
    iterator insert(const_iterator pos, basic_json&& val) // NOLINT(performance-unnecessary-value-param)
    {
        return insert(pos, val);
    }

    /// @brief inserts copies of element into array
    /// @sa https://json.nlohmann.me/api/basic_json/insert/
    iterator insert(const_iterator pos, size_type cnt, const basic_json& val) // NOLINT(performance-unnecessary-value-param)
    {
        // insert only works for arrays
        if (JSON_HEDLEY_LIKELY(is_array()))
        {
            // check if iterator pos fits to this JSON value
            if (JSON_HEDLEY_UNLIKELY(pos.m_object != this))
            {
                JSON_THROW(invalid_iterator::create(202, "iterator does not fit current value", this));
            }

            // insert to array and return iterator
            return insert_iterator(pos, cnt, val);
        }

        JSON_THROW(type_error::create(309, detail::concat("cannot use insert() with ", type_name()), this));
    }

    /// @brief inserts range of elements into array
    /// @sa https://json.nlohmann.me/api/basic_json/insert/
    iterator insert(const_iterator pos, const_iterator first, const_iterator last) // NOLINT(performance-unnecessary-value-param)
    {
        // insert only works for arrays
        if (JSON_HEDLEY_UNLIKELY(!is_array()))
        {
            JSON_THROW(type_error::create(309, detail::concat("cannot use insert() with ", type_name()), this));
        }

        // check if iterator pos fits to this JSON value
        if (JSON_HEDLEY_UNLIKELY(pos.m_object != this))
        {
            JSON_THROW(invalid_iterator::create(202, "iterator does not fit current value", this));
        }

        // check if range iterators belong to the same JSON object
        if (JSON_HEDLEY_UNLIKELY(first.m_object != last.m_object))
        {
            JSON_THROW(invalid_iterator::create(210, "iterators do not fit", this));
        }

        if (JSON_HEDLEY_UNLIKELY(first.m_object == this))
        {
            JSON_THROW(invalid_iterator::create(211, "passed iterators may not belong to container", this));
        }

        // insert to array and return iterator
        return insert_iterator(pos, first.m_it.array_iterator, last.m_it.array_iterator);
    }

    /// @brief inserts elements from initializer list into array
    /// @sa https://json.nlohmann.me/api/basic_json/insert/
    iterator insert(const_iterator pos, initializer_list_t ilist) // NOLINT(performance-unnecessary-value-param)
    {
        // insert only works for arrays
        if (JSON_HEDLEY_UNLIKELY(!is_array()))
        {
            JSON_THROW(type_error::create(309, detail::concat("cannot use insert() with ", type_name()), this));
        }

        // check if iterator pos fits to this JSON value
        if (JSON_HEDLEY_UNLIKELY(pos.m_object != this))
        {
            JSON_THROW(invalid_iterator::create(202, "iterator does not fit current value", this));
        }

        // insert to array and return iterator
        return insert_iterator(pos, ilist.begin(), ilist.end());
    }

    /// @brief inserts range of elements into object
    /// @sa https://json.nlohmann.me/api/basic_json/insert/
    void insert(const_iterator first, const_iterator last) // NOLINT(performance-unnecessary-value-param)
    {
        // insert only works for objects
        if (JSON_HEDLEY_UNLIKELY(!is_object()))
        {
            JSON_THROW(type_error::create(309, detail::concat("cannot use insert() with ", type_name()), this));
        }

        // check if range iterators belong to the same JSON object
        if (JSON_HEDLEY_UNLIKELY(first.m_object != last.m_object))
        {
            JSON_THROW(invalid_iterator::create(210, "iterators do not fit", this));
        }

        // passed iterators must belong to objects
        if (JSON_HEDLEY_UNLIKELY(!first.m_object->is_object()))
        {
            JSON_THROW(invalid_iterator::create(202, "iterators first and last must point to objects", this));
        }

        m_data.m_value.object->insert(first.m_it.object_iterator, last.m_it.object_iterator);
        set_parents();
    }

    /// @brief updates a JSON object from another object, overwriting existing keys
    /// @sa https://json.nlohmann.me/api/basic_json/update/
    void update(const_reference j, bool merge_objects = false)
    {
        update(j.begin(), j.end(), merge_objects);
    }

    /// @brief updates a JSON object from another object, overwriting existing keys
    /// @sa https://json.nlohmann.me/api/basic_json/update/
    void update(const_iterator first, const_iterator last, bool merge_objects = false) // NOLINT(performance-unnecessary-value-param)
    {
        // implicitly convert null value to an empty object
        if (is_null())
        {
            m_data.m_type = value_t::object;
            m_data.m_value.object = create<object_t>();
            assert_invariant();
        }

        if (JSON_HEDLEY_UNLIKELY(!is_object()))
        {
            JSON_THROW(type_error::create(312, detail::concat("cannot use update() with ", type_name()), this));
        }

        // check if range iterators belong to the same JSON object
        if (JSON_HEDLEY_UNLIKELY(first.m_object != last.m_object))
        {
            JSON_THROW(invalid_iterator::create(210, "iterators do not fit", this));
        }

        // passed iterators must belong to objects
        if (JSON_HEDLEY_UNLIKELY(!first.m_object->is_object()))
        {
            JSON_THROW(type_error::create(312, detail::concat("cannot use update() with ", first.m_object->type_name()), first.m_object));
        }

        for (auto it = first; it != last; ++it)
        {
            if (merge_objects && it.value().is_object())
            {
                auto it2 = m_data.m_value.object->find(it.key());
                if (it2 != m_data.m_value.object->end())
                {
                    it2->second.update(it.value(), true);
                    continue;
                }
            }
            m_data.m_value.object->operator[](it.key()) = it.value();
#if JSON_DIAGNOSTICS
            m_data.m_value.object->operator[](it.key()).m_parent = this;
#endif
        }
    }

    /// @brief exchanges the values
    /// @sa https://json.nlohmann.me/api/basic_json/swap/
    void swap(reference other) noexcept (
        std::is_nothrow_move_constructible<value_t>::value&&
        std::is_nothrow_move_assignable<value_t>::value&&
        std::is_nothrow_move_constructible<json_value>::value&& // NOLINT(cppcoreguidelines-noexcept-swap,performance-noexcept-swap)
        std::is_nothrow_move_assignable<json_value>::value
    )
    {
        std::swap(m_data.m_type, other.m_data.m_type);
        std::swap(m_data.m_value, other.m_data.m_value);

        set_parents();
        other.set_parents();
        assert_invariant();
    }

    /// @brief exchanges the values
    /// @sa https://json.nlohmann.me/api/basic_json/swap/
    friend void swap(reference left, reference right) noexcept (
        std::is_nothrow_move_constructible<value_t>::value&&
        std::is_nothrow_move_assignable<value_t>::value&&
        std::is_nothrow_move_constructible<json_value>::value&& // NOLINT(cppcoreguidelines-noexcept-swap,performance-noexcept-swap)
        std::is_nothrow_move_assignable<json_value>::value
    )
    {
        left.swap(right);
    }

    /// @brief exchanges the values
    /// @sa https://json.nlohmann.me/api/basic_json/swap/
    void swap(array_t& other) // NOLINT(bugprone-exception-escape,cppcoreguidelines-noexcept-swap,performance-noexcept-swap)
    {
        // swap only works for arrays
        if (JSON_HEDLEY_LIKELY(is_array()))
        {
            using std::swap;
            swap(*(m_data.m_value.array), other);
        }
        else
        {
            JSON_THROW(type_error::create(310, detail::concat("cannot use swap(array_t&) with ", type_name()), this));
        }
    }

    /// @brief exchanges the values
    /// @sa https://json.nlohmann.me/api/basic_json/swap/
    void swap(object_t& other) // NOLINT(bugprone-exception-escape,cppcoreguidelines-noexcept-swap,performance-noexcept-swap)
    {
        // swap only works for objects
        if (JSON_HEDLEY_LIKELY(is_object()))
        {
            using std::swap;
            swap(*(m_data.m_value.object), other);
        }
        else
        {
            JSON_THROW(type_error::create(310, detail::concat("cannot use swap(object_t&) with ", type_name()), this));
        }
    }

    /// @brief exchanges the values
    /// @sa https://json.nlohmann.me/api/basic_json/swap/
    void swap(string_t& other) // NOLINT(bugprone-exception-escape,cppcoreguidelines-noexcept-swap,performance-noexcept-swap)
    {
        // swap only works for strings
        if (JSON_HEDLEY_LIKELY(is_string()))
        {
            using std::swap;
            swap(*(m_data.m_value.string), other);
        }
        else
        {
            JSON_THROW(type_error::create(310, detail::concat("cannot use swap(string_t&) with ", type_name()), this));
        }
    }

    /// @brief exchanges the values
    /// @sa https://json.nlohmann.me/api/basic_json/swap/
    void swap(binary_t& other) // NOLINT(bugprone-exception-escape,cppcoreguidelines-noexcept-swap,performance-noexcept-swap)
    {
        // swap only works for strings
        if (JSON_HEDLEY_LIKELY(is_binary()))
        {
            using std::swap;
            swap(*(m_data.m_value.binary), other);
        }
        else
        {
            JSON_THROW(type_error::create(310, detail::concat("cannot use swap(binary_t&) with ", type_name()), this));
        }
    }

    /// @brief exchanges the values
    /// @sa https://json.nlohmann.me/api/basic_json/swap/
    void swap(typename binary_t::container_type& other) // NOLINT(bugprone-exception-escape)
    {
        // swap only works for strings
        if (JSON_HEDLEY_LIKELY(is_binary()))
        {
            using std::swap;
            swap(*(m_data.m_value.binary), other);
        }
        else
        {
            JSON_THROW(type_error::create(310, detail::concat("cannot use swap(binary_t::container_type&) with ", type_name()), this));
        }
    }

    /// @}

    //////////////////////////////////////////
    // lexicographical comparison operators //
    //////////////////////////////////////////

    /// @name lexicographical comparison operators
    /// @{

    // note parentheses around operands are necessary; see
    // https://github.com/nlohmann/json/issues/1530
#define JSON_IMPLEMENT_OPERATOR(op, null_result, unordered_result, default_result)                       \
    const auto lhs_type = lhs.type();                                                                    \
    const auto rhs_type = rhs.type();                                                                    \
    \
    if (lhs_type == rhs_type) /* NOLINT(readability/braces) */                                           \
    {                                                                                                    \
        switch (lhs_type)                                                                                \
        {                                                                                                \
            case value_t::array:                                                                         \
                return (*lhs.m_data.m_value.array) op (*rhs.m_data.m_value.array);                                     \
                \
            case value_t::object:                                                                        \
                return (*lhs.m_data.m_value.object) op (*rhs.m_data.m_value.object);                                   \
                \
            case value_t::null:                                                                          \
                return (null_result);                                                                    \
                \
            case value_t::string:                                                                        \
                return (*lhs.m_data.m_value.string) op (*rhs.m_data.m_value.string);                                   \
                \
            case value_t::boolean:                                                                       \
                return (lhs.m_data.m_value.boolean) op (rhs.m_data.m_value.boolean);                                   \
                \
            case value_t::number_integer:                                                                \
                return (lhs.m_data.m_value.number_integer) op (rhs.m_data.m_value.number_integer);                     \
                \
            case value_t::number_unsigned:                                                               \
                return (lhs.m_data.m_value.number_unsigned) op (rhs.m_data.m_value.number_unsigned);                   \
                \
            case value_t::number_float:                                                                  \
                return (lhs.m_data.m_value.number_float) op (rhs.m_data.m_value.number_float);                         \
                \
            case value_t::binary:                                                                        \
                return (*lhs.m_data.m_value.binary) op (*rhs.m_data.m_value.binary);                                   \
                \
            case value_t::discarded:                                                                     \
            default:                                                                                     \
                return (unordered_result);                                                               \
        }                                                                                                \
    }                                                                                                    \
    else if (lhs_type == value_t::number_integer && rhs_type == value_t::number_float)                   \
    {                                                                                                    \
        return static_cast<number_float_t>(lhs.m_data.m_value.number_integer) op rhs.m_data.m_value.number_float;      \
    }                                                                                                    \
    else if (lhs_type == value_t::number_float && rhs_type == value_t::number_integer)                   \
    {                                                                                                    \
        return lhs.m_data.m_value.number_float op static_cast<number_float_t>(rhs.m_data.m_value.number_integer);      \
    }                                                                                                    \
    else if (lhs_type == value_t::number_unsigned && rhs_type == value_t::number_float)                  \
    {                                                                                                    \
        return static_cast<number_float_t>(lhs.m_data.m_value.number_unsigned) op rhs.m_data.m_value.number_float;     \
    }                                                                                                    \
    else if (lhs_type == value_t::number_float && rhs_type == value_t::number_unsigned)                  \
    {                                                                                                    \
        return lhs.m_data.m_value.number_float op static_cast<number_float_t>(rhs.m_data.m_value.number_unsigned);     \
    }                                                                                                    \
    else if (lhs_type == value_t::number_unsigned && rhs_type == value_t::number_integer)                \
    {                                                                                                    \
        return static_cast<number_integer_t>(lhs.m_data.m_value.number_unsigned) op rhs.m_data.m_value.number_integer; \
    }                                                                                                    \
    else if (lhs_type == value_t::number_integer && rhs_type == value_t::number_unsigned)                \
    {                                                                                                    \
        return lhs.m_data.m_value.number_integer op static_cast<number_integer_t>(rhs.m_data.m_value.number_unsigned); \
    }                                                                                                    \
    else if(compares_unordered(lhs, rhs))\
    {\
        return (unordered_result);\
    }\
    \
    return (default_result);

  JSON_PRIVATE_UNLESS_TESTED:
    // returns true if:
    // - any operand is NaN and the other operand is of number type
    // - any operand is discarded
    // in legacy mode, discarded values are considered ordered if
    // an operation is computed as an odd number of inverses of others
    static bool compares_unordered(const_reference lhs, const_reference rhs, bool inverse = false) noexcept
    {
        if ((lhs.is_number_float() && std::isnan(lhs.m_data.m_value.number_float) && rhs.is_number())
                || (rhs.is_number_float() && std::isnan(rhs.m_data.m_value.number_float) && lhs.is_number()))
        {
            return true;
        }
#if JSON_USE_LEGACY_DISCARDED_VALUE_COMPARISON
        return (lhs.is_discarded() || rhs.is_discarded()) && !inverse;
#else
        static_cast<void>(inverse);
        return lhs.is_discarded() || rhs.is_discarded();
#endif
    }

  private:
    bool compares_unordered(const_reference rhs, bool inverse = false) const noexcept
    {
        return compares_unordered(*this, rhs, inverse);
    }

  public:
#if JSON_HAS_THREE_WAY_COMPARISON
    /// @brief comparison: equal
    /// @sa https://json.nlohmann.me/api/basic_json/operator_eq/
    bool operator==(const_reference rhs) const noexcept
    {
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"
#endif
        const_reference lhs = *this;
        JSON_IMPLEMENT_OPERATOR( ==, true, false, false)
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
    }

    /// @brief comparison: equal
    /// @sa https://json.nlohmann.me/api/basic_json/operator_eq/
    template<typename ScalarType>
    requires std::is_scalar_v<ScalarType>
    bool operator==(ScalarType rhs) const noexcept
    {
        return *this == basic_json(rhs);
    }

    /// @brief comparison: not equal
    /// @sa https://json.nlohmann.me/api/basic_json/operator_ne/
    bool operator!=(const_reference rhs) const noexcept
    {
        if (compares_unordered(rhs, true))
        {
            return false;
        }
        return !operator==(rhs);
    }

    /// @brief comparison: 3-way
    /// @sa https://json.nlohmann.me/api/basic_json/operator_spaceship/
    std::partial_ordering operator<=>(const_reference rhs) const noexcept // *NOPAD*
    {
        const_reference lhs = *this;
        // default_result is used if we cannot compare values. In that case,
        // we compare types.
        JSON_IMPLEMENT_OPERATOR(<=>, // *NOPAD*
                                std::partial_ordering::equivalent,
                                std::partial_ordering::unordered,
                                lhs_type <=> rhs_type) // *NOPAD*
    }

    /// @brief comparison: 3-way
    /// @sa https://json.nlohmann.me/api/basic_json/operator_spaceship/
    template<typename ScalarType>
    requires std::is_scalar_v<ScalarType>
    std::partial_ordering operator<=>(ScalarType rhs) const noexcept // *NOPAD*
    {
        return *this <=> basic_json(rhs); // *NOPAD*
    }

#if JSON_USE_LEGACY_DISCARDED_VALUE_COMPARISON
    // all operators that are computed as an odd number of inverses of others
    // need to be overloaded to emulate the legacy comparison behavior

    /// @brief comparison: less than or equal
    /// @sa https://json.nlohmann.me/api/basic_json/operator_le/
    JSON_HEDLEY_DEPRECATED_FOR(3.11.0, undef JSON_USE_LEGACY_DISCARDED_VALUE_COMPARISON)
    bool operator<=(const_reference rhs) const noexcept
    {
        if (compares_unordered(rhs, true))
        {
            return false;
        }
        return !(rhs < *this);
    }

    /// @brief comparison: less than or equal
    /// @sa https://json.nlohmann.me/api/basic_json/operator_le/
    template<typename ScalarType>
    requires std::is_scalar_v<ScalarType>
    bool operator<=(ScalarType rhs) const noexcept
    {
        return *this <= basic_json(rhs);
    }

    /// @brief comparison: greater than or equal
    /// @sa https://json.nlohmann.me/api/basic_json/operator_ge/
    JSON_HEDLEY_DEPRECATED_FOR(3.11.0, undef JSON_USE_LEGACY_DISCARDED_VALUE_COMPARISON)
    bool operator>=(const_reference rhs) const noexcept
    {
        if (compares_unordered(rhs, true))
        {
            return false;
        }
        return !(*this < rhs);
    }

    /// @brief comparison: greater than or equal
    /// @sa https://json.nlohmann.me/api/basic_json/operator_ge/
    template<typename ScalarType>
    requires std::is_scalar_v<ScalarType>
    bool operator>=(ScalarType rhs) const noexcept
    {
        return *this >= basic_json(rhs);
    }
#endif
#else
    /// @brief comparison: equal
    /// @sa https://json.nlohmann.me/api/basic_json/operator_eq/
    friend bool operator==(const_reference lhs, const_reference rhs) noexcept
    {
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"
#endif
        JSON_IMPLEMENT_OPERATOR( ==, true, false, false)
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
    }

    /// @brief comparison: equal
    /// @sa https://json.nlohmann.me/api/basic_json/operator_eq/
    template<typename ScalarType, typename std::enable_if<
                 std::is_scalar<ScalarType>::value, int>::type = 0>
    friend bool operator==(const_reference lhs, ScalarType rhs) noexcept
    {
        return lhs == basic_json(rhs);
    }

    /// @brief comparison: equal
    /// @sa https://json.nlohmann.me/api/basic_json/operator_eq/
    template<typename ScalarType, typename std::enable_if<
                 std::is_scalar<ScalarType>::value, int>::type = 0>
    friend bool operator==(ScalarType lhs, const_reference rhs) noexcept
    {
        return basic_json(lhs) == rhs;
    }

    /// @brief comparison: not equal
    /// @sa https://json.nlohmann.me/api/basic_json/operator_ne/
    friend bool operator!=(const_reference lhs, const_reference rhs) noexcept
    {
        if (compares_unordered(lhs, rhs, true))
        {
            return false;
        }
        return !(lhs == rhs);
    }

    /// @brief comparison: not equal
    /// @sa https://json.nlohmann.me/api/basic_json/operator_ne/
    template<typename ScalarType, typename std::enable_if<
                 std::is_scalar<ScalarType>::value, int>::type = 0>
    friend bool operator!=(const_reference lhs, ScalarType rhs) noexcept
    {
        return lhs != basic_json(rhs);
    }

    /// @brief comparison: not equal
    /// @sa https://json.nlohmann.me/api/basic_json/operator_ne/
    template<typename ScalarType, typename std::enable_if<
                 std::is_scalar<ScalarType>::value, int>::type = 0>
    friend bool operator!=(ScalarType lhs, const_reference rhs) noexcept
    {
        return basic_json(lhs) != rhs;
    }

    /// @brief comparison: less than
    /// @sa https://json.nlohmann.me/api/basic_json/operator_lt/
    friend bool operator<(const_reference lhs, const_reference rhs) noexcept
    {
        // default_result is used if we cannot compare values. In that case,
        // we compare types. Note we have to call the operator explicitly,
        // because MSVC has problems otherwise.
        JSON_IMPLEMENT_OPERATOR( <, false, false, operator<(lhs_type, rhs_type))
    }

    /// @brief comparison: less than
    /// @sa https://json.nlohmann.me/api/basic_json/operator_lt/
    template<typename ScalarType, typename std::enable_if<
                 std::is_scalar<ScalarType>::value, int>::type = 0>
    friend bool operator<(const_reference lhs, ScalarType rhs) noexcept
    {
        return lhs < basic_json(rhs);
    }

    /// @brief comparison: less than
    /// @sa https://json.nlohmann.me/api/basic_json/operator_lt/
    template<typename ScalarType, typename std::enable_if<
                 std::is_scalar<ScalarType>::value, int>::type = 0>
    friend bool operator<(ScalarType lhs, const_reference rhs) noexcept
    {
        return basic_json(lhs) < rhs;
    }

    /// @brief comparison: less than or equal
    /// @sa https://json.nlohmann.me/api/basic_json/operator_le/
    friend bool operator<=(const_reference lhs, const_reference rhs) noexcept
    {
        if (compares_unordered(lhs, rhs, true))
        {
            return false;
        }
        return !(rhs < lhs);
    }

    /// @brief comparison: less than or equal
    /// @sa https://json.nlohmann.me/api/basic_json/operator_le/
    template<typename ScalarType, typename std::enable_if<
                 std::is_scalar<ScalarType>::value, int>::type = 0>
    friend bool operator<=(const_reference lhs, ScalarType rhs) noexcept
    {
        return lhs <= basic_json(rhs);
    }

    /// @brief comparison: less than or equal
    /// @sa https://json.nlohmann.me/api/basic_json/operator_le/
    template<typename ScalarType, typename std::enable_if<
                 std::is_scalar<ScalarType>::value, int>::type = 0>
    friend bool operator<=(ScalarType lhs, const_reference rhs) noexcept
    {
        return basic_json(lhs) <= rhs;
    }

    /// @brief comparison: greater than
    /// @sa https://json.nlohmann.me/api/basic_json/operator_gt/
    friend bool operator>(const_reference lhs, const_reference rhs) noexcept
    {
        // double inverse
        if (compares_unordered(lhs, rhs))
        {
            return false;
        }
        return !(lhs <= rhs);
    }

    /// @brief comparison: greater than
    /// @sa https://json.nlohmann.me/api/basic_json/operator_gt/
    template<typename ScalarType, typename std::enable_if<
                 std::is_scalar<ScalarType>::value, int>::type = 0>
    friend bool operator>(const_reference lhs, ScalarType rhs) noexcept
    {
        return lhs > basic_json(rhs);
    }

    /// @brief comparison: greater than
    /// @sa https://json.nlohmann.me/api/basic_json/operator_gt/
    template<typename ScalarType, typename std::enable_if<
                 std::is_scalar<ScalarType>::value, int>::type = 0>
    friend bool operator>(ScalarType lhs, const_reference rhs) noexcept
    {
        return basic_json(lhs) > rhs;
    }

    /// @brief comparison: greater than or equal
    /// @sa https://json.nlohmann.me/api/basic_json/operator_ge/
    friend bool operator>=(const_reference lhs, const_reference rhs) noexcept
    {
        if (compares_unordered(lhs, rhs, true))
        {
            return false;
        }
        return !(lhs < rhs);
    }

    /// @brief comparison: greater than or equal
    /// @sa https://json.nlohmann.me/api/basic_json/operator_ge/
    template<typename ScalarType, typename std::enable_if<
                 std::is_scalar<ScalarType>::value, int>::type = 0>
    friend bool operator>=(const_reference lhs, ScalarType rhs) noexcept
    {
        return lhs >= basic_json(rhs);
    }

    /// @brief comparison: greater than or equal
    /// @sa https://json.nlohmann.me/api/basic_json/operator_ge/
    template<typename ScalarType, typename std::enable_if<
                 std::is_scalar<ScalarType>::value, int>::type = 0>
    friend bool operator>=(ScalarType lhs, const_reference rhs) noexcept
    {
        return basic_json(lhs) >= rhs;
    }
#endif

#undef JSON_IMPLEMENT_OPERATOR

    /// @}

    ///////////////////
    // serialization //
    ///////////////////

    /// @name serialization
    /// @{
#ifndef JSON_NO_IO
    /// @brief serialize to stream
    /// @sa https://json.nlohmann.me/api/basic_json/operator_ltlt/
    friend std::ostream& operator<<(std::ostream& o, const basic_json& j)
    {
        // read width member and use it as indentation parameter if nonzero
        const bool pretty_print = o.width() > 0;
        const auto indentation = pretty_print ? o.width() : 0;

        // reset width to 0 for subsequent calls to this stream
        o.width(0);

        // do the actual serialization
        serializer s(detail::output_adapter<char>(o), o.fill());
        s.dump(j, pretty_print, false, static_cast<unsigned int>(indentation));
        return o;
    }

    /// @brief serialize to stream
    /// @sa https://json.nlohmann.me/api/basic_json/operator_ltlt/
    /// @deprecated This function is deprecated since 3.0.0 and will be removed in
    ///             version 4.0.0 of the library. Please use
    ///             operator<<(std::ostream&, const basic_json&) instead; that is,
    ///             replace calls like `j >> o;` with `o << j;`.
    JSON_HEDLEY_DEPRECATED_FOR(3.0.0, operator<<(std::ostream&, const basic_json&))
    friend std::ostream& operator>>(const basic_json& j, std::ostream& o)
    {
        return o << j;
    }
#endif  // JSON_NO_IO
    /// @}

    /////////////////////
    // deserialization //
    /////////////////////

    /// @name deserialization
    /// @{

    /// @brief deserialize from a compatible input
    /// @sa https://json.nlohmann.me/api/basic_json/parse/
    template<typename InputType>
    JSON_HEDLEY_WARN_UNUSED_RESULT
    static basic_json parse(InputType&& i,
                            parser_callback_t cb = nullptr,
                            const bool allow_exceptions = true,
                            const bool ignore_comments = false)
    {
        basic_json result;
        parser(detail::input_adapter(std::forward<InputType>(i)), std::move(cb), allow_exceptions, ignore_comments).parse(true, result); // cppcheck-suppress[accessMoved,accessForwarded]
        return result;
    }

    /// @brief deserialize from a pair of character iterators
    /// @sa https://json.nlohmann.me/api/basic_json/parse/
    template<typename IteratorType>
    JSON_HEDLEY_WARN_UNUSED_RESULT
    static basic_json parse(IteratorType first,
                            IteratorType last,
                            parser_callback_t cb = nullptr,
                            const bool allow_exceptions = true,
                            const bool ignore_comments = false)
    {
        basic_json result;
        parser(detail::input_adapter(std::move(first), std::move(last)), std::move(cb), allow_exceptions, ignore_comments).parse(true, result); // cppcheck-suppress[accessMoved]
        return result;
    }

    JSON_HEDLEY_WARN_UNUSED_RESULT
    JSON_HEDLEY_DEPRECATED_FOR(3.8.0, parse(ptr, ptr + len))
    static basic_json parse(detail::span_input_adapter&& i,
                            parser_callback_t cb = nullptr,
                            const bool allow_exceptions = true,
                            const bool ignore_comments = false)
    {
        basic_json result;
        parser(i.get(), std::move(cb), allow_exceptions, ignore_comments).parse(true, result); // cppcheck-suppress[accessMoved]
        return result;
    }

    /// @brief check if the input is valid JSON
    /// @sa https://json.nlohmann.me/api/basic_json/accept/
    template<typename InputType>
    static bool accept(InputType&& i,
                       const bool ignore_comments = false)
    {
        return parser(detail::input_adapter(std::forward<InputType>(i)), nullptr, false, ignore_comments).accept(true);
    }

    /// @brief check if the input is valid JSON
    /// @sa https://json.nlohmann.me/api/basic_json/accept/
    template<typename IteratorType>
    static bool accept(IteratorType first, IteratorType last,
                       const bool ignore_comments = false)
    {
        return parser(detail::input_adapter(std::move(first), std::move(last)), nullptr, false, ignore_comments).accept(true);
    }

    JSON_HEDLEY_WARN_UNUSED_RESULT
    JSON_HEDLEY_DEPRECATED_FOR(3.8.0, accept(ptr, ptr + len))
    static bool accept(detail::span_input_adapter&& i,
                       const bool ignore_comments = false)
    {
        return parser(i.get(), nullptr, false, ignore_comments).accept(true);
    }

    /// @brief generate SAX events
    /// @sa https://json.nlohmann.me/api/basic_json/sax_parse/
    template <typename InputType, typename SAX>
    JSON_HEDLEY_NON_NULL(2)
    static bool sax_parse(InputType&& i, SAX* sax,
                          input_format_t format = input_format_t::json,
                          const bool strict = true,
                          const bool ignore_comments = false)
    {
        auto ia = detail::input_adapter(std::forward<InputType>(i));
        return format == input_format_t::json
               ? parser(std::move(ia), nullptr, true, ignore_comments).sax_parse(sax, strict)
               : detail::binary_reader<basic_json, decltype(ia), SAX>(std::move(ia), format).sax_parse(format, sax, strict);
    }

    /// @brief generate SAX events
    /// @sa https://json.nlohmann.me/api/basic_json/sax_parse/
    template<class IteratorType, class SAX>
    JSON_HEDLEY_NON_NULL(3)
    static bool sax_parse(IteratorType first, IteratorType last, SAX* sax,
                          input_format_t format = input_format_t::json,
                          const bool strict = true,
                          const bool ignore_comments = false)
    {
        auto ia = detail::input_adapter(std::move(first), std::move(last));
        return format == input_format_t::json
               ? parser(std::move(ia), nullptr, true, ignore_comments).sax_parse(sax, strict)
               : detail::binary_reader<basic_json, decltype(ia), SAX>(std::move(ia), format).sax_parse(format, sax, strict);
    }

    /// @brief generate SAX events
    /// @sa https://json.nlohmann.me/api/basic_json/sax_parse/
    /// @deprecated This function is deprecated since 3.8.0 and will be removed in
    ///             version 4.0.0 of the library. Please use
    ///             sax_parse(ptr, ptr + len) instead.
    template <typename SAX>
    JSON_HEDLEY_DEPRECATED_FOR(3.8.0, sax_parse(ptr, ptr + len, ...))
    JSON_HEDLEY_NON_NULL(2)
    static bool sax_parse(detail::span_input_adapter&& i, SAX* sax,
                          input_format_t format = input_format_t::json,
                          const bool strict = true,
                          const bool ignore_comments = false)
    {
        auto ia = i.get();
        return format == input_format_t::json
               // NOLINTNEXTLINE(hicpp-move-const-arg,performance-move-const-arg)
               ? parser(std::move(ia), nullptr, true, ignore_comments).sax_parse(sax, strict)
               // NOLINTNEXTLINE(hicpp-move-const-arg,performance-move-const-arg)
               : detail::binary_reader<basic_json, decltype(ia), SAX>(std::move(ia), format).sax_parse(format, sax, strict);
    }
#ifndef JSON_NO_IO
    /// @brief deserialize from stream
    /// @sa https://json.nlohmann.me/api/basic_json/operator_gtgt/
    /// @deprecated This stream operator is deprecated since 3.0.0 and will be removed in
    ///             version 4.0.0 of the library. Please use
    ///             operator>>(std::istream&, basic_json&) instead; that is,
    ///             replace calls like `j << i;` with `i >> j;`.
    JSON_HEDLEY_DEPRECATED_FOR(3.0.0, operator>>(std::istream&, basic_json&))
    friend std::istream& operator<<(basic_json& j, std::istream& i)
    {
        return operator>>(i, j);
    }

    /// @brief deserialize from stream
    /// @sa https://json.nlohmann.me/api/basic_json/operator_gtgt/
    friend std::istream& operator>>(std::istream& i, basic_json& j)
    {
        parser(detail::input_adapter(i)).parse(false, j);
        return i;
    }
#endif  // JSON_NO_IO
    /// @}

    ///////////////////////////
    // convenience functions //
    ///////////////////////////

    /// @brief return the type as string
    /// @sa https://json.nlohmann.me/api/basic_json/type_name/
    JSON_HEDLEY_RETURNS_NON_NULL
    const char* type_name() const noexcept
    {
        switch (m_data.m_type)
        {
            case value_t::null:
                return "null";
            case value_t::object:
                return "object";
            case value_t::array:
                return "array";
            case value_t::string:
                return "string";
            case value_t::boolean:
                return "boolean";
            case value_t::binary:
                return "binary";
            case value_t::discarded:
                return "discarded";
            case value_t::number_integer:
            case value_t::number_unsigned:
            case value_t::number_float:
            default:
                return "number";
        }
    }

  JSON_PRIVATE_UNLESS_TESTED:
    //////////////////////
    // member variables //
    //////////////////////

    struct data
    {
        /// the type of the current element
        value_t m_type = value_t::null;

        /// the value of the current element
        json_value m_value = {};

        data(const value_t v)
            : m_type(v), m_value(v)
        {
        }

        data(size_type cnt, const basic_json& val)
            : m_type(value_t::array)
        {
            m_value.array = create<array_t>(cnt, val);
        }

        data() noexcept = default;
        data(data&&) noexcept = default;
        data(const data&) noexcept = delete;
        data& operator=(data&&) noexcept = delete;
        data& operator=(const data&) noexcept = delete;

        ~data() noexcept
        {
            m_value.destroy(m_type);
        }
    };

    data m_data = {};

#if JSON_DIAGNOSTICS
    /// a pointer to a parent value (for debugging purposes)
    basic_json* m_parent = nullptr;
#endif

#if JSON_DIAGNOSTIC_POSITIONS
    /// the start position of the value
    std::size_t start_position = std::string::npos;
    /// the end position of the value
    std::size_t end_position = std::string::npos;
  public:
    constexpr std::size_t start_pos() const noexcept
    {
        return start_position;
    }

    constexpr std::size_t end_pos() const noexcept
    {
        return end_position;
    }
#endif

    //////////////////////////////////////////
    // binary serialization/deserialization //
    //////////////////////////////////////////

    /// @name binary serialization/deserialization support
    /// @{

  public:
    /// @brief create a CBOR serialization of a given JSON value
    /// @sa https://json.nlohmann.me/api/basic_json/to_cbor/
    static std::vector<std::uint8_t> to_cbor(const basic_json& j)
    {
        std::vector<std::uint8_t> result;
        to_cbor(j, result);
        return result;
    }

    /// @brief create a CBOR serialization of a given JSON value
    /// @sa https://json.nlohmann.me/api/basic_json/to_cbor/
    static void to_cbor(const basic_json& j, detail::output_adapter<std::uint8_t> o)
    {
        binary_writer<std::uint8_t>(o).write_cbor(j);
    }

    /// @brief create a CBOR serialization of a given JSON value
    /// @sa https://json.nlohmann.me/api/basic_json/to_cbor/
    static void to_cbor(const basic_json& j, detail::output_adapter<char> o)
    {
        binary_writer<char>(o).write_cbor(j);
    }

    /// @brief create a MessagePack serialization of a given JSON value
    /// @sa https://json.nlohmann.me/api/basic_json/to_msgpack/
    static std::vector<std::uint8_t> to_msgpack(const basic_json& j)
    {
        std::vector<std::uint8_t> result;
        to_msgpack(j, result);
        return result;
    }

    /// @brief create a MessagePack serialization of a given JSON value
    /// @sa https://json.nlohmann.me/api/basic_json/to_msgpack/
    static void to_msgpack(const basic_json& j, detail::output_adapter<std::uint8_t> o)
    {
        binary_writer<std::uint8_t>(o).write_msgpack(j);
    }

    /// @brief create a MessagePack serialization of a given JSON value
    /// @sa https://json.nlohmann.me/api/basic_json/to_msgpack/
    static void to_msgpack(const basic_json& j, detail::output_adapter<char> o)
    {
        binary_writer<char>(o).write_msgpack(j);
    }

    /// @brief create a UBJSON serialization of a given JSON value
    /// @sa https://json.nlohmann.me/api/basic_json/to_ubjson/
    static std::vector<std::uint8_t> to_ubjson(const basic_json& j,
            const bool use_size = false,
            const bool use_type = false)
    {
        std::vector<std::uint8_t> result;
        to_ubjson(j, result, use_size, use_type);
        return result;
    }

    /// @brief create a UBJSON serialization of a given JSON value
    /// @sa https://json.nlohmann.me/api/basic_json/to_ubjson/
    static void to_ubjson(const basic_json& j, detail::output_adapter<std::uint8_t> o,
                          const bool use_size = false, const bool use_type = false)
    {
        binary_writer<std::uint8_t>(o).write_ubjson(j, use_size, use_type);
    }

    /// @brief create a UBJSON serialization of a given JSON value
    /// @sa https://json.nlohmann.me/api/basic_json/to_ubjson/
    static void to_ubjson(const basic_json& j, detail::output_adapter<char> o,
                          const bool use_size = false, const bool use_type = false)
    {
        binary_writer<char>(o).write_ubjson(j, use_size, use_type);
    }

    /// @brief create a BJData serialization of a given JSON value
    /// @sa https://json.nlohmann.me/api/basic_json/to_bjdata/
    static std::vector<std::uint8_t> to_bjdata(const basic_json& j,
            const bool use_size = false,
            const bool use_type = false,
            const bjdata_version_t version = bjdata_version_t::draft2)
    {
        std::vector<std::uint8_t> result;
        to_bjdata(j, result, use_size, use_type, version);
        return result;
    }

    /// @brief create a BJData serialization of a given JSON value
    /// @sa https://json.nlohmann.me/api/basic_json/to_bjdata/
    static void to_bjdata(const basic_json& j, detail::output_adapter<std::uint8_t> o,
                          const bool use_size = false, const bool use_type = false,
                          const bjdata_version_t version = bjdata_version_t::draft2)
    {
        binary_writer<std::uint8_t>(o).write_ubjson(j, use_size, use_type, true, true, version);
    }

    /// @brief create a BJData serialization of a given JSON value
    /// @sa https://json.nlohmann.me/api/basic_json/to_bjdata/
    static void to_bjdata(const basic_json& j, detail::output_adapter<char> o,
                          const bool use_size = false, const bool use_type = false,
                          const bjdata_version_t version = bjdata_version_t::draft2)
    {
        binary_writer<char>(o).write_ubjson(j, use_size, use_type, true, true, version);
    }

    /// @brief create a BSON serialization of a given JSON value
    /// @sa https://json.nlohmann.me/api/basic_json/to_bson/
    static std::vector<std::uint8_t> to_bson(const basic_json& j)
    {
        std::vector<std::uint8_t> result;
        to_bson(j, result);
        return result;
    }

    /// @brief create a BSON serialization of a given JSON value
    /// @sa https://json.nlohmann.me/api/basic_json/to_bson/
    static void to_bson(const basic_json& j, detail::output_adapter<std::uint8_t> o)
    {
        binary_writer<std::uint8_t>(o).write_bson(j);
    }

    /// @brief create a BSON serialization of a given JSON value
    /// @sa https://json.nlohmann.me/api/basic_json/to_bson/
    static void to_bson(const basic_json& j, detail::output_adapter<char> o)
    {
        binary_writer<char>(o).write_bson(j);
    }

    /// @brief create a JSON value from an input in CBOR format
    /// @sa https://json.nlohmann.me/api/basic_json/from_cbor/
    template<typename InputType>
    JSON_HEDLEY_WARN_UNUSED_RESULT
    static basic_json from_cbor(InputType&& i,
                                const bool strict = true,
                                const bool allow_exceptions = true,
                                const cbor_tag_handler_t tag_handler = cbor_tag_handler_t::error)
    {
        basic_json result;
        auto ia = detail::input_adapter(std::forward<InputType>(i));
        detail::json_sax_dom_parser<basic_json, decltype(ia)> sdp(result, allow_exceptions);
        const bool res = binary_reader<decltype(ia)>(std::move(ia), input_format_t::cbor).sax_parse(input_format_t::cbor, &sdp, strict, tag_handler); // cppcheck-suppress[accessMoved]
        return res ? result : basic_json(value_t::discarded);
    }

    /// @brief create a JSON value from an input in CBOR format
    /// @sa https://json.nlohmann.me/api/basic_json/from_cbor/
    template<typename IteratorType>
    JSON_HEDLEY_WARN_UNUSED_RESULT
    static basic_json from_cbor(IteratorType first, IteratorType last,
                                const bool strict = true,
                                const bool allow_exceptions = true,
                                const cbor_tag_handler_t tag_handler = cbor_tag_handler_t::error)
    {
        basic_json result;
        auto ia = detail::input_adapter(std::move(first), std::move(last));
        detail::json_sax_dom_parser<basic_json, decltype(ia)> sdp(result, allow_exceptions);
        const bool res = binary_reader<decltype(ia)>(std::move(ia), input_format_t::cbor).sax_parse(input_format_t::cbor, &sdp, strict, tag_handler); // cppcheck-suppress[accessMoved]
        return res ? result : basic_json(value_t::discarded);
    }

    template<typename T>
    JSON_HEDLEY_WARN_UNUSED_RESULT
    JSON_HEDLEY_DEPRECATED_FOR(3.8.0, from_cbor(ptr, ptr + len))
    static basic_json from_cbor(const T* ptr, std::size_t len,
                                const bool strict = true,
                                const bool allow_exceptions = true,
                                const cbor_tag_handler_t tag_handler = cbor_tag_handler_t::error)
    {
        return from_cbor(ptr, ptr + len, strict, allow_exceptions, tag_handler);
    }

    JSON_HEDLEY_WARN_UNUSED_RESULT
    JSON_HEDLEY_DEPRECATED_FOR(3.8.0, from_cbor(ptr, ptr + len))
    static basic_json from_cbor(detail::span_input_adapter&& i,
                                const bool strict = true,
                                const bool allow_exceptions = true,
                                const cbor_tag_handler_t tag_handler = cbor_tag_handler_t::error)
    {
        basic_json result;
        auto ia = i.get();
        detail::json_sax_dom_parser<basic_json, decltype(ia)> sdp(result, allow_exceptions);
        // NOLINTNEXTLINE(hicpp-move-const-arg,performance-move-const-arg)
        const bool res = binary_reader<decltype(ia)>(std::move(ia), input_format_t::cbor).sax_parse(input_format_t::cbor, &sdp, strict, tag_handler); // cppcheck-suppress[accessMoved]
        return res ? result : basic_json(value_t::discarded);
    }

    /// @brief create a JSON value from an input in MessagePack format
    /// @sa https://json.nlohmann.me/api/basic_json/from_msgpack/
    template<typename InputType>
    JSON_HEDLEY_WARN_UNUSED_RESULT
    static basic_json from_msgpack(InputType&& i,
                                   const bool strict = true,
                                   const bool allow_exceptions = true)
    {
        basic_json result;
        auto ia = detail::input_adapter(std::forward<InputType>(i));
        detail::json_sax_dom_parser<basic_json, decltype(ia)> sdp(result, allow_exceptions);
        const bool res = binary_reader<decltype(ia)>(std::move(ia), input_format_t::msgpack).sax_parse(input_format_t::msgpack, &sdp, strict); // cppcheck-suppress[accessMoved]
        return res ? result : basic_json(value_t::discarded);
    }

    /// @brief create a JSON value from an input in MessagePack format
    /// @sa https://json.nlohmann.me/api/basic_json/from_msgpack/
    template<typename IteratorType>
    JSON_HEDLEY_WARN_UNUSED_RESULT
    static basic_json from_msgpack(IteratorType first, IteratorType last,
                                   const bool strict = true,
                                   const bool allow_exceptions = true)
    {
        basic_json result;
        auto ia = detail::input_adapter(std::move(first), std::move(last));
        detail::json_sax_dom_parser<basic_json, decltype(ia)> sdp(result, allow_exceptions);
        const bool res = binary_reader<decltype(ia)>(std::move(ia), input_format_t::msgpack).sax_parse(input_format_t::msgpack, &sdp, strict); // cppcheck-suppress[accessMoved]
        return res ? result : basic_json(value_t::discarded);
    }

    template<typename T>
    JSON_HEDLEY_WARN_UNUSED_RESULT
    JSON_HEDLEY_DEPRECATED_FOR(3.8.0, from_msgpack(ptr, ptr + len))
    static basic_json from_msgpack(const T* ptr, std::size_t len,
                                   const bool strict = true,
                                   const bool allow_exceptions = true)
    {
        return from_msgpack(ptr, ptr + len, strict, allow_exceptions);
    }

    JSON_HEDLEY_WARN_UNUSED_RESULT
    JSON_HEDLEY_DEPRECATED_FOR(3.8.0, from_msgpack(ptr, ptr + len))
    static basic_json from_msgpack(detail::span_input_adapter&& i,
                                   const bool strict = true,
                                   const bool allow_exceptions = true)
    {
        basic_json result;
        auto ia = i.get();
        detail::json_sax_dom_parser<basic_json, decltype(ia)> sdp(result, allow_exceptions);
        // NOLINTNEXTLINE(hicpp-move-const-arg,performance-move-const-arg)
        const bool res = binary_reader<decltype(ia)>(std::move(ia), input_format_t::msgpack).sax_parse(input_format_t::msgpack, &sdp, strict); // cppcheck-suppress[accessMoved]
        return res ? result : basic_json(value_t::discarded);
    }

    /// @brief create a JSON value from an input in UBJSON format
    /// @sa https://json.nlohmann.me/api/basic_json/from_ubjson/
    template<typename InputType>
    JSON_HEDLEY_WARN_UNUSED_RESULT
    static basic_json from_ubjson(InputType&& i,
                                  const bool strict = true,
                                  const bool allow_exceptions = true)
    {
        basic_json result;
        auto ia = detail::input_adapter(std::forward<InputType>(i));
        detail::json_sax_dom_parser<basic_json, decltype(ia)> sdp(result, allow_exceptions);
        const bool res = binary_reader<decltype(ia)>(std::move(ia), input_format_t::ubjson).sax_parse(input_format_t::ubjson, &sdp, strict); // cppcheck-suppress[accessMoved]
        return res ? result : basic_json(value_t::discarded);
    }

    /// @brief create a JSON value from an input in UBJSON format
    /// @sa https://json.nlohmann.me/api/basic_json/from_ubjson/
    template<typename IteratorType>
    JSON_HEDLEY_WARN_UNUSED_RESULT
    static basic_json from_ubjson(IteratorType first, IteratorType last,
                                  const bool strict = true,
                                  const bool allow_exceptions = true)
    {
        basic_json result;
        auto ia = detail::input_adapter(std::move(first), std::move(last));
        detail::json_sax_dom_parser<basic_json, decltype(ia)> sdp(result, allow_exceptions);
        const bool res = binary_reader<decltype(ia)>(std::move(ia), input_format_t::ubjson).sax_parse(input_format_t::ubjson, &sdp, strict); // cppcheck-suppress[accessMoved]
        return res ? result : basic_json(value_t::discarded);
    }

    template<typename T>
    JSON_HEDLEY_WARN_UNUSED_RESULT
    JSON_HEDLEY_DEPRECATED_FOR(3.8.0, from_ubjson(ptr, ptr + len))
    static basic_json from_ubjson(const T* ptr, std::size_t len,
                                  const bool strict = true,
                                  const bool allow_exceptions = true)
    {
        return from_ubjson(ptr, ptr + len, strict, allow_exceptions);
    }

    JSON_HEDLEY_WARN_UNUSED_RESULT
    JSON_HEDLEY_DEPRECATED_FOR(3.8.0, from_ubjson(ptr, ptr + len))
    static basic_json from_ubjson(detail::span_input_adapter&& i,
                                  const bool strict = true,
                                  const bool allow_exceptions = true)
    {
        basic_json result;
        auto ia = i.get();
        detail::json_sax_dom_parser<basic_json, decltype(ia)> sdp(result, allow_exceptions);
        // NOLINTNEXTLINE(hicpp-move-const-arg,performance-move-const-arg)
        const bool res = binary_reader<decltype(ia)>(std::move(ia), input_format_t::ubjson).sax_parse(input_format_t::ubjson, &sdp, strict); // cppcheck-suppress[accessMoved]
        return res ? result : basic_json(value_t::discarded);
    }

    /// @brief create a JSON value from an input in BJData format
    /// @sa https://json.nlohmann.me/api/basic_json/from_bjdata/
    template<typename InputType>
    JSON_HEDLEY_WARN_UNUSED_RESULT
    static basic_json from_bjdata(InputType&& i,
                                  const bool strict = true,
                                  const bool allow_exceptions = true)
    {
        basic_json result;
        auto ia = detail::input_adapter(std::forward<InputType>(i));
        detail::json_sax_dom_parser<basic_json, decltype(ia)> sdp(result, allow_exceptions);
        const bool res = binary_reader<decltype(ia)>(std::move(ia), input_format_t::bjdata).sax_parse(input_format_t::bjdata, &sdp, strict); // cppcheck-suppress[accessMoved]
        return res ? result : basic_json(value_t::discarded);
    }

    /// @brief create a JSON value from an input in BJData format
    /// @sa https://json.nlohmann.me/api/basic_json/from_bjdata/
    template<typename IteratorType>
    JSON_HEDLEY_WARN_UNUSED_RESULT
    static basic_json from_bjdata(IteratorType first, IteratorType last,
                                  const bool strict = true,
                                  const bool allow_exceptions = true)
    {
        basic_json result;
        auto ia = detail::input_adapter(std::move(first), std::move(last));
        detail::json_sax_dom_parser<basic_json, decltype(ia)> sdp(result, allow_exceptions);
        const bool res = binary_reader<decltype(ia)>(std::move(ia), input_format_t::bjdata).sax_parse(input_format_t::bjdata, &sdp, strict); // cppcheck-suppress[accessMoved]
        return res ? result : basic_json(value_t::discarded);
    }

    /// @brief create a JSON value from an input in BSON format
    /// @sa https://json.nlohmann.me/api/basic_json/from_bson/
    template<typename InputType>
    JSON_HEDLEY_WARN_UNUSED_RESULT
    static basic_json from_bson(InputType&& i,
                                const bool strict = true,
                                const bool allow_exceptions = true)
    {
        basic_json result;
        auto ia = detail::input_adapter(std::forward<InputType>(i));
        detail::json_sax_dom_parser<basic_json, decltype(ia)> sdp(result, allow_exceptions);
        const bool res = binary_reader<decltype(ia)>(std::move(ia), input_format_t::bson).sax_parse(input_format_t::bson, &sdp, strict); // cppcheck-suppress[accessMoved]
        return res ? result : basic_json(value_t::discarded);
    }

    /// @brief create a JSON value from an input in BSON format
    /// @sa https://json.nlohmann.me/api/basic_json/from_bson/
    template<typename IteratorType>
    JSON_HEDLEY_WARN_UNUSED_RESULT
    static basic_json from_bson(IteratorType first, IteratorType last,
                                const bool strict = true,
                                const bool allow_exceptions = true)
    {
        basic_json result;
        auto ia = detail::input_adapter(std::move(first), std::move(last));
        detail::json_sax_dom_parser<basic_json, decltype(ia)> sdp(result, allow_exceptions);
        const bool res = binary_reader<decltype(ia)>(std::move(ia), input_format_t::bson).sax_parse(input_format_t::bson, &sdp, strict); // cppcheck-suppress[accessMoved]
        return res ? result : basic_json(value_t::discarded);
    }

    template<typename T>
    JSON_HEDLEY_WARN_UNUSED_RESULT
    JSON_HEDLEY_DEPRECATED_FOR(3.8.0, from_bson(ptr, ptr + len))
    static basic_json from_bson(const T* ptr, std::size_t len,
                                const bool strict = true,
                                const bool allow_exceptions = true)
    {
        return from_bson(ptr, ptr + len, strict, allow_exceptions);
    }

    JSON_HEDLEY_WARN_UNUSED_RESULT
    JSON_HEDLEY_DEPRECATED_FOR(3.8.0, from_bson(ptr, ptr + len))
    static basic_json from_bson(detail::span_input_adapter&& i,
                                const bool strict = true,
                                const bool allow_exceptions = true)
    {
        basic_json result;
        auto ia = i.get();
        detail::json_sax_dom_parser<basic_json, decltype(ia)> sdp(result, allow_exceptions);
        // NOLINTNEXTLINE(hicpp-move-const-arg,performance-move-const-arg)
        const bool res = binary_reader<decltype(ia)>(std::move(ia), input_format_t::bson).sax_parse(input_format_t::bson, &sdp, strict); // cppcheck-suppress[accessMoved]
        return res ? result : basic_json(value_t::discarded);
    }
    /// @}

    //////////////////////////
    // JSON Pointer support //
    //////////////////////////

    /// @name JSON Pointer functions
    /// @{

    /// @brief access specified element via JSON Pointer
    /// @sa https://json.nlohmann.me/api/basic_json/operator%5B%5D/
    reference operator[](const json_pointer& ptr)
    {
        return ptr.get_unchecked(this);
    }

    template<typename BasicJsonType, detail::enable_if_t<detail::is_basic_json<BasicJsonType>::value, int> = 0>
    JSON_HEDLEY_DEPRECATED_FOR(3.11.0, basic_json::json_pointer or nlohmann::json_pointer<basic_json::string_t>) // NOLINT(readability/alt_tokens)
    reference operator[](const ::nlohmann::json_pointer<BasicJsonType>& ptr)
    {
        return ptr.get_unchecked(this);
    }

    /// @brief access specified element via JSON Pointer
    /// @sa https://json.nlohmann.me/api/basic_json/operator%5B%5D/
    const_reference operator[](const json_pointer& ptr) const
    {
        return ptr.get_unchecked(this);
    }

    template<typename BasicJsonType, detail::enable_if_t<detail::is_basic_json<BasicJsonType>::value, int> = 0>
    JSON_HEDLEY_DEPRECATED_FOR(3.11.0, basic_json::json_pointer or nlohmann::json_pointer<basic_json::string_t>) // NOLINT(readability/alt_tokens)
    const_reference operator[](const ::nlohmann::json_pointer<BasicJsonType>& ptr) const
    {
        return ptr.get_unchecked(this);
    }

    /// @brief access specified element via JSON Pointer
    /// @sa https://json.nlohmann.me/api/basic_json/at/
    reference at(const json_pointer& ptr)
    {
        return ptr.get_checked(this);
    }

    template<typename BasicJsonType, detail::enable_if_t<detail::is_basic_json<BasicJsonType>::value, int> = 0>
    JSON_HEDLEY_DEPRECATED_FOR(3.11.0, basic_json::json_pointer or nlohmann::json_pointer<basic_json::string_t>) // NOLINT(readability/alt_tokens)
    reference at(const ::nlohmann::json_pointer<BasicJsonType>& ptr)
    {
        return ptr.get_checked(this);
    }

    /// @brief access specified element via JSON Pointer
    /// @sa https://json.nlohmann.me/api/basic_json/at/
    const_reference at(const json_pointer& ptr) const
    {
        return ptr.get_checked(this);
    }

    template<typename BasicJsonType, detail::enable_if_t<detail::is_basic_json<BasicJsonType>::value, int> = 0>
    JSON_HEDLEY_DEPRECATED_FOR(3.11.0, basic_json::json_pointer or nlohmann::json_pointer<basic_json::string_t>) // NOLINT(readability/alt_tokens)
    const_reference at(const ::nlohmann::json_pointer<BasicJsonType>& ptr) const
    {
        return ptr.get_checked(this);
    }

    /// @brief return flattened JSON value
    /// @sa https://json.nlohmann.me/api/basic_json/flatten/
    basic_json flatten() const
    {
        basic_json result(value_t::object);
        json_pointer::flatten("", *this, result);
        return result;
    }

    /// @brief unflatten a previously flattened JSON value
    /// @sa https://json.nlohmann.me/api/basic_json/unflatten/
    basic_json unflatten() const
    {
        return json_pointer::unflatten(*this);
    }

    /// @}

    //////////////////////////
    // JSON Patch functions //
    //////////////////////////

    /// @name JSON Patch functions
    /// @{

    /// @brief applies a JSON patch in-place without copying the object
    /// @sa https://json.nlohmann.me/api/basic_json/patch/
    void patch_inplace(const basic_json& json_patch)
    {
        basic_json& result = *this;
        // the valid JSON Patch operations
        enum class patch_operations {add, remove, replace, move, copy, test, invalid};

        const auto get_op = [](const string_t& op)
        {
            if (op == "add")
            {
                return patch_operations::add;
            }
            if (op == "remove")
            {
                return patch_operations::remove;
            }
            if (op == "replace")
            {
                return patch_operations::replace;
            }
            if (op == "move")
            {
                return patch_operations::move;
            }
            if (op == "copy")
            {
                return patch_operations::copy;
            }
            if (op == "test")
            {
                return patch_operations::test;
            }

            return patch_operations::invalid;
        };

        // wrapper for "add" operation; add value at ptr
        const auto operation_add = [&result](json_pointer & ptr, const basic_json & val)
        {
            // adding to the root of the target document means replacing it
            if (ptr.empty())
            {
                result = val;
                return;
            }

            // make sure the top element of the pointer exists
            json_pointer const top_pointer = ptr.top();
            if (top_pointer != ptr)
            {
                result.at(top_pointer);
            }

            // get reference to parent of JSON pointer ptr
            const auto last_path = ptr.back();
            ptr.pop_back();
            // parent must exist when performing patch add per RFC6902 specs
            basic_json& parent = result.at(ptr);

            switch (parent.m_data.m_type)
            {
                case value_t::null:
                case value_t::object:
                {
                    // use operator[] to add value
                    parent[last_path] = val;
                    break;
                }

                case value_t::array:
                {
                    if (last_path == "-")
                    {
                        // special case: append to back
                        parent.push_back(val);
                    }
                    else
                    {
                        const auto idx = json_pointer::template array_index<basic_json_t>(last_path);
                        if (JSON_HEDLEY_UNLIKELY(idx > parent.size()))
                        {
                            // avoid undefined behavior
                            JSON_THROW(out_of_range::create(401, detail::concat("array index ", std::to_string(idx), " is out of range"), &parent));
                        }

                        // default case: insert add offset
                        parent.insert(parent.begin() + static_cast<difference_type>(idx), val);
                    }
                    break;
                }

                // if there exists a parent it cannot be primitive
                case value_t::string: // LCOV_EXCL_LINE
                case value_t::boolean: // LCOV_EXCL_LINE
                case value_t::number_integer: // LCOV_EXCL_LINE
                case value_t::number_unsigned: // LCOV_EXCL_LINE
                case value_t::number_float: // LCOV_EXCL_LINE
                case value_t::binary: // LCOV_EXCL_LINE
                case value_t::discarded: // LCOV_EXCL_LINE
                default:            // LCOV_EXCL_LINE
                    JSON_ASSERT(false); // NOLINT(cert-dcl03-c,hicpp-static-assert,misc-static-assert) LCOV_EXCL_LINE
            }
        };

        // wrapper for "remove" operation; remove value at ptr
        const auto operation_remove = [this, & result](json_pointer & ptr)
        {
            // get reference to parent of JSON pointer ptr
            const auto last_path = ptr.back();
            ptr.pop_back();
            basic_json& parent = result.at(ptr);

            // remove child
            if (parent.is_object())
            {
                // perform range check
                auto it = parent.find(last_path);
                if (JSON_HEDLEY_LIKELY(it != parent.end()))
                {
                    parent.erase(it);
                }
                else
                {
                    JSON_THROW(out_of_range::create(403, detail::concat("key '", last_path, "' not found"), this));
                }
            }
            else if (parent.is_array())
            {
                // note erase performs range check
                parent.erase(json_pointer::template array_index<basic_json_t>(last_path));
            }
        };

        // type check: top level value must be an array
        if (JSON_HEDLEY_UNLIKELY(!json_patch.is_array()))
        {
            JSON_THROW(parse_error::create(104, 0, "JSON patch must be an array of objects", &json_patch));
        }

        // iterate and apply the operations
        for (const auto& val : json_patch)
        {
            // wrapper to get a value for an operation
            const auto get_value = [&val](const string_t& op,
                                          const string_t& member,
                                          bool string_type) -> basic_json &
            {
                // find value
                auto it = val.m_data.m_value.object->find(member);

                // context-sensitive error message
                const auto error_msg = (op == "op") ? "operation" : detail::concat("operation '", op, '\''); // NOLINT(bugprone-unused-local-non-trivial-variable)

                // check if desired value is present
                if (JSON_HEDLEY_UNLIKELY(it == val.m_data.m_value.object->end()))
                {
                    // NOLINTNEXTLINE(performance-inefficient-string-concatenation)
                    JSON_THROW(parse_error::create(105, 0, detail::concat(error_msg, " must have member '", member, "'"), &val));
                }

                // check if result is of type string
                if (JSON_HEDLEY_UNLIKELY(string_type && !it->second.is_string()))
                {
                    // NOLINTNEXTLINE(performance-inefficient-string-concatenation)
                    JSON_THROW(parse_error::create(105, 0, detail::concat(error_msg, " must have string member '", member, "'"), &val));
                }

                // no error: return value
                return it->second;
            };

            // type check: every element of the array must be an object
            if (JSON_HEDLEY_UNLIKELY(!val.is_object()))
            {
                JSON_THROW(parse_error::create(104, 0, "JSON patch must be an array of objects", &val));
            }

            // collect mandatory members
            const auto op = get_value("op", "op", true).template get<string_t>();
            const auto path = get_value(op, "path", true).template get<string_t>();
            json_pointer ptr(path);

            switch (get_op(op))
            {
                case patch_operations::add:
                {
                    operation_add(ptr, get_value("add", "value", false));
                    break;
                }

                case patch_operations::remove:
                {
                    operation_remove(ptr);
                    break;
                }

                case patch_operations::replace:
                {
                    // the "path" location must exist - use at()
                    result.at(ptr) = get_value("replace", "value", false);
                    break;
                }

                case patch_operations::move:
                {
                    const auto from_path = get_value("move", "from", true).template get<string_t>();
                    json_pointer from_ptr(from_path);

                    // the "from" location must exist - use at()
                    basic_json const v = result.at(from_ptr);

                    // The move operation is functionally identical to a
                    // "remove" operation on the "from" location, followed
                    // immediately by an "add" operation at the target
                    // location with the value that was just removed.
                    operation_remove(from_ptr);
                    operation_add(ptr, v);
                    break;
                }

                case patch_operations::copy:
                {
                    const auto from_path = get_value("copy", "from", true).template get<string_t>();
                    const json_pointer from_ptr(from_path);

                    // the "from" location must exist - use at()
                    basic_json const v = result.at(from_ptr);

                    // The copy is functionally identical to an "add"
                    // operation at the target location using the value
                    // specified in the "from" member.
                    operation_add(ptr, v);
                    break;
                }

                case patch_operations::test:
                {
                    bool success = false;
                    JSON_TRY
                    {
                        // check if "value" matches the one at "path"
                        // the "path" location must exist - use at()
                        success = (result.at(ptr) == get_value("test", "value", false));
                    }
                    JSON_INTERNAL_CATCH (out_of_range&)
                    {
                        // ignore out of range errors: success remains false
                    }

                    // throw an exception if test fails
                    if (JSON_HEDLEY_UNLIKELY(!success))
                    {
                        JSON_THROW(other_error::create(501, detail::concat("unsuccessful: ", val.dump()), &val));
                    }

                    break;
                }

                case patch_operations::invalid:
                default:
                {
                    // op must be "add", "remove", "replace", "move", "copy", or
                    // "test"
                    JSON_THROW(parse_error::create(105, 0, detail::concat("operation value '", op, "' is invalid"), &val));
                }
            }
        }
    }

    /// @brief applies a JSON patch to a copy of the current object
    /// @sa https://json.nlohmann.me/api/basic_json/patch/
    basic_json patch(const basic_json& json_patch) const
    {
        basic_json result = *this;
        result.patch_inplace(json_patch);
        return result;
    }

    /// @brief creates a diff as a JSON patch
    /// @sa https://json.nlohmann.me/api/basic_json/diff/
    JSON_HEDLEY_WARN_UNUSED_RESULT
    static basic_json diff(const basic_json& source, const basic_json& target,
                           const string_t& path = "")
    {
        // the patch
        basic_json result(value_t::array);

        // if the values are the same, return empty patch
        if (source == target)
        {
            return result;
        }

        if (source.type() != target.type())
        {
            // different types: replace value
            result.push_back(
            {
                {"op", "replace"}, {"path", path}, {"value", target}
            });
            return result;
        }

        switch (source.type())
        {
            case value_t::array:
            {
                // first pass: traverse common elements
                std::size_t i = 0;
                while (i < source.size() && i < target.size())
                {
                    // recursive call to compare array values at index i
                    auto temp_diff = diff(source[i], target[i], detail::concat<string_t>(path, '/', detail::to_string<string_t>(i)));
                    result.insert(result.end(), temp_diff.begin(), temp_diff.end());
                    ++i;
                }

                // We now reached the end of at least one array
                // in a second pass, traverse the remaining elements

                // remove my remaining elements
                const auto end_index = static_cast<difference_type>(result.size());
                while (i < source.size())
                {
                    // add operations in reverse order to avoid invalid
                    // indices
                    result.insert(result.begin() + end_index, object(
                    {
                        {"op", "remove"},
                        {"path", detail::concat<string_t>(path, '/', detail::to_string<string_t>(i))}
                    }));
                    ++i;
                }

                // add other remaining elements
                while (i < target.size())
                {
                    result.push_back(
                    {
                        {"op", "add"},
                        {"path", detail::concat<string_t>(path, "/-")},
                        {"value", target[i]}
                    });
                    ++i;
                }

                break;
            }

            case value_t::object:
            {
                // first pass: traverse this object's elements
                for (auto it = source.cbegin(); it != source.cend(); ++it)
                {
                    // escape the key name to be used in a JSON patch
                    const auto path_key = detail::concat<string_t>(path, '/', detail::escape(it.key()));

                    if (target.find(it.key()) != target.end())
                    {
                        // recursive call to compare object values at key it
                        auto temp_diff = diff(it.value(), target[it.key()], path_key);
                        result.insert(result.end(), temp_diff.begin(), temp_diff.end());
                    }
                    else
                    {
                        // found a key that is not in o -> remove it
                        result.push_back(object(
                        {
                            {"op", "remove"}, {"path", path_key}
                        }));
                    }
                }

                // second pass: traverse other object's elements
                for (auto it = target.cbegin(); it != target.cend(); ++it)
                {
                    if (source.find(it.key()) == source.end())
                    {
                        // found a key that is not in this -> add it
                        const auto path_key = detail::concat<string_t>(path, '/', detail::escape(it.key()));
                        result.push_back(
                        {
                            {"op", "add"}, {"path", path_key},
                            {"value", it.value()}
                        });
                    }
                }

                break;
            }

            case value_t::null:
            case value_t::string:
            case value_t::boolean:
            case value_t::number_integer:
            case value_t::number_unsigned:
            case value_t::number_float:
            case value_t::binary:
            case value_t::discarded:
            default:
            {
                // both primitive type: replace value
                result.push_back(
                {
                    {"op", "replace"}, {"path", path}, {"value", target}
                });
                break;
            }
        }

        return result;
    }
    /// @}

    ////////////////////////////////
    // JSON Merge Patch functions //
    ////////////////////////////////

    /// @name JSON Merge Patch functions
    /// @{

    /// @brief applies a JSON Merge Patch
    /// @sa https://json.nlohmann.me/api/basic_json/merge_patch/
    void merge_patch(const basic_json& apply_patch)
    {
        if (apply_patch.is_object())
        {
            if (!is_object())
            {
                *this = object();
            }
            for (auto it = apply_patch.begin(); it != apply_patch.end(); ++it)
            {
                if (it.value().is_null())
                {
                    erase(it.key());
                }
                else
                {
                    operator[](it.key()).merge_patch(it.value());
                }
            }
        }
        else
        {
            *this = apply_patch;
        }
    }

    /// @}
};

/// @brief user-defined to_string function for JSON values
/// @sa https://json.nlohmann.me/api/basic_json/to_string/
NLOHMANN_BASIC_JSON_TPL_DECLARATION
std::string to_string(const NLOHMANN_BASIC_JSON_TPL& j)
{
    return j.dump();
}

inline namespace literals
{
inline namespace json_literals
{

/// @brief user-defined string literal for JSON values
/// @sa https://json.nlohmann.me/api/basic_json/operator_literal_json/
JSON_HEDLEY_NON_NULL(1)
#if !defined(JSON_HEDLEY_GCC_VERSION) || JSON_HEDLEY_GCC_VERSION_CHECK(4,9,0)
    inline nlohmann::json operator ""_json(const char* s, std::size_t n)
#else
    inline nlohmann::json operator "" _json(const char* s, std::size_t n)
#endif
{
    return nlohmann::json::parse(s, s + n);
}

/// @brief user-defined string literal for JSON pointer
/// @sa https://json.nlohmann.me/api/basic_json/operator_literal_json_pointer/
JSON_HEDLEY_NON_NULL(1)
#if !defined(JSON_HEDLEY_GCC_VERSION) || JSON_HEDLEY_GCC_VERSION_CHECK(4,9,0)
    inline nlohmann::json::json_pointer operator ""_json_pointer(const char* s, std::size_t n)
#else
    inline nlohmann::json::json_pointer operator "" _json_pointer(const char* s, std::size_t n)
#endif
{
    return nlohmann::json::json_pointer(std::string(s, n));
}

}  // namespace json_literals
}  // namespace literals
NLOHMANN_JSON_NAMESPACE_END

///////////////////////
// nonmember support //
///////////////////////

namespace std // NOLINT(cert-dcl58-cpp)
{

/// @brief hash value for JSON objects
/// @sa https://json.nlohmann.me/api/basic_json/std_hash/
NLOHMANN_BASIC_JSON_TPL_DECLARATION
struct hash<nlohmann::NLOHMANN_BASIC_JSON_TPL> // NOLINT(cert-dcl58-cpp)
{
    std::size_t operator()(const nlohmann::NLOHMANN_BASIC_JSON_TPL& j) const
    {
        return nlohmann::detail::hash(j);
    }
};

// specialization for std::less<value_t>
template<>
struct less< ::nlohmann::detail::value_t> // do not remove the space after '<', see https://github.com/nlohmann/json/pull/679
{
    /*!
    @brief compare two value_t enum values
    @since version 3.0.0
    */
    bool operator()(::nlohmann::detail::value_t lhs,
                    ::nlohmann::detail::value_t rhs) const noexcept
    {
#if JSON_HAS_THREE_WAY_COMPARISON
        return std::is_lt(lhs <=> rhs); // *NOPAD*
#else
        return ::nlohmann::detail::operator<(lhs, rhs);
#endif
    }
};

// C++20 prohibit function specialization in the std namespace.
#ifndef JSON_HAS_CPP_20

/// @brief exchanges the values of two JSON objects
/// @sa https://json.nlohmann.me/api/basic_json/std_swap/
NLOHMANN_BASIC_JSON_TPL_DECLARATION
inline void swap(nlohmann::NLOHMANN_BASIC_JSON_TPL& j1, nlohmann::NLOHMANN_BASIC_JSON_TPL& j2) noexcept(  // NOLINT(readability-inconsistent-declaration-parameter-name, cert-dcl58-cpp)
    is_nothrow_move_constructible<nlohmann::NLOHMANN_BASIC_JSON_TPL>::value&&                          // NOLINT(misc-redundant-expression,cppcoreguidelines-noexcept-swap,performance-noexcept-swap)
    is_nothrow_move_assignable<nlohmann::NLOHMANN_BASIC_JSON_TPL>::value)
{
    j1.swap(j2);
}

#endif

}  // namespace std

#if JSON_USE_GLOBAL_UDLS
    #if !defined(JSON_HEDLEY_GCC_VERSION) || JSON_HEDLEY_GCC_VERSION_CHECK(4,9,0)
        using nlohmann::literals::json_literals::operator ""_json; // NOLINT(misc-unused-using-decls,google-global-names-in-headers)
        using nlohmann::literals::json_literals::operator ""_json_pointer; //NOLINT(misc-unused-using-decls,google-global-names-in-headers)
    #else
        using nlohmann::literals::json_literals::operator "" _json; // NOLINT(misc-unused-using-decls,google-global-names-in-headers)
        using nlohmann::literals::json_literals::operator "" _json_pointer; //NOLINT(misc-unused-using-decls,google-global-names-in-headers)
    #endif
#endif

// #include <nlohmann/detail/macro_unscope.hpp>
//     __ _____ _____ _____
//  __|  |   __|     |   | |  JSON for Modern C++
// |  |  |__   |  |  | | | |  version 3.12.0
// |_____|_____|_____|_|___|  https://github.com/nlohmann/json
//
// SPDX-FileCopyrightText: 2013 - 2025 Niels Lohmann <https://nlohmann.me>
// SPDX-License-Identifier: MIT



// restore clang diagnostic settings
#if defined(__clang__)
    #pragma clang diagnostic pop
#endif

// clean up
#undef JSON_ASSERT
#undef JSON_INTERNAL_CATCH
#undef JSON_THROW
#undef JSON_PRIVATE_UNLESS_TESTED
#undef NLOHMANN_BASIC_JSON_TPL_DECLARATION
#undef NLOHMANN_BASIC_JSON_TPL
#undef JSON_EXPLICIT
#undef NLOHMANN_CAN_CALL_STD_FUNC_IMPL
#undef JSON_INLINE_VARIABLE
#undef JSON_NO_UNIQUE_ADDRESS
#undef JSON_DISABLE_ENUM_SERIALIZATION
#undef JSON_USE_GLOBAL_UDLS

#ifndef JSON_TEST_KEEP_MACROS
    #undef JSON_CATCH
    #undef JSON_TRY
    #undef JSON_HAS_CPP_11
    #undef JSON_HAS_CPP_14
    #undef JSON_HAS_CPP_17
    #undef JSON_HAS_CPP_20
    #undef JSON_HAS_CPP_23
    #undef JSON_HAS_FILESYSTEM
    #undef JSON_HAS_EXPERIMENTAL_FILESYSTEM
    #undef JSON_HAS_THREE_WAY_COMPARISON
    #undef JSON_HAS_RANGES
    #undef JSON_HAS_STATIC_RTTI
    #undef JSON_USE_LEGACY_DISCARDED_VALUE_COMPARISON
#endif

// #include <nlohmann/thirdparty/hedley/hedley_undef.hpp>
//     __ _____ _____ _____
//  __|  |   __|     |   | |  JSON for Modern C++
// |  |  |__   |  |  | | | |  version 3.12.0
// |_____|_____|_____|_|___|  https://github.com/nlohmann/json
//
// SPDX-FileCopyrightText: 2013 - 2025 Niels Lohmann <https://nlohmann.me>
// SPDX-License-Identifier: MIT



#undef JSON_HEDLEY_ALWAYS_INLINE
#undef JSON_HEDLEY_ARM_VERSION
#undef JSON_HEDLEY_ARM_VERSION_CHECK
#undef JSON_HEDLEY_ARRAY_PARAM
#undef JSON_HEDLEY_ASSUME
#undef JSON_HEDLEY_BEGIN_C_DECLS
#undef JSON_HEDLEY_CLANG_HAS_ATTRIBUTE
#undef JSON_HEDLEY_CLANG_HAS_BUILTIN
#undef JSON_HEDLEY_CLANG_HAS_CPP_ATTRIBUTE
#undef JSON_HEDLEY_CLANG_HAS_DECLSPEC_DECLSPEC_ATTRIBUTE
#undef JSON_HEDLEY_CLANG_HAS_EXTENSION
#undef JSON_HEDLEY_CLANG_HAS_FEATURE
#undef JSON_HEDLEY_CLANG_HAS_WARNING
#undef JSON_HEDLEY_COMPCERT_VERSION
#undef JSON_HEDLEY_COMPCERT_VERSION_CHECK
#undef JSON_HEDLEY_CONCAT
#undef JSON_HEDLEY_CONCAT3
#undef JSON_HEDLEY_CONCAT3_EX
#undef JSON_HEDLEY_CONCAT_EX
#undef JSON_HEDLEY_CONST
#undef JSON_HEDLEY_CONSTEXPR
#undef JSON_HEDLEY_CONST_CAST
#undef JSON_HEDLEY_CPP_CAST
#undef JSON_HEDLEY_CRAY_VERSION
#undef JSON_HEDLEY_CRAY_VERSION_CHECK
#undef JSON_HEDLEY_C_DECL
#undef JSON_HEDLEY_DEPRECATED
#undef JSON_HEDLEY_DEPRECATED_FOR
#undef JSON_HEDLEY_DIAGNOSTIC_DISABLE_CAST_QUAL
#undef JSON_HEDLEY_DIAGNOSTIC_DISABLE_CPP98_COMPAT_WRAP_
#undef JSON_HEDLEY_DIAGNOSTIC_DISABLE_DEPRECATED
#undef JSON_HEDLEY_DIAGNOSTIC_DISABLE_UNKNOWN_CPP_ATTRIBUTES
#undef JSON_HEDLEY_DIAGNOSTIC_DISABLE_UNKNOWN_PRAGMAS
#undef JSON_HEDLEY_DIAGNOSTIC_DISABLE_UNUSED_FUNCTION
#undef JSON_HEDLEY_DIAGNOSTIC_POP
#undef JSON_HEDLEY_DIAGNOSTIC_PUSH
#undef JSON_HEDLEY_DMC_VERSION
#undef JSON_HEDLEY_DMC_VERSION_CHECK
#undef JSON_HEDLEY_EMPTY_BASES
#undef JSON_HEDLEY_EMSCRIPTEN_VERSION
#undef JSON_HEDLEY_EMSCRIPTEN_VERSION_CHECK
#undef JSON_HEDLEY_END_C_DECLS
#undef JSON_HEDLEY_FLAGS
#undef JSON_HEDLEY_FLAGS_CAST
#undef JSON_HEDLEY_GCC_HAS_ATTRIBUTE
#undef JSON_HEDLEY_GCC_HAS_BUILTIN
#undef JSON_HEDLEY_GCC_HAS_CPP_ATTRIBUTE
#undef JSON_HEDLEY_GCC_HAS_DECLSPEC_ATTRIBUTE
#undef JSON_HEDLEY_GCC_HAS_EXTENSION
#undef JSON_HEDLEY_GCC_HAS_FEATURE
#undef JSON_HEDLEY_GCC_HAS_WARNING
#undef JSON_HEDLEY_GCC_NOT_CLANG_VERSION_CHECK
#undef JSON_HEDLEY_GCC_VERSION
#undef JSON_HEDLEY_GCC_VERSION_CHECK
#undef JSON_HEDLEY_GNUC_HAS_ATTRIBUTE
#undef JSON_HEDLEY_GNUC_HAS_BUILTIN
#undef JSON_HEDLEY_GNUC_HAS_CPP_ATTRIBUTE
#undef JSON_HEDLEY_GNUC_HAS_DECLSPEC_ATTRIBUTE
#undef JSON_HEDLEY_GNUC_HAS_EXTENSION
#undef JSON_HEDLEY_GNUC_HAS_FEATURE
#undef JSON_HEDLEY_GNUC_HAS_WARNING
#undef JSON_HEDLEY_GNUC_VERSION
#undef JSON_HEDLEY_GNUC_VERSION_CHECK
#undef JSON_HEDLEY_HAS_ATTRIBUTE
#undef JSON_HEDLEY_HAS_BUILTIN
#undef JSON_HEDLEY_HAS_CPP_ATTRIBUTE
#undef JSON_HEDLEY_HAS_CPP_ATTRIBUTE_NS
#undef JSON_HEDLEY_HAS_DECLSPEC_ATTRIBUTE
#undef JSON_HEDLEY_HAS_EXTENSION
#undef JSON_HEDLEY_HAS_FEATURE
#undef JSON_HEDLEY_HAS_WARNING
#undef JSON_HEDLEY_IAR_VERSION
#undef JSON_HEDLEY_IAR_VERSION_CHECK
#undef JSON_HEDLEY_IBM_VERSION
#undef JSON_HEDLEY_IBM_VERSION_CHECK
#undef JSON_HEDLEY_IMPORT
#undef JSON_HEDLEY_INLINE
#undef JSON_HEDLEY_INTEL_CL_VERSION
#undef JSON_HEDLEY_INTEL_CL_VERSION_CHECK
#undef JSON_HEDLEY_INTEL_VERSION
#undef JSON_HEDLEY_INTEL_VERSION_CHECK
#undef JSON_HEDLEY_IS_CONSTANT
#undef JSON_HEDLEY_IS_CONSTEXPR_
#undef JSON_HEDLEY_LIKELY
#undef JSON_HEDLEY_MALLOC
#undef JSON_HEDLEY_MCST_LCC_VERSION
#undef JSON_HEDLEY_MCST_LCC_VERSION_CHECK
#undef JSON_HEDLEY_MESSAGE
#undef JSON_HEDLEY_MSVC_VERSION
#undef JSON_HEDLEY_MSVC_VERSION_CHECK
#undef JSON_HEDLEY_NEVER_INLINE
#undef JSON_HEDLEY_NON_NULL
#undef JSON_HEDLEY_NO_ESCAPE
#undef JSON_HEDLEY_NO_RETURN
#undef JSON_HEDLEY_NO_THROW
#undef JSON_HEDLEY_NULL
#undef JSON_HEDLEY_PELLES_VERSION
#undef JSON_HEDLEY_PELLES_VERSION_CHECK
#undef JSON_HEDLEY_PGI_VERSION
#undef JSON_HEDLEY_PGI_VERSION_CHECK
#undef JSON_HEDLEY_PREDICT
#undef JSON_HEDLEY_PRINTF_FORMAT
#undef JSON_HEDLEY_PRIVATE
#undef JSON_HEDLEY_PUBLIC
#undef JSON_HEDLEY_PURE
#undef JSON_HEDLEY_REINTERPRET_CAST
#undef JSON_HEDLEY_REQUIRE
#undef JSON_HEDLEY_REQUIRE_CONSTEXPR
#undef JSON_HEDLEY_REQUIRE_MSG
#undef JSON_HEDLEY_RESTRICT
#undef JSON_HEDLEY_RETURNS_NON_NULL
#undef JSON_HEDLEY_SENTINEL
#undef JSON_HEDLEY_STATIC_ASSERT
#undef JSON_HEDLEY_STATIC_CAST
#undef JSON_HEDLEY_STRINGIFY
#undef JSON_HEDLEY_STRINGIFY_EX
#undef JSON_HEDLEY_SUNPRO_VERSION
#undef JSON_HEDLEY_SUNPRO_VERSION_CHECK
#undef JSON_HEDLEY_TINYC_VERSION
#undef JSON_HEDLEY_TINYC_VERSION_CHECK
#undef JSON_HEDLEY_TI_ARMCL_VERSION
#undef JSON_HEDLEY_TI_ARMCL_VERSION_CHECK
#undef JSON_HEDLEY_TI_CL2000_VERSION
#undef JSON_HEDLEY_TI_CL2000_VERSION_CHECK
#undef JSON_HEDLEY_TI_CL430_VERSION
#undef JSON_HEDLEY_TI_CL430_VERSION_CHECK
#undef JSON_HEDLEY_TI_CL6X_VERSION
#undef JSON_HEDLEY_TI_CL6X_VERSION_CHECK
#undef JSON_HEDLEY_TI_CL7X_VERSION
#undef JSON_HEDLEY_TI_CL7X_VERSION_CHECK
#undef JSON_HEDLEY_TI_CLPRU_VERSION
#undef JSON_HEDLEY_TI_CLPRU_VERSION_CHECK
#undef JSON_HEDLEY_TI_VERSION
#undef JSON_HEDLEY_TI_VERSION_CHECK
#undef JSON_HEDLEY_UNAVAILABLE
#undef JSON_HEDLEY_UNLIKELY
#undef JSON_HEDLEY_UNPREDICTABLE
#undef JSON_HEDLEY_UNREACHABLE
#undef JSON_HEDLEY_UNREACHABLE_RETURN
#undef JSON_HEDLEY_VERSION
#undef JSON_HEDLEY_VERSION_DECODE_MAJOR
#undef JSON_HEDLEY_VERSION_DECODE_MINOR
#undef JSON_HEDLEY_VERSION_DECODE_REVISION
#undef JSON_HEDLEY_VERSION_ENCODE
#undef JSON_HEDLEY_WARNING
#undef JSON_HEDLEY_WARN_UNUSED_RESULT
#undef JSON_HEDLEY_WARN_UNUSED_RESULT_MSG
#undef JSON_HEDLEY_FALL_THROUGH



#endif  // INCLUDE_NLOHMANN_JSON_HPP_

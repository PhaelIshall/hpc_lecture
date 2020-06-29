/******************************************************************************
 * Copyright (c) 2017, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#pragma once

/**
 * \file
 * \brief C++ features that may be otherwise unimplemented for CUDA device functions.
 *
 * This file has three components:
 *
 *   (1) Macros:
 *       - Empty macro defines for C++ keywords not supported by the current
 *         version of C++. These simply allow compilation to proceed (but do
 *         not provide the added semantics).
 *           - \p noexcept
 *           - \p constexpr
 *           - \p nullptr
 *           - \p static_assert
 *
 *       - Macro functions that we need in constant expressions because the
 *         C++ equivalents require constexpr compiler support.  These are
 *         prefixed with \p __NV_STD_*
 *           - \p __NV_STD_MAX
 *           - \p __NV_STD_MIN
 *   (2) Stop-gap implementations of unsupported STL functions and types:
 *       - STL functions and types defined by C++ 11/14/17/etc. that are not
 *         provided by the current version of C++. These are placed into the
 *         \p nv_std namespace
 *           - \p integral_constant
 *           - \p nullptr_t
 *           - \p true_type
 *           - \p false_type
 *           - \p bool_constant
 *           - \p enable_if
 *           - \p conditional
 *           - \p is_same
 *           - \p is_base_of
 *           - \p remove_const
 *           - \p remove_volatile
 *           - \p remove_cv
 *           - \p is_volatile
 *           - \p is_pointer
 *           - \p is_void
 *           - \p is_integral
 *           - \p is_floating_point
 *           - \p is_arithmetic
 *           - \p is_fundamental
 *           - \p is_trivially_copyable
 *           - \p alignment_of
 *           - \p aligned_storage
 *
 *   (3) Functions and types that are STL-like (but aren't in the STL):
 *           - \p TODO: min and max functors?
 *
 * The idea is that, as we drop support for older compilers, we can simply #define
 * the \p __NV_STD_XYZ macros and \p nv_std namespace to alias their C++
 * counterparts (or trivially find-and-replace their occurrences in code text).
 */


//-----------------------------------------------------------------------------
// Include STL files that nv_std provides functionality for
//-----------------------------------------------------------------------------

#include <cstddef>          // nullptr_t
#include <algorithm>        // Minimum/maximum operations
#include <functional>       // Arithmetic operations
#include <utility>          // For methods on std::pair
#if (!defined(_MSC_VER) && (__cplusplus >= 201103L)) || (defined(_MSC_VER) && (_MS_VER >= 1500))
    #include <type_traits>  // For integral constants, conditional metaprogramming, and type traits
#endif



/******************************************************************************
 * Macros
 ******************************************************************************/
//-----------------------------------------------------------------------------
// Keywords
//-----------------------------------------------------------------------------

/// noexcept, constexpr
#if (!defined(_MSC_VER) && (__cplusplus < 201103L)) || (defined(_MSC_VER) && (_MSC_VER < 1900))
    #ifndef noexcept
        #define noexcept
    #endif
    #ifndef constexpr
        #define constexpr
    #endif
#endif

/// nullptr
#if (!defined(_MSC_VER) && (__cplusplus < 201103L)) || (defined(_MSC_VER) && (_MSC_VER < 1310 ))
    #ifndef nullptr
        #define nullptr 0
    #endif
#endif

/// static_assert
#if (!defined(_MSC_VER) && (__cplusplus < 201103L)) || (defined(_MSC_VER) && (_MSC_VER < 1600 ))
    #ifndef static_assert
        #define __nv_std_cat_(a, b) a ## b
        #define __nv_std_cat(a, b) __nv_std_cat_(a, b)
        #define static_assert(__e, __m) typedef int __nv_std_cat(AsSeRt, __LINE__)[(__e) ? 1 : -1]
    #endif
#endif


//-----------------------------------------------------------------------------
// Functions
//-----------------------------------------------------------------------------

/// Select maximum(a, b)
#ifndef __NV_STD_MAX
    #define __NV_STD_MAX(a, b) (((b) > (a)) ? (b) : (a))
#endif

/// Select minimum(a, b)
#ifndef __NV_STD_MIN
    #define __NV_STD_MIN(a, b) (((b) < (a)) ? (b) : (a))
#endif

namespace nv_std {
using std::integral_constant;
    using std::pair;

    /// The type used as a compile-time boolean with true value.
    typedef integral_constant<bool, true>   true_type;

    /// The type used as a compile-time boolean with false value.
    typedef integral_constant<bool, false>  false_type;


#if (!defined(_MSC_VER) && (__cplusplus < 201402L)) || (defined(_MSC_VER) && (_MSC_VER < 1900))
    template <bool V>
    struct bool_constant : nv_std::integral_constant<bool, V>
    {};
#else
    using std::bool_constant;
#endif
    using std::nullptr_t;
    using std::enable_if;
    using std::conditional;
    using std::remove_const;
    using std::remove_volatile;
    using std::remove_cv;
    using std::is_same;
    using std::is_base_of;
    using std::is_volatile;
    using std::is_pointer;
    using std::is_void;
    using std::is_integral;
    using std::is_floating_point;
    using std::is_arithmetic;
    using std::is_fundamental;
    using std::is_trivially_copyable;
    using std::aligned_storage;
    template <typename value_t>
    struct alignment_of : std::alignment_of<value_t> {};
    /* 16B specializations where 32-bit Win32 host compiler disagrees with device compiler */
    template <> struct alignment_of<int4>                 { enum { value = 16 }; };
    template <> struct alignment_of<uint4>                { enum { value = 16 }; };
    template <> struct alignment_of<float4>               { enum { value = 16 }; };
    template <> struct alignment_of<long4>                { enum { value = 16 }; };
    template <> struct alignment_of<ulong4>               { enum { value = 16 }; };
    template <> struct alignment_of<longlong2>            { enum { value = 16 }; };
    template <> struct alignment_of<ulonglong2>           { enum { value = 16 }; };
    template <> struct alignment_of<double2>              { enum { value = 16 }; };
    template <> struct alignment_of<longlong4>            { enum { value = 16 }; };
    template <> struct alignment_of<ulonglong4>           { enum { value = 16 }; };
    template <> struct alignment_of<double4>              { enum { value = 16 }; };
    template <typename value_t> struct alignment_of<volatile value_t>       : alignment_of<value_t> {};
    template <typename value_t> struct alignment_of<const value_t>          : alignment_of<value_t> {};
    template <typename value_t> struct alignment_of<const volatile value_t> : alignment_of<value_t> {};
}; // namespace nv_std


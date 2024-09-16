/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
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

/**
 * @file
 * Thread reduction over statically-sized array-like types
 */

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/detail/array_utils.cuh> // to_array()
#include <cub/detail/type_traits.cuh> // are_same()
#include <cub/thread/thread_operators.cuh> // cub_operator_to_dpx_t
#include <cub/util_namespace.cuh> // CUB_NAMESPACE_BEGIN

#include <cuda/std/bit> // bit_cast
#include <cuda/std/cstdint> // uint16_t

CUB_NAMESPACE_BEGIN

// forward declaration
/**
 * @brief Reduction over statically-sized array-like types.
 *
 * @tparam Input
 *   <b>[inferred]</b> The data type to be reduced having member
 *   <tt>operator[](int i)</tt> and must be statically-sized (size() method or static array)
 *
 * @tparam ReductionOp
 *   <b>[inferred]</b> Binary reduction operator type having member
 *   <tt>T operator()(const T &a, const T &b)</tt>
 *
 * @param[in] input
 *   Input array
 *
 * @param[in] reduction_op
 *   Binary reduction operator
 *
 * @return Aggregate of type <tt>cuda::std::__accumulator_t<ReductionOp, ValueT, PrefixT></tt>
 */
template <typename Input,
          typename ReductionOp,
#ifndef DOXYGEN_SHOULD_SKIP_THIS // Do not document
          typename ValueT = ::cuda::std::__remove_cvref_t<decltype(::cuda::std::declval<Input>()[0])>,
#endif // !DOXYGEN_SHOULD_SKIP_THIS
          typename AccumT = ::cuda::std::__accumulator_t<ReductionOp, ValueT>>
_CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE AccumT ThreadReduce(const Input& input, ReductionOp reduction_op);

/***********************************************************************************************************************
 * Internal Reduction Implementations
 **********************************************************************************************************************/

/// Internal namespace (to prevent ADL mishaps between static functions when mixing different CUB installations)
namespace internal
{

#ifndef DOXYGEN_SHOULD_SKIP_THIS // Do not document

/***********************************************************************************************************************
 * Enable SIMD/Tree reduction heuristics
 **********************************************************************************************************************/

/// DPX instructions compute min, max, and sum for up to three 16 and 32-bit signed or unsigned integer parameters
/// see DPX documetation https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#dpx
/// NOTE: The compiler is able to automatically vectorize all cases with 3 operands
///       However, all other cases with per-halfword comparison need to be explicitly vectorized
///
/// DPX reduction is enabled if the following conditions are met:
/// - Hopper+ architectures. DPX instructions are emulated before Hopper
/// - The number of elements must be large enough for performance reasons (see below)
/// - All types must be the same
/// - Only works with integral types of 2 bytes
/// - DPX instructions provide Min, Max SIMD operations
/// If the number of instructions is the same, we favor the compiler
///
/// length | Standard |  DPX
///  2     |    1     |  NA
///  3     |    1     |  NA
///  4     |    2     |  3
///  5     |    2     |  3
///  6     |    3     |  3
///  7     |    3     |  3
///  8     |    4     |  4
///  9     |    4     |  4
/// 10     |    5     |  4 // ***
/// 11     |    5     |  4 // ***
/// 12     |    6     |  5 // ***
/// 13     |    6     |  5 // ***
/// 14     |    7     |  5 // ***
/// 15     |    7     |  5 // ***
/// 16     |    8     |  6 // ***

// TODO: add Blackwell support

template <typename T, typename ReductionOp, int Length>
_CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE constexpr bool enable_sm90_simd_reduction()
{
  using cub::detail::is_one_of;
  // cub::Sum not handled: IADD3 always produces less instructions than VIADD2
  return is_one_of<T, ::cuda::std::int16_t, ::cuda::std::uint16_t>() && //
         is_one_of<ReductionOp, cub::Min, cub::Max>() && Length >= 10;
}

template <typename T, typename ReductionOp, int Length>
_CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE constexpr bool enable_sm80_simd_reduction()
{
  using cub::detail::is_one_of;
  using ::cuda::std::is_same;
#  if _CCCL_HAS_NVFP16 && _CCCL_HAS_NVBF16
  return ((is_same<T, __nv_bfloat16>::value && is_one_of<ReductionOp, cub::Sum, cub::Mul>())
          || (is_one_of<T, __half, __nv_bfloat16>() && is_one_of<ReductionOp, cub::Min, cub::Max>()))
      && Length >= 4;
#  else
  return false;
#  endif
}

template <typename T, typename ReductionOp, int Length>
_CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE constexpr bool enable_sm70_simd_reduction()
{
  using cub::detail::is_one_of;
  using ::cuda::std::is_same;
#  if _CCCL_HAS_NVFP16
  return is_same<T, __half>::value && is_one_of<ReductionOp, cub::Sum, cub::Mul>() && Length >= 4;
#  else
  return false;
#  endif
}

template <typename Input, typename ReductionOp, typename AccumT>
_CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE _CCCL_CONSTEXPR_CXX14 bool enable_simd_reduction()
{
  using cub::detail::is_one_of;
  using ::cuda::std::is_same;
  using T = decltype(::cuda::std::declval<Input>()[0]);
  if _CCCL_CONSTEXPR_CXX17 (!is_same<T, AccumT>::value)
  {
    return false;
  }
  else
  {
    constexpr auto length = cub::detail::static_size<Input>();
    // clang-format off
    _NV_TARGET_DISPATCH(
      NV_PROVIDES_SM_90,
        (enable_sm90_simd_reduction<T, ReductionOp, length>() || enable_sm80_simd_reduction<T, ReductionOp, length>() ||
         enable_sm70_simd_reduction<T, ReductionOp, length>())
      (NV_PROVIDES_SM_80,
        (enable_sm80_simd_reduction<T, ReductionOp, length>() || enable_sm70_simd_reduction<T, ReductionOp, length>()))
      (NV_PROVIDES_SM_70,
        enable_sm70_simd_reduction<T, ReductionOp, length>())
      (return false;)
    );
    // clang-format on
  }
}

template <typename Input, typename ReductionOp, typename AccumT>
_CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE _CCCL_CONSTEXPR_CXX14 bool enable_ternary_reduction()
{
  using cub::detail::is_one_of;
  using ::cuda::std::is_same;
  using T               = decltype(::cuda::std::declval<Input>()[0]);
  constexpr auto length = cub::detail::static_size<Input>();
  if _CCCL_CONSTEXPR_CXX17 (!is_same<T, AccumT>::value || length < 6)
  {
    return false;
  }
  else
  {
    // clang-format off
    _NV_TARGET_DISPATCH(
      NV_PROVIDES_SM_90,
        (is_one_of<T, ::cuda::std::int32_t, ::cuda::std::uint32_t, ::cuda::std::int64_t, ::cuda::std::uint64_t,
                   __half2, __nv_bfloat162>() &&
         is_one_of<ReductionOp, cub::Min, cub::Max>())
      (NV_PROVIDES_SM_70,
        (is_one_of<T, ::cuda::std::int32_t, ::cuda::std::uint32_t, ::cuda::std::int64_t, ::cuda::std::uint64_t>() &&
         is_one_of<ReductionOp, cub::Sum, cub::BitAnd, cub::BitOr, cub::BitXor>()))
      (return false;)
    );
    // clang-format on
  }
}

template <typename Input, typename ReductionOp, typename AccumT>
_CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE constexpr bool enable_promotion()
{
  using cub::detail::is_one_of;
  using ::cuda::std::is_same;
  using T = decltype(::cuda::std::declval<Input>()[0]);
  return ::cuda::std::is_integral<T>::value && sizeof(T) <= 2
      && is_one_of<ReductionOp, cub::Sum, cub::Mul, cub::BitAnd, cub::BitOr, cub::BitXor, cub::Max, cub::Min>();
}

/***********************************************************************************************************************
 * Internal Reduction Algorithms: Sequential, Binary, Ternary
 **********************************************************************************************************************/

template <typename AccumT, typename Input, typename ReductionOp>
_CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE AccumT
ThreadReduceSequential(const Input& input, ReductionOp reduction_op)
{
  AccumT retval = input[0];
#  pragma unroll
  for (int i = 1; i < cub::detail::static_size<Input>(); ++i)
  {
    retval = reduction_op(retval, input[i]);
  }
  return retval;
}

template <typename AccumT, typename Input, typename ReductionOp>
_CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE AccumT
ThreadReduceBinaryTree(const Input& input, ReductionOp reduction_op)
{
  constexpr auto length = cub::detail::static_size<Input>();
  auto array            = cub::detail::to_array<AccumT>(input);
#  pragma unroll
  for (int i = 1; i < length; i *= 2)
  {
#  pragma unroll
    for (int j = 0; j + i < length; j += i * 2)
    {
      array[j] = reduction_op(array[j], array[j + i]);
    }
  }
  return array[0];
}

template <typename AccumT, typename Input, typename ReductionOp>
_CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE AccumT
ThreadReduceTernaryTree(const Input& input, ReductionOp reduction_op)
{
  constexpr auto length = cub::detail::static_size<Input>();
  auto array            = cub::detail::to_array<AccumT>(input);
#  pragma unroll
  for (int i = 1; i < length; i *= 3)
  {
#  pragma unroll
    for (int j = 0; j + i < length; j += i * 3)
    {
      array[j] = (j + i * 2 < length) ? reduction_op(array[j], reduction_op(array[j + i], array[j + i * 2]))
                                      : reduction_op(array[j], array[j + i]);
    }
  }
  return array[0];
}

/***********************************************************************************************************************
 * SIMD Reduction
 **********************************************************************************************************************/

template <typename Input, typename ReductionOp>
_CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE auto
ThreadReduceSimd(const Input& input, ReductionOp reduction_op) -> ::cuda::std::__remove_cvref_t<decltype(input[0])>
{
  using T                       = ::cuda::std::__remove_cvref_t<decltype(input[0])>;
  using SimdReduceOp            = cub_operator_to_simd_operator_t<ReductionOp, T>;
  constexpr auto simd_ratio     = SimdReduceOp::ratio;
  constexpr auto length         = cub::detail::static_size<Input>();
  constexpr auto length_rounded = (length / simd_ratio) * simd_ratio; // TODO: replace with round_up()
  using ArrayRounded            = T[length_rounded];
  using SimdType                = simd_type_t<T>;
  using SimdArray               = ::cuda::std::array<SimdType, simd_ratio>;
  using UnpackedType            = ::cuda::std::array<T, simd_ratio>;
  // TODO: switch to std::span when C++11 is dropped
  auto simd_input      = ::cuda::std::bit_cast<SimdArray>(*reinterpret_cast<const ArrayRounded*>(input));
  auto simd_reduction  = ThreadReduce(simd_input, SimdReduceOp{});
  auto unpacked_values = ::cuda::std::bit_cast<UnpackedType>(simd_reduction);
  // TODO: extend to simd_ratio > 2 if needed with a SWAR butterfly reduction
  static_assert(simd_ratio <= 2, "Only SIMD size <= 2 is supported");
  if _CCCL_CONSTEXPR_CXX17 (simd_ratio == 1)
  {
    return unpacked_values[0];
  }
  else // if _CCCL_CONSTEXPR_CXX17 (simd_ratio == 2)
  {
    // Create a reversed copy of the SIMD reduction result and apply the SIMD operator.
    // This avoids redundant instructions for converting to and from 32-bit register size
    SimdArray unpacked_values_rev{unpacked_values[1], unpacked_values[0]};
    auto simd_reduction_rev = ::cuda::std::bit_cast<SimdType>(unpacked_values_rev);
    auto result             = SimdReduceOp{}(simd_reduction, simd_reduction_rev);
    // repeat the same optimization for the last element
    if _CCCL_CONSTEXPR_CXX17 (length % simd_ratio == 1)
    {
      SimdArray tail{input[length - 1], 0};
      auto tail_simd = ::cuda::std::bit_cast<SimdType>(tail);
      result         = SimdReduceOp{}(result, tail_simd);
    }
    return ::cuda::std::bit_cast<UnpackedType>(result)[0];
  }
}
#endif // !DOXYGEN_SHOULD_SKIP_THIS

} // namespace internal

/***********************************************************************************************************************
 * Reduction Interface/Dispatch (public)
 **********************************************************************************************************************/

template <typename Input, typename ReductionOp, typename ValueT, typename AccumT>
_CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE AccumT ThreadReduce(const Input& input, ReductionOp reduction_op)
{
  static_assert(cub::detail::has_subscript<Input>::value, "Input must support the subscript operator[]");
  static_assert(cub::detail::has_size<Input>::value, "Input must support the constexpr size() method");
  static_assert(cub::detail::has_binary_call_operator<ReductionOp, ValueT>::value,
                "ReductionOp must have the binary call operator: operator(ValueT, ValueT)");
  using cub::internal::enable_promotion;
  using cub::internal::enable_simd_reduction;
  using cub::internal::enable_ternary_reduction;
  using PromT = ::cuda::std::_If<enable_promotion<Input, ReductionOp, AccumT>(), decltype(+AccumT{}), AccumT>;
  if _CCCL_CONSTEXPR_CXX17 (enable_simd_reduction<Input, ReductionOp, AccumT>())
  {
    return cub::internal::ThreadReduceSimd(input, reduction_op);
  }
  else if _CCCL_CONSTEXPR_CXX17 (enable_ternary_reduction<Input, ReductionOp, PromT>())
  {
    return cub::internal::ThreadReduceTernaryTree<PromT>(input, reduction_op);
  }
  else
  {
    return cub::internal::ThreadReduceBinaryTree<PromT>(input, reduction_op);
  }
}

/**
 * @brief Reduction over statically-sized array-like types, seeded with the specified @p prefix.
 *
 * @tparam Input
 *   <b>[inferred]</b> The data type to be reduced having member
 *   <tt>operator[](int i)</tt> and must be statically-sized (size() method or static array)
 *
 * @tparam ReductionOp
 *   <b>[inferred]</b> Binary reduction operator type having member
 *   <tt>T operator()(const T &a, const T &b)</tt>
 *
 * @tparam PrefixT
 *   <b>[inferred]</b> The prefix type
 *
 * @param[in] input
 *   Input array
 *
 * @param[in] reduction_op
 *   Binary reduction operator
 *
 * @param[in] prefix
 *   Prefix to seed reduction with
 *
 * @return Aggregate of type <tt>cuda::std::__accumulator_t<ReductionOp, ValueT, PrefixT></tt>
 */
template <typename Input,
          typename ReductionOp,
          typename PrefixT,
#ifndef DOXYGEN_SHOULD_SKIP_THIS // Do not document
          typename ValueT = ::cuda::std::__remove_cvref_t<decltype(::cuda::std::declval<Input>()[0])>,
#endif // !DOXYGEN_SHOULD_SKIP_THIS
          typename AccumT = ::cuda::std::__accumulator_t<ReductionOp, ValueT, PrefixT>>
_CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE AccumT
ThreadReduce(const Input& input, ReductionOp reduction_op, PrefixT prefix)
{
  static_assert(detail::has_subscript<Input>::value, "Input must support the subscript operator[]");
  static_assert(detail::has_size<Input>::value, "Input must support the constexpr size() method");
  static_assert(detail::has_binary_call_operator<ReductionOp, ValueT>::value,
                "ReductionOp must have the binary call operator: operator(ValueT, ValueT)");
  constexpr int length = cub::detail::static_size<Input>();
  // copy to a temporary array of type AccumT
  AccumT array[length + 1];
  array[0] = prefix;
#pragma unroll
  for (int i = 0; i < length; ++i)
  {
    array[i + 1] = input[i];
  }
  return ThreadReduce<decltype(array), ReductionOp, AccumT, AccumT>(array, reduction_op);
}

/***********************************************************************************************************************
 * Pointer Interfaces with explicit Length (internal use only)
 *********************************************************************************************************************
 */

/// Internal namespace (to prevent ADL mishaps between static functions when mixing different CUB installations)
namespace internal
{

#ifndef DOXYGEN_SHOULD_SKIP_THIS // Do not document

/**
 * @remark The pointer interface adds little value and requires Length to be explicit.
 *         Prefer using the array-like interface
 *
 * @brief Perform a sequential reduction over @p length elements of the @p input pointer. The aggregate is returned.
 *
 * @tparam T
 *   <b>[inferred]</b> The data type to be reduced
 *
 * @tparam ReductionOp
 *   <b>[inferred]</b> Binary reduction operator type having member
 *   <tt>T operator()(const T &a, const T &b)</tt>
 *
 * @param[in] input
 *   Input pointer
 *
 * @param[in] reduction_op
 *   Binary reduction operator
 *
 * @return Aggregate of type <tt>cuda::std::__accumulator_t<ReductionOp, T></tt>
 */
template <int Length, typename T, typename ReductionOp, typename AccumT = ::cuda::std::__accumulator_t<ReductionOp, T>>
_CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE AccumT ThreadReduce(const T* input, ReductionOp reduction_op)
{
  static_assert(Length > 0, "Length must be greater than 0");
  static_assert(cub::detail::has_binary_call_operator<ReductionOp, T>::value,
                "ReductionOp must have the binary call operator: operator(V1, V2)");
  using ArrayT = T[Length];
  auto array   = reinterpret_cast<const T(*)[Length]>(input);
  return ThreadReduce(*array, reduction_op);
}

/**
 * @remark The pointer interface adds little value and requires Length to be explicit.
 *         Prefer using the array-like interface
 *
 * @brief Perform a sequential reduction over @p length elements of the @p input pointer, seeded with the specified @p
 *        prefix. The aggregate is returned.
 *
 * @tparam length
 *   Length of input pointer
 *
 * @tparam T
 *   <b>[inferred]</b> The data type to be reduced
 *
 * @tparam ReductionOp
 *   <b>[inferred]</b> Binary reduction operator type having member
 *   <tt>T operator()(const T &a, const T &b)</tt>
 *
 * @tparam PrefixT
 *   <b>[inferred]</b> The prefix type
 *
 * @param[in] input
 *   Input pointer
 *
 * @param[in] reduction_op
 *   Binary reduction operator
 *
 * @param[in] prefix
 *   Prefix to seed reduction with
 *
 * @return Aggregate of type <tt>cuda::std::__accumulator_t<ReductionOp, T, PrefixT></tt>
 */
template <int Length,
          typename T,
          typename ReductionOp,
          typename PrefixT,
          typename AccumT = ::cuda::std::__accumulator_t<ReductionOp, T, PrefixT>,
          _CUB_TEMPLATE_REQUIRES(Length > 0)>
_CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE AccumT
ThreadReduce(const T* input, ReductionOp reduction_op, PrefixT prefix)
{
  static_assert(detail::has_binary_call_operator<ReductionOp, T>::value,
                "ReductionOp must have the binary call operator: operator(V1, V2)");
  auto array = reinterpret_cast<const T(*)[Length]>(input);
  return ThreadReduce(*array, reduction_op, prefix);
}

template <int Length, typename T, typename ReductionOp, typename PrefixT, _CUB_TEMPLATE_REQUIRES(Length == 0)>
_CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE T ThreadReduce(const T*, ReductionOp, PrefixT prefix)
{
  return prefix;
}

#endif // !DOXYGEN_SHOULD_SKIP_THIS

} // namespace internal
CUB_NAMESPACE_END

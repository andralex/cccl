/******************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

#include <cub/thread/thread_reduce.cuh>
#include <cub/util_macro.cuh>

#include <thrust/iterator/constant_iterator.h>

#include <cuda/std/functional>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include <functional>
#include <limits>
#include <numeric>
#include <string>
#include <tuple>
#include <type_traits>

#include "bfloat16.h"
#include "c2h/custom_type.cuh"
#include "catch2_test_helper.h"
#include "cub/detail/type_traits.cuh"
#include "half.h"

template <int NUM_ITEMS, typename T, typename ReduceOperator>
__global__ void thread_reduce_kernel(const T* d_in, T* d_out, ReduceOperator reduce_operator)
{
  T thread_data[NUM_ITEMS];
#pragma unroll
  for (int i = 0; i < NUM_ITEMS; ++i)
  {
    thread_data[i] = d_in[i];
  }
  *d_out = cub::ThreadReduce(thread_data, reduce_operator);
}

/***********************************************************************************************************************
 * CUB operator to STD operator
 **********************************************************************************************************************/

template <typename T, typename>
struct cub_operator_to_std;

template <typename T>
struct cub_operator_to_std<T, cub::Sum>
{
  using type = ::std::plus<T>;
};

template <typename T>
struct cub_operator_to_std<T, cub::Mul>
{
  using type = ::std::multiplies<T>;
};

template <typename T>
struct cub_operator_to_std<T, cub::BitAnd>
{
  using type = ::std::bit_and<T>;
};

template <typename T>
struct cub_operator_to_std<T, cub::BitOr>
{
  using type = ::std::bit_or<T>;
};

template <typename T>
struct cub_operator_to_std<T, cub::BitXor>
{
  using type = ::std::bit_xor<T>;
};

struct min_operator
{
  template <typename T>
  T operator()(const T& valueA, const T& valueB)
  {
    return ::std::min(valueA, valueB);
  }
};

struct max_operator
{
  template <typename T>
  T operator()(const T& valueA, const T& valueB)
  {
    return ::std::max(valueA, valueB);
  }
};

template <typename T>
struct cub_operator_to_std<T, cub::Min>
{
  using type = min_operator;
};

template <typename T>
struct cub_operator_to_std<T, cub::Max>
{
  using type = max_operator;
};

template <typename T, typename Operator>
using cub_operator_to_std_t = typename cub_operator_to_std<T, Operator>::type;

/***********************************************************************************************************************
 * CUB operator to identity
 **********************************************************************************************************************/

template <typename T, typename Operator, typename = void>
struct cub_operator_to_identity;

template <typename T>
struct cub_operator_to_identity<T, cub::Sum>
{
  static constexpr T value()
  {
    return T{};
  };
};

template <typename T>
struct cub_operator_to_identity<T, cub::Mul>
{
  static constexpr T value()
  {
    return T{1};
  };
};

template <typename T>
struct cub_operator_to_identity<T, cub::BitAnd>
{
  static constexpr T value()
  {
    return ~T{0};
  };
};

template <typename T>
struct cub_operator_to_identity<T, cub::BitOr>
{
  static constexpr T value()
  {
    return T{0};
  };
};

template <typename T>
struct cub_operator_to_identity<T, cub::BitXor>
{
  static constexpr T value()
  {
    return T{0};
  };
};

template <typename T>
struct cub_operator_to_identity<T, cub::Min>
{
  static constexpr T value()
  {
    return ::std::numeric_limits<T>::max();
  };
};

template <typename T>
struct cub_operator_to_identity<T, cub::Max>
{
  static constexpr T value()
  {
    return ::std::numeric_limits<T>::min();
  };
};

/***********************************************************************************************************************
 * Type list definition
 **********************************************************************************************************************/

using narrow_precision_type_list = c2h::type_list<
#ifdef TEST_HALF_T
  half_t,
#endif
#ifdef TEST_BF_T
  bfloat16_t
#endif
  >;

using fp_type_list = c2h::type_list<float, double>;

using integral_type_list = c2h::
  type_list<::cuda::std::int8_t, ::cuda::std::int16_t, ::cuda::std::uint16_t, ::cuda::std::int32_t, ::cuda::std::int64_t>;

using cub_operator_integral_list =
  c2h::type_list<cub::Sum, cub::Mul, cub::BitAnd, cub::BitOr, cub::BitXor, cub::Min, cub::Max>;

using cub_operator_fp_list = c2h::type_list<cub::Sum, cub::Mul, cub::Min, cub::Max>;

/***********************************************************************************************************************
 * Verify results and kernel launch
 **********************************************************************************************************************/

template <typename T, _CUB_TEMPLATE_REQUIRES(::cuda::std::is_floating_point<T>::value)>
void verify_results(const T& expected_data, const T& test_results)
{
  REQUIRE(expected_data == Approx(test_results));
}

template <typename T, _CUB_TEMPLATE_REQUIRES(!::cuda::std::is_floating_point<T>::value)>
void verify_results(const T& expected_data, const T& test_results)
{
  REQUIRE(expected_data == test_results);
}

template <typename T, typename ReduceOperator>
void run_thread_reduce_kernel(
  int num_items, const c2h::device_vector<T>& in, c2h::device_vector<T>& out, ReduceOperator reduce_operator)
{
  switch (num_items)
  {
    case 1:
      thread_reduce_kernel<1>
        <<<1, 1>>>(thrust::raw_pointer_cast(in.data()), thrust::raw_pointer_cast(out.data()), reduce_operator);
      break;
    case 2:
      thread_reduce_kernel<2>
        <<<1, 1>>>(thrust::raw_pointer_cast(in.data()), thrust::raw_pointer_cast(out.data()), reduce_operator);
      break;
    case 3:
      thread_reduce_kernel<3>
        <<<1, 1>>>(thrust::raw_pointer_cast(in.data()), thrust::raw_pointer_cast(out.data()), reduce_operator);
      break;
    case 4:
      thread_reduce_kernel<4>
        <<<1, 1>>>(thrust::raw_pointer_cast(in.data()), thrust::raw_pointer_cast(out.data()), reduce_operator);
      break;
    case 5:
      thread_reduce_kernel<5>
        <<<1, 1>>>(thrust::raw_pointer_cast(in.data()), thrust::raw_pointer_cast(out.data()), reduce_operator);
      break;
    case 6:
      thread_reduce_kernel<6>
        <<<1, 1>>>(thrust::raw_pointer_cast(in.data()), thrust::raw_pointer_cast(out.data()), reduce_operator);
      break;
    case 7:
      thread_reduce_kernel<7>
        <<<1, 1>>>(thrust::raw_pointer_cast(in.data()), thrust::raw_pointer_cast(out.data()), reduce_operator);
      break;
    case 8:
      thread_reduce_kernel<8>
        <<<1, 1>>>(thrust::raw_pointer_cast(in.data()), thrust::raw_pointer_cast(out.data()), reduce_operator);
      break;
    case 9:
      thread_reduce_kernel<9>
        <<<1, 1>>>(thrust::raw_pointer_cast(in.data()), thrust::raw_pointer_cast(out.data()), reduce_operator);
      break;
    case 10:
      thread_reduce_kernel<10>
        <<<1, 1>>>(thrust::raw_pointer_cast(in.data()), thrust::raw_pointer_cast(out.data()), reduce_operator);
      break;
    case 11:
      thread_reduce_kernel<11>
        <<<1, 1>>>(thrust::raw_pointer_cast(in.data()), thrust::raw_pointer_cast(out.data()), reduce_operator);
      break;
    case 12:
      thread_reduce_kernel<12>
        <<<1, 1>>>(thrust::raw_pointer_cast(in.data()), thrust::raw_pointer_cast(out.data()), reduce_operator);
      break;
    case 13:
      thread_reduce_kernel<13>
        <<<1, 1>>>(thrust::raw_pointer_cast(in.data()), thrust::raw_pointer_cast(out.data()), reduce_operator);
      break;
    case 14:
      thread_reduce_kernel<14>
        <<<1, 1>>>(thrust::raw_pointer_cast(in.data()), thrust::raw_pointer_cast(out.data()), reduce_operator);
      break;
    case 15:
      thread_reduce_kernel<15>
        <<<1, 1>>>(thrust::raw_pointer_cast(in.data()), thrust::raw_pointer_cast(out.data()), reduce_operator);
      break;
    default:
      FAIL("Unsupported number of items");
  }
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
}

static constexpr int max_size = 16;

/***********************************************************************************************************************
 * Test cases
 **********************************************************************************************************************/

CUB_TEST("ThreadReduce Integral Type Tests", "[reduce][thread]", integral_type_list, cub_operator_integral_list)
{
  using value_t                    = c2h::get<0, TestType>;
  constexpr auto reduce_op         = c2h::get<1, TestType>{};
  constexpr auto std_reduce_op     = cub_operator_to_std_t<value_t, c2h::get<1, TestType>>{};
  constexpr auto operator_identity = cub_operator_to_identity<value_t, c2h::get<1, TestType>>::value();
  CAPTURE(c2h::type_name<value_t>(), max_size, c2h::type_name<decltype(reduce_op)>());
  c2h::device_vector<value_t> d_in(max_size);
  c2h::device_vector<value_t> d_out(1);
  c2h::gen(CUB_SEED(10), d_in, std::numeric_limits<value_t>::min());
  c2h::host_vector<value_t> h_in = d_in;
  for (int num_items = 1; num_items < max_size; ++num_items)
  {
    auto reference_result = std::accumulate(h_in.begin(), h_in.begin() + num_items, operator_identity, std_reduce_op);
    run_thread_reduce_kernel(num_items, d_in, d_out, reduce_op);
    verify_results(reference_result, c2h::host_vector<value_t>(d_out)[0]);
  }
}

CUB_TEST("ThreadReduce Floating-Point Type Tests", "[reduce][thread]", fp_type_list, cub_operator_fp_list)
{
  using value_t                = c2h::get<0, TestType>;
  constexpr auto reduce_op     = c2h::get<1, TestType>{};
  constexpr auto std_reduce_op = cub_operator_to_std_t<value_t, c2h::get<1, TestType>>{};
  const auto operator_identity = cub_operator_to_identity<value_t, c2h::get<1, TestType>>::value();
  CAPTURE(c2h::type_name<value_t>(), max_size, c2h::type_name<decltype(reduce_op)>());
  c2h::device_vector<value_t> d_in(max_size);
  c2h::device_vector<value_t> d_out(1);
  c2h::gen(CUB_SEED(10), d_in, std::numeric_limits<value_t>::min());
  c2h::host_vector<value_t> h_in = d_in;
  for (int num_items = 1; num_items < max_size; ++num_items)
  {
    auto reference_result = std::accumulate(h_in.begin(), h_in.begin() + num_items, operator_identity, std_reduce_op);
    run_thread_reduce_kernel(num_items, d_in, d_out, reduce_op);
    verify_results(reference_result, c2h::host_vector<value_t>(d_out)[0]);
  }
}

CUB_TEST("ThreadReduce Narrow PrecisionType Tests", "[reduce][thread]", narrow_precision_type_list, cub_operator_fp_list)
{
  using value_t                = c2h::get<0, TestType>;
  constexpr auto reduce_op     = c2h::get<1, TestType>{};
  constexpr auto std_reduce_op = cub_operator_to_std_t<float, c2h::get<1, TestType>>{};
  const auto operator_identity = cub_operator_to_identity<float, c2h::get<1, TestType>>::value();
  CAPTURE(c2h::type_name<value_t>(), max_size, c2h::type_name<decltype(reduce_op)>());
  c2h::device_vector<value_t> d_in(max_size);
  c2h::device_vector<value_t> d_out(1);
  c2h::gen(CUB_SEED(10), d_in, std::numeric_limits<value_t>::min());
  c2h::host_vector<float> h_in_float = d_in;
  for (int num_items = 1; num_items < max_size; ++num_items)
  {
    auto reference_result =
      std::accumulate(h_in_float.begin(), h_in_float.begin() + num_items, operator_identity, std_reduce_op);
    run_thread_reduce_kernel(num_items, d_in, d_out, reduce_op);
    verify_results(reference_result, float{c2h::host_vector<value_t>(d_out)[0]});
  }
}

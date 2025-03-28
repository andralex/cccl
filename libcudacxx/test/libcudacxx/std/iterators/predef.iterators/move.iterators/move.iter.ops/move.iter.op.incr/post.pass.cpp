//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/iterator>

// move_iterator

// constexpr auto operator++(int); // Return type was move_iterator until C++20

#include <cuda/std/cassert>
#include <cuda/std/iterator>
#include <cuda/std/utility>

#include "test_iterators.h"
#include "test_macros.h"

template <class It>
__host__ __device__ void test_single_pass(It i, It x)
{
  cuda::std::move_iterator<It> r(cuda::std::move(i));
  r++;
  assert(cuda::std::move(r).base() == x);
}

template <class It>
__host__ __device__ void test(It i, It x)
{
  cuda::std::move_iterator<It> r(i);
  cuda::std::move_iterator<It> rr = r++;
  assert(r.base() == x);
  assert(rr.base() == i);
}

int main(int, char**)
{
  char s[] = "123";
  test_single_pass(cpp17_input_iterator<char*>(s), cpp17_input_iterator<char*>(s + 1));
  test(forward_iterator<char*>(s), forward_iterator<char*>(s + 1));
  test(bidirectional_iterator<char*>(s), bidirectional_iterator<char*>(s + 1));
  test(random_access_iterator<char*>(s), random_access_iterator<char*>(s + 1));
  test(s, s + 1);

  {
    constexpr const char* p = "123456789";
    typedef cuda::std::move_iterator<const char*> MI;
    constexpr MI it1 = cuda::std::make_move_iterator(p);
    constexpr MI it2 = cuda::std::make_move_iterator(p + 1);
    static_assert(it1 != it2, "");
    constexpr MI it3 = cuda::std::make_move_iterator(p)++;
    static_assert(it1 == it3, "");
    static_assert(it2 != it3, "");
  }

  // Forward iterators return a copy.
  {
    int a[]        = {1, 2, 3};
    using MoveIter = cuda::std::move_iterator<forward_iterator<int*>>;

    MoveIter i = MoveIter(forward_iterator<int*>(a));
    static_assert(cuda::std::is_same_v<decltype(i++), MoveIter>);
    auto j = i++;
    assert(base(j.base()) == a);
    assert(base(i.base()) == a + 1);
  }

  // Non-forward iterators return void.
  {
    int a[]        = {1, 2, 3};
    using MoveIter = cuda::std::move_iterator<cpp20_input_iterator<int*>>;

    MoveIter i = MoveIter(cpp20_input_iterator<int*>(a));
    static_assert(cuda::std::is_same_v<decltype(i++), void>);
    i++;
    assert(base(i.base()) == a + 1);
  }

  return 0;
}

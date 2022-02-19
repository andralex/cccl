//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/array>

// NVRTC won't be able to compile min_allocator
// UNSUPPORTED: nvrtc

// class array

// bool empty() const noexcept;

#include <cuda/std/array>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "min_allocator.h"

int main(int, char**)
{
    {
    typedef cuda::std::array<int, 2> C;
    C c;
    ASSERT_NOEXCEPT(c.empty());
    assert(!c.empty());
    }
    {
    typedef cuda::std::array<int, 0> C;
    C c;
    ASSERT_NOEXCEPT(c.empty());
    assert( c.empty());
    }

  return 0;
}

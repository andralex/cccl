#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Order independent GPU kernels with an STF **token** (no extra buffer).

Cholesky / POTRI / Warp examples focus on large numerical pipelines. This script
shows a smaller STF idea: a :meth:`cuda.stf.context.token` is logical data with
no payload—it exists only to add dependency edges. Here the same two arrays are
read/written in two Numba AXPY launches; the shared ``token.rw()`` dependency
forces the second kernel to run after the first even though their ``lX``/``lY``
patterns alone would not distinguish order.

Requires: NumPy, Numba with CUDA, and ``cuda.stf`` (experimental STF wheel).

Run from ``python/cuda_cccl_experimental/tests``::

    python stf/example_token_ordering.py
"""

from __future__ import annotations

import sys

import numpy as np
from numba import cuda

import cuda.stf as stf
from numba_helpers import get_arg_numba, numba_arguments

numba.cuda.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0


@cuda.jit
def _axpy(alpha, x, y):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(start, x.size, stride):
        y[i] = alpha * x[i] + y[i]


def run_token_ordered_axpy(n: int = 1 << 20, alpha: float = 2.0) -> float:
    """Run two AXPY passes on device memory ordered by a token; return final Y[0]."""
    x = np.ones(n, dtype=np.float32)
    y = np.ones(n, dtype=np.float32)

    ctx = stf.context()
    lx = ctx.logical_data(x, name="X")
    ly = ctx.logical_data(y, name="Y")
    order = ctx.token()

    blocks = 32
    threads = 256

    with ctx.task(lx.read(), ly.rw(), order.rw()) as t:
        stream = cuda.external_stream(t.stream_ptr())
        dx = get_arg_numba(t, 0)
        dy = get_arg_numba(t, 1)
        _axpy[blocks, threads, stream](alpha, dx, dy)

    with ctx.task(lx.read(), ly.rw(), order.rw()) as t:
        stream = cuda.external_stream(t.stream_ptr())
        dx, dy = numba_arguments(t)
        _axpy[blocks, threads, stream](alpha, dx, dy)

    ctx.finalize()
    return float(y[0])


def main() -> int:
    y0 = run_token_ordered_axpy()
    # First pass: y = alpha * 1 + 1 = alpha + 1. Second: y = alpha * 1 + (alpha+1) = 2*alpha + 1
    expected = 2.0 * 2.0 + 1.0
    if abs(y0 - expected) > 1e-3:
        print(f"unexpected Y[0]: got {y0}, expected ~{expected}")
        return 1
    print(f"token-ordered AXPY OK: Y[0] = {y0} (expected {expected})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

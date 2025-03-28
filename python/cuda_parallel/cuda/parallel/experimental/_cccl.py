# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from __future__ import annotations

import ctypes
import functools
from typing import Callable

import numba
import numpy as np
from numba import cuda, types

from ._utils.protocols import get_data_pointer, get_dtype, is_contiguous
from .iterators._iterators import IteratorBase
from .typing import DeviceArrayLike, GpuStruct


# MUST match `cccl_type_enum` in c/include/cccl/c/types.h
class TypeEnum(ctypes.c_int):
    INT8 = 0
    INT16 = 1
    INT32 = 2
    INT64 = 3
    UINT8 = 4
    UINT16 = 5
    UINT32 = 6
    UINT64 = 7
    FLOAT32 = 8
    FLOAT64 = 9
    STORAGE = 10


# MUST match `cccl_op_kind_t` in c/include/cccl/c/types.h
class OpKind(ctypes.c_int):
    STATELESS = 0
    STATEFUL = 1


# MUST match `cccl_iterator_kind_t` in c/include/cccl/c/types.h
class IteratorKind(ctypes.c_int):
    POINTER = 0
    ITERATOR = 1


# MUST match `cccl_type_info` in c/include/cccl/c/types.h
class TypeInfo(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_size_t),
        ("alignment", ctypes.c_size_t),
        ("type", TypeEnum),
    ]


# MUST match `cccl_op_t` in c/include/cccl/c/types.h
class Op(ctypes.Structure):
    _fields_ = [
        ("type", OpKind),
        ("name", ctypes.c_char_p),
        ("ltoir", ctypes.c_char_p),
        ("ltoir_size", ctypes.c_size_t),
        ("size", ctypes.c_size_t),
        ("alignment", ctypes.c_size_t),
        ("state", ctypes.c_void_p),
    ]


# MUST match `cccl_iterator_t` in c/include/cccl/c/types.h
class Iterator(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_size_t),
        ("alignment", ctypes.c_size_t),
        ("type", IteratorKind),
        ("advance", Op),
        ("dereference", Op),
        ("value_type", TypeInfo),
        ("state", ctypes.c_void_p),
    ]


# MUST match `cccl_device_merge_sort_build_result_t` in c/include/cccl/c/merge_sort.h
class DeviceMergeSortBuildResult(ctypes.Structure):
    _fields_ = [
        ("cc", ctypes.c_int),
        ("cubin", ctypes.c_void_p),
        ("cubin_size", ctypes.c_size_t),
        ("library", ctypes.c_void_p),
        ("block_sort_kernel", ctypes.c_void_p),
        ("partition_kernel", ctypes.c_void_p),
        ("merge_kernel", ctypes.c_void_p),
    ]


# MUST match `cccl_device_reduce_build_result_t` in c/include/cccl/c/reduce.h
class DeviceReduceBuildResult(ctypes.Structure):
    _fields_ = [
        ("cc", ctypes.c_int),
        ("cubin", ctypes.c_void_p),
        ("cubin_size", ctypes.c_size_t),
        ("library", ctypes.c_void_p),
        ("accumulator_size", ctypes.c_uint64),
        ("single_tile_kernel", ctypes.c_void_p),
        ("single_tile_second_kernel", ctypes.c_void_p),
        ("reduction_kernel", ctypes.c_void_p),
    ]


# MUST match `cccl_device_scan_build_result_t` in c/include/cccl/c/scan.h
class DeviceScanBuildResult(ctypes.Structure):
    _fields_ = [
        ("cc", ctypes.c_int),
        ("cubin", ctypes.c_void_p),
        ("cubin_size", ctypes.c_size_t),
        ("library", ctypes.c_void_p),
        ("accumulator_type", TypeInfo),
        ("init_kernel", ctypes.c_void_p),
        ("scan_kernel", ctypes.c_void_p),
        ("force_inclusive", ctypes.c_bool),
        ("description_bytes_per_tile", ctypes.c_size_t),
        ("payload_bytes_per_tile", ctypes.c_size_t),
    ]


# MUST match `cccl_device_segmented_reduce_build_result_t` in c/include/cccl/c/segmented_reduce.h
class DeviceSegmentedReduceBuildResult(ctypes.Structure):
    _fields_ = [
        ("cc", ctypes.c_int),
        ("cubin", ctypes.c_void_p),
        ("cubin_size", ctypes.c_size_t),
        ("library", ctypes.c_void_p),
        ("accumulator_size", ctypes.c_uint64),
        ("segmented_reduce_kernel", ctypes.c_void_p),
    ]


# MUST match `cccl_device_unique_by_key_build_result_t` in c/include/cccl/c/unique_by_key.h
class DeviceUniqueByKeyBuildResult(ctypes.Structure):
    _fields_ = [
        ("cc", ctypes.c_int),
        ("cubin", ctypes.c_void_p),
        ("cubin_size", ctypes.c_size_t),
        ("library", ctypes.c_void_p),
        ("compact_init_kernel", ctypes.c_void_p),
        ("sweep_kernel", ctypes.c_void_p),
        ("description_bytes_per_tile", ctypes.c_size_t),
        ("payload_bytes_per_tile", ctypes.c_size_t),
    ]


# MUST match `cccl_value_t` in c/include/cccl/c/types.h
class Value(ctypes.Structure):
    _fields_ = [("type", TypeInfo), ("state", ctypes.c_void_p)]


_TYPE_TO_ENUM = {
    types.int8: TypeEnum(TypeEnum.INT8),
    types.int16: TypeEnum(TypeEnum.INT16),
    types.int32: TypeEnum(TypeEnum.INT32),
    types.int64: TypeEnum(TypeEnum.INT64),
    types.uint8: TypeEnum(TypeEnum.UINT8),
    types.uint16: TypeEnum(TypeEnum.UINT16),
    types.uint32: TypeEnum(TypeEnum.UINT32),
    types.uint64: TypeEnum(TypeEnum.UINT64),
    types.float32: TypeEnum(TypeEnum.FLOAT32),
    types.float64: TypeEnum(TypeEnum.FLOAT64),
}


def _type_to_enum(numba_type: types.Type) -> TypeEnum:
    if numba_type in _TYPE_TO_ENUM:
        return _TYPE_TO_ENUM[numba_type]
    return TypeEnum(TypeEnum.STORAGE)


# TODO: replace with functools.cache once our docs build environment
# is upgraded to at least Python 3.9
@functools.lru_cache(maxsize=None)
def _numba_type_to_info(numba_type: types.Type) -> TypeInfo:
    context = cuda.descriptor.cuda_target.target_context
    value_type = context.get_value_type(numba_type)
    if isinstance(numba_type, types.Record):
        # then `value_type` is a pointer and we need the
        # alignment of the pointee.
        value_type = value_type.pointee
    size = value_type.get_abi_size(context.target_data)
    alignment = value_type.get_abi_alignment(context.target_data)
    return TypeInfo(size, alignment, _type_to_enum(numba_type))


@functools.lru_cache(maxsize=None)
def _numpy_type_to_info(numpy_type: np.dtype) -> TypeInfo:
    numba_type = numba.from_dtype(numpy_type)
    return _numba_type_to_info(numba_type)


def _device_array_to_cccl_iter(array: DeviceArrayLike) -> Iterator:
    if not is_contiguous(array):
        raise ValueError("Non-contiguous arrays are not supported.")
    info = _numpy_type_to_info(get_dtype(array))
    # state is a pointer, size and alignment of iterator
    # is that of a integral type that holds a pointer
    state_info = _numpy_type_to_info(np.dtype(np.uintp))
    return Iterator(
        state_info.size,
        state_info.alignment,
        IteratorKind.POINTER,
        Op(),
        Op(),
        info,
        # Note: this is slightly slower, but supports all ndarray-like objects
        # as long as they support CAI
        # TODO: switch to use gpumemoryview once it's ready
        array.__cuda_array_interface__["data"][0],
    )


def _iterator_to_cccl_iter(it: IteratorBase) -> Iterator:
    context = cuda.descriptor.cuda_target.target_context
    numba_type = it.numba_type
    size = context.get_value_type(numba_type).get_abi_size(context.target_data)
    alignment = context.get_value_type(numba_type).get_abi_alignment(
        context.target_data
    )
    (advance_abi_name, advance_ltoir), (deref_abi_name, deref_ltoir) = it.ltoirs.items()
    advance_op = Op(
        OpKind.STATELESS,
        advance_abi_name.encode("utf-8"),
        ctypes.c_char_p(advance_ltoir),
        len(advance_ltoir),
        1,
        1,
        None,
    )
    deref_op = Op(
        OpKind.STATELESS,
        deref_abi_name.encode("utf-8"),
        ctypes.c_char_p(deref_ltoir),
        len(deref_ltoir),
        1,
        1,
        None,
    )
    return Iterator(
        size,
        alignment,
        OpKind.STATEFUL,
        advance_op,
        deref_op,
        _numba_type_to_info(it.value_type),
        it.state,
    )


def _none_to_cccl_iter() -> Iterator:
    # Any type could be used here, we just need to pass NULL.
    info = _numpy_type_to_info(np.uint8)
    return Iterator(
        info.size,
        info.alignment,
        IteratorKind.POINTER,
        Op(),
        Op(),
        info,
        None,
    )


def type_enum_as_name(enum_value: int) -> str:
    return (
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float32",
        "float64",
        "STORAGE",
    )[enum_value]


def to_cccl_iter(array_or_iterator) -> Iterator:
    if array_or_iterator is None:
        return _none_to_cccl_iter()
    if isinstance(array_or_iterator, IteratorBase):
        return _iterator_to_cccl_iter(array_or_iterator)
    return _device_array_to_cccl_iter(array_or_iterator)


def to_cccl_value(array_or_struct: np.ndarray | GpuStruct) -> Value:
    if isinstance(array_or_struct, np.ndarray):
        info = _numpy_type_to_info(array_or_struct.dtype)
        data = ctypes.cast(array_or_struct.ctypes.data, ctypes.c_void_p)
        return Value(info, data)
    else:
        # it's a GpuStruct, use the array underlying it
        return to_cccl_value(array_or_struct._data)


def to_cccl_op(op: Callable, sig) -> Op:
    ltoir, _ = cuda.compile(op, sig=sig, output="ltoir")
    name = op.__name__.encode("utf-8")
    return Op(
        OpKind.STATELESS,
        name,
        ctypes.c_char_p(ltoir),
        len(ltoir),
        1,
        1,
        None,
        _data=(ltoir, name),  # keep a reference to these in a _data attribute
    )


def set_cccl_iterator_state(cccl_it: Iterator, input_it):
    if cccl_it.type.value == IteratorKind.POINTER:
        cccl_it.state = get_data_pointer(input_it)
    else:
        cccl_it.state = input_it.state

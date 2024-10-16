//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___UTILITY_TYPEID_H
#define _LIBCUDACXX___UTILITY_TYPEID_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

// BUGBUG
#define _CCCL_NO_TYPEID

#ifndef _LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR
#  include <cuda/std/compare>
#endif
#include <cuda/std/__utility/integer_sequence.h>
#include <cuda/std/cstddef>

#ifndef _CCCL_NO_TYPEID
#  include <typeinfo>
#endif

#ifndef _CCCL_NO_TYPEID

_LIBCUDACXX_BEGIN_NAMESPACE_STD
#  define _CCCL_TYPEID(...) typeid(__VA_ARGS__)
using __type_info = ::std::type_info;
_LIBCUDACXX_END_NAMESPACE_STD

#else // ^^^ !_CCCL_NO_TYPEID ^^^ / vvv _CCCL_NO_TYPEID

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// TODO: replace this with `cuda::std::string_view` when available.
struct __string_view
{
  template <size_t _Np>
  _LIBCUDACXX_HIDE_FROM_ABI constexpr explicit __string_view(
    char const (&str)[_Np], size_t __prefix = 0, size_t __suffix = 0) noexcept
      : __str_{str + __prefix}
      , __len_{_Np - 1 - __prefix - __suffix}
  {}

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr size_t size() const noexcept
  {
    return __len_;
  }

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr char const* data() const noexcept
  {
    return __str_;
  }

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr char const* begin() const noexcept
  {
    return __str_;
  }

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr char const* end() const noexcept
  {
    return __str_ + __len_;
  }

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr char const& operator[](ptrdiff_t __n) const noexcept
  {
    return __str_[__n];
  }

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr __string_view
  substr(ptrdiff_t __start, ptrdiff_t __stop) const noexcept
  {
    return __string_view(__str_ + __start, __stop - __start);
  }

  // C++11 constexpr string comparison
#  if _CCCL_STD_VER < 2014 || __cpp_constexpr < 201304L

private:
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr int
  __compare(char const* __s1, size_t __len1, char const* __s2, size_t __len2, size_t __n) noexcept
  {
    return __n
           ? ((*__s1 < *__s2) //
                ? -1
                : ((*__s2 < *__s1) //
                     ? 1
                     : __compare(__s1 + 1, __len1, __s2 + 1, __len2, __n - 1)))
           : int(__len1) - int(__len2);
  }

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr ptrdiff_t __try_find(
    size_t __i,
    const char* __needle,
    size_t __needle_size,
    char const* __haystack_begin,
    char const* __haystack_end,
    char const* __it) noexcept
  {
    return __i == __needle_size //
           ? __it - __haystack_begin
           : __needle[__i] != __it[__i]
               ? -1
               : __try_find(__i + 1, __needle, __needle_size, __haystack_begin, __haystack_end, __it);
  }

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr ptrdiff_t __find(
    ptrdiff_t __found,
    const char* __needle,
    size_t __needle_size,
    char const* __haystack_begin,
    char const* __haystack_end,
    char const* __it) noexcept
  {
    return __found != -1 //
           ? __found
           : __it == __haystack_end
               ? -1
               : __find(__try_find(0, __needle, __needle_size, __haystack_begin, __haystack_end, __it),
                        __needle,
                        __needle_size,
                        __haystack_begin,
                        __haystack_end,
                        __it + 1);
  }

public:
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr int compare(__string_view const& __other) const noexcept
  {
    return __compare(__str_, __len_, __other.__str_, __other.__len_, __min_(__len_, __other.__len_));
  }

  template <size_t _Np>
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr ptrdiff_t find(const char (&__other)[_Np]) const noexcept
  {
    return ((_Np - 1) > __len_) ? -1 : __find(-1, __other, _Np - 1, __str_, __str_ + __len_ - (_Np - 1), __str_);
  }

#  else // ^^^ C++11 ^^^ / vvv C++14 and beyond vvv

  template <size_t _Np>
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr ptrdiff_t find(const char (&__other)[_Np]) const noexcept
  {
    if ((_Np - 1) > __len_)
    {
      return -1;
    }

    char const *__it = begin(), *__end = end() - (_Np - 1);
    for (; __it != __end; ++__it)
    {
      size_t __i = 0;
      for (; __i < (_Np - 1); ++__i)
      {
        if (__it[__i] != __other[__i])
        {
          break;
        }
      }
      if (__i == (_Np - 1))
      {
        return __it - begin();
      }
    }
    return -1;
  }

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr int compare(__string_view const& __other) const noexcept
  {
    size_t __n       = __min_(__len_, __other.__len_);
    char const *__s1 = __str_, *__s2 = __other.__str_;
    for (; __n; --__n, ++__s1, ++__s2)
    {
      if (*__s1 < *__s2)
      {
        return -1;
      }
      if (*__s2 < *__s1)
      {
        return 1;
      }
    }
    return int(__len_) - int(__other.__len_);
  }

#  endif

  _CCCL_NODISCARD_FRIEND _LIBCUDACXX_HIDE_FROM_ABI constexpr bool
  operator==(__string_view const& __lhs, __string_view const& __rhs) noexcept
  {
    return __lhs.__len_ == __rhs.__len_ && __lhs.compare(__rhs) == 0;
  }

#  ifndef _LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR

  _CCCL_NODISCARD_FRIEND _LIBCUDACXX_HIDE_FROM_ABI constexpr auto
  operator<=>(__string_view const& __lhs, __string_view const& __rhs) noexcept
  {
    return __lhs.compare(__rhs) <=> 0;
  }

#  else // ^^^ !_LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR ^^^ / vvv _LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR

  _CCCL_NODISCARD_FRIEND _LIBCUDACXX_HIDE_FROM_ABI constexpr bool
  operator!=(__string_view const& __lhs, __string_view const& __rhs) noexcept
  {
    return !(__lhs == __rhs);
  }

  _CCCL_NODISCARD_FRIEND _LIBCUDACXX_HIDE_FROM_ABI constexpr bool
  operator<(__string_view const& __lhs, __string_view const& __rhs) noexcept
  {
    return __lhs.compare(__rhs) < 0;
  }

  _CCCL_NODISCARD_FRIEND _LIBCUDACXX_HIDE_FROM_ABI constexpr bool
  operator<=(__string_view const& __lhs, __string_view const& __rhs) noexcept
  {
    return __lhs.compare(__rhs) <= 0;
  }

  _CCCL_NODISCARD_FRIEND _LIBCUDACXX_HIDE_FROM_ABI constexpr bool
  operator>(__string_view const& __lhs, __string_view const& __rhs) noexcept
  {
    return __lhs.compare(__rhs) > 0;
  }

  _CCCL_NODISCARD_FRIEND _LIBCUDACXX_HIDE_FROM_ABI constexpr bool
  operator>=(__string_view const& __lhs, __string_view const& __rhs) noexcept
  {
    return __lhs.compare(__rhs) >= 0;
  }

#  endif // _LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR

private:
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr size_t __min_(size_t __x, size_t __y) noexcept
  {
    return __x < __y ? __x : __y;
  }

  _LIBCUDACXX_HIDE_FROM_ABI constexpr __string_view(char const* __str, size_t __len) noexcept
      : __str_(__str)
      , __len_(__len)
  {}

  char const* __str_;
  size_t __len_;
};

// We find a type T's name as follows:
// 1. Use __PRETTY_FUNCTION__ in a function template parameterized by
//    __pretty_name_begin<T>::__pretty_name_end.
// 2. Find the substrings "__pretty_name_begin<" and ">::__pretty_name_end".
//    Everything between them is the name of type T.
#  if defined(_CCCL_COMPILER_MSVC)
#    define _CCCL_PRETTY_FUNCTION __FUNCSIG__
#  else
#    define _CCCL_PRETTY_FUNCTION __PRETTY_FUNCTION__
#  endif

// Earlier versions of gcc (before gcc-9) do not treat __PRETTY_FUNCTION__ as a
// constexpr value after a reference to it has been returned from the function.
// Instead, arrange things so that the pretty name gets stored in a class static
// data member, where it can be referenced from other constexpr contexts.

#  if defined(_CCCL_COMPILER_GCC) && _CCCL_GCC_VERSION < 90000

template <size_t _Np>
struct __static_string
{
  char __str_[_Np];
};

template <size_t _Np, size_t... _Is>
constexpr auto __pretty_nameof_5(char const (&a)[_Np], index_sequence<_Is...>)
{
  return __static_string<_Np>{{a[_Is]...}};
}

template <class _Tp>
constexpr auto __pretty_nameof_4()
{
  return __pretty_nameof_5(_CCCL_PRETTY_FUNCTION, make_index_sequence<sizeof(_CCCL_PRETTY_FUNCTION)>());
}

template <class _Tp>
using __static_name_for = decltype(__pretty_nameof_4<_Tp>());

template <class _Tp>
struct __pretty_name_begin
{
  static constexpr __static_name_for<_Tp> value = __pretty_nameof_4<_Tp>();
  struct __pretty_name_end;
};

#    if !defined(_CCCL_NO_INLINE_VARIABLES)
template <class _Tp>
constexpr __static_name_for<_Tp> __pretty_name_begin<_Tp>::value;
#    endif // !_CCCL_NO_INLINE_VARIABLES

#  else // ^^^ _CCCL_GCC_VERSION < 90000 ^^^ / vvv _CCCL_GCC_VERSION >= 90000 vvv

template <class _Tp>
struct __pretty_name_begin
{
  struct __pretty_name_end;
};

#  endif // _CCCL_GCC_VERSION < 90000

// Get the type name from the pretty name by trimming the front and back.
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr __string_view __pretty_nameof_3(__string_view __sv) noexcept
{
  return __sv.substr(__sv.find("__pretty_name_begin<") + sizeof("__pretty_name_begin<") - 1,
                     __sv.find(">::__pretty_name_end"));
}

template <class _Tp>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr __string_view __pretty_nameof_2() noexcept
{
#  if defined(_CCCL_COMPILER_GCC) && _CCCL_GCC_VERSION < 90000
  return __pretty_nameof_3(__string_view(__pretty_name_begin<_Tp>::value.__str_));
#  else
  return __pretty_nameof_3(__string_view(_CCCL_PRETTY_FUNCTION));
#  endif
}

template <class _Tp>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr __string_view __pretty_nameof() noexcept
{
  return __pretty_nameof_2<typename __pretty_name_begin<_Tp>::__pretty_name_end>();
}

// A quick smoke test to ensure that the pretty name extraction is working.
static_assert(__pretty_nameof<int>() == __string_view("int"), "__pretty_nameof<int>() == __string_view(\"int\")");
static_assert(__pretty_nameof<float>() < __pretty_nameof<int>(), "__pretty_nameof<float>() < __pretty_nameof<int>()");

/// @brief A minimal implementation of `std::type_info` for platforms that do
/// not support RTTI.
struct __type_info
{
  _LIBCUDACXX_HIDE_FROM_ABI explicit constexpr __type_info(__string_view __name) noexcept
      : __name_(__name)
  {}

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr char const* name() const noexcept
  {
    return __name_.begin();
  }

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr bool before(const __type_info& __other) const noexcept
  {
    return __name_ < __other.__name_;
  }

  // Not yet implemented:
  // _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr size_t hash_code() const noexcept
  // {
  //   return ;
  // }

  _CCCL_NODISCARD_FRIEND _LIBCUDACXX_HIDE_FROM_ABI constexpr bool
  operator==(const __type_info& __lhs, const __type_info& __rhs) noexcept
  {
    return &__lhs == &__rhs || __lhs.__name_ == __rhs.__name_;
  }

#  if _CCCL_STD_VER <= 2017
  _CCCL_NODISCARD_FRIEND _LIBCUDACXX_HIDE_FROM_ABI constexpr bool
  operator!=(const __type_info& __lhs, const __type_info& __rhs) noexcept
  {
    return !(__lhs == __rhs);
  }
#  endif

private:
  __string_view __name_;
};

#  if !defined(_CCCL_NO_INLINE_VARIABLES)

template <class _Tp>
_CCCL_INLINE_VAR _CCCL_CONSTEXPR_GLOBAL auto __typeid = __type_info(__pretty_nameof<_Tp>());

#    define _CCCL_TYPEID(...) _CUDA_VSTD::__typeid<__VA_ARGS__>

#  else

template <class _Tp>
struct __typeid_value
{
  static constexpr __type_info value = __type_info(__pretty_nameof<_Tp>());
};

// Before the addition of inline variables, it was necessary to
// provide a definition for constexpr class static data members.
template <class _Tp>
constexpr __type_info __typeid_value<_Tp>::value;

#    ifndef _CCCL_NO_VARIABLE_TEMPLATES

template <class _Tp>
_CCCL_CONSTEXPR_GLOBAL __type_info& __typeid = __typeid_value<_Tp>::value;

#      define _CCCL_TYPEID(...) _CUDA_VSTD::__typeid<__VA_ARGS__>

#    else // !_CCCL_NO_VARIABLE_TEMPLATES

template <class _Tp>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr __type_info const& __typeid() noexcept
{
  return __typeid_value<_Tp>::value;
}

#      define _CCCL_TYPEID(...) _CUDA_VSTD::__typeid<__VA_ARGS__>()

#    endif // !_CCCL_NO_VARIABLE_TEMPLATES

#  endif // !_CCCL_NO_INLINE_VARIABLES

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _CCCL_NO_TYPEID

#endif // _LIBCUDACXX___UTILITY_TYPEID_H

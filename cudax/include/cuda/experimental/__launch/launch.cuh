//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__LAUNCH_LAUNCH
#define _CUDAX__LAUNCH_LAUNCH
#include <cuda_runtime.h>

#include <cuda/std/__exception/cuda_error.h>
#include <cuda/stream_ref>

#include <cuda/experimental/__launch/configuration.cuh>
#include <cuda/experimental/__launch/hierarchy_draft.cuh>
#include <cuda/experimental/__launch/kernel_launchers.cuh>
#include <cuda/experimental/__launch/launch_transform.cuh>
#include <cuda/experimental/__utility/ensure_current_device.cuh>

#if _CCCL_STD_VER >= 2017
namespace cuda::experimental
{

namespace detail
{
template <typename Config, typename Kernel, typename... Args>
_CCCL_NODISCARD cudaError_t
launch_impl(::cuda::stream_ref stream, Config conf, const Kernel& kernel_fn, const Args&... args)
{
  cudaLaunchConfig_t config               = {0};
  cudaError_t status                      = cudaSuccess;
  constexpr bool has_cluster_level        = has_level<cluster_level, decltype(conf.dims)>;
  constexpr unsigned int num_attrs_needed = detail::kernel_config_count_attr_space(conf) + has_cluster_level;
  cudaLaunchAttribute attrs[num_attrs_needed == 0 ? 1 : num_attrs_needed];
  config.attrs    = &attrs[0];
  config.numAttrs = 0;
  config.stream   = stream.get();

  status = detail::apply_kernel_config(conf, config, reinterpret_cast<void*>(kernel_fn));
  if (status != cudaSuccess)
  {
    return status;
  }

  config.blockDim = conf.dims.extents(thread, block);
  config.gridDim  = conf.dims.extents(block, grid);

  if constexpr (has_cluster_level)
  {
    auto cluster_dims                            = conf.dims.extents(block, cluster);
    config.attrs[config.numAttrs].id             = cudaLaunchAttributeClusterDimension;
    config.attrs[config.numAttrs].val.clusterDim = {
      static_cast<unsigned int>(cluster_dims.x),
      static_cast<unsigned int>(cluster_dims.y),
      static_cast<unsigned int>(cluster_dims.z)};
    config.numAttrs++;
  }

  // TODO lower to cudaLaunchKernelExC?
  return cudaLaunchKernelEx(&config, kernel_fn, args...);
}
} // namespace detail

// TODO launch docs should be updated to mention the need of using finalized_t on the input,
// but the question is should it be all usage or just when using meta dimensions
/**
 * @brief Launch a kernel functor with specified configuration and arguments
 *
 * Launches a kernel functor object on the specified stream and with specified configuration.
 * Kernel functor object is a type with __device__ operator().
 * Functor might or might not accept the configuration as its first argument.
 *
 *
 * @par Snippet
 * @code
 * #include <cstdio>
 * #include <cuda/experimental/launch.cuh>
 *
 * struct kernel {
 *     template <typename Configuration>
 *     __device__ void operator()(Configuration conf, unsigned int thread_to_print) {
 *         if (conf.dims.rank(cudax::thread, cudax::grid) == thread_to_print) {
 *             printf("Hello from the GPU\n");
 *         }
 *     }
 * };
 *
 * void launch_kernel(cuda::stream_ref stream) {
 *     auto dims    = cudax::make_hierarchy(cudax::block_dims<128>(), cudax::grid_dims(4));
 *     auto confing = cudax::make_config(dims, cudax::launch_cooperative());
 *
 *     cudax::launch(stream, config, kernel(), 42);
 * }
 * @endcode
 *
 * @param stream
 * cuda::stream_ref to launch the kernel into
 *
 * @param conf
 * configuration for this launch
 *
 * @param kernel
 * kernel functor to be launched
 *
 * @param args
 * arguments to be passed into the kernel functor
 */
template <typename... Args,
          typename... Config,
          typename Dimensions,
          typename Kernel,
          typename = ::cuda::std::enable_if_t<!::cuda::std::is_function_v<::cuda::std::remove_pointer_t<Kernel>>>>
void launch(
  ::cuda::stream_ref stream, const kernel_config<Dimensions, Config...>& conf, const Kernel& kernel, Args... args)
{
  __ensure_current_device __dev_setter(stream);
  cudaError_t status;
  auto finalized = finalize(stream, conf, kernel, args...);
  if constexpr (::cuda::std::is_invocable_v<Kernel, decltype(finalized), as_kernel_arg_t<Args>...>,
                || __nv_is_extended_device_lambda_closure_type(Kernel))
  {
    auto launcher = detail::kernel_launcher<decltype(finalized), Kernel, as_kernel_arg_t<Args>...>;
    status        = detail::launch_impl(
      stream,
      finalized,
      launcher,
      finalized,
      kernel,
      static_cast<as_kernel_arg_t<Args>>(detail::__launch_transform(stream, args))...);
  }
  else
  {
    static_assert(::cuda::std::is_invocable_v<Kernel, as_kernel_arg_t<Args>...>);
    auto launcher = detail::kernel_launcher_no_config<Kernel, as_kernel_arg_t<Args>...>;
    status        = detail::launch_impl(
      stream,
      finalized,
      launcher,
      kernel,
      static_cast<as_kernel_arg_t<Args>>(detail::__launch_transform(stream, args))...);
  }
  if (status != cudaSuccess)
  {
    ::cuda::__throw_cuda_error(status, "Failed to launch a kernel");
  }
}

/**
 * @brief Launch a kernel functor with specified thread hierarchy and arguments
 *
 * Launches a kernel functor object on the specified stream and with specified thread hierarchy.
 * Kernel functor object is a type with __device__ operator().
 * Functor might or might not accept the hierarchy as its first argument.
 *
 *
 * @par Snippet
 * @code
 * #include <cstdio>
 * #include <cuda/experimental/launch.cuh>
 *
 * struct kernel {
 *     template <typename Dimensions>
 *     __device__ void operator()(Dimensions dims, unsigned int thread_to_print) {
 *         if (dims.rank(cudax::thread, cudax::grid) == thread_to_print) {
 *             printf("Hello from the GPU\n");
 *         }
 *     }
 * };
 *
 * void launch_kernel(cuda::stream_ref stream) {
 *     auto dims = cudax::make_hierarchy(cudax::block_dims<128>(), cudax::grid_dims(4));
 *
 *     cudax::launch(stream, dims, kernel(), 42);
 * }
 * @endcode
 *
 * @param stream
 * cuda::stream_ref to launch the kernel into
 *
 * @param dims
 * thread hierarchy dimensions for this launch
 *
 * @param kernel
 * kernel functor to be launched
 *
 * @param args
 * arguments to be passed into the kernel functor
 */
template <typename... Args,
          typename... Levels,
          typename Kernel,
          typename = ::cuda::std::enable_if_t<!::cuda::std::is_function_v<std::remove_pointer_t<Kernel>>>>
void launch(::cuda::stream_ref stream, const hierarchy_dimensions<Levels...>& dims, const Kernel& kernel, Args... args)
{
  __ensure_current_device __dev_setter(stream);
  cudaError_t status;
  auto finalized = finalize(stream, dims, kernel, args...);
  if constexpr (::cuda::std::is_invocable_v<Kernel, decltype(finalized), as_kernel_arg_t<Args>...>
                || __nv_is_extended_device_lambda_closure_type(Kernel))
  {
    auto launcher = detail::kernel_launcher<decltype(finalized), Kernel, as_kernel_arg_t<Args>...>;
    status        = detail::launch_impl(
      stream,
      kernel_config(finalized),
      launcher,
      finalized,
      kernel,
      static_cast<as_kernel_arg_t<Args>>(detail::__launch_transform(stream, args))...);
  }
  else
  {
    static_assert(::cuda::std::is_invocable_v<Kernel, as_kernel_arg_t<Args>...>);
    auto launcher = detail::kernel_launcher_no_config<Kernel, as_kernel_arg_t<Args>...>;
    status        = detail::launch_impl(
      stream,
      kernel_config(finalized),
      launcher,
      kernel,
      static_cast<as_kernel_arg_t<Args>>(detail::__launch_transform(stream, args))...);
  }
  if (status != cudaSuccess)
  {
    ::cuda::__throw_cuda_error(status, "Failed to launch a kernel");
  }
}
/**
 * @brief Launch a kernel function with specified configuration and arguments
 *
 * Launches a kernel function on the specified stream and with specified configuration.
 * Kernel function is a function with __global__ annotation.
 * Function might or might not accept the configuration as its first argument.
 *
 *
 * @par Snippet
 * @code
 * #include <cstdio>
 * #include <cuda/experimental/launch.cuh>
 *
 * template <typename Congifuration>
 * __global__ void kernel(Configuration conf, unsigned int thread_to_print) {
 *     if (conf.dims.rank(cudax::thread, cudax::grid) == thread_to_print) {
 *         printf("Hello from the GPU\n");
 *     }
 * }
 *
 * void launch_kernel(cuda::stream_ref stream) {
 *     auto dims    = cudax::make_hierarchy(cudax::block_dims<128>(), cudax::grid_dims(4));
 *     auto confing = cudax::make_config(dims, cudax::launch_cooperative());
 *
 *     cudax::launch(stream, config, kernel<decltype(config)>, 42);
 * }
 * @endcode
 *
 * @param stream
 * cuda::stream_ref to launch the kernel into
 *
 * @param conf
 * configuration for this launch
 *
 * @param kernel
 * kernel function to be launched
 *
 * @param args
 * arguments to be passed into the kernel function
 */
template <typename... ExpArgs,
          typename... ActArgs,
          typename... Config,
          typename Dimensions,
          typename = ::cuda::std::enable_if_t<sizeof...(ExpArgs) == sizeof...(ActArgs)>>
void launch(::cuda::stream_ref stream,
            const kernel_config<Dimensions, Config...>& conf,
            void (*kernel)(finalized_t<kernel_config<Dimensions, Config...>>, ExpArgs...),
            ActArgs&&... args)
{
  __ensure_current_device __dev_setter(stream);
  auto finalized     = finalize(stream, conf, kernel);
  cudaError_t status = detail::launch_impl(
    stream, //
    finalized,
    kernel,
    finalized,
    static_cast<as_kernel_arg_t<ActArgs>>(detail::__launch_transform(stream, std::forward<ActArgs>(args)))...);

  if (status != cudaSuccess)
  {
    ::cuda::__throw_cuda_error(status, "Failed to launch a kernel");
  }
}

/**
 * @brief Launch a kernel function with specified thread hierarchy and arguments
 *
 * Launches a kernel function on the specified stream and with specified thread hierarchy.
 * Kernel function is a function with __global__ annotation.
 * Function might or might not accept the hierarchy as its first argument.
 *
 *
 * @par Snippet
 * @code
 * #include <cstdio>
 * #include <cuda/experimental/launch.cuh>
 *
 * template <typename Dimensions>
 * __global__ void kernel(Dimensions dims, unsigned int thread_to_print) {
 *     if (dims.rank(cudax::thread, cudax::grid) == thread_to_print) {
 *         printf("Hello from the GPU\n");
 *     }
 * }
 *
 * void launch_kernel(cuda::stream_ref stream) {
 *     auto dims = cudax::make_hierarchy(cudax::block_dims<128>(), cudax::grid_dims(4));
 *
 *     cudax::launch(stream, dims, kernel<decltype(dims)>, 42);
 * }
 * @endcode
 *
 * @param stream
 * cuda::stream_ref to launch the kernel into
 *
 * @param dims
 * thread hierarchy dimensions for this launch
 *
 * @param kernel
 * kernel function to be launched
 *
 * @param args
 * arguments to be passed into the kernel function
 */
template <typename... ExpArgs,
          typename... ActArgs,
          typename... Levels,
          typename = ::cuda::std::enable_if_t<sizeof...(ExpArgs) == sizeof...(ActArgs)>>
void launch(::cuda::stream_ref stream,
            const hierarchy_dimensions<Levels...>& dims,
            void (*kernel)(finalized_t<hierarchy_dimensions<Levels...>>, ExpArgs...),
            ActArgs&&... args)
{
  __ensure_current_device __dev_setter(stream);
  auto finalized     = finalize(stream, dims, kernel);
  cudaError_t status = detail::launch_impl(
    stream,
    kernel_config(finalized),
    kernel,
    finalized,
    static_cast<as_kernel_arg_t<ActArgs>>(detail::__launch_transform(stream, std::forward<ActArgs>(args)))...);

  if (status != cudaSuccess)
  {
    ::cuda::__throw_cuda_error(status, "Failed to launch a kernel");
  }
}

/**
 * @brief Launch a kernel function with specified configuration and arguments
 *
 * Launches a kernel function on the specified stream and with specified configuration.
 * Kernel function is a function with __global__ annotation.
 * Function might or might not accept the configuration as its first argument.
 *
 * @par Snippet
 * @code
 * #include <cstdio>
 * #include <cuda/experimental/launch.cuh>
 *
 * template <typename Congifuration>
 * __global__ void kernel(Configuration conf, unsigned int thread_to_print) {
 *     if (conf.dims.rank(cudax::thread, cudax::grid) == thread_to_print) {
 *         printf("Hello from the GPU\n");
 *     }
 * }
 *
 * void launch_kernel(cuda::stream_ref stream) {
 *     auto dims    = cudax::make_hierarchy(cudax::block_dims<128>(), cudax::grid_dims(4));
 *     auto confing = cudax::make_config(dims, cudax::launch_cooperative());
 *
 *     cudax::launch(stream, config, kernel<decltype(config)>, 42);
 * }
 * @endcode
 *
 * @param stream
 * cuda::stream_ref to launch the kernel into
 *
 * @param conf
 * configuration for this launch
 *
 * @param kernel
 * kernel function to be launched
 *
 * @param args
 * arguments to be passed into the kernel function
 */
template <typename... ExpArgs,
          typename... ActArgs,
          typename... Config,
          typename Dimensions,
          typename = ::cuda::std::enable_if_t<sizeof...(ExpArgs) == sizeof...(ActArgs)>>
void launch(::cuda::stream_ref stream,
            const kernel_config<Dimensions, Config...>& conf,
            void (*kernel)(ExpArgs...),
            ActArgs&&... args)
{
  __ensure_current_device __dev_setter(stream);
  auto finalized     = finalize(stream, conf, kernel);
  cudaError_t status = detail::launch_impl(
    stream, //
    finalized,
    kernel,
    static_cast<as_kernel_arg_t<ActArgs>>(detail::__launch_transform(stream, std::forward<ActArgs>(args)))...);

  if (status != cudaSuccess)
  {
    ::cuda::__throw_cuda_error(status, "Failed to launch a kernel");
  }
}

/**
 * @brief Launch a kernel function with specified thread hierarchy and arguments
 *
 * Launches a kernel function on the specified stream and with specified thread hierarchy.
 * Kernel function is a function with __global__ annotation.
 * Function might or might not accept the hierarchy as its first argument.
 *
 *
 * @par Snippet
 * @code
 * #include <cstdio>
 * #include <cuda/experimental/launch.cuh>
 *
 * template <typename Dimensions>
 * __global__ void kernel(Dimensions dims, unsigned int thread_to_print) {
 *     if (dims.rank(cudax::thread, cudax::grid) == thread_to_print) {
 *         printf("Hello from the GPU\n");
 *     }
 * }
 *
 * void launch_kernel(cuda::stream_ref stream) {
 *     auto dims = cudax::make_hierarchy(cudax::block_dims<128>(), cudax::grid_dims(4));
 *
 *     cudax::launch(stream, dims, kernel<decltype(dims)>, 42);
 * }
 * @endcode
 *
 * @param stream
 * cuda::stream_ref to launch the kernel into
 *
 * @param dims
 * thread hierarchy dimensions for this launch
 *
 * @param kernel
 * kernel function to be launched
 *
 * @param args
 * arguments to be passed into the kernel function
 */
template <typename... ExpArgs,
          typename... ActArgs,
          typename... Levels,
          typename = ::cuda::std::enable_if_t<sizeof...(ExpArgs) == sizeof...(ActArgs)>>
void launch(
  ::cuda::stream_ref stream, const hierarchy_dimensions<Levels...>& dims, void (*kernel)(ExpArgs...), ActArgs&&... args)
{
  __ensure_current_device __dev_setter(stream);
  auto finalized     = finalize(stream, dims, kernel);
  cudaError_t status = detail::launch_impl(
    stream,
    kernel_config(finalized),
    kernel,
    static_cast<as_kernel_arg_t<ActArgs>>(detail::__launch_transform(stream, std::forward<ActArgs>(args)))...);

  if (status != cudaSuccess)
  {
    ::cuda::__throw_cuda_error(status, "Failed to launch a kernel");
  }
}

} // namespace cuda::experimental
#endif // _CCCL_STD_VER >= 2017
#endif // _CUDAX__LAUNCH_LAUNCH

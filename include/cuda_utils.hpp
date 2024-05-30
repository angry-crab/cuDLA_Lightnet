#ifndef CUDA_UTILS_HPP
#define CUDA_UTILS_HPP

#include <cuda_runtime_api.h>

#include <memory>

namespace cuda_utils
{

struct CudaDeleter
{
    void operator()(void * p) const { CHECK_CUDA_ERROR(::cudaFree(p)); }
};

template <typename T>
using CudaUniquePtr = std::unique_ptr<T, CudaDeleter>;

struct StreamDeleter
{
  void operator()(cudaStream_t * stream)
  {
    if (stream) {
      cudaStreamDestroy(*stream);
      delete stream;
    }
  }
};

using StreamUniquePtr = std::unique_ptr<cudaStream_t, StreamDeleter>;

inline StreamUniquePtr makeCudaStream(const uint32_t flags = cudaStreamDefault)
{
  StreamUniquePtr stream(new cudaStream_t, StreamDeleter());
  if (cudaStreamCreateWithFlags(stream.get(), flags) != cudaSuccess) {
    stream.reset(nullptr);
  }
  return stream;
}
}  // namespace cuda_utils

#endif  // CUDA_UTILS__STREAM_UNIQUE_PTR_HPP_

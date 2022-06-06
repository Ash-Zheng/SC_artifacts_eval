#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace at;

void cache_sync_cuda(
    const Tensor this_unique,
    const Tensor last_unique,
    Tensor this_emb,
    const Tensor last_emb
    );

int32_t cache_sync_cuda_with_return(
    const Tensor this_unique,
    const Tensor last_unique,
    Tensor this_emb,
    const Tensor last_emb
    );


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cache_sync", &cache_sync_cuda, "cache_sync_cuda()");
  m.def("cache_sync_with_return", &cache_sync_cuda_with_return, "cache_sync_cuda()");
}
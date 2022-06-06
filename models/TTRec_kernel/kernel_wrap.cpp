#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace at;

Tensor tt_embeddings_forward_cuda(
    int32_t batch_count,
    int32_t num_tables,
    int32_t B,
    int32_t D,
    const std::vector<int>& tt_p_shapes,
    const std::vector<int>& tt_q_shapes,
    const std::vector<int>& tt_ranks,
    Tensor L,
    int32_t nnz,
    Tensor indices,
    Tensor rowidx,
    Tensor tableidx,
    const std::vector<Tensor>& tt_cores);

void tt_embeddings_backward_sgd_cuda(
    int32_t batch_count,
    int32_t D,
    float learning_rate,
    const std::vector<int32_t>& tt_p_shapes,
    const std::vector<int32_t>& tt_q_shapes,
    const std::vector<int32_t>& tt_ranks,
    Tensor L,
    int32_t nnz,
    Tensor indices,
    Tensor offsets,
    Tensor tableidx,
    Tensor d_output,
    std::vector<Tensor>& tt_cores);

std::vector<Tensor> tt_embeddings_backward_dense_cuda(
    int32_t batch_count,
    int32_t D,
    const std::vector<int32_t>& tt_p_shapes,
    const std::vector<int32_t>& tt_q_shapes,
    const std::vector<int32_t>& tt_ranks,
    Tensor L,
    int32_t nnz,
    Tensor indices,
    Tensor offsets,
    Tensor tableidx,
    Tensor d_output,
    std::vector<Tensor>& tt_cores);

std::tuple<Tensor, Tensor, Tensor, int32_t, c10::optional<Tensor>>
preprocess_indices_sync_cuda(
    Tensor colidx,
    Tensor offsets,
    int32_t num_tables,
    bool warmup,
    Tensor hashtbl,
    Tensor cache_state);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("Facebook_TT_forward", &tt_embeddings_forward_cuda, "Facebook_TT_forward()");
  m.def("Facebook_TT_sgd_backward", &tt_embeddings_backward_sgd_cuda, "Facebook_TT_sgd_backward()");
  m.def("Facebook_TT_sgd_backward_dense", &tt_embeddings_backward_dense_cuda, "Facebook_TT_sgd_backward_dense()");
  m.def(
      "preprocess_indices_sync",
      &preprocess_indices_sync_cuda,
      "preprocess_colidx_sync()");
}
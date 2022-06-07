/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
// #include <ATen/CUDAGeneratorImpl.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/TensorAccessor.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <THC/THCAtomics.cuh>
#include <mutex>
#include "cub-1.8.0/cub/device/device_partition.cuh"
#include "cub-1.8.0/cub/device/device_radix_sort.cuh"
#include "hashtbl_cuda_utils.cuh"
#include "tt_cuda_utils.cuh"

using namespace at;

namespace {

constexpr int32_t MAX_PROBES = 3;

enum {
  OPTIM_SGD = 0,
  OPTIM_ADAGRAD = 1,
  OPTIM_DENSE = 2,
};

}

__global__ void compute_rowidx_kernel(
    int32_t B,
    int32_t num_tables,
    const int64_t* __restrict__ offsets,
    int64_t* __restrict__ rowidx,
    int64_t* __restrict__ tableidx) {
  int32_t b = blockIdx.x * blockDim.y + threadIdx.y;
  if (b < B * num_tables) {
    int64_t colidx_start = offsets[b];
    int64_t colidx_end = offsets[b + 1];
    int32_t L = colidx_end - colidx_start;
    for (int32_t l = threadIdx.x; l < L; l += blockDim.x) {
      rowidx[l + colidx_start] = b % B;
      tableidx[l + colidx_start] = b / B;
    }
  }
}

__global__ void cache_lookup_kernel(
    int32_t N,
    const int64_t* __restrict__ colidx,
    int32_t hashtbl_size,
    const int64_t* __restrict__ hashtbl,
    const int32_t* __restrict__ cache_state,
    bool* __restrict__ is_tt,
    int32_t* __restrict__ cache_location) {
  int32_t n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    int32_t hashtbl_idx =
        hashtbl_find(colidx[n], hashtbl_size, MAX_PROBES, hashtbl);
    if (hashtbl_idx != -1 && cache_state[hashtbl_idx] != -1) {
      is_tt[n] = false;
      cache_location[n] = cache_state[hashtbl_idx];
    } else {
      is_tt[n] = true;
    }
  }
}

std::tuple<Tensor, Tensor, Tensor, int32_t, c10::optional<Tensor>>
preprocess_indices_sync_cuda(
    Tensor colidx,
    Tensor offsets,
    int32_t num_tables,
    bool warmup,
    Tensor hashtbl,
    Tensor cache_state) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(colidx.get_device());
  auto rowidx = empty_like(colidx);
  auto tableidx = empty_like(colidx);
  if (rowidx.numel() == 0) {
    return {colidx, rowidx, tableidx, rowidx.numel(), c10::nullopt};
  }
  int32_t B = (offsets.numel() - 1) / num_tables;
  int32_t N = colidx.numel();
  int32_t num_rows = offsets.numel() - 1;
  int32_t tx = 8;
  int32_t ty = 32;
  compute_rowidx_kernel<<<
      div_round_up(num_rows, ty),
      dim3(tx, ty),
      0,
      c10::cuda::getCurrentCUDAStream()>>>(
      B,
      num_tables,
      offsets.data_ptr<int64_t>(),
      rowidx.data_ptr<int64_t>(),
      tableidx.data_ptr<int64_t>());
  if (warmup || num_tables != 1) {
    // if in warmup phase or num_tables != 1, we do not lookup cache
    return {colidx, rowidx, tableidx, rowidx.numel(), c10::nullopt};
  } else {
    auto partitioned_colidx = empty_like(colidx);
    auto partitioned_rowidx = empty_like(rowidx);
    auto num_tt_indices = zeros({1}, rowidx.options().dtype(kInt));
    auto cache_locations = empty_like(rowidx, rowidx.options().dtype(kInt));
    auto partitioned_cache_locations =
        empty_like(rowidx, rowidx.options().dtype(kInt));
    {
      auto is_tt = empty_like(rowidx, rowidx.options().dtype(kBool));
      int32_t threads = 256;
      int32_t num_blocks = div_round_up(N, threads);
      cache_lookup_kernel<<<
          num_blocks,
          threads,
          0,
          c10::cuda::getCurrentCUDAStream()>>>(
          N,
          colidx.data_ptr<int64_t>(),
          hashtbl.numel(),
          hashtbl.data_ptr<int64_t>(),
          cache_state.data_ptr<int32_t>(),
          is_tt.data_ptr<bool>(),
          cache_locations.data_ptr<int32_t>());
      size_t temp_storage_bytes = 0;
      AT_CUDA_CHECK(cub::DevicePartition::Flagged(
          nullptr,
          temp_storage_bytes,
          rowidx.data_ptr<int64_t>(),
          is_tt.data_ptr<bool>(),
          partitioned_rowidx.data_ptr<int64_t>(),
          num_tt_indices.data_ptr<int32_t>(),
          rowidx.numel(),
          at::cuda::getCurrentCUDAStream(),
          false));
      auto temp_storage = at::empty(
          {static_cast<int64_t>(temp_storage_bytes)},
          hashtbl.options().dtype(kByte));
      AT_CUDA_CHECK(cub::DevicePartition::Flagged(
          temp_storage.data_ptr(),
          temp_storage_bytes,
          rowidx.data_ptr<int64_t>(),
          is_tt.data_ptr<bool>(),
          partitioned_rowidx.data_ptr<int64_t>(),
          num_tt_indices.data_ptr<int32_t>(),
          rowidx.numel(),
          at::cuda::getCurrentCUDAStream(),
          false));
      AT_CUDA_CHECK(cub::DevicePartition::Flagged(
          temp_storage.data_ptr(),
          temp_storage_bytes,
          colidx.data_ptr<int64_t>(),
          is_tt.data_ptr<bool>(),
          partitioned_colidx.data_ptr<int64_t>(),
          num_tt_indices.data_ptr<int32_t>(),
          colidx.numel(),
          at::cuda::getCurrentCUDAStream(),
          false));
      AT_CUDA_CHECK(cub::DevicePartition::Flagged(
          temp_storage.data_ptr(),
          temp_storage_bytes,
          cache_locations.data_ptr<int32_t>(),
          is_tt.data_ptr<bool>(),
          partitioned_cache_locations.data_ptr<int32_t>(),
          num_tt_indices.data_ptr<int32_t>(),
          cache_locations.numel(),
          at::cuda::getCurrentCUDAStream(),
          false));
    }
    int32_t N_tt_indices;
    cudaMemcpyAsync(
        &N_tt_indices,
        num_tt_indices.data_ptr<int32_t>(),
        sizeof(int32_t),
        cudaMemcpyDeviceToHost,
        at::cuda::getCurrentCUDAStream());
    cudaStreamSynchronize(at::cuda::getCurrentCUDAStream());
    return {
        partitioned_colidx,
        partitioned_rowidx,
        tableidx,
        N_tt_indices,
        partitioned_cache_locations};
  }
}

inline void cuda_gemm_batched_fp32_fp32(
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    float* alpha,
    void** a_array,
    int lda,
    void** b_array,
    int ldb,
    float* beta,
    void** c_array,
    int ldc,
    int batch_count) {
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasSetStream(handle, c10::cuda::getCurrentCUDAStream());
  cublasGemmBatchedEx(
      handle,
      transa,
      transb,
      m,
      n,
      k,
      alpha,
      a_array,
      CUDA_R_32F,
      lda,
      b_array,
      CUDA_R_32F,
      ldb,
      beta,
      c_array,
      CUDA_R_32F,
      ldc,
      batch_count,
      CUDA_R_32F,
      CUBLAS_GEMM_DEFAULT);
}

__global__ void init_batch_gemm_forward_3T_kernel(
    int N,
    const int64_t* __restrict__ L,
    const int64_t* __restrict__ colidx,
    const int64_t* __restrict__ tableidx,
    PackedTensorAccessor32<float, 3, RestrictPtrTraits> tt_cores_0,
    PackedTensorAccessor32<float, 3, RestrictPtrTraits> tt_cores_1,
    PackedTensorAccessor32<float, 3, RestrictPtrTraits> tt_cores_2,
    PackedTensorAccessor32<float, 2, RestrictPtrTraits> tr_0,
    PackedTensorAccessor32<float, 2, RestrictPtrTraits> tr_1,
    float** __restrict__ a_ptr,
    float** __restrict__ b_ptr,
    float** __restrict__ c_ptr) {
  int32_t n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    auto tidx = __ldg(&tableidx[n]);
    auto cidx = __ldg(&colidx[n]);
    auto tt_idx_0 = cidx / L[0];
    cidx = cidx % L[0];
    auto tt_idx_1 = cidx / L[1];
    cidx = cidx % L[1];
    auto tt_idx_2 = cidx / L[2];
    float* tr_0_ptr = (float*)&(tr_0[n][0]);
    a_ptr[0 * N + n] = (float*)&(tt_cores_1[tidx][tt_idx_1][0]);
    b_ptr[0 * N + n] = (float*)&(tt_cores_0[tidx][tt_idx_0][0]);
    c_ptr[0 * N + n] = tr_0_ptr;
    a_ptr[1 * N + n] = (float*)&(tt_cores_2[tidx][tt_idx_2][0]);
    b_ptr[1 * N + n] = tr_0_ptr;
    c_ptr[1 * N + n] = (float*)&(tr_1[n][0]);
  }
}

void init_batch_gemm_forward_cuda(
    int32_t T,
    int32_t N,
    const int64_t* __restrict__ L,
    const int64_t* __restrict__ colidx,
    const int64_t* __restrict__ tableidx,
    const std::vector<Tensor>& tt_cores,
    const std::vector<Tensor>& tr,
    float** __restrict__ a_ptr,
    float** __restrict__ b_ptr,
    float** __restrict__ c_ptr) {
  int32_t threads = (N > 256 ? 256 : 32);
  int32_t num_blocks = (N + threads - 1) / threads;
  if (T == 3) {
    init_batch_gemm_forward_3T_kernel<<<
        num_blocks,
        threads,
        0,
        c10::cuda::getCurrentCUDAStream()>>>
        (
        N,
        L,
        colidx,
        tableidx,
        tt_cores[0].packed_accessor32<float, 3, RestrictPtrTraits>(),
        tt_cores[1].packed_accessor32<float, 3, RestrictPtrTraits>(),
        tt_cores[2].packed_accessor32<float, 3, RestrictPtrTraits>(),
        tr[0].packed_accessor32<float, 2, RestrictPtrTraits>(),
        tr[1].packed_accessor32<float, 2, RestrictPtrTraits>(),
        a_ptr,
        b_ptr,
        c_ptr);
  } 
}

__global__ void reduce_output_kernel(
    int32_t N,
    int32_t B,
    int32_t D,
    const int64_t* __restrict__ rowidx,
    const int64_t* __restrict__ tableidx,
    const float* __restrict__ tr_last,
    float* __restrict__ output) {
  int32_t indice_id = blockIdx.x * blockDim.y + threadIdx.y;
  if (indice_id >= N) {
    // don't have *warp* divergence since we launch full warps in blockDim.x,
    // so we can just exit this warp entirely.
    return;
  }
  // check if this warp is responsible for this whole segment.
  bool segment_start =
      (indice_id == 0 || rowidx[indice_id - 1] != rowidx[indice_id] ||
       tableidx[indice_id - 1] != tableidx[indice_id]);
  if (!segment_start) {
    // don't have *warp* divergence since we launch full warps in blockDim.x,
    // so we can just exit this warp entirely.
    return;
  }
  int64_t row_index = rowidx[indice_id];
  int64_t table_index = tableidx[indice_id];
  // now, find the end of the segment (and thus the segment length `SL`).
  int32_t SL = 1;
  while (indice_id + SL < N && rowidx[indice_id + SL] == row_index &&
         tableidx[indice_id + SL] == table_index) {
    SL += 1;
  }
  for (int32_t d = threadIdx.x; d * 4 < D; d += blockDim.x) {
    Vec4T<float> sum(&output[table_index * B * D + row_index * D + d * 4]);
    for (int32_t sl = 0; sl < SL; ++sl) {
      Vec4T<float> tr(&tr_last[(indice_id + sl) * D + d * 4]);
      sum.acc.x += tr.acc.x;
      sum.acc.y += tr.acc.y;
      sum.acc.z += tr.acc.z;
      sum.acc.w += tr.acc.w;
    }
    sum.store(&output[table_index * B * D + row_index * D + d * 4]);
  }
}


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
    Tensor colidx,
    Tensor rowidx,
    Tensor tableidx,
    const std::vector<Tensor>& tt_cores) {

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(rowidx.get_device());
  int32_t T = tt_p_shapes.size();
  auto output =
      at::zeros({num_tables, B, D}, tt_cores[0].options().dtype(at::kFloat));

  if (nnz == 0) {
    return output;
  }

  TORCH_CHECK(batch_count > 0);
  TORCH_CHECK(D > 0);
  TORCH_CHECK(D % 4 == 0);
  TORCH_CHECK(T > 0);

  // batch gemm parameters
  std::vector<int32_t> m(T - 1);
  std::vector<int32_t> n(T - 1);
  std::vector<int32_t> k(T - 1);
  float alpha = 1.0;
  float beta = 0.0;
  int32_t m_ = tt_q_shapes[0];
  for (int32_t t = 0; t < T - 1; ++t) {
    m[t] = m_;
    k[t] = tt_ranks[t + 1];
    n[t] = tt_q_shapes[t + 1] * tt_ranks[t + 2];
    m_ = m_ * tt_q_shapes[t + 1];
  }

  // allocate the immediate buffers
  std::vector<Tensor> tr;
  int32_t tr_size = tt_q_shapes[0] * tt_ranks[1];
  for (int32_t t = 0; t < T - 1; ++t) {
    tr_size = tr_size * tt_q_shapes[t + 1] * tt_ranks[t + 2] / tt_ranks[t + 1];
    tr.push_back(at::empty(
        {batch_count, tr_size}, tt_cores[0].options().dtype(at::kFloat)));
  }
  auto a_ptr_tensor = at::empty(
      {(T - 1) * batch_count}, tt_cores[0].options().dtype(at::kLong));
  auto b_ptr_tensor = at::empty(
      {(T - 1) * batch_count}, tt_cores[0].options().dtype(at::kLong));
  auto c_ptr_tensor = at::empty(
      {(T - 1) * batch_count}, tt_cores[0].options().dtype(at::kLong));
  float** a_ptr = (float**)a_ptr_tensor.data_ptr<int64_t>();
  float** b_ptr = (float**)b_ptr_tensor.data_ptr<int64_t>();
  float** c_ptr = (float**)c_ptr_tensor.data_ptr<int64_t>();
  for (int32_t start_idx = 0; start_idx < nnz; start_idx += batch_count) {
    int32_t end_idx =
        start_idx + batch_count < nnz ? start_idx + batch_count : nnz;
    int32_t N = end_idx - start_idx;
    init_batch_gemm_forward_cuda(
        T,
        N,
        L.data_ptr<int64_t>(),
        &(colidx.data_ptr<int64_t>()[start_idx]),
        &(tableidx.data_ptr<int64_t>()[start_idx]),
        tt_cores,
        tr,
        a_ptr,
        b_ptr,
        c_ptr);
    // batched GEMM
    // for (int32_t t = 0; t < T - 1; ++t) {
    for (int32_t t = 0; t < T - 1; ++t) {
      cuda_gemm_batched_fp32_fp32(
          CUBLAS_OP_N,
          CUBLAS_OP_N,
          n[t],
          m[t],
          k[t],
          &alpha,
          (void**)&(a_ptr[t * N]),
          n[t],
          (void**)&(b_ptr[t * N]),
          k[t],
          &beta,
          (void**)&(c_ptr[t * N]),
          n[t],
          N);
    }

    int32_t tx = kWarpSize;
    int32_t ty = 1024 / tx;
    dim3 threads(tx, ty);
    int32_t num_blocks = (N + ty - 1) / ty;

    reduce_output_kernel<<<
        num_blocks,
        threads,
        0,
        c10::cuda::getCurrentCUDAStream()>>>(
        N,
        B,
        D,
        &(rowidx.data_ptr<int64_t>()[start_idx]),
        &(tableidx.data_ptr<int64_t>()[start_idx]),
        tr[T - 2].data_ptr<float>(),
        output.data_ptr<float>());
  } // for (int start_idx = 0; start_idx < nnz; start_idx += batch_count)

  return output;
}

__global__ void init_batch_gemm_backward_3T_kernel(
    int32_t N,
    const int64_t* __restrict__ colidx,
    const int64_t* __restrict__ rowidx,
    const int64_t* __restrict__ tableidx,
    const int64_t* __restrict__ L,
    PackedTensorAccessor32<float, 3, RestrictPtrTraits> tt_cores_0,
    PackedTensorAccessor32<float, 3, RestrictPtrTraits> tt_cores_1,
    PackedTensorAccessor32<float, 3, RestrictPtrTraits> tt_cores_2,
    PackedTensorAccessor32<float, 2, RestrictPtrTraits> tr_tt_cores_0,
    PackedTensorAccessor32<float, 2, RestrictPtrTraits> tr_tt_cores_1,
    PackedTensorAccessor32<float, 2, RestrictPtrTraits> tr_tt_cores_2,
    PackedTensorAccessor32<float, 2, RestrictPtrTraits> tr_0,
    PackedTensorAccessor32<float, 3, RestrictPtrTraits> d_output,
    int32_t* __restrict__ tt_idx,
    float** __restrict__ a_ptr,
    float** __restrict__ b_ptr,
    float** __restrict__ c_ptr,
    float** __restrict__ a0_ptr,
    float** __restrict__ b0_ptr,
    float** __restrict__ c0_ptr,
    float** __restrict__ a1_ptr,
    float** __restrict__ b1_ptr,
    float** __restrict__ c1_ptr) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    auto cidx = __ldg(&colidx[n]);
    auto ridx = __ldg(&rowidx[n]);
    auto tidx = __ldg(&tableidx[n]);
    int32_t tt_idx_0 = cidx / L[0];
    cidx = cidx % L[0];
    int32_t tt_idx_1 = cidx / L[1];
    cidx = cidx % L[1];
    int32_t tt_idx_2 = cidx / L[2];
    tt_idx[0 * N + n] = tt_idx_0;
    tt_idx[1 * N + n] = tt_idx_1;
    tt_idx[2 * N + n] = tt_idx_2;
    float* tr_0_ptr = (float*)&(tr_0[n][0]);
    float* d_output_ptr = (float*)&(d_output[tidx][ridx][0]);
    float* tt_cores_0_ptr = (float*)&(tt_cores_0[tidx][tt_idx_0][0]);
    float* tt_cores_1_ptr = (float*)&(tt_cores_1[tidx][tt_idx_1][0]);
    a_ptr[0 * N + n] = tt_cores_1_ptr;
    b_ptr[0 * N + n] = tt_cores_0_ptr;
    c_ptr[0 * N + n] = tr_0_ptr;
    a0_ptr[0 * N + n] = tt_cores_0_ptr;
    b0_ptr[0 * N + n] = tr_0_ptr;
    c0_ptr[0 * N + n] = (float*)&(tr_tt_cores_1[n][0]);
    a1_ptr[0 * N + n] = tr_0_ptr;
    b1_ptr[0 * N + n] = tt_cores_1_ptr;
    c1_ptr[0 * N + n] = (float*)&(tr_tt_cores_0[n][0]);
    a0_ptr[1 * N + n] = tr_0_ptr;
    b0_ptr[1 * N + n] = d_output_ptr;
    c0_ptr[1 * N + n] = (float*)&(tr_tt_cores_2[n][0]);
    a1_ptr[1 * N + n] = d_output_ptr;
    b1_ptr[1 * N + n] = (float*)&(tt_cores_2[tidx][tt_idx_2][0]);
    c1_ptr[1 * N + n] = tr_0_ptr;
  }
}

void init_batch_gemm_backward_cuda(
    int32_t T,
    int32_t N,
    const int64_t* __restrict__ colidx,
    const int64_t* __restrict__ rowidx,
    const int64_t* __restrict__ tableidx,
    const int64_t* __restrict__ L,
    const std::vector<Tensor>& tt_cores,
    const std::vector<Tensor>& tr_tt_cores,
    const std::vector<Tensor>& tr,
    Tensor d_output,
    int32_t* __restrict__ tt_idx,
    float** __restrict__ a_ptr,
    float** __restrict__ b_ptr,
    float** __restrict__ c_ptr,
    float** __restrict__ a0_ptr,
    float** __restrict__ b0_ptr,
    float** __restrict__ c0_ptr,
    float** __restrict__ a1_ptr,
    float** __restrict__ b1_ptr,
    float** __restrict__ c1_ptr) {
  int32_t threads = (N > 256 ? 256 : 32);
  int32_t num_blocks = (N + threads - 1) / threads;
  if (T == 3) {
    init_batch_gemm_backward_3T_kernel<<<
        num_blocks,
        threads,
        0,
        c10::cuda::getCurrentCUDAStream()>>>(
        N,
        colidx,
        rowidx,
        tableidx,
        L,
        tt_cores[0].packed_accessor32<float, 3, RestrictPtrTraits>(),
        tt_cores[1].packed_accessor32<float, 3, RestrictPtrTraits>(),
        tt_cores[2].packed_accessor32<float, 3, RestrictPtrTraits>(),
        tr_tt_cores[0].packed_accessor32<float, 2, RestrictPtrTraits>(),
        tr_tt_cores[1].packed_accessor32<float, 2, RestrictPtrTraits>(),
        tr_tt_cores[2].packed_accessor32<float, 2, RestrictPtrTraits>(),
        tr[0].packed_accessor32<float, 2, RestrictPtrTraits>(),
        d_output.packed_accessor32<float, 3, RestrictPtrTraits>(),
        tt_idx,
        a_ptr,
        b_ptr,
        c_ptr,
        a0_ptr,
        b0_ptr,
        c0_ptr,
        a1_ptr,
        b1_ptr,
        c1_ptr);
  } 
}

__global__ void update_tt_cores_sgd_kernel(
    int32_t B,
    int32_t D,
    int32_t num_tables,
    float learning_rate,
    PackedTensorAccessor32<float, 3, RestrictPtrTraits> d_tt_cores,
    PackedTensorAccessor32<float, 3, RestrictPtrTraits> tt_cores) {
  int32_t b = blockIdx.x * blockDim.y + threadIdx.y;
  if (b >= B) {
    return;
  }
  for (int32_t i = 0; i < num_tables; i++) {
    for (int32_t d = threadIdx.x; d < D; d += blockDim.x) {
      tt_cores[i][b][d] -= learning_rate * d_tt_cores[i][b][d];
    }
  }
}

__global__ void update_d_tt_cores_kernel(
    int32_t N,
    int32_t D,
    const int32_t* __restrict__ tt_idx,
    const int64_t* __restrict__ tableidx,
    PackedTensorAccessor32<float, 2, RestrictPtrTraits> tr_tt_cores,
    PackedTensorAccessor32<float, 3, RestrictPtrTraits> d_tt_cores) {
  int32_t n = blockIdx.x * blockDim.y + threadIdx.y;
  if (n < N) {
    auto idx = __ldg(&tt_idx[n]);
    auto tidx = __ldg(&tableidx[n]);
    for (int32_t d = threadIdx.x; d < D; d += blockDim.x) {
      atomicAdd(&(d_tt_cores[tidx][idx][d]), tr_tt_cores[n][d]);
    }
  }
}

std::vector<Tensor> tt_embeddings_backward_cuda(
    int32_t optim,
    int32_t batch_count,
    int32_t D,
    float learning_rate,
    float eps,
    const std::vector<int32_t>& tt_p_shapes,
    const std::vector<int32_t>& tt_q_shapes,
    const std::vector<int32_t>& tt_ranks,
    Tensor L,
    int32_t nnz,
    Tensor colidx,
    Tensor rowidx,
    Tensor tableidx,
    Tensor d_output,
    c10::optional<std::vector<Tensor>> optimizer_state,
    std::vector<Tensor>& tt_cores) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(d_output.get_device());
  int32_t T = tt_p_shapes.size();  //3
  int32_t num_tables = tt_cores[0].size(0); //1

  std::vector<Tensor> d_tt_cores;
  std::vector<Tensor> tr_tt_cores;
  for (int32_t t = 0; t < T; ++t) {
    d_tt_cores.push_back(at::zeros_like(tt_cores[t]));
    tr_tt_cores.push_back(
        at::empty({batch_count, tt_cores[t].size(2)}, tt_cores[t].options()));
    // printf("tt_core size(2):%d,size(1):%d\n",tt_cores[t].size(2), tt_cores[t].size(1));
  }
  if (nnz == 0) {
    return d_tt_cores;
  }

  // batch gemm parameters
  std::vector<int32_t> m(T - 1);
  std::vector<int32_t> n(T - 1);
  std::vector<int32_t> k(T - 1);
  float alpha = 1.0;
  float beta = 0.0;
  int32_t m_ = tt_q_shapes[0]; 
  for (int32_t t = 0; t < T - 1; ++t) {
    m[t] = m_; //m[0]=j1 m[1]=j1*j2
    k[t] = tt_ranks[t + 1]; //k[0]=r1 k[1]=r2
    n[t] = tt_q_shapes[t + 1] * tt_ranks[t + 2]; //n[0]=j2*r2 n[1]=j3
    m_ = m_ * tt_q_shapes[t + 1];
  }

  
  // allocate the immediate buffers
  std::vector<Tensor> tr;

  int64_t tr_size = tt_q_shapes[0] * tt_ranks[1];
  for (int32_t t = 0; t < T - 2; ++t) {
    tr_size = tr_size * tt_q_shapes[t + 1] * tt_ranks[t + 2] / tt_ranks[t + 1];
    tr.push_back(at::empty({batch_count, tr_size}, tt_cores[0].options()));
  }

  auto tt_idx =
      at::empty({T * batch_count}, tt_cores[0].options().dtype(at::kInt));
  auto a_ptr_tensor = at::empty(
      {(T - 2) * batch_count}, tt_cores[0].options().dtype(at::kLong));
  auto b_ptr_tensor = at::empty(
      {(T - 2) * batch_count}, tt_cores[0].options().dtype(at::kLong));
  auto c_ptr_tensor = at::empty(
      {(T - 2) * batch_count}, tt_cores[0].options().dtype(at::kLong));
  float** a_ptr = (float**)a_ptr_tensor.data_ptr<int64_t>();
  float** b_ptr = (float**)b_ptr_tensor.data_ptr<int64_t>();
  float** c_ptr = (float**)c_ptr_tensor.data_ptr<int64_t>();
  auto a0_ptr_tensor = at::empty(
      {(T - 1) * batch_count}, tt_cores[0].options().dtype(at::kLong));
  auto b0_ptr_tensor = at::empty(
      {(T - 1) * batch_count}, tt_cores[0].options().dtype(at::kLong));
  auto c0_ptr_tensor = at::empty(
      {(T - 1) * batch_count}, tt_cores[0].options().dtype(at::kLong));
  float** a0_ptr = (float**)a0_ptr_tensor.data_ptr<int64_t>();
  float** b0_ptr = (float**)b0_ptr_tensor.data_ptr<int64_t>();
  float** c0_ptr = (float**)c0_ptr_tensor.data_ptr<int64_t>();
  auto a1_ptr_tensor = at::empty(
      {(T - 1) * batch_count}, tt_cores[0].options().dtype(at::kLong));
  auto b1_ptr_tensor = at::empty(
      {(T - 1) * batch_count}, tt_cores[0].options().dtype(at::kLong));
  auto c1_ptr_tensor = at::empty(
      {(T - 1) * batch_count}, tt_cores[0].options().dtype(at::kLong));
  float** a1_ptr = (float**)a1_ptr_tensor.data_ptr<int64_t>();
  float** b1_ptr = (float**)b1_ptr_tensor.data_ptr<int64_t>();
  float** c1_ptr = (float**)c1_ptr_tensor.data_ptr<int64_t>();
  for (int32_t start_idx = 0; start_idx < nnz; start_idx += batch_count) {
    int32_t end_idx =
        start_idx + batch_count < nnz ? start_idx + batch_count : nnz;
    int32_t N = end_idx - start_idx;
    init_batch_gemm_backward_cuda(
        T,
        N,
        &(colidx.data_ptr<int64_t>()[start_idx]),
        &(rowidx.data_ptr<int64_t>()[start_idx]),
        &(tableidx.data_ptr<int64_t>()[start_idx]),
        L.data_ptr<int64_t>(),
        tt_cores,
        tr_tt_cores,
        tr,
        d_output,
        tt_idx.data_ptr<int32_t>(),
        a_ptr,
        b_ptr,
        c_ptr,
        a0_ptr,
        b0_ptr,
        c0_ptr,
        a1_ptr,
        b1_ptr,
        c1_ptr);
    // recompute forward
    for (int32_t t = 0; t < T - 2; ++t) {
      cuda_gemm_batched_fp32_fp32(
          CUBLAS_OP_N,
          CUBLAS_OP_N,
          n[t],
          m[t],
          k[t],
          &alpha,
          (void**)&(a_ptr[t * N]),
          n[t],
          (void**)&(b_ptr[t * N]),
          k[t],
          &beta,
          (void**)&(c_ptr[t * N]),
          n[t],
          N);
    } // for (int32_t t = 0; t < T - 2; ++t)
    // backward propagation
    for (int32_t t = T - 2; t >= 0; --t) {
      cuda_gemm_batched_fp32_fp32(
          CUBLAS_OP_N,
          CUBLAS_OP_T,
          n[t],
          k[t],
          m[t],
          &alpha,
          (void**)&(b0_ptr[t * N]),
          n[t],
          (void**)&(a0_ptr[t * N]),
          k[t],
          &beta,
          (void**)&(c0_ptr[t * N]),
          n[t],
          N);
      int32_t D_0 = tt_cores[t + 1].size(2);
      int32_t tx_0 = std::min(1024, D_0);
      int32_t ty_0 = 1024 / tx_0;
      update_d_tt_cores_kernel<<<
          div_round_up(N, ty_0),
          dim3(tx_0, ty_0),
          0,
          c10::cuda::getCurrentCUDAStream()>>>(
          N,
          D_0,
          &(tt_idx.data_ptr<int32_t>()[(t + 1) * N]),
          &(tableidx.data_ptr<int64_t>()[start_idx]),
          tr_tt_cores[t + 1].packed_accessor32<float, 2, RestrictPtrTraits>(),
          d_tt_cores[t + 1].packed_accessor32<float, 3, RestrictPtrTraits>());
      cuda_gemm_batched_fp32_fp32(
          CUBLAS_OP_T,
          CUBLAS_OP_N,
          k[t],
          m[t],
          n[t],
          &alpha,
          (void**)&(b1_ptr[t * N]),
          n[t],
          (void**)&(a1_ptr[t * N]),
          n[t],
          &beta,
          (void**)&(c1_ptr[t * N]),
          k[t],
          N);
      if (t == 0) {
        int32_t D_1 = tt_cores[0].size(2);
        int32_t tx_1 = std::min(1024, D_1);
        int32_t ty_1 = 1024 / tx_1;
        update_d_tt_cores_kernel<<<
            div_round_up(N, ty_1),
            dim3(tx_1, ty_1),
            0,
            c10::cuda::getCurrentCUDAStream()>>>(
            N,
            D_1,
            &(tt_idx.data_ptr<int32_t>()[t * N]),
            &(tableidx.data_ptr<int64_t>()[start_idx]),
            tr_tt_cores[0].packed_accessor32<float, 2, RestrictPtrTraits>(),
            d_tt_cores[0].packed_accessor32<float, 3, RestrictPtrTraits>());
      }
    } // for (int32_t t = T - 2; t >=0 ; --t)
  } // for (int32_t start_idx = 0; start_idx < nnz; start_idx += batch_count)

  if (optim == OPTIM_SGD) {
    for (int32_t t = 0; t < T; ++t) {
      int32_t y_size = tt_cores[t].size(1);
      int32_t x_size = tt_cores[t].size(2);
      int32_t tx = std::min(1024, y_size);
      int32_t ty = 1024 / tx;
      update_tt_cores_sgd_kernel<<<
          div_round_up(x_size, ty),
          dim3(tx, ty),
          0,
          c10::cuda::getCurrentCUDAStream()>>>(
          y_size,
          x_size,
          num_tables,
          learning_rate,
          d_tt_cores[t].packed_accessor32<float, 3, RestrictPtrTraits>(),
          tt_cores[t].packed_accessor32<float, 3, RestrictPtrTraits>());
    }
  }

  return d_tt_cores;
}

//backward dense
std::vector<Tensor> tt_embeddings_backward_dense_cuda(
    int32_t batch_count,
    int32_t D,
    const std::vector<int32_t>& tt_p_shapes,
    const std::vector<int32_t>& tt_q_shapes,
    const std::vector<int32_t>& tt_ranks,
    Tensor L,
    int32_t nnz,
    Tensor colidx,
    Tensor rowidx,
    Tensor tableidx,
    Tensor d_output,
    std::vector<Tensor>& tt_cores) {
  return tt_embeddings_backward_cuda(
      OPTIM_DENSE,
      batch_count,
      D,
      0.0,
      0.0,
      tt_p_shapes,
      tt_q_shapes,
      tt_ranks,
      L,
      nnz,
      colidx,
      rowidx,
      tableidx,
      d_output,
      c10::nullopt,
      tt_cores);
}

//backward sgd
void tt_embeddings_backward_sgd_cuda(
    int32_t batch_count,
    int32_t D,
    float learning_rate,
    const std::vector<int32_t>& tt_p_shapes,
    const std::vector<int32_t>& tt_q_shapes,
    const std::vector<int32_t>& tt_ranks,
    Tensor L,
    int32_t nnz,
    Tensor colidx,
    Tensor rowidx,
    Tensor tableidx,
    Tensor d_output,
    std::vector<Tensor>& tt_cores) {
  tt_embeddings_backward_cuda(
      OPTIM_SGD,
      batch_count,
      D,
      learning_rate,
      0.0,
      tt_p_shapes,
      tt_q_shapes,
      tt_ranks,
      L,
      nnz,
      colidx,
      rowidx,
      tableidx,
      d_output,
      c10::nullopt,
      tt_cores);
}
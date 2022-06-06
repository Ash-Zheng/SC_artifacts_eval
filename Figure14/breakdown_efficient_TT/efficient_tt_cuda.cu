#include <assert.h>
#include <ATen/ATen.h>

#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/TensorAccessor.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include "hashtbl_cuda_utils.cuh"
#include "tt_cuda_utils.cuh"
#include <iostream>

// #include "cub-1.8.0/cub/device/device_radix_sort.cuh"

#define WARP_SIZE 32
#define MAX_BATCH_SIZE 8192

namespace {

constexpr int32_t MAX_PROBES = 3;

enum {
  OPTIM_SGD = 0,
  OPTIM_ADAGRAD = 1,
  OPTIM_DENSE = 2,
};

}

#define checkKernelErrors(expr)                             \
  do {                                                      \
    expr;                                                   \
                                                            \
    cudaError_t __err = cudaGetLastError();                 \
    if (__err != cudaSuccess) {                             \
      printf("Line %d: '%s' failed: %s\n", __LINE__, #expr, \
             cudaGetErrorString(__err));                    \
      abort();                                              \
    }                                                       \
  } while (0)


using namespace at;

float** group_map; // content point to result of intermediate result.
int32_t* group_flag; // indicate group_map is/not have update
int32_t* group_idx;
float *cache;
float *output_d;
int32_t *group_idx_h;


void init_cuda(
    int32_t device_id,
    const std::vector<int>& tt_q_shape,
    const std::vector<int>& tt_ranks, //[1,r1,r2,1]
    int32_t batch_size,
    int32_t feature_dim
)
{
    cudaSetDevice(device_id);

    if(!group_map)
    {
      int min_length = 370 * 370;
      int32_t cache_dim = tt_q_shape[0] * tt_q_shape[1] * tt_ranks[2];

      cudaMalloc(&group_map, min_length*sizeof(float*));
      cudaMalloc(&group_flag, min_length*sizeof(int32_t));
      // cudaMalloc(&group_idx, sizeof(int32_t));  // GPU Mem
      cudaMallocManaged(&group_idx, sizeof(int32_t));  // unified Mem
      cudaMalloc(&cache, min_length * cache_dim * sizeof(float));
      cudaMalloc(&output_d, batch_size * feature_dim * sizeof(float));
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
    int batch_count) 
{
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

inline void stream_cuda_gemm_batched_fp32_fp32(
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
    int batch_count,
    cudaStream_t stream) 
{
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasSetStream(handle, stream);
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

__global__ void prepare_batch_gemm_pointers_3_core(
    int32_t batch_size,
    int32_t table_length,
    int32_t feature_dim,
    int32_t index_length,
    int32_t output_length,
    int32_t cache_length,
    int32_t cache_dim,
    const int64_t* index,
    const int64_t* tt_p_shape,
    const int64_t* tt_q_shape,
    const int64_t* tt_ranks,
    float* tt_core_0,
    float* tt_core_1,
    float* tt_core_2,
    float* cache,
    float* result,
    float** __restrict__ a_ptr,
    float** __restrict__ b_ptr,
    float** __restrict__ c_ptr,
    float** group_map,
    int32_t* group_flag,
    int32_t* group_idx
    )
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  
  if(n >= batch_size)
    return;

  int idx_start = n;
  int idx_end = n + 1;

  int group;
  for(int i=idx_start;i<idx_end;i++){
    int idx = *(index + i);
    float tmp = float(idx)/tt_p_shape[2];
    group = floor(tmp);
    int I3 = idx % tt_p_shape[2];
    if(atomicCAS(group_flag + group, 0, 1)==0)
    {
      int cache_idx = atomicAdd(group_idx, 1);
      int I1 = floor(float(group)/tt_p_shape[1]);
      int I2 = group%tt_p_shape[1];
      // int I2 = floor(float(group)/tt_p_shape[0]);
      // int I1 = group%tt_p_shape[0];

      a_ptr[cache_idx] = tt_core_1 + I2 * tt_ranks[1] * tt_q_shape[1] * tt_ranks[2]; 
      b_ptr[cache_idx] = tt_core_0 + I1 * tt_q_shape[0] * tt_ranks[1];
      c_ptr[cache_idx] = cache + cache_idx * cache_dim;
      group_map[group] = cache + cache_idx * cache_dim;
    }
    a_ptr[cache_length + n] = tt_core_2 + I3 * (tt_q_shape[2] * tt_ranks[2]); 
    b_ptr[cache_length + n] = group_map[group]; // from cache
    c_ptr[cache_length + n] = result + n * output_length;
  }
}


__global__ void update_group_map(
  int32_t batch_size,
  int32_t cache_length,
  int32_t index_length,
  const int64_t* index,
  const int64_t* tt_p_shape,
  float** group_map,
  float** __restrict__ b_ptr
)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;

  if(n >= batch_size)
    return;
  
  int idx = *(index + n);
  float tmp = float(idx)/tt_p_shape[2];
  int group = floor(tmp);
  // if(!b_ptr[cache_length + n])
  {
    b_ptr[cache_length + n] = group_map[group];
    // printf("%p\n",b_ptr[cache_length + n]);
  }
}

Tensor Efficient_TT_forward_cuda(
    int32_t batch_size,
    int32_t table_length,
    int32_t feature_dim,
    const Tensor index,
    const std::vector<int>& tt_p_shape, //[i1,i2,i3]
    const std::vector<int>& tt_q_shape, //[j1,j2,j3]
    const std::vector<int>& tt_ranks, //[1,r1,r2,1]
    const Tensor tensor_p_shape, //[i1,i2,i3]
    const Tensor tensor_q_shape, //[j1,j2,j3]
    const Tensor tensor_ranks, //[1,r1,r2,1]
    const std::vector<Tensor>& tt_cores
){
    auto output =
      at::zeros({batch_size, feature_dim}, tt_cores[0].options().dtype(at::kFloat)).to(at::kCUDA);
    
    int32_t index_length = index.sizes()[0];
    int32_t num_core = tt_p_shape.size();
    int32_t num_rank = tt_p_shape.size() + 1;
    int32_t cache_length = tt_p_shape[1] * tt_p_shape[2];
    int32_t cache_dim = tt_q_shape[0] * tt_q_shape[1] * tt_ranks[2];
    int32_t output_length = tt_q_shape[0] * tt_q_shape[1] * tt_q_shape[2];

    // // printf("\ncache_dim:%d,cache_length:%d,num_core:%d,num_rank:%d\n",cache_dim,cache_length,num_core,num_rank);

    auto a_ptr_tensor = at::empty(
      {cache_length + batch_size}, tt_cores[0].options().dtype(at::kLong)).to(at::kCUDA);
    auto b_ptr_tensor = at::empty(
      {cache_length + batch_size}, tt_cores[0].options().dtype(at::kLong)).to(at::kCUDA);
    auto c_ptr_tensor = at::empty(
      {cache_length + batch_size}, tt_cores[0].options().dtype(at::kLong)).to(at::kCUDA);
    float** a_ptr = (float**)a_ptr_tensor.data_ptr<int64_t>();
    float** b_ptr = (float**)b_ptr_tensor.data_ptr<int64_t>();
    float** c_ptr = (float**)c_ptr_tensor.data_ptr<int64_t>();

    // if(!group_map)
    // {
    //   int min_length = 370 * 370;
    //   cudaMalloc(&group_map, min_length*sizeof(float*));
    //   cudaMalloc(&group_flag, min_length*sizeof(int32_t));
    //   // cudaMalloc(&group_idx, sizeof(int32_t));  // GPU Mem
    //   cudaMallocManaged(&group_idx, sizeof(int32_t));  // unified Mem
    //   cudaMalloc(&cache, min_length * cache_dim * sizeof(float));
    //   cudaMalloc(&output_d, batch_size * feature_dim * sizeof(float));
    // }

    cudaMemset(group_flag, 0, cache_length*sizeof(int32_t)); //set to zero
    cudaMemset(group_idx, 0, sizeof(int32_t)); //set to zero

    // int32_t threads = 256;
    int32_t threads = (batch_size > 256 ? 256 : 32);
    int32_t num_blocks = (table_length + threads - 1) / threads; 
    prepare_batch_gemm_pointers_3_core<<<num_blocks, threads>>>(  // one thread lookup one row
      batch_size,
      table_length,
      feature_dim,
      index_length,
      output_length,
      cache_length,
      cache_dim,

      (const int64_t*)index.data_ptr(),
      (const int64_t*)tensor_p_shape.data_ptr(),
      (const int64_t*)tensor_q_shape.data_ptr(),
      (const int64_t*)tensor_ranks.data_ptr(),
      (float*)tt_cores[0].data_ptr(),
      (float*)tt_cores[1].data_ptr(),
      (float*)tt_cores[2].data_ptr(),
      cache,
      output_d,
      a_ptr,
      b_ptr,
      c_ptr,
      group_map,
      group_flag,
      group_idx
    );

    // int32_t *group_idx_h = (int32_t*)malloc(sizeof(int32_t));
    // cudaMemcpy(group_idx_h, group_idx, sizeof(int32_t), cudaMemcpyDeviceToHost);
    // printf("final group idx:%d\n",*group_idx_h);

    // use cuBlas batched gemm compute cache
    float alpha = 1.0;
    float beta = 0.0;
    // int batch_cnt = *group_idx_h;
    int batch_cnt = *group_idx;
    cuda_gemm_batched_fp32_fp32(
      CUBLAS_OP_N,
      CUBLAS_OP_N,
      tt_q_shape[1]*tt_ranks[2], //n
      tt_q_shape[0],//m
      tt_ranks[1],//k
      &alpha,
      (void**)a_ptr,
      tt_q_shape[1]*tt_ranks[2], //n
      (void**)b_ptr,
      tt_ranks[1], // k
      &beta,
      (void**)c_ptr,
      tt_q_shape[1]*tt_ranks[2], // n
      batch_cnt
    );

    update_group_map<<<num_blocks, threads>>>(
      batch_size,
      cache_length,
      index_length,
      (const int64_t*)index.data_ptr(),
      (const int64_t*)tensor_p_shape.data_ptr(),
      group_map,
      b_ptr
    );

    cuda_gemm_batched_fp32_fp32(
      CUBLAS_OP_N,
      CUBLAS_OP_N,
      tt_q_shape[2], //n
      tt_q_shape[0] * tt_q_shape[1],//m
      tt_ranks[2],//k
      &alpha,
      (void**)(a_ptr+cache_length),
      tt_q_shape[2], //n
      (void**)(b_ptr+cache_length),
      tt_ranks[2], // k
      &beta,
      (void**)(c_ptr+cache_length),
      tt_q_shape[2], // n
      batch_size
    );

    cudaMemcpy((float*)output.data_ptr(), output_d, batch_size * feature_dim * sizeof(float), cudaMemcpyDeviceToHost);

    return output;
}

//=============================================================================================================================
__global__ void prepare_batch_gemm_pointers_3_core_backward(
    int32_t N,
    const int64_t* __restrict__ index,
    const int64_t* tt_p_shape,

    PackedTensorAccessor32<float, 2, RestrictPtrTraits> tt_cores_0,
    PackedTensorAccessor32<float, 2, RestrictPtrTraits> tt_cores_1,
    PackedTensorAccessor32<float, 2, RestrictPtrTraits> tt_cores_2,
    PackedTensorAccessor32<float, 2, RestrictPtrTraits> tr_tt_cores_0,
    PackedTensorAccessor32<float, 2, RestrictPtrTraits> tr_tt_cores_1,
    PackedTensorAccessor32<float, 2, RestrictPtrTraits> tr_tt_cores_2,
    PackedTensorAccessor32<float, 2, RestrictPtrTraits> tr_0,
    PackedTensorAccessor32<float, 2, RestrictPtrTraits> d_output,
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
    int32_t idx = *(index+n);

    float tmp = float(idx)/tt_p_shape[2];
    int group = floor(tmp);
    int I3 = idx % tt_p_shape[2];
    int I1 = floor(float(group)/tt_p_shape[1]);
    int I2 = group%tt_p_shape[1];

    tt_idx[0 * N + n] = I1;
    tt_idx[1 * N + n] = I2;
    tt_idx[2 * N + n] = I3;
    float* tr_0_ptr = (float*)&(tr_0[n][0]);
    float* d_output_ptr = (float*)&(d_output[n][0]);
    float* tt_cores_0_ptr = (float*)&(tt_cores_0[I1][0]);
    float* tt_cores_1_ptr = (float*)&(tt_cores_1[I2][0]);
    a_ptr[0 * N + n] = tt_cores_1_ptr;
    b_ptr[0 * N + n] = tt_cores_0_ptr;
    c_ptr[0 * N + n] = tr_0_ptr;

    a0_ptr[1 * N + n] = tr_0_ptr;
    b0_ptr[1 * N + n] = d_output_ptr;
    c0_ptr[1 * N + n] = (float*)&(tr_tt_cores_2[n][0]);
    a1_ptr[1 * N + n] = d_output_ptr;
    b1_ptr[1 * N + n] = (float*)&(tt_cores_2[I3][0]);
    c1_ptr[1 * N + n] = tr_0_ptr;

    a0_ptr[0 * N + n] = tt_cores_0_ptr;
    b0_ptr[0 * N + n] = tr_0_ptr;
    c0_ptr[0 * N + n] = (float*)&(tr_tt_cores_1[n][0]);
    a1_ptr[0 * N + n] = tr_0_ptr;
    b1_ptr[0 * N + n] = tt_cores_1_ptr;
    c1_ptr[0 * N + n] = (float*)&(tr_tt_cores_0[n][0]);
  }
}

__global__ void update_d_tt_cores_kernel(
    int32_t N,
    int32_t D,
    const int32_t* __restrict__ tt_idx,
    PackedTensorAccessor32<float, 2, RestrictPtrTraits> tr_tt_cores,
    PackedTensorAccessor32<float, 2, RestrictPtrTraits> d_tt_cores) {
  int32_t n = blockIdx.x * blockDim.y + threadIdx.y;
  if (n < N) {
    auto idx = __ldg(&tt_idx[n]);
    for (int32_t d = threadIdx.x; d < D; d += blockDim.x) {
      atomicAdd(&(d_tt_cores[idx][d]), 0.1 * tr_tt_cores[n][d]);
    }
  }
}

__global__ void update_tt_cores_sgd_kernel(
    int32_t B,
    int32_t D,
    float learning_rate,
    PackedTensorAccessor32<float, 2, RestrictPtrTraits> d_tt_cores,
    PackedTensorAccessor32<float, 2, RestrictPtrTraits> tt_core
    ) {
  int32_t b = blockIdx.x * blockDim.y + threadIdx.y;
  if (b >= B) {
    return;
  }
  for (int32_t d = threadIdx.x; d < D; d += blockDim.x) {
    // tt_core[b][d] -= learning_rate * d_tt_cores[b][d];
    tt_core[b][d] -= d_tt_cores[b][d];
  }
}

__global__ void fused_update_tt_cores_sgd_kernel(
    int32_t B,
    int32_t D,
    float learning_rate,
    const int32_t* __restrict__ tt_idx,
    PackedTensorAccessor32<float, 2, RestrictPtrTraits> tr_tt_cores,
    PackedTensorAccessor32<float, 2, RestrictPtrTraits> tt_core
    ) {
  int32_t n = blockIdx.x * blockDim.y + threadIdx.y;
  if (n >= B) {
    return;
  }
  learning_rate *= -1;
  auto idx = __ldg(&tt_idx[n]);
  for (int32_t d = threadIdx.x; d < D; d += blockDim.x) {
    auto delta = learning_rate * tr_tt_cores[n][d];
    atomicAdd(&(tt_core[idx][d]), delta);
  }
}

void Efficient_TT_backward_sgd_cuda(
    int32_t batch_size,
    int32_t table_length,
    int32_t feature_dim,
    float learning_rate,

    const Tensor index,
    const std::vector<int>& tt_p_shapes, //[i1,i2,i3]
    const std::vector<int>& tt_q_shapes, //[j1,j2,j3]
    const std::vector<int>& tt_ranks, //[1,r1,r2,1]
    const Tensor tensor_p_shapes, //[i1,i2,i3]
    const Tensor tensor_q_shapes, //[j1,j2,j3]
    const Tensor tensor_ranks, //[1,r1,r2,1]
    Tensor d_output,
    std::vector<Tensor>& tt_cores
)
{
    at::cuda::OptionalCUDAGuard device_guard;
    device_guard.set_index(d_output.get_device());
    int32_t T = 3;  //3
    int32_t batch_count = batch_size;

    std::vector<Tensor> d_tt_cores;
    std::vector<Tensor> tr_tt_cores;
    d_tt_cores.push_back(at::zeros_like(tt_cores[0]));
    d_tt_cores.push_back(at::zeros_like(tt_cores[1]));
    d_tt_cores.push_back(at::zeros_like(tt_cores[2]));
    
    tr_tt_cores.push_back(at::empty({batch_size, tt_cores[0].size(1)}, tt_cores[0].options()));
    tr_tt_cores.push_back(at::empty({batch_size, tt_cores[1].size(1)}, tt_cores[1].options()));
    tr_tt_cores.push_back(at::empty({batch_size, tt_cores[2].size(1)}, tt_cores[2].options()));

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

    int32_t start_idx = 0;
    int32_t end_idx = start_idx + batch_count;
    int32_t N = end_idx - start_idx;

    int32_t threads = (N > 256 ? 256 : 32);
    int32_t num_blocks = (N + threads - 1) / threads;

    prepare_batch_gemm_pointers_3_core_backward<<<
        num_blocks,
        threads,
        0,
        c10::cuda::getCurrentCUDAStream()>>>(
        N,
        (const int64_t*)index.data_ptr(),
        (const int64_t*)tensor_p_shapes.data_ptr(),

        tt_cores[0].packed_accessor32<float, 2, RestrictPtrTraits>(),
        tt_cores[1].packed_accessor32<float, 2, RestrictPtrTraits>(),
        tt_cores[2].packed_accessor32<float, 2, RestrictPtrTraits>(),

        tr_tt_cores[0].packed_accessor32<float, 2, RestrictPtrTraits>(),
        tr_tt_cores[1].packed_accessor32<float, 2, RestrictPtrTraits>(),
        tr_tt_cores[2].packed_accessor32<float, 2, RestrictPtrTraits>(),
        tr[0].packed_accessor32<float, 2, RestrictPtrTraits>(),
        d_output.packed_accessor32<float, 2, RestrictPtrTraits>(),
        tt_idx.data_ptr<int32_t>(),
        a_ptr,
        b_ptr,
        c_ptr,
        a0_ptr,
        b0_ptr,
        c0_ptr,
        a1_ptr,
        b1_ptr,
        c1_ptr
    );
    
    cuda_gemm_batched_fp32_fp32(
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        n[0],
        m[0],
        k[0],
        &alpha,
        (void**)&(a_ptr[0]),
        n[0],
        (void**)&(b_ptr[0]),
        k[0],
        &beta,
        (void**)&(c_ptr[0]),
        n[0],
        N);
    // for (int32_t t = 0; t < T - 2; ++t)
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
        int32_t D_0 = tt_cores[t + 1].size(1);
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
            tr_tt_cores[t + 1].packed_accessor32<float, 2, RestrictPtrTraits>(),
            d_tt_cores[t + 1].packed_accessor32<float, 2, RestrictPtrTraits>());
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
            int32_t D_1 = tt_cores[0].size(1);
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
                tr_tt_cores[0].packed_accessor32<float, 2, RestrictPtrTraits>(),
                d_tt_cores[0].packed_accessor32<float, 2, RestrictPtrTraits>());
        }
    } // for (int32_t t = T - 2; t >=0 ; --t)

    for (int32_t t = 0; t < T; ++t) {
        int32_t y_size = tt_cores[t].size(0);
        int32_t x_size = tt_cores[t].size(1);
        int32_t tx = std::min(1024, y_size);
        int32_t ty = 1024 / tx;
      
        update_tt_cores_sgd_kernel<<<
        div_round_up(x_size, ty),
        dim3(tx, ty),
        0,
        c10::cuda::getCurrentCUDAStream()>>>(
          y_size,
          x_size,
          learning_rate, // hard code
          d_tt_cores[t].packed_accessor32<float, 2, RestrictPtrTraits>(),
          tt_cores[t].packed_accessor32<float, 2, RestrictPtrTraits>()
        );
    }
    
  return;
}


void Fused_Efficient_TT_backward_sgd_cuda(
    int32_t batch_size,
    int32_t table_length,
    int32_t feature_dim,
    float learning_rate,

    const Tensor index,
    const std::vector<int>& tt_p_shapes, //[i1,i2,i3]
    const std::vector<int>& tt_q_shapes, //[j1,j2,j3]
    const std::vector<int>& tt_ranks, //[1,r1,r2,1]
    const Tensor tensor_p_shapes, //[i1,i2,i3]
    const Tensor tensor_q_shapes, //[j1,j2,j3]
    const Tensor tensor_ranks, //[1,r1,r2,1]
    Tensor d_output,
    std::vector<Tensor>& tt_cores
)
{
    at::cuda::OptionalCUDAGuard device_guard;
    device_guard.set_index(d_output.get_device());
    int32_t T = 3;  //3
    int32_t batch_count = batch_size;

    std::vector<Tensor> tr_tt_cores;
    tr_tt_cores.push_back(at::empty({batch_size, tt_cores[0].size(1)}, tt_cores[0].options()));
    tr_tt_cores.push_back(at::empty({batch_size, tt_cores[1].size(1)}, tt_cores[1].options()));
    tr_tt_cores.push_back(at::empty({batch_size, tt_cores[2].size(1)}, tt_cores[2].options()));

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

    int32_t start_idx = 0;
    int32_t end_idx = start_idx + batch_count;
    int32_t N = end_idx - start_idx;

    int32_t threads = (N > 256 ? 256 : 32);
    int32_t num_blocks = (N + threads - 1) / threads;

    prepare_batch_gemm_pointers_3_core_backward<<<
        num_blocks,
        threads,
        0,
        c10::cuda::getCurrentCUDAStream()>>>(
        N,
        (const int64_t*)index.data_ptr(),
        (const int64_t*)tensor_p_shapes.data_ptr(),

        tt_cores[0].packed_accessor32<float, 2, RestrictPtrTraits>(),
        tt_cores[1].packed_accessor32<float, 2, RestrictPtrTraits>(),
        tt_cores[2].packed_accessor32<float, 2, RestrictPtrTraits>(),

        tr_tt_cores[0].packed_accessor32<float, 2, RestrictPtrTraits>(),
        tr_tt_cores[1].packed_accessor32<float, 2, RestrictPtrTraits>(),
        tr_tt_cores[2].packed_accessor32<float, 2, RestrictPtrTraits>(),
        tr[0].packed_accessor32<float, 2, RestrictPtrTraits>(),
        d_output.packed_accessor32<float, 2, RestrictPtrTraits>(),
        tt_idx.data_ptr<int32_t>(),
        a_ptr,
        b_ptr,
        c_ptr,
        a0_ptr,
        b0_ptr,
        c0_ptr,
        a1_ptr,
        b1_ptr,
        c1_ptr
    );
    
    cuda_gemm_batched_fp32_fp32(
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        n[0],
        m[0],
        k[0],
        &alpha,
        (void**)&(a_ptr[0]),
        n[0],
        (void**)&(b_ptr[0]),
        k[0],
        &beta,
        (void**)&(c_ptr[0]),
        n[0],
        N);

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
            N
            );
        
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
            N
            );
    } // for (int32_t t = T - 2; t >=0 ; --t)

    for (int32_t t = 0; t < T; ++t) {
        int32_t D_0 = tt_cores[t].size(1);
        int32_t tx_0 = std::min(1024, D_0);
        int32_t ty_0 = 1024 / tx_0;
        fused_update_tt_cores_sgd_kernel<<<
        div_round_up(N, ty_0),
        dim3(tx_0, ty_0),
        0,
        c10::cuda::getCurrentCUDAStream()>>>(
          N,
          D_0,
          learning_rate, // hard code
          &(tt_idx.data_ptr<int32_t>()[t * N]),
          tr_tt_cores[t].packed_accessor32<float, 2, RestrictPtrTraits>(),
          tt_cores[t].packed_accessor32<float, 2, RestrictPtrTraits>()
        );
    }
    
  return;
}


// Extra_Eff_Fused ============================================================================

__global__ void Extra_Eff_prepare_batch_gemm_pointers_3_core_backward(
    int32_t unique_num,
    const int64_t* __restrict__ unique_index,
    const int64_t* tt_p_shape,

    PackedTensorAccessor32<float, 2, RestrictPtrTraits> tt_cores_0,
    PackedTensorAccessor32<float, 2, RestrictPtrTraits> tt_cores_1,
    PackedTensorAccessor32<float, 2, RestrictPtrTraits> tt_cores_2,
    PackedTensorAccessor32<float, 2, RestrictPtrTraits> tr_tt_cores_0,
    PackedTensorAccessor32<float, 2, RestrictPtrTraits> tr_tt_cores_1,
    PackedTensorAccessor32<float, 2, RestrictPtrTraits> tr_tt_cores_2,
    PackedTensorAccessor32<float, 2, RestrictPtrTraits> tr_0,
    PackedTensorAccessor32<float, 2, RestrictPtrTraits> d_output,
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
  if (n < unique_num) {
    int32_t idx = *(unique_index+n);

    float tmp = float(idx)/tt_p_shape[2];
    int group = floor(tmp);
    int I3 = idx % tt_p_shape[2];
    int I1 = floor(float(group)/tt_p_shape[1]);
    int I2 = group%tt_p_shape[1];

    tt_idx[0 * unique_num + n] = I1;
    tt_idx[1 * unique_num + n] = I2;
    tt_idx[2 * unique_num + n] = I3;
    float* tr_0_ptr = (float*)&(tr_0[n][0]);
    float* d_output_ptr = (float*)&(d_output[n][0]);
    float* tt_cores_0_ptr = (float*)&(tt_cores_0[I1][0]);
    float* tt_cores_1_ptr = (float*)&(tt_cores_1[I2][0]);
    a_ptr[0 * unique_num + n] = tt_cores_1_ptr; //T1*T2
    b_ptr[0 * unique_num + n] = tt_cores_0_ptr;
    c_ptr[0 * unique_num + n] = tr_0_ptr;

    a0_ptr[1 * unique_num + n] = tr_0_ptr;
    b0_ptr[1 * unique_num + n] = d_output_ptr;
    c0_ptr[1 * unique_num + n] = (float*)&(tr_tt_cores_2[n][0]);
    a1_ptr[1 * unique_num + n] = d_output_ptr;
    b1_ptr[1 * unique_num + n] = (float*)&(tt_cores_2[I3][0]);
    c1_ptr[1 * unique_num + n] = tr_0_ptr;

    a0_ptr[0 * unique_num + n] = tt_cores_0_ptr; 
    b0_ptr[0 * unique_num + n] = tr_0_ptr;
    c0_ptr[0 * unique_num + n] = (float*)&(tr_tt_cores_1[n][0]);
    a1_ptr[0 * unique_num + n] = tr_0_ptr;
    b1_ptr[0 * unique_num + n] = tt_cores_1_ptr;
    c1_ptr[0 * unique_num + n] = (float*)&(tr_tt_cores_0[n][0]);
  }
}


__global__ void compute_unique_gradient(
  int batch_size,
  int feature_dim,
  const int64_t* inverse, 
  PackedTensorAccessor32<float, 2, RestrictPtrTraits> d_input,
  PackedTensorAccessor32<float, 2, RestrictPtrTraits> d_output
)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= batch_size)
    return;

  int idx = inverse[n];
  for(int i=0;i<feature_dim;i++)
  {
    atomicAdd(&(d_output[idx][i]), d_input[n][i]);
  }
}


__global__ void extra_fused_update_tt_cores_sgd_kernel(
    int32_t B,
    int32_t D,
    float learning_rate,
    const int32_t* __restrict__ tt_idx,
    PackedTensorAccessor32<float, 2, RestrictPtrTraits> tr_tt_cores,
    PackedTensorAccessor32<float, 2, RestrictPtrTraits> tt_core
    ) {
  int32_t n = blockIdx.x * blockDim.y + threadIdx.y;
  if (n >= B) {
    return;
  }
  auto idx = __ldg(&tt_idx[n]);
  for (int32_t d = threadIdx.x; d < D; d += blockDim.x) {
    auto delta = -1 * learning_rate * tr_tt_cores[n][d];
    atomicAdd(&(tt_core[idx][d]), delta);
    // tt_core[idx][d] -= learning_rate * tr_tt_cores[n][d];
  }
}


void Fused_Extra_Efficient_TT_backward_sgd_cuda(
    int32_t batch_size,
    int32_t table_length,
    int32_t feature_dim,
    float learning_rate,

    const Tensor index,
    const std::vector<int>& tt_p_shapes, //[i1,i2,i3]
    const std::vector<int>& tt_q_shapes, //[j1,j2,j3]
    const std::vector<int>& tt_ranks, //[1,r1,r2,1]
    const Tensor tensor_p_shapes, //[i1,i2,i3]
    const Tensor tensor_q_shapes, //[j1,j2,j3]
    const Tensor tensor_ranks, //[1,r1,r2,1]
    Tensor d_output,
    std::vector<Tensor>& tt_cores,
    Tensor sorted_idx,
    Tensor sorted_key
)
{
    at::cuda::OptionalCUDAGuard device_guard;
    device_guard.set_index(d_output.get_device());
    int32_t T = 3;  //3
    int32_t batch_count = batch_size;
    int32_t N = batch_size;
    int32_t unique_num = sorted_idx.size(0);


    auto unique_d = at::zeros({sorted_idx.size(0), d_output.size(1)}, tt_cores[0].options());

    int32_t threads = (N > 256 ? 256 : 32);
    int32_t num_blocks = (N + threads - 1) / threads;

    compute_unique_gradient<<<num_blocks, threads>>>(
      batch_size,
      feature_dim,
      (const int64_t*)sorted_key.data_ptr(),
      d_output.packed_accessor32<float, 2, RestrictPtrTraits>(),
      unique_d.packed_accessor32<float, 2, RestrictPtrTraits>()
    );

    //===================================================================================================================
    std::vector<Tensor> tr_tt_cores;
    tr_tt_cores.push_back(at::empty({unique_num, tt_cores[0].size(1)}, tt_cores[0].options()));
    tr_tt_cores.push_back(at::empty({unique_num, tt_cores[1].size(1)}, tt_cores[1].options()));
    tr_tt_cores.push_back(at::empty({unique_num, tt_cores[2].size(1)}, tt_cores[2].options()));

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

    std::vector<Tensor> tr;

    int64_t tr_size = tt_q_shapes[0] * tt_ranks[1];
    for (int32_t t = 0; t < T - 2; ++t) {
        tr_size = tr_size * tt_q_shapes[t + 1] * tt_ranks[t + 2] / tt_ranks[t + 1];
        tr.push_back(at::empty({unique_num, tr_size}, tt_cores[0].options()));
    }

    auto tt_idx =
      at::empty({T * unique_num}, tt_cores[0].options().dtype(at::kInt));
    auto a_ptr_tensor = at::empty(
        {(T - 2) * unique_num}, tt_cores[0].options().dtype(at::kLong));
    auto b_ptr_tensor = at::empty(
        {(T - 2) * unique_num}, tt_cores[0].options().dtype(at::kLong));
    auto c_ptr_tensor = at::empty(
        {(T - 2) * unique_num}, tt_cores[0].options().dtype(at::kLong));
    float** a_ptr = (float**)a_ptr_tensor.data_ptr<int64_t>();
    float** b_ptr = (float**)b_ptr_tensor.data_ptr<int64_t>();
    float** c_ptr = (float**)c_ptr_tensor.data_ptr<int64_t>();
    auto a0_ptr_tensor = at::empty(
        {(T - 1) * unique_num}, tt_cores[0].options().dtype(at::kLong));
    auto b0_ptr_tensor = at::empty(
        {(T - 1) * unique_num}, tt_cores[0].options().dtype(at::kLong));
    auto c0_ptr_tensor = at::empty(
        {(T - 1) * unique_num}, tt_cores[0].options().dtype(at::kLong));
    float** a0_ptr = (float**)a0_ptr_tensor.data_ptr<int64_t>();
    float** b0_ptr = (float**)b0_ptr_tensor.data_ptr<int64_t>();
    float** c0_ptr = (float**)c0_ptr_tensor.data_ptr<int64_t>();
    auto a1_ptr_tensor = at::empty(
        {(T - 1) * unique_num}, tt_cores[0].options().dtype(at::kLong));
    auto b1_ptr_tensor = at::empty(
        {(T - 1) * unique_num}, tt_cores[0].options().dtype(at::kLong));
    auto c1_ptr_tensor = at::empty(
        {(T - 1) * unique_num}, tt_cores[0].options().dtype(at::kLong));
    float** a1_ptr = (float**)a1_ptr_tensor.data_ptr<int64_t>();
    float** b1_ptr = (float**)b1_ptr_tensor.data_ptr<int64_t>();
    float** c1_ptr = (float**)c1_ptr_tensor.data_ptr<int64_t>();

    threads = (unique_num > 256 ? 256 : 32);
    num_blocks = (unique_num + threads - 1) / threads;

    Extra_Eff_prepare_batch_gemm_pointers_3_core_backward<<<
        num_blocks,
        threads,
        0,
        c10::cuda::getCurrentCUDAStream()>>>(
        unique_num,
        (const int64_t*)sorted_idx.data_ptr(),
        (const int64_t*)tensor_p_shapes.data_ptr(),

        tt_cores[0].packed_accessor32<float, 2, RestrictPtrTraits>(),
        tt_cores[1].packed_accessor32<float, 2, RestrictPtrTraits>(),
        tt_cores[2].packed_accessor32<float, 2, RestrictPtrTraits>(),

        tr_tt_cores[0].packed_accessor32<float, 2, RestrictPtrTraits>(),
        tr_tt_cores[1].packed_accessor32<float, 2, RestrictPtrTraits>(),
        tr_tt_cores[2].packed_accessor32<float, 2, RestrictPtrTraits>(),
        tr[0].packed_accessor32<float, 2, RestrictPtrTraits>(),
        unique_d.packed_accessor32<float, 2, RestrictPtrTraits>(),
        tt_idx.data_ptr<int32_t>(),
        a_ptr,
        b_ptr,
        c_ptr,
        a0_ptr,
        b0_ptr,
        c0_ptr,
        a1_ptr,
        b1_ptr,
        c1_ptr
    );
    
    cuda_gemm_batched_fp32_fp32(
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        n[0],
        m[0],
        k[0],
        &alpha,
        (void**)&(a_ptr[0]),
        n[0],
        (void**)&(b_ptr[0]),
        k[0],
        &beta,
        (void**)&(c_ptr[0]),
        n[0],
        unique_num);

// //======================================================
    cuda_gemm_batched_fp32_fp32(
      CUBLAS_OP_N,
      CUBLAS_OP_T,
      n[1],
      k[1],
      m[1],
      &alpha,
      (void**)&(b0_ptr[unique_num]),
      n[1],
      (void**)&(a0_ptr[unique_num]),
      k[1],
      &beta,
      (void**)&(c0_ptr[unique_num]),
      n[1],
      unique_num
    );

    cuda_gemm_batched_fp32_fp32(
      CUBLAS_OP_T,
      CUBLAS_OP_N,
      k[1],
      m[1],
      n[1],
      &alpha,
      (void**)&(b1_ptr[unique_num]),
      n[1],
      (void**)&(a1_ptr[unique_num]),
      n[1],
      &beta,
      (void**)&(c1_ptr[unique_num]),
      k[1],
      unique_num
    );
//=========================================================

    cuda_gemm_batched_fp32_fp32(
      CUBLAS_OP_N,
      CUBLAS_OP_T,
      n[0],
      k[0],
      m[0],
      &alpha,
      (void**)&(b0_ptr[0]),
      n[0],
      (void**)&(a0_ptr[0]),
      k[0],
      &beta,
      (void**)&(c0_ptr[0]),
      n[0],
      unique_num
    );

    cuda_gemm_batched_fp32_fp32(
      CUBLAS_OP_T,
      CUBLAS_OP_N,
      k[0],
      m[0],
      n[0],
      &alpha,
      (void**)&(b1_ptr[0]),
      n[0],
      (void**)&(a1_ptr[0]),
      n[0],
      &beta,
      (void**)&(c1_ptr[0]),
      k[0],
      unique_num
    );

//=========================================================
  // return (a + b - 1) / b;
  for (int32_t t = 0; t < T; ++t) {
      int32_t D_0 = tt_cores[t].size(1);
      int32_t tx_0 = std::min(1024, D_0);
      int32_t ty_0 = 1024 / tx_0;
      extra_fused_update_tt_cores_sgd_kernel<<<
      div_round_up(unique_num, ty_0),
      dim3(tx_0, ty_0),
      0,
      c10::cuda::getCurrentCUDAStream()>>>(
        unique_num,
        D_0,
        learning_rate,
        &(tt_idx.data_ptr<int32_t>()[t * unique_num]),
        tr_tt_cores[t].packed_accessor32<float, 2, RestrictPtrTraits>(),
        tt_cores[t].packed_accessor32<float, 2, RestrictPtrTraits>()
      );
  }
  return;
}


// ==================================================================================================================
// ==================================================================================================================
// ==================================================================================================================
// ==================================================================================================================
// ==================================================================================================================
// ==================================================================================================================
// ==================================================================================================================
// ==================================================================================================================
// ==================================================================================================================
// ==================================================================================================================
// ==================================================================================================================
// ==================================================================================================================
// ==================================================================================================================

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

__global__ void init_batch_gemm_forward_3T_kernel(
    int N,
    const int64_t* __restrict__ L,
    const int64_t* __restrict__ colidx,
    const int64_t* __restrict__ tableidx,
    PackedTensorAccessor32<float, 2, RestrictPtrTraits> tt_cores_0,
    PackedTensorAccessor32<float, 2, RestrictPtrTraits> tt_cores_1,
    PackedTensorAccessor32<float, 2, RestrictPtrTraits> tt_cores_2,
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
    a_ptr[0 * N + n] = (float*)&(tt_cores_1[tt_idx_1][0]);
    b_ptr[0 * N + n] = (float*)&(tt_cores_0[tt_idx_0][0]);
    c_ptr[0 * N + n] = tr_0_ptr;
    a_ptr[1 * N + n] = (float*)&(tt_cores_2[tt_idx_2][0]);
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
        tt_cores[0].packed_accessor32<float, 2, RestrictPtrTraits>(),
        tt_cores[1].packed_accessor32<float, 2, RestrictPtrTraits>(),
        tt_cores[2].packed_accessor32<float, 2, RestrictPtrTraits>(),
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
      at::zeros({B, D}, tt_cores[0].options().dtype(at::kFloat));

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


// ========================================================================


__global__ void init_batch_gemm_backward_3T_kernel(
    int32_t N,
    const int64_t* __restrict__ colidx,
    const int64_t* __restrict__ rowidx,
    const int64_t* __restrict__ tableidx,
    const int64_t* __restrict__ L,
    PackedTensorAccessor32<float, 2, RestrictPtrTraits> tt_cores_0,
    PackedTensorAccessor32<float, 2, RestrictPtrTraits> tt_cores_1,
    PackedTensorAccessor32<float, 2, RestrictPtrTraits> tt_cores_2,
    PackedTensorAccessor32<float, 2, RestrictPtrTraits> tr_tt_cores_0,
    PackedTensorAccessor32<float, 2, RestrictPtrTraits> tr_tt_cores_1,
    PackedTensorAccessor32<float, 2, RestrictPtrTraits> tr_tt_cores_2,
    PackedTensorAccessor32<float, 2, RestrictPtrTraits> tr_0,
    PackedTensorAccessor32<float, 2, RestrictPtrTraits> d_output,
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
    float* d_output_ptr = (float*)&(d_output[ridx][0]);
    float* tt_cores_0_ptr = (float*)&(tt_cores_0[tt_idx_0][0]);
    float* tt_cores_1_ptr = (float*)&(tt_cores_1[tt_idx_1][0]);
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
    b1_ptr[1 * N + n] = (float*)&(tt_cores_2[tt_idx_2][0]);
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
        tt_cores[0].packed_accessor32<float, 2, RestrictPtrTraits>(),
        tt_cores[1].packed_accessor32<float, 2, RestrictPtrTraits>(),
        tt_cores[2].packed_accessor32<float, 2, RestrictPtrTraits>(),
        tr_tt_cores[0].packed_accessor32<float, 2, RestrictPtrTraits>(),
        tr_tt_cores[1].packed_accessor32<float, 2, RestrictPtrTraits>(),
        tr_tt_cores[2].packed_accessor32<float, 2, RestrictPtrTraits>(),
        tr[0].packed_accessor32<float, 2, RestrictPtrTraits>(),
        d_output.packed_accessor32<float, 2, RestrictPtrTraits>(),
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
    PackedTensorAccessor32<float, 2, RestrictPtrTraits> d_tt_cores,
    PackedTensorAccessor32<float, 2, RestrictPtrTraits> tt_cores) {
  int32_t b = blockIdx.x * blockDim.y + threadIdx.y;
  if (b >= B) {
    return;
  }
  for (int32_t i = 0; i < num_tables; i++) {
    for (int32_t d = threadIdx.x; d < D; d += blockDim.x) {
      tt_cores[b][d] -= learning_rate * d_tt_cores[b][d];
    }
  }
}

__global__ void update_d_tt_cores_kernel(
    int32_t N,
    int32_t D,
    const int32_t* __restrict__ tt_idx,
    const int64_t* __restrict__ tableidx,
    PackedTensorAccessor32<float, 2, RestrictPtrTraits> tr_tt_cores,
    PackedTensorAccessor32<float, 2, RestrictPtrTraits> d_tt_cores) {
  int32_t n = blockIdx.x * blockDim.y + threadIdx.y;
  if (n < N) {
    auto idx = __ldg(&tt_idx[n]);
    auto tidx = __ldg(&tableidx[n]);
    for (int32_t d = threadIdx.x; d < D; d += blockDim.x) {
      atomicAdd(&(d_tt_cores[idx][d]), tr_tt_cores[n][d]);
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
  int32_t T = 3;  //3
  int32_t num_tables = 1; //1

  std::vector<Tensor> d_tt_cores;
  std::vector<Tensor> tr_tt_cores;
  for (int32_t t = 0; t < T; ++t) {
    d_tt_cores.push_back(at::zeros_like(tt_cores[t]));
    tr_tt_cores.push_back(
        at::empty({batch_count, tt_cores[t].size(1)}, tt_cores[t].options()));
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
      int32_t D_0 = tt_cores[t + 1].size(1);
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
          d_tt_cores[t + 1].packed_accessor32<float, 2, RestrictPtrTraits>());
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
        int32_t D_1 = tt_cores[0].size(1);
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
            d_tt_cores[0].packed_accessor32<float, 2, RestrictPtrTraits>());
      }
    } // for (int32_t t = T - 2; t >=0 ; --t)
  } // for (int32_t start_idx = 0; start_idx < nnz; start_idx += batch_count)

  if (optim == OPTIM_SGD) {
    for (int32_t t = 0; t < T; ++t) {
      int32_t y_size = tt_cores[t].size(0);
      int32_t x_size = tt_cores[t].size(1);
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
          d_tt_cores[t].packed_accessor32<float, 2, RestrictPtrTraits>(),
          tt_cores[t].packed_accessor32<float, 2, RestrictPtrTraits>());
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



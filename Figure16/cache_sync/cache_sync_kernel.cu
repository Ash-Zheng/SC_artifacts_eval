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
#include <thrust/pair.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>

#include <cassert>
#include <iterator>
#include <limits>
#include <type_traits>


using namespace at;

int32_t* pair_num;

__global__ void get_target_index(
  int32_t* pair_num,
  int32_t this_length,
  int32_t last_length,
  const int64_t* this_index,
  const int64_t* last_index,
  PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> target_index
)
{
  int n = blockIdx.x;
  int t_id = threadIdx.x;

  // printf("b_id:%d,t_id:%d\n", n, t_id);

  if(t_id*32 >= last_length)
    return;
  
  int64_t this_idx = this_index[n];
  int end_idx = (t_id+1)*32;
  if(end_idx > last_length)
    end_idx = last_length;

  for(int i=t_id*32; i<end_idx ;i++) // check 32 idx
  {
    if(last_index[i]==this_idx)
    {
        target_index[n] = i;
        // printf("find:%d,%ld\n",last_index[i],this_idx);
        atomicAdd(pair_num, 1);
        return;
    }
  }
  target_index[n] = -1;
}

__global__ void updata_emb(
  int32_t this_length,
  int32_t feature_dim,
  PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> target_index,
  // PackedTensorAccessor32<float, 2, RestrictPtrTraits> this_emb,
  float* this_emb,
  PackedTensorAccessor32<float, 2, RestrictPtrTraits> last_emb
)
{
  int n = blockIdx.x;
  int t_id = threadIdx.x;
  
  int32_t target_id = target_index[n];
  if(target_id < 0 or n > this_length)
    return;

  *(this_emb + n*feature_dim + t_id) = last_emb[target_id][t_id];
  // this_emb[n][t_id] = last_emb[target_id][t_id];
}

void cache_sync_cuda(
    const Tensor this_unique,
    const Tensor last_unique,
    Tensor this_emb,
    const Tensor last_emb
)
{
    int32_t this_length = this_unique.size(0);
    int32_t last_length = last_unique.size(0);
    int32_t feature_dim = last_emb.size(1);

    cudaMallocManaged(&pair_num, sizeof(int32_t));  // unified Mem
    cudaMemset(pair_num, 0, sizeof(int32_t)); //set to zero

    auto target_index =
      at::zeros({this_length}, this_unique.options().dtype(at::kInt));

    int32_t block_num = this_length;
    int32_t thread_num = last_length/32 + 1;

    get_target_index<<<block_num, thread_num, 0, c10::cuda::getCurrentCUDAStream()>>>(
        pair_num,
        this_length,
        last_length,
        (const int64_t*)this_unique.data_ptr(),
        (const int64_t*)last_unique.data_ptr(),
        target_index.packed_accessor32<int, 1, RestrictPtrTraits>()
    );

    thread_num = feature_dim;
    cudaDeviceSynchronize();
    // printf("pair num:%d\n",*pair_num);
    updata_emb<<<block_num, thread_num, 0, c10::cuda::getCurrentCUDAStream()>>>(
        this_length,
        feature_dim,
        target_index.packed_accessor32<int, 1, RestrictPtrTraits>(),
        // this_emb.packed_accessor32<float, 2, RestrictPtrTraits>(),
        (float*)this_emb.data_ptr(),
        last_emb.packed_accessor32<float, 2, RestrictPtrTraits>()
    );
    // printf("pair num:%d\n",*pair_num);

}

int32_t cache_sync_cuda_with_return(
    const Tensor this_unique,
    const Tensor last_unique,
    Tensor this_emb,
    const Tensor last_emb
)
{
    int32_t this_length = this_unique.size(0);
    int32_t last_length = last_unique.size(0);
    int32_t feature_dim = last_emb.size(1);

    cudaMallocManaged(&pair_num, sizeof(int32_t));  // unified Mem
    cudaMemset(pair_num, 0, sizeof(int32_t)); //set to zero

    auto target_index =
      at::zeros({this_length}, this_unique.options().dtype(at::kInt));

    int32_t block_num = this_length;
    int32_t thread_num = last_length/32 + 1;

    get_target_index<<<block_num, thread_num, 0, c10::cuda::getCurrentCUDAStream()>>>(
        pair_num,
        this_length,
        last_length,
        (const int64_t*)this_unique.data_ptr(),
        (const int64_t*)last_unique.data_ptr(),
        target_index.packed_accessor32<int, 1, RestrictPtrTraits>()
    );

    thread_num = feature_dim;
    cudaDeviceSynchronize();
    // printf("pair num:%d\n",*pair_num);
    updata_emb<<<block_num, thread_num, 0, c10::cuda::getCurrentCUDAStream()>>>(
        this_length,
        feature_dim,
        target_index.packed_accessor32<int, 1, RestrictPtrTraits>(),
        // this_emb.packed_accessor32<float, 2, RestrictPtrTraits>(),
        (float*)this_emb.data_ptr(),
        last_emb.packed_accessor32<float, 2, RestrictPtrTraits>()
    );
    // printf("pair num:%d\n",*pair_num);
    return *pair_num;
}

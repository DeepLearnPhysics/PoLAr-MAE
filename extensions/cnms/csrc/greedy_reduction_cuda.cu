// greedy_reduction_cuda.cu
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void greedy_reduction_kernel(
    const int* __restrict__ sorted_indices,
    const int* __restrict__ idx,
    const int* __restrict__ lengths,
    bool* __restrict__ retain,
    int num_batches,
    int num_spheres,
    int num_neighbors,
    int ignore_idx
) {
    int batch_idx = blockIdx.x;
    int sphere_offset = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (batch_idx >= num_batches) return;
    
    int valid_length = lengths[batch_idx];
    
    // init retain array
    for (int i = sphere_offset; i < num_spheres; i += blockDim.x * gridDim.y) {
        retain[batch_idx * num_spheres + i] = (i < valid_length);
    }
    __syncthreads();
    
    // process spheres in sorted order
    for (int i = 0; i < num_spheres; i++) {
        int sphere_idx = sorted_indices[batch_idx * num_spheres + i];
        
        // all threads need this check
        bool should_process = retain[batch_idx * num_spheres + sphere_idx];
        __syncthreads();
        
        if (!should_process) continue;
        
        // process neighbors
        for (int j = sphere_offset; j < num_neighbors; j += blockDim.x * gridDim.y) {
            int neighbor = idx[batch_idx * num_spheres * num_neighbors + sphere_idx * num_neighbors + j];
            if (neighbor != sphere_idx && neighbor != ignore_idx) {
                atomicAnd((int*)&retain[batch_idx * num_spheres + neighbor], 0);
            }
        }
        __syncthreads();
    }
}

__global__ void shared_memory_greedy_reduction_kernel(
    const int* __restrict__ sorted_indices,
    const int* __restrict__ idx,
    const int* __restrict__ lengths,
    bool* __restrict__ retain,
    int num_batches,
    int num_spheres,
    int num_neighbors,
    int ignore_idx
) {
    extern __shared__ bool shared_retain[];
    
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch_idx >= num_batches) return;
    
    int valid_length = lengths[batch_idx];
    
    // init shared memory
    for (int i = tid; i < num_spheres; i += blockDim.x) {
        shared_retain[i] = (i < valid_length);
    }
    __syncthreads();
    
    // process spheres in sorted order
    for (int i = 0; i < num_spheres; i++) {
        int sphere_idx = sorted_indices[batch_idx * num_spheres + i];
        
        if (!shared_retain[sphere_idx]) continue;
        
        // process neighbors
        for (int j = tid; j < num_neighbors; j += blockDim.x) {
            int neighbor = idx[batch_idx * num_spheres * num_neighbors + sphere_idx * num_neighbors + j];
            if (neighbor != sphere_idx && neighbor != ignore_idx) {
                shared_retain[neighbor] = false;
            }
        }
        __syncthreads();
    }
    
    // write back to global memory
    for (int i = tid; i < num_spheres; i += blockDim.x) {
        retain[batch_idx * num_spheres + i] = shared_retain[i];
    }
}


void launch_greedy_reduction_cuda_kernel(
    const int* sorted_indices,
    const int* idx,
    const int* lengths,
    bool* retain,
    int num_batches,
    int num_spheres,
    int num_neighbors,
    int ignore_idx
) {
    cudaDeviceProp deviceProp;
    cudaError_t err = cudaGetDeviceProperties(&deviceProp, 0);
    if (err != cudaSuccess) {
        printf("Failed to get device properties: %s\n", cudaGetErrorString(err));
        return;
    }
    
    int required_shared_mem = num_spheres * sizeof(bool);
    
    if (required_shared_mem <= deviceProp.sharedMemPerBlock) {
        int threads_shared = 256;
        int blocks_shared = num_batches;
        int shared_mem_size = required_shared_mem;
        
        shared_memory_greedy_reduction_kernel<<<blocks_shared, threads_shared, shared_mem_size>>>(
            sorted_indices, idx, lengths, retain,
            num_batches, num_spheres, num_neighbors, ignore_idx);
    } else {
        dim3 blocks(num_batches, min(32, num_spheres));  // up to 32 blocks in y dimension.
        int threads = 256;
        
        greedy_reduction_kernel<<<blocks, threads>>>(
            sorted_indices, idx, lengths, retain,
            num_batches, num_spheres, num_neighbors, ignore_idx);
    }
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Kernel Failed: %s\n", cudaGetErrorString(err));
    }
}
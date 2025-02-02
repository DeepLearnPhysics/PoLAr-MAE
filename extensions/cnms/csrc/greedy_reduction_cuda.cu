// greedy_reduction_cuda.cu
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA Kernel Definition
__global__ void greedy_reduction_cuda_kernel(
    const int* __restrict__ sorted_indices,    // Shape: (N, P)
    const int* __restrict__ idx,              // Shape: (N, P, K)
    const int* __restrict__ lengths,          // Shape: (N,)
    bool* __restrict__ retain,                // Shape: (N, P)
    int num_batches,
    int num_spheres,
    int num_neighbors,
    int ignore_idx
) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= num_batches) return;

    // Initialize retain array for this batch -- ones for valid length, zeros for invalid length
    int valid_length = lengths[batch_idx];
    for (int i = 0; i < num_spheres; ++i) {
        if (i >= valid_length) {
            retain[batch_idx * num_spheres + i] = false;
        } else {
            retain[batch_idx * num_spheres + i] = true;
        }
    }

    // Perform greedy reduction for this batch
    for (int i = 0; i < num_spheres; ++i) {
        int sphere_idx = sorted_indices[batch_idx * num_spheres + i];
        if (!retain[batch_idx * num_spheres + sphere_idx]) {
            continue; // Already removed
        }

        // Iterate through neighbors
        for (int j = 0; j < num_neighbors; ++j) {
            int neighbor = idx[batch_idx * num_spheres * num_neighbors + sphere_idx * num_neighbors + j];
            if (neighbor == sphere_idx || neighbor == ignore_idx) {
                continue; // Exclude self
            }
            retain[batch_idx * num_spheres + neighbor] = false;
        }
    }
}

// Host function to launch the CUDA kernel
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
    // Define CUDA grid and block dimensions
    int threads = 256;
    int blocks = (num_batches + threads - 1) / threads;

    // Launch the CUDA kernel
    greedy_reduction_cuda_kernel<<<blocks, threads>>>(
        sorted_indices,
        idx,
        lengths,
        retain,
        num_batches,
        num_spheres,
        num_neighbors,
        ignore_idx
    );

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Kernel Failed: %s\n", cudaGetErrorString(err));
    }
}

// greedy_reduction_cpu.cpp
#include <torch/extension.h>
#include <vector>

// CPU Greedy Reduction Function
void greedy_reduction_cpu_kernel(
    const int* sorted_indices,    // Shape: (N, P)
    const int* idx,              // Shape: (N, P, K)
    const int* lengths,          // Shape: (N,)
    bool* retain,                // Shape: (N, P)
    int num_batches,
    int num_spheres,
    int num_neighbors,
    int ignore_idx
) {
    for (int batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
        // Initialize retain array for this batch
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
}

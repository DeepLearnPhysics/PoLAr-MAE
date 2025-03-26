// greedy_reduction_cpu.cpp
#include <torch/extension.h>
#include <vector>
#include <omp.h>

void greedy_reduction_cpu_kernel(
    const int *sorted_indices,
    const int *idx,
    const int *lengths,
    bool *retain,
    int num_batches,
    int num_spheres,
    int num_neighbors,
    int ignore_idx)
{
    #pragma omp parallel for
        for (int batch_idx = 0; batch_idx < num_batches; ++batch_idx)
        {
        int valid_length = lengths[batch_idx];

        #pragma omp simd
        for (int i = 0; i < num_spheres; ++i)
        {
            retain[batch_idx * num_spheres + i] = (i < valid_length);
        }

        // process spheres in sorted order
        for (int i = 0; i < valid_length; ++i)
        {
            int sphere_idx = sorted_indices[batch_idx * num_spheres + i];

            if (!retain[batch_idx * num_spheres + sphere_idx])
            {
                continue; // already removed
            }

            int base_offset = batch_idx * num_spheres * num_neighbors + sphere_idx * num_neighbors;

            #pragma omp simd
            for (int j = 0; j < num_neighbors; ++j)
            {
                int neighbor = idx[base_offset + j];
                if (neighbor != sphere_idx && neighbor != ignore_idx && neighbor < valid_length)
                {
                    retain[batch_idx * num_spheres + neighbor] = false;
                }
            }
        }
    }
}
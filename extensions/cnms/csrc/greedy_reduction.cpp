// greedy_reduction.cpp
#include <torch/extension.h>
#include <vector>

void greedy_reduction_cpu_kernel(
    const int* sorted_indices,
    const int* idx,
    const int* lengths,
    bool* retain,
    int num_batches,
    int num_spheres,
    int num_neighbors,
    int ignore_idx
);

void launch_greedy_reduction_cuda_kernel(
    const int* sorted_indices,
    const int* idx,
    const int* lengths,
    bool* retain,
    int num_batches,
    int num_spheres,
    int num_neighbors,
    int ignore_idx
);

torch::Tensor greedy_reduction(
    torch::Tensor sorted_indices,
    torch::Tensor idx,
    torch::Tensor lengths,
    int ignore_idx
) {
    TORCH_CHECK(sorted_indices.dim() == 2, "sorted_indices must be a 2D tensor");
    TORCH_CHECK(idx.dim() == 3, "idx must be a 3D tensor");

    TORCH_CHECK(sorted_indices.size(0) == idx.size(0),
                "Batch size of sorted_indices and idx must match");
    TORCH_CHECK(sorted_indices.size(1) == idx.size(1),
                "Number of spheres in sorted_indices and idx must match");
    TORCH_CHECK(sorted_indices.device() == idx.device(),
                "sorted_indices and idx must be on the same device");

    bool is_cuda = sorted_indices.is_cuda();

    // get dimensions
    int num_batches = sorted_indices.size(0);
    int num_spheres = sorted_indices.size(1);
    int num_neighbors = idx.size(2);

    // init retain tensor
    auto retain = torch::ones({num_batches, num_spheres}, torch::dtype(torch::kBool).device(sorted_indices.device()));

    // ensure tensors are contiguous and of type int32 or int64
    TORCH_CHECK(sorted_indices.dtype() == torch::kInt32 || sorted_indices.dtype() == torch::kInt64,
                "sorted_indices must be of type int32 or int64");
    TORCH_CHECK(idx.dtype() == torch::kInt32 || idx.dtype() == torch::kInt64,
                "idx must be of type int32 or int64");
    TORCH_CHECK(lengths.dtype() == torch::kInt32 || lengths.dtype() == torch::kInt64,
                "lengths must be of type int32 or int64");


    // to int32 if they are int64
    if (sorted_indices.dtype() == torch::kInt64) {
        sorted_indices = sorted_indices.to(torch::kInt32).contiguous();
    }
    if (idx.dtype() == torch::kInt64) {
        idx = idx.to(torch::kInt32).contiguous();
    }
    if (lengths.dtype() == torch::kInt64) {
        lengths = lengths.to(torch::kInt32).contiguous();
    }

    const int* sorted_indices_ptr = sorted_indices.data_ptr<int>();
    const int* idx_ptr = idx.data_ptr<int>();
    const int* lengths_ptr = lengths.data_ptr<int>();
    bool* retain_ptr = retain.data_ptr<bool>();

    if (is_cuda) {
        launch_greedy_reduction_cuda_kernel(
            sorted_indices_ptr,
            idx_ptr,
            lengths_ptr,
            retain_ptr,
            num_batches,
            num_spheres,
            num_neighbors,
            ignore_idx
        );
    } else {
        greedy_reduction_cpu_kernel(
            sorted_indices_ptr,
            idx_ptr,
            lengths_ptr,
            retain_ptr,
            num_batches,
            num_spheres,
            num_neighbors,
            ignore_idx
        );
    }

    return retain;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("greedy_reduction", &greedy_reduction, "Greedy Reduction (CPU and CUDA)");
}

// greedy_reduction.cpp
#include <torch/extension.h>
#include <vector>

// Forward declarations of CPU and CUDA functions
void greedy_reduction_cpu_kernel(
    const int* sorted_indices,
    const int* idx,
    const int* lengths, // New
    bool* retain,
    int num_batches,
    int num_spheres,
    int num_neighbors,
    int ignore_idx
);

void launch_greedy_reduction_cuda_kernel(
    const int* sorted_indices,
    const int* idx,
    const int* lengths, // New
    bool* retain,
    int num_batches,
    int num_spheres,
    int num_neighbors,
    int ignore_idx
);

// C++ Interface Function
torch::Tensor greedy_reduction(
    torch::Tensor sorted_indices,
    torch::Tensor idx,
    torch::Tensor lengths,
    int ignore_idx,
    torch::Tensor out
) {
    // Check that input tensors are 2D and 3D respectively
    TORCH_CHECK(sorted_indices.dim() == 2, "sorted_indices must be a 2D tensor");
    TORCH_CHECK(idx.dim() == 3, "idx must be a 3D tensor");

    // Ensure that the batch sizes and sphere counts match
    TORCH_CHECK(sorted_indices.size(0) == idx.size(0),
                "Batch size of sorted_indices and idx must match");
    TORCH_CHECK(sorted_indices.size(1) == idx.size(1),
                "Number of spheres in sorted_indices and idx must match");

    // Ensure inputs are on the same device
    TORCH_CHECK(sorted_indices.device() == idx.device(),
                "sorted_indices and idx must be on the same device");

    // Determine device type
    bool is_cuda = sorted_indices.is_cuda();

    // Get dimensions
    int num_batches = sorted_indices.size(0);
    int num_spheres = sorted_indices.size(1);
    int num_neighbors = idx.size(2);

    // Initialize retain tensor
    // auto retain = torch::ones({num_batches, num_spheres}, torch::dtype(torch::kBool).device(sorted_indices.device()));
    // ensure retain tensor is on device
    TORCH_CHECK(out.device() == sorted_indices.device(), "out must be on the same device as sorted_indices");
    // Ensure batch size and sphere count match
    TORCH_CHECK(out.size(0) == num_batches, "Batch size of out must match num_batches");
    TORCH_CHECK(out.size(1) == num_spheres, "Sphere count of out must match num_spheres");

    // Ensure tensors are contiguous and of type int32 or int64
    TORCH_CHECK(sorted_indices.dtype() == torch::kInt32 || sorted_indices.dtype() == torch::kInt64,
                "sorted_indices must be of type int32 or int64");
    TORCH_CHECK(idx.dtype() == torch::kInt32 || idx.dtype() == torch::kInt64,
                "idx must be of type int32 or int64");
    TORCH_CHECK(lengths.dtype() == torch::kInt32 || lengths.dtype() == torch::kInt64,
                "lengths must be of type int32 or int64");

    // Convert tensors to int32 if they are int64 for better performance
    if (sorted_indices.dtype() == torch::kInt64) {
        sorted_indices = sorted_indices.to(torch::kInt32);
    }
    if (idx.dtype() == torch::kInt64) {
        idx = idx.to(torch::kInt32);
    }
    if (lengths.dtype() == torch::kInt64) {
        lengths = lengths.to(torch::kInt32);
    }

    // Get raw pointers
    const int* sorted_indices_ptr = sorted_indices.data_ptr<int>();
    const int* idx_ptr = idx.data_ptr<int>();
    const int* lengths_ptr = lengths.data_ptr<int>();
    bool* retain_ptr = out.data_ptr<bool>();

    if (is_cuda) {
        // Launch CUDA kernel
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
        // Launch CPU kernel
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

    return out;
}

// Binding the C++ Interface Function to Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("greedy_reduction", &greedy_reduction, "Greedy Reduction (CPU and CUDA)");
}

# What is C-NMS?

C-NMS (Centrality-Based Non-Maximum Suppression) identifies overlapping spheres around each centroid in a batch of point clouds, then greedily retains only those centroids that are geometrically or statistically most central.

In practice, you might use this for e.g.

- De-duplicating point proposals in 3D object detection.
- Intelligent downsampling or “patchifying” point clouds without incurring voxel aliasing artifacts. (<-- us!)
- Reducing noise by only retaining the "most representative" sphere center within a neighborhood.

# Installation

```bash
cd extensions/cnms
pip install -e .
```

> [!IMPORTANT]
>
> - A functioning CUDA environment is required for GPU acceleration. If you don't have a GPU, the CPU implementation will be used.
>
> - `pytorch3d` must be installed separately. See the [main README](../README.md) for more details.


# Usage

```python
import torch
from cnms import cnms

# centroids: shape (N, P, 3)
# N = batch size, P = number of centroids, each with xyz coordinates
centroids = torch.randn((2, 50, 3), device="cuda")

# Radius for the spheres and overlap factor
radius = 1.0
overlap_factor = 0.7

# Optionally define K (estimated max overlapping neighbors) and lengths (points per batch).
# If omitted, defaults to K = P and lengths = P for all.

culled_centroids, culled_lengths = cnms(
    centroids, 
    radius=radius, 
    overlap_factor=overlap_factor, 
    K=None, 
    lengths=None
)

# culled_centroids: shape (N, P_culled, 3)
# culled_lengths: shape (N,) => number of culled points per batch
```

Parameters:

- `centroids`: (N, P, 3) float tensor of 3D coordinates.
- `radius`: Float that specifies the base sphere radius.
- `overlap_factor`: Fraction indicating allowed overlap of spheres (e.g., 0.7 = 70% diameter overlap).
- `K`: (Optional) integer bounding the neighbor search. Defaults to all points (P).
- `lengths`: (Optional) an (N,)-shaped integer tensor denoting the valid centroids in each batch. Defaults to P for each batch.


> [!NOTE]
> This has not been tested with gradients, as our initial use case is preprocessing for later training. I think it should work, but if you try it and it doesn't, let me know.
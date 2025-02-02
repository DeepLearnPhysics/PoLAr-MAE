# PILArNet-Medium

We provide the 156 GB **PILArNet-M** dataset, an continuation of [PILArNet](https://arxiv.org/abs/2006.01993), consisting of >1M [LArTPC](https://www.symmetrymagazine.org/article/october-2012/time-projection-chambers-a-milestone-in-particle-detector-technology?language_content_entity=und) events. Download the dataset from this [link](https://drive.google.com/drive/folders/1nec9WYPRqMn-_3m6TdM12TmpoInHDosb?usp=drive_link) or via the following command:

```bash
gdown --folder 1nec9WYPRqMn-_3m6TdM12TmpoInHDosb -O /path/to/save/dataset
```

> [!NOTE] 
> `gdown` must be installed (e.g., `pip install gdown` or `conda install gdown`). If you encounter issues with large file downloads from Google Drive, check for any quota limitations or network issues. You may also need to verify that your gdown version supports folder downloads.

## Indexing the Dataset

Before loading the dataset, it's necessary to index the point cloud sizes. This step ensures that only events with a sufficient number of points are used—effectively filtering out sparse events that might otherwise introduce noise or errors into downstream analysis. Run the following command to create an index:

```bash
python -m polarmae.datasets.build_index /path/to/dataset/**/*.h5 -j N
```

- The `-j N` argument is optional and allows for parallel processing, speeding up the indexing process.
- This command will create a corresponding `*_points.npy` file for each `.h5` file. These `.npy` files contain the number of points in each event and are used by the dataloader to decide which events to include.

## Directory Structure

The dataset is stored in HDF5 format and organized as follows:

```plaintext
/path/to/dataset/
    /train/
        /generic_v2_196200_v1.h5
        /generic_v2_153600_v1.h5
        ...
    /val/
        /generic_v2_10880_v1.h5
        ...
```

Here, the number preceding `v1` indicates the number of events contained in the file. The dataset is split into train and validation sets with:
- 1,199,200 events in the train set
- 10,880 events in the validation set

## Dataset Attributes

Each HDF5 file contains two main attributes:

- **`point`:**  
  Each entry corresponds to the number of spacepoints in a single event, containing:
  - 3D point coordinates
  - Energy deposit
  - Absolute time
  - Number of electrons
  - dx
  
  The raw data is stored as a flattened 1D array, which should be reshaped to `(N, 8)` for an event with `N` points. For example:
  
  """python
  import numpy as np
  
  # Assuming `data` is the flattened array loaded from an event
  N = len(data) // 8
  reshaped_points = data.reshape((N, 8))
  """
  
- **`cluster`:**  
  Each entry corresponds to a cluster of spacepoints, containing:
  - Number of points in the cluster
  - Fragment ID
  - Group ID
  - Interaction ID
  - Semantic type
  
  Similarly, reshape the flattened array to `(N, 5)` for an event with `N` clusters:
  
  ```python
  # Assuming `cluster_data` is the flattened array loaded from an event
  N = len(cluster_data) // 5
  reshaped_clusters = cluster_data.reshape((N, 5))
  ```
  
*Note:* Points in the `point` array are ordered by the cluster they belong to, enabling an association with the corresponding attributes in `cluster`.

A [Colab notebook](https://colab.research.google.com/drive/1x8WatdJa5D7Fxd3sLX5XSJiMkT_sG_im) is provided for a hands-on introduction to loading and inspecting the dataset.

## Usage in PoLAr-MAE

The dataset and its corresponding dataloader are used in this repository. The data path is specified in the config file as follows:

```yaml
data:
  class_path: polarmae.datasets.PILArNetDataModule
  init_args:
    data_path: /path/to/dataset/train/*.h5
    batch_size: 32
    num_workers: 4
    dataset_kwargs:
      energy_threshold: 0.13        # Minimum energy for a point to be included.
      remove_low_energy_scatters: true  # Filter out points with low energy deposits (semantic ID 4).
      emin: 1.0e-2                  # Lower bound for energy
      emax: 20.0                    # Upper bound for energy
      maxlen: 10000                 # Maximum number of events to load.
      min_points: 1024              # Minimum number of points per event.
      return_semantic_id: false     # Set to true if semantic segmentation labels are needed.
      return_cluster_id: false      # Set to true if cluster identification is required.
```

Keyword arguments:

- **`energy_threshold`:** Helps exclude low-interest points by ensuring only points with sufficient energy deposits are processed.
- **`remove_low_energy_scatters`:** Low energy scatters appear as scattered points in each image that seemingly have no relation to other particle trajectories, and are thus often removed.
- **`emin` and `emax`:** Define the energy range for log-transformation, aiding in numerical stability and performance.
- **`maxlen`:** Allows quick iterations by limiting the dataset size during testing or debugging.
- **`min_points`:** Ensures that only events with enough data points are used, which is critical for reliable analysis.
- **`return_semantic_id` & `return_cluster_id`:** Toggle additional labels depending on the downstream task (e.g., segmentation vs. clustering).

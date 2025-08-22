
<div align="center">
  <h2>Particle Trajectory Representation Learning with Masked Point Modeling<br>(PoLAr-MAE)</h2>
</div>

<div align="center">
<a href="https://arxiv.org/abs/2502.02558">[Paper]</a>
<a href="./DATASET.md">[Dataset]</a>
<a href="https://youngsm.com/polarmae">[Project Site]</a>
<a href="./tutorial">[Tutorial]</a>
<a href="#citing-polar-mae">[BibTeX]</a>
</div>

\
![arch](images/arch.png)

## Installation

This codebase relies on a number of dependencies, some of which are difficult to get running. If you're using conda on Linux, use the following to create an environment and install the dependencies:

```bash
git clone https://github.com/DeepLearnPhysics/PoLAr-MAE.git
cd PoLAr-MAE
bash ./install.sh
```

> [!NOTE]
>This script will create a conda environment using `environment.yml` before installing `pytorch3d`, `cnms`, and `polarmae`. environment.yml is a full environment specification, which includes an install of cuda 12.4, pytorch 2.5.1, python 3.10.
> 
> `pytorch3d` and `cnms` are compiled from source, and will only be compiled for the CUDA device architecture of the visible GPU(s) available on the system.

Set the number of cores to use via `N_JOBS=# bash ./install.sh`. At least 4 cores is recommended to compile `pytorch3d` in a reasonable amount of time. 

If you'd like to do the installation on your own, you will need the following dependencies:

- [CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#)
- PyTorch
- gcc & g++ >= 4.9 and < 13
- [pytorch3d](https://github.com/facebookresearch/pytorch3d)
- [PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning)
- [NumPy](https://github.com/numpy/numpy)
- [Lightning-Utilities](https://github.com/Lightning-AI/utilities)
- [Scikit-Learn](https://github.com/scikit-learn/scikit-learn)
- [Omegaconf](https://github.com/omry/omegaconf)
- [h5py](https://github.com/h5py/h5py)

`pytorch3d` is usually the most difficult dependency to install. See the `pytorch3d` [INSTALL.md](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) for more details.

There are a couple of extra dependencies that are optional, but recommended:

```bash
conda install wandb jupyter matplotlib
```

## Tutorial Notebooks

Tutorial notebooks for understanding the dataset, model architecture, pretraining, and finetuning are available in the [`tutorial`](tutorial) directory.


## PILArNet-M Dataset

We use and provide the 156 GB **PILArNet-M** dataset of ~1.2M [LArTPC](https://www.symmetrymagazine.org/article/october-2012/time-projection-chambers-a-milestone-in-particle-detector-technology?language_content_entity=und) events. See [DATASET.md](DATASET.md) for more details, but the dataset is available at this [link](https://drive.google.com/drive/folders/1nec9WYPRqMn-_3m6TdM12TmpoInHDosb?usp=drive_link), or can be downloaded with the following command:

```bash
gdown --folder 1nec9WYPRqMn-_3m6TdM12TmpoInHDosb -O /path/to/save/dataset
```

> [!NOTE]
> `gdown` must be installed via e.g. `pip install gdown` or `conda install gdown`.


## Models


### Pretraining
| Model | Num. Events |  Config | SVM $F_1$ | Download |
|-------|-------------|---------|-------|----------|
| Point-MAE | 1M | [pointmae.yml](https://github.com/DeepLearnPhysics/PoLAr-MAE/blob/main/configs/pointmae.yml) | 0.719 | [here](https://github.com/DeepLearnPhysics/PoLAr-MAE/releases/download/weights/mae_pretrain.ckpt) |
| PoLAr-MAE | 1M | [polarmae.yml](https://github.com/DeepLearnPhysics/PoLAr-MAE/blob/main/configs/polarmae.yml) | 0.732 | [here](https://github.com/DeepLearnPhysics/PoLAr-MAE/releases/download/weights/polarmae_pretrain.ckpt) |

Our evaluation consists of training an ensemble of linear SVMs to classify individual tokens (i.e., groups) as containing one or more classes. This is done via a One vs Rest strategy, where each SVM is trained to classify a single class against all others. $F_1$ is the mean $F_1$ score over all semantic categories in the validation set of the PILArNet-M dataset.

After installing the dependencies, you can run the following commands in an interactive Python session to load the pretrained model(s):

```python
>>> from polarmae.models.ssl import PointMAE, PoLArMAE
>>> !wget https://github.com/DeepLearnPhysics/PoLAr-MAE/releases/download/weights/mae_pretrain.ckpt
>>> !wget https://github.com/DeepLearnPhysics/PoLAr-MAE/releases/download/weights/polarmae_pretrain.ckpt
>>> model = PointMAE.load_from_checkpoint("mae_pretrain.ckpt") # or
>>> model = PoLArMAE.load_from_checkpoint("polarmae_pretrain.ckpt")
```

### Semantic Segmentation

| Model | Training Method | Num. Events |  Config | $F_1$ | Download |
|-------|-----------------|-------------|---------|-------|----------|
| Point-MAE | Linear probing | 10k | [part_segmentation_mae_peft.yml](https://github.com/DeepLearnPhysics/PoLAr-MAE/blob/main/configs/part_segmentation_mae_peft.yml) | 0.772 | [here](https://github.com/DeepLearnPhysics/PoLAr-MAE/releases/download/weights/mae_peft_segsem_10k.ckpt) |
| PoLAr-MAE | Linear probing | 10k | [part_segmentation_polarmae_peft.yml](https://github.com/DeepLearnPhysics/PoLAr-MAE/blob/main/configs/part_segmentation_polarmae_peft.yml) | 0.798 | [here](https://github.com/DeepLearnPhysics/PoLAr-MAE/releases/download/weights/polarmae_peft_segsem_10k.ckpt) |
| Point-MAE | FFT | 10k | [part_segmentation_mae_fft.yml](https://github.com/DeepLearnPhysics/PoLAr-MAE/blob/main/configs/part_segmentation_mae_fft.yml) | 0.831 | [here](https://github.com/DeepLearnPhysics/PoLAr-MAE/releases/download/weights/mae_fft_segsem_10k.ckpt) |
| PoLAr-MAE | FFT | 10k | [part_segmentation_polarmae_fft.yml](https://github.com/DeepLearnPhysics/PoLAr-MAE/blob/main/configs/part_segmentation_polarmae_fft.yml) | 0.837 | [here](https://github.com/DeepLearnPhysics/PoLAr-MAE/releases/download/weights/polarmae_fft_segsem_10k.ckpt) |

Our evaluation for semantic segmentation consists of 1:1 comparisons between the predicted and ground truth segmentations. $F_1$ is the mean $F_1$ score over all semantic categories in the validation set of the PILArNet-M dataset.

After installing the dependencies, you can run the following commands in an interactive Python session to load the pretrained model(s):

```python
>>> from polarmae.models.finetune import SemanticSegmentation
>>> from polarmae.utils.checkpoint import load_finetune_checkpoint
>>> !wget https://github.com/DeepLearnPhysics/PoLAr-MAE/releases/download/weights/{mae,polarmae}_{fft,peft}_segsem.ckpt
>>> model = load_finetune_checkpoint("{mae,polarmae}_{fft,peft}_segsem.ckpt")
```

Here, the brackets {} denote the model and the training method -- choose one from `{mae,polarmae}` and `{fft,peft}`. Note that you must use the `load_finetune_checkpoint` function to load the model, not `SemanticSegmentation.load_from_checkpoint()`.

## Training

<details>
  <summary>Important: Learning rate instructions</summary>

  The following commands use the configurations we used for our experiments. Particularly, our learning rates are set assuming a batch size of 128 (i.e., 32 across 4 GPUs). If you want to train on a single GPU with the same batch size in the configuration file, you will need to scale the learning rate accordingly. We recommend scaling the learning rate by the square root of the ratio of the batch sizes. I.e., if your batch size is $b$, you should set the learning rate $l \rightarrow l \times \sqrt{b/128}$.
</details>


### Pretraining

To pretrain Point-MAE, modify the [config file](https://github.com/DeepLearnPhysics/PoLAr-MAE/blob/main/configs/pointmae.yml) to include the path to the PILArNet-M dataset, and run the following command:

```bash
python -m polarmae.tasks.pointmae fit --config configs/pointmae.yml
```

To pretrain PoLAr-MAE, modify the [config file](https://github.com/DeepLearnPhysics/PoLAr-MAE/blob/main/configs/polarmae.yml) to include the path to the PILArNet-M dataset, and run the following command:

```bash
python -m polarmae.tasks.polarmae fit --config configs/polarmae.yml
```

<details>
  <summary>Example training plots</summary>

  <img src="images/pretrain_loss.png" alt="training plots" width="300">
  <img src="images/svm_f1_scores.png" alt="svm plots" width="300">
</details>

### Semantic Segmentation

To train a semantic segmentation model, modify the [config file](https://github.com/DeepLearnPhysics/PoLAr-MAE/blob/main/configs/part_segmentation_mae_peft.yml) to include the path to the PILArNet-M dataset, and run the following command:

```bash
python -m polarmae.tasks.part_segmentation fit --config configs/part_segmentation_{mae,polarmae}_{peft,fft}.yml \
                        --model.pretrained_ckpt_path path/to/pretrained/checkpoint.ckpt
```

where `{mae,polarmae}` is either `mae` or `polarmae`, and `{peft,fft}` is either `peft` or `fft`. You can either specify the pretrained checkpoint path in the config, or pass it as an argument to the command like above.

<details>
  <summary>Example training plots</summary>

  <img src="images/ft_segsem_loss.png" alt="training plots" width="300">
  <img src="images/ft_segsem_accprec.png" alt="svm plots" width="300">
</details>


## Acknowledgements

This repository is built upon the lovely [Point-MAE](https://github.com/Pang-Yatian/Point-MAE) and [point2vec](https://github.com/kabouzeid/point2vec) repositories.

## Citing PoLAr-MAE

If you find this work useful, please consider citing the following paper:

```bibtex
@misc{young2025particletrajectoryrepresentationlearning,
      title={Particle Trajectory Representation Learning with Masked Point Modeling}, 
      author={Sam Young and Yeon-jae Jwa and Kazuhiro Terao},
      year={2025},
      eprint={2502.02558},
      archivePrefix={arXiv},
      primaryClass={hep-ex},
      url={https://arxiv.org/abs/2502.02558}, 
}
```

## Contact

Any questions? Any suggestions? Want to collaborate? Feel free to raise an issue on Github or email Sam Young [youngsam@stanford.edu](mailto:youngsam@stanford.edu).

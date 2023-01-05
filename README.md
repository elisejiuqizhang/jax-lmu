# jax-lmu
Unofficial implementation of Legendre Memory Unit (LMU) using Jax (out of interest and it's a work in progress!😅)

Original paper: [Legendre Memory Units: Continuous-Time Representation in Recurrent Neural Networks](https://papers.nips.cc/paper/2019/hash/952285b9b7e7a1be5aa7849f32ffff05-Abstract.html); Repo of the orginal paper used [Keras](https://github.com/nengo/keras-lmu);

Another paper is [parallelizing LMU training](https://arxiv.org/abs/2102.11417) and its [LMU PyTorch implementation](https://github.com/hrshtv/pytorch-lmu) (which is mostly what I am basing my adaptation on).

## Models

Original LMU and its training/evaluation are named `jax_lmu.py` and `jax_lmu_run_exp.py` (can also run from the `jax_lmu_mnist.ipynb` file); 

The parallelized version of LMU and its training/evaluation are named `jax_lmufft.py` and `jax_lmufft_run_exp.py`(can also run from the `jax_lmu_parallel_mnist.ipynb` file).

## Environment Setup

Single GPU: NVIDIA GeForce RTX 3090, CUDA version 11.5.

### Minimum Requirements

Python version is 3.9.15. Required packages installed using a mixture of conda and pip:

    conda install cudatoolkit=11.3
    conda install cudnn=8.2.1
    pip install "jax[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    pip install flax==0.6.2
    
### Experiments on MNIST

MNIST dataset will be loaded using TensorFlow. 

    pip install tqdm
    pip install tensorflow tensorflow_datasets
    

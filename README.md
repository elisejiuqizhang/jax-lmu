# jax-lmu
Legendre Memory Unit (LMU) implementation using Jax (work in progress!)

Original paper: [Legendre Memory Units: Continuous-Time Representation in Recurrent Neural Networks](https://papers.nips.cc/paper/2019/hash/952285b9b7e7a1be5aa7849f32ffff05-Abstract.html)

## Environment Setup

My GPU is NVIDIA GeForce RTX 3090 and CUDA version 11.5.

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
    

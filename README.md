# jax-lmu
Legendre Memory Unit (LMU) implementation using Jax (work in progress!)

## Environment Setup

### Minimum Requirements

Python version is 3.9.15, and I'm setting up my environment in Conda. 
After creating the corresponding virtual environment, I ran the following commands to install required packages:

    conda install cudatoolkit=11.3
    conda install cudnn=8.2.1
    pip install "jax[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    pip install flax==0.6.2
    
My GPU is NVIDIA GeForce RTX 3090 (can be queried in commandline with `nvidia-smi --query-gpu=name --format=csv,noheader)` and CUDA version 11.5 (can be queried with `nvidia-smi`). The Jax cuda version might need to be adjusted depending on your device.

### Experiments on MNIST

MNIST dataset will be loaded from PyTorch, since the jax library doesn't include any dataloading feature. I ran the following command to for the Jupyter Notebook experiments on MNIST.

    pip install torch torchvision
    pip install scikit-learn==1.1.3
    pip install tqdm==4.64.1
    pip install ipywidgets

# jax-lmu
Legendre Memory Unit (LMU) implementation using Jax 

## Environment
Python version is 3.9.15. After creating the corresponding virtual environment, I ran the following commands to set up the environment:

    conda install cudatoolkit=11.3
    conda install cudnn=8.2.1
    pip install "jax[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    pip install flax==0.6.2
    
My GPU is NVIDIA GeForce RTX 3090 (can be queried in commandline with `nvidia-smi --query-gpu=name --format=csv,noheader)` and CUDA version 11.5 (can be queried with `nvidia-smi`). The Jax cuda version might need to be adjusted depending on your device.

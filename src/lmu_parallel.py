# Unofficial Implementation of Parallel LMU in JAX
# Author: Elise Zhang
# Date: 2022-12-03

# LMU paper: https://proceedings.neurips.cc/paper/2019/hash/952285b9b7e7a1be5aa7849f32ffff05-Abstract.html
# Parallelizing LMU Training paper: https://proceedings.mlr.press/v139/chilkuri21a.html
# Original implementation using Keras: https://github.com/nengo/keras-lmu
# Implementation using PyTorch: https://github.com/hrshtv/pytorch-lmu (which is what I based my adaptation on)

import jax
import jax.numpy as jnp
from jax.numpy import fft
from scipy.signal import cont2discrete

from jax import random
from jax.nn import initializers

import flax.linen as nn

# ----- The LMUFFT Layer ---------------------------------------------------------
# Parallel Training of LMU Based on the 2021 ICML paper https://proceedings.mlr.press/v139/chilkuri21a.html 
# Specifically following Equation (26) in the paper which uses fast Fourier transform

class LMUFFT(nn.Module):
    """
    Parallelized LMU Layer

    Parameters:
        input_size (int) : 
            Size of the input vector (x_t)
        hidden_size (int) : 
            Size of the hidden vector (h_t)
        memory_size (int) :
            Size of the memory vector (m_t)
        seq_len (int) :
            Size of the sequence length (n)
        theta (int) :
            The number of timesteps in the sliding window that is represented using the LTI system
    """

    def __init__(self, input_size, hidden_size, memory_size, seq_len, theta):
        super(LMUFFT, self).__init__()

        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.seq_len = seq_len
        self.theta = theta

        self.W_u=nn.linear(input_size, 1) # in_features = input_size, out_features = 1
        self.f_u=nn.relu()
        self.W_h=nn.linear(memory_size + input_size, hidden_size) # in_features = memory_size + input_size, out_features = hidden_size
        self.f_h=nn.relu()

        # A: # [memory_size, memory_size]
        # B: # [memory_size, 1]        
        A, B = self.stateSpaceMatrices()

        # H: [memory_size, seq_len]
        # fft_H: [memory_size, seq_len + 1]
        H, fft_H = self.impulse()



    def stateSpaceMatrices(self):
        """ Returns the discretized state space matrices A and B """
        Q = jnp.arange(self.memory_size, dtype = jnp.float64).reshape(-1, 1)
        R = (2*Q + 1) / self.theta
        i, j = jnp.meshgrid(Q, Q, indexing = "ij")

        # Continuous
        A = R * jnp.where(i < j, -1, (-1.0)**(i - j + 1))
        B = R * ((-1.0)**Q)
        C = jnp.ones((1, self.memory_size))
        D = jnp.zeros((1,))

        # Convert to discrete
        A, B, C, D, dt = cont2discrete(
                                system = (A, B, C, D), 
                                dt = 1.0, 
                                method = "zoh"
                                )

        return A, B

    def impulse(self):
        """ Returns the matrices H and the 1D Fourier transform of H (Equations 23, 26 of the paper) """
            
        H = []
        A_i=jnp.eye(self.memory_size, dtype = jnp.float64)
        for i in range(self.seq_len):
            H.append(jnp.matmul(A_i, self.B))
            A_i = jnp.matmul(self.A, A_i)

        H=jnp.concatenate(H, axis = -1) # H: [memory_size, seq_len]
        fft_H=fft.rfft(H, n = 2*self.seq_len, dim = -1) # [memory_size, seq_len + 1]

        return H, fft_H

    def forward(self, x):
        """
        Parameters:
            x (array):
                Input of size [batch_size, seq_len, input_size]
        """

        batch_size, seq_len, input_size = x.shape

        # Equation 18 of the paper
        u=self.f_u(self.W_u(x)) # u: [batch_size, seq_len, 1]

        # Equation 26 of the paper
        fft_input=u.transpose(0, 2, 1) # fft_input: [batch_size, 1, seq_len]
        fft_u=fft.rfft(fft_input, n = 2*self.seq_len, dim = -1) # fft_u: [batch_size, 1, seq_len + 1]

        # Element-wise multiplication (under broadcasting)
        # [batch_size, 1, seq_len+1] * [1, memory_size, seq_len+1]
        temp=fft_u * self.fft_H.expand_dims(0) # temp: [batch_size, memory_size, seq_len+1]

        m = fft.irfft(temp, n = 2*seq_len, dim = -1) # m: [batch_size, memory_size, seq_len+1]
        m = m[:, :, :seq_len] # m: [batch_size, memory_size, seq_len]
        m = m.transpose(0, 2, 1) # m: [batch_size, seq_len, memory_size]

        # Equation 20 of the paper 
        input_h=jnp.concatenate([m,x], axis = -1) # input_h: [batch_size, seq_len, memory_size + input_size]])
        h=self.f_h(self.W_h(input_h)) # h: [batch_size, seq_len, hidden_size]

        h_n=h[:, -1, :] # h_n: [batch_size, hidden_size]
        
        return h, h_n
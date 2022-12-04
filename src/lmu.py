# Unofficial Implementation of Original LMU in JAX
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


# ------ One Cell of LMU ------------------------------------------------------

class LMUCell(nn.Module):
    """ 
    LMU Cell

    Parameters:
        input_size (int) : 
            Size of the input vector (x_t)
        hidden_size (int) : 
            Size of the hidden vector (h_t)
        memory_size (int) :
            Size of the memory vector (m_t)
        theta (int) :
            The number of timesteps in the sliding window that is represented using the LTI system

    """

    def __init__(self, input_size, hidden_size, memory_size, theta):
        super(LMUCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.memory_size = memory_size

        self.f=nn.tanh() # activation function

        # State space matrices: A, B
        A, B = self.stateSpaceMatrices(memory_size, theta)
        self.A = A
        self.B = B

        # The Model parameters:
        ## Encoding vectors
        self.e_x=jnp.empty(1,input_size)
        self.e_h=jnp.empty(1,hidden_size)
        self.e_m=jnp.empty(1,memory_size)
        ## Kernels
        self.W_x=jnp.empty(hidden_size,input_size)
        self.W_h=jnp.empty(hidden_size,hidden_size)
        self.W_m=jnp.empty(hidden_size,memory_size)
        ## Initialize parameters
        self.init_params()


    def stateSpaceMatrices(self, memory_size, theta):
        """ Returns the discretized state space matrices A and B """
        Q = jnp.arange(memory_size, dtype = jnp.float64).reshape(-1, 1)
        R = (2*Q + 1) / theta
        i, j = jnp.meshgrid(Q, Q, indexing = "ij")

        # Continuous
        A = R * jnp.where(i < j, -1, (-1.0)**(i - j + 1))
        B = R * ((-1.0)**Q)
        C = jnp.ones((1, memory_size))
        D = jnp.zeros((1,))

        # Convert to discrete
        A, B, C, D, dt = cont2discrete(
                                system = (A, B, C, D), 
                                dt = 1.0, 
                                method = "zoh"
                            )
            
        return A, B

    def init_params(self):
        """ Initialize model parameters """
        self.e_x = initializers.lecun_uniform()(self.e_x)
        self.e_h = initializers.lecun_uniform()(self.e_h)
        self.e_m = initializers.constant(0.0)(self.e_m)
        self.W_x = initializers.xavier_normal()(self.W_x)
        self.W_h = initializers.xavier_normal()(self.W_h)
        self.W_m = initializers.xavier_normal()(self.W_m)

    def forward(self, x, state):
        """
        Parameters:
            x (array) : 
                Input vector of size [batch_size, input_size]
            state (array) : 
                The hidden state:
                h: [batch_size, hidden_size]
                m: [batch_size, memory_size]
        """

        # Get the hidden state and memory
        h, m = state

        # Equation (7) of the paper
        ## u: [batch_size, 1]
        u=jnp.matmul(self.e_x,x)+jnp.matmul(self.e_h,h)+jnp.matmul(self.e_m,m)

        # Equation (4) of the paper
        ## m: [batch_size, memory_size]
        m=jnp.matmul(self.A,m)+jnp.matmul(self.B,u)

        # Equation (6) of the paper
        ## h: [batch_size, hidden_size]
        h=self.f(
                jnp.matmul(self.W_x,x)
                +jnp.matmul(self.W_h,h)
                +jnp.matmul(self.W_m,m)
            )

        return h, m


# ----- The LMU Layer ---------------------------------------------------------
# Original LMU Based on the 2019 NIPS paper https://proceedings.neurips.cc/paper/2019/hash/952285b9b7e7a1be5aa7849f32ffff05-Abstract.html


class LMU(nn.Module):
    """ 
    The LMU Layer
    
    Parameters:
        input_size (int) : 
            Size of the input vector (x_t)
        hidden_size (int) : 
            Size of the hidden vector (h_t)
        memory_size (int) :
            Size of the memory vector (m_t)
        theta (int) :
            The number of timesteps in the sliding window that is represented using the LTI system
    """

    def __init__(self, input_size, hidden_size, memory_size, theta):
        super(LMU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.theta = theta

        # LMU Cell
        self.cell = LMUCell(input_size, hidden_size, memory_size, theta)

    def forward(self, x, state=None):
        """
        Parameters:
            x (array):
                Input vector of size [batch_size, seq_len, input_size]
            state (array):
                h: [batch_size, hidden_size]
                m: [batch_size, memory_size]
        
        """

        # Assume the order of dimensions is [batch_size, seq_len, ......]
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        # Initial State (h_0, m_0)
        if state is None:
            h_0 = jnp.zeros((batch_size, self.hidden_size))
            m_0 = jnp.zeros((batch_size, self.memory_size))
            if x.is_cuda: # If the input is on the GPU, put to device
                h_0 = h_0.cuda()
                m_0 = m_0.cuda()
            state = (h_0, m_0)

        # Iterate over time steps
        output = []
        for t in range(seq_len):
            x_t=x[:,t,:] # x_t: [batch_size, input_size]
            h_t, m_t = self.cell(x_t, state) # h_t: [batch_size, hidden_size], m_t: [batch_size, memory_size]
            state=(h_t, m_t)
            output.append(h_t)

        output = jnp.stack(output) # output: [seq_len, batch_size, hidden_size]
        output = output.transpose(1, 0, 2) # output: [batch_size, seq_len, hidden_size]

        return output, state
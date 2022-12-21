from functools import partial
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import jax
from jax import lax, random, numpy as jnp
from jax import grad, jit, vmap, value_and_grad

from flax import linen as nn

from scipy.signal import cont2discrete


#---------------------- Parallelized LMU ----------------------#
class LMUFFT(nn.Module):
    """ Parallelized LMU Layer
        
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

    input_size: int
    hidden_size: int
    memory_size: int
    seq_len: int
    theta: int

    def setup(self):
        """
        A: [memory_size, memory_size]
        B: [memory_size, 1]
        """
        self.A, self.B = self.stateSpaceMatrices(self.memory_size, self.theta)
        self.H, self.H_fft = self.impulse()

    @nn.compact
    def __call__(self, x):
        """
        Parameters:
            x (torch.tensor): 
                Input of size [batch_size, seq_len, input_size]
        """
        batch_size, seq_len, input_size = x.shape

        # Equation 18 of the paper
        u_pre_act = nn.Dense(features=1,
                            use_bias=True)
        u = nn.relu(u_pre_act(x)) # [batch_size, seq_len, 1]

        # Equation 26 of the paper
        fft_input=u.transpose(0, 2, 1) # [batch_size, 1, seq_len]
        fft_u=jnp.fft.rfft(fft_input, n = 2*seq_len, axis = -1) # [batch_size, seq_len, seq_len+1]

        # Element-wise multiplication (uses broadcasting)
        # fft_u:[batch_size, 1, seq_len+1] 
        # self.H_fft: [memory_size, seq_len+1] -> to be expanded in dimension 0
        H_fft=self.H_fft.reshape(1, self.memory_size, self.seq_len+1) # [1, memory_size, seq_len+1]
        # [batch_size, 1, seq_len+1] * [1, memory_size, seq_len+1]
        temp=jnp.multiply(fft_u, H_fft) # [batch_size, memory_size, seq_len+1]

        m=jnp.fft.irfft(temp, n = 2*seq_len, axis = -1) # [batch_size, memory_size, seq_len+1]
        m=m[:, :, :seq_len] # [batch_size, memory_size, seq_len]
        m=m.transpose(0, 2, 1) # [batch_size, seq_len, memory_size]

        # Equation 20 of the paper
        input_h=jnp.concatenate((x, m), axis=-1) # [batch_size, seq_len, input_size + memory_size]
        h_pre_act = nn.Dense(features=self.hidden_size,
                            use_bias=True)
        h=nn.relu(h_pre_act(input_h)) # [batch_size, seq_len, hidden_size]

        h_n=h[:, -1, :] # [batch_size, hidden_size]

        return h, h_n



    def impulse(self):
        def impulse_body(n, carry:Tuple):
            """The body function to be used in the fori_loop of the impluse function above"""
            H, A_i=carry
            H_n=jnp.matmul(A_i, self.B) # [memory_size, 1]
            # H.append(H_n)
            H = H.at[:, n].set(H_n.reshape(-1))
            A_i=jnp.matmul(self.A, A_i)
            return (H, A_i)
        """ Returns the matrices H and the 1D Fourier transform of H (Equations 23, 26 of the paper) """
        H_init=jnp.empty((self.memory_size, self.seq_len)) # [memory_size, seq_len]
        A_i_init=jnp.eye(self.memory_size, dtype = np.float32) # [memory_size, memory_size]
        val_init = (H_init, A_i_init)
        H, A_i=lax.fori_loop(0, self.seq_len, impulse_body, val_init)        
        # H=np.concatenate(H_fin, axis=-1) # [memory_size, seq_len]
        H_fft=np.fft.rfft(H, n = 2*self.seq_len, axis = -1) # [memory_size, seq_len + 1]
        return H, H_fft

    # def impulse(self):
    #     """ Returns the matrices H and the 1D Fourier transform of H (Equations 23, 26 of the paper) """
    #     H = np.empty((self.memory_size, self.seq_len), dtype = np.float32) # [memory_size, seq_len]
    #     A_i = np.eye(self.memory_size, dtype = np.float32) # [memory_size, memory_size]
    #     for n in range(self.seq_len):
    #         H_n = np.matmul(A_i, self.B) # [memory_size, 1]
    #         H[:, n]=H_n.reshape(-1)
    #         A_i = np.matmul(self.A, A_i)
    #     H_fft = np.fft.rfft(H, n = 2*self.seq_len, axis = -1) # [memory_size, seq_len + 1]
    #     return H, H_fft

    def stateSpaceMatrices(self, memory_size, theta):
        """ Returns the discretized state space matrices A and B """
        Q = np.arange(memory_size, dtype = np.float32)
        R = (2*Q + 1) / theta
        i, j = np.meshgrid(Q, Q, indexing = "ij")

        # Continuous
        A = R * np.where(i < j, -1, (-1.0)**(i - j + 1))
        B = R * ((-1.0)**Q)
        B = B.reshape(-1, 1)
        C = np.ones((1, memory_size))
        D = np.zeros((1,))

        # Convert to discrete
        A, B, C, D, dt = cont2discrete(
                                system = (A, B, C, D), 
                                dt = 1.0, 
                                method = "zoh"
                            )
            
        return A, B

#---------------------- Parallelized LMU for pMNIST Classification ----------------------#
class Model(nn.Module):
    input_size: int
    output_size: int
    hidden_size: int
    memory_size: int
    theta: int
    seq_len:int

    @nn.compact
    def __call__(self, x):
        _, h_n = LMUFFT(input_size=self.input_size, hidden_size=self.hidden_size, memory_size=self.memory_size, seq_len=self.seq_len, theta=self.theta)(x)
        x = nn.relu(h_n)
        output = nn.Dense(features=self.output_size)(x)
        return output

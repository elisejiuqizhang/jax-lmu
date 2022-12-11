import numpy as np
import jax
from jax import lax, random, numpy as jnp
from jax import grad, jit, vmap, value_and_grad

from typing import Any, Callable, Sequence

from flax import linen as nn

from scipy.signal import cont2discrete

#---------------------- One LMU Cell ----------------------#
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

    input_size: int
    hidden_size: int
    memory_size: int
    theta: int

    init_lecun_uni: Callable = nn.initializers.lecun_normal()
    init_zero: Callable = nn.initializers.constant(0.)
    init_xav_norm: Callable = nn.initializers.xavier_normal()

    def setup(self):
        """ Initialize the parameters 

        Trainable parameters:
            Encoding vectors:
                e_x: [1, input_size]
                e_h: [1, hidden_size]
                e_m: [1, memory_size]
            Kernel matrices:
                W_x: [hidden_size, input_size]
                W_h: [hidden_size, hidden_size]
                W_m: [hidden_size, memory_size]

        A and B are fixed state space matrices:
            A: [memory_size, memory_size]?
            B: [memory_size, 1]?
        
        """
        self.e_x=self.param('e_x', self.init_lecun_uni, (1,self.input_size))
        self.e_h=self.param('e_h', self.init_lecun_uni, (1,self.hidden_size))
        self.e_m=self.param('e_m', self.init_zero, (1,self.memory_size))
        self.W_x=self.param('W_x', self.init_xav_norm, (self.hidden_size,self.input_size))
        self.W_h=self.param('W_h', self.init_xav_norm, (self.hidden_size,self.hidden_size))
        self.W_m=self.param('W_m', self.init_xav_norm, (self.hidden_size,self.memory_size))
        self.A, self.B = self.stateSpaceMatrices(self.memory_size, self.theta)
        

    @staticmethod
    def initialize_state(batch_size, hidden_size, memory_size):
        return jnp.zeros((batch_size, hidden_size)), jnp.zeros((batch_size, memory_size))

    @nn.compact
    def __call__(self, x, state):
        """
        Inputs:
            x (array) : 
                Input vector of size [batch_size, input_size]
            state (array) : 
                The hidden state:
                h: [batch_size, hidden_size]
                m: [batch_size, memory_size]

        """


        # Unpack the hidden state and memory
        h, m = state

        # Eq (7) of the paper
        # u: [batch_size, 1]
        u1=self.vmap_matmul(x,self.e_x.T)
        u2=self.vmap_matmul(h,self.e_h.T)
        u3=self.vmap_matmul(m,self.e_m.T)
        u=u1+u2+u3

        # Eq (4) of the paper
        # m: [batch_size, memory_size]
        m1=self.vmap_matmul(m,self.A.T)
        m2=self.vmap_matmul(u,self.B.T)
        m=m1+m2

        # Eq (6) of the paper
        # h: [batch_size, hidden_size]
        h1=self.vmap_matmul(x,self.W_x.T)
        h2=self.vmap_matmul(h,self.W_h.T)
        h3=self.vmap_matmul(m,self.W_m.T)
        h=nn.tanh(h1+h2+h3)

        return h, m

    def vmap_matmul(self, input, params):
        """Following the example from vmap documentation: 
        https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html

        Dimensions:
            input: [b,a] - [batch_size, input_size]
            params: [a,c] - [input_size, ... (anything) ]
            output: [b,c] - [batch_size, ... (anything) ]
        
        """
        vv=lambda x, y: jnp.dot(x, y)  #  ([a], [a]) -> []
        mv=vmap(vv, in_axes=(0, None), out_axes=0)  #  ([b,a], [a]) -> [b] 
        mm=vmap(mv, in_axes=(None, 1), out_axes=1)  #  ([b,a], [a,c]) -> [b,c]
        return mm(input, params)


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

    
#---------------------- LMU Layer ----------------------#
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
        pmnist (boolean) :
            Uses different parameter initializers when training on psMNIST (as specified in the paper)
    """

    input_size: int
    hidden_size: int
    memory_size: int
    theta: int
    pmnist: bool = False

    def setup(self):
        self.cell = LMUCell(self.input_size, self.hidden_size, self.memory_size, self.theta)

    @nn.compact
    def __call__(self, x, state=None):
        """
        Parameters:
            x (array):
                Input vector of size [batch_size, seq_len, input_size]
            state (array): set to None by default
                h: [batch_size, hidden_size]
                m: [batch_size, memory_size]
        
        """
        # Assume the order of dimensions is [batch_size, seq_len, ......]
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        # Initial State (h_0, m_0)
        if state is None:
            state=self.cell.initialize_state(batch_size, self.hidden_size, self.memory_size)
            # h_0 = jnp.zeros((batch_size, self.hidden_size))
            # m_0 = jnp.zeros((batch_size, self.memory_size))
            # state = (h_0, m_0)     

        # Operation over time steps
        output = []
        for t in range(seq_len):
            x_t=x[:,t,:] # x_t: [batch_size, input_size]
            h_t, m_t = self.cell(x_t, state) # h_t: [batch_size, hidden_size], m_t: [batch_size, memory_size]
            state=(h_t, m_t)
            output.append(h_t)

        output = jnp.stack(output) # output: [seq_len, batch_size, hidden_size]
        output = output.transpose(1, 0, 2) # output: [batch_size, seq_len, hidden_size]

        return output, state

#---------------------- LMU Model ----------------------#

class Model(nn.Module):
    input_size: int
    output_size: int
    hidden_size: int
    memory_size: int
    theta: int

    def setup(self):
        self.lmu = LMU(self.input_size, self.hidden_size, self.memory_size, self.theta)
        self.classifier = nn.Dense(self.output_size)

    @nn.compact
    def __call__(self, x):
        _, (h_n, _) = self.lmu(x) # [batch_size, hidden_size]
        output = self.classifier(h_n)
        return output # [batch_size, output_size]

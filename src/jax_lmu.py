from functools import partial
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import jax
from jax import lax, random, numpy as jnp
from jax import grad, jit, vmap, value_and_grad

from flax import linen as nn

from scipy.signal import cont2discrete

#---------------------- Cell Base ----------------------#
# The other classes inherit from this class

class RecurrentCellBase(nn.Module):
  """Recurrent cell base class."""

  @staticmethod
  def initialize_carry(batch_dims, size, init_fn=nn.initializers.zeros):
    """Initialize the RNN cell carry.

    Args:
      batch_dims: a tuple providing the shape of the batch dimensions.
      size: the size or number of features of the memory.
      init_fn: initializer function for the carry.
    Returns:
      An initialized carry for the given cell.
    """
    raise NotImplementedError

#---------------------- One LMU Cell ----------------------#
class LMUCell(RecurrentCellBase):
    """LMU Cell
    
    Parameters:
        input_size (int) : 
            Size of the input vector (x_t)
        hidden_size (int) : 
            Size of the hidden vector (h_t)
        memory_size (int) :
            Size of the memory vector (m_t)
        theta (int) :
            The number of timesteps in the sliding window that is represented using the LTI system  
    
    Other Attributes:
        State Space Matrices:
            A: [memory_size, memory_size]
            B: [memory_size, 1]
        Initializers:
            init_lecun_uni: lecun_normal, for two encoding vectors, e_x and e_h
            init_zero: constant(0), for the encoding vector e_m
            init_xav_norm: xavier_normal, for three kernel matrices, W_x, W_h, W_m
    """

    input_size: int
    hidden_size: int
    memory_size: int
    theta: int

    init_lecun_uni: Callable = nn.initializers.lecun_normal()
    init_zero: Callable = nn.initializers.constant(0.)
    init_xav_norm: Callable = nn.initializers.xavier_normal()

    activation_fn: Callable = nn.tanh

    def setup(self):
        self.A, self.B = self.stateSpaceMatrices(self.memory_size, self.theta)

    @nn.compact
    def __call__(self, carry, x):
        """
        Args:
            carry (tuple): the previous state:
                h: the hidden vector, [batch_size, hidden_size]
                m: the memory vector, [batch_size, memory_size]
                
            x (array): the input vector,[batch_size, input_size]
                
        """

        # Unpack the hidden state and memory
        h, m = carry

        # Eq (7) of the paper, compute the input signal u
        # u: [batch_size, 1]
        u_x = nn.Dense(features=1,
                        use_bias=False,
                        kernel_init=self.init_lecun_uni,
                        bias_init=self.init_zero,
                        param_dtype=jnp.float32)
            # partial(nn.Dense, 
            #             features=self.input_size,
            #             use_bias=False,
            #             kernel_init=self.init_lecun_uni,
            #             bias_init=self.init_zero,
            #             param_dtype=jnp.float32)
        u_h = nn.Dense(features=1,
                        use_bias=False,
                        kernel_init=self.init_lecun_uni,
                        bias_init=self.init_zero,
                        param_dtype=jnp.float32)
            # partial(nn.Dense,
            #             features=self.hidden_size,
            #             use_bias=False,
            #             kernel_init=self.init_lecun_uni,
            #             bias_init=self.init_zero,
            #             param_dtype=jnp.float32)
        u_m = nn.Dense(features=1,
                        use_bias=False,
                        kernel_init=self.init_zero,
                        bias_init=self.init_zero,
                        param_dtype=jnp.float32)
            # partial(nn.Dense,
            #             features=self.memory_size, 
            #             use_bias=False,
            #             kernel_init=self.init_zero,
            #             bias_init=self.init_zero,
            #             param_dtype=jnp.float32)
        u = u_x(x) + u_h(h) + u_m(m)


        # Eq (4) of the paper, compute the memory
        # m: [batch_size, memory_size]
        m_m = jnp.matmul(m, self.A) 
        m_u = jnp.matmul(u, self.B.T)
        new_m= m_m + m_u

        # Eq (6) of the paper, compute the hidden state
        # h: [batch_size, hidden_size]
        h_x = nn.Dense(features=self.hidden_size,
                        use_bias=False,
                        kernel_init=self.init_xav_norm,
                        bias_init=self.init_zero,
                        param_dtype=jnp.float32)
            # partial(nn.Dense,
            #             features=self.hidden_size,
            #             use_bias=False,
            #             kernel_init=self.init_xav_norm,
            #             bias_init=self.init_zero,
            #             param_dtype=jnp.float32)
        h_h = nn.Dense(features=self.hidden_size,
                        use_bias=False,
                        kernel_init=self.init_xav_norm,
                        bias_init=self.init_zero,
                        param_dtype=jnp.float32)
            # partial(nn.Dense,
            #             features=self.hidden_size,
            #             use_bias=False,
            #             kernel_init=self.init_xav_norm,
            #             bias_init=self.init_zero,
            #             param_dtype=jnp.float32)
        h_m = nn.Dense(features=self.hidden_size,
                        use_bias=False,
                        kernel_init=self.init_xav_norm,
                        bias_init=self.init_zero,
                        param_dtype=jnp.float32)
            # partial(nn.Dense,
            #             features=self.hidden_size,
            #             use_bias=False,
            #             kernel_init=self.init_xav_norm,
            #             bias_init=self.init_zero,
            #             param_dtype=jnp.float32)
        new_h = self.activation_fn(h_x(x) + h_h(h) + h_m(m))
        return (new_h, new_m), new_h

    @staticmethod
    def initialized_carry(batch_size, hidden_size, memory_size):
        """Initialize the LMU cell carry.
        Args:
            batch_size (int) : 
                Size of the batch
            hidden_size (int) : 
                Size of the hidden vector (h_t)
            memory_size (int) :
                Size of the memory vector (m_t)
        Returns:
            An initialized carry for the given cell.
        """
        h = jnp.zeros((batch_size, hidden_size))
        m = jnp.zeros((batch_size, memory_size))
        return h, m

    

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
            Uses different parameter initializers when training on psMNIST (as specified in the paper),
            Set to false by default
    
    """

    input_size: int
    hidden_size: int
    memory_size: int
    theta: int
    pmnist: bool = False

    @nn.compact
    def __call__(self, carry, x):
        """
        Parameters:
            carry (tuple): the previous state, set to None initially
                h: the hidden vector, [batch_size, hidden_size]
                m: the memory vector, [batch_size, memory_size]
            x (array): the input vector, [batch_size, seq_len, input_size]
        """

        LMULayer = nn.scan(LMUCell,
                            variable_broadcast="params",
                            split_rngs={"params": False},
                            in_axes=1,
                            out_axes=1)

        return LMULayer(self.input_size, self.hidden_size, self.memory_size, self.theta)(carry, x)
    
    @staticmethod
    def initialize_carry(batch_size, hidden_size, memory_size):
        return LMUCell.initialized_carry(batch_size, hidden_size, memory_size)

#----------------- LMU Model for pMNIST Classification ----------------#
class Model(nn.Module):
    input_size: int
    output_size: int
    hidden_size: int
    memory_size: int
    theta: int

    @nn.compact
    def __call__(self, x):

        # Initialize the carry at the first time step
        # Get initial carry/state (h_0, m_0)
        init_carry = LMU.initialize_carry(x.shape[0], self.hidden_size, self.memory_size)

        # Run the LMU layer
        # h_n: [batch_size, hidden_size]
        (h_n,_),_= LMU(self.input_size, self.hidden_size, self.memory_size, self.theta)(init_carry, x)

        # Run the output layer
        output=nn.Dense(features=self.output_size)(h_n)

        return output

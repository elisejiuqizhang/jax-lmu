import sys
sys.path.append('/usr/local/data/elisejzh/Projects/Mine/jax-lmu')

from src.jax_lmu import *
from flax.training import train_state

from tqdm.notebook import tqdm

import torch
from torch.utils import data
from torch.utils.data import Dataset, DataLoader

from torchvision import datasets, transforms
from torchvision.datasets import MNIST

import optax

#---------------------- Hyperparameters ----------------------#

N_x = 1 # dimension of the input, a single pixel
N_t = 784 # number of time steps (sequence length)
N_h = 212 # dimension of the hidden state
N_m = 256 # dimension of the memory
N_c = 10 # number of classes 
THETA = N_t
N_b = 100 # batch size
N_epochs = 10


#---------------------- Load pMNIST ----------------------#

class psMNIST(Dataset):
    """ Dataset that defines the psMNIST dataset, given the MNIST data and a fixed permutation """

    def __init__(self, mnist, perm):
        self.mnist = mnist # also a torch.data.Dataset object
        self.perm  = perm # file path to the permutation

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        img, label = self.mnist[idx]
        unrolled = img.reshape(-1)
        permuted = unrolled[self.perm]
        permuted = permuted.reshape(-1, 1)
        return permuted, label


transform = transforms.ToTensor()
mnist_train = datasets.MNIST("../data", train = True, download = True, transform = transform)
mnist_val   = datasets.MNIST("../data", train = False, download = True, transform = transform)

perm = torch.load("./mnist_exp/permutation.pt").long() # created using torch.randperm(784)
ds_train = psMNIST(mnist_train, perm)
ds_val   = psMNIST(mnist_val, perm) 

dl_train = DataLoader(ds_train, batch_size = N_b, shuffle = True, num_workers = 2)
dl_val   = DataLoader(ds_val, batch_size = N_b, shuffle = True, num_workers = 2)


#---------------------- Classifier for pMNIST ----------------------#
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

model = Model(
            input_size  = N_x,
            output_size = N_c,
            hidden_size = N_h, 
            memory_size = N_m, 
            theta = THETA
        )

#---------------------- Optimizer ----------------------#
learning_rate=1e-3
optimizer=optax.adam(learning_rate)

#---------------------- Loss & Metrics ----------------------#
"""Following the Flax example: https://flax.readthedocs.io/en/latest/getting_started.html"""

def cross_entropy_loss(*, logits, labels):
  labels_onehot = jax.nn.one_hot(labels, num_classes=10)
  return optax.softmax_cross_entropy(logits=logits, labels=labels_onehot).mean()

def compute_metrics(*, logits, labels):
  loss = cross_entropy_loss(logits=logits, labels=labels)
  accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
  metrics = {
      'loss': loss,
      'accuracy': accuracy,
  }
  return metrics

#---------------------- Training ----------------------#

def train(model, optimizer, dl_train, dl_val, num_epochs):

    @jit
    def loss_fn(params, x, y):
        logits = model.apply(params, x)
        return cross_entropy_loss(logits=logits, labels=y)

    @jit
    def update(params, x, y, state):

        l, grads=value_and_grad(loss_fn)(params, x, y)
        updates, state=optimizer.update(grads, state)
        params=optax.apply_updates(params, updates)
        return l, params, state

    # Model initialization (hmmm this really takes quite long)
    params = model.init(random.PRNGKey(0), jnp.empty((1, N_t, N_x))) # initialize model parameters by passing a template input
    state = optimizer.init(params)
    print("Model initialized.")

    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1} of {num_epochs}")
        for batch in tqdm(dl_train):
            x, y=batch

            # convert to jnp arrays
            x=x.detach().cpu().numpy()
            y=y.detach().cpu().numpy()
            x=jnp.array(x)
            y=jnp.array(y)

            loss, params, state = update(params, x, y, state)
        print(f"Training loss: {loss}")

        # Validation loop
        for batch in tqdm(dl_val):

            inputs, labels = batch
            
            # convert to jnp arrays
            inputs = inputs.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            inputs=jnp.array(inputs)
            labels=jnp.array(labels)

            logits = model.apply(params, inputs)
            metrics = compute_metrics(logits=logits, labels=labels)
            print(f"Validation loss: {metrics['loss']}")
            print(f"Validation accuracy: {metrics['accuracy']}")

    return params

params = train(model, optimizer, dl_train, dl_val, N_epochs)

import sys
sys.path.append('/usr/local/data/elisejzh/Projects/Mine/jax-lmu')

from src.jax_lmufft import *

import tensorflow_datasets as tfds 

import optax
from flax.training import train_state

#---------------------- Hyperparameters ----------------------#

N_x = 1 # dimension of the input, a single pixel
N_t = 784 # number of time steps (sequence length) - here it's 28 * 28 since we are using MNIST and making it 1D
N_h = 128 # dimension of the hidden state
N_m = 64 # dimension of the memory
N_c = 10 # number of classes 
THETA = N_t
N_b = 256 # batch size
N_epochs = 20 # number of epochs


lr=1e-4 # learning rate for adam optimizer

#---------------------- Load MNIST ----------------------#


def get_datasets():
    """Load MNIST train and test datasets into memory."""

    ds_builder = tfds.builder('mnist')
    ds_builder.download_and_prepare()
    train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
    val_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))


    train_ds['image'] = jnp.float32(train_ds['image'])
    val_ds['image'] = jnp.float32(val_ds['image'])

    return train_ds, val_ds


train_ds, val_ds = get_datasets()


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

#---------------------- Utility Functions for Training ----------------------#

def create_train_state(rng, learning_rate=lr):
    model = Model(
            input_size  = N_x,
            output_size = N_c,
            hidden_size = N_h, 
            memory_size = N_m, 
            seq_len=N_t,
            theta = THETA
        )

    
    params = model.init(rng, jnp.ones((1, N_t, N_x)))['params']
    print("Model initialized.")

    optimizer = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)



def train_step(state, batch):
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch['image'])
        loss = cross_entropy_loss(logits=logits, labels=batch['label'])
        return loss
    grad_fn = value_and_grad(loss_fn, has_aux=False)
    loss, grads = grad_fn(state.params)
    new_state = state.apply_gradients(grads=grads)

    logits = new_state.apply_fn({'params': new_state.params}, batch['image'])
    metrics = compute_metrics(logits=logits, labels=batch['label'])

    return new_state, metrics 

def eval_step(state, batch, seq_len=N_t, input_size=N_x):
    batch['image']=batch['image'].reshape((-1, seq_len, input_size))
    logits = state.apply_fn({'params': state.params}, batch['image'])
    return compute_metrics(logits=logits, labels=batch['label'])


def train_epoch(state, train_ds, epoch, rng, batch_size=N_b, seq_len=N_x, input_size=N_x):

    train_ds_size = len(train_ds['image'])
    steps_per_epoch = train_ds_size // batch_size

    perms = jax.random.permutation(rng, train_ds_size) # get a randomized index array
    perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
    perms = perms.reshape((steps_per_epoch, batch_size)) # index array, where each row is a batch

    batch_metrics = []
    for perm in perms:
        batch = {k: v[perm, ...] for k, v in train_ds.items()} # dict{'image': array, 'label': array}
        batch['image']=batch['image'].reshape((batch_size, -1, input_size)) # reshape to the required input dimensions
        state, metrics = train_step(state, batch)
        batch_metrics.append(metrics)
    
    # compute mean of metrics across each batch in epoch.
    batch_metrics_np = jax.device_get(batch_metrics)
    epoch_metrics_np = {
        k: np.mean([metrics[k] for metrics in batch_metrics_np])
        for k in batch_metrics_np[0]} # jnp.mean does not work on lists

    print('train epoch: %d, loss: %.4f, accuracy: %.2f' % (epoch, epoch_metrics_np['loss'], epoch_metrics_np['accuracy'] * 100))

    return state


def eval_model(state, test_ds):
    metrics = eval_step(state, test_ds)
    metrics = jax.device_get(metrics)
    summary = jax.tree_util.tree_map(lambda x: x.item(), metrics) # map the function over all leaves in metrics
    return summary['loss'], summary['accuracy']

#---------------------- Training ----------------------#

# Random seed
rng = jax.random.PRNGKey(0)
rng, init_rng = jax.random.split(rng)

# Initialize model
state=create_train_state(rng, learning_rate=lr)
del init_rng

for epoch in range(N_epochs):
    # Use a separate PRNG key to permute image data during shuffling
    rng, input_rng = jax.random.split(rng)
    state = train_epoch(state, train_ds, epoch, input_rng)
    test_loss, test_accuracy = eval_model(state, val_ds)
    print(' test epoch: %d, loss: %.2f, accuracy: %.2f' % (epoch, test_loss, test_accuracy * 100))

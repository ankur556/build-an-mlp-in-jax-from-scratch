"""
Build an MLP in JAX from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - make_prng_key
import jax
import jax.numpy as jnp


def make_prng_key(seed):
    # TODO: wrap a Python integer seed into a JAX PRNG key (uint32 array of shape (2,))
    return jax.random.PRNGKey(seed)

# Step 2 - split_prng_key
import jax

def split_prng_key(key, num):
    # TODO: split `key` into `num` independent subkeys and return them as a (num, 2) array.
    return  jax.random.split(key,num)

# Step 3 - sample_normal_matrix
import jax
import jax.numpy as jnp

def sample_normal_matrix(key, shape):
    return jax.random.normal(key,shape)

# Step 4 - sample_input_features
import jax
import jax.numpy as jnp

def sample_input_features(key, batch_size, num_features):
    """Sample a (batch_size, num_features) standard-normal feature batch."""
    return jax.random.normal(key,(batch_size,num_features))

# Step 5 - assign_class_labels
import jax.numpy as jnp

def assign_class_labels(inputs, num_classes):
    # Use only the first num_classes columns of inputs
    relevant_scores = inputs[:, :num_classes]
    # Find the index of the max value along axis=1 (per row)
    labels = jnp.argmax(relevant_scores, axis=1)
    # Convert to int32
    return labels.astype(jnp.int32)

# Step 6 - one_hot_encode_labels
def one_hot_encode_labels(inputs, num_classes):
     return jnp.eye(num_classes)[labels]

# Step 7 - init_linear_layer
def init_linear_layer(key, in_dim, out_dim, scale=0.1):
    W = scale * sample_normal_matrix(key, (in_dim, out_dim))
    b = jnp.zeros((out_dim,))
    return {'W': W, 'b': b}

# Step 8 - init_mlp_params
def init_mlp_params(key, layer_sizes, scale=0.1):
    # TODO: build a list of per-layer parameter dicts from adjacent layer sizes.
    num_layers=len(layer_sizes)-1
    keys=jax.random.split(key,num_layers)
    params=[]
    for i in range(num_layers):
        in_dim=layer_sizes[i]
        out_dim=layer_sizes[i+1]
        layer_key=keys[i]
        layer_params=init_linear_layer(layer_key,in_dim,out_dim,scale)
        params.append(layer_params)

    return params

# Step 9 - linear_forward
def linear_forward(x, layer_params):
    # TODO: compute x @ W + b using layer_params['W'] and layer_params['b'].
    W = layer_params['W']
    b = layer_params['b']
    return x @ W + b

# Step 10 - relu_activation
import jax.numpy as jnp


def relu_activation(x):
    """Apply the ReLU activation elementwise to a JAX array."""
    # TODO: return an array of the same shape with negatives replaced by zero.
    return jnp.maximum(x, 0)

# Step 11 - softmax_probabilities
import jax.numpy as jnp

def softmax_probabilities(logits):
    # TODO: convert logits into a numerically stable softmax along the last axis
    shifted_logits=logits-jnp.max(logits,axis=-1,keepdims=True)
    exp_logits=jnp.exp(shifted_logits)
    sum_exp=jnp.sum(exp_logits,axis=-1,keepdims=True)
    return exp_logits/sum_exp

# Step 12 - mlp_forward
def mlp_forward(params, x):
    # For each hidden layer (all except last), apply linear + ReLU
    for layer_params in params[:-1]:
        # Swap the order: input 'x' comes first, then the parameters
        x = linear_forward(x, layer_params)
        x = relu_activation(x)
        
    # Final layer: linear only, output logits
    logits = linear_forward(x, params[-1])
    
    return logits

# Step 13 - log_softmax_logits
def log_softmax_logits(logits):
    # TODO: return the numerically stable log-softmax of logits along the last axis.
    m = np.max(logits, axis=-1, keepdims=True)
    shifted_logits = logits - m
    log_sum_exp = np.log(np.sum(np.exp(shifted_logits), axis=-1, keepdims=True))
    return shifted_logits - log_sum_exp

# Step 14 - cross_entropy_loss (not yet solved)
# TODO: implement

# Step 15 - classification_accuracy (not yet solved)
# TODO: implement

# Step 16 - loss_fn_of_params (not yet solved)
# TODO: implement

# Step 17 - compute_param_grads (not yet solved)
# TODO: implement

# Step 18 - sgd_update_params (not yet solved)
# TODO: implement

# Step 19 - training_step (not yet solved)
# TODO: implement

# Step 20 - train_mlp (not yet solved)
# TODO: implement

# Step 21 - predict_classes (not yet solved)
# TODO: implement


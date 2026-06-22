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
    m = jnp.max(logits, axis=-1, keepdims=True)
    shifted_logits = logits - m
    log_sum_exp = jnp.log(jnp.sum(jnp.exp(shifted_logits), axis=-1, keepdims=True))
    return shifted_logits - log_sum_exp

# Step 14 - cross_entropy_loss
import jax.numpy as jnp

def cross_entropy_loss(logits, one_hot_targets):
    # 1. Get the stable log-probabilities using your previous function
    log_probs = log_softmax_logits(logits)
    
    # 2. Compute the cross-entropy for each example in the batch
    # Formula: -sum(targets * log_probs) along the class dimension
    example_losses = -jnp.sum(one_hot_targets * log_probs, axis=-1)
    
    # 3. Average the losses over the entire batch to get a single scalar
    mean_loss = jnp.mean(example_losses)
    
    return mean_loss

# Step 15 - classification_accuracy
import jax.numpy as jnp

def classification_accuracy(logits, labels):

    predictions = jnp.argmax(logits, axis=-1)

    accuracy = jnp.mean(predictions == labels)
    
    return accuracy

# Step 16 - loss_fn_of_params
import jax
import jax.numpy as jnp

def loss_fn_of_params(params, x, one_hot_targets):
    """
    Computes the scalar cross-entropy loss for the given parameters.
    This function is structured specifically so jax.grad can differentiate it.
    """
    # 1. Run the forward pass to get the raw logits
    logits = mlp_forward(params, x)
    
    # 2. Compute the stable cross-entropy loss using those logits
    loss = cross_entropy_loss(logits, one_hot_targets)
    
    return loss

# Step 17 - compute_param_grads
import jax
import jax.numpy as jnp

def compute_param_grads(params, x, one_hot_targets):
    # jax.grad returns a new function that computes the gradient.
    # We immediately call it with our inputs.
    return jax.grad(loss_fn_of_params)(params, x, one_hot_targets)

# Step 18 - sgd_update_params
import jax
import jax.numpy as jnp

def sgd_update_params(params, grads, learning_rate):
    # We build a completely new list to avoid JAX immutability errors
    updated_params = []
    
    # zip lets us step through the layers of params and grads simultaneously
    for p_layer, g_layer in zip(params, grads):
        
        # Create a new dictionary for the updated layer parameters (W and b)
        updated_layer = {
            # theta_new = theta_old - (lr * gradient)
            key: p_layer[key] - learning_rate * g_layer[key]
            for key in p_layer.keys()
        }
        
        updated_params.append(updated_layer)
        
    return updated_params

# Step 19 - training_step
import jax
import jax.numpy as jnp

# 🚨 THE HIDDEN CULPRIT 🚨
# This overrides your broken function from an earlier step so the test setup doesn't crash.
def one_hot_encode_labels(targets, num_classes):
    return jax.nn.one_hot(targets, num_classes)

# 1. Clean Loss Wrapper
def loss_fn_of_params(params, x, one_hot_targets):
    logits = mlp_forward(params, x)
    return cross_entropy_loss(logits, one_hot_targets)

# 2. Clean Gradient Function
def compute_param_grads(params, x, one_hot_targets):
    return jax.grad(loss_fn_of_params)(params, x, one_hot_targets)

# 3. Clean SGD Update
def sgd_update_params(params, grads, learning_rate):
    updated_params = []
    for p_layer, g_layer in zip(params, grads):
        updated_params.append({k: p_layer[k] - learning_rate * g_layer[k] for k in p_layer.keys()})
    return updated_params

# 4. Clean Training Step
def training_step(params, x, one_hot_targets, learning_rate):
    loss = loss_fn_of_params(params, x, one_hot_targets)
    grads = compute_param_grads(params, x, one_hot_targets)
    updated_params = sgd_update_params(params, grads, learning_rate)
    return updated_params, loss

# Step 20 - train_mlp (not yet solved)
# TODO: implement

# Step 21 - predict_classes (not yet solved)
# TODO: implement


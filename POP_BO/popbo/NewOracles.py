import numpy as np
from Oracles import sigmoid, Bernoulli_sample
#For the model implementation
from collections import namedtuple
from Stackelberg.src.utils.utility_functions import matyas
import jax.numpy as jnp
from Preference_Embeddings.JAXEmbeddings import ComplexPreference
import flax.serialization as serialization
import json
import jax
from typing import Callable
from pathlib import Path
import os.path as path
import os

#Model loading logic
def get_model(name: str) -> Callable:
    # -----------------------------------------------------------------------------
    # 2.2) Read hyperparameters from JSON
    # -----------------------------------------------------------------------------
    base_path = '/Users/josecruz/Desktop/S25 Files/MIE Lab/Code/'
    hparam_path = path.join(
        base_path,
        f"Embedding_Model_Weights/{name}.json"
    )
    with open(hparam_path, "r") as f:
        metadata = json.load(f)
    hparams = metadata["hparams"]
    in_dim = hparams["in_dim"]
    factor = hparams["factor"]
    sizes  = hparams["sizes"]  # e.g. [128, 64]

    # Re‐instantiate the Flax model with these hyperparameters
    model_def = ComplexPreference(in_dim=in_dim, factor=factor, sizes=sizes)

    # -----------------------------------------------------------------------------
    # 2.3) Build a dummy “params” PyTree to use as a template
    # -----------------------------------------------------------------------------
    # ANY batch_size ≥ 1 will do; we only need correct shapes.
    rng = jax.random.PRNGKey(0)
    dummy_x  = jnp.zeros((1, in_dim), dtype=jnp.float32)
    dummy_xp = jnp.zeros((1, in_dim), dtype=jnp.float32)

    variables    = model_def.init(rng, dummy_x, dummy_xp)
    dummy_params = variables["params"]  # a FrozenDict tree matching the saved structure

    # -----------------------------------------------------------------------------
    # 2.4) Read the saved parameter bytes and re‐hydrate them
    # -----------------------------------------------------------------------------
    param_path = path.join(base_path, f"Embedding_Model_Weights/{name}.msgpack")
    with open(param_path, "rb") as f:
        loaded_bytes = f.read()

    restored_params = serialization.from_bytes(dummy_params, loaded_bytes)
    return lambda x, y: model_def.apply({"params": restored_params}, x, y)


def det_oracle(fx, fx_prime):
    return fx > fx_prime

def model_oracle(model):
    def oracle(fx, fx_prime):
        score = model(fx, fx_prime)
        score = np.clip(score, -6, 6) #prevents overflow in sigmoid
        p = sigmoid(score)
        xwin = Bernoulli_sample(p)
        return xwin
    return oracle

def det_model_oracle(model):
    def oracle(fx, fx_prime):
        return model(fx, fx_prime) > 0 #since p = sigmoid(score) and sigmoid(0) = 0.5
    return oracle

def get_named_oracle(name: str, deterministic: bool = False):
    model = get_model(name)
    if deterministic:
        return det_model_oracle(model)
    else:
        return model_oracle(model)
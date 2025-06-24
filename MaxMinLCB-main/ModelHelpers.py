from typing import Callable
import jax.numpy as jnp
from Preference_Embeddings.JAXEmbeddings import ComplexPreference
import flax.serialization as serialization
import json
import jax

def get_model(name: str) -> Callable:
    # -----------------------------------------------------------------------------
    # 2.2) Read hyperparameters from JSON
    # -----------------------------------------------------------------------------
    hparam_path = f"../Embedding_Model_Weights/{name}.json"
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
    param_path = f"../Embedding_Model_Weights/{name}.msgpack"
    with open(param_path, "rb") as f:
        loaded_bytes = f.read()

    restored_params = serialization.from_bytes(dummy_params, loaded_bytes)
    return lambda x, y: model_def.apply({"params": restored_params}, x, y)
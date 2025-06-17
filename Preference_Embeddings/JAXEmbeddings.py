# jax_native_embeddings.py

import numpy as np
from typing import Callable, Sequence, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import random

import flax.linen as nn
import optax

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
)


# -----------------------------------------------------------------------------
# 1) ComplexPreference (Flax/Linen)
# -----------------------------------------------------------------------------
class ComplexPreference(nn.Module):
    """
    Implements a “preference score” as the imaginary part of a Hermitian inner product
    of two embedded vectors.  Given inputs x, x' ∈ R^in_dim, we embed each into C^(in_dim*factor/2)
    by producing “radii” and “angles,” then computing
        im_part = sum_i [ r_i(x) * r_i(x') * sin(θ_i(x) - θ_i(x')) ].

    - in_dim: input dimension (e.g. 2).
    - factor: an even integer, so that the complex embedding has dimension in_dim * (factor/2).
    - sizes: optional list of hidden‐layer sizes for both the “radii” and “angles” MLPs.
             If None, we default to a single hidden layer of size 128.
    """
    in_dim: int
    factor: int = 2
    sizes: Optional[Sequence[int]] = None
    branches: int = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, x_prime: jnp.ndarray) -> jnp.ndarray:
        # Set up hidden sizes
        sizes = self.sizes if self.sizes is not None else [128]
        branches = 1 if self.branches is None else self.branches
        branch_sizes = sizes[:branches]
        hidden_dims = sizes[branches:]

        total_complex_dim = self.in_dim * self.factor
        half_dim = total_complex_dim // 2  # Dimension of the “complex” embedding per side
        # -----------------------
        # Trunk network
        # -----------------------
        def trunk_mlp(z: jnp.ndarray) -> jnp.ndarray:
            for h in branch_sizes:
                z = nn.Dense(h)(z)
                z = nn.swish(z)
            return z

        # ------------------------
        # Radii network (r(x))
        # ------------------------
        def radii_mlp(z: jnp.ndarray) -> jnp.ndarray:
            z = trunk_mlp(z)
            for h in hidden_dims:
                z = nn.Dense(h)(z)
                z = nn.swish(z)      # “Hardswish” ≈ swish in JAX
            z = nn.Dense(half_dim)(z)
            z = nn.softplus(z)      # ensure radii ≥ 0
            return z  # shape: [batch, half_dim]

        # ------------------------
        # Angles network (θ(x))
        # ------------------------
        def angles_mlp(z: jnp.ndarray) -> jnp.ndarray:
            z = trunk_mlp(z)
            for h in hidden_dims:
                z = nn.Dense(h)(z)
                z = nn.swish(z)
            z = nn.Dense(half_dim)(z)
            z = nn.softplus(z)      # produce positive angles
            z = z * (2.0 * jnp.pi)  # scale to [0, 2π) range
            return z  # shape: [batch, half_dim]

        # embed x and x_prime
        r_x   = radii_mlp(x)        # [batch, half_dim]
        theta_x   = angles_mlp(x)       # [batch, half_dim]
        r_xp  = radii_mlp(x_prime)  # [batch, half_dim]
        theta_xp  = angles_mlp(x_prime) # [batch, half_dim]

        # Compute imaginary‐part (vectorized over batch):
        #   im = sum_i ( r_i(x)*r_i(x') * sin(θ_i(x) - θ_i(x')) )
        im = jnp.sum(r_x * r_xp * jnp.sin(theta_x - theta_xp), axis=-1)  # shape: [batch]
        return im

# -----------------------------------------------------------------------------
# 2) FuncDataset (simple NumPy/JAX version + batching)
# -----------------------------------------------------------------------------
def get_ranges_and_evals(
        num_pairs: int,
        func: Callable[[np.ndarray], np.ndarray],
        in_dim: int = 2,
        bounds: Optional[Sequence[Tuple[float, float]]] = None,
        seed: Optional[int] = None,
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Builds and returns the ranges and evaluations of `func` over `num_pairs` random pairs."""
    if bounds is None:
        # default to [0,1]^in_dim
        bounds = [(0.0, 1.0)] * in_dim

    rng = np.random.RandomState(seed)
    lows  = np.array([b[0] for b in bounds])
    highs = np.array([b[1] for b in bounds])

    X = rng.uniform(low=lows, high=highs, size=(num_pairs, in_dim))
    Y = rng.uniform(low=lows, high=highs, size=(num_pairs, in_dim))

    fX = func(X)  # shape: [num_pairs]
    fY = func(Y)

    return X, Y, fX, fY

def generate_func_data(
        num_pairs: int,
        func: Callable[[np.ndarray], np.ndarray],
        in_dim: int = 2,
        bounds: Optional[Sequence[Tuple[float, float]]] = None,
        seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate `num_pairs` random input pairs (X[i], Y[i]) in [bounds]^in_dim
    and compute binary labels “1 if func(X) < func(Y) (minimization), else 0.”

    Returns:
      X: np.ndarray of shape [num_pairs, in_dim]
      Y: np.ndarray of shape [num_pairs, in_dim]
      labels: np.ndarray of shape [num_pairs] in {0,1}
    """
    if bounds is None:
        # default to [0,1]^in_dim
        bounds = [(0.0, 1.0)] * in_dim

    X, Y, fX, fY = get_ranges_and_evals(num_pairs, func, in_dim, bounds, seed)
    labels = (fX < fY).astype(np.int32)
    return X, Y, labels


def data_batch_generator(
        X: np.ndarray,
        Y: np.ndarray,
        labels: np.ndarray,
        batch_size: int,
        shuffle: bool = True,
        seed: Optional[int] = None
):
    """
    Yields minibatches of (x_batch, x_prime_batch, label_batch) as JAX arrays.
    """
    num_pairs = X.shape[0]
    idxs = np.arange(num_pairs)
    if shuffle:
        rng = np.random.RandomState(seed)
        rng.shuffle(idxs)

    for start in range(0, num_pairs, batch_size):
        end = start + batch_size
        batch_idxs = idxs[start:end]
        x_b   = X[batch_idxs]
        xp_b  = Y[batch_idxs]
        lbl_b = labels[batch_idxs]
        yield jnp.array(x_b, dtype=jnp.float32), jnp.array(xp_b, dtype=jnp.float32), jnp.array(lbl_b, dtype=jnp.float32)

def unlabeled_batch_generator(
        X: np.ndarray,
        Y: np.ndarray,
        batch_size: int = 256,
        shuffle: bool = True,
        seed: Optional[int] = None
):
    """
    Yields minibatches of (x_batch, x_prime_batch, label_batch) as JAX arrays.
    """
    num_pairs = X.shape[0]
    Xidxs = np.arange(num_pairs)
    Yidxs = np.arange(num_pairs)
    if shuffle:
        rng = np.random.RandomState(seed)
        rng.shuffle(Xidxs)
        rng.shuffle(Yidxs)

    for start in range(0, num_pairs, batch_size):
        end = start + batch_size
        Xbatch_idxs = Xidxs[start:end]
        Ybatch_idxs = Yidxs[start:end]
        x_b   = X[Xbatch_idxs]
        xp_b  = Y[Ybatch_idxs]
        lbl_b = x_b >= xp_b
        yield jnp.array(x_b, dtype=jnp.float32), jnp.array(xp_b, dtype=jnp.float32), jnp.array(lbl_b, dtype=jnp.float32)

# -----------------------------------------------------------------------------
# 3) train_on_func (JAX training loop with Optax)
# -----------------------------------------------------------------------------
def train_on_func(
        rng_key: jax.random.PRNGKey,
        model_def: nn.Module,
        func: Callable[[np.ndarray], np.ndarray],
        bounds: Optional[Sequence[Tuple[float, float]]] = None,
        in_dim: int = 2,
        factor: int = 2,
        sizes: Optional[Sequence[int]] = None,
        num_pairs: int = 50_000,
        batch_size: int = 512,
        epochs: int = 10,
        lr: float = 1e-1,
        patience: int = 10
) -> dict:
    """
    Trains `model_def` (an uninitialized Flax module) on randomly generated pairs
    from `func` (minimization). Returns a dict containing:
      - 'params': learned parameters
      - 'train_losses': list of per‐epoch final loss
    """
    # 1) Generate the entire dataset once (NumPy)
    X_all, Y_all, fX_all, fY_all = get_ranges_and_evals(
        num_pairs=num_pairs,
        func=func,
        in_dim=in_dim,
        bounds=bounds,
        seed=0
    )

    # 2) Initialize model parameters
    init_batch = jnp.zeros((batch_size, in_dim), dtype=jnp.float32)
    variables = model_def.init(rng_key, init_batch, init_batch)  # e.g. {'params': {...}}
    params = variables['params']

    # 3) Build learning‐rate schedule: warmup → cosine‐decay → 0
    total_steps = epochs * ((num_pairs + batch_size - 1) // batch_size)
    warmup_steps = int(0.1 * total_steps)
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=lr,
        warmup_steps=warmup_steps,
        decay_steps=total_steps - warmup_steps,
        end_value=0.0
    )

    # 4) Create optimizer (Adam) with that schedule
    optimizer = optax.adamw(learning_rate=lr_schedule)
    opt_state = optimizer.init(params)

    # 5) Define one training‐step (JIT‐compiled)
    @jax.jit
    def train_step(params, opt_state, x_b, xp_b, lbl_b):
        def loss_fn(p):
            logits = model_def.apply({'params': p}, x_b, xp_b)  # shape [batch]
            # Use BCE with logits:
            bce = optax.sigmoid_binary_cross_entropy(logits=logits, labels=lbl_b)
            return jnp.mean(bce)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    # 6) Training loop with early‐stopping
    best_loss = np.inf
    bad_epochs = 0
    train_losses = []

    step = 0
    for epoch in range(1, epochs + 1):
        # Shuffle + batch generator for this epoch
        epoch_seed = int(jax.random.randint(rng_key, (), 0, 1e6))
        batch_gen = unlabeled_batch_generator(
            fX_all, fY_all, batch_size, shuffle=True, seed=epoch_seed
        )

        epoch_loss_accum = 0.0
        batch_count = 0

        for x_b, xp_b, lbl_b in batch_gen:
            params, opt_state, loss = train_step(params, opt_state, x_b, xp_b, lbl_b)
            epoch_loss_accum += float(loss)
            batch_count += 1
            step += 1

        avg_epoch_loss = epoch_loss_accum / batch_count
        train_losses.append(avg_epoch_loss)

        # Early stopping check
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping at epoch {epoch} (avg loss {avg_epoch_loss:.6f})")
                break

        if (epoch % 10) == 0:
            print(f"Epoch {epoch:03d} — avg loss: {avg_epoch_loss:.6f}")

    return {
        'params': params,
        'train_losses': train_losses
    }

# -----------------------------------------------------------------------------
# 4) evaluate (compute metrics on fresh random pairs)
# -----------------------------------------------------------------------------
def evaluate(
        rng_key: jax.random.PRNGKey,
        model_def: nn.Module,
        params: dict,
        func: Callable[[np.ndarray], np.ndarray],
        bounds: Optional[Sequence[Tuple[float, float]]] = None,
        in_dim: int = 2,
        num_pairs: int = 50_000,
        batch_size: int = 512
) -> dict:
    """
    Generates a fresh test set of `num_pairs` random (X,Y) from func, then
    runs the trained model to compute accuracy, precision, recall, f1, ROC‐AUC, and confusion matrix.

    Returns a dict containing all of those metrics.
    """
    # 1) Generate fresh test data (NumPy)
    X_all, Y_all, L_all = generate_func_data(
        num_pairs=num_pairs,
        func=func,
        in_dim=in_dim,
        bounds=bounds,
        seed=42,
    )

    all_preds = []
    all_scores = []
    all_targets = []

    # 2) Iterate in minibatches
    test_batches = data_batch_generator(
        X_all, Y_all, L_all, batch_size, shuffle=False
    )
    for x_b, xp_b, lbl_b in test_batches:
        logits = model_def.apply({'params': params}, x_b, xp_b)  # [batch]
        probs  = jax.nn.sigmoid(logits)                          # [batch]
        preds  = (probs > 0.5).astype(jnp.int32)                 # [batch]

        all_scores.append(np.asarray(probs))
        all_preds.append(np.asarray(preds))
        all_targets.append(np.asarray(lbl_b))

    # 3) Flatten and compute sklearn metrics
    scores  = np.concatenate(all_scores, axis=0).ravel()
    preds   = np.concatenate(all_preds, axis=0).ravel()
    targets = np.concatenate(all_targets, axis=0).ravel()

    acc = accuracy_score(targets, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(targets, preds, average='binary')
    cm = confusion_matrix(targets, preds)
    try:
        auc = roc_auc_score(targets, scores)
    except ValueError:
        auc = float('nan')

    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")
    print("Confusion Matrix:")
    print(cm)

    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'roc_auc': auc,
        'confusion_matrix': cm
    }

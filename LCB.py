'''stackelberg.py
A reference implementation of the **MAXMIN‑LCB** (a.k.a. Stackelberg) algorithm from
*“Bandits with Preference Feedback: A Stackelberg Game Perspective”* (Pasztor et al., 2024).

═══════════════════════════════════════════════════════════════════════════════
Quick theory crib‑sheet (see comments inline)
───────────────────────────────────────────────────────────────────────────────
• **Representer theorem** – turns the infinite‑dimensional RKHS optimisation of
  Eq. (6) into a finite‑dimensional search over dual coefficients **α**.  The
  optimal \(h_t\) is a kernel expansion `h(x,x′)=Σ α_i k_D((x,x′),z_i)` and we
  solve for **α** once per round (see `_update_model`).
• **Dueling‑kernel diagonal ≤ 4** – for any pair (x,x′) one has
  `k_D((x,x′),(x,x′)) = k(x,x)+k(x′,x′)−2k(x,x′) ≤ 2+2 = 4` when the base
  kernel is bounded by 1.  Hence all eigenvalues of **Kₜ** are ≤ 4, giving the
  log‑det bound used in `_default_beta`.
• **β‑schedule** – The exact information‑gain term γₜ is expensive (O(t³)), so
  we upper‑bound `det(I+λ⁻¹Kₜ)`.  Details are spelled out around
  line 290.
• **Lipschitz constant of σ** – σ′(z)≤¼ for all z, but we keep a configurable
  variable so you can loosen/tighten the exploration width.

The rest of the file mirrors Algorithm 1 line‑by‑line, with every update step
labelled accordingly.  Only *comments* changed in this revision.
»Author: OpenAI ChatGPT o3,2025‑05‑21
'''

from __future__ import annotations

import math
#import warnings
from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple, Optional

import numpy as np
import torch

try:  # optional heavy deps ---------------------------------------------------
    import nevergrad as ng  # type: ignore
except ImportError:  # pragma: no cover – optional
    ng = None  # noqa: N816 – keep sentinel lowercase for clarity

try:
    from scipy import optimize  # type: ignore
    from scipy.stats import qmc
except ImportError:  # pragma: no cover
    optimize = None  # type: ignore

################################################################################
# Helper utilities
################################################################################

def sigmoid(z: torch.Tensor) -> torch.Tensor:
    """Numerically‑stable sigmoid that accepts *and* returns ``torch.Tensor``."""
    return torch.sigmoid(z)  # PyTorch already guards against overflow


def gaussian_kernel(lengthscale: float = 1.0) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Factory for an RBF kernel that matches the call signature expected here.

    The lambda ensures broadcasting over leading dims while keeping the last
    dim as the feature dimension.
    """

    ls2 = 2 * (lengthscale ** 2)

    def _k(x: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        diff = x - x2
        return torch.exp(-diff.pow(2).sum(dim=-1) / ls2)

    return _k

################################################################################
# Config containers
################################################################################


@dataclass
class StackelbergConfig:
    """All hyper‑parameters required by the algorithm (§3 of the paper).
    :param bounds: sequence of (low, high) pairs, one per dimension
    :param kernel: callable that returns a float in [0,1] from a pair of tensors
    :param beta_schedule: callable that returns a float from an int representing the time step
    :param lamb: float, L2 regularisation strength
    :param B: float, ball radius in the RKHS
    :param optimizer_backend: "scipy" | "nevergrad" | "random"
    :param random_search_budget: int, used if backend == "random" max amount of random search iterations
    """

    bounds: Sequence[Tuple[float, float]]  # problem domain ℳ ⊂ ℝᵈ
    kernel: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = gaussian_kernel()
    beta_schedule: Optional[Callable[[int], float]] = None  # fall‑back added later
    lamb: float = 1.0  # λ  (ridge reg.)
    B: float = 3.0  # ball radius in 𝓗_k
    optimizer_backend: str = "scipy"  # "scipy" | "nevergrad" | "random"
    random_search_budget: int = 128  # used if backend == "random"
    decision_buffer: float = 0.05  # ε in the ≥½−ε condition
    candidate_check_points: int = 64  # K Monte‑Carlo samples for each check

    # ==== BEGIN  max-min-LCB closure ======================================

def make_max_min_lcb(
        *,
        beta: float,
        decision_buffer: float = 0.0,
        argmax_tol: float = 1e-4,
        use_candidate_set: bool = True,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.double,
) -> Callable[
    [torch.Tensor, torch.Tensor, Optional[int], Optional[torch.Tensor]],
    Tuple[int, int]
]:
    """
    Factory that returns a *state-free* max-min-LCB selector.

    The returned function expects:
        h_mat    : (n,n)  – GP posterior mean  h_t(x_i,x_j)
        var_mat  : (n,n)  – posterior variance σ_t^D(x_i,x_j)^2
        key      :  int   – pseudo-RNG seed for tie-breaks
        mask_M   : (n,)   – boolean viability mask for leaders (optional)

    and returns `(i,j)` indices of the next query (leader i, follower j).
    """
    def _selector(
            h_mat: torch.Tensor,
            var_mat: torch.Tensor,
            key: Optional[int] = None,
            mask_M: Optional[torch.Tensor] = None,
    ) -> Tuple[int, int]:
        n = h_mat.shape[0]
        if key is not None:
            torch.random.manual_seed(key)

        # -----------------------------------------------------------------
        # 1.  Lower Confidence Bound  ℓ_ij = μ_ij − β·σ_ij
        # -----------------------------------------------------------------
        lcb = h_mat - beta * var_mat.sqrt()      # (n,n)

        # -----------------------------------------------------------------
        # 2.  Candidate-set pruning  (viability predicate)
        # -----------------------------------------------------------------
        if use_candidate_set:
            #   Leader i is viable  ⇔  ∀j:  μ_ij + β σ_ij ≥ ½ − ε
            ucb = h_mat + beta * var_mat.sqrt()
            leader_ok = torch.all(
                torch.logical_or(
                    ucb > 0.5 - decision_buffer,
                    torch.eye(n, dtype=torch.bool, device=device)
                ), dim=1)                        # (n,)
            if mask_M is not None:               # allow user override
                leader_ok &= mask_M.to(torch.bool)

            # broadcast masks & forbid self-duels
            lcb = torch.where(
                leader_ok[:, None] & leader_ok[None, :],
                lcb,
                torch.full_like(lcb, math.nan)
            )
            lcb.fill_diagonal_(math.nan)
        else:
            leader_ok = torch.ones(n, dtype=torch.bool, device=device)
            lcb.fill_diagonal_(0.5)              # ignored but finite

        # -----------------------------------------------------------------
        # 3.  min_j LCB(i,j)  and  argmax_i  of that quantity
        # -----------------------------------------------------------------
        min_j, _  = torch.nanmin(lcb, dim=1)     # (n,)
        # randomise *among* ties for j
        diff_j = torch.abs(lcb - min_j[:, None])
        rand_j = torch.rand_like(lcb) * (diff_j < argmax_tol)
        argmin_j = torch.argmax(rand_j, dim=1)   # (n,)

        maxmin = torch.nanmax(min_j)             # scalar
        diff_i = torch.abs(min_j - maxmin)
        rand_i = torch.rand(n, device=device) * (diff_i < argmax_tol)
        next_i = int(torch.argmax(rand_i))
        next_j = int(argmin_j[next_i])

        # Edge case: only *one* viable leader remains
        if leader_ok.sum() == 1:
            only = int(torch.nonzero(leader_ok).squeeze())
            return only, only
        return next_i, next_j
    return _selector
# ====  END  max-min-LCB closure =======================================
#Quasi-Random Sampler for continuous case
class CandidateSampler:
    """
    Draws quasi-random points and supports progressive shrinking.
    """
    def __init__(self, dim, method="sobol", adaptive=True, seed=0):
        self.dim, self.adaptive, self.seed = dim, adaptive, seed
        if method == "sobol":
            self._sampler = qmc.Sobol(d=dim, scramble=True, seed=seed)
            self._m = 3   # start with 2**3 = 8 points
        elif method == "lhs":
            self._sampler = qmc.LatinHypercube(d=dim, seed=seed)  #:contentReference[oaicite:2]{index=2}
            self._n = 16
        else:
            raise ValueError("method must be sobol|lhs")

    def next_batch(self):
        """Return a new set of points; size doubles each call if adaptive."""
        if hasattr(self, "_m"):
            pts = self._sampler.random_base2(m=self._m)
            if self.adaptive: self._m += 1                         # shrink ⇢ grow
        else:
            pts = self._sampler.random(n=self._n)
            if self.adaptive: self._n *= 2
        return torch.as_tensor(pts, dtype=torch.double)
#Viable points
# ============  BEGIN  ViableLeaderPool  =======================================
class ViableLeaderPool:
    """
    Maintains (i) a master Sobol/LHS grid  X_all  and
             (ii) a shrinking boolean mask  M_t  of viable leaders.
    """
    def __init__(self, sampler: CandidateSampler):
        self.sampler = sampler
        self.X_all: List[torch.Tensor] = []     # full grid
        self.M: Optional[torch.Tensor] = None   # bool mask  (|X_all|,)

    # -- public API ------------------------------------------------------------
    def extend_with(self, new_pts: torch.Tensor) -> None:
        """Append new points to X_all and grow the viability mask."""
        if not len(new_pts):
            return
        self.X_all.append(new_pts)
        new_mask = torch.ones(len(new_pts), dtype=torch.bool,
                              device=new_pts.device)
        self.M = new_mask if self.M is None else torch.cat([self.M, new_mask])

    def prune_by_ucb(
            self,
            h_mat: torch.Tensor,
            sigma_mat: torch.Tensor,
            beta: float,
            eps: float = 0.0,
    ) -> None:
        """
        Update M  ←  1{ μ_ij + β σ_ij ≥ ½ − ε  ∀j }
        """
        ucb = h_mat + beta * sigma_mat
        cond = torch.all(
            torch.logical_or(
                ucb > 0.5 - eps,
                torch.eye(ucb.shape[0], dtype=torch.bool, device=ucb.device)
            ), dim=1)
        self.M &= cond
# ============  END  ViableLeaderPool  =========================================


################################################################################
# Core class
################################################################################


class StackelbergBandit:
    """Implements Algorithm 1 of Pasztor *et al.* (2024) end‑to‑end.

    The main entry‑point is :py:meth:`run`, which executes *T* rounds and
    returns the entire history for downstream analysis.
    """

    def __init__(
            self,
            oracle: Callable[[torch.Tensor, torch.Tensor], int],
            config: StackelbergConfig,
            *,
            device: torch.device | str = "cpu",
            dtype: torch.dtype = torch.float32,
    ) -> None:
        self.oracle = oracle
        self.cfg = config
        self.device = torch.device(device)
        self.dtype = dtype
        self.dim = len(config.bounds)
        self.random_search_budget = config.random_search_budget
        # Pre‑compute bound tensors for fast sampling -------------------------
        self._bounds_low = torch.tensor([lo for lo, _ in config.bounds], device=self.device, dtype=self.dtype)
        self._bounds_high = torch.tensor([hi for _, hi in config.bounds], device=self.device, dtype=self.dtype)

        # Derived constants ----------------------------------------------------
        B_tensor = torch.tensor(config.B, device=self.device, dtype=self.dtype)
        self.kappa = 1.0 / (sigmoid(B_tensor) * (1.0 - sigmoid(B_tensor)))  # κ

        #   β(t) fall‑back ------------------------------------------------------
        if self.cfg.beta_schedule is None:
            self.cfg.beta_schedule = self._default_beta  # type: ignore

        # Mutable state --------------------------------------------------------
        self.history: List[Tuple[torch.Tensor, torch.Tensor, int]] = []  # (x,x',y)
        self.alpha: Optional[torch.Tensor] = None  # dual coefficients of h_t
        self.K: Optional[torch.Tensor] = None
        self.K_inv: Optional[torch.Tensor] = None  # (K + λκI)⁻¹  up-to-date
        self.sampler = CandidateSampler(dim=self.dim)
        self.pool = ViableLeaderPool(sampler=self.sampler)
        self._grid = None
        # create the closure ONCE (e.g. in __init__)
        self.select_pair = make_max_min_lcb(
            beta        = None,          # we pass β at call time, see below
            decision_buffer = self.cfg.decision_buffer,
            use_candidate_set = False,   # we handle M_t ourselves
            device = self.device,
            dtype  = self.dtype,
        )


    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def run(self, T: int) -> List[Tuple[torch.Tensor, torch.Tensor, int]]:
        """Execute *T* rounds of MAXMIN‑LCB and return the collected history."""

        # Pre‑initialisation (Alg. 1, lines 3–6) -----------------------------
        x0, x0p = self._sample_uniform(), self._sample_uniform()
        pref0 = self.oracle(x0, x0p)
        self.history.append((x0, x0p, pref0))
        self._append_kernel_row(x0, x0p)
        self._update_model()  # sets self.alpha and helpers seen below

        # Start the main loop (Alg. 1, lines 8–19) ---------------------------
        for t in range(1, T + 1):
            print(t)
            beta_t = self.cfg.beta_schedule(t)

            # — Leader step: choose x_t = argmax_x min_{x'} LCB_t(x,x')
            def follower_response(x: torch.Tensor) -> torch.Tensor:
                """Given *x*, return x' = argmin_{x'} LCB_t(x,x')."""
                return self._continuous_argmin(lambda xp: self._lcb(x, xp, beta_t))

            def leader_objective(x: torch.Tensor) -> float:
                if not self._in_M(x, beta_t):
                    return -1e9  # large negative — will be ignored by max
                xp = follower_response(x)
                return self._lcb(x, xp, beta_t).item()

            x_t = self._continuous_argmax(leader_objective)
            x_t_prime = follower_response(x_t)

            # — Query preference oracle and update state --------------------
            y_t = self.oracle(x_t, x_t_prime)
            self.history.append((x_t, x_t_prime, y_t))
            self._append_kernel_row(x_t, x_t_prime)
            self._update_model()  # recalculates α, σ_t^D, etc.

        return self.history, self.best_x()

    # ------------------------------------------------------------------
    # Candidate‑set predicate  M_t
    # ------------------------------------------------------------------

    def _in_M(self, x: torch.Tensor, beta_t: float) -> bool:
        """Monte‑Carlo test of the feasibility predicate for a leader x."""
        K = self.cfg.candidate_check_points
        # Sample K opponent points uniformly from bounds
        r = torch.rand((K, self.dim), device=self.device, dtype=self.dtype)
        xp = self._bounds_low + r * (self._bounds_high - self._bounds_low)
        # Vectorised σ(h)+βσᴰ ≥ ½−ε
        h_vals = torch.tensor([self._h(x, xp_i) for xp_i in xp], device=self.device, dtype=self.dtype)
        sig_h = sigmoid(h_vals)
        sig_d = torch.tensor([self._sigma_D(x, xp_i) for xp_i in xp], device=self.device, dtype=self.dtype)
        lhs = sig_h + beta_t * sig_d
        return bool(torch.all(lhs >= 0.5 - self.cfg.decision_buffer).item())

    # ------------------------------------------------------------------
    #  Internal helpers – maths
    # ------------------------------------------------------------------

    def _update_model(self) -> None:
        """Solve Eq.(6) over the entire history and cache dual coefficients α."""
        n = len(self.history)
        if n == 0:
            return                 # ← already up-to-date

        y = torch.tensor([p for *_, p in [(0, 0, 0)]], dtype=self.dtype)  # placeholder
        y = torch.tensor([pref for *_, pref in self.history], device=self.device, dtype=self.dtype)

        # Representer‑theorem form: h(⋅) = Σ α_i k_D(⋅, z_i)
        if self.alpha is None:
            alpha = torch.nn.Parameter(torch.zeros(n, device=self.device, dtype=self.dtype))
        else:
            #add 1 zero if alpha already exists to allow for warm restarts
            alpha = torch.nn.Parameter(torch.cat([self.alpha, torch.zeros(1, device=self.device, dtype=self.dtype)]))
        opt = torch.optim.LBFGS([alpha,], max_iter=5, line_search_fn="strong_wolfe")
        lam = self.cfg.lamb

        def closure() -> torch.Tensor:  # negative log‑likelihood + ridge term
            opt.zero_grad()
            logits = torch.mv(self.K, alpha)
            p = sigmoid(logits)
            nll = -(y * torch.log(p + 1e-9) + (1 - y) * torch.log(1 - p + 1e-9)).sum()
            rkhs = 0.5 * lam * torch.dot(alpha, torch.mv(self.K, alpha))
            loss = nll + rkhs
            loss.backward()
            return loss

        opt.step(closure)
        self.alpha = alpha.detach()

    # ..................................................................

    def _h(self, x: torch.Tensor, xp: torch.Tensor) -> torch.Tensor:
        """h_t(x,x') evaluated via the dual coefficients."""
        if self.alpha is None:
            return torch.tensor(0.0, device=self.device, dtype=self.dtype)
        k_vec = torch.stack([
            self._dueling_kernel(x, xp, xi, xpi) for xi, xpi, _ in self.history
        ])
        return torch.dot(self.alpha, k_vec)

    def _sigma_D(self, x: torch.Tensor, xp: torch.Tensor) -> torch.Tensor:
        """σ_t^D(x,x') as in Alg. 1 (square‑root of the posterior variance)."""
        k_xx = self._dueling_kernel(x, xp, x, xp)
        k_hist = torch.stack([
            self._dueling_kernel(x, xp, xi, xpi) for xi, xpi, _ in self.history
        ])
        var = k_xx - torch.dot(k_hist, torch.mv(self.K_inv, k_hist))
        return torch.sqrt(torch.clamp(var, min=1e-9))

    def _lcb(self, x: torch.Tensor, xp: torch.Tensor, beta_t: float) -> torch.Tensor:
        return sigmoid(self._h(x, xp)) - beta_t * self._sigma_D(x, xp)

    # ------------------------------------------------------------------
    #  Internal helpers – optimisation wrappers
    # ------------------------------------------------------------------

    def _continuous_argmax(self, fun: Callable[[torch.Tensor], float]) -> torch.Tensor:
        """Maximise *fun* over the hyper‑rectangle self.cfg.bounds."""
        return self._optim_wrapper(fun, maximise=True)

    def _continuous_argmin(self, fun: Callable[[torch.Tensor], float]) -> torch.Tensor:
        return self._optim_wrapper(fun, maximise=False)

    def _optim_wrapper(self, fun: Callable[[torch.Tensor], float], *, maximise: bool) -> torch.Tensor:
        """Select backend and run numeric optimisation in box‑constraints."""
        bounds = self.cfg.bounds
        dim = self.dim

        def _neg(x: np.ndarray) -> float:  # helper since SciPy minimises only
            tx = torch.from_numpy(x).type(self.dtype).to(self.device)
            val = fun(tx)
            if isinstance(val, torch.Tensor):
                val: float = val.item()
            else:
                val = float(val)
            return -val if maximise else val

        if self.cfg.optimizer_backend == "scipy" and optimize is not None:
            res = optimize.dual_annealing(
                _neg,
                bounds=bounds,
                seed=None,
            )
            best = res.x
        elif self.cfg.optimizer_backend == "nevergrad" and ng is not None:
            param = ng.p.Array(shape=(dim,)).set_bounds(*zip(*bounds))
            instr = ng.optimizers.OnePlusOne(param, budget=self.random_search_budget)
            best = instr.minimize(_neg).value
        else:  # fallback: pure random search
            best_val: float = float("inf")
            best_x: np.ndarray | None = None
            for _ in range(self.cfg.random_search_budget):
                rnd = np.array([np.random.uniform(low, high) for low, high in bounds])
                v = _neg(rnd)
                if v < best_val:
                    best_val, best_x = v, rnd
            best = best_x if best_x is not None else np.zeros(dim)

        best_t = torch.from_numpy(best).type(self.dtype).to(self.device, dtype=self.dtype)
        return best_t

    # ------------------------------------------------------------------
    #  Kernel utilities
    # ------------------------------------------------------------------

    def _dueling_kernel(self, x: torch.Tensor, xp: torch.Tensor, y: torch.Tensor, yp: torch.Tensor) -> torch.Tensor:
        k = self.cfg.kernel
        return k(x, y) + k(xp, yp) - k(x, yp) - k(xp, y)

    def _dueling_kernel_matrix(self) -> torch.Tensor:
        n = len(self.history)
        K = torch.empty((n, n), device=self.device, dtype=self.dtype)
        for i, (xi, xpi, _) in enumerate(self.history):
            for j, (xj, xpj, _) in enumerate(self.history):
                K[i, j] = self._dueling_kernel(xi, xpi, xj, xpj)
        return K #TODO update so k is a self parameter
    # ==== BEGIN incremental kernel update  ================================
    # ------------------------------------------------------------------
    #  Incremental kernel bookkeeping
    # ------------------------------------------------------------------

    def _update_kernel_inverse(
            self,
            k_new: torch.Tensor,
            kappa_new: float,
    ) -> None:
        """
        Rank-1 Sherman–Morrison update of (K + λκI)⁻¹.

        Parameters
        ----------
        k_new : (n,)  kernel column between the *new* pair (x,x') and the
                      previous n pairs.
        kappa_new :   k_D((x,x'),(x,x'))  — self-kernel of the new pair.
        """
        if self.K_inv is None:                           # first datum
            self.K_inv = torch.tensor([[1.0 / (kappa_new + self.cfg.lamb *
                                               self.kappa)]],
                                      dtype=self.dtype, device=self.device)
            return

        # Schur-complement update
        v = self.K_inv @ k_new.unsqueeze(1)              # (n,1)
        s = (kappa_new + self.cfg.lamb * self.kappa) - (k_new @ v).item()
        s = max(s, 1e-8)                                 # numerical guard
        top_left  = self.K_inv + (v @ v.T) / s
        top_right = -v / s
        bottom_left  = -v.T / s
        bottom_right = torch.tensor([[1.0 / s]],
                                    dtype=self.dtype, device=self.device)
        self.K_inv = torch.cat([torch.cat([top_left,  top_right],  dim=1),
                                torch.cat([bottom_left, bottom_right], dim=1)],
                               dim=0)

    def _append_kernel_row(self, x: torch.Tensor, xp: torch.Tensor) -> None:
        """
        Extend self.K (full kernel matrix) and self.K_inv in-place with the
        dueling-kernel values for the newly observed pair (x,x').
        """
        if self.K is None:                   # first point
            self.K = torch.tensor([[0.0]],  # dummy, overwritten
                                  dtype=self.dtype,
                                  device=self.device)

        k_new = torch.stack([
            self._dueling_kernel(x, xp, xi, xpi)
            for xi, xpi, _ in self.history[:-1]          # n previous rows
        ]) if len(self.history) > 1 else torch.empty(0, device=self.device)

        kappa_new = self._dueling_kernel(x, xp, x, xp).item()

        # grow the *dense* matrix (cheap for ≤2 000 pts)
        if k_new.numel():
            top = torch.cat([self.K, k_new.unsqueeze(1)], dim=1)
            bottom = torch.cat([k_new, torch.tensor([kappa_new],
                                                    dtype=self.dtype,
                                                    device=self.device)])
            self.K = torch.cat([top,
                                bottom.unsqueeze(0)], dim=0)
        else:                                            # very first row
            self.K[0, 0] = kappa_new

        # finally update the inverse
        self._update_kernel_inverse(k_new, kappa_new)
    # ==== END incremental kernel update  ==================================


    # ------------------------------------------------------------------
    #  Misc helpers
    # ------------------------------------------------------------------

    def _default_beta(self, t: int) -> float:
        """Closed‑form exploration coefficient (Theorem 2)."""
        """
        Theory:
         γₜ = max_(x_1...x_t) ½ log det(I + (λkappa)⁻¹Kₜ).
        0 <= k(x, x') <= 1, so k^D(pair1, pair2) <= 1 + 1 - 0 - 0 = 2
        AND k^D >= 0 + 0 - 1 - 1 = -2
        K_t is symmetric, so it must diagonalize and thus it's determinant is equal to the product of the eigenvalues
        By the Gershgorin circle theorem, the eigenvalues of a symmetric matrix are real and contained inside intervals on the real line
        centered around the diagonal entries with radius equal to row sums. As each entry <= 2 and -2 <= k^D <= 2, our maximal balls have
        radius = (t-1)*2 and are centered around -2 and 2. Since we are maximizing, we take the ball centered around 2, and conclude that our
        eigenvalues must be bounded by 2 + (t-1)*2 = 2t. Since we are dividing by lambda * kappa, this turns into 2t/(lambda kappa) post-diagonalization
        """
        gamma_t = 0.5 * t * math.log(1 + 2*t / (self.cfg.lamb * self.kappa))  # crude upper bound
        L = 0.25  # Lipschitz of sigmoid on [‑B,B] – worst‑case = 0.25 but shorter
        term = 4 * L * self.cfg.B + 2 * L * math.sqrt(self.kappa / self.cfg.lamb) * math.sqrt(2 * math.log(t + 1) + 2 * gamma_t)
        return term

    def _sample_uniform(self) -> torch.Tensor:
        low = torch.tensor([b[0] for b in self.cfg.bounds], device=self.device, dtype=self.dtype)
        high = torch.tensor([b[1] for b in self.cfg.bounds], device=self.device, dtype=self.dtype)
        r = torch.rand(self.dim, device=self.device, dtype=self.dtype)
        return low + r * (high - low)

    # ==== BEGIN best-x helper  ============================================
    # ------------------------------------------------------------------
    #  Final recommendation after T rounds
    # ------------------------------------------------------------------

    def best_x(self) -> torch.Tensor:
        """
        Return the optimistic leader \hat{x}_T := argmax_x  \hat{P}(x wins).

        Here we use a simple empirical-Bayes score:
            win_rate(x) = (# times x won) / (# times x played as leader)
        tying is broken by picking the *most recent* of the best.
        """
        counts, wins, last_seen = {}, {}, {}
        for t, (x, _, y) in enumerate(self.history):
            key = x.tobytes()
            counts[key] = counts.get(key, 0) + 1
            wins[key]   = wins.get(key,   0) + int(y)
            last_seen[key] = t

        scores = {k: wins[k] / max(counts[k], 1) for k in counts}
        # argmax with "latest-seen" tie-break
        best_key = max(scores, key=lambda k: (scores[k], last_seen[k]))
        return torch.frombuffer(best_key, dtype=self.dtype).to(self.device)
    # ==== END best-x helper  ==============================================


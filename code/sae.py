"""
TopK Sparse Autoencoder for document embeddings.

This is a compact, CPU-friendly implementation of the TopK SAE described in
Gao et al., "Scaling and evaluating sparse autoencoders" (the
`sparse_autoencoders.pdf` in this repo). The key ingredients we reuse:

  * A linear encoder/decoder around a pre-encoder bias `b_pre` that is tied to
    the data mean (Section 2.1 / 2.2).
  * A `TopK` activation that keeps only the k largest pre-activations and zeros
    the rest, which directly controls sparsity (no L1 penalty / tuning needed).
  * Unit-norm decoder rows (each latent has a unit-length "direction" in
    embedding space) so that feature activations are comparable.
  * An auxiliary "AuxK" loss that revives dead latents by asking the currently
    dead units to reconstruct the residual error (Section 2.4, "Preventing dead
    latents").

For our extension the SAE is trained on *document embeddings* (e.g. 384-d
all-MiniLM or 768-d mpnet vectors), not on language-model activations, but the
recipe is identical. The trained decoder rows `W_dec[j]` are interpretable
"directions"; the encoder activations tell us which documents excite each
latent, which is exactly what we need to discover latent confounders.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKSAE(nn.Module):
    """A single-layer TopK sparse autoencoder.

    Parameters
    ----------
    d_in : int
        Dimension of the input embeddings.
    n_latents : int
        Number of dictionary features (usually an expansion factor times d_in).
    k : int
        Number of active latents per example (the TopK sparsity level).
    aux_k : int, optional
        Number of dead latents used by the AuxK auxiliary loss. Defaults to
        2 * k. Set to 0 to disable dead-latent revival.
    dead_steps_threshold : int
        A latent that has not fired for this many optimization steps is
        considered "dead" and becomes eligible for the AuxK loss. `train_sae`
        sets this from `dead_epochs`, so that "dead" means the same thing on
        every corpus regardless of how many batches an epoch holds.
    """

    def __init__(
        self,
        d_in: int,
        n_latents: int,
        k: int,
        aux_k: int | None = None,
        dead_steps_threshold: int = 200,
    ):
        super().__init__()
        self.d_in = d_in
        self.n_latents = n_latents
        self.k = min(k, n_latents)
        self.aux_k = (2 * self.k) if aux_k is None else aux_k
        self.dead_steps_threshold = dead_steps_threshold

        # b_pre is subtracted before encoding and added back after decoding.
        # It is initialised to the data mean by `init_from_data`.
        self.b_pre = nn.Parameter(torch.zeros(d_in))
        self.b_enc = nn.Parameter(torch.zeros(n_latents))

        # Decoder rows are unit-norm "feature directions"; encoder is tied to the
        # decoder at init (a standard, well-behaved initialisation).
        W_dec = torch.randn(n_latents, d_in)
        W_dec = W_dec / W_dec.norm(dim=1, keepdim=True)
        self.W_dec = nn.Parameter(W_dec)
        self.W_enc = nn.Parameter(W_dec.t().clone())  # (d_in, n_latents)

        # Steps since each latent last fired (for dead-latent detection).
        self.register_buffer("steps_since_fired", torch.zeros(n_latents))

    # ------------------------------------------------------------------ #
    # Initialisation helpers
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def init_from_data(self, X: torch.Tensor) -> None:
        """Tie the pre-encoder bias to the data mean (geometric median proxy)."""
        self.b_pre.data = X.mean(dim=0)

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        """Project decoder rows back to the unit sphere after each step."""
        self.W_dec.data = self.W_dec.data / (
            self.W_dec.data.norm(dim=1, keepdim=True) + 1e-8
        )

    # ------------------------------------------------------------------ #
    # Forward / encode
    # ------------------------------------------------------------------ #
    def preactivations(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.b_pre) @ self.W_enc + self.b_enc

    def topk_encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the sparse latent activations (TopK of the pre-activations)."""
        pre = self.preactivations(x)
        values, indices = pre.topk(self.k, dim=-1)
        values = F.relu(values)  # keep activations non-negative
        z = torch.zeros_like(pre)
        z.scatter_(-1, indices, values)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W_dec + self.b_pre

    def forward(self, x: torch.Tensor):
        pre = self.preactivations(x)
        values, indices = pre.topk(self.k, dim=-1)
        values = F.relu(values)
        z = torch.zeros_like(pre)
        z.scatter_(-1, indices, values)
        x_hat = self.decode(z)
        return x_hat, z, pre

    # ------------------------------------------------------------------ #
    # AuxK auxiliary loss (dead-latent revival)
    # ------------------------------------------------------------------ #
    def auxk_loss(self, x: torch.Tensor, x_hat: torch.Tensor, pre: torch.Tensor):
        """Ask the currently-dead latents to model the reconstruction residual.

        Mirrors the AuxK term in Gao et al.: among latents that have not fired
        recently, take the aux_k largest pre-activations and use *only* those to
        reconstruct the residual (x - x_hat). This gives dead latents a gradient
        and keeps the dictionary fully utilised.
        """
        if self.aux_k <= 0:
            return torch.tensor(0.0, device=x.device)

        dead = self.steps_since_fired >= self.dead_steps_threshold
        if dead.sum() == 0:
            return torch.tensor(0.0, device=x.device)

        residual = x - x_hat.detach()
        pre_dead = pre.clone()
        pre_dead[:, ~dead] = -float("inf")  # restrict to dead latents
        n_aux = int(min(self.aux_k, int(dead.sum().item())))
        values, indices = pre_dead.topk(n_aux, dim=-1)
        values = F.relu(values)
        z_aux = torch.zeros_like(pre)
        z_aux.scatter_(-1, indices, values)
        residual_hat = z_aux @ self.W_dec
        return F.mse_loss(residual_hat, residual)

    @torch.no_grad()
    def update_dead_stats(self, z: torch.Tensor) -> None:
        fired = (z > 0).any(dim=0)
        self.steps_since_fired += 1
        self.steps_since_fired[fired] = 0


def train_sae(
    X: np.ndarray | torch.Tensor,
    n_latents: int | None = None,
    expansion_factor: int = 8,
    k: int = 32,
    aux_k: int | None = None,
    aux_coef: float = 1.0 / 32.0,
    lr: float = 1e-3,
    n_epochs: int = 200,
    batch_size: int = 1024,
    device: str | None = None,
    seed: int = 0,
    verbose: bool = True,
    dead_epochs: int = 20,
):
    """Train a TopK SAE on a matrix of embeddings.

    `dead_epochs` defines a dead latent as one that has not fired for that many
    **epochs** (full passes over the corpus). The threshold is converted to
    optimizer steps from the corpus size, so it means the same thing on every
    corpus: with the old fixed 200-step threshold a 700-document corpus (one
    batch per epoch) never revived anything in 200 epochs while a 5,000-document
    corpus (five batches per epoch) started reviving after 40, and the reported
    `frac_dead` was not comparable across corpora either.

    Returns
    -------
    sae : TopKSAE
        The trained model (on CPU, in eval mode).
    info : dict
        Training diagnostics: final reconstruction R^2, fraction of variance
        explained, fraction of dead latents, and the loss history.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)
    X = X.float()
    d_in = X.shape[1]
    n_latents = n_latents or expansion_factor * d_in
    steps_per_epoch = max(1, int(np.ceil(X.shape[0] / batch_size)))
    dead_steps = max(1, int(dead_epochs) * steps_per_epoch)

    sae = TopKSAE(d_in, n_latents, k=k, aux_k=aux_k,
                  dead_steps_threshold=dead_steps).to(device)
    sae.init_from_data(X.to(device))
    opt = torch.optim.Adam(sae.parameters(), lr=lr)

    X_dev = X.to(device)
    n = X_dev.shape[0]
    total_var = (X_dev - X_dev.mean(0)).pow(2).sum().item()
    history = []

    for epoch in range(n_epochs):
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            xb = X_dev[idx]

            x_hat, z, pre = sae(xb)
            recon = F.mse_loss(x_hat, xb)
            aux = sae.auxk_loss(xb, x_hat, pre)
            loss = recon + aux_coef * aux

            opt.zero_grad()
            loss.backward()
            opt.step()
            sae.normalize_decoder()
            sae.update_dead_stats(z)
            epoch_loss += recon.item() * xb.shape[0]

        history.append(epoch_loss / n)
        if verbose and (epoch % 20 == 0 or epoch == n_epochs - 1):
            with torch.no_grad():
                x_hat, z, _ = sae(X_dev)
                resid = (X_dev - x_hat).pow(2).sum().item()
                r2 = 1.0 - resid / total_var
                dead = (sae.steps_since_fired >= sae.dead_steps_threshold).float().mean().item()
            print(
                f"[SAE] epoch {epoch:3d}  recon_mse={history[-1]:.5f}  "
                f"R^2={r2:.3f}  dead={dead:.1%}"
            )

    sae = sae.to("cpu").eval()
    with torch.no_grad():
        x_hat, z, _ = sae(X.cpu())
        resid = (X.cpu() - x_hat).pow(2).sum().item()
        r2 = 1.0 - resid / total_var
        dead = (sae.steps_since_fired.cpu() >= sae.dead_steps_threshold).float().mean().item()
    info = {
        "r2": r2,
        "frac_dead": dead,
        "loss_history": history,
        "n_latents": n_latents,
        "k": k,
        "dead_epochs": int(dead_epochs),
        "dead_steps_threshold": int(dead_steps),
    }
    if verbose:
        print(f"[SAE] done. R^2={r2:.3f}  dead_latents={dead:.1%}")
    return sae, info


@torch.no_grad()
def encode_features(sae: TopKSAE, X: np.ndarray | torch.Tensor) -> np.ndarray:
    """Return the (n_docs, n_latents) sparse activation matrix for X."""
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)
    z = sae.topk_encode(X.float().cpu())
    return z.numpy()


@torch.no_grad()
def encode_preactivations(sae: TopKSAE, X: np.ndarray | torch.Tensor) -> np.ndarray:
    """Return the (n_docs, n_latents) *pre*-activation matrix for X.

    Unlike the TopK activations (exactly zero for ~97% of doc/latent entries),
    pre-activations are dense, so ranking documents by *lowest* pre-activation
    gives a meaningful "most anti-activating" set — the right contrast for
    interpreting a latent's bottom examples (the TopK zeros are an arbitrary
    tie-break and say nothing about which documents oppose the feature).
    """
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)
    return sae.preactivations(X.float().cpu()).numpy()

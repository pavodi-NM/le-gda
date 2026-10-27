"""Ledoit-Wolf shrinkage helpers, used by the configuration-transfer ablation
(scripts/run_config_transfer.py) to set each space's ridge in closed form from
the support set, with no validation data.
"""

import torch
import torch.nn.functional as F


def lw_intensity(R):
    """Ledoit-Wolf shrinkage intensity toward (tr S / D) I, from the support set.

    Sigma_LW = (1-rho) S + rho m I with m = tr(S)/D. Our ridge family is
    S + lambda m I, and Sigma_LW = (1-rho)[S + (rho/(1-rho)) m I], so the two
    coincide at lambda = rho/(1-rho); the leading (1-rho) is a global scale that
    the fusion weight alpha absorbs.

    The b_bar^2 sum is computed in closed form rather than by looping over
    samples: sum_i ||x_i x_i^T - S||_F^2 = sum_i ||x_i||^4 - N ||S||_F^2, using
    sum_i x_i^T S x_i = N ||S||_F^2.

    Returns (rho, lambda).
    """
    N, D = R.shape
    S = R.t() @ R / N
    m = S.diagonal().mean()
    s_fro2 = (S ** 2).sum()
    d2 = (s_fro2 - D * m ** 2) / D
    quartic = ((R ** 2).sum(1) ** 2).sum()
    b2 = ((quartic - N * s_fro2) / (N ** 2 * D)).clamp_min(0.0)
    b2 = torch.minimum(b2, d2.clamp_min(1e-30))
    rho = (b2 / d2.clamp_min(1e-30)).clamp(0.0, 1.0 - 1e-6)
    return float(rho), float(rho / (1.0 - rho))


def residuals(ftr, ytr, n_cls):
    """Within-class residuals exactly as gda_weights builds them.

    gda_weights centres each class on its L2-normalised mean, so the residuals
    are computed the same way here; otherwise the Ledoit-Wolf estimate would
    describe a different covariance from the one the classifier inverts.
    """
    res = []
    for c in range(n_cls):
        Xc = ftr[ytr == c]
        res.append(Xc - F.normalize(Xc.mean(0), dim=-1))
    return torch.cat(res, 0)

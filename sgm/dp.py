"""Patient-level DP-SGD and a Renyi-DP accountant.

Privacy unit: one patient (all windows of that patient held by a site).
Mechanism per step at site k:
    - Poisson-sample patients with rate q_k,
    - per-patient gradient g_p = sum of the gradients of that patient's sampled
      windows, clipped to L2 norm C over *all* trainable parameters,
    - noisy sum  sum_p clip(g_p) + N(0, sigma^2 C^2 I_P),
    - divide by the expected batch size q_k * n_k (data-independent).
This is the sampled Gaussian mechanism; its RDP is computed exactly for integer
orders (Mironov, Talwar & Zhang 2019) and composed over every step, local epoch
and round the site runs.  Conversion to (eps, delta) uses the bound of
Balle et al. (2020), as in Opacus.
"""
import math

import numpy as np
import torch
from torch.func import functional_call, grad, vmap

ORDERS = list(range(2, 257))


def _log_comb(n, k):
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def rdp_sampled_gaussian(q, sigma, orders=ORDERS):
    """RDP epsilon(alpha) of one step of the sampled Gaussian mechanism."""
    out = []
    for a in orders:
        if q == 0:
            out.append(0.0)
            continue
        if q == 1.0:
            out.append(a / (2 * sigma ** 2))
            continue
        terms = [
            _log_comb(a, k) + (a - k) * math.log1p(-q) + k * math.log(q)
            + (k * k - k) / (2 * sigma ** 2)
            for k in range(a + 1)
        ]
        m = max(terms)
        log_a = m + math.log(sum(math.exp(t - m) for t in terms))
        out.append(log_a / (a - 1))
    return np.array(out)


def eps_from_rdp(rdp, delta, orders=ORDERS):
    orders = np.asarray(orders, dtype=float)
    eps = rdp - (np.log(delta) + np.log(orders)) / (orders - 1) + np.log((orders - 1) / orders)
    i = int(np.nanargmin(eps))
    return float(eps[i]), float(orders[i])


def epsilon(q, sigma, steps, delta):
    return eps_from_rdp(steps * rdp_sampled_gaussian(q, sigma), delta)[0]


def calibrate_sigma(q, steps, target_eps, delta, lo=0.3, hi=50.0, tol=1e-3):
    if epsilon(q, hi, steps, delta) > target_eps:
        raise ValueError("target epsilon unreachable with sigma <= %.1f" % hi)
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if epsilon(q, mid, steps, delta) > target_eps:
            lo = mid
        else:
            hi = mid
    return hi


class Accountant:
    """Tracks the RDP of every site separately (sites hold disjoint patients,
    so a patient's privacy loss is that of the one site holding them)."""

    def __init__(self, delta):
        self.delta = delta
        self.rdp = {}

    def step(self, site, q, sigma, n=1):
        self.rdp[site] = self.rdp.get(site, 0.0) + n * rdp_sampled_gaussian(q, sigma)

    def spent(self):
        per_site = {k: eps_from_rdp(v, self.delta)[0] for k, v in self.rdp.items()}
        return max(per_site.values()) if per_site else 0.0, per_site


def per_sample_grads(model, loss_fn, batch):
    """Per-sample gradients via torch.func; `loss_fn(model_out, sample)` must
    return the scalar loss of ONE sample."""
    params = {k: v.detach() for k, v in model.named_parameters() if v.requires_grad}
    buffers = {k: v.detach() for k, v in model.named_buffers()}

    def one(p, b, sample):
        s = {k: v.unsqueeze(0) for k, v in sample.items()}
        out = functional_call(model, (p, b), (s,))
        return loss_fn(out, s)

    g = vmap(grad(one), in_dims=(None, None, 0), randomness="different")(params, buffers, batch)
    return g


def dp_patient_step(model, loss_fn, batch, patient_index, n_patients_in_batch,
                    clip, sigma, expected_batch, generator=None):
    """Returns the privatised gradient (dict name->tensor) for one DP step.

    patient_index: LongTensor [B] mapping each sample to 0..n_patients_in_batch-1.
    """
    params = {k: v for k, v in model.named_parameters() if v.requires_grad}
    if n_patients_in_batch == 0:
        per_patient = None
    else:
        g = per_sample_grads(model, loss_fn, batch)
        per_patient = {}
        for k, v in g.items():
            acc = torch.zeros((n_patients_in_batch,) + v.shape[1:], dtype=v.dtype, device=v.device)
            per_patient[k] = acc.index_add_(0, patient_index, v)
        sq = sum(v.flatten(1).pow(2).sum(1) for v in per_patient.values())
        factor = (clip / (sq.sqrt() + 1e-12)).clamp(max=1.0)
    out = {}
    for k, p in params.items():
        s = torch.zeros_like(p) if per_patient is None else \
            (per_patient[k] * factor.view(-1, *[1] * (per_patient[k].dim() - 1))).sum(0)
        noise = torch.randn(p.shape, generator=generator, device=p.device) * sigma * clip
        out[k] = (s + noise) / expected_batch
    return out

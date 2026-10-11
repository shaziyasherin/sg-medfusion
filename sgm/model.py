"""SG-MedFusion model (R1).  Shapes: B batch, W minutes, L = 60*hz*W grid
samples, d model width, F_m = 12 EHR channels (4 vitals, 4 masks, 4 device
flags); grid channels: standardised log power, asinh(first difference / 5 W).

  X_g [B,L,2] --GridEncoder--> H_g [B,W,d] --2x SelfBlock--> H_g
  X_m [B,W,12] --Linear+pos--> H_m [B,W,d] --2x SelfBlock--> H_m
  match-predict:  c_g_hat = MLP(H_m) [B,W]      (grid envelope from EHR)
                  c_m_hat = MLP(H_g) [B,W,4]    (device flags from grid)
                  r = log1p([mse_g, mse_m])  [B,2]  (detached residuals)
  fusion (xattn): H_g' = H_g + MHA(H_g -> H_m), H_m' = H_m + MHA(H_m -> H_g)
                  z = [mean_t H_g' ; mean_t H_m']  [B,2d]
  access head:    s = MLP([z ; r])  [B]   (logit of "inconsistent")
  forecast head:  y_hat = MLP(z)    [B,2] (HR, MAP at tau + 60 min)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MHA(nn.Module):
    def __init__(self, d, h, noise_sigma=0.0):
        super().__init__()
        self.h, self.dh, self.noise_sigma = h, d // h, noise_sigma
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))

    def forward(self, xq, xkv):
        B, Tq, d = xq.shape
        Tk = xkv.shape[1]
        q = self.q(xq).view(B, Tq, self.h, self.dh).transpose(1, 2)
        k = self.k(xkv).view(B, Tk, self.h, self.dh).transpose(1, 2)
        v = self.v(xkv).view(B, Tk, self.h, self.dh).transpose(1, 2)
        logits = q @ k.transpose(-1, -2) / math.sqrt(self.dh)          # [B,h,Tq,Tk]
        if self.noise_sigma > 0 and self.training:
            logits = logits + self.noise_sigma * torch.randn_like(logits)
        a = logits.softmax(-1)
        return self.o((a @ v).transpose(1, 2).reshape(B, Tq, d))


class Block(nn.Module):
    """Pre-norm transformer block; cross-attention when kv is given."""

    def __init__(self, d, h, mult, p, noise_sigma=0.0):
        super().__init__()
        self.n1, self.nkv, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = MHA(d, h, noise_sigma)
        self.ff = nn.Sequential(nn.Linear(d, mult * d), nn.GELU(), nn.Linear(mult * d, d))
        self.drop = nn.Dropout(p)

    def forward(self, x, kv=None):
        q = self.n1(x)
        kv = q if kv is None else self.nkv(kv)
        x = x + self.drop(self.att(q, kv))
        return x + self.drop(self.ff(self.n2(x)))


class GridEncoder(nn.Module):
    def __init__(self, d, W):
        super().__init__()
        c = d // 2
        self.net = nn.Sequential(
            nn.Conv1d(2, c, 7, stride=4, padding=3), nn.GELU(), nn.GroupNorm(1, c),
            nn.Conv1d(c, d, 5, stride=3, padding=2), nn.GELU(), nn.GroupNorm(1, d),
            nn.Conv1d(d, d, 3, stride=5, padding=1), nn.GELU(), nn.GroupNorm(1, d),
            nn.AdaptiveAvgPool1d(W))

    def forward(self, x):                      # [B,L,1] -> [B,W,d]
        return self.net(x.transpose(1, 2)).transpose(1, 2)


def mlp(i, o, d):
    return nn.Sequential(nn.Linear(i, d), nn.GELU(), nn.Linear(d, o))


class SGMedFusion(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        m = cfg["model"]
        d, h, W = m["d"], m["heads"], cfg["window_minutes"]
        self.m = m
        self.use_grid, self.use_ehr = m["use_grid"], m["use_ehr"]
        self.use_cycle = m["use_cycle"] and self.use_grid and self.use_ehr
        self.xattn = m["fusion"] == "xattn" and self.use_grid and self.use_ehr
        blk = lambda s=0.0: Block(d, h, m["ffn_mult"], m["dropout"], s)
        if self.use_grid:
            self.genc = GridEncoder(d, W)
            self.gpos = nn.Parameter(torch.zeros(1, W, d))
            self.gsa = nn.ModuleList([blk() for _ in range(m["sa_layers"])])
        if self.use_ehr:
            self.menc = nn.Linear(12, d)
            self.mpos = nn.Parameter(torch.zeros(1, W, d))
            self.msa = nn.ModuleList([blk() for _ in range(m["sa_layers"])])
        if self.xattn:
            self.g2m = blk(m["attn_noise_sigma"])
            self.m2g = blk(m["attn_noise_sigma"])
        if self.use_cycle:
            self.pred_cg = mlp(d, 1, d)
            self.pred_cm = mlp(d, 4, d)
        zd = d * (int(self.use_grid) + int(self.use_ehr))
        self.acc = mlp(zd + (2 if self.use_cycle else 0), 1, d)
        self.fc = mlp(zd, 2, d)
        for name in ("gpos", "mpos"):
            if hasattr(self, name):
                nn.init.normal_(getattr(self, name), std=0.02)

    def forward(self, b):
        out = {}
        if self.use_grid:
            hg = self.genc(b["xg"]) + self.gpos
            for layer in self.gsa:
                hg = layer(hg)
        if self.use_ehr:
            hm = self.menc(b["xm"]) + self.mpos
            for layer in self.msa:
                hm = layer(hm)
        if self.use_cycle:
            out["cg_hat"] = self.pred_cg(hm).squeeze(-1)
            out["cm_hat"] = self.pred_cm(hg)
            eg = (out["cg_hat"] - b["cg"]).pow(2).mean(1)
            em = (out["cm_hat"] - b["cm"]).pow(2).mean((1, 2))
            out["resid"] = torch.log1p(torch.stack([eg, em], 1)).detach()
        if self.xattn:
            hg, hm = self.g2m(hg, hm), self.m2g(hm, hg)
        zs = []
        if self.use_grid:
            zs.append(hg.mean(1))
        if self.use_ehr:
            zs.append(hm.mean(1))
        z = torch.cat(zs, 1)
        a_in = torch.cat([z, out["resid"]], 1) if self.use_cycle else z
        out["logit"] = self.acc(a_in).squeeze(-1)
        out["yhat"] = self.fc(z)
        return out


def loss_terms(out, b, cfg):
    """Per-sample losses (vector [B]).  Forecast and match-predict terms are
    computed on consistent pairs only (y = 0)."""
    lc = cfg["loss"]
    cons = 1.0 - b["y"]
    l_acc = F.binary_cross_entropy_with_logits(out["logit"], b["y"], reduction="none")
    m = b["ypred_mask"]
    l_pred = cons * ((out["yhat"] - b["ypred"]).pow(2) * m).sum(1) / m.sum(1).clamp(min=1)
    total = l_acc + lc["alpha_pred"] * l_pred
    parts = {"acc": l_acc, "pred": l_pred}
    if "cg_hat" in out:
        l_cyc = cons * ((out["cg_hat"] - b["cg"]).pow(2).mean(1)
                        + (out["cm_hat"] - b["cm"]).pow(2).mean((1, 2)))
        total = total + lc["lambda_cyc"] * l_cyc
        parts["cyc"] = l_cyc
    return total, parts


def sample_loss_fn(cfg, flip_labels=False):
    def f(out, b):
        if flip_labels:
            b = dict(b, y=1.0 - b["y"])
        return loss_terms(out, b, cfg)[0].sum()
    return f

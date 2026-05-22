import argparse
import copy
import gc
import math
import random
import sys
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import Data

try:
    from torch_geometric.loader import NeighborLoader
except ImportError:
    NeighborLoader = None
from torch_geometric.datasets import Planetoid, WikipediaNetwork, Actor, WebKB
from torch_geometric.nn import APPNP, LabelPropagation
from torch_geometric.nn.models import CorrectAndSmooth
from torch_geometric.utils import add_remaining_self_loops, degree, remove_self_loops, softmax, to_undirected
from tqdm import tqdm

try:
    from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
except ImportError:
    Evaluator = None
    PygNodePropPredDataset = None


def nan_to_num_safe(x: Tensor, nan: float=0.0, posinf: float=0.0, neginf: float=0.0) -> Tensor:
    if torch.is_complex(x):
        xr = torch.view_as_real(x)
        xr = torch.nan_to_num(xr, nan=nan, posinf=posinf, neginf=neginf)
        return torch.view_as_complex(xr)
    return torch.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)

def cross_entropy_with_label_smoothing(pred: Tensor, target: Tensor, smoothing: float=0.0) -> Tensor:
    if smoothing <= 0.0:
        return F.cross_entropy(pred, target)
    n_class = pred.size(1)
    log_probs = F.log_softmax(pred, dim=1)
    with torch.no_grad():
        true_dist = torch.zeros_like(log_probs)
        true_dist.fill_(smoothing / max(1, n_class - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), 1.0 - smoothing)
    return torch.mean(torch.sum(-true_dist * log_probs, dim=1))

def dropedge(edge_index: Tensor, p: float, training: bool) -> Tensor:
    if p <= 0.0 or not training:
        return edge_index
    e = edge_index.size(1)
    if e == 0:
        return edge_index
    keep = torch.rand(e, device=edge_index.device) > p
    return edge_index[:, keep]

def collect_provided_args(argv: List[str]) -> set:
    provided = set()
    for item in argv:
        if not item.startswith('--'):
            continue
        key = item.split('=', 1)[0][2:]
        if key.startswith('no-'):
            key = key[3:]
        if key.startswith('no_'):
            key = key[3:]
        provided.add(key.replace('-', '_'))
    return provided

def set_if_unprovided(args, provided: set, name: str, value) -> None:
    if name not in provided:
        setattr(args, name, value)



def set_seed(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    # Keep this deterministic enough for Planetoid/Cora comparisons.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def is_homophilic_builtin(data_id: int) -> bool:
    return int(data_id) in {0, 1, 2}

def inverse_sigmoid(x: float) -> float:
    x = min(max(float(x), 1e-06), 1.0 - 1e-06)
    return math.log(x / (1.0 - x))

def js_consistency(logits1: Tensor, logits2_detached: Tensor, T: float=2.0) -> Tensor:
    p1 = F.softmax(logits1 / T, dim=-1)
    p2 = F.softmax(logits2_detached / T, dim=-1)
    m = 0.5 * (p1 + p2)
    js = 0.5 * (F.kl_div((p1 + 1e-12).log(), m, reduction='batchmean') + F.kl_div((p2 + 1e-12).log(), m, reduction='batchmean'))
    return T * T * js

def complex_linear(x: Tensor, W: Tensor, b: Optional[Tensor]=None) -> Tensor:
    y = x @ W.transpose(0, 1)
    if b is not None:
        y = y + b
    return y

class ComplexLinear(nn.Module):

    def __init__(self, in_features: int, out_features: int, bias: bool=True, dtype=torch.cfloat):
        super().__init__()
        self.W = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
        self.bias = nn.Parameter(torch.empty(out_features, dtype=dtype)) if bias else None
        self.out_features = out_features
        self.reset_parameters()

    def reset_parameters(self) -> None:
        fan_in = self.W.size(1)
        scale = 1.0 / math.sqrt(max(1, fan_in))
        with torch.no_grad():
            self.W.real.uniform_(-scale, scale)
            self.W.imag.uniform_(-scale, scale)
            if self.bias is not None:
                self.bias.real.zero_()
                self.bias.imag.zero_()

    def forward(self, x: Tensor) -> Tensor:
        return complex_linear(x, self.W, self.bias)

class ModReLU(nn.Module):

    def __init__(self, features: int):
        super().__init__()
        self.b = nn.Parameter(torch.zeros(features, dtype=torch.float))

    def forward(self, z: Tensor) -> Tensor:
        mag = torch.abs(z).clamp_min(1e-06)
        gated = F.relu(mag + self.b)
        return nan_to_num_safe(gated * (z / mag))

class PhasePreservingNodeNorm(nn.Module):

    def __init__(self, eps: float=1e-05):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        if torch.is_complex(x):
            rms2 = (x.real.square() + x.imag.square()).mean(dim=-1, keepdim=True)
            out = x * torch.rsqrt(rms2 + self.eps)
            return nan_to_num_safe(out)
        mean = x.mean(dim=-1, keepdim=True)
        var = (x - mean).square().mean(dim=-1, keepdim=True)
        return torch.nan_to_num((x - mean) * torch.rsqrt(var + self.eps))



class LegacyNodeNorm(nn.Module):
    """Node-wise normalization used by the non-adaptive GESC baseline.

    This intentionally differs from PhasePreservingNodeNorm: it normalizes the
    real/imaginary channels over feature dimensions per node, matching the
    attached non-adaptive GESC implementation.
    """

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        if torch.is_complex(x):
            xr = torch.view_as_real(x)
            mean = xr.mean(dim=-2, keepdim=True)
            std = xr.std(dim=-2, keepdim=True).clamp_min(self.eps)
            xn = (xr - mean) / std
            return torch.view_as_complex(torch.nan_to_num(xn))
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True).clamp_min(self.eps)
        return torch.nan_to_num((x - mean) / std)


class GESCSoftmaxLayer(nn.Module):
    """Non-adaptive GESC/GET-SIC layer from the attached strong baseline.

    The key difference from ASCiiLayer is that the SIC coefficient is fixed
    per layer (`sic_strength`) instead of being edge-adaptive. This is useful
    for Cora, where the simpler baseline was reported to work better.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        gamma: float = 1.0,
        use_bias: bool = False,
        attn_dropout: float = 0.5,
        use_activation: bool = True,
        use_nodenorm: bool = True,
    ):
        super().__init__()
        self.in_dim = dim
        self.out_dim = dim
        self.M = num_heads
        self.gamma = gamma
        self.attn_dropout = attn_dropout
        self.use_activation = use_activation
        self.W = nn.ModuleList([ComplexLinear(dim, dim, bias=use_bias) for _ in range(self.M)])
        self.Q = nn.ModuleList([ComplexLinear(dim, dim, bias=False) for _ in range(self.M)])
        self._msg_gate = nn.Parameter(torch.tensor(0.5))
        self.act = ModReLU(dim) if use_activation else nn.Identity()
        self.nodenorm = LegacyNodeNorm() if use_nodenorm else nn.Identity()

    @property
    def msg_gate(self) -> Tensor:
        return torch.sigmoid(self._msg_gate)

    def forward(self, h: Tensor, edge_index: Tensor, sic_strength: float = 0.0) -> Tensor:
        if edge_index.numel() == 0:
            return self.act(self.nodenorm(h))

        src, dst = edge_index
        N = h.size(0)
        h_src_in = h[src]
        h_dst_in = h[dst]
        h_dst_norm2 = (h_dst_in.real.square() + h_dst_in.imag.square()).sum(dim=1, keepdim=True)
        h_dst_norm2 = h_dst_norm2.clamp_min(1e-6)
        updates_sum = torch.zeros((N, self.out_dim), dtype=torch.cfloat, device=h.device)
        gate = self.msg_gate

        for m in range(self.M):
            Wh_src = self.W[m](h_src_in)
            transported = nan_to_num_safe(Wh_src)

            hi_conj_dot = torch.sum(torch.conj(h_dst_in) * transported, dim=1, keepdim=True)
            hi_conj_dot = nan_to_num_safe(hi_conj_dot)
            proj = h_dst_in * (hi_conj_dot / h_dst_norm2)
            proj = nan_to_num_safe(proj)

            r_attn = transported - float(sic_strength) * proj
            r_attn = nan_to_num_safe(r_attn)

            Qhi = nan_to_num_safe(self.Q[m](h_dst_in))
            s = torch.sum(torch.conj(Qhi) * r_attn, dim=1)
            s = nan_to_num_safe(s)
            logits = self.gamma * torch.abs(s) / math.sqrt(max(1, self.out_dim))
            logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
            alpha = softmax(logits, dst, num_nodes=N)
            alpha = torch.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0)
            alpha = F.dropout(alpha, p=self.attn_dropout, training=self.training)

            base = gate.to(transported.dtype) * r_attn + (1.0 - gate).to(transported.dtype) * transported
            with torch.no_grad():
                sim = F.cosine_similarity(h_src_in.real, h_dst_in.real, dim=-1).clamp(min=-1.0, max=1.0)
                sign = torch.sign(sim)
            base = base * (0.5 + 0.5 * sign.to(base.dtype).unsqueeze(-1))

            base = nan_to_num_safe(base)
            msg = alpha.unsqueeze(-1).to(base.dtype) * base
            msg = nan_to_num_safe(msg)
            updates_sum.index_add_(0, dst, msg)

        h_new = h + updates_sum
        h_new = nan_to_num_safe(h_new)
        h_new = self.nodenorm(h_new)
        return self.act(h_new)


class GESCSoftmaxNet(nn.Module):
    """Full non-adaptive GESC baseline, adapted to the ASCii training API."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_classes: int,
        edge_index: Tensor,
        num_nodes: int,
        num_heads: int = 4,
        gamma: float = 0.1,
        attn_dropout: float = 0.2,
        feat_dropout: float = 0.5,
        readout_dropout: float = 0.5,
        layers: int = 2,
        sic_first: float = 1.5,
        alpha_skip: float = 0.1,
        jk_mode: str = 'concat',
        use_nodenorm: bool = True,
    ):
        super().__init__()
        if jk_mode == 'last':
            # The legacy baseline only had concat/mean. Map last to mean rather
            # than failing in sweeps that reuse ASCii CLI options.
            jk_mode = 'mean'
        assert jk_mode in ['concat', 'mean']
        self.edge_index = edge_index
        self.num_nodes = num_nodes
        self.num_classes = num_classes
        self.feat_drop = nn.Dropout(feat_dropout)
        self.alpha_skip = alpha_skip
        self.layers_num = layers
        self.jk_mode = jk_mode
        self.enc = ComplexLinear(in_dim, hidden_dim, bias=True)
        self.layers = nn.ModuleList([
            GESCSoftmaxLayer(
                hidden_dim,
                num_heads=num_heads,
                gamma=gamma,
                attn_dropout=attn_dropout,
                use_activation=(i < layers - 1),
                use_nodenorm=use_nodenorm,
            )
            for i in range(layers)
        ])
        self.sic_vals = [float(sic_first)] + [0.0] * max(0, layers - 1)
        cls_in = hidden_dim * 2 * layers if jk_mode == 'concat' else hidden_dim * 2
        self.cls_norm = nn.LayerNorm(cls_in)
        self.cls_drop = nn.Dropout(readout_dropout)
        self.cls = nn.Linear(cls_in, num_classes)
        self.last_homo_logits = None

    def forward(
        self,
        x_real: Tensor,
        edge_index_override: Optional[Tensor] = None,
        return_aux: bool = False,
        recompute_static: bool = False,
    ):
        del recompute_static
        self.last_homo_logits = None
        x_real = self.feat_drop(torch.nan_to_num(x_real.float(), nan=0.0, posinf=0.0, neginf=0.0))
        x = x_real.to(torch.cfloat)
        h = self.enc(x)
        h0 = h.detach()
        ei = self.edge_index if edge_index_override is None else edge_index_override
        hs = []
        for i, layer in enumerate(self.layers):
            h_new = layer(h, ei, sic_strength=self.sic_vals[i])
            h = (1.0 - self.alpha_skip) * h_new + self.alpha_skip * h0
            hs.append(h)
        h_out = torch.cat(hs, dim=-1) if self.jk_mode == 'concat' else torch.stack(hs, dim=0).mean(dim=0)
        z = torch.view_as_real(h_out).reshape(h_out.size(0), -1)
        z = self.cls_norm(z)
        z = self.cls_drop(z)
        logits = self.cls(z)
        if return_aux:
            return logits, logits.new_tensor(0.0)
        return logits


@torch.no_grad()
def normalize_feature_view(x: Tensor, eps: float=1e-12) -> Tensor:
    x = x.float()
    return F.normalize(x, p=2, dim=-1, eps=eps)

@torch.no_grad()
def normalize_homophily_block(x: Tensor, mode: str, eps: float=1e-12) -> Tensor:
    if mode == 'none':
        return x
    if mode == 'row':
        return F.normalize(x, p=2, dim=-1, eps=eps)
    if mode == 'column':
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True).clamp_min(eps)
        return torch.nan_to_num((x - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)
    raise ValueError("homo_feature_norm must be one of: none, row, column")

@torch.no_grad()
def build_homophilic_features(x: Tensor, edge_index: Tensor, K: int=10, alpha: float=0.1, dropout: float=0.0, mode: str='cat', norm: str='none') -> Tensor:
    if mode == 'off':
        return x
    x0 = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
    prop = APPNP(K=int(K), alpha=float(alpha), dropout=float(dropout)).to(x0.device)
    prop.eval()
    xp = prop(x0, edge_index)
    xp = torch.nan_to_num(xp, nan=0.0, posinf=0.0, neginf=0.0)
    x0n = normalize_homophily_block(x0, norm)
    xpn = normalize_homophily_block(xp, norm)
    if mode == 'prop':
        out = xpn
    elif mode == 'cat':
        out = torch.cat([x0n, xpn], dim=-1)
    elif mode == 'cat_diff':
        out = torch.cat([x0n, xpn, normalize_homophily_block(xp - x0, norm)], dim=-1)
    else:
        raise ValueError("homo_feature_mode must be one of: off, prop, cat, cat_diff")
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).to(x.dtype)

@torch.no_grad()
def compute_local_inconsistency(u: Tensor, edge_index: Tensor, num_nodes: int, eps: float=1e-12, edge_chunk_size: int=500000) -> Tensor:
    clean_edge_index, _ = remove_self_loops(edge_index)
    if clean_edge_index.numel() == 0:
        return torch.zeros(num_nodes, device=u.device, dtype=torch.float)

    u_norm = F.normalize(u.float(), p=2, dim=-1, eps=eps)
    acc = torch.zeros(num_nodes, device=u.device, dtype=torch.float)
    cnt = torch.zeros(num_nodes, device=u.device, dtype=torch.float)

    E = clean_edge_index.size(1)
    chunk = int(edge_chunk_size) if edge_chunk_size and edge_chunk_size > 0 else E
    for start in range(0, E, chunk):
        end = min(start + chunk, E)
        src = clean_edge_index[0, start:end]
        dst = clean_edge_index[1, start:end]
        sim = (u_norm[src] * u_norm[dst]).sum(dim=-1).clamp(min=-1.0, max=1.0)
        acc.index_add_(0, dst, sim)
        cnt.index_add_(0, dst, torch.ones_like(sim))

    avg = acc / cnt.clamp_min(1.0)
    q = (1.0 - avg).clamp(min=0.0, max=2.0)
    q = torch.where(cnt > 0, q, torch.zeros_like(q))
    return q

class ASCiiLayer(nn.Module):

    def __init__(self, dim: int, num_heads: int=4, gamma: float=1.0, attn_dropout: float=0.2, eta_max: float=0.75, eta_bar: float=0.5, lambda_attn: float=0.5, layer_id: int=0, num_layers: int=1, use_activation: bool=True, use_nodenorm: bool=True, use_phase: bool=True, phase_scale: float=1.0, eps: float=1e-08, edge_chunk_size: int=200000, attn_mode: str='softmax'):
        super().__init__()
        self.in_dim = dim
        self.out_dim = dim
        self.M = num_heads
        self.gamma = gamma
        self.attn_dropout = attn_dropout
        self.eta_max = float(eta_max)
        self.eta_bar = float(eta_bar)
        self.lambda_attn = float(lambda_attn)
        self.layer_id = layer_id
        self.num_layers = max(1, num_layers)
        self.layer_frac = float(layer_id + 1) / float(self.num_layers)
        self.use_phase = use_phase
        self.phase_scale = phase_scale
        self.eps = eps
        self.edge_chunk_size = int(edge_chunk_size) if edge_chunk_size is not None else 0
        self.attn_mode = str(attn_mode)
        if self.attn_mode not in ['softmax', 'sigmoid_degree', 'degree']:
            raise ValueError("attn_mode must be one of: softmax, sigmoid_degree, degree")
        self.W = nn.ModuleList([ComplexLinear(dim, dim, bias=False) for _ in range(self.M)])
        self.Q = nn.ModuleList([ComplexLinear(dim, dim, bias=False) for _ in range(self.M)])
        desc_dim = 7
        phase_desc_dim = 5
        self.eta_nets = nn.ModuleList([nn.Linear(desc_dim, 1) for _ in range(self.M)])
        self.mix_gate_nets = nn.ModuleList([nn.Linear(desc_dim, 1) for _ in range(self.M)])
        self.phase_nets = nn.ModuleList([nn.Linear(phase_desc_dim, 1) for _ in range(self.M)])
        self.xi_scale = nn.Parameter(torch.ones(self.M, dtype=torch.float))
        self.xi_bias = nn.Parameter(torch.zeros(self.M, dtype=torch.float))
        self.act = ModReLU(dim) if use_activation else nn.Identity()
        self.nodenorm = PhasePreservingNodeNorm() if use_nodenorm else nn.Identity()
        self.reset_scalar_modules()

    def reset_scalar_modules(self) -> None:
        for lin in list(self.eta_nets) + list(self.mix_gate_nets) + list(self.phase_nets):
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)

    def _edge_static_terms(self, edge_index: Tensor, x_view: Tensor, degree_log: Tensor, q_hat: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        src, dst = edge_index
        delta_x = 1.0 - (x_view[dst] * x_view[src]).sum(dim=-1, keepdim=True)
        delta_x = delta_x.clamp(min=0.0, max=2.0)
        delta_d = (degree_log[dst] - degree_log[src]).unsqueeze(-1)
        qi = q_hat[dst].unsqueeze(-1)
        qj = q_hat[src].unsqueeze(-1)
        layer_frac = torch.full_like(delta_x, self.layer_frac)
        return (delta_x, delta_d, qi, qj, layer_frac)

    def _compute_edge_terms(self, h: Tensor, edge_index: Tensor, x_view: Tensor, degree_log: Tensor, q_hat: Tensor, head_id: int) -> Tuple[Tensor, Tensor, Tensor]:
        src, dst = edge_index
        h_src = h[src]
        h_dst = h[dst]
        delta_x, delta_d, qi, qj, layer_frac = self._edge_static_terms(edge_index=edge_index, x_view=x_view, degree_log=degree_log, q_hat=q_hat)
        phase_desc = torch.cat([delta_x, delta_d, qi, qj, layer_frac], dim=-1)

        Wh_src = nan_to_num_safe(self.W[head_id](h_src))
        if self.use_phase:
            theta = math.pi * self.phase_scale * torch.tanh(self.phase_nets[head_id](phase_desc))
            U = torch.complex(torch.cos(theta), torch.sin(theta))
            transported = Wh_src * U
        else:
            transported = Wh_src
        transported = nan_to_num_safe(transported)

        Qhi = nan_to_num_safe(self.Q[head_id](h_dst))
        s = torch.sum(torch.conj(Qhi) * transported, dim=1, keepdim=True)
        s = nan_to_num_safe(s)
        nu = torch.linalg.vector_norm(Qhi, dim=-1, keepdim=True) * torch.linalg.vector_norm(transported, dim=-1, keepdim=True) + self.eps
        rho = torch.real(s) / nu
        chi = torch.abs(s) / nu
        rho = torch.nan_to_num(rho, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=-1.0, max=1.0)
        chi = torch.nan_to_num(chi, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0, max=1.0)

        z = torch.cat([rho, chi, delta_x, delta_d, qi, qj, layer_frac], dim=-1)
        eta = self.eta_max * torch.sigmoid(self.eta_nets[head_id](z))
        eta = torch.nan_to_num(eta, nan=0.0, posinf=self.eta_max, neginf=0.0)
        eta_penalty = (eta - self.eta_bar).square().mean()

        h_dst_norm2 = (h_dst.real.square() + h_dst.imag.square()).sum(dim=1, keepdim=True)
        h_dst_norm2 = h_dst_norm2.clamp_min(self.eps)
        hi_conj_dot = torch.sum(torch.conj(h_dst) * transported, dim=1, keepdim=True)
        hi_conj_dot = nan_to_num_safe(hi_conj_dot)
        proj = h_dst * (hi_conj_dot / h_dst_norm2)
        proj = nan_to_num_safe(proj)

        r = transported - eta.to(transported.dtype) * proj
        r = nan_to_num_safe(r)

        s_r = torch.sum(torch.conj(Qhi) * r, dim=1, keepdim=True)
        denom_r = torch.linalg.vector_norm(Qhi, dim=-1, keepdim=True) * torch.linalg.vector_norm(r, dim=-1, keepdim=True) + self.eps
        rho_r = torch.real(s_r) / denom_r
        rho_r = torch.nan_to_num(rho_r, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=-1.0, max=1.0)

        xi = torch.sigmoid(self.xi_scale[head_id] * rho_r + self.xi_bias[head_id])
        g = torch.sigmoid(self.mix_gate_nets[head_id](z))
        msg_hat = g.to(r.dtype) * xi.to(r.dtype) * r + (1.0 - g).to(transported.dtype) * transported
        msg_hat = nan_to_num_safe(msg_hat)

        s_tilde = torch.sum(torch.conj(Qhi) * msg_hat, dim=1, keepdim=True)
        denom_hat = torch.linalg.vector_norm(Qhi, dim=-1, keepdim=True) * torch.linalg.vector_norm(msg_hat, dim=-1, keepdim=True) + self.eps
        signed = torch.real(s_tilde) / denom_hat
        strength = torch.abs(s_tilde) / math.sqrt(max(1, self.out_dim))
        logits = self.gamma * (self.lambda_attn * strength + (1.0 - self.lambda_attn) * signed)
        logits = torch.nan_to_num(logits.squeeze(-1), nan=0.0, posinf=0.0, neginf=0.0)
        return logits, msg_hat, eta_penalty

    def _scatter_amax_(self, out: Tensor, index: Tensor, src: Tensor) -> None:
        if hasattr(out, 'scatter_reduce_'):
            out.scatter_reduce_(0, index, src, reduce='amax', include_self=True)
        else:
            for i, value in zip(index.tolist(), src.tolist()):
                if value > out[i]:
                    out[i] = value

    def forward(self, h: Tensor, edge_index: Tensor, x_view: Tensor, degree_log: Tensor, q_hat: Tensor, return_aux: bool=False):
        if edge_index.numel() == 0:
            h_next = self.act(self.nodenorm(h))
            zero = h.real.new_tensor(0.0)
            return (h_next, zero) if return_aux else h_next

        N = h.size(0)
        E = edge_index.size(1)
        chunk_size = self.edge_chunk_size if self.edge_chunk_size and self.edge_chunk_size > 0 else E
        chunk_size = max(1, int(chunk_size))
        updates_sum = torch.zeros((N, self.out_dim), dtype=torch.cfloat, device=h.device)
        eta_penalties = []

        if self.attn_mode in ['degree', 'sigmoid_degree']:
            dst_all = edge_index[1]
            deg_inv = degree(dst_all, num_nodes=N, dtype=h.real.dtype).clamp_min(1.0).reciprocal()
            for m in range(self.M):
                eta_weighted_sum = h.real.new_tensor(0.0)
                eta_weight = 0
                for start_i in range(0, E, chunk_size):
                    end_i = min(start_i + chunk_size, E)
                    ei_c = edge_index[:, start_i:end_i]
                    dst_c = ei_c[1]
                    logits_c, msg_hat_c, eta_penalty_c = self._compute_edge_terms(h, ei_c, x_view, degree_log, q_hat, m)
                    if self.attn_mode == 'degree':
                        alpha = deg_inv[dst_c]
                    else:
                        alpha = torch.sigmoid(logits_c) * deg_inv[dst_c]
                    alpha = torch.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0)
                    alpha = F.dropout(alpha, p=self.attn_dropout, training=self.training)
                    msg = alpha.unsqueeze(-1).to(msg_hat_c.dtype) * msg_hat_c
                    msg = nan_to_num_safe(msg)
                    updates_sum.index_add_(0, dst_c, msg)
                    eta_weighted_sum = eta_weighted_sum + eta_penalty_c * float(end_i - start_i)
                    eta_weight += end_i - start_i
                eta_penalties.append(eta_weighted_sum / max(1, eta_weight))
        elif chunk_size >= E:
            dst = edge_index[1]
            for m in range(self.M):
                logits, msg_hat, eta_penalty = self._compute_edge_terms(h, edge_index, x_view, degree_log, q_hat, m)
                alpha = softmax(logits, dst, num_nodes=N)
                alpha = torch.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0)
                alpha = F.dropout(alpha, p=self.attn_dropout, training=self.training)
                msg = alpha.unsqueeze(-1).to(msg_hat.dtype) * msg_hat
                msg = nan_to_num_safe(msg)
                updates_sum.index_add_(0, dst, msg)
                eta_penalties.append(eta_penalty)
        else:
            for m in range(self.M):
                max_per_dst = torch.full((N,), -torch.inf, dtype=h.real.dtype, device=h.device)
                with torch.no_grad():
                    for start_i in range(0, E, chunk_size):
                        end_i = min(start_i + chunk_size, E)
                        ei_c = edge_index[:, start_i:end_i]
                        dst_c = ei_c[1]
                        logits_c, _, _ = self._compute_edge_terms(h, ei_c, x_view, degree_log, q_hat, m)
                        self._scatter_amax_(max_per_dst, dst_c, logits_c)

                sum_exp = torch.zeros((N,), dtype=h.real.dtype, device=h.device)
                for start_i in range(0, E, chunk_size):
                    end_i = min(start_i + chunk_size, E)
                    ei_c = edge_index[:, start_i:end_i]
                    dst_c = ei_c[1]
                    logits_c, _, _ = self._compute_edge_terms(h, ei_c, x_view, degree_log, q_hat, m)
                    exp_c = torch.exp((logits_c - max_per_dst[dst_c]).clamp(min=-80.0, max=80.0))
                    sum_exp.index_add_(0, dst_c, exp_c)

                eta_weighted_sum = h.real.new_tensor(0.0)
                eta_weight = 0
                for start_i in range(0, E, chunk_size):
                    end_i = min(start_i + chunk_size, E)
                    ei_c = edge_index[:, start_i:end_i]
                    dst_c = ei_c[1]
                    logits_c, msg_hat_c, eta_penalty_c = self._compute_edge_terms(h, ei_c, x_view, degree_log, q_hat, m)
                    exp_c = torch.exp((logits_c - max_per_dst[dst_c]).clamp(min=-80.0, max=80.0))
                    alpha = exp_c / sum_exp[dst_c].clamp_min(self.eps)
                    alpha = torch.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0)
                    alpha = F.dropout(alpha, p=self.attn_dropout, training=self.training)
                    msg = alpha.unsqueeze(-1).to(msg_hat_c.dtype) * msg_hat_c
                    msg = nan_to_num_safe(msg)
                    updates_sum.index_add_(0, dst_c, msg)
                    eta_weighted_sum = eta_weighted_sum + eta_penalty_c * float(end_i - start_i)
                    eta_weight += end_i - start_i
                eta_penalties.append(eta_weighted_sum / max(1, eta_weight))
        h_new = h + updates_sum
        h_new = nan_to_num_safe(h_new)
        h_next = self.act(self.nodenorm(h_new))
        eta_penalty = torch.stack(eta_penalties).mean() if eta_penalties else h.real.new_tensor(0.0)
        return (h_next, eta_penalty) if return_aux else h_next

class HomophilyMLPBranch(nn.Module):

    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int, dropout: float=0.5, layers: int=2, use_norm: bool=True):
        super().__init__()
        layers = max(1, int(layers))
        hidden_dim = max(1, int(hidden_dim))
        modules = [nn.Dropout(dropout)]
        if layers == 1:
            modules.append(nn.Linear(in_dim, num_classes))
        else:
            prev = in_dim
            for _ in range(layers - 1):
                modules.append(nn.Linear(prev, hidden_dim))
                if use_norm:
                    modules.append(nn.LayerNorm(hidden_dim))
                modules.append(nn.ReLU())
                modules.append(nn.Dropout(dropout))
                prev = hidden_dim
            modules.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*modules)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0))

class InvariantGramReadout(nn.Module):

    def __init__(self, complex_dim: int, rank: int, num_classes: int, dropout: float=0.5, hidden: int=0):
        super().__init__()
        self.rank = int(rank)
        self.proj = ComplexLinear(complex_dim, self.rank, bias=False)
        phi_dim = self.rank + 2 * self.rank * self.rank + 1
        self.norm = nn.LayerNorm(phi_dim)
        self.drop = nn.Dropout(dropout)
        if hidden and hidden > 0:
            self.cls = nn.Sequential(nn.Linear(phi_dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, num_classes))
        else:
            self.cls = nn.Linear(phi_dim, num_classes)

    def forward(self, h: Tensor) -> Tensor:
        p = self.proj(h)
        gram = p.unsqueeze(2) * torch.conj(p.unsqueeze(1))
        h_norm = torch.linalg.vector_norm(h, dim=-1, keepdim=True).real
        phi = torch.cat([torch.abs(p), gram.real.reshape(h.size(0), -1), gram.imag.reshape(h.size(0), -1), h_norm], dim=-1)
        phi = torch.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0)
        return self.cls(self.drop(self.norm(phi)))

class ASCiiNet(nn.Module):

    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int, edge_index: Tensor, num_nodes: int, num_heads: int=4, gamma: float=0.1, attn_dropout: float=0.2, feat_dropout: float=0.5, readout_dropout: float=0.5, layers: int=2, eta_max: float=0.75, eta_bar: float=0.5, lambda_attn: float=0.5, alpha_skip: float=0.1, jk_mode: str='concat', readout_rank: int=16, readout_hidden: int=0, use_nodenorm: bool=True, use_phase: bool=True, phase_scale: float=1.0, edge_chunk_size: int=200000, static_chunk_size: int=500000, checkpoint_layers: bool=False, attn_mode: str='softmax', use_homo_branch: bool=False, homo_branch_hidden: int=128, homo_branch_dropout: float=0.5, homo_branch_layers: int=2, homo_branch_weight: float=0.35, homo_branch_max_weight: float=0.7, learn_homo_branch_weight: bool=True):
        super().__init__()
        assert jk_mode in ['last', 'concat', 'mean']
        self.edge_index = edge_index
        self.num_nodes = num_nodes
        self.num_classes = num_classes
        self.feat_drop = nn.Dropout(feat_dropout)
        self.alpha_skip = alpha_skip
        self.layers_num = layers
        self.jk_mode = jk_mode
        self.static_chunk_size = int(static_chunk_size) if static_chunk_size is not None else 0
        self.checkpoint_layers = bool(checkpoint_layers)
        self.enc = ComplexLinear(in_dim, hidden_dim, bias=True)
        self.layers = nn.ModuleList([ASCiiLayer(hidden_dim, num_heads=num_heads, gamma=gamma, attn_dropout=attn_dropout, eta_max=eta_max, eta_bar=eta_bar, lambda_attn=lambda_attn, layer_id=i, num_layers=layers, use_activation=i < layers - 1, use_nodenorm=use_nodenorm, use_phase=use_phase, phase_scale=phase_scale, edge_chunk_size=edge_chunk_size, attn_mode=attn_mode) for i in range(layers)])
        readout_dim = hidden_dim * layers if jk_mode == 'concat' else hidden_dim
        self.readout = InvariantGramReadout(complex_dim=readout_dim, rank=readout_rank, num_classes=num_classes, dropout=readout_dropout, hidden=readout_hidden)
        self.homo_branch = HomophilyMLPBranch(in_dim=in_dim, hidden_dim=homo_branch_hidden, num_classes=num_classes, dropout=homo_branch_dropout, layers=homo_branch_layers) if use_homo_branch else None
        self.learn_homo_branch_weight = bool(learn_homo_branch_weight)
        self.homo_branch_max_weight = max(0.0, min(1.0, float(homo_branch_max_weight)))
        self.last_homo_logits = None
        if self.homo_branch is not None:
            init_mix = max(0.0, min(float(homo_branch_weight), self.homo_branch_max_weight))
            if self.learn_homo_branch_weight and self.homo_branch_max_weight > 0.0:
                self.homo_mix_logit = nn.Parameter(torch.tensor(inverse_sigmoid(init_mix / self.homo_branch_max_weight), dtype=torch.float))
            else:
                self.register_buffer('homo_fixed_mix', torch.tensor(init_mix, dtype=torch.float), persistent=False)
                self.homo_mix_logit = None
        else:
            self.homo_mix_logit = None
        self.register_buffer('x_view', torch.empty(0), persistent=False)
        self.register_buffer('degree_log', torch.empty(0), persistent=False)
        self.register_buffer('q_hat', torch.empty(0), persistent=False)

    def homo_mix_weight(self) -> Optional[Tensor]:
        if self.homo_branch is None:
            return None
        if self.homo_mix_logit is not None:
            return self.homo_branch_max_weight * torch.sigmoid(self.homo_mix_logit)
        return self.homo_fixed_mix

    @torch.no_grad()
    def prepare_static_graph(self, x_real: Tensor, edge_index: Optional[Tensor]=None, unary_view: Optional[Tensor]=None) -> None:
        ei = self.edge_index if edge_index is None else edge_index
        self.x_view = normalize_feature_view(x_real.detach()).to(x_real.device)
        deg = degree(ei[1], num_nodes=x_real.size(0), dtype=torch.float).to(x_real.device)
        self.degree_log = torch.log1p(deg)
        q_base = unary_view.detach().float() if unary_view is not None else self.x_view
        self.q_hat = compute_local_inconsistency(q_base, ei, num_nodes=x_real.size(0), edge_chunk_size=self.static_chunk_size).to(x_real.device)

    def _ensure_static_graph(self, x_real: Tensor, edge_index: Tensor, force: bool=False) -> None:
        if force or self.x_view.numel() == 0 or self.x_view.size(0) != x_real.size(0):
            self.prepare_static_graph(x_real, edge_index)

    def forward(self, x_real: Tensor, edge_index_override: Optional[Tensor]=None, return_aux: bool=False, recompute_static: bool=False):
        ei = self.edge_index if edge_index_override is None else edge_index_override
        self._ensure_static_graph(x_real, ei, force=recompute_static)
        x_dropped = self.feat_drop(x_real)
        x = x_dropped.to(torch.cfloat)
        h = self.enc(x)
        h0 = h.detach()
        hs = []
        eta_losses = []
        for layer in self.layers:
            if self.checkpoint_layers and self.training:
                h_new, eta_loss = checkpoint(
                    lambda h_in, layer=layer: layer(h_in, ei, self.x_view, self.degree_log, self.q_hat, return_aux=True),
                    h,
                    use_reentrant=False,
                )
            else:
                h_new, eta_loss = layer(h, ei, self.x_view, self.degree_log, self.q_hat, return_aux=True)
            h = (1.0 - self.alpha_skip) * h_new + self.alpha_skip * h0
            hs.append(h)
            eta_losses.append(eta_loss)
        if self.jk_mode == 'concat':
            h_out = torch.cat(hs, dim=-1)
        elif self.jk_mode == 'mean':
            h_out = torch.stack(hs, dim=0).mean(dim=0)
        else:
            h_out = hs[-1]
        base_logits = self.readout(h_out)
        self.last_homo_logits = None
        if self.homo_branch is not None:
            homo_logits = self.homo_branch(x_real)
            self.last_homo_logits = homo_logits
            mix = self.homo_mix_weight().to(base_logits.device, dtype=base_logits.dtype)
            logits = (1.0 - mix) * base_logits + mix * homo_logits
        else:
            logits = base_logits
        if return_aux:
            eta_reg = torch.stack(eta_losses).mean() if eta_losses else logits.new_tensor(0.0)
            return (logits, eta_reg)
        return logits

def require_ogb() -> None:
    if PygNodePropPredDataset is None or Evaluator is None:
        raise ImportError("OGB datasets require the 'ogb' package. Install with: pip install ogb")


def make_mask(num_nodes: int, idx: Tensor) -> Tensor:
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[idx.view(-1).long()] = True
    return mask


def attach_split_masks(data: Data, split_idx, node_type: Optional[str]=None) -> Data:
    def pick(split_name: str) -> Tensor:
        value = split_idx[split_name]
        if isinstance(value, dict):
            if node_type is None:
                raise ValueError('node_type must be provided for heterogeneous OGB splits.')
            value = value[node_type]
        return value

    data.train_mask = make_mask(data.num_nodes, pick('train'))
    data.val_mask = make_mask(data.num_nodes, pick('valid'))
    data.test_mask = make_mask(data.num_nodes, pick('test'))
    return data


def add_ogbn_proteins_features(data: Data, edge_attr_chunk_size: int=1000000) -> Data:
    if getattr(data, 'x', None) is None:
        if getattr(data, 'edge_attr', None) is None:
            raise ValueError('ogbn-proteins requires edge_attr to build node features.')
        dst_all = data.edge_index[1]
        x = torch.zeros((data.num_nodes, data.edge_attr.size(-1)), dtype=data.edge_attr.dtype)
        E = data.edge_index.size(1)
        chunk = int(edge_attr_chunk_size) if edge_attr_chunk_size and edge_attr_chunk_size > 0 else E
        for start in range(0, E, chunk):
            end = min(start + chunk, E)
            x.index_add_(0, dst_all[start:end], data.edge_attr[start:end])
        data.x = x
    data.edge_attr = None
    gc.collect()
    return data


def infer_num_node_features(dataset, data) -> int:
    value = getattr(dataset, 'num_node_features', None)
    if value is not None and int(value) > 0:
        return int(value)
    value = getattr(dataset, 'num_features', None)
    if value is not None and int(value) > 0:
        return int(value)
    return int(data.x.size(-1))


def infer_num_classes(dataset, data) -> int:
    if getattr(data, 'task_type', 'multiclass') == 'multilabel':
        return int(data.y.size(-1))
    value = getattr(dataset, 'num_classes', None)
    if value is not None and int(value) > 0:
        return int(value)
    return int(data.y.max().item() + 1)


def supervised_loss(logits: Tensor, y: Tensor, mask: Tensor, task_type: str, label_smooth: float=0.0) -> Tensor:
    if task_type == 'multilabel':
        return F.binary_cross_entropy_with_logits(logits[mask], y[mask].float())
    return cross_entropy_with_label_smoothing(logits[mask], y[mask].view(-1).long(), smoothing=label_smooth)


def prediction_consistency(logits1: Tensor, logits2_detached: Tensor, task_type: str, T: float=2.0) -> Tensor:
    if task_type == 'multilabel':
        return F.mse_loss(torch.sigmoid(logits1), torch.sigmoid(logits2_detached))
    return js_consistency(logits1, logits2_detached, T=T)


def supervised_loss_from_logits(logits: Tensor, y: Tensor, task_type: str, label_smooth: float=0.0) -> Tensor:
    if task_type == 'multilabel':
        return F.binary_cross_entropy_with_logits(logits, y.float())
    return cross_entropy_with_label_smoothing(logits, y.view(-1).long(), smoothing=label_smooth)

def model_zero(model: nn.Module) -> Tensor:
    for p in model.parameters():
        if torch.is_complex(p):
            return p.real.new_tensor(0.0)
        return p.new_tensor(0.0)
    return torch.tensor(0.0)

def homophily_auxiliary_loss(model: nn.Module, y: Tensor, mask: Tensor, task_type: str, label_smooth: float=0.0) -> Tensor:
    logits = getattr(model, 'last_homo_logits', None)
    if logits is None:
        return model_zero(model)
    return supervised_loss(logits, y, mask, task_type, label_smooth=label_smooth)

def homophily_auxiliary_root_loss(model: nn.Module, y_root: Tensor, task_type: str, label_smooth: float=0.0) -> Tensor:
    logits = getattr(model, 'last_homo_logits', None)
    if logits is None:
        return model_zero(model)
    return supervised_loss_from_logits(logits[:y_root.size(0)], y_root, task_type, label_smooth=label_smooth)


def require_neighbor_loader() -> None:
    if NeighborLoader is None:
        raise ImportError('Mini-batch training requires torch_geometric.loader.NeighborLoader.')


def parse_num_neighbors(value: Optional[str], layers: int) -> List[int]:
    layers = max(1, int(layers))
    if value is None or str(value).strip() == '':
        return [10] * layers
    text = str(value).strip().lower()
    if text in {'full', 'all'}:
        return [-1] * layers
    nums = [int(x.strip()) for x in text.replace(';', ',').split(',') if x.strip()]
    if not nums:
        return [10] * layers
    if len(nums) == 1:
        nums = nums * layers
    elif len(nums) < layers:
        nums = nums + [nums[-1]] * (layers - len(nums))
    return nums[:layers]


def make_neighbor_loader(data: Data, input_nodes, num_neighbors: List[int], batch_size: int, shuffle: bool, num_workers: int, device: torch.device):
    require_neighbor_loader()
    kwargs = dict(
        data=data,
        input_nodes=input_nodes,
        num_neighbors=num_neighbors,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
    )
    if int(num_workers) > 0:
        kwargs['persistent_workers'] = True
    if device.type == 'cuda':
        kwargs['pin_memory'] = True
    return NeighborLoader(**kwargs)


def finite_gradients(model: nn.Module) -> bool:
    for p in model.parameters():
        if p.grad is not None and (not torch.isfinite(p.grad).all()):
            return False
    return True


def train_minibatch_epoch(model: nn.Module, data: Data, args, device: torch.device, epoch: int, opt, scheduler) -> Tuple[float, float, float, float]:
    task_type = getattr(data, 'task_type', 'multiclass')
    train_neighbors = parse_num_neighbors(args.num_neighbors, args.layers)
    loader = make_neighbor_loader(
        data=data,
        input_nodes=data.train_mask,
        num_neighbors=train_neighbors,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        device=device,
    )
    model.train()
    total_examples = 0
    total_loss = 0.0
    total_ce = 0.0
    total_cons = 0.0
    total_eta = 0.0

    for step, batch in enumerate(loader, start=1):
        batch = batch.to(device, non_blocking=True)
        if args.batch_self_loops:
            batch.edge_index, _ = add_remaining_self_loops(batch.edge_index, num_nodes=batch.num_nodes)
        root_size = int(batch.batch_size)
        ei1 = dropedge(batch.edge_index, args.dropedge, training=True)
        logits1, eta_loss1 = model(batch.x, edge_index_override=ei1, return_aux=True, recompute_static=True)
        root_logits1 = logits1[:root_size]
        root_y = batch.y[:root_size]
        ce_main = supervised_loss_from_logits(root_logits1, root_y, task_type, label_smooth=args.label_smooth)
        homo_aux = homophily_auxiliary_root_loss(model, root_y, task_type, label_smooth=args.label_smooth)
        ce = ce_main + args.homo_aux_w * homo_aux

        if args.consistency_w > 0.0:
            ei2 = dropedge(batch.edge_index, args.dropedge, training=True)
            logits2, eta_loss2 = model(batch.x, edge_index_override=ei2, return_aux=True, recompute_static=True)
            cons = prediction_consistency(root_logits1, logits2[:root_size].detach(), task_type, T=args.cons_T)
            eta_loss = 0.5 * (eta_loss1 + eta_loss2)
        else:
            cons = root_logits1.new_tensor(0.0)
            eta_loss = eta_loss1

        loss = ce + args.consistency_w * cons + args.eta_reg * eta_loss
        if epoch <= args.warmup:
            loss = loss * (epoch / max(1, args.warmup))
        if not torch.isfinite(loss):
            print('[warn] non-finite mini-batch loss detected; skipping step')
            opt.zero_grad(set_to_none=True)
            continue

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if not finite_gradients(model):
            print('[warn] non-finite mini-batch grad detected; skipping step')
            opt.zero_grad(set_to_none=True)
            continue
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, error_if_nonfinite=False)
        opt.step()

        weight = max(1, root_size)
        total_examples += weight
        total_loss += float(loss.detach().item()) * weight
        total_ce += float(ce.detach().item()) * weight
        total_cons += float(cons.detach().item()) * weight
        total_eta += float(eta_loss.detach().item()) * weight

        del batch, logits1, root_logits1, root_y, loss, ce, cons, eta_loss
        if device.type == 'cuda' and args.empty_cache_steps > 0 and step % args.empty_cache_steps == 0:
            torch.cuda.empty_cache()

    scheduler.step()
    denom = max(1, total_examples)
    return (total_loss / denom, total_ce / denom, total_cons / denom, total_eta / denom)


@torch.no_grad()
def evaluate_minibatch(model: nn.Module, data: Data, args, device: torch.device) -> Tuple[float, float, float, Optional[Tensor]]:
    model.eval()
    task_type = getattr(data, 'task_type', 'multiclass')
    eval_metric = getattr(data, 'eval_metric', 'acc')
    eval_neighbors = parse_num_neighbors(args.eval_num_neighbors, args.layers)

    def eval_split(mask: Tensor) -> float:
        if int(mask.sum().item()) == 0:
            return float('nan')
        loader = make_neighbor_loader(
            data=data,
            input_nodes=mask,
            num_neighbors=eval_neighbors,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            device=device,
        )
        if eval_metric == 'rocauc':
            y_true_chunks = []
            y_pred_chunks = []
            for batch in loader:
                batch = batch.to(device, non_blocking=True)
                if args.batch_self_loops:
                    batch.edge_index, _ = add_remaining_self_loops(batch.edge_index, num_nodes=batch.num_nodes)
                root_size = int(batch.batch_size)
                logits = model(batch.x, edge_index_override=batch.edge_index, recompute_static=True)[:root_size]
                y_true_chunks.append(batch.y[:root_size].detach().cpu().float())
                y_pred_chunks.append(torch.sigmoid(logits).detach().cpu())
                del batch, logits
            evaluator = Evaluator(name=data.ogb_name)
            return evaluator.eval({
                'y_true': torch.cat(y_true_chunks, dim=0),
                'y_pred': torch.cat(y_pred_chunks, dim=0),
            })['rocauc']

        correct = 0
        total = 0
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            if args.batch_self_loops:
                batch.edge_index, _ = add_remaining_self_loops(batch.edge_index, num_nodes=batch.num_nodes)
            root_size = int(batch.batch_size)
            logits = model(batch.x, edge_index_override=batch.edge_index, recompute_static=True)[:root_size]
            pred = logits.argmax(dim=-1)
            y = batch.y[:root_size].view(-1).long()
            correct += int((pred == y).sum().item())
            total += int(root_size)
            del batch, logits, pred, y
        return float(correct) / max(1, total)

    train_score = eval_split(data.train_mask) if args.eval_train else float('nan')
    val_score = eval_split(data.val_mask)
    test_score = eval_split(data.test_mask)
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return (train_score, val_score, test_score, None)


def load_dataset(data_id: int, device: Optional[torch.device]=None):
    if data_id == 0:
        dataset = Planetoid(root='/tmp/Cora', name='Cora')
        data = dataset[0]
    elif data_id == 1:
        dataset = Planetoid(root='/tmp/Citeseer', name='Citeseer')
        data = dataset[0]
    elif data_id == 2:
        dataset = Planetoid(root='/tmp/Pubmed', name='Pubmed')
        data = dataset[0]
    elif data_id == 3:
        dataset = WikipediaNetwork(root='/tmp/Chameleon', name='chameleon')
        data = dataset[0]
    elif data_id == 4:
        dataset = WikipediaNetwork(root='/tmp/Squirrel', name='squirrel')
        data = dataset[0]
    elif data_id == 5:
        dataset = Actor(root='/tmp/Actor')
        data = dataset[0]
    elif data_id == 6:
        dataset = WebKB(root='/tmp/Cornell', name='Cornell')
        data = dataset[0]
    elif data_id == 7:
        dataset = WebKB(root='/tmp/Texas', name='Texas')
        data = dataset[0]
    elif data_id == 8:
        dataset = WebKB(root='/tmp/Wisconsin', name='Wisconsin')
        data = dataset[0]
    elif data_id == 9:
        require_ogb()
        dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='/tmp/ogbn_arxiv')
        data = attach_split_masks(dataset[0], dataset.get_idx_split())
        data.y = data.y.view(-1).long()
        data.ogb_name = 'ogbn-arxiv'
        data.eval_metric = 'acc'
        data.task_type = 'multiclass'
    elif data_id == 10:
        require_ogb()
        dataset = PygNodePropPredDataset(name='ogbn-proteins', root='/tmp/ogbn_proteins')
        data = attach_split_masks(add_ogbn_proteins_features(dataset[0]), dataset.get_idx_split())
        data.y = data.y.float()
        data.ogb_name = 'ogbn-proteins'
        data.eval_metric = 'rocauc'
        data.task_type = 'multilabel'
    elif data_id == 11:
        require_ogb()
        dataset = PygNodePropPredDataset(name='ogbn-mag', root='/tmp/ogbn_mag')
        raw_data = dataset[0]
        data = Data(
            x=raw_data['paper'].x,
            edge_index=raw_data['paper', 'cites', 'paper'].edge_index,
            y=raw_data['paper'].y.view(-1).long(),
            num_nodes=raw_data['paper'].num_nodes,
        )
        data = attach_split_masks(data, dataset.get_idx_split(), node_type='paper')
        data.ogb_name = 'ogbn-mag'
        data.eval_metric = 'acc'
        data.task_type = 'multiclass'
    else:
        raise ValueError('Unknown data id. Use 0-8 for built-in datasets, 9=ogbn-arxiv, 10=ogbn-proteins, 11=ogbn-mag.')
    if hasattr(data, 'train_mask') and data.train_mask.dim() == 2:
        data.train_mask = data.train_mask[:, 0]
        data.val_mask = data.val_mask[:, 0]
        data.test_mask = data.test_mask[:, 0]
    if data.y.dim() == 2 and data.y.size(-1) == 1:
        data.y = data.y.view(-1).long()
    data.task_type = getattr(data, 'task_type', 'multiclass')
    data.eval_metric = getattr(data, 'eval_metric', 'acc')
    if device is not None:
        data = data.to(device)
    return (dataset, data)

def correct_and_smooth_compat(logits: Tensor, y: Tensor, train_mask: Tensor, edge_index: Tensor, cs_corr_layers: int=50, cs_corr_alpha: float=0.5, cs_smooth_layers: int=50, cs_smooth_alpha: float=0.8, autoscale: bool=True) -> Tensor:
    with torch.no_grad():
        y_soft = F.softmax(logits, dim=-1)
        y_soft = torch.nan_to_num(y_soft, 0.0, 0.0, 0.0)
        row_sum = y_soft.sum(dim=-1, keepdim=True)
        bad = (row_sum <= 1e-08) | torch.isnan(row_sum)
        if bad.any():
            N, C = y_soft.size()
            y_soft[bad.expand(-1, C)] = 1.0 / C
        y_soft = y_soft / (y_soft.sum(dim=-1, keepdim=True) + 1e-12)
        residual = 1.0 - y_soft.sum(dim=-1, keepdim=True)
        y_soft[:, :1] = y_soft[:, :1] + residual
        y_soft = torch.clamp(y_soft, min=0.0)
        y_soft = y_soft / (y_soft.sum(dim=-1, keepdim=True) + 1e-12)
        cs = CorrectAndSmooth(num_correction_layers=cs_corr_layers, correction_alpha=cs_corr_alpha, num_smoothing_layers=cs_smooth_layers, smoothing_alpha=cs_smooth_alpha, autoscale=autoscale)
        y_true_m = y[train_mask]
        y_corr = cs.correct(y_soft, y_true_m, train_mask, edge_index)
        y_smooth = cs.smooth(y_corr, y_true_m, train_mask, edge_index)
        y_smooth = torch.nan_to_num(y_smooth, 0.0, 0.0, 0.0)
        return y_smooth / (y_smooth.sum(dim=-1, keepdim=True) + 1e-12)

@torch.no_grad()
def evaluate(model: nn.Module, data, args) -> Tuple[float, float, float, Tensor]:
    model.eval()
    logits_eval = model(data.x, edge_index_override=data.edge_index)
    pred_eval = logits_eval
    task_type = getattr(data, 'task_type', 'multiclass')
    eval_metric = getattr(data, 'eval_metric', 'acc')

    if task_type != 'multilabel' and args.use_cs:
        y_smooth = correct_and_smooth_compat(logits_eval, data.y, data.train_mask, data.edge_index, cs_corr_layers=args.cs_corr_layers, cs_corr_alpha=args.cs_corr_alpha, cs_smooth_layers=args.cs_smooth_layers, cs_smooth_alpha=args.cs_smooth_alpha, autoscale=True)
        pred_eval = (y_smooth + 1e-12).log()
    if task_type != 'multilabel' and args.use_lp:
        lp = LabelPropagation(num_layers=args.lp_layers, alpha=args.lp_alpha)
        y_lp = lp(data.y, data.edge_index, mask=data.train_mask)
        probs = F.softmax(pred_eval, dim=-1) * (1.0 - args.lp_blend) + y_lp * args.lp_blend
        pred_eval = (probs + 1e-12).log()

    if eval_metric == 'rocauc':
        evaluator = Evaluator(name=data.ogb_name)
        scores = torch.sigmoid(pred_eval)
        train_score = evaluator.eval({'y_true': data.y[data.train_mask].detach().cpu(), 'y_pred': scores[data.train_mask].detach().cpu()})['rocauc']
        val_score = evaluator.eval({'y_true': data.y[data.val_mask].detach().cpu(), 'y_pred': scores[data.val_mask].detach().cpu()})['rocauc']
        test_score = evaluator.eval({'y_true': data.y[data.test_mask].detach().cpu(), 'y_pred': scores[data.test_mask].detach().cpu()})['rocauc']
        return (train_score, val_score, test_score, pred_eval)

    pred = pred_eval.argmax(dim=1)
    if hasattr(data, 'ogb_name') and data.ogb_name in ['ogbn-arxiv', 'ogbn-mag']:
        evaluator = Evaluator(name=data.ogb_name)
        y_true = data.y.view(-1, 1)
        y_pred = pred.view(-1, 1)
        train_acc = evaluator.eval({'y_true': y_true[data.train_mask].detach().cpu(), 'y_pred': y_pred[data.train_mask].detach().cpu()})['acc']
        val_acc = evaluator.eval({'y_true': y_true[data.val_mask].detach().cpu(), 'y_pred': y_pred[data.val_mask].detach().cpu()})['acc']
        test_acc = evaluator.eval({'y_true': y_true[data.test_mask].detach().cpu(), 'y_pred': y_pred[data.test_mask].detach().cpu()})['acc']
    else:
        train_acc = (pred[data.train_mask] == data.y[data.train_mask]).float().mean().item()
        val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean().item()
        test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()
    return (train_acc, val_acc, test_acc, pred_eval)

def apply_homophily_defaults(args, provided: set) -> None:
    if not is_homophilic_builtin(args.data) or not args.auto_homo_defaults:
        if args.use_homo_feature_boost is None:
            args.use_homo_feature_boost = False
        if args.use_homo_branch is None:
            args.use_homo_branch = False
        return
    if args.use_homo_feature_boost is None:
        args.use_homo_feature_boost = True
    if args.use_homo_branch is None:
        args.use_homo_branch = True
    profile = {
        0: dict(lr=0.003, weight_decay=0.001, hidden=128, readout_rank=24, readout_hidden=128, feat_dropout=0.55, readout_dropout=0.55, attn_dropout=0.15, dropedge=0.10, label_smooth=0.03, consistency_w=0.05, warmup=30, patience=300, homo_branch_weight=0.35, homo_aux_w=0.45, lp_blend=0.25),
        1: dict(lr=0.003, weight_decay=0.001, hidden=128, readout_rank=24, readout_hidden=128, feat_dropout=0.60, readout_dropout=0.55, attn_dropout=0.15, dropedge=0.12, label_smooth=0.04, consistency_w=0.05, warmup=30, patience=300, homo_branch_weight=0.40, homo_aux_w=0.50, lp_blend=0.25),
        2: dict(lr=0.0025, weight_decay=0.0005, hidden=128, readout_rank=24, readout_hidden=128, feat_dropout=0.45, readout_dropout=0.50, attn_dropout=0.10, dropedge=0.08, label_smooth=0.02, consistency_w=0.03, warmup=30, patience=300, homo_branch_weight=0.30, homo_aux_w=0.35, lp_blend=0.20),
    }[int(args.data)]
    for key, value in profile.items():
        set_if_unprovided(args, provided, key, value)
    set_if_unprovided(args, provided, 'layers', 2)
    set_if_unprovided(args, provided, 'heads', 4)
    set_if_unprovided(args, provided, 'jk', 'concat')
    set_if_unprovided(args, provided, 'use_cs', True)
    set_if_unprovided(args, provided, 'use_lp', True)
    set_if_unprovided(args, provided, 'cs_corr_layers', 50)
    set_if_unprovided(args, provided, 'cs_corr_alpha', 0.7)
    set_if_unprovided(args, provided, 'cs_smooth_layers', 50)
    set_if_unprovided(args, provided, 'cs_smooth_alpha', 0.85)
    set_if_unprovided(args, provided, 'lp_layers', 50)
    set_if_unprovided(args, provided, 'lp_alpha', 0.9)
    set_if_unprovided(args, provided, 'homo_feature_mode', 'cat')
    set_if_unprovided(args, provided, 'homo_feature_norm', 'none')
    set_if_unprovided(args, provided, 'homo_prop_K', 10)
    set_if_unprovided(args, provided, 'homo_prop_alpha', 0.1)
    set_if_unprovided(args, provided, 'homo_prop_dropout', 0.0)
    set_if_unprovided(args, provided, 'homo_branch_hidden', 128)
    set_if_unprovided(args, provided, 'homo_branch_dropout', 0.55)
    set_if_unprovided(args, provided, 'homo_branch_layers', 2)
    set_if_unprovided(args, provided, 'homo_branch_max_weight', 0.65)
    set_if_unprovided(args, provided, 'learn_homo_branch_weight', True)


def resolve_base_model(args) -> str:
    """Auto policy: use the non-adaptive GESC baseline on Cora, ASCii elsewhere."""
    requested = str(args.base_model).lower()
    if requested != 'auto':
        return requested
    return 'gesc' if int(args.data) == 0 else 'adaptive'


def apply_gesc_defaults(args, provided: set) -> None:
    """Restore the attached non-adaptive GESC defaults for selected datasets.

    This intentionally runs *after* apply_homophily_defaults so Cora can fall
    back to the simpler baseline unless the user explicitly overrides a value.
    """
    set_if_unprovided(args, provided, 'lr', 0.001)
    set_if_unprovided(args, provided, 'weight_decay', 0.0005)
    set_if_unprovided(args, provided, 'heads', 4)
    set_if_unprovided(args, provided, 'hidden', 64)
    set_if_unprovided(args, provided, 'gamma', 0.1)
    set_if_unprovided(args, provided, 'attn_dropout', 0.2)
    set_if_unprovided(args, provided, 'feat_dropout', 0.5)
    set_if_unprovided(args, provided, 'readout_dropout', 0.5)
    set_if_unprovided(args, provided, 'layers', 2)
    set_if_unprovided(args, provided, 'sic_first', 1.5)
    set_if_unprovided(args, provided, 'alpha_skip', 0.1)
    set_if_unprovided(args, provided, 'jk', 'concat')
    set_if_unprovided(args, provided, 'label_smooth', 0.0)
    set_if_unprovided(args, provided, 'dropedge', 0.0)
    set_if_unprovided(args, provided, 'warmup', 50)
    set_if_unprovided(args, provided, 'patience', 200)
    set_if_unprovided(args, provided, 'consistency_w', 0.1)
    set_if_unprovided(args, provided, 'cons_T', 2.0)
    set_if_unprovided(args, provided, 'use_cs', True)
    set_if_unprovided(args, provided, 'cs_corr_layers', 50)
    set_if_unprovided(args, provided, 'cs_corr_alpha', 0.5)
    set_if_unprovided(args, provided, 'cs_smooth_layers', 50)
    set_if_unprovided(args, provided, 'cs_smooth_alpha', 0.8)
    set_if_unprovided(args, provided, 'use_lp', True)
    set_if_unprovided(args, provided, 'lp_layers', 50)
    set_if_unprovided(args, provided, 'lp_alpha', 0.9)
    set_if_unprovided(args, provided, 'lp_blend', 0.2)
    set_if_unprovided(args, provided, 'use_preprop', False)
    set_if_unprovided(args, provided, 'preprop_K', 10)
    set_if_unprovided(args, provided, 'preprop_alpha', 0.1)
    set_if_unprovided(args, provided, 'preprop_dropout', 0.0)
    set_if_unprovided(args, provided, 'use_homo_feature_boost', False)
    set_if_unprovided(args, provided, 'use_homo_branch', False)
    set_if_unprovided(args, provided, 'homo_aux_w', 0.0)
    set_if_unprovided(args, provided, 'eta_reg', 0.0)

def train_main() -> None:
    parser = argparse.ArgumentParser(description='ASCii: Adaptive Gauge-Invariant Self-Interference Control')
    parser.add_argument('data', type=int, help='0=Cora,1=Citeseer,2=Pubmed,3=Chameleon,4=Squirrel,5=Actor,6=Cornell,7=Texas,8=Wisconsin,9=ogbn-arxiv,10=ogbn-proteins,11=ogbn-mag')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--base_model', '--base-model', dest='base_model', type=str, default='auto', choices=['auto', 'adaptive', 'gesc'], help='auto uses non-adaptive GESC on Cora and adaptive ASCii elsewhere.')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--attn_dropout', type=float, default=0.2)
    parser.add_argument('--feat_dropout', type=float, default=0.5)
    parser.add_argument('--readout_dropout', type=float, default=0.5)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--sic_first', type=float, default=1.5, help='Fixed first-layer SIC strength used by --base_model gesc.')
    parser.add_argument('--alpha_skip', type=float, default=0.1)
    parser.add_argument('--jk', type=str, default='concat', choices=['last', 'concat', 'mean'])
    parser.add_argument('--use_nodenorm', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--eta_max', type=float, default=0.75, help='Upper bound for adaptive SIC coefficient.')
    parser.add_argument('--eta_bar', type=float, default=0.5, help='Center for eta regularization.')
    parser.add_argument('--eta_reg', type=float, default=0.0001, help='Weight for coefficient stabilizer.')
    parser.add_argument('--lambda_attn', type=float, default=0.5, help='Mix between magnitude attention and signed phase attention.')
    parser.add_argument('--readout_rank', type=int, default=16, help='Rank r for magnitude-Gram readout.')
    parser.add_argument('--readout_hidden', type=int, default=0, help='Hidden width for readout MLP; 0 uses linear classifier.')
    parser.add_argument('--use_phase', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--phase_scale', type=float, default=1.0, help='theta range is +/- pi * phase_scale.')
    parser.add_argument('--label_smooth', type=float, default=0.0)
    parser.add_argument('--dropedge', type=float, default=0.0)
    parser.add_argument('--cosine_min_lr_scale', type=float, default=0.1)
    parser.add_argument('--warmup', type=int, default=50)
    parser.add_argument('--patience', type=int, default=200)
    parser.add_argument('--consistency_w', type=float, default=0.1)
    parser.add_argument('--cons_T', type=float, default=2.0)
    parser.add_argument('--use_cs', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--cs_corr_layers', type=int, default=50)
    parser.add_argument('--cs_corr_alpha', type=float, default=0.5)
    parser.add_argument('--cs_smooth_layers', type=int, default=50)
    parser.add_argument('--cs_smooth_alpha', type=float, default=0.8)
    parser.add_argument('--use_lp', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--lp_layers', type=int, default=50)
    parser.add_argument('--lp_alpha', type=float, default=0.9)
    parser.add_argument('--lp_blend', type=float, default=0.2)
    parser.add_argument('--use_preprop', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--preprop_K', type=int, default=10)
    parser.add_argument('--preprop_alpha', type=float, default=0.1)
    parser.add_argument('--preprop_dropout', type=float, default=0.0)
    parser.add_argument('--edge_chunk_size', type=int, default=50000, help='Process edges in chunks to reduce peak memory. Set <=0 for original full-edge mode.')
    parser.add_argument('--static_chunk_size', type=int, default=100000, help='Chunk size for static graph statistics such as local inconsistency.')
    parser.add_argument('--checkpoint_layers', action=argparse.BooleanOptionalAction, default=False, help='Use activation checkpointing for ASCii layers.')
    parser.add_argument('--ogb_safe_defaults', action=argparse.BooleanOptionalAction, default=True, help='For OGB, use mini-batch training and disable memory-heavy full-graph postprocessing.')
    parser.add_argument('--eval_every', type=int, default=1, help='Evaluate every N epochs.')
    parser.add_argument('--undirected', action=argparse.BooleanOptionalAction, default=None, help='Make graph undirected. Default: True for small built-ins, False for OGB to save memory.')
    parser.add_argument('--add_self_loops', action=argparse.BooleanOptionalAction, default=None, help='Add self loops to the stored graph. Default: True for full-batch small graphs, False for OGB mini-batch.')
    parser.add_argument('--batch_self_loops', action=argparse.BooleanOptionalAction, default=True, help='Add self-loops to each sampled mini-batch subgraph.')
    parser.add_argument('--mini_batch', action=argparse.BooleanOptionalAction, default=None, help='Use NeighborLoader mini-batch training. Default: True for OGB, False otherwise.')
    parser.add_argument('--batch_size', type=int, default=None, help='Seed-node batch size for mini-batch training.')
    parser.add_argument('--eval_batch_size', type=int, default=None, help='Seed-node batch size for mini-batch evaluation.')
    parser.add_argument('--num_neighbors', type=str, default=None, help='Comma-separated train fanouts for NeighborLoader, e.g. 10,5. Use -1 for full neighbors.')
    parser.add_argument('--eval_num_neighbors', type=str, default=None, help='Comma-separated eval fanouts. Default: same as --num_neighbors.')
    parser.add_argument('--num_workers', type=int, default=0, help='NeighborLoader worker count. Use 0 if your environment has sampler worker issues.')
    parser.add_argument('--empty_cache_steps', type=int, default=0, help='Call torch.cuda.empty_cache every N mini-batches; 0 disables it.')
    parser.add_argument('--eval_train', action=argparse.BooleanOptionalAction, default=True, help='Evaluate the train split during mini-batch evaluation.')
    parser.add_argument('--attn_mode', type=str, default='auto', choices=['auto', 'softmax', 'sigmoid_degree', 'degree'], help='softmax is the original attention; sigmoid_degree is a one-pass memory-safe gate used by OGB defaults.')
    parser.add_argument('--auto_homo_defaults', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--use_homo_feature_boost', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--homo_feature_mode', type=str, default='cat', choices=['off', 'prop', 'cat', 'cat_diff'])
    parser.add_argument('--homo_feature_norm', type=str, default='none', choices=['none', 'row', 'column'])
    parser.add_argument('--homo_prop_K', type=int, default=10)
    parser.add_argument('--homo_prop_alpha', type=float, default=0.1)
    parser.add_argument('--homo_prop_dropout', type=float, default=0.0)
    parser.add_argument('--use_homo_branch', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--homo_branch_hidden', type=int, default=128)
    parser.add_argument('--homo_branch_dropout', type=float, default=0.55)
    parser.add_argument('--homo_branch_layers', type=int, default=2)
    parser.add_argument('--homo_branch_weight', type=float, default=0.35)
    parser.add_argument('--homo_branch_max_weight', type=float, default=0.65)
    parser.add_argument('--learn_homo_branch_weight', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--homo_aux_w', type=float, default=0.45)
    provided_args = collect_provided_args(sys.argv[1:])
    args = parser.parse_args()
    apply_homophily_defaults(args, provided_args)
    args.resolved_base_model = resolve_base_model(args)
    if args.resolved_base_model == 'gesc':
        apply_gesc_defaults(args, provided_args)
    #set_seed(args.seed)

    if not 0.0 <= args.eta_max <= 1.0:
        raise ValueError('--eta_max must be in [0, 1].')
    if not 0.0 <= args.lambda_attn <= 1.0:
        raise ValueError('--lambda_attn must be in [0, 1].')

    device = torch.device(args.device)

    dataset, data = load_dataset(args.data, device=None)
    is_ogb = hasattr(data, 'ogb_name')

    if args.mini_batch is None:
        args.mini_batch = bool(is_ogb)

    if is_ogb and args.ogb_safe_defaults:
        args.mini_batch = True
        args.use_cs = False
        args.use_lp = False
        args.consistency_w = 0.0
        args.checkpoint_layers = True
        args.eval_train = False
        args.eval_every = max(1, max(args.eval_every, 5))
        if args.num_neighbors is None:
            args.num_neighbors = '5,5' if args.data == 10 else '3,3'
        if args.eval_num_neighbors is None:
            args.eval_num_neighbors = args.num_neighbors
        if args.batch_size is None:
            args.batch_size = 512 if args.data == 10 else 256
        if args.eval_batch_size is None:
            args.eval_batch_size = 1024 if args.data == 10 else 512
        if args.edge_chunk_size and args.edge_chunk_size > 0:
            args.edge_chunk_size = min(args.edge_chunk_size, 25000)
        if args.static_chunk_size and args.static_chunk_size > 0:
            args.static_chunk_size = min(args.static_chunk_size, 50000)
        if args.empty_cache_steps == 0:
            args.empty_cache_steps = 0
        if args.attn_mode == 'auto':
            args.attn_mode = 'sigmoid_degree'
        print('[info] OGB safe defaults enabled: --mini_batch --no-use_cs --no-use_lp --consistency_w 0 --checkpoint_layers --attn_mode sigmoid_degree')

    if args.mini_batch:
        require_neighbor_loader()
        if args.use_preprop:
            raise ValueError('--use_preprop is full-graph propagation and is disabled in --mini_batch mode.')
        if args.use_homo_feature_boost:
            raise ValueError('--use_homo_feature_boost is full-graph propagation and is disabled in --mini_batch mode.')
        args.use_cs = False
        args.use_lp = False

    if args.batch_size is None:
        args.batch_size = 1024
    if args.eval_batch_size is None:
        args.eval_batch_size = max(1024, int(args.batch_size))
    if args.num_neighbors is None:
        args.num_neighbors = '10,10'
    if args.eval_num_neighbors is None:
        args.eval_num_neighbors = args.num_neighbors
    if args.attn_mode == 'auto':
        args.attn_mode = 'softmax'

    make_undirected = args.undirected
    if make_undirected is None:
        make_undirected = not is_ogb

    add_loops = args.add_self_loops
    if add_loops is None:
        add_loops = not (is_ogb and args.mini_batch)

    if make_undirected:
        ei = to_undirected(data.edge_index, num_nodes=data.num_nodes)
    else:
        ei = data.edge_index
    if add_loops:
        ei, _ = add_remaining_self_loops(ei, num_nodes=data.num_nodes)
    data.edge_index = ei.contiguous()

    in_dim = infer_num_node_features(dataset, data)
    num_classes = infer_num_classes(dataset, data)

    if args.mini_batch:
        model_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
    else:
        data = data.to(device)
        if args.use_homo_feature_boost:
            data.x = build_homophilic_features(data.x, data.edge_index, K=args.homo_prop_K, alpha=args.homo_prop_alpha, dropout=args.homo_prop_dropout, mode=args.homo_feature_mode, norm=args.homo_feature_norm)
        elif args.use_preprop:
            appnp = APPNP(K=args.preprop_K, alpha=args.preprop_alpha, dropout=args.preprop_dropout).to(device)
            appnp.eval()
            with torch.no_grad():
                data.x = appnp(data.x, data.edge_index)
        model_edge_index = data.edge_index

    in_dim = int(data.x.size(-1))

    if args.resolved_base_model == 'gesc':
        model = GESCSoftmaxNet(
            in_dim=in_dim,
            hidden_dim=args.hidden,
            num_classes=num_classes,
            edge_index=model_edge_index,
            num_nodes=data.num_nodes,
            num_heads=args.heads,
            gamma=args.gamma,
            attn_dropout=args.attn_dropout,
            feat_dropout=args.feat_dropout,
            readout_dropout=args.readout_dropout,
            layers=args.layers,
            sic_first=args.sic_first,
            alpha_skip=args.alpha_skip,
            jk_mode=args.jk,
            use_nodenorm=args.use_nodenorm,
        ).to(device)
    else:
        model = ASCiiNet(
            in_dim=in_dim,
            hidden_dim=args.hidden,
            num_classes=num_classes,
            edge_index=model_edge_index,
            num_nodes=data.num_nodes,
            num_heads=args.heads,
            gamma=args.gamma,
            attn_dropout=args.attn_dropout,
            feat_dropout=args.feat_dropout,
            readout_dropout=args.readout_dropout,
            layers=args.layers,
            eta_max=args.eta_max,
            eta_bar=args.eta_bar,
            lambda_attn=args.lambda_attn,
            alpha_skip=args.alpha_skip,
            jk_mode=args.jk,
            readout_rank=args.readout_rank,
            readout_hidden=args.readout_hidden,
            use_nodenorm=args.use_nodenorm,
            use_phase=args.use_phase,
            phase_scale=args.phase_scale,
            edge_chunk_size=args.edge_chunk_size,
            static_chunk_size=args.static_chunk_size,
            checkpoint_layers=args.checkpoint_layers,
            attn_mode=args.attn_mode,
            use_homo_branch=bool(args.use_homo_branch),
            homo_branch_hidden=args.homo_branch_hidden,
            homo_branch_dropout=args.homo_branch_dropout,
            homo_branch_layers=args.homo_branch_layers,
            homo_branch_weight=args.homo_branch_weight,
            homo_branch_max_weight=args.homo_branch_max_weight,
            learn_homo_branch_weight=args.learn_homo_branch_weight,
        ).to(device)

    print(f'[info] base_model={args.resolved_base_model} data={args.data} hidden={args.hidden} layers={args.layers} heads={args.heads} lr={args.lr} wd={args.weight_decay}')

    if not args.mini_batch and hasattr(model, 'prepare_static_graph'):
        model.prepare_static_graph(data.x, data.edge_index)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * args.cosine_min_lr_scale)
    best_val_acc = -1.0
    test_acc_at_best_val = 0.0
    best_epoch = 0
    best_state = None
    bad = 0
    metric_label = 'ROC-AUC' if getattr(data, 'eval_metric', 'acc') == 'rocauc' else 'Acc'

    if args.mini_batch:
        train_fanout = parse_num_neighbors(args.num_neighbors, args.layers)
        eval_fanout = parse_num_neighbors(args.eval_num_neighbors, args.layers)
        print(f'[info] Mini-batch mode: batch_size={args.batch_size}, eval_batch_size={args.eval_batch_size}, train_neighbors={train_fanout}, eval_neighbors={eval_fanout}, attn_mode={args.attn_mode}')

    for epoch in tqdm(range(1, args.epochs + 1)):
        if args.mini_batch:
            loss_value, ce_value, cons_value, eta_value = train_minibatch_epoch(model, data, args, device, epoch, opt, scheduler)
        else:
            model.train()
            ei1 = dropedge(data.edge_index, args.dropedge, training=True)
            logits1, eta_loss1 = model(data.x, edge_index_override=ei1, return_aux=True)
            task_type = getattr(data, 'task_type', 'multiclass')
            ce_main = supervised_loss(logits1, data.y, data.train_mask, task_type, label_smooth=args.label_smooth)
            homo_aux = homophily_auxiliary_loss(model, data.y, data.train_mask, task_type, label_smooth=args.label_smooth)
            ce = ce_main + args.homo_aux_w * homo_aux

            if args.consistency_w > 0.0:
                ei2 = dropedge(data.edge_index, args.dropedge, training=True)
                logits2, eta_loss2 = model(data.x, edge_index_override=ei2, return_aux=True)
                cons = prediction_consistency(logits1, logits2.detach(), task_type, T=args.cons_T)
                eta_loss = 0.5 * (eta_loss1 + eta_loss2)
            else:
                cons = logits1.new_tensor(0.0)
                eta_loss = eta_loss1

            loss = ce + args.consistency_w * cons + args.eta_reg * eta_loss
            if epoch <= args.warmup:
                loss = loss * (epoch / max(1, args.warmup))
            if not torch.isfinite(loss):
                print('[warn] non-finite loss detected; skipping step')
                opt.zero_grad(set_to_none=True)
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if not finite_gradients(model):
                print('[warn] non-finite grad detected; skipping step')
                opt.zero_grad(set_to_none=True)
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, error_if_nonfinite=False)
            opt.step()
            scheduler.step()
            loss_value = float(loss.detach().item())
            ce_value = float(ce.detach().item())
            cons_value = float(cons.detach().item())
            eta_value = float(eta_loss.detach().item())

        should_eval = (epoch == 1) or (epoch % max(1, args.eval_every) == 0) or (epoch == args.epochs)
        if should_eval:
            if args.mini_batch:
                train_acc, val_acc, test_acc, _ = evaluate_minibatch(model, data, args, device)
            else:
                train_acc, val_acc, test_acc, _ = evaluate(model, data, args)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_acc_at_best_val = test_acc
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
            print(f'[{epoch}] loss={loss_value:.4f} ce={ce_value:.4f} cons={cons_value:.4f} eta={eta_value:.4f} | train={train_acc:.3f} val={val_acc:.3f} test={test_acc:.3f} | best@val(epoch={best_epoch})={best_val_acc:.3f}/{test_acc_at_best_val:.3f}')
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            if bad >= args.patience:
                break
        else:
            print(f'[{epoch}] loss={loss_value:.4f} ce={ce_value:.4f} cons={cons_value:.4f} eta={eta_value:.4f} | eval skipped | best@val(epoch={best_epoch})={best_val_acc:.3f}/{test_acc_at_best_val:.3f}')

    if best_state is not None:
        model.load_state_dict(best_state)
    if args.mini_batch:
        final_train_acc, final_val_acc, final_test_acc, _ = evaluate_minibatch(model, data, args, device)
    else:
        final_train_acc, final_val_acc, final_test_acc, _ = evaluate(model, data, args)
    print(f'Best Val {metric_label}={best_val_acc:.4f}, Test {metric_label}@BestVal={test_acc_at_best_val:.4f}, Best Epoch={best_epoch}')
    print(f'Loaded Best-Val State: train={final_train_acc:.4f}, val={final_val_acc:.4f}, test={final_test_acc:.4f}')
if __name__ == '__main__':
    train_main()

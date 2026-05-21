import argparse
import copy
import math
import random
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.datasets import Planetoid, WikipediaNetwork, Actor, WebKB
from torch_geometric.nn import APPNP, LabelPropagation
from torch_geometric.nn.models import CorrectAndSmooth
from torch_geometric.utils import add_remaining_self_loops, degree, remove_self_loops, softmax, to_undirected
from tqdm import tqdm

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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

@torch.no_grad()
def normalize_feature_view(x: Tensor, eps: float=1e-12) -> Tensor:
    x = x.float()
    return F.normalize(x, p=2, dim=-1, eps=eps)

@torch.no_grad()
def compute_local_inconsistency(u: Tensor, edge_index: Tensor, num_nodes: int, eps: float=1e-12) -> Tensor:
    clean_edge_index, _ = remove_self_loops(edge_index)
    if clean_edge_index.numel() == 0:
        return torch.zeros(num_nodes, device=u.device, dtype=torch.float)
    src, dst = clean_edge_index
    u_norm = F.normalize(u.float(), p=2, dim=-1, eps=eps)
    sim = (u_norm[src] * u_norm[dst]).sum(dim=-1).clamp(min=-1.0, max=1.0)
    acc = torch.zeros(num_nodes, device=u.device, dtype=torch.float)
    cnt = torch.zeros(num_nodes, device=u.device, dtype=torch.float)
    acc.index_add_(0, dst, sim)
    cnt.index_add_(0, dst, torch.ones_like(sim))
    avg = acc / cnt.clamp_min(1.0)
    q = (1.0 - avg).clamp(min=0.0, max=2.0)
    q = torch.where(cnt > 0, q, torch.zeros_like(q))
    return q

class ASCiiLayer(nn.Module):

    def __init__(self, dim: int, num_heads: int=4, gamma: float=1.0, attn_dropout: float=0.2, eta_max: float=0.75, eta_bar: float=0.5, lambda_attn: float=0.5, layer_id: int=0, num_layers: int=1, use_activation: bool=True, use_nodenorm: bool=True, use_phase: bool=True, phase_scale: float=1.0, eps: float=1e-08):
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

    def forward(self, h: Tensor, edge_index: Tensor, x_view: Tensor, degree_log: Tensor, q_hat: Tensor, return_aux: bool=False):
        if edge_index.numel() == 0:
            h_next = self.act(self.nodenorm(h))
            zero = h.real.new_tensor(0.0)
            return (h_next, zero) if return_aux else h_next
        src, dst = edge_index
        N = h.size(0)
        h_src = h[src]
        h_dst = h[dst]
        delta_x, delta_d, qi, qj, layer_frac = self._edge_static_terms(edge_index=edge_index, x_view=x_view, degree_log=degree_log, q_hat=q_hat)
        phase_desc = torch.cat([delta_x, delta_d, qi, qj, layer_frac], dim=-1)
        h_dst_norm2 = (h_dst.real.square() + h_dst.imag.square()).sum(dim=1, keepdim=True)
        h_dst_norm2 = h_dst_norm2.clamp_min(self.eps)
        updates_sum = torch.zeros((N, self.out_dim), dtype=torch.cfloat, device=h.device)
        eta_penalties = []
        for m in range(self.M):
            Wh_src = nan_to_num_safe(self.W[m](h_src))
            if self.use_phase:
                theta = math.pi * self.phase_scale * torch.tanh(self.phase_nets[m](phase_desc))
                U = torch.complex(torch.cos(theta), torch.sin(theta))
                transported = Wh_src * U
            else:
                transported = Wh_src
            transported = nan_to_num_safe(transported)
            Qhi = nan_to_num_safe(self.Q[m](h_dst))
            s = torch.sum(torch.conj(Qhi) * transported, dim=1, keepdim=True)
            s = nan_to_num_safe(s)
            nu = torch.linalg.vector_norm(Qhi, dim=-1, keepdim=True) * torch.linalg.vector_norm(transported, dim=-1, keepdim=True) + self.eps
            rho = torch.real(s) / nu
            chi = torch.abs(s) / nu
            rho = torch.nan_to_num(rho, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=-1.0, max=1.0)
            chi = torch.nan_to_num(chi, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0, max=1.0)
            z = torch.cat([rho, chi, delta_x, delta_d, qi, qj, layer_frac], dim=-1)
            eta = self.eta_max * torch.sigmoid(self.eta_nets[m](z))
            eta = torch.nan_to_num(eta, nan=0.0, posinf=self.eta_max, neginf=0.0)
            eta_penalties.append((eta - self.eta_bar).square().mean())
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
            xi = torch.sigmoid(self.xi_scale[m] * rho_r + self.xi_bias[m])
            g = torch.sigmoid(self.mix_gate_nets[m](z))
            msg_hat = g.to(r.dtype) * xi.to(r.dtype) * r + (1.0 - g).to(transported.dtype) * transported
            msg_hat = nan_to_num_safe(msg_hat)
            s_tilde = torch.sum(torch.conj(Qhi) * msg_hat, dim=1, keepdim=True)
            denom_hat = torch.linalg.vector_norm(Qhi, dim=-1, keepdim=True) * torch.linalg.vector_norm(msg_hat, dim=-1, keepdim=True) + self.eps
            signed = torch.real(s_tilde) / denom_hat
            strength = torch.abs(s_tilde) / math.sqrt(max(1, self.out_dim))
            logits = self.gamma * (self.lambda_attn * strength + (1.0 - self.lambda_attn) * signed)
            logits = torch.nan_to_num(logits.squeeze(-1), nan=0.0, posinf=0.0, neginf=0.0)
            alpha = softmax(logits, dst, num_nodes=N)
            alpha = torch.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0)
            alpha = F.dropout(alpha, p=self.attn_dropout, training=self.training)
            msg = alpha.unsqueeze(-1).to(msg_hat.dtype) * msg_hat
            msg = nan_to_num_safe(msg)
            updates_sum.index_add_(0, dst, msg)
        h_new = h + updates_sum
        h_new = nan_to_num_safe(h_new)
        h_next = self.act(self.nodenorm(h_new))
        eta_penalty = torch.stack(eta_penalties).mean() if eta_penalties else h.real.new_tensor(0.0)
        return (h_next, eta_penalty) if return_aux else h_next

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

    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int, edge_index: Tensor, num_nodes: int, num_heads: int=4, gamma: float=0.1, attn_dropout: float=0.2, feat_dropout: float=0.5, readout_dropout: float=0.5, layers: int=2, eta_max: float=0.75, eta_bar: float=0.5, lambda_attn: float=0.5, alpha_skip: float=0.1, jk_mode: str='concat', readout_rank: int=16, readout_hidden: int=0, use_nodenorm: bool=True, use_phase: bool=True, phase_scale: float=1.0):
        super().__init__()
        assert jk_mode in ['last', 'concat', 'mean']
        self.edge_index = edge_index
        self.num_nodes = num_nodes
        self.num_classes = num_classes
        self.feat_drop = nn.Dropout(feat_dropout)
        self.alpha_skip = alpha_skip
        self.layers_num = layers
        self.jk_mode = jk_mode
        self.enc = ComplexLinear(in_dim, hidden_dim, bias=True)
        self.layers = nn.ModuleList([ASCiiLayer(hidden_dim, num_heads=num_heads, gamma=gamma, attn_dropout=attn_dropout, eta_max=eta_max, eta_bar=eta_bar, lambda_attn=lambda_attn, layer_id=i, num_layers=layers, use_activation=i < layers - 1, use_nodenorm=use_nodenorm, use_phase=use_phase, phase_scale=phase_scale) for i in range(layers)])
        readout_dim = hidden_dim * layers if jk_mode == 'concat' else hidden_dim
        self.readout = InvariantGramReadout(complex_dim=readout_dim, rank=readout_rank, num_classes=num_classes, dropout=readout_dropout, hidden=readout_hidden)
        self.register_buffer('x_view', torch.empty(0), persistent=False)
        self.register_buffer('degree_log', torch.empty(0), persistent=False)
        self.register_buffer('q_hat', torch.empty(0), persistent=False)

    @torch.no_grad()
    def prepare_static_graph(self, x_real: Tensor, edge_index: Optional[Tensor]=None, unary_view: Optional[Tensor]=None) -> None:
        ei = self.edge_index if edge_index is None else edge_index
        self.x_view = normalize_feature_view(x_real.detach()).to(x_real.device)
        deg = degree(ei[1], num_nodes=x_real.size(0), dtype=torch.float).to(x_real.device)
        self.degree_log = torch.log1p(deg)
        q_base = unary_view.detach().float() if unary_view is not None else self.x_view
        self.q_hat = compute_local_inconsistency(q_base, ei, num_nodes=x_real.size(0)).to(x_real.device)

    def _ensure_static_graph(self, x_real: Tensor) -> None:
        if self.x_view.numel() == 0 or self.x_view.size(0) != x_real.size(0):
            self.prepare_static_graph(x_real, self.edge_index)

    def forward(self, x_real: Tensor, edge_index_override: Optional[Tensor]=None, return_aux: bool=False):
        self._ensure_static_graph(x_real)
        x_dropped = self.feat_drop(x_real)
        x = x_dropped.to(torch.cfloat)
        h = self.enc(x)
        h0 = h.detach()
        ei = self.edge_index if edge_index_override is None else edge_index_override
        hs = []
        eta_losses = []
        for layer in self.layers:
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
        logits = self.readout(h_out)
        if return_aux:
            eta_reg = torch.stack(eta_losses).mean() if eta_losses else logits.new_tensor(0.0)
            return (logits, eta_reg)
        return logits

def load_dataset(data_id: int, device: Optional[torch.device]=None):
    if data_id == 0:
        dataset = Planetoid(root='/tmp/Cora', name='Cora')
    elif data_id == 1:
        dataset = Planetoid(root='/tmp/Citeseer', name='Citeseer')
    elif data_id == 2:
        dataset = Planetoid(root='/tmp/Pubmed', name='Pubmed')
    elif data_id == 3:
        dataset = WikipediaNetwork(root='/tmp/Chameleon', name='chameleon')
    elif data_id == 4:
        dataset = WikipediaNetwork(root='/tmp/Squirrel', name='squirrel')
    elif data_id == 5:
        dataset = Actor(root='/tmp/Actor')
    elif data_id == 6:
        dataset = WebKB(root='/tmp/Cornell', name='Cornell')
    elif data_id == 7:
        dataset = WebKB(root='/tmp/Texas', name='Texas')
    else:
        dataset = WebKB(root='/tmp/Wisconsin', name='Wisconsin')
    data = dataset[0]
    if data.train_mask.dim() == 2:
        data.train_mask = data.train_mask[:, 0]
        data.val_mask = data.val_mask[:, 0]
        data.test_mask = data.test_mask[:, 0]
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
    if args.use_cs:
        y_smooth = correct_and_smooth_compat(logits_eval, data.y, data.train_mask, data.edge_index, cs_corr_layers=args.cs_corr_layers, cs_corr_alpha=args.cs_corr_alpha, cs_smooth_layers=args.cs_smooth_layers, cs_smooth_alpha=args.cs_smooth_alpha, autoscale=True)
        pred_eval = (y_smooth + 1e-12).log()
    if args.use_lp:
        lp = LabelPropagation(num_layers=args.lp_layers, alpha=args.lp_alpha)
        y_lp = lp(data.y, data.edge_index, mask=data.train_mask)
        probs = F.softmax(pred_eval, dim=-1) * (1.0 - args.lp_blend) + y_lp * args.lp_blend
        pred_eval = (probs + 1e-12).log()
    pred = pred_eval.argmax(dim=1)
    train_acc = (pred[data.train_mask] == data.y[data.train_mask]).float().mean().item()
    val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean().item()
    test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()
    return (train_acc, val_acc, test_acc, pred_eval)

def train_main() -> None:
    parser = argparse.ArgumentParser(description='ASCii: Adaptive Gauge-Invariant Self-Interference Control')
    parser.add_argument('data', type=int, help='0=Cora,1=Citeseer,2=Pubmed,3=Chameleon,4=Squirrel,5=Actor,6=Cornell,7=Texas,else=Wisconsin')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
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
    args = parser.parse_args()
    if not 0.0 <= args.eta_max <= 1.0:
        raise ValueError('--eta_max must be in [0, 1].')
    if not 0.0 <= args.lambda_attn <= 1.0:
        raise ValueError('--lambda_attn must be in [0, 1].')
    set_seed(args.seed)
    device = torch.device(args.device)
    dataset, data = load_dataset(args.data, device=device)
    ei = to_undirected(data.edge_index, num_nodes=data.num_nodes)
    ei, _ = add_remaining_self_loops(ei, num_nodes=data.num_nodes)
    data.edge_index = ei.to(device)
    if args.use_preprop:
        appnp = APPNP(K=args.preprop_K, alpha=args.preprop_alpha, dropout=args.preprop_dropout).to(device)
        with torch.no_grad():
            data.x = appnp(data.x, data.edge_index)
    model = ASCiiNet(in_dim=dataset.num_node_features, hidden_dim=args.hidden, num_classes=dataset.num_classes, edge_index=data.edge_index, num_nodes=data.num_nodes, num_heads=args.heads, gamma=args.gamma, attn_dropout=args.attn_dropout, feat_dropout=args.feat_dropout, readout_dropout=args.readout_dropout, layers=args.layers, eta_max=args.eta_max, eta_bar=args.eta_bar, lambda_attn=args.lambda_attn, alpha_skip=args.alpha_skip, jk_mode=args.jk, readout_rank=args.readout_rank, readout_hidden=args.readout_hidden, use_nodenorm=args.use_nodenorm, use_phase=args.use_phase, phase_scale=args.phase_scale).to(device)
    model.prepare_static_graph(data.x, data.edge_index)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * args.cosine_min_lr_scale)
    best_val_acc = -1.0
    test_acc_at_best_val = 0.0
    best_epoch = 0
    best_state = None
    bad = 0
    for epoch in tqdm(range(1, args.epochs + 1)):
        model.train()
        ei1 = dropedge(data.edge_index, args.dropedge, training=True)
        ei2 = dropedge(data.edge_index, args.dropedge, training=True)
        logits1, eta_loss1 = model(data.x, edge_index_override=ei1, return_aux=True)
        logits2, eta_loss2 = model(data.x, edge_index_override=ei2, return_aux=True)
        ce = cross_entropy_with_label_smoothing(logits1[data.train_mask], data.y[data.train_mask], smoothing=args.label_smooth)
        cons = js_consistency(logits1, logits2.detach(), T=args.cons_T)
        eta_loss = 0.5 * (eta_loss1 + eta_loss2)
        loss = ce + args.consistency_w * cons + args.eta_reg * eta_loss
        if epoch <= args.warmup:
            loss = loss * (epoch / max(1, args.warmup))
        if not torch.isfinite(loss):
            print('[warn] non-finite loss detected; skipping step')
            opt.zero_grad(set_to_none=True)
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        finite_grad = True
        for p in model.parameters():
            if p.grad is not None and (not torch.isfinite(p.grad).all()):
                finite_grad = False
                break
        if not finite_grad:
            print('[warn] non-finite grad detected; skipping step')
            opt.zero_grad(set_to_none=True)
            continue
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, error_if_nonfinite=False)
        opt.step()
        scheduler.step()
        train_acc, val_acc, test_acc, _ = evaluate(model, data, args)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc_at_best_val = test_acc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
        print(f'[{epoch}] loss={loss.item():.4f} ce={ce.item():.4f} cons={cons.item():.4f} eta={eta_loss.item():.4f} | train={train_acc:.3f} val={val_acc:.3f} test={test_acc:.3f} | best@val(epoch={best_epoch})={best_val_acc:.3f}/{test_acc_at_best_val:.3f}')
        if bad >= args.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    final_train_acc, final_val_acc, final_test_acc, _ = evaluate(model, data, args)
    print(f'Best Val Acc={best_val_acc:.4f}, Test Acc@BestVal={test_acc_at_best_val:.4f}, Best Epoch={best_epoch}')
    print(f'Loaded Best-Val State: train={final_train_acc:.4f}, val={final_val_acc:.4f}, test={final_test_acc:.4f}')
if __name__ == '__main__':
    train_main()

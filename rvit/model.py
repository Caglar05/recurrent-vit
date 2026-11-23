"""One ViT class that covers four architectures from two papers.

  ViT / DeiT III  n_blocks == depth, n_experts == 1   (every step has its own block)
  bViT            n_blocks == 1,     n_experts == 1   (Byra et al., arXiv 2605.10661)
  reViT           n_blocks == 1,     n_experts  > 1   (Bulat et al., arXiv 2610.12448)
  partial sharing 1 < n_blocks < depth                (bViT App. A, Table 3/4)

`depth` is COMPUTATIONAL depth (number of block evaluations, the papers' L / T).
`n_blocks` is PARAMETER depth (number of independently parameterized block templates).
Step t uses template `t % n_blocks` — the stack of n_blocks templates is looped
depth/n_blocks times, matching bViT Table 3's "(2B,6S)" = 2 blocks x 6 steps.

reViT (n_experts = E > 1): the FFN of every template is replaced by a bank of E expert
FFNs. A tiny router MLP maps the normalized depth s_t = t/(depth-1) to gates
g_t = softmax(psi(s_t)); the experts' WEIGHTS are merged, W_t = sum_e g_t,e W_e, and
only the merged FFN is evaluated (reViT Eq. 1-3). The gate depends on depth only — the
same for every image and token — so this is not token routing.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    img_size: int = 224
    patch_size: int = 16
    num_classes: int = 1000
    embed_dim: int = 768
    num_heads: int = 12
    mlp_ratio: float = 4.0
    depth: int = 12  # block evaluations (computational depth)
    n_blocks: int = 1  # independent block templates (parameter depth); depth % n_blocks == 0
    n_experts: int = 1  # >1 = reViT depth-programmed FFN expert bank
    router_hidden: int = 16  # reViT App. A.1
    norm: str = "rms"  # "rms" (bViT) | "layer" (reViT, DeiT III)
    qkv_bias: bool = True
    layer_scale: float = 0.0  # 0 = off; DeiT III uses 1e-4
    drop_path: float = 0.0  # max stochastic-depth rate, linear over steps
    step_embed: bool = False  # bViT-TE (App. B Eq. 8): x_t + TE_t before each step
    num_registers: int = 0  # bViT-R (App. B)

    def __post_init__(self):
        assert self.depth % self.n_blocks == 0, "depth must be a multiple of n_blocks"
        assert self.embed_dim % self.num_heads == 0
        assert self.norm in ("rms", "layer")


def make_norm(kind: str, d: int) -> nn.Module:
    return nn.RMSNorm(d, eps=1e-6) if kind == "rms" else nn.LayerNorm(d, eps=1e-6)


def drop_path(x: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    if p == 0.0 or not training:
        return x
    keep = 1.0 - p
    mask = x.new_empty(x.shape[0], *([1] * (x.ndim - 1))).bernoulli_(keep)
    return x * mask / keep


class Attention(nn.Module):
    def __init__(self, d: int, heads: int, qkv_bias: bool):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(d, 3 * d, bias=qkv_bias)
        self.proj = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        q, k, v = self.qkv(x).view(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(q, k, v)
        return self.proj(o.transpose(1, 2).reshape(B, N, C))


class ExpertFFN(nn.Module):
    """GELU MLP stored as an expert bank of size E (E = 1 is a plain ViT FFN).

    With gate g (shape [E]) the expert weights AND biases are merged before use
    (reViT Eq. 2-3); only the merged FFN touches the tokens.
    """

    def __init__(self, d: int, hidden: int, n_experts: int):
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(n_experts, hidden, d))
        self.b1 = nn.Parameter(torch.zeros(n_experts, hidden))
        self.w2 = nn.Parameter(torch.empty(n_experts, d, hidden))
        self.b2 = nn.Parameter(torch.zeros(n_experts, d))
        nn.init.trunc_normal_(self.w1, std=0.02)
        nn.init.trunc_normal_(self.w2, std=0.02)

    def forward(self, x: torch.Tensor, gate: torch.Tensor | None) -> torch.Tensor:
        if gate is None:  # E == 1
            w1, b1, w2, b2 = self.w1[0], self.b1[0], self.w2[0], self.b2[0]
        else:
            g = gate.to(self.w1.dtype)
            w1 = torch.einsum("e,eij->ij", g, self.w1)
            b1 = g @ self.b1
            w2 = torch.einsum("e,eij->ij", g, self.w2)
            b2 = g @ self.b2
        return F.linear(F.gelu(F.linear(x, w1, b1)), w2, b2)

    def flat_experts(self) -> torch.Tensor:
        """[E, P]: each expert's full parameter vector q_e (reViT Eq. 9)."""
        return torch.cat(
            [self.w1.flatten(1), self.b1, self.w2.flatten(1), self.b2], dim=1
        )


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.embed_dim
        self.norm1 = make_norm(cfg.norm, d)
        self.attn = Attention(d, cfg.num_heads, cfg.qkv_bias)
        self.norm2 = make_norm(cfg.norm, d)
        self.ffn = ExpertFFN(d, int(d * cfg.mlp_ratio), cfg.n_experts)
        if cfg.layer_scale > 0:
            self.ls1 = nn.Parameter(torch.full((d,), cfg.layer_scale))
            self.ls2 = nn.Parameter(torch.full((d,), cfg.layer_scale))
        else:
            self.ls1 = self.ls2 = None

    def forward(self, x, gate: torch.Tensor | None, dp: float) -> torch.Tensor:
        a = self.attn(self.norm1(x))
        if self.ls1 is not None:
            a = a * self.ls1
        x = x + drop_path(a, dp, self.training)
        f = self.ffn(self.norm2(x), gate)
        if self.ls2 is not None:
            f = f * self.ls2
        return x + drop_path(f, dp, self.training)


class DepthRouter(nn.Module):
    """psi: R -> R^E, 2-layer MLP (16 hidden, SiLU), zero-init output => uniform gates."""

    def __init__(self, n_experts: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(1, hidden)
        self.fc2 = nn.Linear(hidden, n_experts)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, s: torch.Tensor) -> torch.Tensor:  # s: [L, 1] -> logits [L, E]
        return self.fc2(F.silu(self.fc1(s)))


class RecurrentViT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.embed_dim
        n_patches = (cfg.img_size // cfg.patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, d, cfg.patch_size, cfg.patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, d))
        self.reg_tokens = (
            nn.Parameter(torch.zeros(1, cfg.num_registers, d)) if cfg.num_registers else None
        )
        self.step_embed = (
            nn.Parameter(torch.zeros(cfg.depth, d)) if cfg.step_embed else None
        )
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_blocks))
        self.router = (
            DepthRouter(cfg.n_experts, cfg.router_hidden) if cfg.n_experts > 1 else None
        )
        self.norm = make_norm(cfg.norm, d)
        self.head = nn.Linear(d, cfg.num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.reg_tokens is not None:
            nn.init.trunc_normal_(self.reg_tokens, std=0.02)
        for m in self.blocks.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.zeros_(self.head.weight)  # torchvision ViT recipe
        nn.init.zeros_(self.head.bias)
        # depth coordinate s_t = t / (L-1), reViT Eq. 1
        s = torch.arange(cfg.depth, dtype=torch.float32) / max(cfg.depth - 1, 1)
        self.register_buffer("depth_coord", s[:, None], persistent=False)

    # ---------------------------------------------------------------- router
    def router_logits(self) -> torch.Tensor | None:
        if self.router is None:
            return None
        return self.router(self.depth_coord)  # [L, E]

    def gates(self) -> torch.Tensor | None:
        z = self.router_logits()
        return None if z is None else torch.softmax(z.float(), dim=-1)

    def aux_losses(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """reViT Eq. 6, 8, 9 (computed in fp32). Depend on parameters only, not data."""
        E = self.cfg.n_experts
        with torch.autocast(device_type=z.device.type, enabled=False):
            z = z.float()
            g = torch.softmax(z, dim=-1)
            ent = -(g * torch.log(g.clamp_min(1e-12))).sum(-1)
            bal = (1.0 - ent / math.log(E)).mean()
            zl = torch.logsumexp(z, dim=-1).pow(2).mean()
            div = z.new_zeros(())
            iu = torch.triu_indices(E, E, offset=1, device=z.device)
            for blk in self.blocks:
                q = F.normalize(blk.ffn.flat_experts().float(), dim=1)
                c = (q @ q.t())[iu[0], iu[1]]
                div = div + c.pow(2).mean()
            div = div / len(self.blocks)
        return {"bal": bal, "z": zl, "div": div}

    @torch.no_grad()
    def expert_weight_cos(self) -> float | None:
        """Max pairwise cosine between experts' WEIGHT matrices (w1, w2; biases excluded),
        averaged over block templates. Diagnostic only: L_div includes the biases, which
        have no weight decay, so it can look healthy while the weights are identical
        (-> 1.0 = experts collapsed into copies; router then gets no gradient)."""
        if self.router is None:
            return None
        vals = []
        for blk in self.blocks:
            f = blk.ffn
            q = F.normalize(torch.cat([f.w1.flatten(1), f.w2.flatten(1)], 1).float(), dim=1)
            c = q @ q.t()
            vals.append(c[~torch.eye(len(c), dtype=torch.bool, device=c.device)].max())
        return torch.stack(vals).mean().item()

    # ---------------------------------------------------------------- forward
    def forward(self, img: torch.Tensor, all_steps: bool = False):
        """Returns (logits, aux).

        logits: [B, C], or [depth, B, C] if all_steps (classifier applied to the CLS
        token after every step — bViT Fig. 2a "accuracy after each recurrent step").
        aux: reViT auxiliary losses ({} when n_experts == 1).
        """
        cfg = self.cfg
        x = self.patch_embed(img).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1) + self.pos_embed
        if self.reg_tokens is not None:
            x = torch.cat([x[:, :1], self.reg_tokens.expand(x.shape[0], -1, -1), x[:, 1:]], 1)

        z = self.router_logits()
        g = None if z is None else torch.softmax(z.float(), dim=-1)
        dpr = [cfg.drop_path * t / max(cfg.depth - 1, 1) for t in range(cfg.depth)]
        cls_per_step = []
        for t in range(cfg.depth):
            if self.step_embed is not None:
                x = x + self.step_embed[t]
            blk = self.blocks[t % cfg.n_blocks]
            x = blk(x, None if g is None else g[t], dpr[t])
            if all_steps:
                cls_per_step.append(x[:, 0])

        if all_steps:
            logits = self.head(self.norm(torch.stack(cls_per_step)))
        else:
            logits = self.head(self.norm(x[:, 0]))
        aux = self.aux_losses(z) if (z is not None and self.training) else {}
        return logits, aux


# ------------------------------------------------------------------ presets
_SIZES = {  # embed_dim, heads, depth  (ViT-S / B / L, patch 16)
    "s": (384, 6, 12),
    "b": (768, 12, 12),
    "l": (1024, 16, 24),
}
# DeiT III stochastic-depth rates (Touvron et al. 2022, IN-1k from scratch)
_DEIT3_DP = {"s": 0.05, "b": 0.1, "l": 0.4}


def model_preset(name: str) -> ModelConfig:
    """`<family>-<size>`, family in {vit, bvit, revit, deit3}, size in {s, b, l}."""
    family, size = name.lower().split("-")
    d, h, L = _SIZES[size]
    base = dict(embed_dim=d, num_heads=h, depth=L)
    if family == "vit":  # bViT paper's ViT baseline (RMSNorm, no stochastic depth)
        return ModelConfig(**base, n_blocks=L, norm="rms")
    if family == "bvit":
        return ModelConfig(**base, n_blocks=1, norm="rms")
    if family == "revit":
        return ModelConfig(**base, n_blocks=1, n_experts=4, norm="layer")
    if family == "deit3":  # reViT paper's baseline
        return ModelConfig(
            **base, n_blocks=L, norm="layer", layer_scale=1e-4, drop_path=_DEIT3_DP[size]
        )
    raise ValueError(f"unknown model family {family!r}")


MODEL_PRESETS = [f"{f}-{s}" for f in ("vit", "bvit", "revit", "deit3") for s in "sbl"]

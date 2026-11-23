# Methods

This page describes what `rvit/model.py` computes and how each part maps to the two papers.

## One class, two depths

A ViT has a **computational depth** *L*: how many transformer blocks a token passes
through. Normally *L* also equals the **parameter depth**: how many independently
parameterised blocks exist. `RecurrentViT` separates the two:

| field | meaning |
|---|---|
| `depth` | *L*, the number of block evaluations |
| `n_blocks` | *K*, the number of block templates (`depth % n_blocks == 0`) |

Step *t* = 0 … *L*−1 runs template *t* mod *K*, so the stack of *K* templates is looped
*L*/*K* times:

```
x_0     = [CLS; patches] + pos_embed
x_{t+1} = Block_{t mod K}(x_t)          t = 0 … L-1
logits  = head(norm(x_L[CLS]))
```

| *K* | model |
|---|---|
| *L* | standard ViT / DeiT III (no sharing) |
| 1 | **bViT**, a single block looped *L* times |
| 1 < *K* < *L* | partial sharing, e.g. bViT Table 3 "(2B, 6S)" = 2 blocks × 6 loops |

The block is a pre-norm transformer block: multi-head self-attention, then a GELU MLP with
ratio 4, each with a residual connection. Optional LayerScale and drop-path are only used
by the DeiT III baseline.

## bViT variants (Byra et al., App. B)

- **bViT-TE** (`step_embed=true`): a learned, zero-initialised vector *TE_t* is added to
  every token before step *t*, so the shared block can tell the steps apart
  (`x_t ← x_t + TE_t`).
- **bViT-R** (`num_registers=N`): *N* register tokens are appended after the CLS token.

## reViT: depth-programmed experts (Bulat et al.)

With `n_experts = E > 1`, the FFN of every template becomes a bank of *E* experts
`(W1_e, b1_e, W2_e, b2_e)`. A router ψ, a 2-layer MLP (1 → 16 → *E*, SiLU) with a
zero-initialised output, sees only the normalised depth:

```
s_t  = t / (L - 1)                      depth coordinate        (Eq. 1)
g_t  = softmax(ψ(s_t))                  gates, one per expert
W_t  = Σ_e g_{t,e} W_e   (same for b)   weight merge            (Eq. 2-3)
FFN_t(x) = W2_t · GELU(W1_t · x + b1_t) + b2_t
```

The experts are merged in **weight space**, and only the merged FFN touches the tokens.
The cost per token is that of one FFN, whatever *E* is. The gates depend on depth alone, so
they are the same for every image and token: this is depth programming, not token routing.
At initialisation the gates are uniform (1/*E*).

### Auxiliary losses

Added to the task loss with weights `lambda_bal` (1e-2), `lambda_z` (1e-3) and
`lambda_div` (1e-3). They depend only on parameters, not on the data:

| loss | formula | purpose |
|---|---|---|
| balance (Eq. 6) | mean_t [1 − H(g_t) / log E] | keeps the gates from turning one-hot too early |
| router z (Eq. 8) | mean_t [logsumexp(z_t)²] | keeps the router logits small |
| diversity (Eq. 9) | mean over expert pairs of cos²(q_i, q_j), q_e = flattened expert parameters | keeps the experts from becoming copies |

### Collapse diagnostics

Two extra quantities are logged at every log step:

- `experts/w_cos_max`: the largest pairwise cosine similarity between the experts' weight
  matrices (biases excluded). Values near 1 mean the experts have collapsed into copies, and
  the router then receives no useful gradient. The diversity loss includes the biases, so it
  can look healthy while the weights are identical. This metric catches that case.
- `experts/gate_max`: the largest gate over all steps. 1/*E* means the router is still
  uniform.

## Per-step evaluation

`model(x, all_steps=True)` applies the final norm and classifier to the CLS token after
every step and returns logits of shape `[L, B, C]`. Evaluation reports top-1 for each step,
which reproduces bViT Fig. 2a and shows how much each additional loop contributes.

## Implementation choices the papers leave open

| choice | ours | why |
|---|---|---|
| reViT qkv bias, pooling, expert init | qkv bias on, CLS pooling, trunc-normal(0.02) per expert | standard ViT / DeiT III defaults |
| reViT LayerScale, drop-path | off | not mentioned in the paper |
| sharing order for 1 < *K* < *L* | cyclic (the *K*-block stack is looped) | reading of bViT Table 3 "(2B, 6S)" |
| expert biases | merged with the same gates as the weights; no weight decay | Eq. 2-3 merge all expert parameters |

The reViT paper's B and L parameter counts are about 1 M above what its own stated shapes
give. Our S count matches its 6.1 M exactly.

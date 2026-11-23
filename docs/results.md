# Results

All runs train from scratch on ImageNet-1k at 224 px on 4× NVIDIA A100 (40 or 80 GB), with the
default recipe settings and seed 0. "Best" is the best validation top-1 over the whole run
(raw model or EMA, whichever is higher), evaluated every 10 epochs. These are single runs:
with one seed, differences under about half a point should not be over-read.

## Main table

| run | params | recipe | epochs | best top-1 | final top-1 | final top-5 | paper |
|---|---:|---|---:|---:|---:|---:|---:|
| ViT-S | 22.04 M | bvit | 300 | **79.64** | 79.41 | 94.25 | 78.2 ± 0.3 (bViT, Table 1) |
| bViT-S | 2.53 M | bvit | 300 | **68.38** | 68.35 | 88.75 | 68.1 ± 0.3 (bViT, Table 1) |
| DeiT III-S | 22.06 M | deit3 | 270 / 300 | 78.13 | 78.13 | 94.20 | 79.9 (reViT, Table 1) |
| reViT-S, *E* = 1 | 2.53 M | deit3 | 300 | 67.25 | 67.25 | 87.80 | – |

**bViT reproduces.** bViT-S lands within the paper's error bar: 68.4 against
68.1 ± 0.3. Sharing a single block costs about 11 points at S size, against a 9× reduction
in parameters.

**Our ViT-S baseline is stronger than the bViT paper's** (79.6 against 78.2), so the gap
between ViT-S and bViT-S is slightly larger here (11.3 against 10.1 points).

**DeiT III-S** was stopped at epoch 270 by a node failure, while its cosine schedule was
still climbing. 78.1 % is therefore a lower bound for this run.

**reViT-S with E = 1** is a single shared block with LayerNorm, trained with the DeiT III
recipe. It is effectively bViT-S with a different norm and recipe. It reaches 67.3, about one
point below bViT-S.

## Accuracy after each recurrent step

Top-1 when the classifier reads the CLS token after step *t*, at the final checkpoint:

| run | t = 9 | t = 10 | t = 11 | t = 12 |
|---|---:|---:|---:|---:|
| ViT-S | 17.0 | 39.1 | 70.3 | 79.4 |
| DeiT III-S | 4.4 | 31.5 | 70.5 | 78.1 |
| bViT-S | 4.2 | 18.2 | 59.2 | 68.4 |
| reViT-S, *E* = 1 | 15.8 | 39.7 | 59.3 | 67.2 |
| reViT-S, *E* = 4 (epoch 30, in progress) | 35.9 | 42.2 | 44.8 | 45.4 |

For the four completed runs, steps 1–6 stay near chance (≤ 0.4 %; chance is 0.1 %) and
accuracy builds up over the last four steps. The loss only sees the final step, so nothing
forces the intermediate CLS token to be linearly decodable.

The reViT model with four experts looks different, even this early in training. It already
reaches 8.0 % at step 6 and 25.9 % at step 8, and its last steps are close together
(44.8 → 45.4). One reading is that the depth-routed experts let the shared block refine its
prediction gradually rather than all at the end. That is a single run at epoch 30, though,
so treat it as an observation, not a result.

## reViT-S with E = 4: training stability

| run | recipe | status | top-1 |
|---|---|---|---|
| `revit-s_deit3` | deit3, as published | collapsed, stopped at epoch 150 | 0.5 % |
| `revit-s_deit3clip` | deit3 + gradient clipping 1.0 | stopped at epoch 26 | 14.4 % |
| `revit-s_bvit` | bvit | in progress, epoch 30 | 45.4 % (bViT-S at epoch 30: 43.4 %) |

With the published DeiT III recipe (peak lr 4e-3, BCE, no clipping), a loss spike around
step 2,300 made the four experts' weights identical (`experts/w_cos_max` = 1.000). From then
on the router got no gradient and stayed at uniform gates (0.25). The diversity loss then
pushed the experts' **biases** apart, which are not weight-decayed, while the weights stayed
identical. The model never recovered.

With the bViT recipe (lr 5e-4, clipping 1.0) the run trains normally. It is ahead of bViT-S
at the same epoch. `experts/w_cos_max` rises slowly (0.92 at epoch 30) while
`experts/gate_max` grows to 0.84, so the router does specialise the steps. Whether the
experts stay distinct for the whole schedule is still open. This table will be updated when
the run finishes.

We have not yet tracked down why the published recipe diverges here and not in the paper.
Candidate causes are the higher learning rate at batch 2048, the missing gradient clipping
and implementation details the paper does not specify (see [methods.md](methods.md)).

## Throughput and memory

Median training throughput on 4× A100, bf16 autocast, `torch.compile`, 256 images per GPU.
Runs landed on 40 GB and 80 GB nodes, so compare these as rough figures:

| run | images / s | peak memory per GPU |
|---|---:|---:|
| ViT-S (bvit recipe) | 5,690 | 9.4 GB |
| bViT-S (bvit recipe) | 6,540 | 8.9 GB |
| DeiT III-S (deit3 recipe) | 5,210 | 10.1 GB |
| reViT-S, *E* = 1 (deit3 recipe) | 6,120 | 9.0 GB |
| reViT-S, *E* = 4 (bvit recipe) | 6,740 | 9.0 GB |

The recurrent models compute as many FLOPs per image as their baselines, yet ran somewhat
faster in these runs. This is plausibly because there are ~10× fewer weights to read and to
update in the optimizer step; we have not profiled it. Merging four experts adds no
measurable cost.

## Reproducing the figure

```bash
python scripts/plot_results.py <out_dir>
```

This writes `assets/results-{light,dark}.png` from the runs' `metrics.jsonl` files.

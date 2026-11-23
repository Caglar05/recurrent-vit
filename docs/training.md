# Training guide

## Command line

```bash
python -m rvit.train --model <preset> --recipe <bvit|deit3> --run_name <name> [--key value ...]
```

- Every `TrainConfig` field is a flag (`--peak_lr 1e-3`).
- Every `ModelConfig` field is a flag with a `model.` prefix (`--model.n_experts 8`).
- Booleans accept `true/false`, `1/0` and `yes/no`.
- Precedence: dataclass default < recipe < command-line flag.

Multi-GPU runs use `torchrun` (or `python -m torch.distributed.run`) with the same flags.
The global batch is fixed by the recipe. Gradient accumulation is derived automatically as
`global_batch / (batch_per_gpu × world_size)`, so the same run works on 1, 2, 4 or 8 GPUs.
If you change `--global_batch`, the peak learning rate scales linearly
(`peak_lr × global_batch / lr_ref_batch`).

## Recipes

| | `bvit`: Byra et al. §3.2 | `deit3`: Bulat et al. Table 6 (shortened DeiT III) |
|---|---|---|
| optimiser | AdamW, wd 0.05 | AdamW, wd 0.05 |
| batch / peak lr | 1024 / 5e-4 | 2048 / 4e-3 (S), 3e-3 (B, L) |
| schedule | 30-epoch linear warmup from 0.033×, cosine to 0 | 5-epoch warmup from 1e-6, cosine to 1e-5 |
| epochs | 300 | 300 |
| loss | soft-target cross-entropy, label smoothing 0.1 | binary cross-entropy |
| mixup / cutmix | 0.8 / 1.0 | 0.8 / 1.0 |
| augmentation | RandomResizedCrop + flip + RandAugment(2, 9) + RandomErasing 0.25 | RandomResizedCrop + flip + 3-Augment + colour jitter 0.3 |
| repeated augmentation | – | ×3 |
| EMA | 0.99995 (torchvision semantics) | – |
| gradient clipping | 1.0 | off |
| eval | resize 256 → centre crop 224, bilinear | resize 224 → centre crop 224, bicubic |

Gradient clipping, RandomErasing and the eval crop for `bvit` follow the torchvision ViT
reference recipe, which the bViT paper builds on. For `deit3`, repeated augmentation keeps
the 3 copies on one rank and mixes them through the shuffle buffer. timm's `RASampler`
spreads them across ranks instead.

## Main options

| flag | default | meaning |
|---|---|---|
| `--train_dir`, `--val_dir` | `$RVIT_TRAIN_DIR`, `$RVIT_VAL_DIR` | dataset locations (see README) |
| `--out_dir` | `$RVIT_OUT_DIR` or `runs` | runs go to `<out_dir>/<run_name>/` |
| `--epochs`, `--max_steps` | 300, 0 | `max_steps > 0` overrides epochs (smoke tests) |
| `--batch_per_gpu` | 256 | micro-batch per GPU |
| `--num_workers`, `--shuffle_buffer` | 14, 2000 | loader workers per GPU, samples in the shuffle buffer |
| `--eval_every_epochs` | 10 | validation cadence |
| `--val_limit` | 0 | > 0 evaluates on an evenly spaced subset (smoke tests) |
| `--compile` | true | `torch.compile` the model |
| `--time_limit_min` | 0 | > 0: save and exit cleanly after this many minutes |
| `--save_every_min` | 30 | periodic `last.pt` |
| `--wandb_project`, `--wandb_entity` | empty | empty project = no wandb |
| `--lambda_bal`, `--lambda_z`, `--lambda_div` | 1e-2, 1e-3, 1e-3 | reViT auxiliary-loss weights |
| `--model.depth`, `--model.n_blocks`, `--model.n_experts` | preset | computational depth, weight sets, experts |
| `--model.step_embed`, `--model.num_registers` | false, 0 | bViT-TE, bViT-R |
| `--model.norm` | preset | `rms` or `layer` |
| `--model.layer_scale`, `--model.drop_path` | preset | DeiT III regularisers |

The full list is in `rvit/config.py` (`TrainConfig`) and `rvit/model.py` (`ModelConfig`).

## Outputs and resuming

```
<out_dir>/<run_name>/
├── config.json     full configuration (model + training)
├── metrics.jsonl   one JSON line per log step and per evaluation
├── last.pt         model, optimiser, EMA, step, best score, wandb id
├── best.pt         weights of the best validation top-1 so far (model or EMA)
└── DONE            written when the schedule finishes
```

Running the same command again resumes from `last.pt`. The trainer refuses to resume if a
setting that changes the experiment differs from the checkpoint. Only infrastructure
settings may change: paths, batch per GPU, workers, logging, time limit, wandb. The data
order after a resume is not bit-identical, because each chunk reseeds its shuffle from its
start step.

Evaluation lines contain `val/top1`, `val/top5`, `val/top1_per_step`, the EMA equivalents
and, for reViT, the current gates. Training lines contain the loss, learning rate, gradient
norm, throughput, peak memory, the auxiliary losses and the expert-collapse diagnostics.

## SLURM

`slurm/launch.sh` submits a run as an array of short chunks that run one after another.
Each chunk resumes from `last.pt` and saves before its time limit, so long runs fit short
queues.

```bash
cp slurm/site.env.example slurm/site.env      # account, partition, data paths, Python, proxy
MODEL=bvit-b RECIPE=bvit SEED=0 bash slurm/launch.sh
MODEL=bvit-b RECIPE=bvit BENCH=1 bash slurm/launch.sh                  # 30-min throughput/memory probe
MODEL=revit-b RECIPE=deit3 EXTRA_ARGS="--model.n_experts 8" RUN_NAME=revit-b-e8 bash slurm/launch.sh
```

| variable | default | meaning |
|---|---|---|
| `NGPU`, `GPU_TYPE` | 4, any | GPUs per job (`--gpus-per-node=<GPU_TYPE>:<NGPU>`) |
| `CHUNKS`, `TIME` | 30, 2 h | array size and wall time per chunk |
| `TIME_LIMIT_MIN` | 100 | trainer saves and exits after this |
| `MEM`, `CPUS` | 400G, 16 × NGPU | memory and CPUs per job |
| `OUT_DIR`, `WANDB_PROJECT` | `./runs`, `recurrent-vit` | outputs, wandb project (`""` disables it) |
| `EXTRA_ARGS` | – | extra trainer flags |

Surplus chunks exit within seconds once the run has written `DONE`. To extend a run that
stopped early, submit it again with the same `RUN_NAME`.

## Tests

```bash
pytest
```

The tests run on CPU in about 30 seconds. They cover:

- parameter counts against both papers, including bViT's partial-sharing table
- the cyclic template schedule
- per-step logits
- uniform gates at init and the auxiliary-loss values
- weight merging with identical experts equal to a dense FFN
- gradients reaching every parameter
- weight-decay groups and the learning-rate schedule
- CLI precedence
- shard partitioning and repeated augmentation in the parquet stream

"""Training config, the two paper recipes, and a flat CLI.

    python -m rvit.train --model bvit-b --recipe bvit --run_name X [--key value ...]

Every TrainConfig field is a flag (`--peak_lr 1e-3`), every ModelConfig field is a flag
with a `model.` prefix (`--model.n_experts 8`). Order of precedence:
dataclass default < recipe < CLI flag.
"""

import argparse
import os
from dataclasses import asdict, dataclass, fields, replace

from .data import DEFAULT_TRAIN_DIR, DEFAULT_VAL_DIR
from .model import MODEL_PRESETS, ModelConfig, model_preset


@dataclass
class TrainConfig:
    run_name: str = ""
    out_dir: str = os.environ.get("RVIT_OUT_DIR", "runs")  # runs go to <out_dir>/<run_name>/
    train_dir: str = DEFAULT_TRAIN_DIR
    val_dir: str = DEFAULT_VAL_DIR
    # schedule
    epochs: float = 300
    max_steps: int = 0  # >0 overrides epochs (smoke tests)
    global_batch: int = 1024
    batch_per_gpu: int = 256  # grad accumulation = global / (batch_per_gpu * world)
    peak_lr: float = 5e-4
    lr_ref_batch: int = 1024  # peak_lr is defined at this global batch; scaled linearly
    min_lr: float = 0.0
    warmup_epochs: float = 30
    warmup_start_factor: float = 0.033  # warmup starts at peak * this
    # optimizer
    weight_decay: float = 0.05
    beta1: float = 0.9
    beta2: float = 0.999
    grad_clip: float = 1.0  # 0 = off
    # loss / regularization
    loss: str = "ce"  # "ce" (soft-target CE) | "bce" (DeiT III)
    label_smoothing: float = 0.1
    mixup: float = 0.8
    cutmix: float = 1.0
    lambda_bal: float = 1e-2  # reViT aux losses (ignored when n_experts == 1)
    lambda_z: float = 1e-3
    lambda_div: float = 1e-3
    # data
    train_aug: str = "randaug"  # "randaug" (bViT) | "3aug" (DeiT III / reViT)
    interpolation: str = "bilinear"
    rand_erase: float = 0.25
    color_jitter: float = 0.3  # used by 3aug only
    repeated_aug: int = 1
    eval_crop_ratio: float = 0.875
    num_workers: int = 14
    shuffle_buffer: int = 2000
    val_limit: int = 0  # >0: evaluate on an evenly spaced val subset (smoke tests)
    # EMA (torchvision reference semantics: update every ema_every steps)
    ema_decay: float = 0.99995  # 0 = off
    ema_every: int = 32
    # infra
    seed: int = 0
    compile: bool = True
    log_every: int = 100
    eval_every_epochs: float = 10
    save_every_min: float = 30
    time_limit_min: float = 0  # >0: save + exit cleanly after this much wall time
    wandb_project: str = ""
    wandb_entity: str = ""


def recipe(name: str, model_name: str) -> dict:
    """Recipe overrides on top of TrainConfig defaults (which ARE the bViT recipe)."""
    size = model_name.split("-")[1]
    if name == "bvit":
        # Byra et al. Sec 3.2: AdamW lr 5e-4 wd 0.05, 30-ep linear warmup from 0.033x, cosine
        # to 0, label smoothing 0.1, mixup 0.8 / cutmix 1.0, batch 4x256, EMA 0.99995,
        # RRC(0.08-1) + flip + RandAugment + RandomErasing, no stochastic depth.
        return {}
    if name == "deit3":
        # Bulat et al. Table 6 (reViT, "shortened DeiT III recipe"): batch 2048, AdamW
        # wd 0.05, 5-ep warmup from 1e-6, cosine, BCE, no label smoothing, 3-Augment +
        # color jitter 0.3, mixup 0.8 / cutmix 1.0, repeated aug, no random erasing,
        # eval crop ratio 1.0.
        peak = {"s": 4e-3, "b": 3e-3, "l": 3e-3}[size]
        return dict(
            global_batch=2048, lr_ref_batch=2048, peak_lr=peak, min_lr=1e-5,
            warmup_epochs=5, warmup_start_factor=1e-6 / peak, grad_clip=0.0,
            loss="bce", label_smoothing=0.0, train_aug="3aug", interpolation="bicubic",
            rand_erase=0.0, repeated_aug=3, eval_crop_ratio=1.0, ema_decay=0.0,
        )
    raise ValueError(f"unknown recipe {name!r}")


RECIPES = ["bvit", "deit3"]


def _str2bool(v: str) -> bool:
    if v.lower() in ("1", "true", "yes", "y"):
        return True
    if v.lower() in ("0", "false", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"not a bool: {v}")


def _add(p: argparse.ArgumentParser, flag: str, dest: str, ftype: type):
    typ = _str2bool if ftype is bool else ftype  # declared type: `epochs: float = 300` -> float
    p.add_argument(flag, dest=dest, type=typ, default=None, metavar=ftype.__name__.upper())


def parse_args(argv=None) -> tuple[TrainConfig, ModelConfig, str, str]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=MODEL_PRESETS)
    p.add_argument("--recipe", required=True, choices=RECIPES)
    for f in fields(TrainConfig):
        _add(p, f"--{f.name}", f"t__{f.name}", f.type)
    for f in fields(ModelConfig):
        _add(p, f"--model.{f.name}", f"m__{f.name}", f.type)
    a = vars(p.parse_args(argv))

    t = replace(TrainConfig(), **recipe(a["recipe"], a["model"]))
    t = replace(t, **{k[3:]: v for k, v in a.items() if k.startswith("t__") and v is not None})
    m = replace(model_preset(a["model"]),
                **{k[3:]: v for k, v in a.items() if k.startswith("m__") and v is not None})
    if not t.run_name:
        p.error("--run_name is required")
    return t, m, a["model"], a["recipe"]


def to_dict(t: TrainConfig, m: ModelConfig, model_name: str, recipe_name: str) -> dict:
    return {"model_preset": model_name, "recipe": recipe_name, "train": asdict(t), "model": asdict(m)}

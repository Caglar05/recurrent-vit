"""Supervised ImageNet-1k training for ViT / bViT / reViT / DeiT III.

Single GPU / CPU:  python -m rvit.train --model bvit-s --recipe bvit --run_name dbg ...
Multi GPU:         python -m torch.distributed.run --standalone --nproc_per_node 4 \
                       -m rvit.train --model bvit-b --recipe bvit --run_name bvit-b-s0

Resumable: if <out_dir>/<run_name>/last.pt exists, training continues from it. With
--time_limit_min the job saves and exits cleanly, so a SLURM array of short jobs
(`--array=0-N%1`) chains into one long run. A finished run writes a DONE marker.

Outputs in <out_dir>/<run_name>/: config.json, metrics.jsonl (every log/eval line),
last.pt (full state), best.pt (weights of the best val top-1 so far, model or EMA).
"""

import json
import math
import os
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from .config import TrainConfig, parse_args, to_dict
from .data import N_TRAIN, ParquetStream, eval_transform, train_files, train_transform, val_subset
from .model import RecurrentViT

# fields that may change between resumed chunks without changing the experiment
_INFRA_FIELDS = {
    "out_dir", "train_dir", "val_dir", "batch_per_gpu", "num_workers", "shuffle_buffer",
    "val_limit", "compile", "log_every", "eval_every_epochs", "save_every_min",
    "time_limit_min", "wandb_project", "wandb_entity",
}


def param_groups(model: torch.nn.Module, wd: float):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        leaf = n.rsplit(".", 1)[-1]
        if p.ndim < 2 or leaf in ("bias", "b1", "b2") or n in (
            "pos_embed", "cls_token", "reg_tokens", "step_embed"
        ):
            no_decay.append(p)
        else:
            decay.append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


def lr_at(step: int, t: TrainConfig, peak: float, warmup: int, total: int) -> float:
    if step < warmup:
        f = t.warmup_start_factor
        return peak * (f + (1 - f) * step / warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return t.min_lr + (peak - t.min_lr) * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))


def soft_targets(y: torch.Tensor, n_cls: int, smoothing: float) -> torch.Tensor:
    off = smoothing / n_cls
    return torch.full((y.shape[0], n_cls), off, device=y.device).scatter_(
        1, y[:, None], 1.0 - smoothing + off)


def task_loss(logits: torch.Tensor, target: torch.Tensor, kind: str) -> torch.Tensor:
    logits = logits.float()
    if kind == "bce":
        return F.binary_cross_entropy_with_logits(logits, target)
    return torch.sum(-target * F.log_softmax(logits, dim=-1), dim=-1).mean()


@torch.no_grad()
def evaluate(net: RecurrentViT, loader, device, amp) -> dict:
    """Top-1 after EVERY recurrent step (bViT Fig. 2a) + final top-5, all-reduced."""
    was_training = net.training
    net.eval()
    L = net.cfg.depth
    correct = torch.zeros(L, device=device, dtype=torch.float64)
    top5 = torch.zeros((), device=device, dtype=torch.float64)
    n = torch.zeros((), device=device, dtype=torch.float64)
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with amp:
            logits, _ = net(x, all_steps=True)  # [L, B, C]
        correct += (logits.argmax(-1) == y[None]).sum(1).double()
        top5 += (logits[-1].topk(5, -1).indices == y[:, None]).any(-1).sum().double()
        n += y.numel()
    if dist.is_initialized():
        for v in (correct, top5, n):
            dist.all_reduce(v)
    net.train(was_training)
    per_step = (correct / n).tolist()
    return {"top1": per_step[-1], "top5": (top5 / n).item(), "top1_per_step": per_step,
            "n": int(n.item())}


@torch.no_grad()
def ema_update(ema: torch.nn.Module, model: torch.nn.Module, alpha: float):
    e = [p for p in ema.parameters()]
    m = [p.detach() for p in model.parameters()]
    torch._foreach_lerp_(e, m, alpha)


def atomic_save(obj, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main(argv=None):
    t, mcfg, model_name, recipe_name = parse_args(argv)

    # ------------------------------------------------------------ distributed
    distributed = "RANK" in os.environ
    if distributed:
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
        rank, world = dist.get_rank(), dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank, world, local_rank = 0, 1, 0
    cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{local_rank}" if cuda else "cpu")
    if cuda:
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if cuda else nullcontext()
    is_main = rank == 0

    def log(msg: str):
        if is_main:
            print(msg, flush=True)

    # ------------------------------------------------------------ schedule
    micro = t.batch_per_gpu * world
    assert t.global_batch % micro == 0, f"global_batch {t.global_batch} not divisible by {micro}"
    accum = t.global_batch // micro
    steps_per_epoch = N_TRAIN / t.global_batch
    total = t.max_steps or round(t.epochs * steps_per_epoch)
    warmup = round(t.warmup_epochs * steps_per_epoch)
    if t.max_steps:
        warmup = min(warmup, max(1, total // 10))
    eval_every = max(1, round(t.eval_every_epochs * steps_per_epoch))
    peak = t.peak_lr * t.global_batch / t.lr_ref_batch

    run_dir = Path(t.out_dir) / t.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "DONE").exists():
        log(f"[rvit] {run_dir} already DONE — nothing to do")
        return

    # ------------------------------------------------------------ model / opt
    torch.manual_seed(t.seed)
    model = RecurrentViT(mcfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    ema = None
    if t.ema_decay > 0:
        ema = RecurrentViT(mcfg).to(device)
        ema.load_state_dict(model.state_dict())
        ema.requires_grad_(False).eval()
    # torchvision reference EMA: alpha = (1-decay) * global_batch * ema_every / epochs
    ema_alpha = min(1.0, (1 - t.ema_decay) * t.global_batch * t.ema_every / t.epochs)
    opt = torch.optim.AdamW(param_groups(model, t.weight_decay), lr=peak,
                            betas=(t.beta1, t.beta2), fused=cuda)

    start_step, best, wandb_id = 0, -1.0, None
    cfg_dict = to_dict(t, mcfg, model_name, recipe_name)
    ckpt_path = run_dir / "last.pt"
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        old = ck["config"]
        if old["model"] != asdict(mcfg):
            raise SystemExit(f"[rvit] model config differs from {ckpt_path}; refusing to resume")
        diff = {k for k in old["train"] if k not in _INFRA_FIELDS and old["train"][k] != asdict(t)[k]}
        if diff:
            raise SystemExit(f"[rvit] train config differs from checkpoint in {sorted(diff)}")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        if ema is not None:
            ema.load_state_dict(ck["ema"])
        start_step, best, wandb_id = ck["step"], ck["best"], ck.get("wandb_id")
        log(f"[rvit] resumed from {ckpt_path} at step {start_step}/{total}")
        del ck
    elif is_main:
        (run_dir / "config.json").write_text(json.dumps(cfg_dict, indent=2))

    log(f"[rvit] {model_name} / {recipe_name}: {n_params / 1e6:.2f}M params | depth "
        f"{mcfg.depth}, blocks {mcfg.n_blocks}, experts {mcfg.n_experts}, norm {mcfg.norm}")
    log(f"[rvit] world {world} x batch {t.batch_per_gpu} x accum {accum} = {t.global_batch}; "
        f"{total} steps ({steps_per_epoch:.1f}/epoch), warmup {warmup}, peak lr {peak:.2e}")

    train_model = DDP(model, device_ids=[local_rank] if cuda else None) if distributed else model
    if t.compile:
        train_model = torch.compile(train_model)

    # ------------------------------------------------------------ data
    torch.manual_seed(t.seed * 1_000_003 + rank + 7919 * start_step)  # worker aug RNG
    tf = train_transform(t.train_aug, mcfg.img_size, t.interpolation, t.rand_erase, t.color_jitter)
    stream = ParquetStream(train_files(t.train_dir), tf, seed=f"{t.seed}-{start_step}",
                           rank=rank, world=world, repeats=t.repeated_aug, buffer=t.shuffle_buffer)
    loader = DataLoader(stream, batch_size=t.batch_per_gpu, num_workers=t.num_workers,
                        pin_memory=cuda, drop_last=True,
                        persistent_workers=t.num_workers > 0,
                        prefetch_factor=4 if t.num_workers > 0 else None)
    val_ds, n_val = val_subset(t.val_dir, eval_transform(mcfg.img_size, t.eval_crop_ratio, t.interpolation),
                               rank, world, t.val_limit)
    val_loader = DataLoader(val_ds, batch_size=min(256, t.batch_per_gpu * 2),
                            num_workers=min(8, t.num_workers), pin_memory=cuda)

    mixup = None
    if t.mixup > 0 or t.cutmix > 0:
        from timm.data import Mixup
        mixup = Mixup(mixup_alpha=t.mixup, cutmix_alpha=t.cutmix,
                      label_smoothing=t.label_smoothing, num_classes=mcfg.num_classes)

    # ------------------------------------------------------------ logging
    wb = None
    if is_main and t.wandb_project:
        import wandb
        wandb_id = wandb_id or wandb.util.generate_id()
        wb = wandb.init(project=t.wandb_project, entity=t.wandb_entity or None, name=t.run_name,
                        id=wandb_id, resume="allow", config=cfg_dict,
                        dir=os.environ.get("WANDB_DIR", str(run_dir)))
        wb.summary["n_params"] = n_params
    metrics_f = open(run_dir / "metrics.jsonl", "a") if is_main else None

    def emit(rec: dict):
        if not is_main:
            return
        metrics_f.write(json.dumps(rec) + "\n")
        metrics_f.flush()
        if wb is not None:
            wb.log({k: v for k, v in rec.items() if not isinstance(v, list)}, step=rec["step"])

    def save(step: int):
        if not is_main:
            return
        atomic_save({"model": model.state_dict(), "opt": opt.state_dict(),
                     "ema": ema.state_dict() if ema is not None else None,
                     "step": step, "best": best, "wandb_id": wandb_id, "config": cfg_dict},
                    ckpt_path)

    def run_eval(step: int):
        nonlocal best
        rec = {"step": step, "epoch": step / steps_per_epoch, "kind": "eval"}
        r = evaluate(model, val_loader, device, amp)
        rec.update({"val/top1": r["top1"], "val/top5": r["top5"], "val/top1_per_step": r["top1_per_step"]})
        cand = {"model": r["top1"]}
        if ema is not None:
            re = evaluate(ema, val_loader, device, amp)
            rec.update({"val_ema/top1": re["top1"], "val_ema/top5": re["top5"],
                        "val_ema/top1_per_step": re["top1_per_step"]})
            cand["ema"] = re["top1"]
        which = max(cand, key=cand.get)
        if model.router is not None:
            rec["gates"] = model.gates().tolist()
        rec["val/best_top1"] = best = max(best, cand[which])
        if is_main and cand[which] >= best:
            src = ema if which == "ema" else model
            atomic_save({"model": src.state_dict(), "which": which, "step": step, "top1": cand[which],
                         "config": cfg_dict}, run_dir / "best.pt")
        log(f"[eval] step {step} ep {rec['epoch']:.1f}: top1 {r['top1']:.4f}"
            + (f" | ema {cand['ema']:.4f}" if "ema" in cand else "") + f" | best {best:.4f}"
            + f" | per-step {' '.join(f'{a:.3f}' for a in r['top1_per_step'])} (n={r['n']})")
        emit(rec)

    # ------------------------------------------------------------ train loop
    model.train()
    it = iter(loader)
    t0 = last_log_t = last_save_t = time.time()
    last_log_step = step = start_step
    stop = False
    while step < total and not stop:
        lr = lr_at(step, t, peak, warmup, total)
        for g in opt.param_groups:
            g["lr"] = lr
        loss_acc = torch.zeros((), device=device)
        aux_acc: dict[str, torch.Tensor] = {}
        for _ in range(accum):
            x, y = next(it)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if mixup is not None:
                x, tgt = mixup(x, y)
            else:
                tgt = soft_targets(y, mcfg.num_classes, t.label_smoothing)
            with amp:
                logits, aux = train_model(x)
            loss = task_loss(logits, tgt, t.loss)
            if aux:
                loss = loss + t.lambda_bal * aux["bal"] + t.lambda_z * aux["z"] + t.lambda_div * aux["div"]
                for k, v in aux.items():
                    aux_acc[k] = v.detach()
            (loss / accum).backward()
            loss_acc += loss.detach() / accum
        # clip_grad_norm_ with max_norm=inf only measures (always logged)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               t.grad_clip if t.grad_clip > 0 else float("inf"))
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1

        if ema is not None:
            if step <= warmup:  # torchvision resets the EMA during warmup
                ema.load_state_dict(model.state_dict())
            elif step % t.ema_every == 0:
                ema_update(ema, model, ema_alpha)

        if step % t.log_every == 0 or step == start_step + 1:
            now = time.time()
            ips = (step - last_log_step) * t.global_batch / max(now - last_log_t, 1e-9)
            mean_loss = loss_acc.clone()
            if distributed:
                dist.all_reduce(mean_loss, op=dist.ReduceOp.AVG)
            rec = {"step": step, "epoch": step / steps_per_epoch, "kind": "train", "loss": mean_loss.item(),
                   "lr": lr, "grad_norm": float(gnorm), "img_per_s": ips,
                   **{f"aux/{k}": v.item() for k, v in aux_acc.items()}}
            if model.router is not None:  # collapse diagnostics (see expert_weight_cos)
                rec["experts/w_cos_max"] = model.expert_weight_cos()
                rec["experts/gate_max"] = model.gates().max().item()
            if cuda:
                rec["mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
            mem = f" mem {rec['mem_gb']:.1f}GB" if cuda else ""
            log(f"[train] step {step}/{total} ep {rec['epoch']:.2f} loss {rec['loss']:.4f} lr {lr:.2e} "
                f"gn {rec['grad_norm']:.2f} {ips:.0f} img/s{mem}")
            emit(rec)
            last_log_t, last_log_step = now, step

        if step % eval_every == 0 or step == total:
            run_eval(step)

        if is_main and t.save_every_min and time.time() - last_save_t > t.save_every_min * 60:
            save(step)
            last_save_t = time.time()

        if t.time_limit_min and step % 10 == 0:
            flag = torch.tensor(float(time.time() - t0 > t.time_limit_min * 60), device=device)
            if distributed:
                dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            stop = bool(flag.item())

    save(step)
    if step >= total and is_main:
        (run_dir / "DONE").write_text(f"finished at step {step}, best top1 {best:.4f}\n")
        log(f"[rvit] DONE: best top1 {best:.4f}")
    elif is_main:
        log(f"[rvit] time limit reached at step {step}/{total}; checkpoint saved, resubmit to continue")
    if wb is not None:
        wb.finish()
    if metrics_f is not None:
        metrics_f.close()
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

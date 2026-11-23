"""ImageNet-1k input pipeline.

Train: the HuggingFace `ILSVRC/imagenet-1k` parquet shards (image bytes + int label),
streamed by an infinite IterableDataset. Labels 0..999 follow sorted-synset order,
which is exactly the class order torchvision's ImageFolder gives the val folder
(checked: `classes.py` key order == sorted(os.listdir(val_dir))).

Val: the raw synset-named ImageFolder (50 000 images).
"""

import io
import os
import random
from pathlib import Path

import pyarrow.parquet as pq
import torch
from PIL import Image, ImageFilter, ImageOps
from torch.utils.data import IterableDataset, get_worker_info
from torchvision import datasets
from torchvision import transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
N_TRAIN = 1_281_167

# Dataset locations: `--train_dir` / `--val_dir`, or these environment variables.
DEFAULT_TRAIN_DIR = os.environ.get("RVIT_TRAIN_DIR", "")
DEFAULT_VAL_DIR = os.environ.get("RVIT_VAL_DIR", "")

_INTERP = {"bilinear": T.InterpolationMode.BILINEAR, "bicubic": T.InterpolationMode.BICUBIC}


class ThreeAugment:
    """DeiT III 3-Augment: one of {grayscale, solarize, gaussian blur}, uniformly."""

    def __call__(self, img: Image.Image) -> Image.Image:
        op = random.randrange(3)
        if op == 0:
            return ImageOps.grayscale(img).convert("RGB")
        if op == 1:
            return ImageOps.solarize(img, threshold=128)
        return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 2.0)))


def train_transform(aug: str, img_size: int, interpolation: str, rand_erase: float,
                    color_jitter: float) -> T.Compose:
    interp = _INTERP[interpolation]
    ops: list = [
        T.RandomResizedCrop(img_size, scale=(0.08, 1.0), ratio=(3 / 4, 4 / 3), interpolation=interp),
        T.RandomHorizontalFlip(),
    ]
    if aug == "randaug":  # bViT: torchvision reference recipe (RandAugment, 2 ops, magnitude 9)
        ops.append(T.RandAugment(num_ops=2, magnitude=9, interpolation=interp))
    elif aug == "3aug":  # reViT: DeiT III recipe
        ops.append(ThreeAugment())
        if color_jitter > 0:
            ops.append(T.ColorJitter(color_jitter, color_jitter, color_jitter))
    else:
        raise ValueError(f"unknown train aug {aug!r}")
    ops += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if rand_erase > 0:
        ops.append(T.RandomErasing(p=rand_erase, value="random"))
    return T.Compose(ops)


def eval_transform(img_size: int, crop_ratio: float, interpolation: str) -> T.Compose:
    return T.Compose([
        T.Resize(int(round(img_size / crop_ratio)), interpolation=_INTERP[interpolation]),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def train_files(train_dir: str) -> list[str]:
    if not train_dir:
        raise ValueError("no training data: pass --train_dir or set RVIT_TRAIN_DIR "
                         "(a directory of ImageNet-1k train-*.parquet shards)")
    files = sorted(str(p) for p in Path(train_dir).glob("train-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no train-*.parquet under {train_dir}")
    return files


class ParquetStream(IterableDataset):
    """Infinite, shuffled stream of (image, label) over parquet shards.

    Every pass p shuffles the shard list with the same seed on all consumers and gives
    consumer c (= rank * num_workers + worker) the shards c, c+n, c+2n, ... Row groups
    are visited in random order and samples go through a shuffle buffer. Because the
    stream never ends, ranks never run out at different times (no DDP hang), and one
    "epoch" is simply N_TRAIN samples worth of steps.

    `repeats` > 1 = repeated augmentation (DeiT III): each image is yielded `repeats`
    times with independent augmentations. Unlike timm's RASampler the copies stay on one
    rank, but the shuffle buffer scatters them over different batches.
    """

    def __init__(self, files: list[str], transform, seed: str, rank: int, world: int,
                 repeats: int = 1, buffer: int = 2000):
        self.files, self.transform, self.seed = files, transform, seed
        self.rank, self.world, self.repeats, self.buffer = rank, world, repeats, buffer

    def _decode(self, b: bytes):
        return self.transform(Image.open(io.BytesIO(b)).convert("RGB"))

    def __iter__(self):
        wi = get_worker_info()
        nw, wid = (wi.num_workers, wi.id) if wi else (1, 0)
        consumer, n = self.rank * nw + wid, self.world * nw
        rng = random.Random(f"{self.seed}-local-{consumer}")
        buf: list[tuple[bytes, int]] = []
        p = 0
        while True:
            order = list(self.files)
            random.Random(f"{self.seed}-pass-{p}").shuffle(order)
            mine = order[consumer::n] or [order[consumer % len(order)]]
            for f in mine:
                pf = pq.ParquetFile(f)
                rgs = list(range(pf.num_row_groups))
                rng.shuffle(rgs)
                for rg in rgs:
                    t = pf.read_row_group(rg, columns=["image", "label"])
                    imgs = t.column("image").combine_chunks().field("bytes").to_pylist()
                    labels = t.column("label").to_pylist()
                    for item in zip(imgs, labels, strict=True):
                        if len(buf) < self.buffer:
                            buf.append(item)
                            continue
                        i = rng.randrange(len(buf))
                        (b, y), buf[i] = buf[i], item
                        for _ in range(self.repeats):
                            yield self._decode(b), y
            p += 1


def val_subset(val_dir: str, transform, rank: int, world: int, limit: int = 0):
    if not val_dir:
        raise ValueError("no validation data: pass --val_dir or set RVIT_VAL_DIR "
                         "(an ImageFolder with one sub-directory per synset)")
    ds = datasets.ImageFolder(val_dir, transform=transform)
    idx = list(range(len(ds)))
    if limit:  # evenly spaced subset (covers all classes) — smoke tests only
        idx = idx[:: max(1, len(idx) // limit)][:limit]
    return torch.utils.data.Subset(ds, idx[rank::world]), len(idx)

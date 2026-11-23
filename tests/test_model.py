import math

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from rvit.config import parse_args
from rvit.data import ParquetStream
from rvit.model import ModelConfig, RecurrentViT, model_preset
from rvit.train import lr_at, param_groups


def n_params(name: str, **kw) -> float:
    cfg = model_preset(name)
    for k, v in kw.items():
        setattr(cfg, k, v)
    with torch.device("meta"):
        m = RecurrentViT(cfg)
    return sum(p.numel() for p in m.parameters()) / 1e6


# Paper-reported parameter counts (M). bViT Table 1; reViT Table 1 (S only — the paper's
# B/L counts are ~1M above what its own stated shapes give, see README).
@pytest.mark.parametrize("name,paper", [
    ("vit-s", 22.0), ("vit-b", 86.6), ("vit-l", 304.3),
    ("bvit-s", 2.5), ("bvit-b", 8.6), ("bvit-l", 14.6),
    ("revit-s", 6.1), ("deit3-s", 22.1), ("deit3-b", 86.6),
])
def test_param_counts_match_papers(name, paper):
    assert n_params(name) == pytest.approx(paper, rel=0.02)


def test_partial_sharing_counts_match_bvit_table3():
    # bViT App. A Table 3: ViT-S (2B,6S) 4.3M, (3B,4S) 6.1M, (6B,2S) 11.4M
    for k, paper in [(2, 4.3), (3, 6.1), (6, 11.4)]:
        assert n_params("vit-s", n_blocks=k) == pytest.approx(paper, rel=0.03)


def tiny(**kw) -> ModelConfig:
    base = dict(img_size=32, patch_size=8, num_classes=10, embed_dim=32, num_heads=2, depth=4)
    base.update(kw)
    return ModelConfig(**base)


def test_all_steps_last_equals_final():
    torch.manual_seed(0)
    m = RecurrentViT(tiny(n_experts=3, step_embed=True, num_registers=2)).eval()
    torch.nn.init.normal_(m.head.weight)  # head is zero-init
    x = torch.randn(2, 3, 32, 32)
    final, aux = m(x)
    per_step, _ = m(x, all_steps=True)
    assert aux == {}  # aux losses only in training mode
    assert per_step.shape == (4, 2, 10)
    torch.testing.assert_close(per_step[-1], final)


def test_cyclic_template_schedule():
    m = RecurrentViT(tiny(depth=6, n_blocks=2))
    calls = []
    for i, b in enumerate(m.blocks):
        b.register_forward_hook(lambda mod, inp, out, i=i: calls.append(i))
    m(torch.randn(1, 3, 32, 32))
    assert calls == [0, 1, 0, 1, 0, 1]


def test_router_starts_uniform_and_aux_losses():
    m = RecurrentViT(tiny(n_experts=4)).train()
    g = m.gates()
    torch.testing.assert_close(g, torch.full((4, 4), 0.25))
    _, aux = m(torch.randn(2, 3, 32, 32))
    assert aux["bal"].item() == pytest.approx(0.0, abs=1e-6)  # max entropy
    assert aux["z"].item() == pytest.approx(math.log(4) ** 2, rel=1e-5)
    assert 0.0 <= aux["div"].item() < 0.05  # random high-dim experts ~ orthogonal


def test_weight_merge_with_identical_experts_equals_dense_ffn():
    torch.manual_seed(0)
    dense = RecurrentViT(tiny(n_experts=1)).eval()
    moe = RecurrentViT(tiny(n_experts=3)).eval()
    sd = dense.state_dict()
    for k, v in moe.state_dict().items():
        if ".ffn." in k:
            v.copy_(sd[k].expand_as(v))  # every expert = the dense FFN
        elif k in sd:
            v.copy_(sd[k])
    torch.nn.init.normal_(moe.router.fc2.weight)  # arbitrary non-uniform gates
    torch.nn.init.normal_(dense.head.weight)
    moe.head.weight.data.copy_(dense.head.weight)
    x = torch.randn(2, 3, 32, 32)
    torch.testing.assert_close(moe(x)[0], dense(x)[0], rtol=1e-4, atol=1e-5)


def test_gradients_reach_every_parameter():
    m = RecurrentViT(tiny(n_experts=2, n_blocks=2, step_embed=True, layer_scale=1e-4)).train()
    torch.nn.init.normal_(m.head.weight)
    torch.nn.init.normal_(m.router.fc2.weight)
    logits, aux = m(torch.randn(2, 3, 32, 32))
    (logits.square().mean() + sum(aux.values())).backward()
    missing = [n for n, p in m.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
    assert missing == []


def test_weight_decay_groups():
    m = RecurrentViT(tiny(n_experts=2))
    decay, no_decay = param_groups(m, 0.05)
    names = {id(p): n for n, p in m.named_parameters()}
    nd = {names[id(p)] for p in no_decay["params"]}
    d = {names[id(p)] for p in decay["params"]}
    assert "pos_embed" in nd and "cls_token" in nd and "blocks.0.ffn.b1" in nd
    assert "blocks.0.ffn.w1" in d and "blocks.0.attn.qkv.weight" in d
    assert not any(n.endswith("bias") for n in d)


def test_lr_schedule():
    t, *_ = parse_args(["--model", "bvit-s", "--recipe", "bvit", "--run_name", "x"])
    peak, warm, total = 5e-4, 100, 1000
    assert lr_at(0, t, peak, warm, total) == pytest.approx(peak * 0.033)
    assert lr_at(warm, t, peak, warm, total) == pytest.approx(peak)
    assert lr_at(total, t, peak, warm, total) == pytest.approx(0.0, abs=1e-12)


def test_cli_precedence():
    t, m, *_ = parse_args(["--model", "revit-b", "--recipe", "deit3", "--run_name", "x",
                           "--peak_lr", "1e-3", "--model.n_experts", "8", "--compile", "false"])
    assert t.loss == "bce" and t.global_batch == 2048  # recipe
    assert t.peak_lr == 1e-3 and t.compile is False  # CLI beats recipe
    assert m.n_experts == 8 and m.norm == "layer" and m.depth == 12


def _toy_shards(tmp_path, n_files=3, rows=10):
    import io

    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, format="JPEG")
    files = []
    for f in range(n_files):
        labels = [f * 100 + i for i in range(rows)]
        tbl = pa.table({"image": [{"bytes": buf.getvalue(), "path": ""}] * rows, "label": labels})
        p = tmp_path / f"train-{f:05d}.parquet"
        pq.write_table(tbl, p, row_group_size=4)
        files.append(str(p))
    return files


def test_parquet_stream_partitions_shards(tmp_path):
    files = _toy_shards(tmp_path)
    seen = []
    for rank in range(3):
        s = ParquetStream(files, transform=lambda im: 0, seed="t", rank=rank, world=3, buffer=1)
        it = iter(s)
        seen.append({next(it)[1] // 100 for _ in range(9)})  # one shard's worth (buffer=1)
    # each rank reads exactly one distinct shard in the first pass
    assert all(len(x) == 1 for x in seen) and len(set.union(*seen)) == 3


def test_parquet_stream_repeats(tmp_path):
    files = _toy_shards(tmp_path, n_files=1)
    s = ParquetStream(files, transform=lambda im: 0, seed="t", rank=0, world=1, repeats=3, buffer=2)
    it = iter(s)
    ys = [next(it)[1] for _ in range(9)]
    assert ys[0] == ys[1] == ys[2] and ys[3] == ys[4] == ys[5]

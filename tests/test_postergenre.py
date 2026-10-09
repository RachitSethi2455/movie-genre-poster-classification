"""Fast offline tests: random-init models and synthetic images, with no Kaggle or torchvision downloads."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from postergenre.data import PosterDataset, eval_tf, stratified_split, train_tf
from postergenre.models import (BACKBONES, build_model, count_params, load_trained_state, param_groups,
                                trained_state, unfreeze_last_block)
from postergenre.train import fit, predict

MODELS = Path(__file__).resolve().parents[1] / "models"


@pytest.mark.parametrize("name", list(BACKBONES))
def test_backbone_head_shape_and_freezing(name):
    model = build_model(name, num_classes=4, pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 4)
    trainable, total = count_params(model)
    assert 0 < trainable < 0.01 * total  # only the new head trains


def test_unfreeze_last_block_and_param_groups():
    model = build_model("ResNet18", pretrained=False)
    head_only = count_params(model)[0]
    unfreeze_last_block(model, "ResNet18")
    assert count_params(model)[0] > 100 * head_only
    groups = param_groups(model, "ResNet18", head_lr=1e-3, body_lr=1e-5)
    assert [g["lr"] for g in groups] == [1e-3, 1e-5]
    assert sum(p.numel() for g in groups for p in g["params"]) == count_params(model)[0]


def test_trained_state_roundtrip_is_small():
    torch.manual_seed(0)
    a = build_model("ResNet18", pretrained=False)
    b = copy.deepcopy(a)
    torch.nn.init.normal_(b.fc.weight)
    state = trained_state(a)
    assert sum(v.numel() for v in state.values()) < 0.02 * sum(p.numel() for p in a.parameters())
    load_trained_state(b, state)
    assert torch.equal(a.fc.weight, b.fc.weight)


def test_stratified_split_is_disjoint_stratified_and_reproducible():
    paths = [f"img_{i}.jpg" for i in range(400)]
    labels = [i % 4 for i in range(400)]
    split = stratified_split(paths, labels, seed=3)
    parts = [set(split[k][0]) for k in ("train", "val", "test")]
    assert sum(map(len, parts)) == 400 and not (parts[0] & parts[1] or parts[0] & parts[2] or parts[1] & parts[2])
    assert np.bincount(split["test"][1]).tolist() == [15, 15, 15, 15]
    assert split["test"][0] == stratified_split(paths, labels, seed=3)["test"][0]
    assert split["test"][0] != stratified_split(paths, labels, seed=4)["test"][0]


def synthetic_dataset(n=16):
    rng = np.random.default_rng(0)
    paths = [Path(f"poster_{i}.jpg") for i in range(n)]
    cache = {p: Image.fromarray(rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)) for p in paths}
    return paths, [i % 4 for i in range(n)], cache


def test_dataset_and_transforms():
    paths, labels, cache = synthetic_dataset()
    x, y = PosterDataset(paths, labels, cache, train_tf)[0]
    assert x.shape == (3, 224, 224) and x.dtype == torch.float32 and y == 0
    assert eval_tf(Image.new("RGB", (300, 450))).shape == (3, 224, 224)  # posters are resized, any aspect ratio


def test_fit_restores_best_checkpoint_and_predicts():
    torch.manual_seed(0)
    paths, labels, cache = synthetic_dataset()
    loader = DataLoader(PosterDataset(paths, labels, cache, eval_tf), batch_size=8)
    model = build_model("ResNet18", pretrained=False)
    history, best = fit(model, loader, loader, "cpu", param_groups(model, "ResNet18", 1e-2, 0), epochs=2, verbose=False)
    assert best["val_loss"] == min(history["val_loss"])
    y_true, y_pred, probs = predict(model, loader, "cpu")
    assert probs.shape == (16, 4) and np.allclose(probs.sum(1), 1, atol=1e-5)


@pytest.mark.skipif(not (MODELS / "poster_genre.pth").exists(), reason="run the notebook to create models/")
def test_saved_demo_model_and_app():
    import app

    card = json.loads((MODELS / "model_card.json").read_text(encoding="utf-8"))
    model = app.load_model(pretrained=False)  # random backbone; checks that the checkpoint matches the architecture
    probs = app.classify(Image.new("RGB", (300, 450), "red"), model=model)
    assert list(probs) == card["classes"] and abs(sum(probs.values()) - 1) < 1e-5
    probs2, heat = app.explain(Image.new("RGB", (300, 450), "red"), model=model)
    assert probs2 == pytest.approx(probs, abs=1e-5)
    assert heat.shape == (450, 300, 3) and heat.dtype == np.uint8  # overlay keeps the poster's own shape


def test_gradcam_heatmaps_are_normalised_and_hooks_removed():
    from postergenre.gradcam import GradCAM, overlay

    torch.manual_seed(0)
    model = build_model("ResNet18", pretrained=False).eval()
    x = torch.randn(2, 3, 224, 224)
    with GradCAM(model, model.layer4) as cam:
        heat, probs = cam(x)
    assert heat.shape == (2, 224, 224) and np.isfinite(heat).all()
    assert heat.min() >= 0 and heat.max() <= 1 + 1e-6
    assert probs.shape == (2, 4) and np.allclose(probs.sum(1), 1, atol=1e-5)
    assert not model.layer4._forward_hooks  # context manager cleaned up
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    blended = overlay(Image.new("RGB", (224, 224), "white"), heat[0])
    assert blended.shape == (224, 224, 3) and blended.dtype == np.uint8

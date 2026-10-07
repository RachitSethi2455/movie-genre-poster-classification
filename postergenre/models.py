"""Backbones with a new classification head, and lightweight checkpoints."""
import torch
import torch.nn as nn
from torchvision import models

BACKBONES = {
    "ResNet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
    "ResNet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
    "DenseNet121": (models.densenet121, models.DenseNet121_Weights.IMAGENET1K_V1),
    "VGG16": (models.vgg16, models.VGG16_Weights.IMAGENET1K_V1),
}
LAST_BLOCK = {"ResNet18": "layer4", "ResNet50": "layer4", "DenseNet121": "features.denseblock4", "VGG16": "features.28"}
HEAD = {"ResNet18": "fc", "ResNet50": "fc", "DenseNet121": "classifier", "VGG16": "classifier.6"}


def build_model(name, num_classes=4, pretrained=True):
    """Pretrained backbone with every weight frozen and a new trainable linear head."""
    ctor, weights = BACKBONES[name]
    model = ctor(weights=weights if pretrained else None)
    for p in model.parameters():
        p.requires_grad = False
    if name.startswith("ResNet"):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "DenseNet121":
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif name == "VGG16":
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    return model


def unfreeze_last_block(model, name):
    for pname, p in model.named_parameters():
        if pname.startswith(LAST_BLOCK[name]):
            p.requires_grad = True
    return model


def param_groups(model, name, head_lr, body_lr):
    """Separate learning rates for the new head and any unfrozen backbone weights."""
    head, body = [], []
    for pname, p in model.named_parameters():
        if p.requires_grad:
            (head if pname.startswith(HEAD[name]) else body).append(p)
    return [g for g in ({"params": head, "lr": head_lr}, {"params": body, "lr": body_lr}) if g["params"]]


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad), sum(p.numel() for p in model.parameters())


def trained_state(model):
    """Only what training changed: trainable weights and BatchNorm buffers. The rest comes from torchvision."""
    keep = {n for n, p in model.named_parameters() if p.requires_grad} | {n for n, _ in model.named_buffers()}
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if k in keep}


def load_trained_state(model, state):
    if not isinstance(state, dict):
        state = torch.load(state, map_location="cpu")
    result = model.load_state_dict(state, strict=False)
    assert not result.unexpected_keys, f"checkpoint does not match model: {result.unexpected_keys[:3]}"
    return model

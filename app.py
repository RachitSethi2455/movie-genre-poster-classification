"""Gradio demo: upload a movie poster and get genre probabilities.

Run locally with ``python app.py`` after the notebook has saved ``models/``. On first start, torchvision
downloads the pretrained backbone weights. Only the trained layers are stored in this repo.
"""
import json
from functools import lru_cache
from pathlib import Path

import gradio as gr
import torch

from postergenre.data import eval_tf
from postergenre.models import build_model, load_trained_state

MODEL_DIR = Path(__file__).parent / "models"
card = json.loads((MODEL_DIR / "model_card.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_model(pretrained=True):
    model = build_model(card["backbone"], len(card["classes"]), pretrained=pretrained)
    load_trained_state(model, torch.load(MODEL_DIR / "poster_genre.pth", map_location="cpu"))
    return model.float().eval()


@torch.no_grad()
def classify(image, model=None):
    """PIL image -> {genre: probability}."""
    model = model or load_model()
    probs = model(eval_tf(image.convert("RGB")).unsqueeze(0)).softmax(1).squeeze(0)
    return {genre: float(p) for genre, p in zip(card["classes"], probs)}


demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil", label="Movie poster"),
    outputs=gr.Label(num_top_classes=4, label="Predicted genre"),
    title="Movie genre from poster",
    description=(
        f"{card['backbone']}{' (fine-tuned)' if card['fine_tuned'] else ' (frozen backbone + linear head)'} "
        f"trained on 1,325 IMDB posters. It averages about {card['test_acc_mean_5_splits']:.0f}% test accuracy "
        "across 5 splits, where random guessing would score 25%. "
        "It only knows Action, Comedy, Horror and Romance."
    ),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()

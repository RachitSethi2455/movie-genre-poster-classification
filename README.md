# Movie Genre Classification from Posters

Classifies a movie poster into one of four genres (**Action, Comedy, Horror, Romance**) using transfer learning in PyTorch. It compares four ImageNet-pretrained CNN backbones under the same training setup.

## Approach

- **Data** – poster images in `train / val / test` folders loaded with `ImageFolder`. The test set has 200 images.
- **Preprocessing** – resize to 224×224 and apply ImageNet normalization. Training-time augmentation: random horizontal flip, ±10° rotation and colour jitter.
- **Transfer learning** – freeze the pretrained backbone and replace the final classification layer with a 4-class head.
- **Backbones compared** – ResNet18, ResNet50, DenseNet121, VGG16.
- **Training** – Adam, cross-entropy loss, 8 epochs, batch size 32 (lr 1e-3; 5e-4 for VGG16), on a Colab GPU.
- **Evaluation** – test accuracy, per-class precision/recall/F1, a confusion matrix, and accuracy/loss curves.

## Results

| Backbone | Best val acc | Test acc |
|---|---|---|
| **ResNet50** | 68.7% | **69.0%** |
| DenseNet121 | 69.7% | 64.0% |
| VGG16 | 64.7% | 57.5% |
| ResNet18 | 65.2% | — |

The best model, **ResNet50**, reached **69% test accuracy (0.69 macro-F1)**. That's about 2.75× the 25% random baseline for 4 classes.

Per-class results for ResNet50:

| Genre | Precision | Recall | F1 |
|---|---|---|---|
| Action | 0.61 | 0.74 | 0.67 |
| Comedy | 0.76 | 0.57 | 0.65 |
| Horror | 0.80 | 0.73 | 0.77 |
| Romance | 0.62 | 0.71 | 0.66 |

Horror is the easiest genre to identify, probably because of its distinctive dark palette. Comedy has the lowest recall and is often confused with Romance.

VGG16 overfit clearly: train accuracy reached 88% while val accuracy fell to about 61%. This is because its large fully connected head was retrained from scratch.

> **Known issue:** in the test-evaluation cell, the ResNet18 entry is computed with the DenseNet model (`test_accuracy(model_densenet)`), so its reported 64% is DenseNet's number. The ResNet18 test score needs a re-run. The other backbones are unaffected.

## Run it

```bash
pip install -r requirements.txt
jupyter notebook movie_genre_classification.ipynb
```

The notebook expects the dataset as `four_genre_posters_split.zip` containing `train/`, `val/` and `test/` subfolders with one folder per genre. The dataset and trained weights (`.pth`) aren't included in this repo.

## Next steps

- Unfreeze and fine-tune the last backbone blocks with a lower learning rate
- Early stopping and learning-rate scheduling (VGG16 overfit after about 4 epochs)
- More data per genre, or multi-label genres (real posters often belong to more than one)
- A small Gradio/FastAPI demo for poster upload → genre prediction

## Tech

Python · PyTorch · torchvision · scikit-learn · seaborn · matplotlib

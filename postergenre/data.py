"""Poster download, reproducible splits, an in-memory image cache and transforms."""
import shutil
from pathlib import Path

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

KAGGLE_DATASET = "zulkarnainsaurav/four-genre-movie-poster-images"
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}
IMG_SIZE = 224
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def find_genre_root(base):
    """Return the first directory whose sub-folders all contain images (one folder per genre)."""
    base = Path(base)
    for d in [base, *sorted(p for p in base.rglob("*") if p.is_dir())]:
        subs = [s for s in d.iterdir() if s.is_dir()]
        if len(subs) >= 2 and all(any(f.suffix.lower() in IMG_EXT for f in s.iterdir()) for s in subs):
            return d
    raise FileNotFoundError(f"No genre folders found under {base}")


def download_posters(raw_dir="data/four_genre_posters"):
    """Return the folder of genre sub-folders, downloading it from Kaggle on first use."""
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        import kagglehub
        downloaded = Path(kagglehub.dataset_download(KAGGLE_DATASET))
        # The archive also ships a "four_genre_posters_updated" variant; this project uses the original 1,325 images.
        shutil.copytree(find_genre_root(downloaded / "four_genre_posters"), raw_dir)
    return find_genre_root(raw_dir)


def list_images(root):
    """Return sorted image paths, their integer labels, and the class names (sorted genre folder names)."""
    root = Path(root)
    classes = sorted(d.name for d in root.iterdir() if d.is_dir())
    paths = sorted(f for f in root.rglob("*") if f.suffix.lower() in IMG_EXT)
    labels = [classes.index(p.parent.name) for p in paths]
    return paths, labels, classes


def stratified_split(paths, labels, seed, val=0.15, test=0.15):
    """Stratified train/val/test split -> {"train": (paths, labels), "val": ..., "test": ...}."""
    tr_p, rest_p, tr_y, rest_y = train_test_split(paths, labels, test_size=val + test, stratify=labels, random_state=seed)
    va_p, te_p, va_y, te_y = train_test_split(rest_p, rest_y, test_size=test / (val + test), stratify=rest_y, random_state=seed)
    return {"train": (tr_p, tr_y), "val": (va_p, va_y), "test": (te_p, te_y)}


def load_image(path, size=IMG_SIZE):
    return Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)


class ImageCache(dict):
    """Decode and resize every poster once. 1,325 posters at 224×224 RGB is about 200 MB."""

    def __init__(self, paths, size=IMG_SIZE):
        super().__init__((Path(p), load_image(p, size)) for p in paths)


class PosterDataset(Dataset):
    def __init__(self, paths, labels, cache, transform):
        self.paths, self.labels, self.cache, self.transform = [Path(p) for p in paths], list(labels), cache, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.transform(self.cache[self.paths[i]]), self.labels[i]


def _finish(*ops):
    return transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), *ops, transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


# Stage-1 augmentation (unchanged from the original project).
train_tf = _finish(transforms.RandomHorizontalFlip(), transforms.RandomRotation(10),
                   transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))
# Stronger augmentation for fine-tuning, where overfitting is the main risk.
finetune_tf = _finish(transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)), transforms.RandomHorizontalFlip(),
                      transforms.RandomRotation(10), transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3))
eval_tf = _finish()

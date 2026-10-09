"""Grad-CAM: which parts of a poster drive the predicted genre."""
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    """Grad-CAM (Selvaraju et al., 2017) on one convolutional layer.

    It weights each channel of the layer's activation map by the average gradient of the target class score,
    then keeps only the positive evidence. Use it as a context manager so the hooks are removed afterwards.
    """

    def __init__(self, model, layer):
        self.model, self.layer = model, layer
        self._acts = self._grads = None
        self._hooks = [layer.register_forward_hook(self._save_acts)]

    def _save_acts(self, module, inputs, output):
        self._acts = output
        output.register_hook(self._save_grads)

    def _save_grads(self, grad):
        self._grads = grad

    def __call__(self, x, class_idx=None):
        """x: (N, 3, H, W) normalized batch -> (heatmaps (N, H, W) in [0, 1], probabilities (N, classes)).

        ``class_idx`` defaults to each image's predicted class. Gradients are enabled locally,
        so the weights stay frozen and are never updated.
        """
        self.model.eval()
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)  # lets gradients reach frozen layers' activations
            logits = self.model(x)
            idx = logits.argmax(1) if class_idx is None else torch.as_tensor(class_idx, device=x.device).expand(len(x))
            self.model.zero_grad(set_to_none=True)
            logits.gather(1, idx.view(-1, 1)).sum().backward()
        weights = self._grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self._acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
        flat = cam.flatten(1)
        lo, hi = flat.min(1).values.view(-1, 1, 1), flat.max(1).values.view(-1, 1, 1)
        cam = (cam - lo) / (hi - lo).clamp_min(1e-8)
        return cam.detach().cpu().numpy(), logits.softmax(1).detach().cpu().numpy()

    def remove(self):
        for h in self._hooks:
            h.remove()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()


def overlay(image, heatmap, alpha=0.45, cmap="jet"):
    """Blend a [0, 1] heatmap onto an RGB PIL image (same size) -> uint8 array."""
    import matplotlib

    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
    heat = matplotlib.colormaps[cmap](heatmap)[..., :3]
    return (255 * ((1 - alpha) * rgb + alpha * heat)).astype(np.uint8)

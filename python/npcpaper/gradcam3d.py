from __future__ import annotations

import torch
import torch.nn.functional as F


class GradCAM3D:
    """Minimal Grad-CAM for 3D ResNet endpoints."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._hooks = []
        self._register()

    def _register(self):
        def fwd_hook(module, inp, out):
            self.activations = out.detach()
        def bwd_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        self._hooks.append(self.target_layer.register_forward_hook(fwd_hook))
        self._hooks.append(self.target_layer.register_full_backward_hook(bwd_hook))

    def remove(self):
        for h in self._hooks:
            h.remove()

    def __call__(self, x: torch.Tensor, endpoint: str) -> torch.Tensor:
        self.model.zero_grad(set_to_none=True)
        out = self.model(x)[endpoint]
        score = out.sum()
        score.backward(retain_graph=True)
        weights = self.gradients.mean(dim=(2, 3, 4), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[2:], mode="trilinear", align_corners=False)
        cam_min = cam.flatten(1).min(dim=1)[0].view(-1, 1, 1, 1, 1)
        cam_max = cam.flatten(1).max(dim=1)[0].view(-1, 1, 1, 1, 1)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam.detach()

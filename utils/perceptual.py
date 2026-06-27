"""Train-only VGG16 perceptual (LPIPS-style) loss for the video tokenizer.

Pixel smooth_l1/L2 alone optimizes toward the per-pixel mean and over-smooths
high-frequency detail -> blurry reconstructions (the dominant cause of the
tokenizer's ~31 dB recon ceiling). Adding a perceptual loss on frozen VGG16
features restores texture/edges. The VGG net is FROZEN and used only during
training, so it adds ZERO inference parameters (respects the no-param-increase
constraint). Computed in the training loop, never stored in the model/checkpoint.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights


class VGGPerceptualLoss(nn.Module):
    # feature distance at relu1_2, relu2_2, relu3_3 (indices into vgg16.features)
    LAYER_IDX = (3, 8, 15)

    def __init__(self, device, resize_to: int = 64):
        super().__init__()
        feats = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.eval()
        for p in feats.parameters():
            p.requires_grad_(False)
        self.slices = nn.ModuleList()
        prev = 0
        for idx in self.LAYER_IDX:
            self.slices.append(nn.Sequential(*[feats[i] for i in range(prev, idx + 1)]))
            prev = idx + 1
        # ImageNet normalization constants (inputs must be in [0,1]).
        # Register BEFORE self.to(device) so the buffers move to the GPU too (else mean/std
        # stay on CPU and (x - self.mean) raises a cuda-vs-cpu device mismatch).
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.to(device)

    def _prep(self, x):
        # x: [B, T, C, H, W] in [-1, 1] (tokenizer convention) -> [B*T, 3, H, W] ImageNet-normalized
        if x.dim() == 5:
            x = x.reshape(-1, *x.shape[2:])
        x = (x + 1.0) * 0.5            # [-1,1] -> [0,1]
        x = (x - self.mean) / self.std
        return x

    @torch.no_grad()
    def _features_target(self, y):
        return self._features(y)

    def _features(self, x):
        outs = []
        h = x
        for s in self.slices:
            h = s(h)
            outs.append(h)
        return outs

    def forward(self, pred, target):
        # pred, target: [B, T, C, H, W] in [-1, 1]. VGG net is frozen; grad flows only to pred.
        p = self._prep(pred)
        with torch.no_grad():
            t = self._prep(target)
        loss = 0.0
        ph = p
        th = t
        for s in self.slices:
            ph = s(ph)
            th = s(th)
            loss = loss + F.l1_loss(ph, th.detach())
        return loss

"""Frozen patch-feature extractors and per-object PCA onboarding for AlignPose."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from method.render import fibonacci_sphere, look_at_origin, rasterize


class DinoFeatures:
    """DINOv2 patch tokens from an intermediate layer, normalised by the model's final LayerNorm.

    This matches ``get_intermediate_layers(..., norm=True)`` of the official
    DINOv2 code: the encoder is truncated after ``layer`` blocks and the final
    LayerNorm is applied to the output tokens.
    """

    def __init__(self, model_dir, layer):
        import torch
        from transformers import AutoModel

        self.torch = torch
        cfg = json.loads(Path(model_dir, "preprocessor_config.json").read_text())
        self.mean = torch.tensor(cfg.get("image_mean", [0.485, 0.456, 0.406]), dtype=torch.float32)[None, :, None, None]
        self.std = torch.tensor(cfg.get("image_std", [0.229, 0.224, 0.225]), dtype=torch.float32)[None, :, None, None]
        model = AutoModel.from_pretrained(model_dir, local_files_only=True)
        self.layer = int(layer)
        try:
            model.encoder.layer = model.encoder.layer[:layer]
            self.truncated = len(model.encoder.layer) == layer
        except (AttributeError, TypeError):
            self.truncated = False
        model.eval()
        self.model = model
        self.patch = int(model.config.patch_size)
        self.skip = 1 + int(getattr(model.config, "num_register_tokens", 0) or 0)
        self.name = f"dinov2:{Path(model_dir).name}:layer{layer}"

    def __call__(self, images):
        torch = self.torch
        x = torch.from_numpy(np.ascontiguousarray(images, dtype=np.float32)).permute(0, 3, 1, 2)
        x = (x - self.mean) / self.std
        b, _, h, w = x.shape
        gh, gw = h // self.patch, w // self.patch
        out = []
        with torch.no_grad():
            for i in range(0, b, 8):
                if self.truncated:
                    tokens = self.model(pixel_values=x[i:i + 8]).last_hidden_state
                else:
                    hidden = self.model(pixel_values=x[i:i + 8], output_hidden_states=True).hidden_states[self.layer]
                    tokens = self.model.layernorm(hidden)
                tokens = tokens[:, self.skip:, :]
                out.append(tokens.reshape(tokens.shape[0], gh, gw, -1).numpy())
        return np.concatenate(out, 0).astype(np.float64)


class PixelPatchFeatures:
    """Raw-pixel patch descriptors (photometric alignment control, no learned features)."""

    def __init__(self, patch=14):
        self.patch = patch
        self.name = f"pixels:patch{patch}"

    def __call__(self, images):
        images = np.asarray(images, dtype=np.float64)
        b, h, w, c = images.shape
        p = self.patch
        gh, gw = h // p, w // p
        x = images[:, : gh * p, : gw * p].reshape(b, gh, p, gw, p, c).transpose(0, 1, 3, 2, 4, 5)
        return x.reshape(b, gh, gw, p * p * c)


def patch_centers(crop, patch):
    g = crop // patch
    return np.arange(g) * patch + patch // 2


def onboard_pca(mesh, extractor, crop, n_templates, n_components):
    """Mesh-only onboarding: PCA basis from masked features of rendered templates."""
    dirs = fibonacci_sphere(n_templates)
    dist = 3.0 * mesh.diameter
    f = 0.8 * crop * dist / mesh.diameter
    k = np.array([[f, 0.0, crop / 2.0], [0.0, f, crop / 2.0], [0.0, 0.0, 1.0]])
    renders, masks = [], []
    centers = patch_centers(crop, extractor.patch)
    ii, jj = np.meshgrid(centers, centers, indexing="ij")
    for d in dirs:
        r_co, t_co = look_at_origin(d, dist)
        rgb, depth = rasterize(mesh, r_co, t_co, k, crop, crop)
        renders.append(rgb)
        masks.append(np.isfinite(depth[ii, jj]))
    feats = extractor(np.stack(renders))
    x = np.concatenate([feats[i][masks[i]] for i in range(len(renders))], 0)
    mu = x.mean(0)
    xc = x - mu
    cov = xc.T @ xc / max(1, len(xc) - 1)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1][:n_components]
    comps = evecs[:, order].T
    signs = np.sign(comps[np.arange(len(comps)), np.abs(comps).argmax(1)])
    comps = comps * signs[:, None]
    return {"mean": mu, "components": comps}


def project(feats, pca):
    y = (feats - pca["mean"]) @ pca["components"].T
    return y / np.maximum(np.linalg.norm(y, axis=-1, keepdims=True), 1e-8)

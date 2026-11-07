"""Vision models used by search, loaded lazily so the rest of the platform does not need them.

* ClipEmbedder: OpenCLIP. Default ViT-B-32 (LAION-2B), or the multilingual
  ``xlm-roberta-base-ViT-B-32`` variant so investigators can query in Hindi and
  other Indian languages. Model and weight licences must be confirmed before
  commercial deployment [VERIFY].
* ObjectDetector: torchvision Faster R-CNN MobileNetV3 (COCO). It finds
  people, two-wheelers, cars, buses, trucks and bags, so small subjects in wide
  CCTV views get their own embeddings.

Everything runs on CPU (on-premises, no cloud calls). Weights are downloaded once
and cached; an air-gapped install pre-loads the cache.
"""

from __future__ import annotations

import threading

import cv2
import numpy as np

COCO_CLASSES = {1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 6: "bus", 8: "truck", 27: "backpack", 31: "handbag", 33: "suitcase"}

MODELS = {
    "en": ("ViT-B-32", "laion2b_s34b_b79k"),
    "multilingual": ("xlm-roberta-base-ViT-B-32", "laion5b_s13b_b90k"),
}

PROMPTS = ["{}", "a CCTV image of {}", "a surveillance camera photo of {}", "a low quality security camera frame of {}"]

_lock = threading.Lock()
_cache: dict = {}


class ClipEmbedder:
    def __init__(self, variant: str = "en"):
        import open_clip
        import torch

        name, pretrained = MODELS[variant]
        self.variant = variant
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(name, pretrained=pretrained)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(name)
        self.torch = torch
        self.dim = int(self.model.visual.output_dim)

    def _prep(self, bgr: np.ndarray):
        from PIL import Image

        return self.preprocess(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))

    def encode_images(self, images: list, batch: int = 32) -> np.ndarray:
        out = []
        with self.torch.no_grad():
            for i in range(0, len(images), batch):
                x = self.torch.stack([self._prep(im) for im in images[i : i + batch]])
                v = self.model.encode_image(x).float()
                out.append(v / v.norm(dim=-1, keepdim=True))
        if not out:
            return np.zeros((0, self.dim), np.float32)
        return self.torch.cat(out).numpy().astype(np.float32)

    def encode_text(self, query: str) -> np.ndarray:
        texts = [p.format(query) for p in PROMPTS]
        with self.torch.no_grad():
            v = self.model.encode_text(self.tokenizer(texts)).float()
            v = v / v.norm(dim=-1, keepdim=True)
            m = v.mean(0)
        return (m / m.norm()).numpy().astype(np.float32)


class ObjectDetector:
    def __init__(self, score: float = 0.4, min_size: int = 480):
        import torch
        from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_FPN_Weights, fasterrcnn_mobilenet_v3_large_fpn

        self.torch = torch
        self.score = score
        self.model = fasterrcnn_mobilenet_v3_large_fpn(weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT, min_size=min_size, max_size=int(min_size * 4 / 3)).eval()

    def detect(self, frames: list) -> list:
        """Returns per frame a list of (class_name, score, (x, y, w, h))."""
        out = []
        with self.torch.no_grad():
            for i in range(0, len(frames), 4):
                batch = [self.torch.from_numpy(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255 for f in frames[i : i + 4]]
                for r in self.model(batch):
                    dets = []
                    for b, lab, s in zip(r["boxes"].tolist(), r["labels"].tolist(), r["scores"].tolist()):
                        if s >= self.score and lab in COCO_CLASSES:
                            x0, y0, x1, y1 = b
                            dets.append((COCO_CLASSES[lab], float(s), (int(x0), int(y0), int(x1 - x0), int(y1 - y0))))
                    out.append(dets)
        return out


def get_embedder(variant: str = "en") -> ClipEmbedder:
    with _lock:
        key = ("clip", variant)
        if key not in _cache:
            _cache[key] = ClipEmbedder(variant)
        return _cache[key]


def get_detector() -> ObjectDetector:
    with _lock:
        if "det" not in _cache:
            _cache["det"] = ObjectDetector()
        return _cache["det"]

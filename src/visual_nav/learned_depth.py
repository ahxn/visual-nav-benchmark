from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Any


class LearnedMetricDepthEstimator:
    """Lazy Depth Anything V2 indoor metric-depth inference adapter."""

    DEFAULT_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"

    def __init__(
        self, model_name: str = DEFAULT_MODEL, device: str | None = None,
        cache_size: int = 256,
    ):
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except ImportError as exc:
            raise RuntimeError("Install the 'deep' optional dependencies") from exc
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name).to(self.device).eval()
        self.cache_size = max(0, cache_size)
        self._cache: OrderedDict[bytes, Any] = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0

    def estimate(self, rgb_frame: Any):
        """Estimate an absolute indoor depth map in meters from one RGB frame."""
        import numpy as np
        from PIL import Image

        frame = np.ascontiguousarray(rgb_frame)
        key = hashlib.blake2b(memoryview(frame), digest_size=16).digest()
        if key in self._cache:
            self.cache_hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]
        self.cache_misses += 1
        image = Image.fromarray(frame)
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
        result = self.processor.post_process_depth_estimation(
            outputs, target_sizes=[(image.height, image.width)]
        )[0]["predicted_depth"]
        depth = result.detach().float().cpu().numpy()
        if self.cache_size:
            self._cache[key] = depth
            self._cache.move_to_end(key)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return depth


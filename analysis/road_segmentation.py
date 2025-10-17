"""Road segmentation utilities for satellite imagery."""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation


MODEL_NAME = "valeo/segformer-b0-finetuned-cityscapes-1024-1024"


@dataclass
class RoadSegmentationModel:
    """Wrapper around a pretrained model to extract road masks."""

    model_name: str = MODEL_NAME
    device: str = "cpu"
    _processor: Optional[AutoImageProcessor] = None
    _model: Optional[AutoModelForSemanticSegmentation] = None
    _road_label_id: Optional[int] = None
    _torch_device: torch.device = field(init=False)

    def __post_init__(self) -> None:
        self._torch_device = torch.device(self.device)
        self._load_model()

    def _load_model(self) -> None:
        self._processor = AutoImageProcessor.from_pretrained(self.model_name)
        self._model = AutoModelForSemanticSegmentation.from_pretrained(self.model_name)
        self._model.to(self._torch_device)
        self._model.eval()

        id2label = {int(key): value for key, value in self._model.config.id2label.items()}
        for label_id, label in id2label.items():
            if label.lower() == "road":
                self._road_label_id = label_id
                break

        if self._road_label_id is None:
            raise ValueError(
                "The configured model does not expose a 'road' label."
            )

    @property
    def road_label_id(self) -> int:
        assert self._road_label_id is not None
        return self._road_label_id

    @torch.inference_mode()
    def predict_from_image(self, image: Image.Image) -> Image.Image:
        if image.mode != "RGB":
            image = image.convert("RGB")

        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {key: value.to(self._torch_device) for key, value in inputs.items()}

        outputs = self._model(**inputs)
        logits = outputs.logits
        upsampled_logits = F.interpolate(
            logits,
            size=image.size[::-1],
            mode="bilinear",
            align_corners=False,
        )

        predicted_classes = upsampled_logits.argmax(dim=1)
        road_mask = (predicted_classes == self.road_label_id).to(torch.uint8)

        mask = road_mask.squeeze(0).cpu().numpy() * 255
        return Image.fromarray(mask, mode="L")

    def predict_from_bytes(self, image_bytes: bytes) -> Image.Image:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.load()
            return self.predict_from_image(image)

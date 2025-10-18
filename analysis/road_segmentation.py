"""Road segmentation utilities for satellite imagery."""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from PIL import Image

import io
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PIL import Image

try:
    from mmengine.registry import init_default_scope
    from mmseg.apis import SegInferencer
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency error path
    raise ModuleNotFoundError(
        "mmsegmentation and its dependencies must be installed to use the road "
        "segmentation tools"
    ) from exc

MODEL_NAME = "deeplabv3plus_r101-d8_769x769_80k_deepglobe"
MODEL_CHECKPOINT = (
    "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus/"
    "deeplabv3plus_r101-d8_769x769_80k_deepglobe_20211020_204507-a8f35505.pth"
)
ROAD_LABEL_ID = 1


@dataclass
class RoadSegmentationModel:
    """Wrapper around a pretrained DeepLabV3+ model to extract road masks."""

    model_name: str = MODEL_NAME
    checkpoint_url: str = MODEL_CHECKPOINT
    device: str = "cpu"
    _inferencer: Optional[SegInferencer] = field(init=False, default=None)

    def __post_init__(self) -> None:
        # Ensure the mmseg registry scope is available so that bundled config aliases
        # (like the DeepGlobe road model) resolve correctly.
        init_default_scope("mmseg")
        self._inferencer = SegInferencer(
            model=self.model_name,
            pretrained=self.checkpoint_url,
            device=self.device,
        )

    @property
    def road_label_id(self) -> int:
        return ROAD_LABEL_ID

    def _run_inference(self, image: Image.Image) -> np.ndarray:
        # MMDetection / MMSegmentation pipelines operate on BGR numpy arrays.
        rgb_image = image.convert("RGB")
        bgr_image = np.array(rgb_image)[:, :, ::-1]
        result = self._inferencer(
            bgr_image,
            show=False,
            no_save_vis=True,
            return_datasample=False,
        )
        prediction = result["predictions"][0]
        return prediction

    def predict_from_image(self, image: Image.Image) -> Image.Image:
        prediction = self._run_inference(image)
        road_mask = (prediction == self.road_label_id).astype(np.uint8) * 255
        return Image.fromarray(road_mask, mode="L")

    def predict_from_bytes(self, image_bytes: bytes) -> Image.Image:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.load()
            return self.predict_from_image(image)


def segment_and_save_mask(
    input_image_path: Union[str, Path],
    output_mask_path: Union[str, Path],
    model: Optional[RoadSegmentationModel] = None,
    device: str = "cpu",
) -> None:
    """
    Run road segmentation on an image file and save the mask with yellow/violet colormap.

    Args:
        input_image_path: Path to the input image file
        output_mask_path: Path where the colored mask will be saved (PNG format)
        model: Optional pre-initialized RoadSegmentationModel. If None, a new model will be created
        device: Device to run the model on ("cpu" or "cuda")
    """
    # Load the model if not provided
    if model is None:
        model = RoadSegmentationModel(device=device)

    # Load the input image
    input_image = Image.open(input_image_path)

    # Run segmentation
    mask = model.predict_from_image(input_image)

    # Convert mask to numpy array and normalize to [0, 1]
    mask_array = np.array(mask) / 255.0

    # Apply matplotlib's viridis colormap (yellow/violet)
    cmap = plt.get_cmap("viridis")
    colored_mask = cmap(mask_array)

    # Convert to RGB (remove alpha channel) and scale to 0-255
    colored_mask_rgb = (colored_mask[:, :, :3] * 255).astype(np.uint8)

    # Save the colored mask
    output_image = Image.fromarray(colored_mask_rgb, mode="RGB")
    output_image.save(output_mask_path)


if __name__ == '__main__':
    segment_and_save_mask(
        "C:\\Users\\dell\\OneDrive\\Desktop\\sputnik\\data\\collections\\Zvarivank\\images\\2020-06-10T07-46-19_024000Z__46_153049_39_037053_46_207037_39_061982.png",
        "C:\\Users\\dell\\OneDrive\\Desktop\\sputnik\\data\\collections\\Zvarivank\\road_masks\\2020-06-10T07-46-19_024000Z__46_153049_39_037053_46_207037_39_061982.png"
    )
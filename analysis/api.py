"""API endpoints exposing satellite image analysis tools."""

from __future__ import annotations

import io

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response

from .road_segmentation import RoadSegmentationModel

app = FastAPI(title="Sputnik Analysis API")
_model = RoadSegmentationModel()


@app.post("/road-segmentation", summary="Generate a binary road segmentation mask")
async def road_segmentation(file: UploadFile = File(...)) -> Response:
    """Return a PNG mask highlighting road pixels for the provided image."""

    image_bytes = await file.read()
    mask = _model.predict_from_bytes(image_bytes)

    buffer = io.BytesIO()
    mask.save(buffer, format="PNG")
    return Response(content=buffer.getvalue(), media_type="image/png")

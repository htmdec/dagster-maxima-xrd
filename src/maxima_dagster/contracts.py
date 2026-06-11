"""Data structures, schemas, and metadata builders for the XRD workflow."""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib.metadata import version
from typing import Any


# XRD scan H5 files: scan_point_<id>_data_<counter>.h5
H5_SCAN_PATTERN = re.compile(r"^scan_point_(\d+)_data_\d+\.h5$", re.IGNORECASE)

# Calibrant XRD scan H5 files: xrd_calibrant_data_<id>.h5
CALIBRANT_SCAN_PATTERN = re.compile(r"^xrd_calibrant_data_(\d+)\.h5$", re.IGNORECASE)


@dataclass
class GirderPointer:
    """
    Signals that data already exists in Girder. 
    The IOManager will fetch this file into memory when passed to downstream assets.
    """
    file_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GirderPayload:
    """
    Signals that new data has been computed in RAM. 
    The IOManager will upload this stream to Girder and convert it to a GirderPointer.
    """
    stream: io.BytesIO
    filename: str
    folder_id: str
    mime_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


def build_item_link(girder_url: str, item_id: str) -> str:
    base = str(girder_url or "").rstrip("/")
    if base.endswith("/api/v1"):
        base = base[: -len("/api/v1")]
    return f"{base}/#item/{item_id}"


def build_prov_metadata(run_id: str | None) -> dict[str, Any]:
    return {
        "workflow_version": version("MaximaDagster"),
        "run_id": run_id,
        "time": datetime.now(timezone.utc).isoformat(),
    }


def build_model_metadata(
    model_version: str,
    model_item_id: str,
    girder_url: str,
) -> dict[str, Any]:
    return {
        "version": model_version,
        "item_id": model_item_id,
        "link": build_item_link(girder_url, model_item_id),
    }


def build_calibrant_metadata(
    calibrant_item_id: str,
    girder_url: str,
    igsn: str | None,
) -> dict[str, Any]:
    payload = {
        "item_id": calibrant_item_id,
        "link": build_item_link(girder_url, calibrant_item_id),
    }
    if igsn:
        payload["igsn"] = igsn
    return payload


def build_poni_linkage_metadata(
    poni_item_id: str,
    girder_url: str,
    geometry: Any,
) -> dict[str, Any]:
    return {
        "item_id": poni_item_id,
        "link": build_item_link(girder_url, poni_item_id),
        "geometry": {
            "dist": float(geometry.dist),
            "poni1": float(geometry.poni1),
            "poni2": float(geometry.poni2),
            "rot1": float(geometry.rot1),
            "rot2": float(geometry.rot2),
            "rot3": float(geometry.rot3),
        },
    }


__all__ = [
    "H5_SCAN_PATTERN",
    "CALIBRANT_SCAN_PATTERN",
    "GirderPointer",
    "GirderPayload",
    "build_item_link",
    "build_prov_metadata",
    "build_model_metadata",
    "build_calibrant_metadata",
    "build_poni_linkage_metadata",
]
"""Helpers for publishing XRD artifacts and metadata to Girder."""

from __future__ import annotations

import functools
import io
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def upload_artifact(
    gc: Any,
    folder_id: str,
    filename: str,
    payload: bytes,
    mime_type: str,
    metadata: dict[str, Any],
) -> str:
    """Uploads arbitrary bytes to a Girder folder and returns the new Item ID."""
    item = gc.loadOrCreateItem(filename, folder_id)
    existing_files = gc.get(f"item/{item['_id']}/files", parameters={"limit": 1})

    stream = io.BytesIO(payload)
    size = len(payload)

    if existing_files:
        gc.uploadFileContents(existing_files[0]["_id"], stream, size)
    else:
        file_meta = gc.post(
            "file",
            parameters={
                "parentType": "item",
                "parentId": item["_id"],
                "name": filename,
                "size": size,
                "mimeType": mime_type,
            },
        )
        gc._uploadContents(file_meta, stream, size)

    gc.addMetadataToItem(item["_id"], metadata)
    
    return str(item["_id"])


@functools.cache
def get_workflow_version() -> str:
    """Get the workflow version from pyproject.toml.
    
    Returns:
        Version string from pyproject.toml, or a default fallback if unable to read.
    """
    try:
        current = Path(__file__).resolve()
        for parent in [current.parent, *current.parent.parents]:
            pyproject_path = parent / "pyproject.toml"
            if pyproject_path.exists():
                with open(pyproject_path, "rb") as f:
                    data = tomllib.load(f)
                    return data.get("project", {}).get("version", "unknown")

        return "unknown"
    except Exception:
        return "unknown"


def build_item_link(girder_url: str, item_id: str) -> str:
    base = str(girder_url or "").rstrip("/")
    if base.endswith("/api/v1"):
        base = base[: -len("/api/v1")]
    return f"{base}/#item/{item_id}"


def build_prov_metadata(run_id: str | None) -> dict[str, Any]:
    return {
        "workflow_version": get_workflow_version(),
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
    "upload_artifact",
    "build_item_link",
    "build_prov_metadata",
    "build_model_metadata",
    "build_calibrant_metadata",
    "build_poni_linkage_metadata",
]

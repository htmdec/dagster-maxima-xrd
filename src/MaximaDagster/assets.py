import os
import tempfile
import io
from pathlib import Path
from typing import Any

import h5py
from pyFAI.geometry import Geometry
from pyFAI.integrator.azimuthal import AzimuthalIntegrator as PyFAIAzimuthalIntegrator
from dagster import (
    AssetIn,
    AssetExecutionContext,
    Failure,
    asset,
)

from .modules import azimuthal_integrator, calibrate
from .partition_mapping import ClosestPrecedingPartitionMapping
from .sensors import calibrant_partitions, experiment_partitions
from .contracts import (
    CALIBRANT_SCAN_PATTERN,
    H5_SCAN_PATTERN,
    GirderPointer,
    GirderPayload,
    build_calibrant_metadata,
    build_model_metadata,
    build_poni_linkage_metadata,
    build_prov_metadata,
)


def load_geometry_from_mem(poni_bytes: bytes) -> Geometry:
    ai = PyFAIAzimuthalIntegrator()
    with tempfile.TemporaryDirectory() as tmpdir:
        poni_path = Path(tmpdir) / "temp.poni"
        poni_path.write_bytes(poni_bytes)

        ai.load(str(poni_path))
    
    return Geometry(
        dist=ai.dist,
        poni1=ai.poni1,
        poni2=ai.poni2,
        rot1=ai.rot1,
        rot2=ai.rot2,
        rot3=ai.rot3,
        detector=ai.detector,
        wavelength=ai.wavelength,
    )


@asset(required_resource_keys={"GirderConnection"})
def calibration_model(context: AssetExecutionContext) -> GirderPointer:
    """Downloads and caches canonical .pth calibration model."""
    conn = context.resources.GirderConnection
    model_item_id = os.getenv("GIRDER_MODEL_ITEM_ID")
    if not model_item_id:
        raise ValueError("GIRDER_MODEL_ITEM_ID is not configured.")

    item_info = conn.client.getItem(model_item_id)
    meta = item_info.get("meta", {})
    meta_fields = meta.get("params") if isinstance(meta.get("params"), dict) else meta

    item_files = list(conn.client.listFile(item_info["_id"]))
    pth_files = [f for f in item_files if f["name"].lower().endswith(".pth")]

    if not pth_files:
        raise ValueError(f"No .pth file found in item {model_item_id}.")

    file_info = pth_files[0]

    metadata = {
            "source_file_id": str(file_info["_id"]),
            "calibrant": str(meta_fields["calibrant"]),
            "detector": str(meta_fields["detector"]),
            "energy": float(meta_fields["energy"]),
            "version": str(meta_fields["version"]),
    }
    
    context.log.info(f"Discovered calibration model: {file_info['name']}")

    return GirderPointer(
        file_id=str(file_info["_id"]),
        metadata=metadata,
    )




@asset(required_resource_keys={"GirderConnection"}, partitions_def=calibrant_partitions)
def poni(
    context: AssetExecutionContext,
    calibration_model: Any,
) -> GirderPayload:
    """Generates a PONI geometry file for the given partition using the provided calibration model."""
    conn = context.resources.GirderConnection
    run_id = str(context.run.run_id)
    girder_url = str(os.getenv("GIRDER_API_URL") or "")
    partition_key = context.partition_key

    rows = conn.resolve_partition_details(partition_key, "xrd_calibrant_raw")
    target_row = next((r for r in rows if CALIBRANT_SCAN_PATTERN.match(conn.get_fname(r) or "")), None)

    if not target_row:
        raise Failure(f"Valid calibrant file missing in partition {partition_key}.")

    calibrant_fname = conn.get_fname(target_row)
    folder_id = conn.get_folder_id(target_row)
    item_id = conn.get_item_id(target_row)

    item_info = conn.client.getItem(item_id)
    exp_date = item_info.get("meta", {}).get("experiment_date")
    
    files = list(conn.client.listFile(item_id))
    h5_file_id = next((f["_id"] for f in files if str(f["name"]).lower().endswith(".h5")), None)

    calibrant_stream = conn.get_stream(h5_file_id)
    with h5py.File(calibrant_stream, "r") as h5f:
        calibration_pattern = h5f["entry/data/data"][:][0]

    model_metadata = calibration_model.metadata

    calibrator = calibrate.MaximaCalibrator(
        calibration_model,
        str(model_metadata["calibrant"]),
        str(model_metadata["detector"]),
        float(model_metadata["energy"]),
    )

    geometry = calibrator.calibrate(calibration_pattern)

    poni_file_name = Path(calibrant_fname).stem + ".poni"

    with tempfile.TemporaryDirectory() as tmpdir:    # pyFAI expects a file path, it seems like a C-level requirement, so I believe we have to write to disk here 
        poni_path = Path(tmpdir) / "temp.poni"
        geometry.save(str(poni_path))
        poni_bytes = poni_path.read_bytes()

    metadata = {
        "prov": build_prov_metadata(run_id),
        "model": build_model_metadata(
            model_version=model_metadata["version"],
            model_item_id=model_metadata["source_file_id"],
            girder_url=girder_url,
        ),
        "calibrant": build_calibrant_metadata(
            calibrant_item_id=item_id,
            girder_url=girder_url,
            igsn=conn.get_igsn(target_row),
        ),
        "data_type": "xrd_calibrant_derived",
        "experiment_date": exp_date,
    }

    context.log.info(f"Generated new PONI geometry for {partition_key}")
    return GirderPayload(
        stream=io.BytesIO(poni_bytes),
        filename=poni_file_name,
        folder_id=folder_id,
        mime_type="text/plain",
        metadata=metadata,
    )


@asset(required_resource_keys={"GirderConnection"}, partitions_def=experiment_partitions)
def xrd_raw(context: AssetExecutionContext) -> dict[str, Any]:
    conn = context.resources.GirderConnection 
    partition_key = context.partition_key
    
    rows = conn.resolve_partition_details(partition_key, "xrd_raw")
    scans = {}
    folder_id = None

    for row in rows:
        fname = conn.get_fname(row)
        
        h5_match = H5_SCAN_PATTERN.match(fname or "")
        if not fname or not h5_match:
            continue

        if not folder_id:
            raw_folder_id = conn.get_folder_id(row)
            raw_folder = conn.client.getFolder(raw_folder_id)
            folder_id = str(raw_folder.get("parentId", ""))

        item_id = conn.get_item_id(row)
        item_info = conn.client.getItem(item_id)
        exp_date = item_info.get("meta", {}).get("experiment_date")

        files = list(conn.client.listFile(item_id))
        file_id = next((f["_id"] for f in files if str(f["name"]).lower().endswith(".h5")), None)

        if file_id:
            scan_num = int(h5_match.group(1))
            pointer_meta = {"item_id": item_id, "igsn": conn.get_igsn(row), "experiment_date": exp_date}
            scans[scan_num] = {"xrd": GirderPointer(file_id=file_id, metadata=pointer_meta)}

    context.log.info(f"Mapped {len(scans)} scan(s) to pointers.")
    
    return {
        "experiment_folder_id": folder_id,
        "scans": scans,
    }


@asset(
    partitions_def=experiment_partitions,
    ins={"poni": AssetIn(partition_mapping=ClosestPrecedingPartitionMapping())},
)
def azimuthal_integration(
    context: AssetExecutionContext,
    xrd_raw: dict[str, Any],
    poni: Any,
) -> dict[int, GirderPayload]:
    """Performs azimuthal integration on the raw XRD scans using the provided PONI geometry, and uploads results to Girder."""
    run_id = str(context.run.run_id)
    girder_url = str(os.getenv("GIRDER_API_URL") or "")
    
    folder_id = xrd_raw["experiment_folder_id"]
    scans = xrd_raw["scans"]

    geometry = load_geometry_from_mem(poni.getvalue())

    xrd_scans_in_memory = {}
    for scan_id, scan_data in scans.items():
        if "xrd" in scan_data:
            with h5py.File(scan_data["xrd"], "r") as h5f:
                xrd_scans_in_memory[scan_id] = h5f["entry/data/data"][:][0]

    results = azimuthal_integrator.integrate_dict(xrd_scans_in_memory, geometry)

    ai_metadata_base = {
        "prov": build_prov_metadata(run_id),
        "poni": build_poni_linkage_metadata(
            poni_item_id=poni.metadata.get("item_id", "unknown"),
            girder_url=girder_url,
            geometry=geometry,
        ),
        "data_type": "xrd_derived",
    }

    payloads = {}
    for scan_id, dataframe in results.items():
        csv_bytes = dataframe.to_csv(index=False).encode("utf-8")

        scan_id_str = str(scan_id)
        
        igsn = scans[scan_id_str]["xrd"].metadata.get("igsn")
        exp_date = scans[scan_id_str]["xrd"].metadata.get("experiment_date")
        item_meta = {**ai_metadata_base, "igsn": igsn, "experiment_date": exp_date} if igsn else ai_metadata_base

        payloads[scan_id] = GirderPayload(
            stream=io.BytesIO(csv_bytes),
            filename=f"scan_point_{int(scan_id)}_azimuthal.csv",
            folder_id=folder_id,
            mime_type="text/csv",
            metadata=item_meta,
        )

    context.log.info(f"Yielding {len(payloads)} integration result(s) for network upload.")
    return payloads
import os
import tempfile
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

from .utils.discovery import (
    get_base_parent_id,
    get_base_parent_type,
    call_with_retries,
    fetch_partition_details,
    get_fname,
    get_item_id,
    get_folder_id,
    get_igsn,
)

from .modules import AzimuthalIntegrator, calibrate
from .utils.patterns import CALIBRANT_SCAN_PATTERN, H5_SCAN_PATTERN
from .partition_mapping import ClosestPrecedingPartitionMapping
from .utils.results_publisher import (
    build_calibrant_metadata,
    build_model_metadata,
    build_poni_linkage_metadata,
    build_prov_metadata,
    upload_artifact
)

from .sensors import calibrant_partitions, experiment_partitions


def _read_xrd_h5_from_item_id(
    context: AssetExecutionContext,
    gc: Any,
    item_id: str,
) -> Any:
    item_files = list(gc.listFile(item_id))
    h5_files = [
        f for f in item_files 
        if str(f.get("name", "")).lower().endswith(".h5")
    ]
    
    if not h5_files:
        raise Failure(
            description=(
                "Expected an .h5 file inside item, but found: "
                f"{[f.get('name') for f in item_files]}"
            ),
            metadata={
                "context": {
                    "partition_key": context.partition_key,
                    "run_id": str(context.run.run_id),
                },
                "item_id": item_id,
            },
        )
        
    file_id = str(h5_files[0]["_id"])

    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = Path(tmpdir) / f"{item_id}.h5"
        gc.downloadFile(file_id, str(local_path))
        
        with h5py.File(local_path, "r") as h5f:
            return h5f["entry/data/data"][:][0]


def _resolve_model_file(gc: Any, model_item_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    item_info = gc.getItem(model_item_id)
    meta = item_info.get("meta") or {}
    
    item_files = list(gc.listFile(item_info["_id"]))
    pth_files = [f for f in item_files if str(f.get("name", "")).lower().endswith(".pth")]
    
    if not pth_files:
        raise ValueError(f"No .pth files found in Girder item {model_item_id}.")
    
    return pth_files[0], meta


def load_geometry_from_poni(poni_path: Path | str) -> Geometry:
    ai = PyFAIAzimuthalIntegrator()
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


@asset(required_resource_keys={"GirderClient"})
def calibration_model(context: AssetExecutionContext) -> dict[str, Any]:
    """Downloads and caches canonical .pth calibration model."""
    gc = context.resources.GirderClient

    model_item_id = os.getenv("GIRDER_MODEL_ITEM_ID")
    if not model_item_id:
        raise ValueError("GIRDER_MODEL_ITEM_ID is not configured.")

    file_info, meta = _resolve_model_file(gc, model_item_id)
    meta_fields = meta.get("params") if isinstance(meta.get("params"), dict) else meta

    local_dir = Path("data") / "models"
    local_dir.mkdir(parents=True, exist_ok=True)
    file_name = str(file_info["name"])
    local_path = local_dir / file_name

    if not local_path.exists():
        gc.downloadFile(file_info["_id"], str(local_path))
        context.log.info(f"Downloaded calibration model to {local_path}")
    else:
        context.log.info(f"Using cached calibration model at {local_path}")

    return {
        "model_path": str(local_path),
        "metadata": {
            "source_file_id": str(file_info["_id"]),
            "calibrant": str(meta_fields["calibrant"]),
            "detector": str(meta_fields["detector"]),
            "energy": float(meta_fields["energy"]),
            "version": str(meta_fields["version"]),
        },
    }


@asset(required_resource_keys={"GirderClient"}, partitions_def=calibrant_partitions)
def poni(
    context: AssetExecutionContext,
    calibration_model: dict[str, Any],
) -> dict[str, Any]:
    gc = context.resources.GirderClient
    run_id = str(context.run.run_id)
    girder_url = str(os.getenv("GIRDER_API_URL") or "")
    partition_key = context.partition_key
    base_id = get_base_parent_id()
    base_type = get_base_parent_type()

    details = call_with_retries(
        fetch_partition_details,
        gc,
        base_id=base_id,
        base_type=base_type,
        key=partition_key,
        data_type="xrd_calibrant_raw",
    )

    item_id: str | None = None
    fname: str | None = None
    igsn: str | None = None
    folder_id: str | None = None
    for row in details:
        fname = get_fname(row)

        if not fname or not CALIBRANT_SCAN_PATTERN.match(fname):
            continue

        item_id = get_item_id(row)
        igsn = get_igsn(row)
        folder_id = get_folder_id(row)
        break

    if item_id is None or fname is None or folder_id is None:
        raise Failure(
            description=(
                "Could not find a valid calibrant file matching the expected pattern "
                f"in partition {partition_key}."
            ),
            metadata={
                "context": {
                    "partition_key": partition_key,
                    "run_id": run_id,
                },
                "data_type": "xrd_calibrant_raw",
            },
        )

    calibration_pattern = _read_xrd_h5_from_item_id(context, gc, item_id)
    
    model_metadata = calibration_model["metadata"]

    calibrator = calibrate.MaximaCalibrator(
        calibration_model["model_path"],
        str(model_metadata["calibrant"]),
        str(model_metadata["detector"]),
        float(model_metadata["energy"]),
    )

    geometry = calibrator.calibrate(calibration_pattern)
    del calibration_pattern

    poni_file_name = Path(fname).stem
    with tempfile.TemporaryDirectory() as tmpdir:
        poni_path = Path(tmpdir) / f"{poni_file_name}.poni"
        geometry.save(str(poni_path))
        poni_bytes = poni_path.read_bytes()

    poni_metadata = {
        "prov": build_prov_metadata(run_id),
        "model": build_model_metadata(
            model_version=model_metadata["version"],
            model_item_id=model_metadata["source_file_id"],
            girder_url=girder_url,
        ),
        "calibrant": build_calibrant_metadata(
            calibrant_item_id=item_id,
            girder_url=girder_url,
            igsn=igsn,
        ),
        "data_type": "xrd_calibrant_derived",
    }

    poni_item_id = upload_artifact(
        gc=gc,
        folder_id=folder_id,
        filename=f"{fname}.poni",
        payload=poni_bytes,
        mime_type="text/plain",
        metadata=poni_metadata,
    )

    context.log.info(f"Generated new PONI geometry for {partition_key}")
    return {
        "poni_bytes": poni_bytes,
        "poni_file_name": f"{fname}.poni",
        "poni_item_id": poni_item_id,
        "calibrant_item_id": item_id,
    }


@asset(required_resource_keys={"GirderClient"}, partitions_def=experiment_partitions)
def xrd_raw(context: AssetExecutionContext) -> dict[str, Any]:
    gc = context.resources.GirderClient 
    partition_key = context.partition_key
    base_id = get_base_parent_id()
    base_type = get_base_parent_type()

    details = call_with_retries(
        fetch_partition_details, 
        gc, 
        base_id=base_id,
        base_type=base_type,
        key=partition_key, 
        data_type="xrd_raw"
    )

    scans: dict[int, dict[str, Any]] = {}
    folder_id: str | None = None

    for row in details:
        fname = get_fname(row)
        
        h5_match = H5_SCAN_PATTERN.match(fname)
        if not fname or not h5_match:
            continue

        current_folder_id = get_folder_id(row)
        if folder_id is None:
            folder_id = current_folder_id
        elif folder_id != current_folder_id:
            raise ValueError(
                f"Unexpected pattern: Multiple raw folder IDs found in partition {partition_key} "
                f"({folder_id} and {current_folder_id})."
            )

        item_id = get_item_id(row)
        if not item_id:
            context.log.warning(f"Skipping {fname}: Could not extract item ID.")
            continue

        scan_num = int(h5_match.group(1))
        scans.setdefault(scan_num, {})

        item_igsn = get_igsn(row)
        if item_igsn:
            scans[scan_num]["igsn"] = item_igsn

        scans[scan_num]["xrd"] = _read_xrd_h5_from_item_id(context, gc, item_id)
        scans[scan_num].setdefault("source_files", []).append(fname)
        scans[scan_num].setdefault("source_item_ids", []).append(item_id)

    if not folder_id:
        raise ValueError(
            f"Could not find a folder containing valid XRD files matching "
            f"the expected pattern in partition {partition_key}."
        )

    raw_folder = gc.getFolder(folder_id)
    
    if raw_folder.get("parentCollection") != "folder":
        raise ValueError(
            f"Raw folder {folder_id} is directly under a {raw_folder.get('parentCollection')}."
        )
        
    experiment_folder_id = str(raw_folder.get("parentId", ""))
    experiment_folder = gc.getFolder(experiment_folder_id)
    experiment_name = str(experiment_folder.get("name", experiment_folder_id))

    context.log.info(
        f"Loaded {len(scans)} scan(s) from experiment {experiment_name}"
    )

    return {
        "experiment_folder_id": experiment_folder_id,
        "experiment_name": experiment_name,
        "scans": scans,
    }


@asset(
    required_resource_keys={"GirderClient"},
    partitions_def=experiment_partitions,
    ins={"poni": AssetIn(partition_mapping=ClosestPrecedingPartitionMapping())},
)
def azimuthal_integration(
    context: AssetExecutionContext,
    xrd_raw: dict[str, Any],
    poni: dict[str, Any],
) -> dict[int, Any]:

    gc = context.resources.GirderClient
    run_id = str(context.run.run_id)
    girder_url = str(os.getenv("GIRDER_API_URL") or "")
    
    experiment_folder_id = xrd_raw["experiment_folder_id"]
    scans = xrd_raw["scans"]

    xrd_scans = {scan_id: scan["xrd"] for scan_id, scan in scans.items() if "xrd" in scan}
    with tempfile.TemporaryDirectory() as tmpdir:
        poni_path = Path(tmpdir) / "poni.poni"
        poni_path.write_bytes(poni["poni_bytes"])
        geometry = load_geometry_from_poni(poni_path)
    results = AzimuthalIntegrator.integrate_dict(xrd_scans, geometry)

    ai_metadata = {
        "prov": build_prov_metadata(run_id),
        "poni": build_poni_linkage_metadata(
            poni_item_id=poni["poni_item_id"],
            girder_url=girder_url,
            geometry=geometry,
        ),
        "data_type": "xrd_derived",
    }

    uploaded_files = []

    for scan_id, integration_result in results.items():
        filename = f"scan_point_{int(scan_id)}_azimuthal.csv"
        payload = integration_result.to_csv(index=False).encode("utf-8")
        
        igsn = scans[scan_id].get("igsn")
        item_meta = {**ai_metadata, "igsn": igsn} if igsn else ai_metadata

        upload_artifact(
            gc=gc,
            folder_id=experiment_folder_id,
            filename=filename,
            payload=payload,
            mime_type="text/csv",
            metadata=item_meta,
        )
        
        uploaded_files.append(filename)

    context.log.info(f"Uploaded {len(uploaded_files)} azimuthal integration result(s) to experiment folder {xrd_raw['experiment_name']}")

    return results
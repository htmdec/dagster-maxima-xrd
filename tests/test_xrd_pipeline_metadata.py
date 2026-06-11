from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
from dagster import build_asset_context

from MaximaDagster import assets
from MaximaDagster.contracts import GirderPayload, GirderPointer


def _write_test_h5(path: Path) -> None:
    with h5py.File(path, "w") as h5f:
        h5f.create_dataset("entry/data/data", data=np.arange(4, dtype=np.float32).reshape(1, 2, 2))


class _XrdRawClient:
    def __init__(self, detail_rows):
        self._detail_rows = detail_rows

    def get(self, route, parameters=None):
        if route == "aimdl/partition/details":
            return list(self._detail_rows)
        raise AssertionError(route)

    def listFile(self, item_id):
        return [{"_id": f"file_{item_id}", "name": "scan_point_0_data_00001.h5"}]

    def getItem(self, item_id):
        _ = item_id
        return {"meta": {"experiment_date": "2026-01-02T00:00:00+00:00"}}

    def getFolder(self, folder_id):
        if folder_id == "raw_01":
            return {
                "_id": "raw_01",
                "name": "raw",
                "parentCollection": "folder",
                "parentId": "exp_01",
            }
        if folder_id == "exp_01":
            return {"_id": "exp_01", "name": "experiment_01"}
        raise AssertionError(folder_id)


class _PoniClient:
    def __init__(self, detail_rows) -> None:
        self._detail_rows = detail_rows

    def get(self, route, parameters=None):
        if route == "aimdl/partition/details":
            return list(self._detail_rows)
        raise AssertionError(route)

    def listFile(self, item_id):
        return [{"_id": f"file_{item_id}", "name": "xrd_calibrant_data_000001.h5"}]

    def getItem(self, item_id):
        _ = item_id
        return {"meta": {"experiment_date": "2026-01-01T00:00:00+00:00"}}


class _ConnectionStub:
    def __init__(self, client, h5_path: Path, detail_rows):
        self.client = client
        self._h5_path = h5_path
        self._detail_rows = detail_rows

    def resolve_partition_details(self, key: str, data_type: str):
        _ = (key, data_type)
        return list(self._detail_rows)

    @staticmethod
    def get_item_id(row):
        return str(row.get("_id")) if row.get("_modelType") == "item" else None

    @staticmethod
    def get_folder_id(row):
        return str(row.get("folderId")) if row.get("folderId") else None

    @staticmethod
    def get_fname(row):
        return str(row.get("name")) if row.get("name") else None

    @staticmethod
    def get_igsn(row):
        meta = row.get("meta")
        if isinstance(meta, dict) and meta.get("igsn"):
            return str(meta.get("igsn"))
        return None

    def get_stream(self, file_id: str):
        _ = file_id
        return io.BytesIO(self._h5_path.read_bytes())


class _LogStub:
    def info(self, msg):
        _ = msg

    def warning(self, msg):
        _ = msg


class _AssetContextStub:
    def __init__(self, resources, partition_key, run_id):
        self.resources = SimpleNamespace(**resources)
        self.partition_key = partition_key
        self.run = SimpleNamespace(run_id=run_id)
        self.log = _LogStub()


def test_xrd_raw_preserves_igsn_and_source_shape(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    h5_path = tmp_path / "scan.h5"
    _write_test_h5(h5_path)

    rows = [
        {
            "_modelType": "item",
            "_id": "item_1",
            "name": "scan_point_0_data_00001.h5",
            "folderId": "raw_01",
            "meta": {"igsn": "IGSN-123"},
        }
    ]
    gc = _XrdRawClient(detail_rows=rows)
    connection = _ConnectionStub(client=gc, h5_path=h5_path, detail_rows=rows)

    context = build_asset_context(resources={"GirderConnection": connection}, partition_key="exp_01")
    result = assets.xrd_raw(context)

    assert result["experiment_folder_id"] == "exp_01"
    scan_pointer = result["scans"][0]["xrd"]
    assert isinstance(scan_pointer, GirderPointer)
    assert scan_pointer.file_id == "file_item_1"
    assert scan_pointer.metadata["item_id"] == "item_1"
    assert scan_pointer.metadata["igsn"] == "IGSN-123"


def test_azimuthal_integration_uploads_exact_metadata_shape(monkeypatch) -> None:
    monkeypatch.setenv("GIRDER_API_URL", "https://girder.example/api/v1")
    monkeypatch.setattr(
        assets.azimuthal_integrator,
        "integrate_dict",
        lambda xrd_scans, geometry: {
            0: pd.DataFrame({"q_nm^-1": [1.0], "intensity": [2.0]}),
            1: pd.DataFrame({"q_nm^-1": [3.0], "intensity": [4.0]}),
        },
    )
    geometry = SimpleNamespace(
        dist=1.1,
        poni1=2.2,
        poni2=3.3,
        rot1=4.4,
        rot2=5.5,
        rot3=6.6,
    )
    monkeypatch.setattr(assets, "load_geometry_from_mem", lambda payload: geometry)

    context = _AssetContextStub(
        resources={"GirderConnection": object()},
        partition_key="exp_01",
        run_id="run_123",
    )

    xrd_payload = {
        "experiment_folder_id": "exp_01",
        "scans": {
            "0": {
                "xrd": GirderPointer(
                    file_id="scan_0",
                    metadata={"igsn": "IGSN-123", "experiment_date": "2026-01-02T00:00:00+00:00"},
                )
            },
            "1": {
                "xrd": GirderPointer(
                    file_id="scan_1",
                    metadata={"experiment_date": "2026-01-03T00:00:00+00:00"},
                )
            },
        },
    }

    class _PoniBuffer(io.BytesIO):
        def __init__(self, data: bytes, metadata):
            super().__init__(data)
            self.metadata = metadata

    poni_payload = _PoniBuffer(b"poni", {"item_id": "poni_item_1"})

    class _FakeH5:
        def __enter__(self):
            return {"entry/data/data": np.arange(4, dtype=np.float32).reshape(1, 2, 2)}

        def __exit__(self, exc_type, exc, tb):
            _ = (exc_type, exc, tb)
            return False

    monkeypatch.setattr(assets.h5py, "File", lambda obj, mode: _FakeH5())

    result = assets.azimuthal_integration.op.compute_fn.decorated_fn(
        context,
        xrd_raw=xrd_payload,
        poni=poni_payload,
    )

    assert set(result.keys()) == {0, 1}
    first_payload = result[0]
    second_payload = result[1]
    assert isinstance(first_payload, GirderPayload)
    assert isinstance(second_payload, GirderPayload)

    first = first_payload.metadata
    second = second_payload.metadata

    assert set(first.keys()) == {"prov", "poni", "data_type", "igsn", "experiment_date"}
    assert set(second.keys()) == {"prov", "poni", "data_type"}

    assert set(first["prov"].keys()) == {"workflow_version", "run_id", "time"}
    assert first["prov"]["run_id"] == "run_123"
    assert first["data_type"] == "xrd_derived"
    assert first["igsn"] == "IGSN-123"

    assert set(first["poni"].keys()) == {"item_id", "link", "geometry"}
    assert first["poni"]["item_id"] == "poni_item_1"
    assert first["poni"]["link"] == "https://girder.example/#item/poni_item_1"
    assert set(first["poni"]["geometry"].keys()) == {"dist", "poni1", "poni2", "rot1", "rot2", "rot3"}


def test_poni_uploads_exact_metadata_shape(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GIRDER_API_URL", "https://girder.example/api/v1")

    h5_path = tmp_path / "calibrant.h5"
    _write_test_h5(h5_path)

    rows = [
        {
            "_modelType": "item",
            "_id": "cal_item_1",
            "name": "xrd_calibrant_data_000001.h5",
            "folderId": "cal_folder",
            "meta": {"igsn": "CAL-IGSN-1"},
        }
    ]
    gc = _PoniClient(detail_rows=rows)
    connection = _ConnectionStub(client=gc, h5_path=h5_path, detail_rows=rows)

    class _FakeCalibrator:
        def __init__(self, model_path, calibrant, detector, energy):
            _ = (model_path, calibrant, detector, energy)

        def calibrate(self, pattern):
            _ = pattern
            return _GeometryStub()

    class _GeometryStub:
        dist = 1.0
        poni1 = 2.0
        poni2 = 3.0
        rot1 = 4.0
        rot2 = 5.0
        rot3 = 6.0

        def save(self, path):
            Path(path).write_text("poni", encoding="utf-8")

    monkeypatch.setattr(assets.calibrate, "MaximaCalibrator", _FakeCalibrator)

    context = _AssetContextStub(
        resources={"GirderConnection": connection},
        partition_key="cal_01",
        run_id="run_poni",
    )

    result = assets.poni.op.compute_fn.decorated_fn(
        context,
        calibration_model=GirderPointer(
            file_id="model_file_1",
            metadata={
                "calibrant": "alpha_Al2O3",
                "detector": "Eiger2Cdte_1M",
                "energy": 12.0,
                "version": "1.2.3",
                "source_file_id": "model_file_1",
            },
        ),
    )

    assert isinstance(result, GirderPayload)

    metadata = result.metadata
    assert set(metadata.keys()) == {"prov", "model", "calibrant", "data_type", "experiment_date"}
    assert set(metadata["prov"].keys()) == {"workflow_version", "run_id", "time"}
    assert metadata["prov"]["run_id"] == "run_poni"
    assert set(metadata["model"].keys()) == {"version", "item_id", "link"}
    assert metadata["model"] == {
        "version": "1.2.3",
        "item_id": "model_file_1",
        "link": "https://girder.example/#item/model_file_1",
    }
    assert set(metadata["calibrant"].keys()) == {"item_id", "link", "igsn"}
    assert metadata["calibrant"] == {
        "item_id": "cal_item_1",
        "link": "https://girder.example/#item/cal_item_1",
        "igsn": "CAL-IGSN-1",
    }
    assert metadata["data_type"] == "xrd_calibrant_derived"
    assert metadata["experiment_date"] == "2026-01-01T00:00:00+00:00"
    assert result.filename == "xrd_calibrant_data_000001.poni"
    assert result.folder_id == "cal_folder"
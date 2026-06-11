from __future__ import annotations

import json
from typing import Any

import pytest
from dagster import AssetMaterialization, DagsterInstance, build_sensor_context

import maxima_dagster.sensors as sensor_module
from maxima_dagster.sensors import xrd_calibration_sensor, xrd_experiment_sensor


EXPERIMENT_KEY_1 = "IGSN-1//2026-01-02T00:00:00+00:00"
EXPERIMENT_KEY_2 = "IGSN-2//2026-01-03T00:00:00+00:00"
CALIBRANT_KEY_1 = "CAL-1//2026-01-01T00:00:00+00:00"
CALIBRANT_KEY_2 = "CAL-2//2026-01-04T00:00:00+00:00"


def _materialize_poni(instance: DagsterInstance, partition_key: str) -> None:
    instance.report_runless_asset_event(
        AssetMaterialization(asset_key="poni", partition=partition_key)
    )


class _PartitionClient:
    def __init__(self, responses_by_data_type: dict[str, dict[str, str]]) -> None:
        self.responses_by_data_type = responses_by_data_type

    def get(
        self,
        route: str,
        parameters: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        assert route == "aimdl/partition"
        data_type = (parameters or {}).get("dataType")
        return dict(self.responses_by_data_type.get(data_type, {}))


class _PartitionConnection:
    def __init__(self, responses_by_data_type: dict[str, dict[str, str]]) -> None:
        self.responses_by_data_type = responses_by_data_type
        self.client = _PartitionClient(responses_by_data_type)


@pytest.fixture(autouse=True)
def _patch_fetch_partitions(monkeypatch):
    def _fake_fetch_partitions(gc: Any, data_type: str, since: str, base_id: str | None = None, base_type: str | None = None) -> dict[str, str]:
        _ = since
        return dict(gc.responses_by_data_type.get(data_type, {}))

    monkeypatch.setattr(sensor_module, "fetch_partitions", _fake_fetch_partitions)


def test_xrd_sensor_bootstrap_sets_cursor_without_runs() -> None:
    with DagsterInstance.ephemeral() as instance:
        _materialize_poni(instance, CALIBRANT_KEY_1)
        connection = _PartitionConnection(
            {
                "xrd_raw": {EXPERIMENT_KEY_1: "chk_1"},
                "xrd_calibrant_raw": {CALIBRANT_KEY_1: "chk_a"},
            }
        )

        context = build_sensor_context(resources={"GirderConnection": connection}, instance=instance)
        evaluation = xrd_experiment_sensor(context)

    assert evaluation.run_requests == []
    assert evaluation.dynamic_partitions_requests == []
    assert evaluation.cursor
    payload = json.loads(evaluation.cursor)
    assert set(payload.keys()) == {"since", "checksums_by_partition"}
    assert payload["checksums_by_partition"] == {EXPERIMENT_KEY_1: "chk_1"}


def test_xrd_sensor_checksum_change_triggers_run() -> None:
    with DagsterInstance.ephemeral() as instance:
        _materialize_poni(instance, CALIBRANT_KEY_1)
        seed_connection = _PartitionConnection(
            {
                "xrd_raw": {EXPERIMENT_KEY_1: "chk_1"},
                "xrd_calibrant_raw": {CALIBRANT_KEY_1: "chk_a"},
            }
        )
        seed_context = build_sensor_context(resources={"GirderConnection": seed_connection}, instance=instance)
        seed = xrd_experiment_sensor(seed_context)

        next_connection = _PartitionConnection(
            {
                "xrd_raw": {EXPERIMENT_KEY_1: "chk_2"},
                "xrd_calibrant_raw": {CALIBRANT_KEY_1: "chk_a"},
            }
        )
        next_context = build_sensor_context(
            resources={"GirderConnection": next_connection},
            cursor=seed.cursor,
            instance=instance,
        )
        evaluation = xrd_experiment_sensor(next_context)

    assert len(evaluation.run_requests) == 1
    request = evaluation.run_requests[0]
    assert request.partition_key == EXPERIMENT_KEY_1
    assert request.run_key == f"xrd_raw:{EXPERIMENT_KEY_1}:chk_2"
    assert request.tags == {
        "partition_key": EXPERIMENT_KEY_1,
        "data_checksum": "chk_2",
        "data_type": "xrd_raw",
    }
    assert evaluation.dynamic_partitions_requests
    assert evaluation.dynamic_partitions_requests[0].partition_keys == [EXPERIMENT_KEY_1]


def test_xrd_sensor_unchanged_checksum_emits_no_runs() -> None:
    with DagsterInstance.ephemeral() as instance:
        _materialize_poni(instance, CALIBRANT_KEY_1)
        seed_connection = _PartitionConnection(
            {
                "xrd_raw": {EXPERIMENT_KEY_1: "chk_1"},
                "xrd_calibrant_raw": {CALIBRANT_KEY_1: "chk_a"},
            }
        )
        seed_context = build_sensor_context(resources={"GirderConnection": seed_connection}, instance=instance)
        seed = xrd_experiment_sensor(seed_context)

        next_connection = _PartitionConnection(
            {
                "xrd_raw": {EXPERIMENT_KEY_1: "chk_1"},
                "xrd_calibrant_raw": {CALIBRANT_KEY_1: "chk_a"},
            }
        )
        next_context = build_sensor_context(
            resources={"GirderConnection": next_connection},
            cursor=seed.cursor,
            instance=instance,
        )
        evaluation = xrd_experiment_sensor(next_context)

    assert evaluation.run_requests == []
    assert evaluation.dynamic_partitions_requests == []


def test_xrd_sensor_new_partition_adds_only_new_partition() -> None:
    with DagsterInstance.ephemeral() as instance:
        _materialize_poni(instance, CALIBRANT_KEY_1)
        _materialize_poni(instance, CALIBRANT_KEY_2)
        seed_connection = _PartitionConnection(
            {
                "xrd_raw": {EXPERIMENT_KEY_1: "chk_1"},
                "xrd_calibrant_raw": {
                    CALIBRANT_KEY_1: "chk_a",
                    CALIBRANT_KEY_2: "chk_b",
                },
            }
        )
        seed_context = build_sensor_context(resources={"GirderConnection": seed_connection}, instance=instance)
        seed = xrd_experiment_sensor(seed_context)

        next_connection = _PartitionConnection(
            {
                "xrd_raw": {
                    EXPERIMENT_KEY_1: "chk_1",
                    EXPERIMENT_KEY_2: "chk_9",
                },
                "xrd_calibrant_raw": {
                    CALIBRANT_KEY_1: "chk_a",
                    CALIBRANT_KEY_2: "chk_b",
                },
            }
        )
        next_context = build_sensor_context(
            resources={"GirderConnection": next_connection},
            cursor=seed.cursor,
            instance=instance,
        )
        evaluation = xrd_experiment_sensor(next_context)

    assert len(evaluation.run_requests) == 1
    assert evaluation.run_requests[0].partition_key == EXPERIMENT_KEY_2
    assert evaluation.dynamic_partitions_requests
    assert evaluation.dynamic_partitions_requests[0].partition_keys == [EXPERIMENT_KEY_2]


def test_xrd_sensor_skips_when_poni_missing() -> None:
    with DagsterInstance.ephemeral() as instance:
        connection = _PartitionConnection(
            {
                "xrd_raw": {EXPERIMENT_KEY_1: "chk_1"},
                "xrd_calibrant_raw": {CALIBRANT_KEY_1: "chk_a"},
            }
        )

        context = build_sensor_context(resources={"GirderConnection": connection}, instance=instance)
        evaluation = xrd_experiment_sensor(context)

    assert evaluation.run_requests == []
    payload = json.loads(evaluation.cursor)
    assert payload["checksums_by_partition"] == {}


def test_calibration_sensor_bootstrap_sets_cursor_without_runs() -> None:
    connection = _PartitionConnection({"xrd_calibrant_raw": {CALIBRANT_KEY_1: "chk_a"}})

    context = build_sensor_context(resources={"GirderConnection": connection})
    evaluation = xrd_calibration_sensor(context)

    assert evaluation.run_requests == []
    assert evaluation.dynamic_partitions_requests == []
    payload = json.loads(evaluation.cursor)
    assert payload["checksums_by_partition"] == {CALIBRANT_KEY_1: "chk_a"}


def test_calibration_sensor_checksum_change_triggers_run() -> None:
    with DagsterInstance.ephemeral() as instance:
        seed_connection = _PartitionConnection({"xrd_calibrant_raw": {CALIBRANT_KEY_1: "chk_a"}})
        seed_context = build_sensor_context(resources={"GirderConnection": seed_connection}, instance=instance)
        seed = xrd_calibration_sensor(seed_context)

        next_connection = _PartitionConnection({"xrd_calibrant_raw": {CALIBRANT_KEY_1: "chk_b"}})
        next_context = build_sensor_context(
            resources={"GirderConnection": next_connection},
            cursor=seed.cursor,
            instance=instance,
        )
        evaluation = xrd_calibration_sensor(next_context)

    assert len(evaluation.run_requests) == 1
    request = evaluation.run_requests[0]
    assert request.partition_key == CALIBRANT_KEY_1
    assert request.run_key == f"xrd_calibrant_raw:{CALIBRANT_KEY_1}:chk_b"
    assert request.tags == {
        "partition_key": CALIBRANT_KEY_1,
        "data_checksum": "chk_b",
        "data_type": "xrd_calibrant_raw",
    }
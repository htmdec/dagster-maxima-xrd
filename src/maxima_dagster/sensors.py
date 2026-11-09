import os
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from dagster import (
    AssetKey,
    DynamicPartitionsDefinition,
    RunRequest,
    SensorEvaluationContext,
    SensorDefinition,
    SensorResult,
    sensor,
)

from .partition_mapping import parse_partition_datetime, select_closest_preceding_partition

def fetch_partitions(gc: Any, base_id: str, base_type: str, data_type: str, since: str) -> dict[str, str]:
    response = gc.client.get(
        f"aimdl/partition",
        parameters={
            "dataType": data_type,
            "since": since,
            "baseParentId": base_id,
            "baseParentType": base_type,
        },
    )
    if response is None:
        return {}
    if not isinstance(response, dict):
        raise RuntimeError(f"Expected dict response from aimdl/partition")

    normalized: dict[str, str] = {}
    for key, checksum in response.items():
        partition_key = str(key).strip()
        checksum_text = str(checksum).strip()
        if not partition_key or not checksum_text:
            continue
        normalized[partition_key] = checksum_text
    return normalized

def _parse_cursor_payload(cursor: str | None) -> dict[str, Any]:
    if not cursor:
        return {}
    try:
        payload = json.loads(cursor)
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def _default_since() -> str:
    return "1970-01-01T00:00:00.000000+00:00"


def _next_since() -> str:
    return (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()


def _parse_checksum_map(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    checksums: dict[str, str] = {}
    for key, checksum in value.items():
        partition_key = str(key or "").strip()
        checksum_text = str(checksum or "").strip()
        if not partition_key or not checksum_text:
            continue
        checksums[partition_key] = checksum_text
    return checksums


def _parse_girder_cursor(cursor: str | None) -> tuple[str, dict[str, str], dict[str, str]]:
    payload = _parse_cursor_payload(cursor)
    since = str(payload.get("since") or "")
    checksums_by_partition = _parse_checksum_map(payload.get("checksums_by_partition"))
    pending_by_partition = _parse_checksum_map(payload.get("pending_by_partition"))
    return since, checksums_by_partition, pending_by_partition


def _serialize_girder_cursor(
    since: str,
    checksums_by_partition: dict[str, str],
    pending_by_partition: dict[str, str] | None = None,
) -> str:
    return json.dumps(
        {
            "since": since,
            "checksums_by_partition": dict(sorted(checksums_by_partition.items())),
            "pending_by_partition": dict(sorted((pending_by_partition or {}).items())),
        }
    )


# Given the changed partitions (key -> checksum), return the keys that are ready to run.
PartitionGate = Callable[[SensorEvaluationContext, Any, dict[str, str]], list[str]]


def _fetch_calibrant_partition_keys(gc: Any) -> list[str]:
    calibrant_partitions = fetch_partitions(
        gc,
        data_type="xrd_calibrant_raw",
        since=_default_since(),
        base_id = os.environ.get("BASE_PARENT_ID"),
        base_type = os.environ.get("BASE_PARENT_TYPE"),
    )
    return list(calibrant_partitions.keys())


def _poni_ready_for_experiments(
    context: SensorEvaluationContext,
    gc: Any,
    changed_partitions: dict[str, str],
) -> list[str]:
    if not changed_partitions:
        return []

    calibrant_keys = _fetch_calibrant_partition_keys(gc)
    materialized = context.instance.get_materialized_partitions(AssetKey("poni"))

    ready: list[str] = []
    for partition_key in changed_partitions:
        try:
            experiment_date = parse_partition_datetime(partition_key)
            target_calibrant_key = select_closest_preceding_partition(calibrant_keys, experiment_date)
        except ValueError as exc:
            context.log.warning(str(exc))
            continue

        if target_calibrant_key is None:
            context.log.info(
                "No calibrant partition exists before the experiment. "
                f"Skipping partition {partition_key}."
            )
            continue

        if target_calibrant_key not in materialized:
            context.log.info(
                "Required PONI partition is not materialized yet. "
                f"Skipping partition {partition_key} (calibrant {target_calibrant_key})."
            )
            continue

        ready.append(partition_key)

    return ready


def build_girder_partition_sensor(
    sensor_name: str,
    job_name: str,
    data_type: str,
    partitions_def: DynamicPartitionsDefinition,
    partition_gate: PartitionGate | None = None,
) -> SensorDefinition:
    """
    Factory function to generate a dynamically partitioned Girder sensor.
    """
    @sensor(
        name=sensor_name,
        job_name=job_name,
        minimum_interval_seconds=30,
        required_resource_keys={"GirderConnection"},
    )
    def _generic_sensor(
        context: SensorEvaluationContext,
        GirderConnection: Any | None = None,
    ) -> SensorResult:
        gc = GirderConnection or context.resources.GirderConnection

        since, checksums_by_partition, pending_by_partition = _parse_girder_cursor(context.cursor)
        poll_since = since or _default_since()
        cursor_since = _next_since()

        partition_updates = fetch_partitions(
            gc, 
            data_type=data_type, 
            since=poll_since,
            base_id = os.environ.get("BASE_PARENT_ID"),
            base_type = os.environ.get("BASE_PARENT_TYPE"),
            )

        # Partitions blocked by the gate on earlier ticks are carried in the cursor
        # and re-checked here, so ``since`` can always advance.
        partition_updates = {**pending_by_partition, **partition_updates}

        changed_partitions = {
            partition_key: checksum
            for partition_key, checksum in partition_updates.items()
            if checksums_by_partition.get(partition_key) != checksum
        }

        if partition_gate:
            eligible = set(partition_gate(context, gc, changed_partitions))
        else:
            eligible = set(changed_partitions)
        eligible_partition_keys = [key for key in changed_partitions if key in eligible]

        pending_partitions = {
            partition_key: checksum
            for partition_key, checksum in changed_partitions.items()
            if partition_key not in eligible
        }

        merged_checksums = dict(checksums_by_partition)
        for partition_key in eligible_partition_keys:
            merged_checksums[partition_key] = partition_updates[partition_key]

        if not eligible_partition_keys:
            return SensorResult(
                cursor=_serialize_girder_cursor(cursor_since, merged_checksums, pending_partitions),
                run_requests=[],
                dynamic_partitions_requests=[],
            )

        existing_partitions = set(context.instance.get_dynamic_partitions(partitions_def.name))
        new_partitions = [
            partition_key 
            for partition_key in eligible_partition_keys 
            if partition_key not in existing_partitions
        ]

        run_requests = [
            RunRequest(
                run_key=f"{data_type}:{partition_key}:{partition_updates[partition_key]}",
                job_name=job_name,
                partition_key=partition_key,
                tags={
                    "partition_key": partition_key,
                    "data_checksum": partition_updates[partition_key],
                    "data_type": data_type,
                },
            )
            for partition_key in eligible_partition_keys
        ]

        dynamic_requests = []
        if new_partitions:
            dynamic_requests = [partitions_def.build_add_request(new_partitions)]

        return SensorResult(
            cursor=_serialize_girder_cursor(cursor_since, merged_checksums, pending_partitions),
            run_requests=run_requests,
            dynamic_partitions_requests=dynamic_requests,
        )

    return _generic_sensor


experiment_partitions = DynamicPartitionsDefinition(name="experiments")
calibrant_partitions = DynamicPartitionsDefinition(name="calibrants")


xrd_experiment_sensor = build_girder_partition_sensor(
    sensor_name="xrd_experiment_sensor",
    job_name="xrd",
    data_type="xrd_raw",
    partitions_def=experiment_partitions,
    partition_gate=_poni_ready_for_experiments,
)

xrd_calibration_sensor = build_girder_partition_sensor(
    sensor_name="xrd_calibration_sensor",
    job_name="calibration_precompute",
    data_type="xrd_calibrant_raw",
    partitions_def=calibrant_partitions,
)

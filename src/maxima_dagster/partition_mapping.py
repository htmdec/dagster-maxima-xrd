from __future__ import annotations

from datetime import datetime, timezone
from typing import Sequence

from dagster import PartitionMapping, PartitionsDefinition, DagsterInstance
from dagster._core.definitions.partitions.mapping.partition_mapping import UpstreamPartitionsResult
from dagster._core.definitions.partitions.subset.partitions_subset import PartitionsSubset
from dagster._core.instance.types import DynamicPartitionsStore


def parse_partition_datetime(partition_key: str) -> datetime:
    parts = partition_key.split("//")
    if len(parts) < 2:
        raise ValueError(
            "Invalid partition key format. Expected '<prefix>//<iso_date>', "
            f"got: {partition_key}."
        )

    date_str = parts[-1].strip()
    try:
        parsed = datetime.fromisoformat(date_str)
    except ValueError as exc:
        raise ValueError(
            f"Invalid partition date in key '{partition_key}': '{date_str}'."
        ) from exc

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)

    return parsed.astimezone(timezone.utc)


def select_closest_preceding_partition(
    partition_keys: Sequence[str],
    target: datetime,
) -> str | None:
    if target.tzinfo is None:
        target = target.replace(tzinfo=timezone.utc)
    else:
        target = target.astimezone(timezone.utc)

    closest_key: str | None = None
    closest_time: datetime | None = None
    for key in partition_keys:
        key_time = parse_partition_datetime(key)
        if key_time > target:
            continue
        if closest_time is None or key_time > closest_time:
            closest_key = key
            closest_time = key_time
    return closest_key


def _ensure_partitions_store(
    dynamic_partitions_store: DynamicPartitionsStore | None,
) -> DynamicPartitionsStore:
    if dynamic_partitions_store is not None:
        return dynamic_partitions_store
    
    return DagsterInstance.get()


def _get_partition_keys(
    partitions_def: PartitionsDefinition,
    dynamic_partitions_store: DynamicPartitionsStore,
) -> Sequence[str]:
    return list(partitions_def.get_partition_keys(dynamic_partitions_store=dynamic_partitions_store))


def _get_subset_keys(partitions_subset: PartitionsSubset | None) -> Sequence[str]:
    if partitions_subset is None:
        return []
    return list(partitions_subset.get_partition_keys())


def _select_upstream_keys(
    upstream_keys: Sequence[str],
    downstream_keys: Sequence[str],
) -> set[str]:
    selected: set[str] = set()
    for downstream_key in downstream_keys:
        target = parse_partition_datetime(downstream_key)
        closest = select_closest_preceding_partition(upstream_keys, target)
        if closest is not None:
            selected.add(closest)
    return selected


class ClosestPrecedingPartitionMapping(PartitionMapping):
    @property
    def description(self) -> str:
        return "Maps each experiment partition to the closest preceding calibrant partition."

    def validate_partition_mapping(
        self,
        upstream_partitions_def: PartitionsDefinition,
        downstream_partitions_def: PartitionsDefinition | None,
    ) -> None:
        _ = upstream_partitions_def
        _ = downstream_partitions_def

    def get_upstream_mapped_partitions_result_for_partitions(
        self,
        downstream_partitions_subset: PartitionsSubset | None,
        downstream_partitions_def: PartitionsDefinition | None,
        upstream_partitions_def: PartitionsDefinition,
        current_time: datetime | None = None,
        dynamic_partitions_store: DynamicPartitionsStore | None = None,
    ) -> UpstreamPartitionsResult:
        _ = current_time
        store = _ensure_partitions_store(dynamic_partitions_store)

        upstream_keys = _get_partition_keys(upstream_partitions_def, store)
        if downstream_partitions_subset is None:
            if downstream_partitions_def is None:
                raise ValueError("downstream_partitions_def is required when no subset is provided.")
            downstream_keys = _get_partition_keys(downstream_partitions_def, store)
        else:
            downstream_keys = _get_subset_keys(downstream_partitions_subset)

        selected_upstream = _select_upstream_keys(upstream_keys, downstream_keys)
        upstream_subset = upstream_partitions_def.subset_with_partition_keys(
            sorted(selected_upstream)
        )

        return UpstreamPartitionsResult(
            partitions_subset=upstream_subset,
            required_but_nonexistent_subset=upstream_partitions_def.empty_subset(),
        )

    def get_downstream_partitions_for_partitions(
        self,
        upstream_partitions_subset: PartitionsSubset,
        upstream_partitions_def: PartitionsDefinition,
        downstream_partitions_def: PartitionsDefinition,
        current_time: datetime | None = None,
        dynamic_partitions_store: DynamicPartitionsStore | None = None,
    ) -> PartitionsSubset:
        _ = current_time
        store = _ensure_partitions_store(dynamic_partitions_store)

        upstream_keys = _get_partition_keys(upstream_partitions_def, store)
        downstream_keys = _get_partition_keys(downstream_partitions_def, store)
        selected_upstream = set(upstream_partitions_subset.get_partition_keys())

        selected_downstream: set[str] = set()
        for downstream_key in downstream_keys:
            target = parse_partition_datetime(downstream_key)
            closest = select_closest_preceding_partition(upstream_keys, target)
            if closest is None or closest not in selected_upstream:
                continue
            selected_downstream.add(downstream_key)

        return downstream_partitions_def.subset_with_partition_keys(
            sorted(selected_downstream)
        )

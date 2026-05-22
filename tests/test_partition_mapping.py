from __future__ import annotations

from dagster import DagsterInstance

from MaximaDagster.partition_mapping import ClosestPrecedingPartitionMapping
from MaximaDagster.sensors import calibrant_partitions, experiment_partitions


def test_partition_mapping_selects_closest_preceding_calibrant() -> None:
    mapping = ClosestPrecedingPartitionMapping()
    calibrant_key_1 = "CAL-1//2026-01-01T00:00:00+00:00"
    calibrant_key_2 = "CAL-2//2026-01-10T00:00:00+00:00"
    experiment_key = "IGSN-1//2026-01-05T00:00:00+00:00"

    with DagsterInstance.ephemeral() as instance:
        instance.add_dynamic_partitions(calibrant_partitions.name, [calibrant_key_1, calibrant_key_2])
        instance.add_dynamic_partitions(experiment_partitions.name, [experiment_key])

        downstream_subset = experiment_partitions.subset_with_partition_keys([experiment_key])
        result = mapping.get_upstream_mapped_partitions_result_for_partitions(
            downstream_subset,
            experiment_partitions,
            calibrant_partitions,
            dynamic_partitions_store=instance,
        )

    assert set(result.partitions_subset.get_partition_keys()) == {calibrant_key_1}


def test_partition_mapping_maps_many_experiments_to_calibrant() -> None:
    mapping = ClosestPrecedingPartitionMapping()
    calibrant_key_1 = "CAL-1//2026-01-01T00:00:00+00:00"
    calibrant_key_2 = "CAL-2//2026-01-10T00:00:00+00:00"
    experiment_key_1 = "IGSN-1//2026-01-02T00:00:00+00:00"
    experiment_key_2 = "IGSN-2//2026-01-08T00:00:00+00:00"
    experiment_key_3 = "IGSN-3//2026-01-12T00:00:00+00:00"

    with DagsterInstance.ephemeral() as instance:
        instance.add_dynamic_partitions(calibrant_partitions.name, [calibrant_key_1, calibrant_key_2])
        instance.add_dynamic_partitions(
            experiment_partitions.name,
            [experiment_key_1, experiment_key_2, experiment_key_3],
        )

        upstream_subset = calibrant_partitions.subset_with_partition_keys([calibrant_key_1])
        result = mapping.get_downstream_partitions_for_partitions(
            upstream_subset,
            calibrant_partitions,
            experiment_partitions,
            dynamic_partitions_store=instance,
        )

    assert set(result.get_partition_keys()) == {experiment_key_1, experiment_key_2}

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Union

from dagster import (
    ConfigurableIOManager,
    Field,
    InitResourceContext,
    InputContext,
    OutputContext,
    StringSource,
    io_manager,
)


class SanitizedFilesystemIOManager(ConfigurableIOManager):
    base_dir: str

    def _get_path(self, context: Union[InputContext, OutputContext]) -> Path:
        base_dir = Path(self.base_dir)
        if context.has_asset_key and context.asset_key:
            path_parts = list(context.asset_key.path)
        else:
            path_parts = [part for part in [context.step_key, context.name] if part]
        if context.has_asset_partitions:
            sanitized_partition = (
                context.asset_partition_key.replace(":", "-").replace("/", "-")
            )
            path_parts.append(sanitized_partition)
        return base_dir.joinpath(*path_parts)

    def handle_output(self, context: OutputContext, obj: object) -> None:
        output_path = self._get_path(context)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as handle:
            pickle.dump(obj, handle)

    def load_input(self, context: InputContext) -> object:
        input_path = self._get_path(context)
        with input_path.open("rb") as handle:
            return pickle.load(handle)


@io_manager(config_schema={"base_dir": Field(StringSource, is_required=False)})
def sanitized_fs_io_manager(init_context: InitResourceContext) -> SanitizedFilesystemIOManager:
    configured_base_dir = init_context.resource_config.get("base_dir")
    base_dir = configured_base_dir or init_context.instance.storage_directory()
    return SanitizedFilesystemIOManager(base_dir=base_dir)

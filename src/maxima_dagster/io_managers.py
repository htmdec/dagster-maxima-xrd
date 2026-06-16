from __future__ import annotations

import io
import json
import re
from typing import Any, Union

from dagster import (
    ConfigurableIOManager,
    Failure,
    InputContext,
    OutputContext,
)

from .contracts import GirderPayload, GirderPointer
from .resources import GirderConnection


class GirderStream(io.BytesIO):
    """
    A subclass of io.BytesIO that seamlessly carries Girder tracking coordinates.
    Acts exactly like a standard byte stream for scientific libraries, but allows
    metadata helpers to inspect its origin tracking IDs.
    """

    def __init__(
        self,
        initial_bytes: bytes = b"",
        file_id: str | None = None,
        item_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        super().__init__(initial_bytes)
        self.file_id = file_id
        self.item_id = item_id
        self.metadata = metadata or {}


class GirderIOManager(ConfigurableIOManager):
    """
    Manages asset execution boundaries. Handles automated network uploading for
    outputs, pointer serialization for local tracking, and automatic partition key
    resolution for direct Girder file downloading.
    """

    storage_folder_id: str
    girder_connection: GirderConnection

    def _get_path(self, context: Union[InputContext, OutputContext]) -> str:
        """Determines the local path for storing lightweight pointer files."""
        if context.has_asset_key and context.asset_key:
            path_parts = list(context.asset_key.path)
        else:
            path_parts = [part for part in [context.step_key, context.name] if part]
        if context.has_asset_partitions:
            sanitized_partition = context.asset_partition_key.replace(":", "-").replace(
                "/", "-"
            )
            path_parts.append(sanitized_partition)
        return "_".join(path_parts) + ".json"

    def _serialize_obj(self, conn: Any, obj: Any) -> Any:
        """Recursively traverses data structures, converting Payloads/Pointers to JSON-safe dicts."""
        if isinstance(obj, GirderPayload):
            pointer = self._upload_payload(conn, obj)
            return {
                "_type": "GirderPointer",
                "file_id": pointer.file_id,
                "metadata": pointer.metadata,
            }
        elif isinstance(obj, GirderPointer):
            return {
                "_type": "GirderPointer",
                "file_id": obj.file_id,
                "metadata": obj.metadata,
            }
        elif isinstance(obj, dict):
            return {k: self._serialize_obj(conn, v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_obj(conn, v) for v in obj]
        else:
            return obj

    def _deserialize_obj(self, conn: Any, obj: Any) -> Any:
        """Recursively traverses data structures, converting serialized Pointers back into GirderStreams."""
        if isinstance(obj, dict):
            if obj.get("_type") == "GirderPointer":
                return self._download_to_stream(
                    conn, obj["file_id"], obj.get("metadata", {})
                )
            return {k: self._deserialize_obj(conn, v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._deserialize_obj(conn, v) for v in obj]
        else:
            return obj

    def handle_output(self, context: OutputContext, obj: Any) -> None:
        """Processes outputs by streaming them to Girder and caching tracking metadata."""
        conn = self.girder_connection

        serialized_data = self._serialize_obj(conn, obj)
        serialized = {"type": "recursive_tree", "data": serialized_data}

        filename = self._get_path(context)
        buffer = io.BytesIO(json.dumps(serialized).encode("utf-8"))
        conn.upload_file_to_folder(
            self.storage_folder_id, buffer, filename, mime_type="application/json"
        )

    def _upload_payload(self, conn: Any, payload: GirderPayload) -> GirderPointer:
        """Helper to stream bytes to the server and update item metadata fields."""
        file_info = conn.upload_file_to_folder(
            folder_id=payload.folder_id,
            stream=payload.stream,
            filename=payload.filename,
            mime_type=payload.mime_type,
        )
        file_id = file_info["_id"]
        item_id = file_info.get("itemId") or file_info.get("parentId")

        if item_id and payload.metadata:
            conn.client.addMetadataToItem(item_id, payload.metadata)

        return GirderPointer(file_id=file_id, metadata=payload.metadata)

    def load_input(self, context: InputContext) -> Any:
        """Loads inputs either from cached tracking files or dynamic Girder queries."""
        conn = self.girder_connection
        filename = self._get_path(context)

        item = None
        children = conn.client.listItem(self.storage_folder_id, name=filename)
        try:
            item = next(children)
        except StopIteration:
            pass

        if item:
            with conn.client.sendRestRequest(
                "get", f"item/{item['_id']}/download", stream=True, jsonResp=False
            ) as stream:
                buffer = io.BytesIO()
                for chunk in stream.iter_content(chunk_size=65536):
                    buffer.write(chunk)
                buffer.seek(0)
                serialized = json.loads(buffer.getvalue().decode("utf-8"))

            if serialized["type"] == "recursive_tree":
                return self._deserialize_obj(conn, serialized["data"])

            elif serialized["type"] == "pointer":
                data = serialized["data"]
                return self._download_to_stream(
                    conn, data["file_id"], data.get("metadata", {})
                )
            elif serialized["type"] == "dict_of_pointers":
                result_dict = {}
                for k, v in serialized["data"].items():
                    if isinstance(v, dict) and "file_id" in v:
                        result_dict[k] = self._download_to_stream(
                            conn, v["file_id"], v.get("metadata", {})
                        )
                    else:
                        result_dict[k] = v
                return result_dict

        if context.has_asset_partitions:
            partition_key = context.asset_partition_key
            asset_name = context.asset_key.path[-1]

            if asset_name == "xrd_raw":
                return self._resolve_raw_xrd_partition(conn, partition_key)
            elif asset_name == "calibration_model":
                return self._resolve_calibration_partition(conn, partition_key)

        raise Failure(
            f"Could not load input for asset {context.asset_key.to_string()}. "
            f"No tracking token found and no fallback resolution matches."
        )

    def _download_to_stream(
        self, conn: Any, file_id: str, metadata: dict[str, Any]
    ) -> GirderStream:
        """Downloads a file and bundles it into a metadata-enriched GirderStream."""
        raw_stream = conn.get_stream(file_id)

        item_id = metadata.get("item_id")
        if not item_id:
            try:
                file_info = conn.client.get(f"file/{file_id}")
                item_id = file_info.get("itemId")
            except Exception:
                item_id = None

        return GirderStream(
            raw_stream.getvalue(),
            file_id=file_id,
            item_id=item_id,
            metadata=metadata,
        )

    def _resolve_xrd_partition(self, conn: Any, partition_key: str) -> dict[str, Any]:
        """Resolves an external partition key directly into structured file streams."""
        rows = conn.resolve_partition_details(partition_key, "xrd_raw")
        if not rows:
            raise Failure(f"No partition data found for key {partition_key} (xrd_raw)")

        experiment_folder_id = None
        scans_dict = {}

        for row in rows:
            if not experiment_folder_id:
                experiment_folder_id = conn.get_folder_id(row)

            file_id = row.get("fileId") or row.get("_id")
            if row.get("_modelType") == "item":
                files = list(conn.client.listFile(row["_id"]))
                if files:
                    file_id = files[0]["_id"]

            if not file_id:
                continue

            fname = conn.get_fname(row) or ""
            scan_id_match = re.search(r"\d+", fname)
            if not scan_id_match:
                continue
            scan_id = scan_id_match.group(0)

            igsn = conn.get_igsn(row)
            stream = self._download_to_stream(conn, file_id, row.get("meta", {}))

            scans_dict[scan_id] = {"xrd": stream, "igsn": igsn}

        return {"experiment_folder_id": experiment_folder_id, "scans": scans_dict}

    def _resolve_calibration_partition(self, conn: Any, partition_key: str) -> Any:
        """Resolves an external calibration partition into accessible file streams."""
        rows = conn.resolve_partition_details(partition_key, "calibration_model")
        if not rows:
            raise Failure(
                f"No partition data found for key {partition_key} (calibration_model)"
            )

        streams = []
        for row in rows:
            file_id = row.get("fileId")
            if row.get("_modelType") == "item":
                files = list(conn.client.listFile(row["_id"]))
                if files:
                    file_id = files[0]["_id"]
            if file_id:
                streams.append(
                    self._download_to_stream(conn, file_id, row.get("meta", {}))
                )

        if not streams:
            raise Failure(
                f"No files resolved for calibration partition: {partition_key}"
            )

        return streams[0] if len(streams) == 1 else streams

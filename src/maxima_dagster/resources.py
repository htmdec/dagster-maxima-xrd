import io
import os
import threading
from contextlib import contextmanager
from importlib.metadata import version, PackageNotFoundError
from typing import Any, Generator

from girder_client import GirderClient as gc
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dagster import ConfigurableResource
from pydantic import PrivateAttr


class GirderClientWithSession(gc):
    """Extends standard GirderClient to accept a persistent requests.Session."""
    def __init__(self, apiUrl=None, apiKey=None, session=None, **kwargs):
        super().__init__(apiUrl=apiUrl, **kwargs)
        if apiKey:
            self.authenticate(apiKey=apiKey)
        self._session = session


_girder_client_cache: dict[tuple, GirderClientWithSession] = {}
_girder_client_cache_lock = threading.Lock()


class GirderConnection(ConfigurableResource):
    """Dagster resource managing pooled, auto-retrying connections to Girder."""
    api_url: str
    api_key: str
    
    base_parent_id: str = os.getenv("BASE_PARENT_ID")
    base_parent_type: str = os.getenv("BASE_PARENT_TYPE")
    
    _client: GirderClientWithSession | None = PrivateAttr(default=None)

    def _make_client(self) -> GirderClientWithSession:
        session = requests.Session()

        try:
            gc_version = version("girder-client")
        except PackageNotFoundError:
            gc_version = "unknown"
            
        try:
            req_version = version("requests")
        except PackageNotFoundError:
            req_version = "unknown"
        session.headers.update({
            "User-Agent": (
                f"maxima-dagster/{version('MaximaDagster')} "
                f"girder-client/{gc_version} "
                f"python-requests/{req_version}"
            )
        })
        
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST", "PUT"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        return GirderClientWithSession(
            apiUrl=self.api_url,
            apiKey=self.api_key,
            session=session,
        )

    @contextmanager
    def yield_for_execution(self, context: Any) -> Generator["GirderConnection", None, None]:
        key = (self.api_url, self.api_key)
        with _girder_client_cache_lock:
            client = _girder_client_cache.get(key)
            if client is None or client.get("user/me") is None:
                _girder_client_cache[key] = self._make_client()
        self._client = _girder_client_cache[key]
        yield self

    @property
    def client(self) -> GirderClientWithSession:
        if not self._client:
            raise RuntimeError("Girder client not initialized. Use yield_for_execution.")
        return self._client

    # --- File/Data Operations ---

    def get_stream(self, file_id: str) -> io.BytesIO:
        """Downloads a file directly into an in-memory byte stream."""
        data = io.BytesIO()
        self.client.downloadFile(file_id, data)
        data.seek(0)
        return data

    def upload_file_to_folder(
        self, folder_id: str, stream: io.BytesIO, filename: str, mime_type: str | None = None
    ) -> dict[str, Any]:
        """Uploads a byte stream to Girder, creating or updating the file."""
        if mime_type is None:
            mime_type = "application/octet-stream"
            
        existing = self.existing_file(folder_id, filename)
        size = stream.seek(0, os.SEEK_END)
        stream.seek(0)
        
        if existing:
            self.client.uploadFileContents(existing["_id"], stream, size)
            return existing
        else:
            item = self.client.loadOrCreateItem(filename, folder_id)
            file_meta = self.client.post(
                "file",
                parameters={
                    "parentType": "item",
                    "parentId": item["_id"],
                    "name": filename,
                    "size": size,
                    "mimeType": mime_type,
                },
            )
            return self.client._uploadContents(file_meta, stream, size)

    def existing_file(self, folder_id: str, filename: str) -> dict[str, Any] | None:
        for item in self.client.listItem(folder_id, name=filename):
            return next(self.client.listFile(item["_id"]), None)

    # --- Partition & ID Resolution ---

    def resolve_partition_details(self, key: str, data_type: str) -> list[dict[str, Any]]:
        """Returns the raw rows from aimdl/partition/details."""
        response = self.client.get(
            "aimdl/partition/details",
            parameters={
                "key": key,
                "dataType": data_type,
                "baseParentId": self.base_parent_id,
                "baseParentType": self.base_parent_type,
            },
        )
        return response if isinstance(response, list) else []

    @staticmethod
    def get_item_id(row: dict[str, Any]) -> str | None:
        return str(row.get("_id")) if row.get("_modelType") == "item" else None

    @staticmethod
    def get_folder_id(row: dict[str, Any]) -> str | None:
        return str(row.get("folderId")) if row.get("folderId") else None

    @staticmethod
    def get_fname(row: dict[str, Any]) -> str | None:
        return str(row.get("name")) if row.get("name") else None

    @staticmethod
    def get_igsn(row: dict[str, Any]) -> str | None:
        meta = row.get("meta")
        if isinstance(meta, dict) and meta.get("igsn"):
            return str(meta.get("igsn"))
        return None
import traceback

import girder_client
import pytest
import requests

from maxima_dagster import resources


class _FakeClient:
    def __init__(self, apiUrl: str, apiKey: str, session):
        self.apiUrl = apiUrl
        self.apiKey = apiKey
        self.session = session


def test_girder_connection_builds_client_with_api_credentials(monkeypatch) -> None:
    monkeypatch.setattr(resources, "GirderClientWithSession", _FakeClient)

    connection = resources.GirderConnection(
        api_url="https://girder.example/api/v1",
        api_key="secret",
    )
    client = connection._make_client()

    assert isinstance(client, _FakeClient)
    assert client.apiUrl == "https://girder.example/api/v1"
    assert client.apiKey == "secret"
    assert "User-Agent" in client.session.headers


_KEY = "s3cr3tK3y+/="
_URL = "https://girder.example/api/v1"


@pytest.mark.parametrize(
    "error",
    [
        girder_client.HttpError(
            502, "Bad Gateway", f"{_URL}/api_key/token?key={_KEY}", "POST"
        ),
        requests.exceptions.ConnectTimeout(
            "HTTPSConnectionPool(host='girder.example', port=443): Max retries "
            "exceeded with url: /api/v1/api_key/token?key=s3cr3tK3y%2B%2F%3D"
        ),
    ],
)
def test_authentication_errors_do_not_leak_api_key(monkeypatch, error) -> None:
    def _fail(self, *args, **kwargs):
        raise error

    monkeypatch.setattr(resources.GirderClientWithSession, "authenticate", _fail)

    with pytest.raises(resources.GirderAuthenticationError) as excinfo:
        resources.GirderClientWithSession(apiUrl=_URL, apiKey=_KEY)

    # Render the way Dagster does (traceback incl. chained exceptions).
    rendered = "".join(
        traceback.format_exception(excinfo.type, excinfo.value, excinfo.tb)
    )
    assert type(error).__name__ in rendered
    assert _KEY not in rendered
    assert "s3cr3tK3y%2B%2F%3D" not in rendered


def test_api_key_is_sent_in_body_not_url(monkeypatch) -> None:
    calls = []

    def _send(self, method, path, parameters=None, data=None, **kwargs):
        calls.append((method, path, parameters, data))
        return {"authToken": {"token": "tok"}}

    monkeypatch.setattr(girder_client.GirderClient, "sendRestRequest", _send)

    client = resources.GirderClientWithSession(apiUrl=_URL, apiKey=_KEY)

    assert client.token == "tok"
    assert calls == [("POST", "api_key/token", None, {"key": _KEY})]

from MaximaDagster import resources


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

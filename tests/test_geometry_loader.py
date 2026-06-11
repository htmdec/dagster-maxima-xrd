from __future__ import annotations

from MaximaDagster import assets


def test_load_geometry_from_poni_uses_pyfai_integrator(monkeypatch) -> None:
    poni_bytes = b"poni-content"

    class _FakeIntegrator:
        last_loaded_path: str | None = None
        last_loaded_bytes: bytes | None = None

        def __init__(self) -> None:
            self.loaded = None
            self.loaded_bytes = None
            self.dist = 1.1
            self.poni1 = 2.2
            self.poni2 = 3.3
            self.rot1 = 4.4
            self.rot2 = 5.5
            self.rot3 = 6.6
            self.detector = "detector"
            self.wavelength = 7.7

        def load(self, path: str) -> None:
            self.loaded = path
            with open(path, "rb") as handle:
                self.loaded_bytes = handle.read()
            _FakeIntegrator.last_loaded_path = self.loaded
            _FakeIntegrator.last_loaded_bytes = self.loaded_bytes

    monkeypatch.setattr(assets, "PyFAIAzimuthalIntegrator", _FakeIntegrator)

    geometry = assets.load_geometry_from_mem(poni_bytes)

    assert geometry.dist == 1.1
    assert geometry.poni1 == 2.2
    assert geometry.poni2 == 3.3
    assert geometry.rot1 == 4.4
    assert geometry.rot2 == 5.5
    assert geometry.rot3 == 6.6
    assert geometry.wavelength == 7.7
    assert _FakeIntegrator.last_loaded_bytes == poni_bytes
    assert _FakeIntegrator.last_loaded_path is not None

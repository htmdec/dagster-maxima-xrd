from __future__ import annotations

from MaximaDagster import assets


def test_load_geometry_from_poni_uses_pyfai_integrator(monkeypatch) -> None:
    class _FakeIntegrator:
        def __init__(self) -> None:
            self.loaded = None
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

    monkeypatch.setattr(assets, "PyFAIAzimuthalIntegrator", _FakeIntegrator)

    geometry = assets.load_geometry_from_poni("calibration.poni")

    assert geometry.dist == 1.1
    assert geometry.poni1 == 2.2
    assert geometry.poni2 == 3.3
    assert geometry.rot1 == 4.4
    assert geometry.rot2 == 5.5
    assert geometry.rot3 == 6.6
    assert geometry.wavelength == 7.7

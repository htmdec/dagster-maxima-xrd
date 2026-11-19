from contextlib import nullcontext
import hashlib
import io
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("fabio")
pytest.importorskip("pyFAI")
pytest.importorskip("skimage")
pytest.importorskip("scipy")

from maxima_dagster.modules import calibrate
from maxima_dagster.modules.refinement import RefinementError


@pytest.mark.parametrize(
    ("energy", "wavelength", "expected"),
    [
        (12.0, None, calibrate.HC_KEV_M / 12.0),
        (None, 1.23e-10, 1.23e-10),
        (None, None, calibrate.DEFAULT_WAVELENGTH_M),
    ],
)
def test_resolve_wavelength_valid_inputs(energy, wavelength, expected):
    assert calibrate.resolve_wavelength(energy=energy, wavelength=wavelength) == expected


def test_resolve_wavelength_rejects_conflicting_inputs():
    with pytest.raises(ValueError, match="either `energy` or `wavelength`"):
        calibrate.resolve_wavelength(energy=12.0, wavelength=1.0)


@pytest.mark.parametrize(
    ("energy", "wavelength"), [(0.0, None), (float("inf"), None), (None, -1.0)]
)
def test_resolve_wavelength_rejects_invalid_inputs(energy, wavelength):
    with pytest.raises(ValueError, match="must be > 0 and finite"):
        calibrate.resolve_wavelength(energy=energy, wavelength=wavelength)


def test_preprocess_excludes_both_masks_and_pads_bottom():
    image = np.array([[0, 1000, 15], [1, 2**32 - 1, 7]], dtype=np.uint32)
    factory_mask = np.array([[0, 2, 0], [0, 0, 0]], dtype=np.uint32)
    mask = calibrate._effective_mask(image, factory_mask)
    actual = calibrate._preprocess(image, mask, image_size=3)

    log_counts = np.log1p(np.array([0, 15, 1, 7], dtype=np.float32))
    low, high = np.percentile(log_counts, (0.1, 99.9))
    zero, high, one, seven = np.clip((log_counts - low) / (high - low), 0, 1)
    expected = [[zero, 0, high], [one, 0, seven], [0, 0, 0]]
    for channel, (mean, std) in enumerate(zip((.485, .456, .406), (.229, .224, .225))):
        np.testing.assert_allclose(actual[0, channel], (np.array(expected) - mean) / std, atol=1e-6)
    assert actual.dtype == torch.float32


def test_wrong_checkpoint_digest_fails_before_deserialization(monkeypatch):
    def unexpected_load(*args, **kwargs):
        pytest.fail("Invalid checkpoint bytes must not be deserialized")

    monkeypatch.setattr(calibrate.torch, "load", unexpected_load)
    with pytest.raises(ValueError, match="Checkpoint SHA-256 mismatch"):
        calibrate._load_checkpoint(io.BytesIO(b"wrong checkpoint"))


def test_checkpoint_state_is_loaded_strictly(monkeypatch):
    instance = calibrate.MaximaCalibrator.__new__(calibrate.MaximaCalibrator)
    instance.model = torch.nn.Linear(2, 1)
    stream = io.BytesIO()
    torch.save({"weight": torch.ones(1, 2)}, stream)
    monkeypatch.setattr(calibrate.recipe, "CHECKPOINT_SHA256", hashlib.sha256(stream.getvalue()).hexdigest())
    with pytest.raises(RuntimeError, match="Missing key"):
        instance._load_weights(stream)


@pytest.fixture
def calibrator(monkeypatch):
    instance = calibrate.MaximaCalibrator.__new__(calibrate.MaximaCalibrator)
    instance.image_size = 2
    instance.device = torch.device("cpu")
    instance.wavelength = 5.1233139848429864e-11
    instance.detector = "Eiger2Cdte_1M"

    def infer(tensor):
        assert tensor.shape == (1, 3, 2, 2)
        return torch.tensor([[1.25, -.2, 0.3, 0.4, -.5, 4]])

    instance.model = infer
    return instance


@pytest.mark.parametrize("with_factory_mask", [False, True])
def test_calibration_passes_mask_and_denormalized_seed_to_refiner(calibrator, tmp_path, with_factory_mask):
    import pyFAI

    image = np.array([[2**32 - 1, 250], [8, 20]], dtype=np.uint32)
    mask_kwargs = {}
    if with_factory_mask:
        mask_kwargs["factory_mask"] = np.array([[0, 1], [0, 0]], dtype=bool)
    fitted = [0.105, 0.038, 0.034, -0.012, -0.009, 0.0]

    def refine(pattern, mask, seed):
        np.testing.assert_array_equal(mask, [[True, with_factory_mask], [False, False]])
        np.testing.assert_array_equal(pattern, image.astype(np.float32))
        expected = np.asarray(calibrate.recipe.PARAMETER_CENTERS) + (
            np.asarray(calibrate.recipe.PARAMETER_SCALES) * [1.25, -.2, .3, .4, -.5]
        )
        np.testing.assert_allclose(seed, [*expected, 0.0], atol=1e-8)
        return SimpleNamespace(parameters=fitted, diagnostics={"rescue_used": False})

    calibrator.refiner = SimpleNamespace(refine=refine)
    output = tmp_path / "fitted.poni"
    geometry = calibrator.calibrate(image, output, **mask_kwargs)
    restored = pyFAI.load(str(output))
    assert geometry.dist == restored.dist == fitted[0]
    assert geometry.wavelength == restored.wavelength == calibrator.wavelength
    assert calibrator.last_refinement_diagnostics == {"rescue_used": False}


def test_failed_refinement_writes_no_poni(calibrator, tmp_path):
    def fail(*args):
        raise RefinementError("insufficient ring support", diagnostics={})

    calibrator.refiner = SimpleNamespace(refine=fail)
    output = tmp_path / "unfitted.poni"
    with pytest.raises(RefinementError, match="insufficient ring support"):
        calibrator.calibrate(np.ones((2, 2)), output)
    assert not output.exists()
    assert calibrator.last_refinement_diagnostics is None


def test_calibrate_image_loads_frame_and_passes_factory_mask(monkeypatch):
    import fabio

    image = np.arange(4).reshape(2, 2)
    factory_mask = np.zeros((2, 2), dtype=bool)

    class StubCalibrator:
        def __init__(self, *, model_src, **kwargs):
            assert model_src == "model.pth"
            assert kwargs["energy"] == 24.2

        def calibrate(self, frame, output_path, *, factory_mask):
            assert output_path == "out.poni"
            return frame, factory_mask

    monkeypatch.setattr(calibrate, "MaximaCalibrator", StubCalibrator)
    monkeypatch.setattr(fabio, "open", lambda path: nullcontext(SimpleNamespace(data=image)))

    frame, mask = calibrate.calibrate_image(
        "frame.tif", "model.pth", "out.poni", energy=24.2, factory_mask=factory_mask
    )
    assert frame is image
    assert mask is factory_mask

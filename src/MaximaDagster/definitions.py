import os
from dagster import Definitions, define_asset_job

from .assets import xrd_raw, calibration_model, poni, azimuthal_integration
from .sensors import xrd_experiment_sensor, xrd_calibration_sensor
from .resources import GirderConnection
from .io_managers import GirderIOManager


xrd = define_asset_job(
    name="xrd",
    selection=["xrd_raw", "azimuthal_integration"],
)

calibration_precompute = define_asset_job(
    name="calibration_precompute",
    selection=["calibration_model", "poni"],
)

defs = Definitions(
    assets=[xrd_raw, calibration_model, poni, azimuthal_integration],
    jobs=[xrd, calibration_precompute],
    sensors=[xrd_experiment_sensor, xrd_calibration_sensor],
    resources={
        "GirderConnection": GirderConnection(
            api_url=os.getenv("GIRDER_API_URL", ""),
            api_key=os.getenv("GIRDER_API_KEY", ""),
        ),
        "io_manager": GirderIOManager(
            base_dir=os.getenv("DAGSTER_STORAGE_DIR", "/tmp/dagster_storage")
        ),
    },
)
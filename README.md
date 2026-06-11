# MAXIMA Dagster Modules

Dagster project for sensor-driven XRD experiment processing against Girder-backed data from MAXIMA.
It discovers new experiment and calibrant partitions, runs a partitioned asset graph, and publishes
derived outputs (PONI + azimuthal CSV) back to Girder with provenance metadata.

![MAXIMA Dagster architecture](maxima-dagster.png)

## Status

- Active scope: XRD workflow (sensor-driven).
- Out of scope: XRF workflow.
- Runtime model: in-memory processing where possible; pyFAI PONI serialization uses a temporary file boundary.

## Pipeline Overview

### Sensors

- `xrd_experiment_sensor`
  - Polls `aimdl/partition` for `xrd_raw` checksums.
  - Launches `xrd` when checksums change.
  - Gates runs until the closest preceding calibrant has a materialized `poni` asset.

- `xrd_calibration_sensor`
  - Polls `aimdl/partition` for `xrd_calibrant_raw` checksums.
  - Launches `calibration_precompute` when checksums change.

### Jobs

- `xrd`
  - Selection: `xrd_raw` -> `azimuthal_integration` (with partition-mapped `poni` input).

- `calibration_precompute`
  - Selection: `calibration_model` -> `poni`.

### Assets

- `calibration_model`: resolves canonical `.pth` model from Girder.
- `poni`: performs calibration on calibrant scan and emits a PONI payload.
- `xrd_raw`: maps experiment scans to Girder pointers.
- `azimuthal_integration`: integrates scans and emits per-scan CSV payloads.

## Repository Layout

- `src/maxima_dagster/assets.py`: Dagster assets and science-flow orchestration.
- `src/maxima_dagster/sensors.py`: Girder-backed sensor factory, cursor/checksum logic, and partition gating.
- `src/maxima_dagster/definitions.py`: `Definitions`, jobs, sensors, and resources wiring.
- `src/maxima_dagster/resources.py`: Girder client resource and retry/session behavior.
- `src/maxima_dagster/io_managers.py`: upload/download boundary and pointer serialization.
- `src/maxima_dagster/contracts.py`: payload/pointer dataclasses and metadata builders.
- `src/maxima_dagster/partition_mapping.py`: closest-preceding calibrant partition selection.
- `src/maxima_dagster/modules/`: calibration, azimuthal integration, and lattice helper modules.
- `tests/`: unit and behavior tests for assets, sensors, resources, partition mapping, and metadata.
- `data/models/`: local model cache path.
- `dagster_home/`: local Dagster state.
- `dagster_storage/`: local IO manager pointer files and run artifacts.
- `docs/source/`: Sphinx documentation source.
- `docker-compose.yml`, `Dockerfile`, `workspace.yaml`: runtime and deployment wiring.

## Requirements

- Python `>=3.11,<3.15`
- Access to target Girder instance and data hierarchy
- Optional: Docker + Docker Compose

## Environment Configuration

Copy `.env.example` to `.env` and provide values:

- `GIRDER_API_URL`: Girder API base URL.
- `GIRDER_API_KEY`: API key for read/write access.
- `GIRDER_MODEL_ITEM_ID`: Girder item ID containing the calibration model `.pth` file.

Also ensure these are set in your runtime environment (used by sensors/resources when querying partition APIs):

- `BASE_PARENT_ID`: Girder parent object ID used for partition discovery scope.
- `BASE_PARENT_TYPE`: Girder parent object type (for example, `folder` or `collection`).

Optional local overrides:

- `DAGSTER_HOME`: defaults to `./dagster_home` in local dev workflows.
- `DAGSTER_STORAGE_DIR`: defaults to `/tmp/dagster_storage` if unset.

## Local Development

### 1. Create and activate environment

```powershell
conda create -n dagster python=3.11 -y
conda activate dagster
```

### 2. Install project

```powershell
pip install -e .
pip install dagster-webserver dagster-dg-cli pytest pytest-cov python-dotenv sphinx sphinx-rtd-theme
```

### 3. Configure environment

```powershell
Copy-Item .env.example .env
# Edit .env and set additional BASE_PARENT_* values in your shell or launcher
$env:DAGSTER_HOME = (Resolve-Path .\dagster_home)
$env:DAGSTER_STORAGE_DIR = (Resolve-Path .\dagster_storage)
```

### 4. Run Dagster

```powershell
dagster dev -w workspace.yaml
```

Dagster UI: http://localhost:3000

Enable `xrd_experiment_sensor` and `xrd_calibration_sensor` in Automation.

## Testing

```powershell
python -m pytest -vv
```

For coverage:

```powershell
python -m pytest --cov=src/MaximaDagster --cov-report=term-missing
```

## Docker Quickstart

1. Prepare `.env`:

```powershell
Copy-Item .env.example .env
# Add BASE_PARENT_ID and BASE_PARENT_TYPE as needed by your deployment
```

2. Build and start:

```powershell
docker compose up --build -d
```

3. Open UI:

- http://localhost:3000

4. Stream logs:

```powershell
docker compose logs -f dagster-daemon dagster-webserver
```

5. Stop:

```powershell
docker compose down
```


# Part 2 · Road-constrained tracking

Fuse sparse, identified perimeter observations with anonymous interior passage events. Compare road-constrained particle filtering (Road-PF), Kalman filtering, and dead reckoning on the CARLA Town05 road network.

## Quick start

Python 3.10+ and GNU Make are required. These tracking experiments run on CPU and do not require a running CARLA or Mininet-WiFi instance.

From this folder, in a separate virtual environment:

```bash
python -m pip install -r requirements.txt
make smoke PYTHON=python
```

The smoke check uses small scenarios and synthetic corruption to verify the pipeline. For the main experiments:

```bash
make tracking_value       # value of indirect observations
make robustness           # missed events, false events, and timing jitter
make estimator_comparison # estimator comparison and blind gaps
make multivehicle         # 1, 2, and 3 concurrent vehicles
make all                  # all four experiments
```

The bundled `dnn_event_v3` profile is the default for both Make and Python entrypoints. It selects `configs/sidechannel_observation_model.dnn_event_v3.json`; the Town05 graph is also included, so no raw captures are needed. Override settings with, for example, `make tracking_value WORKERS=2 PYTHON=python`; `PROFILE=<name>` selects another available detector profile.

Passage detection and its evaluation live in [Part 1](../Part1-testbed/). Part 2 uses the exported detector calibration.

## Sensing conditions

| Condition | Observations |
| --- | --- |
| A · `perimeter_only` | Sparse direct entry/exit observations. |
| B · `oracle_interior` | Direct observations plus perfect anonymous interior events. |
| C · `sidechannel` | Direct observations plus interior events with measured detector errors. |

## Layout and outputs

| Folder | Contents |
| --- | --- |
| `configs/` | Dataset settings and detector calibration. |
| `data/` | Bundled Town05 road graph. |
| `src/` | Simulation, observation models, estimators, and metrics. |
| `experiments/` | `tracking_value/`, `robustness/`, `estimator_comparison/`, and `multivehicle/`. |

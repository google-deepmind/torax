# Simulation Checkpointing in TORAX

TORAX supports opt-in simulation checkpointing to allow long-running simulations
to be resumed after interruption or failure.

Checkpointing reuses the existing NetCDF output format and is fully compatible
with TORAX restart functionality.

---

## Enabling Checkpointing

Checkpointing is configured via the `checkpointing` section of `ToraxConfig`.

```python
from torax import CheckpointConfig, ToraxConfig

config = ToraxConfig(
    ...,
    checkpointing=CheckpointConfig(
        enabled=True,
        path="checkpoint.nc",
    ),
)
```

---

## Checkpoint Trigger Modes

### Solver-step–based

Write a checkpoint every N solver steps:

```python
CheckpointConfig(
    enabled=True,
    every_n_steps=50,
    path="checkpoint.nc",
)
```

---

### Wall-clock–based

Write a checkpoint every N seconds of wall-clock time:

```python
CheckpointConfig(
    enabled=True,
    every_n_seconds=300.0,
    path="checkpoint.nc",
)
```

---

### Simulation-time–based

Write a checkpoint every N units of simulation time:

```python
CheckpointConfig(
    enabled=True,
    every_n_sim_time=0.05,
    path="checkpoint.nc",
)
```

Multiple trigger modes may be enabled simultaneously. A checkpoint is written
whenever any trigger condition is satisfied.

---

## Abnormal Termination Checkpoints

If a simulation terminates early due to an error (e.g. minimum timestep reached
or non-convergence), TORAX writes a final checkpoint marked with:

- `status = "terminated"`
- `termination_reason = <SimError>`

This allows post-mortem inspection and restart from the last valid state.

---

## Restarting from a Checkpoint

Checkpoint files are fully restart-compatible. To resume from a checkpoint:

```python
ToraxConfig(
    ...,
    restart=FileRestart(path="checkpoint.nc"),
)
```

---

## Notes

- Checkpoints overwrite the same file by default.
- Checkpointing is disabled unless explicitly enabled.
- No solver or physics behavior is modified by checkpointing.

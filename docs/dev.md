# Development

Keep device conversion, retargeting, and robot transport in their respective layers. Named rig presets connect supported components without putting device-specific dispatch in a runtime.

## Repository map

| Layer | Location | Responsibility |
| --- | --- | --- |
| Sources | `src/retargetingkit/sources/` | Native data to canonical frames |
| Retargeters | `src/retargetingkit/retargeting/` | Frames to robot commands |
| Targets | `src/retargetingkit/targets/` | Hardware, transport, or visualization |
| Presets | `src/retargetingkit/configs/teleop/` | Source, solver, target, and connection settings |
| Runtimes | `scripts/teleop/` | Component lifecycle and control loop |

## Validate changes

```shell
uv sync --extra vis --group dev
make ci
```

Add the hardware extras required by your work. During development, run `make format`, `make lint`, `make check`, or `make test` for the relevant check.

Every Python example in these docs runs in `tests/test_docs.py`; device pages run against autospecced devices.

Public functions should have doctests. Group pytest cases by feature and use `torch.testing.assert_close` for numerical comparisons.

## Add a component

Give a source an isolated stream check, a target a connection check with its hardware behavior stated, and a retargeter a runnable example. Add a named rig once the complete composition is supported.

Keep device guides focused on setup, a stream or connection check, and a working entrypoint. Validate example paths and command flags against the implementation.

# Interfaces

Each pipeline stage has a runtime interface and a configuration object that builds it. Named rigs compose those configurations for a supported source and robot.

## Contracts

| Configuration | Builds | Runtime method |
| --- | --- | --- |
| `SourceConfig` | `Source` | `get_frame()` returns a frame, or `None` before data arrives |
| `RetargeterConfig` | `Retargeter` | `solve(frame)` returns a robot command |
| `TargetConfig` | `Target` | `set_qpos(qpos)` or `set_command(command)`, depending on the target |

Call `source_config.build()` and `target_config.build()` to start those components. `retargeter_config.build(device)` constructs and warms up the solver on the selected device.

## Match the frame and command

| Retargeter | Input | Output |
| --- | --- | --- |
| `OptimizationHandRetargeter` | `HandFrame` | `QPosCommand` |
| `NeuralHandRetargeter` | `HandFrame` | `QPosCommand` |
| `ArmRetargeter` | `ArmFrame` | `QPosCommand` |
| `HumanoidRetargeter` | `BodyFrame` | `BasePoseQPosCommand` |

`Target` declares both update methods, but concrete targets implement the methods they support. Use the method expected by the selected runtime; passing only joints to a humanoid target would lose its base pose.

## Lifecycle

A runtime builds the components, reads frames, solves, and updates its targets. It calls `close()` on sources and targets when stopping. See [Teleoperation](../guide/teleoperation.md#override-settings) for entrypoints and overrides, and [Commands and targets](../reference/api/targets.md) for the API contracts.

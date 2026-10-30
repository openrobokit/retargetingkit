![RetargetingKit](../assets/retargetingkit-banner.webp)

RetargetingKit maps human motion to robot hands, arms, and humanoids. It combines RoboKit optimization with neural hand models for recorded motion and real-time teleoperation.

## Key features

- **One pipeline.** Sources produce canonical motion frames; retargeters produce robot commands; targets preview or execute them.
- **Hands and bodies.** Solve finger motion, wrist poses, humanoid motion, and combined body-and-hand interactions.
- **Optimization + neural.** Choose geometry-aware optimization or trained hand inference for the task.
- **Recorded or live.** Retarget clips in parallel or drive robots from VR, gloves, and motion capture.

## How it works

```mermaid
flowchart LR
    Source["`Source
    *VR · gloves · mocap · recordings*`"]
    Retargeter["`Retargeter
    *Optimization + Neural*`"]
    Target["`Target
    *Viser · robot hardware*`"]
    Source -- "Frame: hand · arm · body" --> Retargeter
    Retargeter -- "Command: joints · base pose" --> Target
    classDef endpoint fill:#dbeafe,stroke:#2563eb,color:#111827
    classDef solver fill:#dbc8ea,stroke:#8163a1,stroke-width:2px,color:#111827
    class Source,Target endpoint
    class Retargeter solver
```

| Component | Role | Read more |
| --- | --- | --- |
| Source | Read a device or recording. | [Sources](../concepts/pipeline.md) |
| Frame | Represent human hand, arm, or body motion. | [Frames](../concepts/frames.md) |
| Retargeter | Map motion to a robot through optimization or neural inference. | [Interfaces](../concepts/interfaces.md) |
| Command | Carry robot joints and an optional base pose. | [Commands](../reference/api/targets.md) |
| Target | Preview motion or command hardware. | [Targets](../reference/targets/xhand.md) |

## Features

| [![Hand retargeting](../assets/built-in-tasks/hand-retargeting.webp)](hand_retargeting.md) | [![Geometry-aware hand retargeting](../assets/built-in-tasks/hand-geometry-sim.webp)](hand_retargeting.md#inspect-object-contact) |
| --- | --- |
| **[Hand retargeting](hand_retargeting.md)** Map tracked hand motion to robot joints. | **[Geometry-aware hand retargeting](hand_retargeting.md#inspect-object-contact)** Refine hands against object geometry. |

| [![Humanoid retargeting](../assets/built-in-tasks/humanoid-retargeting.webp)](humanoid_retargeting.md) | [![Geometry-aware humanoid retargeting](../assets/built-in-tasks/humanoid-geometry.webp)](humanoid_retargeting.md) |
| --- | --- |
| **[Humanoid retargeting](humanoid_retargeting.md)** Map captured body motion to humanoids. | **[Geometry-aware humanoid retargeting](humanoid_retargeting.md)** Account for terrain and contact. |

| [![Full-body retargeting](../assets/built-in-tasks/full-body-retargeting.webp)](full_body_retargeting.md) | [![Geometry-aware full-body retargeting](../assets/built-in-tasks/full-body-geometry.webp)](full_body_retargeting.md#retarget-an-indoor-interaction) |
| --- | --- |
| **[Full-body retargeting](full_body_retargeting.md)** Combine body and hand motion. | **[Geometry-aware full-body retargeting](full_body_retargeting.md#retarget-an-indoor-interaction)** Preserve scene and object interactions. |

## Quick links

- [Quickstart](quickstart.md)
- [Installation](installation.md)
- [Framework concepts](../concepts/pipeline.md)
- [Asset setup](assets.md)

<p align="center">
  <a href="https://github.com/openrobokit/retargetingkit"><img src="https://openrobokit.com/assets/docs/retargetingkit/assets/retargetingkit-banner.webp" alt="RetargetingKit" width="100%"></a>
</p>

<p align="center">
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+"></a>
  <a href="https://github.com/openrobokit/retargetingkit/actions/workflows/build.yml"><img src="https://img.shields.io/badge/CI-GitHub_Actions-blue" alt="GitHub Actions"></a>
  <a href="https://openrobokit.com/retargetingkit"><img src="https://img.shields.io/badge/homepage-RetargetingKit-blue" alt="Homepage"></a>
  <a href="https://openrobokit.com/docs/retargetingkit/overview/"><img src="https://img.shields.io/badge/docs-RetargetingKit-blue" alt="Docs"></a>
</p>

RetargetingKit maps live or recorded human motion to robot hands, arms, and humanoids using [RoboKit](https://github.com/openrobokit/robokit) optimization and neural hand models.

The two main applications are:

- 🎮 **Real-time teleoperation**
- ⚡ **Offline retargeting in parallel**

<p align="center">
  <a href="https://openrobokit.com/retargetingkit"><img src="https://openrobokit.com/assets/docs/retargetingkit/assets/retargetingkit-paper-teaser.png" alt="RetargetingKit paper teaser showing hand manipulation and humanoid teleoperation" width="100%"></a><br>
  <em>Real-time teleoperation with dexterous hands and humanoids.</em>
</p>

<p align="center">
  <a href="https://openrobokit.com/retargetingkit"><img src="https://openrobokit.com/assets/docs/retargetingkit/assets/showcase/full-body-scene.png" alt="Humanoid retargeting in a furnished scene" width="100%"></a><br>
  <em>Body-and-hand retargeting in a furnished scene.</em>
</p>

## Quick links

- [Homepage](https://openrobokit.com/retargetingkit)
- [Docs](https://github.com/openrobokit/retargetingkit/blob/main/docs/guide/overview.md)
- [Quickstart](https://github.com/openrobokit/retargetingkit/blob/main/docs/guide/quickstart.md)
- [Installation](https://github.com/openrobokit/retargetingkit/blob/main/docs/guide/installation.md)
- [Development](https://github.com/openrobokit/retargetingkit/blob/main/docs/dev.md)

## Installation

Requires Python 3.10 or newer:

```shell
uv add retargetingkit      # or: pip install retargetingkit
```

See [Installation](https://github.com/openrobokit/retargetingkit/blob/main/docs/guide/installation.md) for source installs, extras, and asset access.

## Features

<table width="100%">
  <tr>
    <td width="33%" align="center" valign="top">
      <a href="https://github.com/openrobokit/retargetingkit/blob/main/docs/guide/hand_retargeting.md#work-with-trajectories"><img src="https://openrobokit.com/assets/docs/retargetingkit/assets/built-in-tasks/hand-retargeting.webp" alt="Tracked hand skeleton and retargeted robot hand side by side" width="300"></a><br>
      <strong><a href="https://github.com/openrobokit/retargetingkit/blob/main/docs/guide/hand_retargeting.md#work-with-trajectories">Hand<br>retargeting</a></strong>
    </td>
    <td width="33%" align="center" valign="top">
      <a href="src/retargetingkit/gallery/hand_contact.py"><img src="https://openrobokit.com/assets/docs/retargetingkit/assets/built-in-tasks/hand-geometry-sim.webp" alt="Robot hand grasping a sage cylinder in a white studio" width="300"></a><br>
      <strong><a href="src/retargetingkit/gallery/hand_contact.py">Geometry-aware hand<br>retargeting</a></strong>
    </td>
    <td width="33%" align="center" valign="top">
      <a href="https://github.com/openrobokit/retargetingkit/blob/main/docs/guide/hand_retargeting.md#preview-live-motion"><img src="https://openrobokit.com/assets/docs/retargetingkit/assets/built-in-tasks/hand-teleoperation.webp" alt="Physical robot hand rotating a purple cube" width="300"></a><br>
      <strong><a href="https://github.com/openrobokit/retargetingkit/blob/main/docs/guide/hand_retargeting.md#preview-live-motion">Real-time hand<br>teleoperation</a></strong>
    </td>
  </tr>
  <tr>
    <td width="33%" align="center" valign="top">
      <a href="https://github.com/openrobokit/retargetingkit/blob/main/docs/guide/humanoid_retargeting.md"><img src="https://openrobokit.com/assets/docs/retargetingkit/assets/built-in-tasks/humanoid-retargeting.webp" alt="Human body skeleton and retargeted humanoid side by side" width="300"></a><br>
      <strong><a href="https://github.com/openrobokit/retargetingkit/blob/main/docs/guide/humanoid_retargeting.md">Humanoid<br>retargeting</a></strong>
    </td>
    <td width="33%" align="center" valign="top">
      <a href="src/retargetingkit/gallery/humanoid_terrain.py"><img src="https://openrobokit.com/assets/docs/retargetingkit/assets/built-in-tasks/humanoid-geometry.webp" alt="Rendered humanoid crouching and climbing on stepped terrain" width="300"></a><br>
      <strong><a href="src/retargetingkit/gallery/humanoid_terrain.py">Geometry-aware humanoid<br>retargeting</a></strong>
    </td>
    <td width="33%" align="center" valign="top">
      <a href="https://github.com/openrobokit/retargetingkit/blob/main/docs/guide/humanoid_retargeting.md"><img src="https://openrobokit.com/assets/docs/retargetingkit/assets/built-in-tasks/humanoid-teleoperation.webp" alt="Physical humanoid crouching to pick up a green bottle" width="300"></a><br>
      <strong><a href="https://github.com/openrobokit/retargetingkit/blob/main/docs/guide/humanoid_retargeting.md">Real-time humanoid<br>teleoperation</a></strong>
    </td>
  </tr>
  <tr>
    <td width="33%" align="center" valign="top">
      <a href="https://github.com/openrobokit/retargetingkit/blob/main/docs/guide/full_body_retargeting.md"><img src="https://openrobokit.com/assets/docs/retargetingkit/assets/built-in-tasks/full-body-retargeting.webp" alt="Standing humanoid turning and taking small steps on a white background" width="300"></a><br>
      <strong><a href="https://github.com/openrobokit/retargetingkit/blob/main/docs/guide/full_body_retargeting.md">Full-body<br>retargeting</a></strong>
    </td>
    <td width="33%" align="center" valign="top">
      <a href="https://github.com/openrobokit/retargetingkit/blob/main/docs/guide/full_body_retargeting.md#retarget-an-indoor-interaction"><img src="https://openrobokit.com/assets/docs/retargetingkit/assets/built-in-tasks/full-body-geometry.webp" alt="Rendered humanoid interacting with objects and furniture in a kitchen on a white background" width="300"></a><br>
      <strong><a href="https://github.com/openrobokit/retargetingkit/blob/main/docs/guide/full_body_retargeting.md#retarget-an-indoor-interaction">Geometry-aware full-body<br>retargeting</a></strong>
    </td>
    <td width="33%" align="center" valign="top">
      <a href="https://github.com/openrobokit/retargetingkit/blob/main/docs/guide/humanoid_retargeting.md#control-both-inspire-hands"><img src="https://openrobokit.com/assets/docs/retargetingkit/assets/built-in-tasks/full-body-teleoperation.webp" alt="Physical humanoid handling flowers at a table" width="300"></a><br>
      <strong><a href="https://github.com/openrobokit/retargetingkit/blob/main/docs/guide/humanoid_retargeting.md#control-both-inspire-hands">Real-time full-body<br>teleoperation</a></strong>
    </td>
  </tr>
</table>

Explore recorded motions and robot presets in the [gallery examples](src/retargetingkit/gallery).

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
    Retargeter -- "Command: robot joints · base pose" --> Target
    classDef endpoint fill:#dbeafe,stroke:#2563eb,color:#111827
    classDef solver fill:#dbc8ea,stroke:#8163a1,stroke-width:2px,color:#111827
    class Source,Target endpoint
    class Retargeter solver
```

| Component | Role | Read more |
| --- | --- | --- |
| Source | Read a device or recording and produce a canonical frame. | [Sources](https://github.com/openrobokit/retargetingkit/blob/main/docs/concepts/pipeline.md#three-stages) |
| Frame | Represent human hand, arm, or body motion. | [Frames](https://github.com/openrobokit/retargetingkit/blob/main/docs/concepts/frames.md) |
| Retargeter | Solve hand, arm, or body motion using RoboKit. | [Interfaces](https://github.com/openrobokit/retargetingkit/blob/main/docs/concepts/interfaces.md) |
| Command | Carry robot joint positions and an optional base pose. | [Commands](https://github.com/openrobokit/retargetingkit/blob/main/docs/reference/api/targets.md) |
| Target | Preview commands or send them to a robot. | [Targets](https://github.com/openrobokit/retargetingkit/blob/main/docs/reference/api/targets.md) |

See the [framework overview](https://github.com/openrobokit/retargetingkit/blob/main/docs/concepts/pipeline.md) for frames, commands,
and named rigs.

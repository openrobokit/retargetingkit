<p align="center">
  <a href="https://openrobokit.com/retargetingkit"><img src="https://raw.githubusercontent.com/openrobokit/retargetingkit/main/docs/assets/retargetingkit-banner.webp" alt="RetargetingKit" width="100%"></a>
</p>

<p align="center">
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+"></a>
  <a href="https://github.com/openrobokit/retargetingkit/actions/workflows/build.yml"><img src="https://img.shields.io/badge/CI-GitHub_Actions-blue" alt="GitHub Actions"></a>
</p>

RetargetingKit maps live or recorded human motion to robot hands, arms, and humanoids using [RoboKit](https://openrobokit.com/robokit) optimization and neural hand models.

The two main applications are:

- 🎮 **Real-time teleoperation**
- ⚡ **Offline retargeting in parallel**

## Quick links

- [Docs](https://openrobokit.com/docs/retargetingkit/overview/)
- [Quickstart](https://openrobokit.com/docs/retargetingkit/quickstart/)
- [Installation](https://openrobokit.com/docs/retargetingkit/install/)
- [Development](https://openrobokit.com/docs/retargetingkit/development/)

## Installation

Requires Python 3.10 or newer:

```shell
uv add retargetingkit      # or: pip install retargetingkit
```

See [Installation](https://github.com/openrobokit/retargetingkit/blob/main/docs/guide/installation.md) for source installs, extras, and asset access.

## Features

<table width="100%">
  <tr>
    <td width="50%" align="center" valign="top">
      <a href="https://openrobokit.com/docs/retargetingkit/hand-optimization/#work-with-trajectories"><img src="https://raw.githubusercontent.com/openrobokit/retargetingkit/main/docs/assets/built-in-tasks/hand-retargeting.webp" alt="Tracked hand skeleton and retargeted robot hand side by side" width="300"></a><br>
      <strong><a href="https://openrobokit.com/docs/retargetingkit/hand-optimization/#work-with-trajectories">Hand<br>retargeting</a></strong>
    </td>
    <td width="50%" align="center" valign="top">
      <a href="src/retargetingkit/gallery/hand_contact.py"><img src="https://raw.githubusercontent.com/openrobokit/retargetingkit/main/docs/assets/built-in-tasks/hand-geometry-sim.webp" alt="Robot hand grasping a sage cylinder in a white studio" width="300"></a><br>
      <strong><a href="src/retargetingkit/gallery/hand_contact.py">Geometry-aware hand<br>retargeting</a></strong>
    </td>
  </tr>
  <tr>
    <td width="50%" align="center" valign="top">
      <a href="https://openrobokit.com/docs/retargetingkit/humanoid-retargeting/"><img src="https://raw.githubusercontent.com/openrobokit/retargetingkit/main/docs/assets/built-in-tasks/humanoid-retargeting.webp" alt="Human body skeleton and retargeted humanoid side by side" width="300"></a><br>
      <strong><a href="https://openrobokit.com/docs/retargetingkit/humanoid-retargeting/">Humanoid<br>retargeting</a></strong>
    </td>
    <td width="50%" align="center" valign="top">
      <a href="src/retargetingkit/gallery/humanoid_terrain.py"><img src="https://raw.githubusercontent.com/openrobokit/retargetingkit/main/docs/assets/built-in-tasks/humanoid-geometry.webp" alt="Rendered humanoid crouching and climbing on stepped terrain" width="300"></a><br>
      <strong><a href="src/retargetingkit/gallery/humanoid_terrain.py">Geometry-aware humanoid<br>retargeting</a></strong>
    </td>
  </tr>
  <tr>
    <td width="50%" align="center" valign="top">
      <a href="https://openrobokit.com/docs/retargetingkit/full-body/"><img src="https://raw.githubusercontent.com/openrobokit/retargetingkit/main/docs/assets/built-in-tasks/full-body-retargeting.webp" alt="Standing humanoid turning and taking small steps on a white background" width="300"></a><br>
      <strong><a href="https://openrobokit.com/docs/retargetingkit/full-body/">Full-body<br>retargeting</a></strong>
    </td>
    <td width="50%" align="center" valign="top">
      <a href="https://openrobokit.com/docs/retargetingkit/full-body/#retarget-an-indoor-interaction"><img src="https://raw.githubusercontent.com/openrobokit/retargetingkit/main/docs/assets/built-in-tasks/full-body-geometry.webp" alt="Rendered humanoid interacting with objects and furniture in a kitchen on a white background" width="300"></a><br>
      <strong><a href="https://openrobokit.com/docs/retargetingkit/full-body/#retarget-an-indoor-interaction">Geometry-aware full-body<br>retargeting</a></strong>
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
| Source | Read a device or recording and produce a canonical frame. | [Sources](https://openrobokit.com/docs/retargetingkit/pipeline/#three-stages) |
| Frame | Represent human hand, arm, or body motion. | [Frames](https://openrobokit.com/docs/retargetingkit/frames/) |
| Retargeter | Solve hand, arm, or body motion using RoboKit. | [Interfaces](https://openrobokit.com/docs/retargetingkit/protocols/) |
| Command | Carry robot joint positions and an optional base pose. | [Commands](https://openrobokit.com/docs/retargetingkit/ref-targets/) |
| Target | Preview commands or send them to a robot. | [Targets](https://openrobokit.com/docs/retargetingkit/ref-targets/) |

See the [framework overview](https://openrobokit.com/docs/retargetingkit/pipeline/) for frames, commands,
and named rigs.

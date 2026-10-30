# Axis Studio

Stream Noitom body motion from Axis Studio in Calc broadcast mode. `AxisStudioSource` converts the incoming avatar to a canonical Z-up `BodyFrame` for G1 retargeting.

## Connect Axis Studio

The bundled SDK runtime supports Linux x86-64. From the RetargetingKit repository:

```shell
uv sync --extra axis --extra vis
```

In Axis Studio, enable **Calc data broadcasting**, set the destination to the RetargetingKit workstation, and use UDP port **7012**. If a firewall is active:

```shell
sudo ufw allow 7012/udp
```

## Preview G1

```shell
CUDA_VISIBLE_DEVICES=<FREE_GPU> uv run --extra axis --extra vis \
    scripts/teleop/humanoid.py axisstudio_g1_sonic_viser \
    --source.port 7012 --source.human-height 1.75
```

Set `--source.human-height` to the user's height in meters. Open <http://localhost:8080> and check that G1 follows the streamed body motion. Change `--source.port` if Axis Studio uses a different port.

This rig also publishes commands to SONIC. Use `axisstudio_g1_sonic` to publish without Viser. See [humanoid teleoperation](../../guide/humanoid_retargeting.md) for simulation and G1 deployment.

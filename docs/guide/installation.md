# Installation

RetargetingKit needs Python 3.10 or newer.

=== "uv"
```shell
uv add retargetingkit
```
=== "pip"
```shell
pip install retargetingkit
```

## From source

=== "uv"
```shell
uv add git+https://github.com/openrobokit/retargetingkit.git
```
=== "pip"
```shell
pip install git+https://github.com/openrobokit/retargetingkit.git
```

To work on RetargetingKit itself, clone the repository and run `uv sync`. See the [Development Guide](../dev.md).

## Extras

| Extra | What it adds |
| --- | --- |
| `vis` | Browser galleries and Viser previews. |
| `contact` | Hand-object contact examples. |
| `vision-pro`, `pico`, `manus`, `axis` | Live motion sources; see [Sources](../reference/sources/vision_pro.md). |
| `xarm`, `inspire`, `ability` | Physical arm and hand targets; see [Targets](../reference/targets/xarm.md). |
| `realsense` | RealSense camera source. |
| `rby1`, `dexmate` | Humanoid hardware connections. |

The `pico`, `manus`, and `axis` extras use wheels from a separate index:

=== "uv"
```shell
uv add "retargetingkit[manus]" --index https://jianglongye.github.io/torch_packages_builder
```
=== "pip"
```shell
pip install "retargetingkit[manus]" --extra-index-url https://jianglongye.github.io/torch_packages_builder
```

See [Assets](assets.md), then start with the [Quickstart](quickstart.md).

"""Abstract robot-target contracts shared by the teleop runtimes."""

from abc import ABC, abstractmethod

import numpy as np

from retargetingkit.targets.commands import Command


class Target(ABC):
    """A hardware, transport, or viewer consuming robot commands.

    Built-in implementations include:
        - `XArm6`, `XArm7`: xArm hardware over the xArm SDK.
        - `XHand`, `AbilityHand`, `InspireHand`, `SharpaHand`: robot-hand hardware.
        - `SonicTarget`: SONIC v1 humanoid commands over ZMQ.
        - `Viser`: one or more URDFs in a browser.
    """

    def set_qpos(self, qpos: np.ndarray) -> None:
        """Command actuated joint positions in radians."""
        raise NotImplementedError

    def set_command(self, command: Command) -> None:
        """Command one canonical robot command as one atomic update."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Release hardware or transport."""


class TargetConfig(ABC):
    """Robot target blueprint; subclasses live beside their device and make up `retargetingkit.configs.teleop` rigs."""

    @abstractmethod
    def build(self) -> Target:
        """Construct and start the target."""

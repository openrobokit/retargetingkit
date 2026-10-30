"""Generic human-motion retargeter contract."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import tyro

from retargetingkit.sources.frames import Frame
from retargetingkit.targets.commands import Command


FrameT = TypeVar("FrameT", bound=Frame)
CommandT = TypeVar("CommandT", bound=Command)
# Mark a suppressed field opaque so tyro skips robokit task templates, whose `Robot` hints only resolve for type checkers.
OPAQUE = tyro.constructors.PrimitiveConstructorSpec(
    nargs=0, metavar="", instance_from_str=lambda _: None, is_instance=lambda _: True, str_from_instance=lambda _: []
)


class Retargeter(ABC, Generic[FrameT, CommandT]):
    """Map one canonical human frame to one canonical robot command.

    Built-in implementations include:
        - `ArmRetargeter`: one human wrist to one robot arm, by IK.
        - `OptimizationHandRetargeter`: one human hand to one robot hand, by optimization.
        - `NeuralHandRetargeter`: one human hand to one robot hand, by a trained network.
        - `HumanoidRetargeter`: full human body to a floating-base humanoid.
    """

    @abstractmethod
    def solve(self, frame: FrameT) -> CommandT:
        """Retarget one human frame."""


class RetargeterConfig(ABC, Generic[FrameT, CommandT]):
    """Retargeter blueprint; configs live next to their solver and feed `retargetingkit.configs.teleop` rig presets."""

    @abstractmethod
    def build(self, device: str) -> Retargeter[FrameT, CommandT]:
        """Construct and warm up the retargeter."""

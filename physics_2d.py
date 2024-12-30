from abc import ABC, abstractmethod

import numpy as np

from engine import Entity
from vector2D import Vector2D


class PhysicsObject(Entity, ABC):

    @property
    @abstractmethod
    def v(self) -> Vector2D:
        pass

    @property
    @abstractmethod
    def a(self) -> Vector2D:
        pass

    @property
    @abstractmethod
    def pos(self) -> Vector2D:
        pass

    def distance_to(self, other) -> float:
        vec = self.pos - other.pos
        return vec.magnitude()


from abc import ABC, abstractmethod

import numpy as np

from entity import Entity
from vector2D import Vector2D


class PhysicsObject(Entity, ABC):
    def __init__(self, *args, v: Vector2D = Vector2D(0, 0), a: Vector2D = Vector2D(0, 0), **kwargs):
        super().__init__(*args, **kwargs)
        self._v = v
        self._a = a

    @property
    def v(self) -> Vector2D:
        return self._v

    @v.setter
    def v(self, value: Vector2D):
        self._v = value

    @property
    def a(self) -> Vector2D:
        return self._a

    @a.setter
    def a(self, value: Vector2D):
        self._a = value

    def update_physics(self, time_elapsed: float):
        self.v += self.a * time_elapsed
        self.pos += self.v * time_elapsed
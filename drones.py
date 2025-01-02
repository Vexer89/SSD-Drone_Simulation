import random
from typing import List
import logging

import numpy as np
import pygame

from entity import Entity
from physics_2d import PhysicsObject

from game_settings import GameSettings
from vector2D import Vector2D

logger = logging.getLogger(__name__)

#TODO:
# verify
# max speed and acc


class DroneFlock:
    def __init__(self, game_settings: GameSettings):
        self._drones: List[Drone] = None
        self.game_settings = game_settings

    def generate_drones(self, n_drones, rules=None, **kwargs):
        self._drones = [
            Drone(
                pos=np.array([random.randint(0, self.game_settings.map_width),
                              random.randint(0, self.game_settings.map_height)]),
                game_settings=self.game_settings,
                rules=rules,
                flock=self,
                **kwargs,
            )
            for _ in range(n_drones)
        ]

    @property
    def drones(self):
        return self._drones

    def get_local_drones(self, drone: Entity):
        return [other_drone for other_drone in self.drones
                if drone.distance_to(other_drone) < drone.local_radius and drone != other_drone]


class Drone(PhysicsObject):
    def __init__(self, *args, flock: 'DroneFlock', colour=None, rules=None, size=10, local_radius=200,
                 max_velocity=30, speed=20, **kwargs):
        super().__init__(*args, **kwargs)

        self.flock = flock
        self.size = size
        self.local_radius = local_radius
        self.max_velocity = max_velocity
        self.speed = speed
        self.rules = rules or []
        self.n_neighbours = 0

        # Domyślny kolor losowy
        self.colour = colour or (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )

    @PhysicsObject.v.setter
    def v(self, value: Vector2D):
        magnitude = value.magnitude()
        if magnitude > self.max_velocity:
            value *= self.max_velocity / magnitude
        self._v = value

    def draw(self, win):
        # Rysowanie w kształcie strzałki
        if self.v.magnitude() == 0:
            direction = Vector2D(0, 1)
        else:
            direction = self.v.normalized()

        direction *= self.size
        perpendicular = Vector2D(-direction.y, direction.x) * 0.5

        points = [
            self.pos + direction,
            self.pos - direction + perpendicular,
            self.pos - direction - perpendicular
        ]
        pygame.draw.polygon(win, self.colour, [(p.x, p.y) for p in points])

    def update_physics(self, time_elapsed: float):
        """
        Aktualizuje fizykę boida, uwzględniając przyspieszenie wynikające z reguł stada.
        """
        # Pobieranie lokalnych boidów
        local_boids = self.flock.get_local_drones(self)
        self.n_neighbours = len(local_boids)

        # Obliczenie przyspieszenia na podstawie reguł
        self.a = self.calculate_rules(local_boids)

        # Aktualizacja prędkości i pozycji
        self.v += self.a * time_elapsed
        super().update_physics(time_elapsed)

    def calculate_rules(self, local_boids: List['Drone']) -> Vector2D:
        """
        Oblicza wektor przyspieszenia na podstawie reguł zachowania stada.
        """
        # Suma wszystkich wektorów wpływu reguł, uwzględniając wagi reguł
        acceleration = sum(
            rule.evaluate(self, local_boids) * rule.weight
            for rule in self.rules
        )
        return acceleration
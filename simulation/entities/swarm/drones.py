import random
from typing import List
import logging

import numpy as np
import pygame

from ..entity import Entity
from ..physics_2d import PhysicsObject

from simulation.game_settings import GameSettings
from simulation.utils.vector2D import Vector2D

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
                pos=Vector2D(random.randint(0, self.game_settings.map_width),
                              random.randint(0, self.game_settings.map_height)),
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
        local_drones = [other_drone for other_drone in self.drones
                    if drone.distance_to(other_drone) < drone.local_radius and drone != other_drone]
        # Sortowanie po odległości i ograniczenie do 5
        local_drones.sort(key=lambda d: drone.distance_to(d))
        return local_drones[:5]


class Drone(PhysicsObject):
    def __init__(self, *args, flock: 'DroneFlock', colour=None, rules=None, size=10, local_radius=100,
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
            self.v = value.normalized() * self.max_velocity
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
        if self.v.magnitude() > self.max_velocity:
            self.v = self.v.normalized() * self.max_velocity
        #self.pos = self.pos.astype(float)
        self.pos += self.v * time_elapsed
        super().update_physics(time_elapsed)

    def calculate_rules(self, local_boids: List['Drone']) -> Vector2D:
        """
        Oblicza wektor przyspieszenia na podstawie reguł zachowania stada.
        """
        # Suma wszystkich wektorów wpływu reguł, uwzględniając wagi reguł
        acceleration = Vector2D(0, 0)
        for rule in self.rules:
            rule_output = rule.evaluate(self, local_boids)
            #print(f"Rule: {rule.name}, Output: {rule_output}")
            acceleration += rule_output
        return acceleration
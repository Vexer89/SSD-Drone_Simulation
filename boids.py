import random
from abc import ABC, abstractmethod
from typing import List
import logging

import numpy as np
import pygame

from engine import Entity, EntityAction
from physics_2d import PhysicsObject

from game_settings import GameSettings
from obstacle import Obstacle


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class BoidFlock:
    def __init__(self, game_settings: GameSettings):
        self._boids: List[Boid] = None
        self.game_settings = game_settings

    def generate_boids(self, n_boids, positions, rules=None, **kwargs):
        positionsc = np.array(positions)

        self._boids = [
            Boid(
                position=positionsc[i],
                game_settings=self.game_settings,
                rules=rules,
                flock=self,
                **kwargs,
            )
            for i in range(n_boids)
        ]

    @property
    def boids(self):
        return self._boids

    def get_local_boids(self, boid: Entity):
        return [other_boid for other_boid in self.boids
                if boid.distance_to(other_boid) < boid.local_radius and boid != other_boid]


class Boid(PhysicsObject):

    @property
    def a(self):
        return 0

    @property
    def pos(self):
        return self._pos

    def __init__(self, *args, flock: BoidFlock, position, colour=None, rules=None, size=10, local_radius=200, max_velocity=30,
                 speed=20, **kwargs):
        super().__init__(*args, **kwargs)
        logging.debug(f"Inicjalizacja Boid: {kwargs}")

        if colour is None:
            colour = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        if rules is None:
            rules = list()

        self.colour = colour
        self.flock = flock
        self.size = size
        self._pos = np.array(position, dtype=np.float64)

        self.local_radius = local_radius
        self.max_velocity = max_velocity
        self.speed = speed
        self._v = np.array([0.0, 0.0], dtype='float')

        self.rules = rules
        self.n_neighbours = 0

    @property
    def v(self):
        return self._v

    @v.setter
    def v(self, v):

        magnitude = np.linalg.norm(v)
        if magnitude > self.max_velocity:
            v = v * (self.max_velocity/magnitude)

        self._v = v

    def draw(self, win):
        # Kierunek prędkości używany do obliczenia orientacji
        if np.linalg.norm(self.v) > 0:
            direction = self.v / np.linalg.norm(self.v)
        else:
            direction = np.array([1, 0])  # Domyślny kierunek, gdy prędkość jest zerowa

        direction *= self.size
        perpendicular_direction = np.cross(np.array([*direction, 0]), np.array([0, 0, 1]))[:2]

        points = [
            0.5 * direction + self.pos,
            -0.5 * direction + 0.25 * perpendicular_direction + self.pos,
            -0.25 * direction + self.pos,
            -0.5 * direction - 0.25 * perpendicular_direction + self.pos,
        ]
        # Konwersja punktów na int przed rysowaniem
        int_points = [(int(x), int(y)) for x, y in points]
        pygame.draw.polygon(win, self.colour, int_points)

    def update_physics(self, actions: List[EntityAction], time_elapsed):

        local_boids: List[Boid] = self.flock.get_local_boids(self)

        direction = self.calculate_rules(local_boids, actions)
        self.n_neighbours = len(local_boids)

        self.v = self.v + direction * self.speed

        self._pos += self.v * time_elapsed

        logging.debug(f"Aktualizacja fizyki Boid na pozycji {self._pos} z prędkością {self._v}")

        # TODO: Game clock

    def get_debug_text(self):
        return super().get_debug_text() + f", n={self.n_neighbours}"

    def calculate_rules(self, local_boids, actions):
        return sum(
            [rule.evaluate(self, local_boids, actions=actions) * rule.weight for rule in self.rules]
        )



class BoidRule(ABC):
    _name = "BoidRule"

    def __init__(self, weighting: float, game_settings: GameSettings):
        self._weight = weighting
        self.game_settings = game_settings

    @abstractmethod
    def _evaluate(self, boid: Boid, local_boids: List[Boid], actions: List[EntityAction]):
        pass

    def evaluate(self, boid, local_boids: List[Boid], actions):
        output = self._evaluate(boid, local_boids, actions=actions)
        if np.isnan(output).any():
            logger.warning(f"NaN encountered in {self.name}")
            return np.array([0, 0])
        return output

    @property
    def name(self) -> str:
        return self._name

    @property
    def weight(self):
        return self._weight

    @weight.setter
    def weight(self, value):
        self._weight = value


class SimpleSeparationRule(BoidRule):
    def __init__(self, *args, push_force=5, random_movement_factor=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.push_force = push_force
        self.random_movement_factor = random_movement_factor  # dodatkowy parametr do sterowania losowym ruchem

    _name = "Separation"

    def _evaluate(self, boid: Boid, local_boids: List[Boid], **kwargs):
        n = len(local_boids)
        if n > 0:
            direction_offsets = np.array([(boid.pos - other_boid.pos) for other_boid in local_boids])
            magnitudes = np.sum(np.abs(direction_offsets)**2, axis=-1)**(1./2)

            # Avoid division by zero
            epsilon = 1e-8
            magnitudes = np.maximum(magnitudes, epsilon)

            normed_directions = direction_offsets / magnitudes[:, np.newaxis]
            v = np.sum(normed_directions * (self.push_force / magnitudes)[:, np.newaxis], axis=0)
        else:
            # Gdy nie ma lokalnych boidów, generuj losowy wektor ruchu
            v = np.random.randn(2) * self.random_movement_factor

        return np.nan_to_num(v)



class AlignmentRule(BoidRule):
    _name = "Alignment"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _evaluate(self, boid: Boid, local_boids, **kwargs):
        if not local_boids:
            # Kontynuacja ruchu w obecnym kierunku prędkości
            if np.linalg.norm(boid.v) > 0:
                return boid.v / np.linalg.norm(boid.v)
            else:
                return np.array([0, 0])  # Brak ruchu, jeśli prędkość początkowa jest zero

        # Wyrównaj prędkości
        avg_velocity = np.mean([b.v for b in local_boids], axis=0)
        if np.linalg.norm(avg_velocity) == 0:
            return np.array([0, 0])

        # Normalize to match boid speed
        return avg_velocity / np.linalg.norm(avg_velocity)




class CohesionRule(BoidRule):
    _name = "Cohesion"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _evaluate(self, boid: Boid, local_boids: List[Boid], **kwargs):
        if len(local_boids) == 0:
            # Losowy kierunek, gdy nie ma sąsiednich boidów
            random_direction = np.random.rand(2) - 0.5
            random_direction = random_direction / np.linalg.norm(random_direction)
            return random_direction

        average_pos = np.array([b.pos for b in local_boids]).mean(axis=0)
        diff = average_pos - boid.pos
        mag = np.sqrt((diff**2).sum())
        if mag == 0:
            return np.array([0, 0])
        return diff / mag



class AvoidWallsRule(BoidRule):
    def __init__(self, *args, push_force=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.push_force = push_force

    def _evaluate(self, boid: Boid, local_boids, **kwargs):
        fake_boids = np.array([
            [0, boid.pos[1]],
            [self.game_settings.map_width, boid.pos[1]],
            [boid.pos[0], 0],
            [boid.pos[0], self.game_settings.map_height],
        ])

        direction_offsets = boid.pos - fake_boids
        magnitudes = np.sum(np.abs(direction_offsets) ** 2, axis=-1) ** (1. / 2)
        normed_directions = direction_offsets / magnitudes[:, np.newaxis]
        adjusted_magnitudes = magnitudes**2
        v = np.sum(normed_directions * (self.push_force / adjusted_magnitudes)[:, np.newaxis], axis=0)

        return v


class AvoidObstaclesRule(BoidRule):
    def __init__(self, *args, obstacles, push_force=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.obstacles = obstacles
        self.push_force = push_force

    def _evaluate(self, boid, local_boids, **kwargs):
        v = np.array([0.0, 0.0])
        for obstacle in self.obstacles:
            if obstacle.contains(boid.pos):
                direction_offset = boid.pos - obstacle.center
                magnitude = np.linalg.norm(direction_offset)
                if magnitude > 0:
                    normed_direction = direction_offset / magnitude
                    adjusted_magnitude = magnitude**2
                    v += normed_direction * (self.push_force / adjusted_magnitude)
            else:
                closest_point = obstacle.closest_point(boid.pos)
                direction_offset = boid.pos - closest_point
                magnitude = np.linalg.norm(direction_offset)
                if magnitude > 0:
                    normed_direction = direction_offset / magnitude
                    adjusted_magnitude = magnitude**2
                    v += normed_direction * (self.push_force / adjusted_magnitude)
        return v

class MoveRightRule(BoidRule):
    _name = "MoveRight"

    def _evaluate(self, boid: Boid, local_boids: List[Boid], **kwargs):
        return np.array([10, 0])


class NoiseRule(BoidRule):
    _name = "Noise"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _evaluate(self, boid, local_boids: List[Boid], **kwargs):
        return np.random.uniform(-1, 1, 2)


# class SpiralRule(BoidRule):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def _evaluate(self, boid: Boid, local_boids: List[Boid], **kwargs):
#         return np.cross(boid.v, np.array([0, 0, 1]))[:2]


class SideBySideFormationRule(BoidRule):
    _name = "SideBySideFormation"

    def __init__(self, *args, spacing=50, noise_factor=0.1, **kwargs):
        """
        spacing: Desired lateral spacing between boids in the line.
        noise_factor: Amount of randomness to introduce in the movement.
        """
        super().__init__(*args, **kwargs)
        self.spacing = spacing
        self.noise_factor = noise_factor

    def _evaluate(self, boid: Boid, local_boids: List[Boid], **kwargs):
        if not local_boids:
            # If no neighbors, introduce some random noise
            return np.random.uniform(-1, 1, 2) * self.noise_factor

        # Fixed direction for the line (horizontal in this case)
        line_direction = np.array([1.0, 0.0])  # Boids will form a horizontal line
        lateral_direction = np.array([-line_direction[1], line_direction[0]])  # Perpendicular to the line

        # Initialize force
        lateral_force = np.array([0.0, 0.0], dtype=float)

        for other_boid in local_boids:
            # Compute relative position
            relative_position = boid.pos - other_boid.pos

            # Project the relative position onto the lateral direction
            lateral_distance = np.dot(relative_position, lateral_direction)

            # Apply force to maintain side-by-side spacing
            if abs(lateral_distance) < self.spacing:
                adjustment = (self.spacing - abs(lateral_distance)) / self.spacing
                sign = -1 if lateral_distance > 0 else 1
                lateral_force += sign * lateral_direction * adjustment

        # Add some randomness (noise) to the movement
        noise = np.random.uniform(-1, 1, 2) * self.noise_factor

        # Combine lateral force and noise
        total_force = lateral_force + noise

        # Ensure the result is finite
        return np.nan_to_num(total_force)



# class ControlRule(BoidRule):
#     def __init__(self, *args, control_factor, **kwargs):
#         self.control_factor = control_factor
#         super().__init__(*args, **kwargs)

#     def _evaluate(self, boid: Boid, local_boids: List[Boid], actions: List[EntityAction]=None):
#         v = np.array([0, 0])

#         if actions is None:
#             return v

#         for action in actions:
#             if action == EntityAction.MOVE_UP:
#                 v[1] -= self.control_factor
#                 continue
#             if action == EntityAction.MOVE_DOWN:
#                 v[1] += self.control_factor
#                 continue
#             if action == EntityAction.MOVE_LEFT:
#                 v[0] -= self.control_factor
#                 continue
#             if action == EntityAction.MOVE_RIGHT:
#                 v[0] += self.control_factor

#         return v

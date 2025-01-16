import logging
from abc import ABC, abstractmethod
from typing import List
import numpy as np
from simulation.entities.swarm.drones import Drone
from simulation.game_settings import GameSettings
from simulation.utils.vector2D import Vector2D

logger = logging.getLogger(__name__)

# TODO:
# rewrite rules to vectors
# attractiveness rule

class BoidRule(ABC):
    _name = "BoidRule"

    def __init__(self, weighting: float, game_settings: GameSettings):
        self._weight = weighting
        self.game_settings = game_settings

    @abstractmethod
    def _evaluate(self, boid: Drone, local_boids: List[Drone]):
        pass

    def evaluate(self, boid, local_boids: List[Drone]):
        output = self._evaluate(boid, local_boids)
        if not np.isfinite(output.x) or not np.isfinite(output.y):
            logger.warning(f"NaN or Inf encountered in {self.name}")
            return Vector2D(0, 0)
        return output

    @property
    def name(self) -> str:
        return self._name

    @property
    def weight(self):
        return self._weight

    @weight.setter
    def weight(self, value):
        if value < 0:
            raise ValueError("Weight must be non-negative")
        self._weight = value


class SimpleSeparationRule(BoidRule):
    def __init__(self, *args, push_force=5, **kwargs):
        super().__init__(*args, **kwargs)
        if push_force <= 0:
            raise ValueError("push_force must be greater than zero")
        self.push_force = push_force

    _name = "Separation"

    def _evaluate(self, boid: Drone, local_boids: List[Drone], **kwargs):
        if not local_boids:
            return Vector2D(0, 0)

        direction_offsets = np.array([boid.pos - other_boid.pos for other_boid in local_boids])
        magnitudes = np.maximum(np.linalg.norm(direction_offsets, axis=-1), 1e-6)
        normed_directions = direction_offsets / magnitudes[:, np.newaxis]
        acceleration = np.sum(normed_directions * (self.push_force / magnitudes)[:, np.newaxis], axis=0)

        return Vector2D(acceleration[0], acceleration[1])


class AlignmentRule(BoidRule):
    _name = "Alignment"

    def _evaluate(self, boid: Drone, local_boids, **kwargs):
        if not local_boids:
            return Vector2D(0, 0)

        avg_velocity = np.mean([b.v for b in local_boids], axis=0)
        norm = np.linalg.norm(avg_velocity)
        if norm == 0:
            return Vector2D(0, 0)

        acceleration = avg_velocity / norm
        return Vector2D(acceleration[0], acceleration[1])


class CohesionRule(BoidRule):
    _name = "Cohesion"

    def _evaluate(self, boid: Drone, local_boids: List[Drone], **kwargs):
        if not local_boids:
            return Vector2D(0, 0)

        average_pos = np.mean([b.pos for b in local_boids], axis=0)
        diff = average_pos - boid.pos
        mag = np.linalg.norm(diff)
        if mag == 0:
            return Vector2D(0, 0)

        acceleration = diff / mag
        return Vector2D(acceleration[0], acceleration[1])


class AvoidWallsRule(BoidRule):
    def __init__(self, *args, push_force=5, **kwargs):
        super().__init__(*args, **kwargs)
        if push_force <= 0:
            raise ValueError("push_force must be greater than zero")
        self.push_force = push_force

    def _evaluate(self, boid: Drone, local_boids, **kwargs):
        fake_boids = np.array([
            [0, boid.pos[1]],
            [self.game_settings.map_width, boid.pos[1]],
            [boid.pos[0], 0],
            [boid.pos[0], self.game_settings.map_height],
        ])

        direction_offsets = boid.pos - fake_boids
        magnitudes = np.maximum(np.linalg.norm(direction_offsets, axis=-1), 1e-6)
        normed_directions = direction_offsets / magnitudes[:, np.newaxis]
        adjusted_magnitudes = magnitudes**2
        acceleration = np.sum(normed_directions * (self.push_force / adjusted_magnitudes)[:, np.newaxis], axis=0)

        return Vector2D(acceleration[0], acceleration[1])


class AvoidObstaclesRule(BoidRule):
    def __init__(self, *args, obstacles, push_force=5, **kwargs):
        super().__init__(*args, **kwargs)
        if push_force <= 0:
            raise ValueError("push_force must be greater than zero")
        self.obstacles = obstacles
        self.push_force = push_force

    def _evaluate(self, boid, local_boids, **kwargs):
        acceleration = np.array([0.0, 0.0])
        for obstacle in self.obstacles:
            if obstacle.contains(boid.pos):
                direction_offset = boid.pos - obstacle.center
                magnitude = np.linalg.norm(direction_offset)
                if magnitude > 0:
                    normed_direction = direction_offset / magnitude
                    acceleration += normed_direction * (self.push_force / magnitude**2)
            else:
                closest_point = obstacle.closest_point(boid.pos)
                direction_offset = boid.pos - closest_point
                magnitude = np.linalg.norm(direction_offset)
                if magnitude > 0:
                    normed_direction = direction_offset / magnitude
                    acceleration += normed_direction * (self.push_force / magnitude**2)
        return Vector2D(acceleration[0], acceleration[1])


class MoveRightRule(BoidRule):
    _name = "MoveRight"

    def _evaluate(self, boid: Drone, local_boids: List[Drone], **kwargs):
        return Vector2D(10, 0)


class SideBySideFormationRule(BoidRule):
    _name = "SideBySideFormation"

    def __init__(self, *args, spacing=50, noise_factor=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        if spacing <= 0:
            raise ValueError("spacing must be greater than zero")
        if noise_factor < 0:
            raise ValueError("noise_factor must be non-negative")
        self.spacing = spacing
        self.noise_factor = noise_factor

    def _evaluate(self, boid: Drone, local_boids: List[Drone], **kwargs):
        if not local_boids:
            noise = np.random.uniform(-1, 1, 2) * self.noise_factor
            return Vector2D(noise[0], noise[1])

        line_direction = np.array([1.0, 0.0])
        lateral_direction = np.array([-line_direction[1], line_direction[0]])

        relative_positions = np.array([boid.pos - other_boid.pos for other_boid in local_boids])
        lateral_distances = np.dot(relative_positions, lateral_direction)
        adjustments = np.clip((self.spacing - np.abs(lateral_distances)) / self.spacing, 0, 1)
        lateral_force = np.sum(adjustments[:, np.newaxis] * np.sign(lateral_distances)[:, np.newaxis] * lateral_direction, axis=0)

        noise = np.random.uniform(-1, 1, 2) * self.noise_factor
        total_force = lateral_force + noise

        return Vector2D(total_force[0], total_force[1])
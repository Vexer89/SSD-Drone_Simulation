import logging
from abc import ABC, abstractmethod
from typing import List

import numpy as np

from drones import Drone
from game_settings import GameSettings
from vector2D import Vector2D

logger = logging.getLogger(__name__)

#TODO:
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

    def evaluate(self, boid, local_boids: List[Drone], actions):
        output = self._evaluate(boid, local_boids)
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
    def __init__(self, *args, push_force=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.push_force = push_force

    _name = "Separation"

    def _evaluate(self, boid: Drone, local_boids: List[Drone], **kwargs):
        n = len(local_boids)
        if n > 1:
            direction_offsets = np.array([(boid.pos - other_boid.pos) for other_boid in local_boids])
            magnitudes = np.sum(np.abs(direction_offsets)**2, axis=-1)**(1./2)
            normed_directions = direction_offsets / magnitudes[:, np.newaxis]
            acceleration = np.sum(normed_directions * (self.push_force/magnitudes)[:, np.newaxis], axis=0)
            acceleration_vector = Vector2D(acceleration[0], acceleration[1])
        else:
            acceleration_vector = Vector2D(0, 0)

        return acceleration_vector


class AlignmentRule(BoidRule):
    _name = "Alignment"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _evaluate(self, boid: Drone, local_boids, **kwargs):
        if not local_boids:
            return Vector2D(0, 0)

        # Align velocities
        avg_velocity = np.mean([b.v for b in local_boids], axis=0)
        if np.linalg.norm(avg_velocity) == 0:
            return Vector2D(0, 0)

        # Normalize to match boid speed
        acceleration = avg_velocity / np.linalg.norm(avg_velocity)
        return Vector2D(acceleration[0], acceleration[1])


class CohesionRule(BoidRule):
    _name = "Cohesion"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _evaluate(self, boid: Drone, local_boids: List[Drone], **kwargs):
        if len(local_boids) == 0:
            return Vector2D(0, 0)
        average_pos = np.array([b.pos for b in local_boids]).mean(axis=0)
        diff = average_pos - boid.pos
        mag = np.sqrt((diff**2).sum())
        if mag == 0:
            return Vector2D(0, 0)
        acceleration = diff / mag
        return Vector2D(acceleration[0], acceleration[1])


class AvoidWallsRule(BoidRule):
    def __init__(self, *args, push_force=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.push_force = push_force

    def _evaluate(self, boid: Drone, local_boids, **kwargs):
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
        acceleration = np.sum(normed_directions * (self.push_force / adjusted_magnitudes)[:, np.newaxis], axis=0)

        return acceleration


class AvoidObstaclesRule(BoidRule):
    def __init__(self, *args, obstacles, push_force=5, **kwargs):
        super().__init__(*args, **kwargs)
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
                    adjusted_magnitude = magnitude**2
                    acceleration += normed_direction * (self.push_force / adjusted_magnitude)
            else:
                closest_point = obstacle.closest_point(boid.pos)
                direction_offset = boid.pos - closest_point
                magnitude = np.linalg.norm(direction_offset)
                if magnitude > 0:
                    normed_direction = direction_offset / magnitude
                    adjusted_magnitude = magnitude**2
                    acceleration += normed_direction * (self.push_force / adjusted_magnitude)
        return acceleration


class MoveRightRule(BoidRule):
    _name = "MoveRight"

    def _evaluate(self, boid: Drone, local_boids: List[Drone], **kwargs):
        return np.array([10, 0])


class SideBySideFormationRule(BoidRule):
    _name = "SideBySideFormation"

    def __init__(self, *args, spacing=50, noise_factor=0.1, **kwargs):
        """
        spacing: Desired lateral spacing between drones in the line.
        noise_factor: Amount of randomness to introduce in the movement.
        """
        super().__init__(*args, **kwargs)
        self.spacing = spacing
        self.noise_factor = noise_factor

    def _evaluate(self, boid: Drone, local_boids: List[Drone], **kwargs):
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

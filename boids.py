import random
from abc import ABC, abstractmethod
from typing import List
import logging

import numpy as np
import pygame

from engine import Entity, EntityAction
from physics_2d import PhysicsObject

from game_settings import GameSettings

#
# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


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
        def is_within_side_view(target):
            # Wektor od boida do celu
            vector_to_target = target.pos - boid.pos
            # Normalizacja wektorów
            vector_to_target_normalized = vector_to_target / np.linalg.norm(vector_to_target)
            boid_direction_normalized = boid.v / np.linalg.norm(boid.v) if np.linalg.norm(boid.v) > 0 else np.array([1, 0])

            # Kąt między wektorem prędkości boida a wektorem do celu
            dot_product = np.dot(vector_to_target_normalized, boid_direction_normalized)
            angle = np.arccos(np.clip(dot_product, -1.0, 1.0))

            # Sprawdź, czy kąt jest w preferowanym zakresie (do 150 stopni w każdą stronę od przodu)
            return (angle > np.radians(30)) and (angle < np.radians(150))

        return [other_boid for other_boid in self.boids
                if boid != other_boid and
                np.linalg.norm(boid.pos - other_boid.pos) < boid.local_radius and
                is_within_side_view(other_boid)]
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
            # logger.warning(f"NaN encountered in {self.name}")
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

    def _evaluate(self, boid: Boid, local_boids, **kwargs):
        if not local_boids:
            if np.linalg.norm(boid.v) > 0:
                return boid.v / np.linalg.norm(boid.v)
            else:
                # Losowy kierunek, gdy prędkość jest zerowa
                random_direction = np.random.randn(2)
                return random_direction / np.linalg.norm(random_direction)

        velocities = np.array([b.v for b in local_boids])
        avg_velocity = np.mean(velocities, axis=0)
        return avg_velocity / np.linalg.norm(avg_velocity) if np.linalg.norm(avg_velocity) > 0 else np.zeros(2)



class CohesionRule(BoidRule):
    _name = "Cohesion"

    def _evaluate(self, boid: Boid, local_boids: List[Boid], **kwargs):
        if len(local_boids) == 0:
            inertia = boid.v * 0.95  # Zachowaj 95% obecnej prędkości
            small_random_component = np.random.randn(2) * 0.05  # Dodaj 5% losowości
            new_direction = inertia + small_random_component
            return new_direction / np.linalg.norm(new_direction)


        average_pos = np.mean([b.pos for b in local_boids], axis=0)
        offset = np.zeros(2)
        for other in local_boids:
            # Przesunięcie każdego boida, aby uniknąć lądowania za innym boidem
            vector_between = boid.pos - other.pos
            distance = np.linalg.norm(vector_between)
            if distance < boid.local_radius / 2:  # Jeśli są zbyt blisko siebie
                offset += vector_between / distance  # Odsuń się proporcjonalnie do odległości

        direction_to_avg = average_pos - boid.pos
        if np.linalg.norm(direction_to_avg) > 0:
            direction_to_avg /= np.linalg.norm(direction_to_avg)

        return direction_to_avg + 0.1 * offset  # Ustawione, aby offset miał mniejszy wpływ


class AntiCollisionRule(BoidRule):
    _name = "AntiCollision"

    def _evaluate(self, boid: Boid, local_boids: List[Boid], **kwargs):
        force = np.zeros(2)
        for other in local_boids:
            diff = boid.pos - other.pos
            dist = np.linalg.norm(diff)
            if dist < 30:  # Minimalna odległość do utrzymania
                force += diff / dist  # Oddal się od innych boidów

        return force



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
        super().__init__(*args, **kwargs)
        self.spacing = spacing
        self.noise_factor = noise_factor

    def _evaluate(self, boid: Boid, local_boids: List[Boid], **kwargs):
        if not local_boids:
            return np.random.uniform(-1, 1, 2) * self.noise_factor

        # Calculate the average velocity of local boids to determine the line direction
        avg_velocity = np.mean([b.v for b in local_boids], axis=0)
        if np.linalg.norm(avg_velocity) > 0:
            line_direction = avg_velocity / np.linalg.norm(avg_velocity)
        else:
            line_direction = np.array([0.0, 0.0])  # Default direction if stationary

        lateral_direction = np.array([-line_direction[1], line_direction[0]])  # Perpendicular to line direction

        lateral_force = np.zeros(2)
        for other_boid in local_boids:
            relative_position = boid.pos - other_boid.pos
            lateral_distance = np.dot(relative_position, lateral_direction)

            if abs(lateral_distance) < self.spacing:
                adjustment = (self.spacing - abs(lateral_distance)) / self.spacing
                sign = -1 if lateral_distance > 0 else 1
                lateral_force += sign * lateral_direction * adjustment

        noise = np.random.uniform(-1, 1, 2) * self.noise_factor
        total_force = lateral_force + noise

        return np.nan_to_num(total_force)


class AttractionRule(BoidRule):
    def __init__(self, *args, sim_map, **kwargs):

        super().__init__(*args, **kwargs)
        self.sim_map = sim_map

    def _evaluate(self, boid: Boid, local_boids: List[Boid], **kwargs):

        max_attractiveness = -float('inf')
        best_sector = None

        for row in self.sim_map.sectors:
            for sector in row:
                attractiveness = sector.attractiveness
                if attractiveness > max_attractiveness:
                    max_attractiveness = attractiveness
                    if self.sim_map.best_sector:
                        if abs(max_attractiveness - self.sim_map.best_sector.attractiveness) <= 0.1:
                            best_sector = self.sim_map.best_sector
                    else:
                        best_sector = sector

        self.sim_map.best_sector = best_sector

        if best_sector:
            direction_to_sector = np.array(best_sector.center) - np.array(boid.pos)
            norm_direction = direction_to_sector / np.linalg.norm(direction_to_sector)
            return norm_direction

        return np.array([0, 0])

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

import numpy as np

class Vector2D(np.ndarray):
    def __new__(cls, x: float, y: float):
        # Tworzymy nowy obiekt klasy Vector2D jako tablicę NumPy
        obj = np.asarray([x, y], dtype=float).view(cls)
        return obj

    def __init__(self, x: float, y: float):
        # `__new__` już utworzyło tablicę, więc tu nie musimy nic robić
        pass

    def magnitude(self) -> float:
        """Zwraca długość wektora."""
        return np.linalg.norm(self)

    def __add__(self, other: "Vector2D") -> "Vector2D":
        """Dodawanie dwóch wektorów."""
        result = np.add(self, other)
        return Vector2D(result[0], result[1])

    def __sub__(self, other: "Vector2D") -> "Vector2D":
        """Odejmowanie dwóch wektorów."""
        result = np.subtract(self, other)
        return Vector2D(result[0], result[1])

    def __mul__(self, scalar: float) -> "Vector2D":
        """Mnożenie wektora przez skalar."""
        result = np.multiply(self, scalar)
        return Vector2D(result[0], result[1])

    def __imul__(self, scalar: float) -> "Vector2D":
        """Mnożenie wektora przez skalar w miejscu (in-place)."""
        self[0] *= scalar
        self[1] *= scalar
        return self

    def __iadd__(self, other: "Vector2D") -> "Vector2D":
        """Dodawanie wektora do wektora w miejscu (in-place)."""
        self[0] += other[0]
        self[1] += other[1]
        return self

    #+, -, *, *=, +=

    def __repr__(self):
        """Reprezentacja techniczna."""
        return f"Vector2D({self[0]:.2f}, {self[1]:.2f})"

    def __str__(self):
        """Przyjazna reprezentacja tekstowa."""
        return f"({self[0]:.2f}, {self[1]:.2f})"

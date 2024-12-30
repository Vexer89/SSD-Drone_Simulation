import numpy as np

class Vector2D(np.ndarray):
    def __new__(cls, x: float, y: float):
        # Tworzymy nowy obiekt klasy Vector2 jako tablicę NumPy
        obj = np.asarray([x, y], dtype=float).view(cls)
        return obj

    def __init__(self, x: float, y: float):
        # `__new__` już utworzyło tablicę, więc tu nie musimy nic robić
        pass
    
    def magnitude(self) -> float:
        """Zwraca długość wektora."""
        return np.linalg.norm(self)

    def normalize(self):
        """Normalizuje wektor do jednostkowego."""
        mag = self.magnitude()
        if mag > 0:
            self /= mag

    def dot(self, other):
        """Iloczyn skalarny z innym wektorem."""
        return np.dot(self, other)

    def __add__(self, other):
        """Dodawanie dwóch wektorów."""
        result = super().__add__(other)
        return Vector2D(result[0], result[1])

    def __sub__(self, other):
        """Odejmowanie dwóch wektorów."""
        result = super().__sub__(other)
        return Vector2D(result[0], result[1])

    def __repr__(self):
        """Reprezentacja wektora w formie tekstowej."""
        return f"Vector2D({self[0]}, {self[1]})"

# Przykład użycia
v1 = Vector2D(3, 4)
v2 = Vector2D(1, 2)

print("Wektor v1:", v1)  # Vector2(3.0, 4.0)
print("Długość v1:", v1.magnitude())  # 5.0

v1.normalize()
print("Znormalizowany v1:", v1)  # Vector2(0.6, 0.8)

v3 = v1 + v2
print("Dodanie wektorów v1 + v2:", v3)  # Vector2(1.6, 2.8)

dot_product = v1.dot(v2)
print("Iloczyn skalarny v1 . v2:", dot_product)  # 2.0

print(v1)
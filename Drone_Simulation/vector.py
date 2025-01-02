import math

class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        """Dodawanie dwóch wektorów."""
        return Vector(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        """Odejmowanie dwóch wektorów."""
        return Vector(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar):
        """Mnożenie wektora przez skalar."""
        return Vector(self.x * scalar, self.y * scalar)

    def __truediv__(self, scalar):
        """Dzielenie wektora przez skalar."""
        if scalar == 0:
            raise ValueError("Nie można dzielić przez zero.")
        return Vector(self.x / scalar, self.y / scalar)

    def magnitude(self):
        """Obliczanie długości wektora."""
        return math.sqrt(self.x**2 + self.y**2)

    def normalize(self):
        """Normalizacja wektora."""
        magnitude = self.magnitude()
        if magnitude == 0:
            return Vector(0, 0)
        return self / magnitude

    def dot(self, other):
        """Iloczyn skalarny dwóch wektorów."""
        return self.x * other.x + self.y * other.y

    def cross(self, other):
        """Iloczyn wektorowy (tylko z projekcją na Z)."""
        return self.x * other.y - self.y * other.x

    def distance_to(self, other):
        """Odległość do innego wektora."""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    def to_array(self):
        """Konwersja na tablicę."""
        return [self.x, self.y]

    def __repr__(self):
        return f"Vector({self.x}, {self.y})"

    def __eq__(self, other):
        """Porównanie dwóch wektorów."""
        return self.x == other.x and self.y == other.y

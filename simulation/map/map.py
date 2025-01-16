from sector import Sector

class Map:
    def __init__(self, width, height, rows, columns):
        self.width = width
        self.height = height
        self.rows = rows
        self.columns = columns
        self.sectors = [
            [Sector(row, col, width // columns) for col in range(columns)]
            for row in range(rows)
        ]
        self.__add_neighbours_to_sectors()

    def __add_neighbours_to_sectors(self):
        for row_idx, row in enumerate(self.sectors):
            for col_idx, sector in enumerate(row):
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                    neighbor_row = row_idx + dy
                    neighbor_col = col_idx + dx
                    if 0 <= neighbor_row < self.rows and 0 <= neighbor_col < self.columns:
                        sector.add_neighbor(self.sectors[neighbor_row][neighbor_col])

    def draw(self, screen):
        for row in self.sectors:
            for sector in row:
                sector.draw(screen)

    def update_attractiveness(self):
        for row in self.sectors:
            for sector in row:
                sector.calculate_attractiveness()
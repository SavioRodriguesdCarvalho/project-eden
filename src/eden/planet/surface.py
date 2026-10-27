import numpy as np

class Surface:
    EARTH_RADIUS_METERS = 6_371_000.0
    REFERENCE_WATER_DEPTH = 4_000.0

    def __init__(self, icosphere, topology):
        self.icosphere = icosphere
        self.topology = topology

        self.elevation = np.zeros(len(icosphere.faces), dtype = np.float64)

        self.sea_level = 0.0
        self.water_depth = np.zeros(len(icosphere.faces), dtype = np.float64)
        self.water_volume = 0.0

    def update_water(self):
        self.water_depth = np.maximum(self.sea_level - self.elevation, 0.0)

    def set_water_abundance(self, abundance):
        if not 0.0 <= abundance <= 1.0:
            raise ValueError("A abundância de água deve estar entre 0 e 1.")

        areas = self.icosphere.get_faces_areas() * 1_000_000.0
        target_volume = (abundance * self.REFERENCE_WATER_DEPTH * areas.sum())

        if abundance == 0.0:
            self.sea_level = float(self.elevation.min())
            self.update_water()
            self.water_volume = 0.0

            return

        lower = float(self.elevation.min())
        upper = float(self.elevation.max() + self.REFERENCE_WATER_DEPTH)

        for _ in range(80):
            middle = (lower + upper) / 2.0
            volume = np.sum(areas * np.maximum(middle - self.elevation, 0.0))

            if volume < target_volume:
                lower = middle
            else:
                upper = middle

        self.sea_level = (lower + upper) / 2.0
        self.update_water()
        self.water_volume = float(np.sum(areas * self.water_depth))
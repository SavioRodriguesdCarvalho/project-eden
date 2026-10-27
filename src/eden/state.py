from dataclasses import dataclass
import numpy as np

@dataclass
class PlanetState:
    elapsed_days: float
    season: float
    soil_water_mm: np.ndarray
    snow_water_equivalent_mm: np.ndarray
    surface_temperature_c: np.ndarray

    @classmethod
    def initialize(cls, planet, initial_soil_water_mm = 0.0):
        face_count = len(planet.surface.elevation)

        if initial_soil_water_mm < 0.0:
            raise ValueError("A umidade do solo não pode ser negativa.")

        return cls(
            elapsed_days = 0.0,
            season = 0.0,
            soil_water_mm = np.full(face_count, initial_soil_water_mm, dtype = np.float64),
            snow_water_equivalent_mm = np.zeros(face_count, dtype = np.float64),
            surface_temperature_c = np.zeros(face_count, dtype = np.float64)
        )
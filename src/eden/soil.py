from dataclasses import dataclass
import numpy as np

@dataclass
class SoilFluxes:
    actual_evapotranspiration_mm: np.ndarray
    drainage_mm: np.ndarray

class Soil:
    def __init__(self, water_capacity_mm = 150.0):
        if not np.isfinite(water_capacity_mm) or water_capacity_mm <= 0:
            raise ValueError("A capacidade do solo deve ser positiva.")

        self.water_capacity_mm = float(water_capacity_mm)

    def step(self, state, precipitation_mm, potential_evapotranspiration_mm):
        soil_water = np.asarray(state.soil_water_mm, dtype = np.float64)

        precipitation = np.broadcast_to(np.asarray(precipitation_mm, dtype = np.float64), soil_water.shape)

        potential_et = np.broadcast_to(np.asarray(potential_evapotranspiration_mm, dtype = np.float64), soil_water.shape)

        if not np.all(np.isfinite(soil_water)):
            raise ValueError("A água armazenada contém valores inválidos.")
        if not np.all(np.isfinite(precipitation)):
            raise ValueError("A precipitação contém valores inválidos.")
        if not np.all(np.isfinite(potential_et)):
            raise ValueError("A evapotranspiração potencial contém valores inválidos.")
        if np.any(soil_water < 0) or np.any(soil_water > self.water_capacity_mm):
            raise ValueError("A água armazenada está fora dos limites do solo.")
        if np.any(precipitation < 0) or np.any(potential_et < 0):
            raise ValueError("Precipitação e evapotranspiração devem ser não negativas.")

        available_water = soil_water + precipitation

        actual_et = np.minimum(potential_et, available_water)

        remaining_water = available_water - actual_et
        drainage = np.maximum(remaining_water - self.water_capacity_mm, 0.0)
        new_soil_water = np.minimum(remaining_water, self.water_capacity_mm)
        state.soil_water_mm = new_soil_water.copy()

        return SoilFluxes(
            actual_evapotranspiration_mm = actual_et.copy(),
            drainage_mm = drainage.copy()
        )
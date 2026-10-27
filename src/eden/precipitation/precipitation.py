from collections import deque
import numpy as np


class Precipitation:
    def __init__(self, base_precipitation = 1200.0, ocean_influence_distance = 8.0, *, base_preciptation = None):
        if base_preciptation is not None:
            base_precipitation = base_preciptation

        self.base_precipitation = float(base_precipitation)
        self.ocean_influence_distance = float(ocean_influence_distance)
        if not np.isfinite(self.base_precipitation) or self.base_precipitation < 0.0:
            raise ValueError("A precipitação-base deve ser finita e não negativa.")
        if not np.isfinite(self.ocean_influence_distance) or self.ocean_influence_distance <= 0.0:
            raise ValueError("A distância de influência oceânica deve ser finita e positiva.")
        self.precipitation = None
        self.precipitation_raw = None
        self.precipitation_mm = None
        self.calibration_factor = None
        self.distance_to_ocean = None
        self.normalization_value = None

    def calculate(self, planet, temperature):
        elevation = np.asarray(planet.surface.elevation, dtype = np.float64)
        faces = np.asarray(planet.icosphere.faces)
        temperature = np.asarray(temperature, dtype = np.float64)

        if len(elevation) != len(faces):
            raise ValueError("A elevação deve conter um valor por face.")
        if len(temperature) != len(faces):
            raise ValueError("A temperatura deve conter um valor por face.")
        if not np.all(np.isfinite(temperature)):
            raise ValueError("A temperatura contém valores não finitos.")
        if self.ocean_influence_distance <= 0.0:
            raise ValueError("A influência oceânica deve ser positiva.")

        ocean = elevation < planet.surface.sea_level
        neighbors = planet.icosphere.get_face_neighbors()
        self.distance_to_ocean = self._calculate_distance_to_ocean(neighbors, ocean)

        ocean_influence = np.exp(-self.distance_to_ocean / self.ocean_influence_distance)
        temperature_factor = 0.4 + 0.6 * np.clip((temperature + 20.0) / 35.0, 0.0, 1.0)
        altitude = np.maximum(elevation - planet.surface.sea_level, 0.0)
        altitude_factor = np.clip(1.0 + altitude / 6000.0, 1.0, 1.5)

        self.precipitation = (
            self.base_precipitation
            * ocean_influence
            * temperature_factor
            * altitude_factor
        )
        self.precipitation[ocean] = 0.0
        return self.precipitation

    def calculate_from_atmosphere(self, planet, atmosphere, normalization_percentile = 99.0, evaporation_mean = None):
        if atmosphere.condensed_moisture is None:
            raise ValueError("Execute o cálculo de umidade antes da precipitação.")
        if evaporation_mean is None and getattr(atmosphere, "total_ocean_evaporated", 0.0):
            evaporation_mean = float(atmosphere.total_ocean_evaporated)
        return self.calculate_from_condensation(
            planet,
            atmosphere.condensed_moisture,
            normalization_percentile = normalization_percentile,
            evaporation_mean = evaporation_mean,
        )

    def calculate_from_condensation(
        self,
        planet,
        condensed_moisture,
        normalization_percentile = 99.0,
        evaporation_mean = None,
        reference_ocean_evaporation_mm = 1200.0,
        calibration_factor = None,
    ):
        if not 0.0 < normalization_percentile <= 100.0:
            raise ValueError("O percentil de normalização deve estar em (0, 100].")
        if not np.isfinite(reference_ocean_evaporation_mm) or reference_ocean_evaporation_mm <= 0.0:
            raise ValueError("A referência de evaporação oceânica deve ser finita e positiva.")
        if evaporation_mean is not None and (not np.isfinite(evaporation_mean) or evaporation_mean < 0.0):
            raise ValueError("A evaporação usada na calibração deve ser finita e não negativa.")
        if calibration_factor is not None and (
            not np.isfinite(calibration_factor) or calibration_factor <= 0.0
        ):
            raise ValueError("O fator de calibração explícito deve ser finito e positivo.")

        elevation = np.asarray(planet.surface.elevation, dtype = np.float64)
        faces = np.asarray(planet.icosphere.faces, dtype = np.int64)
        condensed = np.asarray(condensed_moisture, dtype = np.float64)
        if len(condensed) != len(faces) or len(elevation) != len(faces):
            raise ValueError("A condensação e a elevação devem ter um valor por face.")
        if np.any(~np.isfinite(condensed)) or np.any(condensed < 0.0):
            raise ValueError("A condensação contém valores inválidos.")

        ocean = elevation < planet.surface.sea_level
        land = ~ocean
        self.precipitation_raw = condensed.copy()
        self.precipitation = np.zeros(len(faces), dtype = np.float64)

        self.calibration_factor = None
        self.precipitation_mm = np.zeros(len(faces), dtype = np.float64)
        ocean_count = int(np.count_nonzero(ocean))
        if calibration_factor is not None:
            self.calibration_factor = float(calibration_factor)
            self.precipitation_mm = self.precipitation_raw * self.calibration_factor
        elif evaporation_mean is not None and ocean_count > 0 and float(evaporation_mean) > 0.0:
            self.calibration_factor = float(reference_ocean_evaporation_mm) / (float(evaporation_mean) / ocean_count)
            self.precipitation_mm = self.precipitation_raw * self.calibration_factor

        positive_land = condensed[land & (condensed > 0.0)]
        if positive_land.size == 0:
            self.normalization_value = 0.0
            return self.precipitation

        scale = float(np.percentile(positive_land, normalization_percentile))
        if not np.isfinite(scale) or scale <= 0.0:
            scale = float(positive_land.max())

        self.normalization_value = scale
        self.precipitation = self.precipitation_mm.copy()
        self.precipitation[land] = self.precipitation_mm[land]

        return self.precipitation_mm

    def _calculate_distance_to_ocean(self, neighbors, ocean):
        distances = np.full(len(neighbors), np.inf)
        queue = deque()
        for face_id in np.flatnonzero(ocean):
            distances[face_id] = 0
            queue.append(int(face_id))

        while queue:
            current = queue.popleft()
            for neighbor in neighbors[current]:
                if distances[neighbor] == np.inf:
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)
        return distances
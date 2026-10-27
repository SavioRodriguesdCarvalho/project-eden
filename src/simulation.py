from dataclasses import dataclass
import numpy as np

from src.eden.precipitation.precipitation import Precipitation
from src.eden.atmosphere.atmosphere import Atmosphere
from src.eden.hydrology.hydrology import Hydrology
from src.eden.climate.climate import Climate

@dataclass
class SeasonResult:
    season: float
    temperature: np.ndarray
    moisture: np.ndarray
    condensed_moisture: np.ndarray
    precipitation_mm: np.ndarray
    flow_to: np.ndarray
    flow_accumulation: np.ndarray
    evapotranspiration_mm: np.ndarray
    runoff_mm: np.ndarray
    available_water_mm: np.ndarray
    precipitation_volume_m3: float
    evapotranspiration_volume_m3: float
    runoff_volume_m3: float
    total_area_m2: float
    land_area_m2: float
    ocean_area_m2: float
    temperature_area_weighted_mean_c: float
    land_precipitation_area_weighted_mean_mm: float
    land_evapotranspiration_area_weighted_mean_mm: float
    land_precipitation_volume_m3: float
    ocean_precipitation_volume_m3: float
    water_balance_residual_m3: float

class PlanetSimulation:
    def __init__(
        self,
        climate = None,
        atmosphere = None,
        precipitation = None,
        hydrology = None,
        *,
        reference_ocean_evaporation_mm = 1200.0,
        land_evapotranspiration_demand_mm = 300.0,
        atmosphere_iterations = 80,
        evaporation_fraction = 1.0
    ):
        self.climate = climate or Climate()
        self.atmosphere = atmosphere or Atmosphere()
        self.precipitation = precipitation or Precipitation()
        self.hydrology = hydrology or Hydrology()

        self.reference_ocean_evaporation_mm = float(reference_ocean_evaporation_mm)
        self.land_evapotranspiration_demand_mm = float(land_evapotranspiration_demand_mm)
        self.atmosphere_iterations = int(atmosphere_iterations)
        self.evaporation_fraction = float(evaporation_fraction)

        if not np.isfinite(self.reference_ocean_evaporation_mm) or self.reference_ocean_evaporation_mm <= 0:
            raise ValueError("A evaporação oceânica de referência deve ser finita e positiva.")
        if not np.isfinite(self.land_evapotranspiration_demand_mm) or self.land_evapotranspiration_demand_mm < 0:
            raise ValueError("A demanda de evapotranspiração deve ser finita e não negativa.")
        if self.atmosphere_iterations < 1:
            raise ValueError("O número de iterações atmosféricas deve ser positivo.")
        if not 0.0 < self.evaporation_fraction <= 1.0:
            raise ValueError("A fração evaporável deve estar em [0, 1].")

    @staticmethod
    def _face_areas_m2(planet):
        vertices = np.asarray(planet.icosphere.vertices, dtype=np.float64).copy()
        faces = np.asarray(planet.icosphere.faces, dtype=np.int64)

        vertices /= np.maximum(np.linalg.norm(vertices, axis = 1, keepdims = True), 1e-12)
        areas = np.empty(len(faces), dtype = np.float64)

        for face_id, (ia, ib, ic) in enumerate(faces):
            a, b, c = vertices[ia], vertices[ib], vertices[ic]

            determinant = abs(float(np.dot(a, np.cross(b, c))))
            denominator = 1.0 + float(np.dot(a, b) + np.dot(b, c) + np.dot(c, a))
            areas[face_id] = 2.0 * np.arctan2(determinant, max(denominator, 1e-15))

        if np.any(~np.isfinite(areas)) or np.any(areas <= 0.0):
            raise ValueError("A malha contém faces com áreas inválidas.")

        radius_m = float(planet.radius) * 1000.0

        areas_m2 = areas * radius_m ** 2
        if len(areas_m2) != len(planet.surface.elevation):
            raise ValueError("A área deve conter um valor por face.")
        return areas_m2

    @staticmethod
    def _weighted_mean(values, weights, mask = None):
        values = np.asarray(values, dtype = np.float64)
        weights = np.asarray(weights, dtype = np.float64)
        valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
        if mask is not None:
            valid &= np.asarray(mask, dtype = bool)
        if not np.any(valid):
            return 0.0
        return float(np.average(values[valid], weights = weights[valid]))

    def _calibrate_precipitation(self, planet, condensed, areas_m2):
        elevation = np.asanyarray(planet.surface.elevation, dtype = np.float64)
        areas_m2 = np.asanyarray(areas_m2, dtype = np.float64)
        condensed = np.asanyarray(condensed, dtype = np.float64)

        ocean = elevation < float(planet.surface.sea_level)

        if not np.any(ocean) or not np.any(~ocean):
            raise ValueError("A simulação requer faces oceânicas e terrestres.")
        if condensed.shape != elevation.shape or areas_m2.shape != elevation.shape:
            raise ValueError("Condesação e áreas devem conter um valor por face.")

        if (
            np.any(~np.isfinite(condensed))
            or np.any(condensed < 0.0)
            or np.any(~np.isfinite(areas_m2))
            or np.any(areas_m2 <= 0.0)
        ):
            raise ValueError("Condensação ou áreas contém valores inválidos.")

        total_condensation = float(np.sum(condensed * areas_m2))
        ocean_area = float(np.sum(areas_m2[ocean]))

        if total_condensation <= 0.0:
            raise ValueError("Não houve condensação para calibrar a precipitação.")

        target_volume_mm_m2 = self.reference_ocean_evaporation_mm * ocean_area
        calibration_factor = target_volume_mm_m2 / total_condensation

        self.precipitation.calculate_from_condensation(planet, condensed, calibration_factor = calibration_factor)

        precipitation_mm = np.asarray(self.precipitation.precipitation_mm, dtype = np.float64).copy()

        if precipitation_mm.shape != elevation.shape:
            raise ValueError("A precipitação deve conter um valor por face.")

        return np.maximum(np.nan_to_num(precipitation_mm), 0.0)

    def simulate_season(self, planet, season = 0.0):
        if not 0.0 <= season < 1.0:
            raise ValueError("A estação deve estar no intervalo [0, 1].")

        temperature = np.asarray(self.climate.calculate_temperature(planet, season = season), dtype = np.float64)
        self.atmosphere.calculate_wind(planet, season = season)
        self.atmosphere.calculate_moisture(
            planet,
            temperature,
            iterations = self.atmosphere_iterations,
            season = season
        )

        condensed = np.asarray(self.atmosphere.condensed_moisture, dtype = np.float64).copy()
        areas_m2 = self._face_areas_m2(planet)
        precipitation_mm = self._calibrate_precipitation(planet, condensed, areas_m2)

        flow_to = self.hydrology.calculate_flow(planet)
        flow_accumulation = self.hydrology.calculate_flow_accumulation(planet, flow_to, weights = precipitation_mm * areas_m2)

        land = np.asarray(planet.surface.elevation) >= planet.surface.sea_level
        demand = np.zeros_like(precipitation_mm)
        raw_demand = np.asarray(self.atmosphere.land_evaporation_demand, dtype = np.float64)
        positive_land_demand = raw_demand[land & (raw_demand > 0.0)]

        if self.land_evapotranspiration_demand_mm > 0.0 and positive_land_demand.size:
            reference = float(np.percentile(positive_land_demand, 99.0))
            if reference > 0.0:
                demand[land] = (
                    np.clip(raw_demand[land] / reference, 0.0, 1.0)
                    * self.land_evapotranspiration_demand_mm
                )

        routed = self.hydrology.route_evapotranspiration(
            planet,
            flow_to,
            precipitation_mm,
            demand,
            evaporation_fraction = self.evaporation_fraction,
            face_area = areas_m2
        )

        et_mm = routed["et_mm"]
        runoff_mm = routed["outflow_mm"]
        available_mm = routed["available_mm"]

        downstream = np.asarray(flow_to, dtype = np.int64)
        elevation = np.asarray(planet.surface.elevation, dtype = np.float64)
        ocean = elevation < float(planet.surface.sea_level)
        land = ~ocean
        valid_downstream = ((downstream >= 0) & (downstream < len(elevation)))
        exits_to_ocean = land & valid_downstream & ocean[np.clip(downstream, 0, len(ocean) - 1)]
        exits_to_terminal = land & (downstream < 0)
        exits_land = exits_to_ocean | exits_to_terminal
        external_runoff_volume_m3 = float(
            np.sum(np.maximum(np.asarray(runoff_mm)[exits_land], 0.0) * areas_m2[exits_land]) / 1000.0
        )

        total_area_m2 = float(np.sum(areas_m2))
        land_area_m2 = float(np.sum(areas_m2[land]))
        ocean_area_m2 = float(np.sum(areas_m2[ocean]))
        precipitation_volume_m3 = float(np.sum(precipitation_mm * areas_m2) / 1000.0)
        land_precipitation_volume_m3 = float(
            np.sum(precipitation_mm[land] * areas_m2[land]) / 1000.0
        )
        ocean_precipitation_volume_m3 = float(
            np.sum(precipitation_mm[ocean] * areas_m2[ocean]) / 1000.0
        )
        evapotranspiration_volume_m3 = float(
            np.sum(np.maximum(np.asarray(et_mm)[land], 0.0) * areas_m2[land]) / 1000.0
        )
        water_balance_residual_m3 = (
            land_precipitation_volume_m3
            - evapotranspiration_volume_m3
            - external_runoff_volume_m3
        )

        return SeasonResult(
            season = float(season),
            temperature = temperature.copy(),
            moisture = np.asarray(self.atmosphere.moisture, dtype = np.float64).copy(),
            condensed_moisture = condensed,
            precipitation_mm = precipitation_mm,
            flow_to = np.asarray(flow_to, dtype = np.int64).copy(),
            flow_accumulation = np.asarray(flow_accumulation, dtype = np.float64).copy(),
            evapotranspiration_mm = et_mm.copy(),
            runoff_mm = runoff_mm.copy(),
            available_water_mm = available_mm.copy(),
            precipitation_volume_m3 = precipitation_volume_m3,
            evapotranspiration_volume_m3 = evapotranspiration_volume_m3,
            runoff_volume_m3 = external_runoff_volume_m3,
            total_area_m2 = total_area_m2,
            land_area_m2 = land_area_m2,
            ocean_area_m2 = ocean_area_m2,
            temperature_area_weighted_mean_c = self._weighted_mean(temperature, areas_m2),
            land_precipitation_area_weighted_mean_mm = self._weighted_mean(precipitation_mm, areas_m2, land),
            land_evapotranspiration_area_weighted_mean_mm = self._weighted_mean(et_mm, areas_m2, land),
            land_precipitation_volume_m3 = land_precipitation_volume_m3,
            ocean_precipitation_volume_m3 = ocean_precipitation_volume_m3,
            water_balance_residual_m3 = water_balance_residual_m3,
        )

    @staticmethod
    def print_diagnostics(result):
        """Imprime áreas, médias ponderadas e volumes de uma estação."""
        def volume(value):
            return f"{value:.3e} m³"

        total_km2 = result.total_area_m2 / 1_000_000.0
        land_km2 = result.land_area_m2 / 1_000_000.0
        ocean_km2 = result.ocean_area_m2 / 1_000_000.0
        land_pct = result.land_area_m2 / max(result.total_area_m2, 1.0) * 100.0
        ocean_pct = result.ocean_area_m2 / max(result.total_area_m2, 1.0) * 100.0

        print(f"\\nEstação: {result.season:.2f}")
        print(f"Área total: {total_km2:,.2f} km²")
        print(f"Terra: {land_km2:,.2f} km² ({land_pct:.2f}%)")
        print(f"Oceano: {ocean_km2:,.2f} km² ({ocean_pct:.2f}%)")
        print(f"Temperatura média ponderada por área: {result.temperature_area_weighted_mean_c:.2f} °C")
        print(f"Precipitação média em terra: {result.land_precipitation_area_weighted_mean_mm:.2f} mm")
        print(f"Evapotranspiração média em terra: {result.land_evapotranspiration_area_weighted_mean_mm:.2f} mm")
        print(f"Precipitação total: {volume(result.precipitation_volume_m3)}")
        print(f"  Precipitação em terra: {volume(result.land_precipitation_volume_m3)}")
        print(f"  Precipitação no oceano: {volume(result.ocean_precipitation_volume_m3)}")
        print(f"Evapotranspiração: {volume(result.evapotranspiration_volume_m3)}")
        print(f"Escoamento externo da terra: {volume(result.runoff_volume_m3)}")
        print(f"Resíduo P-terra - ET - escoamento: {volume(result.water_balance_residual_m3)}")

    def simulate_year(self, planet, seasons = 4):
        seasons = int(seasons)
        if seasons < 1:
            raise ValueError("O número de estações deve ser positivo.")

        return [
            self.simulate_season(planet, season = index / seasons)
            for index in range(seasons)
        ]

if __name__ == "__main__":
    from eden.geology.terrain import TerrainGenerator
    from eden.planet_profile import generate_profile
    from eden.planet.planet import Planet

    seed = 4
    profile = generate_profile(seed)
    planet = Planet(profile, subdivision = 3)

    TerrainGenerator(seed).generate(planet)

    simulation = PlanetSimulation()
    results = simulation.simulate_year(planet, seasons = 4)

    for result in results:
        simulation.print_diagnostics(result)
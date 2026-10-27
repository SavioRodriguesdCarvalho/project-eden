from dataclasses import dataclass
import numpy as np

MIN_RADIUS_KM = 1737.4
EARTH_RADIUS_KM = 6371.0
MAX_RADIUS_KM = 17300.0

@dataclass(frozen = True)
class PlanetProfile:
    seed: int
    mass_earth: float
    radius_km: float
    rotation_period: float
    axial_tilt: float
    stellar_flux_earth: float
    atmospheric_pressure_atm: float
    greenhouse_strength: float
    water_abundance: float
    density_earth: float

    @property
    def gravity_earth(self):
        radius_ratio = self.radius_km / EARTH_RADIUS_KM

        return self.mass_earth / radius_ratio ** 2

def generate_profile(seed):
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1001]))
    
    radius_km = float(rng.uniform(MIN_RADIUS_KM, MAX_RADIUS_KM))
    density = float(rng.uniform(0.5, 1.5))
    radius_ratio = radius_km / EARTH_RADIUS_KM
    mass_earth = density * radius_ratio ** 3

    return PlanetProfile(
        seed = seed,
        mass_earth = mass_earth,
        radius_km = radius_km,
        rotation_period = float(rng.uniform(10.0, 100.0)),
        axial_tilt = float(rng.uniform(0.0, 60.0)),
        stellar_flux_earth = float(rng.uniform(0.4, 1.6)),
        atmospheric_pressure_atm = float(10 ** rng.uniform(-1.0, 0.7)),
        greenhouse_strength = float(rng.uniform(0.0, 1.0)),
        water_abundance = float(rng.uniform(0.0, 1.0)),
        density_earth = density
    )
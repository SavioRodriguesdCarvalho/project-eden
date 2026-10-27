import numpy as np

from eden.planet_profile import generate_profile
from eden.planet.planet import Planet
from eden.state import PlanetState
from eden.soil import Soil

planet = Planet(generate_profile(42), subdivision = 3)
state = PlanetState.initialize(planet)

state.soil_water_mm = np.array([100.0, 140.0, 0.0])

soil = Soil(water_capacity_mm = 150.0)

fluxes = soil.step(
    state,
    precipitation_mm = np.array([20.0, 20.0, 10.0]),
    potential_evapotranspiration_mm = np.array([10.0, 30.0, 20.0]),
)

print("Água no solo:", state.soil_water_mm)
print("Evapotranspiração real:", fluxes.actual_evapotranspiration_mm)
print("Drenagem:", fluxes.drainage_mm)
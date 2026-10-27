import numpy as np
import pytest

from eden.geology.terrain import TerrainGenerator
from eden.planet_profile import generate_profile
from eden.hydrology.hydrology import Hydrology
from eden.climate.climate import Climate
from eden.planet.planet import Planet

def test_same_seed_generates_same_profile():
    profile_a = generate_profile(42)
    profile_b = generate_profile(42)

    assert profile_a == profile_b


def test_different_seeds_generate_different_profiles():
    profile_a = generate_profile(42)
    profile_b = generate_profile(43)

    assert profile_a != profile_b


def test_terrain_generation_produces_finite_elevations():
    seed = 42
    profile = generate_profile(seed)
    planet = Planet(profile, subdivision=2)

    TerrainGenerator(seed=seed).generate(planet)

    elevation = planet.surface.elevation

    assert len(elevation) > 0
    assert np.all(np.isfinite(elevation))


def test_water_depth_is_nonnegative():
    seed = 42
    planet = Planet(generate_profile(seed), subdivision=2)

    TerrainGenerator(seed=seed).generate(planet)

    assert np.all(planet.surface.water_depth >= 0)
    assert np.isfinite(planet.surface.water_volume)


def test_zero_water_abundance_produces_no_water():
    seed = 42
    planet = Planet(generate_profile(seed), subdivision=2)

    TerrainGenerator(seed=seed).generate(planet)
    planet.surface.set_water_abundance(0.0)

    assert np.all(planet.surface.water_depth == 0)
    assert planet.surface.water_volume == 0.0


def test_more_water_does_not_lower_sea_level():
    seed = 42
    planet = Planet(generate_profile(seed), subdivision=2)

    TerrainGenerator(seed=seed).generate(planet)

    planet.surface.set_water_abundance(0.2)
    low_water_sea_level = planet.surface.sea_level

    planet.surface.set_water_abundance(0.8)
    high_water_sea_level = planet.surface.sea_level

    assert high_water_sea_level >= low_water_sea_level

def test_climate_produces_finite_temperatures():
    seed = 42
    planet = Planet(generate_profile(seed), subdivision=2)

    TerrainGenerator(seed=seed).generate(planet)
    climate = Climate()

    temperature = climate.calculate_temperature(planet)

    assert len(temperature) == len(planet.surface.elevation)
    assert np.all(np.isfinite(temperature))


def test_season_changes_temperature_distribution():
    seed = 42
    planet = Planet(generate_profile(seed), subdivision=2)

    TerrainGenerator(seed=seed).generate(planet)
    climate = Climate()

    summer = climate.calculate_temperature(planet, season=0.25).copy()
    winter = climate.calculate_temperature(planet, season=0.75).copy()

    assert not np.allclose(summer, winter)

def test_hydrology_calculates_flow_for_all_land_faces():
    profile = generate_profile(seed=42)
    planet = Planet(profile, subdivision=2)
    TerrainGenerator(seed=42).generate(planet)

    hydrology = Hydrology()
    flow_to = hydrology.calculate_flow(planet)

    land_faces = np.where(planet.surface.water_depth == 0)[0]

    assert len(flow_to) == len(planet.surface.elevation)
    assert np.all(np.isfinite(hydrology.filled_elevation))
    assert len(hydrology.flow_order) > 0

def test_hydrology_calculates_flow():
    profile = generate_profile(seed=42)
    planet = Planet(profile, subdivision=2)
    TerrainGenerator(seed=42).generate(planet)

    hydrology = Hydrology()
    flow_to = hydrology.calculate_flow(planet)

    assert len(flow_to) == len(planet.surface.elevation)
    assert len(hydrology.flow_order) == len(flow_to)
    assert np.all(np.isfinite(hydrology.filled_elevation))
    assert np.all(flow_to >= -1)
    assert np.all(flow_to < len(flow_to))


def test_flow_accumulation_is_nonnegative():
    profile = generate_profile(seed=42)
    planet = Planet(profile, subdivision=2)
    TerrainGenerator(seed=42).generate(planet)

    hydrology = Hydrology()
    flow_to = hydrology.calculate_flow(planet)
    accumulation = hydrology.calculate_flow_accumulation(planet, flow_to)

    assert len(accumulation) == len(planet.surface.elevation)
    assert np.all(np.isfinite(accumulation))
    assert np.all(accumulation >= 0)


def test_evapotranspiration_respects_water_availability():
    profile = generate_profile(seed=42)
    planet = Planet(profile, subdivision=2)
    TerrainGenerator(seed=42).generate(planet)

    hydrology = Hydrology()
    flow_to = hydrology.calculate_flow(planet)
    face_count = len(planet.surface.elevation)

    precipitation = np.full(face_count, 100.0)
    demand = np.full(face_count, 40.0)

    result = hydrology.route_evapotranspiration(
        planet,
        flow_to,
        precipitation,
        demand
    )

    assert np.all(np.isfinite(result["et_mm"]))
    assert np.all(np.isfinite(result["outflow_mm"]))
    assert np.all(result["et_mm"] >= 0)
    assert np.all(result["outflow_mm"] >= 0)
    assert np.all(result["et_mm"] <= demand)

def test_evapotranspiration_respects_water_balance():
    profile = generate_profile(seed=42)
    planet = Planet(profile, subdivision=2)
    TerrainGenerator(seed=42).generate(planet)

    hydrology = Hydrology()
    flow_to = hydrology.calculate_flow(planet)
    face_count = len(planet.surface.elevation)

    precipitation = np.full(face_count, 100.0)
    demand = np.full(face_count, 40.0)

    result = hydrology.route_evapotranspiration(
        planet,
        flow_to,
        precipitation,
        demand
    )

    land = planet.surface.elevation >= planet.surface.sea_level
    downstream = flow_to

    outlets = land & (
        (downstream < 0)
        | (planet.surface.elevation[np.maximum(downstream, 0)] < planet.surface.sea_level)
    )

    total_precipitation = precipitation[land].sum()
    total_evapotranspiration = result["et_mm"][land].sum()
    total_outflow = result["outflow_mm"][outlets].sum()

    assert np.isclose(
        total_precipitation,
        total_evapotranspiration + total_outflow
    )

def test_evapotranspiration_conserves_volume_with_different_areas():
    profile = generate_profile(seed=42)
    planet = Planet(profile, subdivision=2)
    TerrainGenerator(seed=42).generate(planet)

    hydrology = Hydrology()
    flow_to = hydrology.calculate_flow(planet)

    face_count = len(planet.surface.elevation)
    precipitation = np.full(face_count, 100.0)
    demand = np.full(face_count, 40.0)
    face_area = np.linspace(1.0, 10.0, face_count)

    result = hydrology.route_evapotranspiration(
        planet,
        flow_to,
        precipitation,
        demand,
        face_area=face_area
    )

    elevation = planet.surface.elevation
    land = elevation >= planet.surface.sea_level

    total_precipitation = np.sum(
        precipitation[land] * face_area[land]
    )

    total_evapotranspiration = np.sum(
        result["et_mm"][land] * face_area[land]
    )

    downstream = np.maximum(flow_to, 0)
    downstream_is_ocean = elevation[downstream] < planet.surface.sea_level

    outlets = land & ((flow_to < 0) | downstream_is_ocean)

    total_outflow = np.sum(
        result["outflow_mm"][outlets] * face_area[outlets]
    )

    assert np.isclose(
        total_precipitation,
        total_evapotranspiration + total_outflow
    )


def test_evapotranspiration_rejects_invalid_precipitation_size():
    planet = Planet(generate_profile(seed=42), subdivision=2)
    TerrainGenerator(seed=42).generate(planet)

    hydrology = Hydrology()
    flow_to = hydrology.calculate_flow(planet)
    face_count = len(planet.surface.elevation)

    with pytest.raises(ValueError):
        hydrology.route_evapotranspiration(
            planet,
            flow_to,
            np.zeros(face_count - 1),
            np.zeros(face_count),
        )


@pytest.mark.parametrize("fraction", [0.0, -0.1, 1.1, 2.0])
def test_evapotranspiration_rejects_invalid_evaporation_fraction(fraction):
    planet = Planet(generate_profile(seed=42), subdivision=2)
    TerrainGenerator(seed=42).generate(planet)

    hydrology = Hydrology()
    flow_to = hydrology.calculate_flow(planet)
    face_count = len(planet.surface.elevation)

    with pytest.raises(ValueError):
        hydrology.route_evapotranspiration(
            planet,
            flow_to,
            np.ones(face_count),
            np.ones(face_count),
            evaporation_fraction=fraction,
        )


@pytest.mark.parametrize(
    "invalid_area",
    [
        0.0,
        -1.0,
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_evapotranspiration_rejects_invalid_face_areas(invalid_area):
    planet = Planet(generate_profile(seed=42), subdivision=2)
    TerrainGenerator(seed=42).generate(planet)

    hydrology = Hydrology()
    flow_to = hydrology.calculate_flow(planet)
    face_count = len(planet.surface.elevation)

    areas = np.ones(face_count)
    areas[0] = invalid_area

    with pytest.raises(ValueError):
        hydrology.route_evapotranspiration(
            planet,
            flow_to,
            np.ones(face_count),
            np.ones(face_count),
            face_area=areas,
        )


def test_flow_destinations_are_valid():
    planet = Planet(generate_profile(seed=42), subdivision=2)
    TerrainGenerator(seed=42).generate(planet)

    hydrology = Hydrology()
    flow_to = hydrology.calculate_flow(planet)
    face_count = len(planet.surface.elevation)

    assert flow_to.shape == (face_count,)
    assert np.all(flow_to >= -1)
    assert np.all(flow_to < face_count)


def test_land_flow_does_not_form_cycles():
    planet = Planet(generate_profile(seed=42), subdivision=2)
    TerrainGenerator(seed=42).generate(planet)

    hydrology = Hydrology()
    flow_to = hydrology.calculate_flow(planet)
    elevation = planet.surface.elevation
    sea_level = planet.surface.sea_level
    land = elevation >= sea_level

    for start in np.flatnonzero(land):
        visited = set()
        current = int(start)

        while current >= 0 and land[current]:
            assert current not in visited, "Ciclo encontrado no fluxo hidrológico."
            visited.add(current)
            current = int(flow_to[current])


def test_land_flow_reaches_ocean_or_valid_outlet():
    planet = Planet(generate_profile(seed=42), subdivision=2)
    TerrainGenerator(seed=42).generate(planet)

    hydrology = Hydrology()
    flow_to = hydrology.calculate_flow(planet)
    elevation = planet.surface.elevation
    sea_level = planet.surface.sea_level
    land = elevation >= sea_level

    for start in np.flatnonzero(land):
        current = int(start)
        visited = set()

        while current >= 0 and land[current]:
            assert current not in visited, "O fluxo entrou em um ciclo."
            visited.add(current)
            current = int(flow_to[current])

        assert current == -1 or not land[current], (
            "O fluxo terminou em uma face terrestre sem saída válida."
        )


def test_flow_moves_down_filled_elevation():
    planet = Planet(generate_profile(seed=42), subdivision=2)
    TerrainGenerator(seed=42).generate(planet)

    hydrology = Hydrology()
    flow_to = hydrology.calculate_flow(planet)
    filled = hydrology.filled_elevation
    land = planet.surface.elevation >= planet.surface.sea_level

    for face_id in np.flatnonzero(land):
        downstream = int(flow_to[face_id])

        if downstream >= 0:
            assert filled[downstream] <= filled[face_id], (
                "O fluxo deve seguir para uma face de elevação preenchida "
                "igual ou inferior."
            )


def test_flow_accumulation_does_not_decrease_downstream():
    planet = Planet(generate_profile(seed=42), subdivision=2)
    TerrainGenerator(seed=42).generate(planet)

    hydrology = Hydrology()
    flow_to = hydrology.calculate_flow(planet)
    accumulation = hydrology.calculate_flow_accumulation(planet, flow_to)
    land = planet.surface.elevation >= planet.surface.sea_level

    for face_id in np.flatnonzero(land):
        downstream = int(flow_to[face_id])

        if downstream >= 0 and land[downstream]:
            assert accumulation[downstream] >= accumulation[face_id], (
                "A acumulação não deve diminuir ao longo do fluxo terrestre."
            )

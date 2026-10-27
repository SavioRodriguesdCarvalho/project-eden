import numpy as np

class Atmosphere:
    def __init__(self):
        self.latitude = None
        self.wind_vectors = None
        self.wind_speed = None
        self.moisture = None
        self.condensed_moisture = None
        self.temperature = None
        self.saturation_capacity_field = None
        self.orographic_lift = None
        self.convective_lift = None
        self.evaporation_field = None
        self.applied_evaporation_field = None
        self.land_evaporation_demand = None
        self.wind_season = None
        self.total_evaporated = 0.0
        self.total_ocean_evaporated = 0.0
        self.total_condensed = 0.0
        self.moisture_balance_error = 0.0
        self.saturation_overage = 0.0

    @staticmethod
    def _smoothstep(edge0, edge1, values):
        t = np.clip((values - edge0) / (edge1 - edge0), 0.0, 1.0)

        return t * t * (3.0 - 2.0 * t)

    @staticmethod
    def saturation_vapor_pressure(temperature_c):
        temperature_c = np.clip(np.asarray(temperature_c, dtype = np.float64), -80.0, 60.0)

        return 0.6108 * np.exp(17.27 * temperature_c / (temperature_c + 237.3))

    def calculate_wind(self, planet, season = 0.0):
        if not 0.0 <= season < 1.0:
            raise ValueError("A estação deve estar no intervalo [0, 1).")

        vertices = np.asarray(planet.icosphere.vertices, dtype = np.float64)
        faces = np.asarray(planet.icosphere.faces, dtype = np.int64)
        centers = vertices[faces].mean(axis = 1)
        lengths = np.linalg.norm(centers, axis = 1, keepdims = True)

        if np.any(lengths <= 1e-12):
            raise ValueError("Não foi possível normalizar o centro de uma face.")

        centers /= lengths

        self.latitude = np.degrees(np.arcsin(np.clip(centers[:, 2], -1.0, 1.0)))

        x = centers[:, 0]
        y = centers[:, 1]
        horizontal_norm = np.hypot(x, y)

        east = np.zeros_like(centers)
        non_polar = horizontal_norm > 1e-8
        east[non_polar, 0] = -y[non_polar] / horizontal_norm[non_polar]
        east[non_polar, 1] = x[non_polar] / horizontal_norm[non_polar]
        east[~non_polar, 1] = 1.0

        north = np.cross(centers, east)
        abs_latitude = np.abs(self.latitude)

        tropical_transition = self._smoothstep(25.0, 35.0, abs_latitude)
        polar_transition = self._smoothstep(55.0, 65.0, abs_latitude)
        polar_zonal_transition = self._smoothstep(53.0, 59.0, abs_latitude)
        zonal_direction = (-1.0 + 2.0 * tropical_transition - 2.0 * polar_zonal_transition)

        thermal_equator = 18.0 * np.sin(2.0 * np.pi * season)

        tropical_strength = 0.35 * (1.0 - tropical_transition)
        midlatitude_strength = 0.15 * tropical_transition * (1.0 - polar_transition)
        polar_strength = 0.15 * polar_transition

        meridional_direction = (
            np.sign(self.latitude) * (midlatitude_strength - polar_strength)
            - np.sign(self.latitude - thermal_equator) * tropical_strength
        )

        direction = (east * zonal_direction[:, None] + north * meridional_direction[:, None])
        direction_norm = np.linalg.norm(direction, axis = 1, keepdims = True)
        stationary = direction_norm[:, 0] <= 1e-8
        direction[stationary] = east[stationary]
        direction_norm = np.linalg.norm(direction, axis = 1, keepdims = True)
        direction /= np.maximum(direction_norm, 1e-8)

        self.wind_speed = (
            6.0
            + 4.0 * tropical_transition
            - 3.0 * polar_transition
        )
        self.wind_vectors = direction * self.wind_speed[:, None]

        return self.wind_vectors

    def _prepare_transport(self, centers, neighbors, wind_direction, elevation, transport_diffusion):
        sources = []
        destinations = []
        for source, adjacent_faces in enumerate(neighbors):
            for destination in adjacent_faces:
                sources.append(source)
                destinations.append(int(destination))

        sources = np.asarray(sources, dtype = np.int64)
        destinations = np.asarray(destinations, dtype = np.int64)
        destination_centers = centers[destinations]
        source_centers = centers[sources]
        directions = destination_centers - (
            np.sum(destination_centers * source_centers, axis = 1)[:, None]
            * source_centers
        )
        direction_lengths = np.linalg.norm(directions, axis = 1, keepdims = True)
        directions /= np.maximum(direction_lengths, 1e-8)

        alignment = np.sum(directions * wind_direction[sources], axis = 1)
        weights = np.maximum(alignment, 0.0)
        weight_sums = np.bincount(sources, weights = weights, minlength = len(centers))
        wind_sources = weight_sums > 1e-12
        weights /= np.maximum(weight_sums[sources], 1e-12)
        wind_weights = weights.copy()
        degree = np.bincount(sources, minlength = len(centers))
        isotropic_weights = 1.0 / np.maximum(degree[sources], 1)

        if transport_diffusion > 0.0:
            weights = ((1.0 - transport_diffusion) * weights + transport_diffusion * isotropic_weights)

            no_wind_edge = ~wind_sources[sources]
            weights[no_wind_edge] = isotropic_weights[no_wind_edge]
            wind_weights[no_wind_edge] = isotropic_weights[no_wind_edge]

        valid_sources = (wind_sources | (transport_diffusion > 0.0)) & (degree > 0)

        rise = np.maximum(elevation[destinations] - elevation[sources], 0.0)
        uplift = np.bincount(sources, weights = wind_weights * rise, minlength = len(centers))
        uplift = np.clip(uplift, 0.0, 1800.0)

        return sources, destinations, weights, valid_sources, uplift

    def calculate_moisture(
        self,
        planet,
        temperature = None,
        iterations = 80,
        evaporation_rate = 0.6,
        transport_rate = 0.65,
        transport_diffusion = 0.15,
        saturation_capacity = 2.5,
        lapse_rate = 6.5,
        season = 0.0,
        convection_strength = 6.0,
        ocean_stability_factor = 2.0,
        relative_humidity = 0.8,
        washout_rate = 0.015,
        ocean_washout_factor = 0.5,
        land_evaporation_factor = 0.55,
        land_evaporation_cap = None,
        initial_moisture = None
    ):
        if iterations < 1:
            raise ValueError("O número de iterações deve ser positivo.")
        if not 0.0 <= transport_rate <= 1.0:
            raise ValueError("A taxa de transporte deve estar entre 0 e 1.")
        if not 0.0 <= transport_diffusion <= 1.0:
            raise ValueError("A difusão do transporte deve estar entre 0 e 1.")
        if evaporation_rate < 0.0:
            raise ValueError("A taxa de evaporação não pode ser negativa.")
        if saturation_capacity <= 0.0:
            raise ValueError("A capacidade de saturação deve ser positiva.")
        if lapse_rate < 0.0:
            raise ValueError("A taxa de resfriamento não pode ser negativa.")
        if not 0.0 <= season < 1.0:
            raise ValueError("A estação deve estar no intervalo [0, 1].")
        if convection_strength < 0.0:
            raise ValueError("A intensidade convectiva não pode ser negativa.")
        if ocean_stability_factor < 1.0:
            raise ValueError("O fator de estabilidade oceânica deve ser >= 1.")
        if not 0.0 < relative_humidity <= 1.0:
            raise ValueError("A umidade relativa de condensação deve estar em (0, 1].")
        if not 0.0 <= washout_rate <= 1.0:
            raise ValueError("A taxa de arraste deve estar entre 0 e 1.")
        if ocean_washout_factor < 0.0:
            raise ValueError("O fator de arraste oceânico não pode ser negativo.")
        if not 0.0 <= land_evaporation_factor <= 1.0:
            raise ValueError("O fator de evapotranspiração terrestre deve estar entre 0 e 1.")

        vertices = np.asarray(planet.icosphere.vertices, dtype = np.float64)
        faces = np.asarray(planet.icosphere.faces, dtype = np.int64)
        elevation = np.asarray(planet.surface.elevation, dtype = np.float64)
        sea_level = float(planet.surface.sea_level)

        if len(elevation) != len(faces):
            raise ValueError("A elevação deve conter um valor por face.")

        centers = vertices[faces].mean(axis = 1)
        centers /= np.maximum(np.linalg.norm(centers, axis = 1, keepdims = True), 1e-12)
        face_count = len(faces)
        ocean = elevation < sea_level
        neighbors = planet.icosphere.get_face_neighbors()

        if (
            self.wind_vectors is None
            or len(self.wind_vectors) != face_count
            or self.wind_season != float(season)
        ):
            self.calculate_wind(planet, season = season)

        wind_norm = np.linalg.norm(self.wind_vectors, axis = 1, keepdims = True)
        wind_direction = self.wind_vectors / np.maximum(wind_norm, 1e-8)

        if temperature is None:
            latitude_fraction = np.abs(self.latitude) / 90.0
            temperature = 28.0 - 45.0 * latitude_fraction ** 1.3
            temperature -= (np.maximum(elevation - sea_level, 0.0) / 1000.0) * lapse_rate

        temperature = np.asarray(temperature, dtype = np.float64)
        if temperature.ndim != 1 or len(temperature) != face_count:
            raise ValueError("A temperatura deve conter um valor por face.")
        if not np.all(np.isfinite(temperature)):
            raise ValueError("A temperatura contém valores não finitos.")

        sources, destinations, weights, valid_sources, uplift = (
            self._prepare_transport(centers, neighbors, wind_direction, elevation, transport_diffusion)
        )

        self.orographic_lift = uplift

        reference_pressure = float(self.saturation_vapor_pressure(25.0))
        vapor_pressure = self.saturation_vapor_pressure(temperature)
        thermal_evaporation = np.clip(np.sqrt(vapor_pressure / reference_pressure), 0.08, 1.35)
        cosine_latitude = np.maximum(np.cos(np.deg2rad(self.latitude)), 0.0)
        solar_factor = 0.2 + 0.8 * np.sqrt(cosine_latitude)
        self.evaporation_field = evaporation_rate * thermal_evaporation * solar_factor

        season_shift = 18.0 * np.sin(2.0 * np.pi * season)
        latitude_distance = np.abs(self.latitude - season_shift)
        convective_profile = np.exp(-((latitude_distance / 8.0) ** 2))
        convective_lift_m = (1000.0 * convection_strength / max(lapse_rate, 1e-8) * convective_profile)
        storm_track_lift_m = 550.0 * np.exp(-((np.abs(self.latitude) - 50.0) / 13.0) ** 2)
        self.convective_lift = convective_lift_m + storm_track_lift_m
        subtropical_subsidence = 2.5 * np.exp(-(((np.abs(self.latitude) - 27.0) / 9.0) ** 2.0))
        effective_temperature = (temperature - ((uplift + self.convective_lift) / 1000.0) * lapse_rate) + subtropical_subsidence
        effective_pressure = self.saturation_vapor_pressure(effective_temperature)
        saturation = saturation_capacity * effective_pressure / reference_pressure
        saturation = np.clip(saturation, 0.2, saturation_capacity * 2.5)
        saturation[ocean] *= ocean_stability_factor
        self.temperature = temperature.copy()
        self.saturation_capacity_field = saturation

        if initial_moisture is None:
            initial_moisture_field = np.zeros(face_count, dtype = np.float64)
        else:
            initial_moisture_field = np.asarray(initial_moisture, dtype = np.float64)
            if initial_moisture_field.ndim != 1 or len(initial_moisture_field) != face_count:
                raise ValueError("A umidade inicial deve conter um valor por face.")
            if np.any(~np.isfinite(initial_moisture_field)) or np.any(initial_moisture_field < 0.0):
                raise ValueError("A umidade inicial contém valores inválidos.")
        self.moisture = initial_moisture_field.copy()
        self.condensed_moisture = np.zeros(face_count, dtype = np.float64)
        self.total_evaporated = 0.0
        self.total_ocean_evaporated = 0.0
        self.total_condensed = 0.0
        self.moisture_balance_error = 0.0
        self.saturation_overage = 0.0

        land_evaporation_demand = self.evaporation_field * land_evaporation_factor
        self.land_evaporation_demand = land_evaporation_demand
        land_evaporation = land_evaporation_demand
        if land_evaporation_cap is not None:
            cap = np.asarray(land_evaporation_cap, dtype = np.float64)
            if cap.ndim != 1 or len(cap) != face_count:
                raise ValueError("O limite de evapotranspiração deve conter um valor por face.")
            if np.any(np.isnan(cap)) or np.any(cap < 0.0):
                raise ValueError("O limite de evapotranspiração contém valores inválidos.")
            land_evaporation = np.minimum(land_evaporation, cap)

        self.applied_evaporation_field = np.where(
            ocean,
            self.evaporation_field,
            land_evaporation,
        )

        for _ in range(iterations):
            added = self.applied_evaporation_field
            self.moisture += added
            self.total_evaporated += float(added.sum())
            self.total_ocean_evaporated += float(added[ocean].sum())

            transfer_fraction = np.where(valid_sources, transport_rate, 0.0)
            transferred_by_face = self.moisture * transfer_fraction
            change = -transferred_by_face
            edge_transfers = transferred_by_face[sources] * weights
            np.add.at(change, destinations, edge_transfers)
            self.moisture = np.maximum(self.moisture + change, 0.0)

            excess = np.maximum(self.moisture - relative_humidity * saturation, 0.0)
            washout = washout_rate * np.where(ocean, ocean_washout_factor, 1.0) * self.moisture
            condensed = np.minimum(excess + washout, self.moisture)
            self.saturation_overage = max(self.saturation_overage, float(np.max(excess)))

            self.moisture -= condensed
            self.condensed_moisture += condensed
            self.total_condensed += float(condensed.sum())

        self.moisture_balance_error = (
            float(initial_moisture_field.sum())
            + self.total_evaporated
            - float(self.moisture.sum())
            - float(self.condensed_moisture.sum())
        )

        return self.moisture
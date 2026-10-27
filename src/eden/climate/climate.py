import numpy as np

class Climate:
    def __init__(
        self,
        lapse_rate = 6.5,
        equatorial_temperature = 36.5,
        latitudinal_range = 45.0,
        land_seasonal_response = 0.8,
        ocean_seasonal_response = 0.25,
    ):
        self.lapse_rate = float(lapse_rate)
        self.equatorial_temperature = float(equatorial_temperature)
        self.latitudinal_range = float(latitudinal_range)
        self.land_seasonal_response = float(land_seasonal_response)
        self.ocean_seasonal_response = float(ocean_seasonal_response)
        self.temperature = None
        self.insolation = None

    @staticmethod
    def _calculate_daily_insolation(latitude, axial_tilt, season):
        solar_declination = np.arcsin(np.sin(axial_tilt) * np.sin(2.0 * np.pi * season))
        hour_angle_argument = -np.tan(latitude) * np.tan(solar_declination)
        hour_angle_argument = np.clip(hour_angle_argument, -1.0, 1.0)
        sunset_angle = np.arccos(hour_angle_argument)

        daily_insolation = (
            sunset_angle * np.sin(latitude) * np.sin(solar_declination)
            + np.cos(latitude)
            * np.cos(solar_declination)
            * np.sin(sunset_angle)
        ) / np.pi

        polar_day = (latitude * np.sign(solar_declination) > (np.pi / 2.0 - np.abs(solar_declination)))
        polar_night = (latitude * np.sign(solar_declination) < -(np.pi / 2.0 - np.abs(solar_declination)))
        daily_insolation = np.where(polar_day, np.maximum(np.sin(latitude) * np.sin(solar_declination), 0.0), daily_insolation,)
        daily_insolation = np.where(polar_night, 0.0, daily_insolation)

        return np.maximum(daily_insolation, 0.0)

    @staticmethod
    def _temperature_from_insolation(relative_insolation, equatorial_temperature, latitudinal_range):
        latitude_effect = latitudinal_range * (1.0 - np.sqrt(np.clip(relative_insolation, 0.0, 1.0)))

        return equatorial_temperature - latitude_effect

    @staticmethod
    def _stellar_temperature_adjustment(stellar_flux):
        if stellar_flux <= 0.0:
            raise ValueError("O fluxo estelar deve ser positivo.")

        return 278.5 * (stellar_flux ** 0.25 - 1.0)

    @staticmethod
    def _greenhouse_temperature_adjustment(pressure, greenhouse_strength):
        if pressure <= 0.0:
            raise ValueError("A pressão atmosférica deve ser positiva.")

        if not 0.0 <= greenhouse_strength <= 1.0:
            raise ValueError("A intensidade do efeito estufa deve estar entre 0 e 1.")

        pressure_factor = pressure / (1.0 + pressure)

        return 33.0 * greenhouse_strength * pressure_factor

    def calculate_temperature(self, planet, season = 0.0):
        if not 0.0 <= season < 1.0:
            raise ValueError("A estação deve estar no intervalo [0, 1).")

        vertices = np.asarray(planet.icosphere.vertices, dtype = np.float64)
        faces = np.asarray(planet.icosphere.faces, dtype = np.int64)
        elevation = np.asarray(planet.surface.elevation, dtype = np.float64)
        sea_level = float(planet.surface.sea_level)
        if len(elevation) != len(faces):
            raise ValueError("A elevação deve conter um valor por face.")

        face_centers = vertices[faces].mean(axis = 1)
        lengths = np.linalg.norm(face_centers, axis = 1, keepdims = True)
        if np.any(lengths <= 1e-12):
            raise ValueError("O centro de uma face não pode ter comprimento zero.")
        face_centers /= lengths

        latitude = np.arcsin(np.clip(face_centers[:, 2], -1.0, 1.0))
        axial_tilt = np.deg2rad(float(planet.axial))
        seasonal_insolation = self._calculate_daily_insolation(latitude, axial_tilt, season)
        seasonal_relative = np.clip(seasonal_insolation / 0.5, 0.0, 1.0)

        annual_insolation = np.zeros_like(latitude)

        for reference_season in (0.0, 0.25, 0.5, 0.75):
            annual_insolation += self._calculate_daily_insolation(latitude, axial_tilt, reference_season)

        annual_insolation /= 4.0
        annual_relative = np.clip(annual_insolation / 0.5, 0.0, 1.0)

        stellar_flux = float(planet.profile.stellar_flux_earth)
        stellar_adjustment = self._stellar_temperature_adjustment(stellar_flux)

        pressure = float(planet.profile.atmospheric_pressure_atm)
        greenhouse_strength = float(planet.profile.greenhouse_strength)
        greenhouse_adjustment = self._greenhouse_temperature_adjustment(pressure, greenhouse_strength)

        equatorial_temperature = (self.equatorial_temperature + stellar_adjustment + greenhouse_adjustment)

        annual_temperature = self._temperature_from_insolation(annual_relative, equatorial_temperature, self.latitudinal_range)
        seasonal_temperature = self._temperature_from_insolation(seasonal_relative, equatorial_temperature, self.latitudinal_range)

        land = elevation >= sea_level
        seasonal_response = np.where(land, self.land_seasonal_response, self.ocean_seasonal_response)
        altitude_above_sea_level = np.maximum(elevation - sea_level, 0.0)
        altitude_effect = (altitude_above_sea_level / 1000.0) * self.lapse_rate

        self.insolation = seasonal_insolation
        self.temperature = (
            annual_temperature
            + seasonal_response * (seasonal_temperature - annual_temperature)
            - altitude_effect
        )

        return self.temperature
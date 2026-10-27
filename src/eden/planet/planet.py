from .icosphere import Icosphere
from .topology import Topology
from .surface import Surface


class Planet:
    def __init__(self, profile, subdivision=0):
        self.profile = profile

        self.radius = profile.radius_km
        self.mass = profile.mass_earth
        self.rotation = profile.rotation_period
        self.axial = profile.axial_tilt
        self.gravity = profile.gravity_earth

        self.icosphere = Icosphere(radius = self.radius, subdivision = subdivision)
        self.topology = Topology(self.icosphere.faces)
        self.surface = Surface(self.icosphere, self.topology)
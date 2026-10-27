import numpy as np

class TerrainGenerator:
    def __init__(self, seed = None):
        self.rng = np.random.default_rng(seed)

        self._seed = int(self.rng.integers(0, 2**31 - 1))

    def generate(self, planet):
        vertices = np.asarray(planet.icosphere.vertices)
        faces = np.asarray(planet.icosphere.faces)

        positions = vertices[faces].mean(axis = 1)
        positions /= np.linalg.norm(positions, axis = 1, keepdims = True)

        center = self.rng.normal(size = 3)
        center /= np.linalg.norm(center)

        continent = self._continent(positions, center)
        regional = self._fbm(positions, frequency = 2.5, octaves = 5)
        mountains = self._ridged(positions, frequency = 4.0, octaves = 5)

        land = self._smoothstep(0.0, 0.15, continent)

        elevation = (
            np.where(continent > 0, continent * 4000.0, continent * 3000.0)
            + regional * 1200.0 * (0.3 + 0.7 * land)
            + mountains * 3500.0 * land * self._smoothstep(0.1, 0.5, continent)
        )

        planet.surface.elevation = elevation

        planet.surface.set_water_abundance(planet.profile.water_abundance)

    def _continent(self, positions, center):
        bias = positions @ center
        shape = self._fbm(positions, frequency = 1.6, octaves = 5)

        value = 0.25 * bias + 1.0 * shape
        threshold = 0.08

        return value - threshold

    def _fbm(self, positions, frequency, octaves):
        value = np.zeros(len(positions))
        amplitude = 1.0
        total = 0.0
        freq = frequency

        for octave in range(octaves):
            value += amplitude * self._value_noise(positions * freq, octave)
            total += amplitude
            freq *= 2.0
            amplitude *= 0.5

        return value / total

    def _ridged(self, positions, frequency, octaves):
        value = np.zeros(len(positions))
        amplitude = 1.0
        total = 0.0
        freq = frequency

        for octave in range(octaves):
            n = self._value_noise(positions * freq, octave)
            ridge = (1.0 - np.abs(n)) ** 2
            value += amplitude * ridge
            total += amplitude
            freq *= 2.0
            amplitude *= 0.5

        return value / total

    def _value_noise(self, points, octave):
        points = points + (octave + 1) * 1000.0

        p0 = np.floor(points).astype(np.int64)
        frac = points - p0

        fade = self._fade(frac)

        corners = {}
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    corner = p0 + np.array([dx, dy, dz])
                    corners[(dx, dy, dz)] = self._hash(corner)

        def lerp(a, b, t):
            return a + t * (b - a)

        x00 = lerp(corners[(0, 0, 0)], corners[(1, 0, 0)], fade[:, 0])
        x10 = lerp(corners[(0, 1, 0)], corners[(1, 1, 0)], fade[:, 0])
        x01 = lerp(corners[(0, 0, 1)], corners[(1, 0, 1)], fade[:, 0])
        x11 = lerp(corners[(0, 1, 1)], corners[(1, 1, 1)], fade[:, 0])

        y0 = lerp(x00, x10, fade[:, 1])
        y1 = lerp(x01, x11, fade[:, 1])

        value = lerp(y0, y1, fade[:, 2])

        return value * 2.0 - 1.0

    def _hash(self, lattice):
        x = lattice[:, 0].astype(np.uint64)
        y = lattice[:, 1].astype(np.uint64)
        z = lattice[:, 2].astype(np.uint64)
        s = np.uint64(self._seed)

        h = x * np.uint64(0x9E3779B1)
        h ^= y * np.uint64(0x85EBCA77)
        h ^= z * np.uint64(0xC2B2AE3D)
        h ^= s * np.uint64(0x27D4EB2F)
        h ^= h >> np.uint64(15)
        h *= np.uint64(0x2545F4914F6CDD1D)
        h ^= h >> np.uint64(13)

        return (h & np.uint64(0xFFFFFF)).astype(np.float64) / float(0xFFFFFF)

    @staticmethod
    def _fade(t):
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

    @staticmethod
    def _smoothstep(edge0, edge1, x):
        t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)
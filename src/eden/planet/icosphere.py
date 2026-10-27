import math

class Icosphere:
    def __init__(self, radius = 1.0, subdivision = 0):
        self.radius = radius
        self.vertices = []
        self.faces = []

        self._create_icosahedron()

        for _ in range(subdivision):
            self._subdivide()

    def _create_icosahedron(self):
        phi = (1 + math.sqrt(5)) / 2

        vertices = [
            (-1, phi, 0),
            (1, phi, 0),
            (-1, -phi, 0),
            (1, -phi, 0),

            (0, -1, phi),
            (0, 1, phi),
            (0, -1, -phi),
            (0, 1, -phi),

            (phi, 0, -1),
            (phi, 0, 1),
            (-phi, 0, -1),
            (-phi, 0, 1)
        ]

        for vertex in vertices:
            self.vertices.append(self._normalize(vertex))

        self.faces = [
            (0, 11, 5),
            (0, 5, 1),
            (0, 1, 7),
            (0, 7, 10),
            (0, 10, 11),

            (1, 5, 9),
            (5, 11, 4),
            (11, 10, 2),
            (10, 7, 6),
            (7, 1, 8),

            (3, 9, 4),
            (3, 4, 2),
            (3, 2, 6),
            (3, 6, 8),
            (3, 8, 9),

            (4, 9, 5),
            (2, 4, 11),
            (6, 2, 10),
            (8, 6, 7),
            (9, 8, 1)
        ]

    def _normalize(self, vertex):
        x, y, z = vertex

        length = math.sqrt(x**2 + y**2 + z**2)

        return (
            x / length * self.radius,
            y / length * self.radius,
            z / length * self.radius
        )

    def _subdivide(self):
        new_faces = []
        midpoint_cache = {}

        def midpoint(a, b):
            edge = tuple(sorted((a, b)))

            if edge in midpoint_cache:
                return midpoint_cache[edge]

            vertex_a = self.vertices[a]
            vertex_b = self.vertices[b]

            vertex = (
                (vertex_a[0] + vertex_b[0]) / 2,
                (vertex_a[1] + vertex_b[1]) / 2,
                (vertex_a[2] + vertex_b[2]) / 2
            )

            index = len(self.vertices)

            self.vertices.append(self._normalize(vertex))
            midpoint_cache[edge] = index

            return index

        for a, b, c in self.faces:
            ab = midpoint(a, b)
            bc = midpoint(b, c)
            ca = midpoint(c, a)

            new_faces.extend([
                (a, ab, ca),
                (b, bc, ab),
                (c, ca, bc),
                (ab, bc, ca),
            ])

        self.faces = new_faces

    def get_face_neighbors(self):
        neighbors = [set() for _ in self.faces]
        edge_faces = {}

        for face_id, face in enumerate(self.faces):
            a, b, c = face

            edges = [
                tuple(sorted((a, b))),
                tuple(sorted((b, c))),
                tuple(sorted((c, a)))
            ]

            for edge in edges:
                if edge in edge_faces:
                    other_face = edge_faces[edge]

                    neighbors[face_id].add(other_face)
                    neighbors[other_face].add(face_id)
                else:
                    edge_faces[edge] = face_id

        return neighbors

    def get_faces_areas(self):
        import numpy as np

        vertices = np.asarray(self.vertices, dtype = np.float64)
        faces = np.asarray(self.faces, dtype = np.int64)

        a = vertices[faces[:, 0]]
        b = vertices[faces[:, 1]]
        c = vertices[faces[:, 2]]

        cross_products = np.cross(b - a, c - a)

        return np.linalg.norm(cross_products, axis = 1) / 2.0
class Topology:
    def __init__(self, faces):
        self.faces = faces
        self.neighbors = []

        self._build_neighbors()

    def _build_neighbors(self):
        edge_map = {}

        for face_id, face in enumerate(self.faces):
            for i in range(3):
                a = face[i]
                b = face[(i + 1) % 3]

                edge = tuple(sorted((a, b)))

                if edge not in edge_map:
                    edge_map[edge] = []

                edge_map[edge].append(face_id)

        self.neighbors = [[] for _ in self.faces]

        for edge_faces in edge_map.values():
            if len(edge_faces) == 2:
                face_a, face_b = edge_faces

                self.neighbors[face_a].append(face_b)
                self.neighbors[face_b].append(face_a)
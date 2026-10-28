import numpy as np
import pyvista as pv

from eden.geology.terrain import TerrainGenerator
from eden.planet_profile import generate_profile
from eden.planet.planet import Planet

def generate_planet():
    seed = 10
    profile = generate_profile(seed)
    planet = Planet(profile, subdivision = 6)

    terrain = TerrainGenerator(seed = seed)
    terrain.generate(planet)

    return planet

def interpolate_colors(values, stops):
    positions = np.array([stop[0] for stop in stops])
    colors = np.array([stop[1] for stop in stops], dtype = np.float32)

    result = np.empty((len(values), 4), dtype=np.float32)

    for channel in range(4):
        result[:, channel] = np.interp(values, positions, colors[:, channel])

    return result

def prepare_mesh(planet):
    vertices = np.asarray(planet.icosphere.vertices, dtype = np.float32)
    faces = np.asarray(planet.icosphere.faces, dtype = np.int64)
    elevation = np.asarray(planet.surface.elevation, dtype = np.float32)
    sea_level = float(planet.surface.sea_level)

    vertex_elevation = np.zeros(len(vertices), dtype = np.float32)
    vertex_count = np.zeros(len(vertices), dtype = np.float32)

    for face_id, face in enumerate(faces):
        for vertex_id in face:
            vertex_elevation[vertex_id] += elevation[face_id]
            vertex_count[vertex_id] += 1

    vertex_elevation /= np.maximum(vertex_count, 1)

    directions = vertices / np.linalg.norm(
        vertices,
        axis = 1,
        keepdims = True,
    )

    relative_elevation = vertex_elevation - sea_level
    max_relief = max(float(np.max(np.abs(relative_elevation))), 1.0)

    relief_strength = 0.10
    radius_factor = 1.0 + (relative_elevation / max_relief) * relief_strength

    deformed_vertices = directions * radius_factor[:, None]

    depth = sea_level - elevation
    height = elevation - sea_level

    water_stops = [
        (0.0, [0.35, 0.70, 0.85, 1.0]),
        (500.0, [0.12, 0.43, 0.72, 1.0]),
        (2000.0, [0.055, 0.23, 0.49, 1.0]),
        (6000.0, [0.025, 0.09, 0.25, 1.0]),
    ]

    land_stops = [
        (0.0, [0.78, 0.70, 0.48, 1.0]),
        (150.0, [0.77, 0.72, 0.43, 1.0]),
        (600.0, [0.38, 0.58, 0.28, 1.0]),
        (1500.0, [0.27, 0.46, 0.23, 1.0]),
        (2500.0, [0.48, 0.43, 0.34, 1.0]),
        (4000.0, [0.66, 0.61, 0.54, 1.0]),
        (6000.0, [0.91, 0.91, 0.88, 1.0]),
    ]

    face_colors = np.empty((len(faces), 4), dtype = np.float32)
    underwater = elevation < sea_level

    if np.any(underwater):
        face_colors[underwater] = interpolate_colors(
            depth[underwater],
            water_stops
        )

    if np.any(~underwater):
        face_colors[~underwater] = interpolate_colors(
            height[~underwater],
            land_stops
        )

    return deformed_vertices, faces, face_colors

def visualize_planet(planet):
    vertices, faces, colors = prepare_mesh(planet)

    vtk_faces = np.column_stack((np.full(len(faces), 3, dtype=np.int64), faces)).ravel()

    mesh = pv.PolyData(vertices, vtk_faces)

    rgba_colors = np.clip(colors * 255, 0, 255).astype(np.uint8)

    mesh.cell_data["colors"] = rgba_colors

    plotter = pv.Plotter(window_size = (1200, 850), title = "Project Eden - Planet Viewer")

    plotter.set_background("#101820")

    plotter.add_mesh(
        mesh,
        scalars = "colors",
        rgba = True,
        show_edges = False,
        smooth_shading = True,
    )

    plotter.add_axes()
    plotter.camera_position = "iso"
    plotter.camera.zoom(1.2)

    elevation = np.asarray(planet.surface.elevation)
    sea_level = float(planet.surface.sea_level)
    land = elevation >= sea_level

    print(f"Vértices: {len(vertices)}")
    print(f"Faces: {len(faces)}")
    print(f"Raio do planeta: {planet.radius:.2f} km")
    print(f"Nível do mar: {sea_level:.2f} m")
    print(f"Elevação mínima: {elevation.min():.2f} m")
    print(f"Elevação máxima: {elevation.max():.2f} m")
    print(f"Área superficial em terra: {np.mean(land) * 100:.2f}%")

    print("Renderizador OpenGL:")
    try:
        plotter.render()
        print(plotter.ren_win.ReportCapabilities())
    except Exception as error:
        print(f"Não foi possível consultar o renderizador: {error}")

    print("Controles: arraste para girar e use a roda do mouse para zoom.")

    plotter.show()

def main():
    planet = generate_planet()
    visualize_planet(planet)


if __name__ == "__main__":
    main()
import numpy as np
import heapq

class Hydrology:
    def calculate_flow(self, planet, epsilon = 1e-3):
        if epsilon < 0.0:
            raise ValueError("O incremento de preenchimento não pode ser negativo.")
        
        surface = planet.surface
        elevation = surface.elevation
        sea_level = surface.sea_level
        neighbors = surface.topology.neighbors

        ocean = elevation < sea_level

        filled_elevation = elevation.copy()
        flow_to = np.full(len(elevation), -1, dtype = np.int64)
        visited = np.zeros(len(elevation), dtype = bool)

        priority_queue = []

        for face_id in np.flatnonzero(ocean):
            face_id = int(face_id)
            visited[face_id] = True

            heapq.heappush(priority_queue, (filled_elevation[face_id], face_id))

        if not priority_queue:
            lowest_face = int(np.argmin(elevation))
            visited[lowest_face] = True
            
            heapq.heappush(priority_queue, (filled_elevation[lowest_face], lowest_face))

        flow_order = []

        while priority_queue:
            _, current = heapq.heappop(priority_queue)
            flow_order.append(current)

            for neighbor in neighbors[current]:
                neighbor = int(neighbor)

                if visited[neighbor]:
                    continue

                visited[neighbor] = True

                filled_elevation[neighbor] = max(elevation[neighbor], filled_elevation[current] + epsilon)

                flow_to[neighbor] = current

                heapq.heappush(priority_queue, (filled_elevation[neighbor], neighbor))

        self.filled_elevation = filled_elevation
        self.flow_order = flow_order

        return flow_to

    def calculate_flow_accumulation(self, planet, flow_to, weights = None):
        elevation = planet.surface.elevation
        land = elevation >= planet.surface.sea_level

        if weights is None:
            accumulation = land.astype(np.float64)
        else:
            weights = np.asarray(weights, dtype = np.float64)
            if len(weights) != len(elevation):
                raise ValueError("Os pesos devem conter um valor por face.")
            accumulation = np.where(land, np.maximum(weights, 0.0), 0.0)

        for face_id in reversed(self.flow_order):
            downstream = flow_to[face_id]

            if downstream >= 0:
                accumulation[downstream] += accumulation[face_id]

        return accumulation

    def route_evapotranspiration(self, planet, flow_to, precipitation_mm, et_demand_mm, evaporation_fraction = 1.0, face_area = None):
        surface = planet.surface
        elevation = np.asarray(surface.elevation, dtype=np.float64)
        face_count = len(elevation)
        land = elevation >= surface.sea_level
        flow_to = np.asarray(flow_to, dtype=np.int64)
        precipitation_mm = np.asarray(precipitation_mm, dtype=np.float64)
        et_demand_mm = np.asarray(et_demand_mm, dtype=np.float64)

        if flow_to.shape != (face_count,):
            raise ValueError("O destino de escoamento deve conter um valor por face.")
        if precipitation_mm.shape != (face_count,) or et_demand_mm.shape != (face_count,):
            raise ValueError("A precipitação e a demanda devem conter um valor por face.")
        if not 0.0 < evaporation_fraction <= 1.0:
            raise ValueError("A fração evaporável deve estar em (0, 1].")
        if face_area is None:
            areas = np.ones(face_count, dtype=np.float64)
        else:
            areas = np.asarray(face_area, dtype=np.float64)
            if areas.shape != (face_count,) or np.any(~np.isfinite(areas)) or np.any(areas <= 0.0):
                raise ValueError("As áreas das faces devem ser finitas, positivas e conter um valor por face.")

        et_mm = np.zeros(face_count, dtype=np.float64)
        outflow_mm = np.zeros(face_count, dtype=np.float64)
        available_volume = np.where(land, np.maximum(precipitation_mm, 0.0) * areas, 0.0)
        for face_id in reversed(self.flow_order):
            if not land[face_id]:
                continue

            available = available_volume[face_id]
            available_depth = available / areas[face_id]
            et_depth = min(max(et_demand_mm[face_id], 0.0), evaporation_fraction * available_depth)
            et_volume = et_depth * areas[face_id]
            et_mm[face_id] = et_depth
            outflow_volume = max(available - et_volume, 0.0)
            outflow_mm[face_id] = outflow_volume / areas[face_id]
            downstream = int(flow_to[face_id])
            if downstream >= 0 and land[downstream]:
                available_volume[downstream] += outflow_volume

        return {
            "et_mm": et_mm,
            "outflow_mm": outflow_mm,
            "available_mm": available_volume / areas,
        }

    def route_runoff(self, planet, flow_to, runoff_mm, face_area = None):
        elevation = np.asarray(planet.surface.elevation, dtype = np.float64)
        face_count = len(elevation)
        land = elevation >= planet.surface.sea_level

        flow_to = np.asarray(flow_to, dtype = np.int64)
        runoff_mm = np.asarray(runoff_mm, dtype = np.float64)

        if flow_to.shape != (face_count):
            raise ValueError("O fluxo deve conter um destino por face.")
        if runoff_mm.shape != (face_count):
            raise ValueError("A drenagem deve conter um valor por face.")
        if not np.all(np.isfinite(runoff_mm)) or np.any(runoff_mm < 0):
            raise ValueError("A drenagem deve ser finita e não negativa.")

        if face_area is None:
            areas = np.ones(face_count, dtype = np.float64)
        else:
            areas = np.asarray(face_area, dtype = np.float64)

            if (
                areas.shape != (face_count)
                or not np.all(np.isfinite(areas))
                or np.any(areas <= 0)
            ):
                raise ValueError("As áreas devem ser finitas e positivas.")

        runoff_volume = np.where(land, runoff_mm * areas, 0.0)
        routed_outflow_mm = np.zeros(face_count, dtype = np.float64)
        external_volume_m3 = 0.0

        for face_id in reversed(self.flow_order):
            if not land[face_id]:
                continue

            volume = runoff_volume[face_id]
            routed_outflow_mm[face_id] = volume / areas[face_id]
            downstream = int(flow_to[face_id])

            if downstream >= 0 and land[downstream]:
                runoff_volume[downstream] += volume
            else:
                external_volume_m3 += volume / 1000.0

        return {
            "outflow_mm": routed_outflow_mm,
            "external_runoff_volume_m3": external_volume_m3
        }
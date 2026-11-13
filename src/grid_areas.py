"""
Cell areas of a regular or irregular latitude/longitude grid given its cell
centres (1-D coordinates), for the mass-conservation loss on grids without
bounds variables (e.g. the CORDEX NZ target grid, whose lat_bnds/lon_bnds are
referenced but not included in the files).

Cell edges are the midpoints between neighbouring centres; the two outer edges
lie half a spacing beyond the first/last centre. Area on a sphere of radius
6371 km: A = R^2 * dlon * (sin(phi_north) - sin(phi_south)), angles in radians.
For a uniform grid this equals src/india_data.cell_areas_km2.

Rotated-pole or projected grids (2-D lat/lon) are not supported: they raise.
"""
import numpy as np

EARTH_RADIUS_KM = 6371.0


def cell_edges(centres):
    """Edges (n + 1,) of 1-D, strictly monotonic cell centres (n,)."""
    c = np.asarray(centres, dtype=np.float64)
    if c.ndim != 1 or len(c) < 2:
        raise ValueError("cell centres must be a 1-D array with at least 2 values (2-D lat/lon grids are not supported)")
    d = np.diff(c)
    if not (np.all(d > 0) or np.all(d < 0)):
        raise ValueError("cell centres must be strictly monotonic")
    mid = (c[:-1] + c[1:]) / 2
    return np.concatenate([[c[0] - d[0] / 2], mid, [c[-1] + d[-1] / 2]])


def cell_areas_from_centres(lat, lon):
    """(n_lat, n_lon) cell areas in km^2 from 1-D latitude and longitude centres (degrees)."""
    lat_e = np.clip(cell_edges(lat), -90.0, 90.0)
    lon_e = cell_edges(lon)
    band = np.abs(np.sin(np.deg2rad(lat_e[1:])) - np.sin(np.deg2rad(lat_e[:-1])))   # (n_lat,)
    width = np.abs(np.deg2rad(np.diff(lon_e)))                                          # (n_lon,)
    return EARTH_RADIUS_KM ** 2 * band[:, None] * width[None, :]

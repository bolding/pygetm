"""A level's depth from a bathymetry file on the layout's grid (one reader for every way a nested model is set up:
pygetm-config's generated scripts, pygetm-nesting's tools)."""

from typing import Optional

import numpy as np

from .geometry import Level


def read_level_depth(path, lev: Level, depth_variable: str, lon_variable: str, lat_variable: str,
                     mask_variable: Optional[str] = None, fill_value: Optional[float] = None,
                     min_depth: Optional[float] = None, scale: float = 1.0) -> np.ndarray:
    """The depth of level ``lev`` (NaN on land) on its domain (window plus band, :attr:`Level.domain_shape`) from the
    NetCDF file ``path``. A file larger by the same number of cells on every side (ocean-prep's level files carry a band
    of 2 cells, also around the root) is cut to the level's grid; the file's cell centres must then lie on the
    layout's (to 1e-3 of a cell). Water: ``mask_variable`` > 0 if given, else depth != ``fill_value`` if given, else a
    finite depth; ``min_depth``: shallower cells are land (as :meth:`pygetm.domain.Domain.mask_shallow`). A curvilinear
    level (``grid``; with an overhang its whole domain): the file's cell centres (2D, or 1D where the grid is separable)
    must be those of the level's supergrid. ``lon_variable``, ``lat_variable``: the file's coordinates (for a Cartesian
    level its x and y); ``scale``: the file's coordinate units per layout unit (e.g. 1000: the file in metres, the
    layout in km)."""
    import netCDF4

    with netCDF4.Dataset(path) as nc:
        nc.set_auto_mask(False)
        H = np.asarray(nc[depth_variable][...], dtype=float)
        if mask_variable:
            wet = np.asarray(nc[mask_variable][...]) > 0
        elif fill_value is not None:
            wet = H != fill_value
        else:
            wet = np.isfinite(H)
        lon = np.asarray(nc[lon_variable][:], dtype=float) / scale  # in layout units
        lat = np.asarray(nc[lat_variable][:], dtype=float) / scale
    ny, nx = lev.domain_shape
    cj, ci = (H.shape[0] - ny) // 2, (H.shape[1] - nx) // 2  # a file larger by the same on every side: cut
    if cj < 0 or ci < 0 or H.shape != (ny + 2 * cj, nx + 2 * ci):
        raise ValueError(f"{lev.name}: the bathymetry {path} has shape {H.shape}, the layout gives this level {(ny, nx)} "
                         "(its window and band)")
    H, wet = H[cj:cj + ny, ci:ci + nx], wet[cj:cj + ny, ci:ci + nx]
    lon = lon[ci:ci + nx] if lon.ndim == 1 else lon[cj:cj + ny, ci:ci + nx]
    lat = lat[cj:cj + ny] if lat.ndim == 1 else lat[cj:cj + ny, ci:ci + nx]
    if lev.grid is not None:  # curvilinear (with an overhang: its whole domain): the supergrid's cell centres
        xs, ys = lev.domain_supergrid()
        xc, yc = xs[1::2, 1::2], ys[1::2, 1::2]
        lon2, lat2 = np.broadcast_to(lon if lon.ndim == 2 else lon[None, :], H.shape), \
            np.broadcast_to(lat if lat.ndim == 2 else lat[:, None], H.shape)
        if np.abs(lon2 - xc).max() > 1e-3 * lev.dlon or np.abs(lat2 - yc).max() > 1e-3 * lev.dlat:
            raise ValueError(f"{lev.name}: the cell centres of the bathymetry {path} are not those of the level's grid "
                             f"({lev.grid})")
    else:
        lon_e, lat_e = lev.edges()
        lonc, latc = 0.5 * (lon_e[1:] + lon_e[:-1]), 0.5 * (lat_e[1:] + lat_e[:-1])
        lon1 = lon if lon.ndim == 1 else lon[0]
        lat1 = lat if lat.ndim == 1 else lat[:, 0]
        if np.abs(lon1 - lonc).max() > 1e-3 * lev.dlon or np.abs(lat1 - latc).max() > 1e-3 * lev.dlat:
            raise ValueError(f"{lev.name}: the cell centres of the bathymetry {path} are not those of the layout")
    if min_depth is not None:
        wet &= H >= float(min_depth)  # min_depth: shallower cells are land (as Domain.mask_shallow)
    return np.where(wet, H, np.nan)

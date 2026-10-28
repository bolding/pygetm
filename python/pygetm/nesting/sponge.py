"""The sponge zone of a nested child (AGRIF-like): damping of the transports
(:attr:`pygetm.momentum.Momentum.An`, momentum only, so volume stays exactly
conserved) in the computed wet cells next to the interface, decaying inward.

The distance is counted through water only and starts at the wet ring cells:
where the child is land at its interface nothing is exchanged, so no sponge
is needed there; a peninsula does not carry the sponge across to the other
side.

With sub-stepping (``time_ratio`` > 1) and flux matching, short waves in the
child are pumped by the loop "flux mismatch -> correction of the parent's
outside cell -> the child's ring"; the sponge damps them. IDEAL (tidal,
time_ratio 3, 48 h, width 1 parent cell): stable for alpha = An dt / dx^2
from 0.03 to 0.2 with the same elevations to 1 mm; 0.02 and less grow, 0.3
exceeds the explicit diffusion limit (0.25). Default 0.1.
"""

import numpy as np

from .geometry import dilate


def water_distance(source: np.ndarray, wet: np.ndarray, max_steps: int = 10_000) -> np.ndarray:
    """Distance (in cells, 8-connected steps) from the ``source`` cells to
    every cell of ``wet``, moving through wet cells only. Unreachable cells
    get ``inf``; source cells 0."""
    wet = np.asarray(wet, dtype=bool)
    dist = np.full(wet.shape, np.inf)
    front = np.asarray(source, dtype=bool).copy()
    dist[front] = 0.0
    seen = front.copy()
    for d in range(1, max_steps + 1):
        front = dilate(front, 1) & wet & ~seen
        if not front.any():
            break
        dist[front] = d
        seen |= front
    return dist


def sponge_weight(distance: np.ndarray, width: float) -> np.ndarray:
    """1 at distance 1 (the first computed cell next to the interface),
    decreasing linearly by 1/``width`` per cell (1/``width`` at ``width``);
    0 beyond and at the source."""
    w = np.where((distance >= 1) & (distance <= width), 1.0 - (distance - 1.0) / width, 0.0)
    return np.where(np.isfinite(distance), w, 0.0)

"""Which variables parent and child exchange, and when.

The halo zones of a pyGETM simulation must hold valid values for every
variable that its equations read from neighbouring cells. Within one
simulation, pyGETM ensures this by calling its halo exchange after each
update. At a child's outer edge, or at the rim of the parent's hole, that
exchange does nothing, so the other side of the nesting must provide those
values instead.

The variables fall into three groups:

* **Exchanged** (:data:`EXCHANGE_GROUPS`): prognostic variables that pyGETM
  updates in the active cells only and then halo-exchanges. They must come
  from the other side of the nesting, at the stage where pyGETM would have
  exchanged them (:data:`pygetm.simulation.COUPLING_STAGES`).
* **Computed locally in the halo**: diagnostics that pyGETM computes over all
  cells including halos, from exchanged variables plus static fields. They
  need no exchange as long as their inputs are valid in the halo:

  - water depth ``D`` on all grids (from ``T.z`` and ``H``), and old elevations
    ``zo``, ``zio``, ``zin`` (copies of ``T.z``);
  - layer thicknesses ``hn``, ``ho`` for sigma and GVC coordinates (from ``D``);
  - density ``rho``, buoyancy ``buoy``, ``NN`` (from temperature and
    salinity), and with them the internal pressure gradients ``idpdx``,
    ``idpdy``.

* **Local inputs**: static fields (bathymetry ``H``, mask, metrics, Coriolis,
  bottom roughness) and forcing (``airsea.sp``, ``taux``, ``tauy``, ``pe``,
  heat fluxes). Each simulation provides these itself for its halo cells,
  from geometry that covers the halo. Horizontal diffusivities and
  viscosities (``Ah``, ``An``) are set by each simulation's own
  configuration.
"""

from typing import NamedTuple, Optional

from .. import core


class ExchangeGroup(NamedTuple):
    """Variables that become valid at the same stage on the same grid, and
    therefore travel together in one message."""

    #: name of the group
    name: str
    #: stage after which the group is valid (see ``COUPLING_STAGES``)
    stage: str
    #: grid: "T", "U" or "V"
    grid: str
    #: attribute paths relative to the simulation (e.g. "momentum.U").
    #: A path ending in "*" refers to a collection; see :func:`resolve`
    paths: tuple[str, ...]
    #: why the halo needs it (the term of the governing equations)
    needed_for: str


#: The variables exchanged between parent and child, in stage order.
EXCHANGE_GROUPS: tuple[ExchangeGroup, ...] = (
    ExchangeGroup(
        "U",
        "transports_2d",
        "U",
        ("momentum.U",),
        "continuity (flux through the interface face), 2D advection and"
        " diffusion of momentum, Coriolis",
    ),
    ExchangeGroup(
        "V",
        "transports_2d",
        "V",
        ("momentum.V",),
        "continuity (flux through the interface face), 2D advection and"
        " diffusion of momentum, Coriolis",
    ),
    ExchangeGroup(
        "z",
        "elevation",
        "T",
        ("T.z",),
        "surface pressure gradient; water depth D and layer thicknesses in the"
        " halo",
    ),
    ExchangeGroup(
        "uk",
        "momentum_3d",
        "U",
        ("momentum.pk", "momentum.uk"),
        "3D advection and diffusion of momentum, tracer advection fluxes"
        " through the interface",
    ),
    ExchangeGroup(
        "vk",
        "momentum_3d",
        "V",
        ("momentum.qk", "momentum.vk"),
        "3D advection and diffusion of momentum, tracer advection fluxes"
        " through the interface",
    ),
    ExchangeGroup(
        "num",
        "momentum_3d",
        "T",
        ("vertical_mixing.num",),
        "turbulent viscosity interpolated to U and V points (3D momentum)",
    ),
    ExchangeGroup(
        "ts",
        "tracers",
        "T",
        ("temp", "salt"),
        "tracer advection and horizontal diffusion; density, hence internal"
        " pressure",
    ),
    ExchangeGroup(
        "bgc",
        "tracers",
        "T",
        ("tracers*",),
        "advection and horizontal diffusion of FABM (and other) tracers",
    ),
    ExchangeGroup(
        "hn",
        "tracers",
        "T",
        ("T.hn",),
        "layer thicknesses, only for adaptive vertical coordinates, whose"
        " update is limited to the active cells",
    ),
)


def resolve(sim, group: ExchangeGroup) -> list[core.Array]:
    """Return the arrays of ``group`` that exist in ``sim``.

    Missing attributes (for instance 3D variables in a 2D run, or temperature
    and salinity in a barotropic run) are skipped, so the result may be empty.
    The collection path ``tracers*`` gives all tracers except temperature and
    salinity.
    """
    arrays: list[core.Array] = []
    for path in group.paths:
        if path == "tracers*":
            tracers = getattr(sim, "tracers", None)
            if tracers is None:
                continue
            exclude = {id(getattr(sim, "temp", None)), id(getattr(sim, "salt", None))}
            arrays.extend(t for t in tracers if id(t) not in exclude)
            continue
        obj = sim
        for name in path.split("."):
            obj = getattr(obj, name, None)
            if obj is None:
                break
        if isinstance(obj, core.Array):
            arrays.append(obj)
    return arrays


def groups_for_stage(stage: str, names: Optional[set[str]] = None):
    """Exchange groups valid after ``stage``, optionally limited to ``names``."""
    return [
        g
        for g in EXCHANGE_GROUPS
        if g.stage == stage and (names is None or g.name in names)
    ]

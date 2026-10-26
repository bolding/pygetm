from typing import Mapping, NamedTuple, Optional

from mpi4py import MPI

from .. import parallel


class NestingContext(NamedTuple):
    """Role of the current process in a nested setup."""

    #: name of the domain the current process belongs to
    name: str
    #: communicator comprising all processes of the current domain.
    #: Pass this to :class:`pygetm.domain.Domain` (``comm``) and
    #: :func:`pygetm.parallel.get_logger` (``comm``).
    comm: MPI.Comm
    #: communicator comprising all processes of all domains, reserved for
    #: exchanges between domains (it has its own tag space)
    nesting_comm: MPI.Comm
    #: rank of the first process of each domain in :attr:`nesting_comm`
    offsets: Mapping[str, int]
    #: number of processes of each domain
    sizes: Mapping[str, int]


def split_world(
    ranks: Mapping[str, int], comm: Optional[MPI.Comm] = None
) -> NestingContext:
    """Divide the MPI processes over the domains of a nested setup.

    Domains receive contiguous blocks of ranks, in the order of ``ranks``. For
    instance, ``{"root": 4, "child": 2}`` assigns ranks 0-3 to ``root`` and 4-5
    to ``child``. This matches the order of program blocks in an MPMD launch
    (``mpiexec -n 4 ... : -n 2 ...``).

    Args:
        ranks: number of processes for each domain
        comm: communicator to divide (default: ``MPI.COMM_WORLD``)
    """
    if comm is None:
        comm = MPI.COMM_WORLD
    if not ranks:
        raise ValueError("At least one domain must be specified")
    for name, n in ranks.items():
        if n < 1:
            raise ValueError(f"Domain {name} must have at least one process ({n=})")
    total = sum(ranks.values())
    if total != comm.size:
        raise ValueError(
            f"Domains need {total} processes in total, but the communicator has"
            f" {comm.size}: {dict(ranks)}"
        )

    offsets: dict[str, int] = {}
    start = 0
    for name, n in ranks.items():
        offsets[name] = start
        start += n

    names = list(ranks)
    color = next(
        i for i, name in enumerate(names)
        if offsets[name] <= comm.rank < offsets[name] + ranks[name]
    )
    domain_comm = parallel.mpi4py_autofree(comm.Split(color, comm.rank))
    nesting_comm = parallel.mpi4py_autofree(comm.Dup())
    return NestingContext(
        names[color], domain_comm, nesting_comm, offsets, dict(ranks)
    )

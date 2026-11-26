"""The nesting's point-to-point messages between a level and its parent and children,
as persistent requests with fixed buffers, in the way of pyGETM's own halo exchange
(:class:`pygetm.parallel.HaloUpdater`).

A message is identified by its communicator (one per interface), partner rank, tag and
size. It is set up the first time it is sent or received - a buffer and an
``MPI_Send_init``/``MPI_Recv_init`` request - and reused at every later sync: a send
copies the values into its buffer and starts the request, a receive starts its request
and returns its buffer. Nothing is allocated per sync, and every message of the nesting
goes through one mechanism.

Message sizes per (interface, partner, tag) are fixed by the layout and the tilings,
so the set of messages is the same at every sync; a send reused before the previous
one with the same key has completed raises an error (every sync waits for its sends).
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
from mpi4py import MPI

from .. import parallel

Key = Tuple[int, int, int, Tuple[int, ...], str]


class Message:
    """One persistent message: its buffer and its request."""

    __slots__ = ("buf", "req")

    def __init__(self, buf: np.ndarray, req: MPI.Prequest):
        self.buf = buf
        self.req = req


class Messages:
    """The persistent messages of one process (shared by its nesting couplers)."""

    __slots__ = ("_sends", "_recvs")

    def __init__(self):
        self._sends: Dict[Key, Message] = {}
        self._recvs: Dict[Key, Message] = {}

    @staticmethod
    def _key(comm: MPI.Comm, rank: int, tag: int, shape: Tuple[int, ...], dtype) -> Key:
        return (comm.py2f(), int(rank), int(tag), tuple(shape), np.dtype(dtype).str)

    def send(
        self, comm: MPI.Comm, dest: int, tag: int, values: np.ndarray
    ) -> MPI.Prequest:
        """Start sending ``values`` to rank ``dest``; returns the request (to be
        waited for before the next sync, as an ``Isend``'s)."""
        values = np.asarray(values)
        key = self._key(comm, dest, tag, values.shape, values.dtype)
        m = self._sends.get(key)
        if m is None:
            buf = np.empty(values.shape, dtype=values.dtype)
            req = parallel.mpi4py_autofree(comm.Send_init(buf, dest, tag))
            m = self._sends[key] = Message(buf, req)
        elif not m.req.Test():
            raise RuntimeError(
                f"nesting: message to rank {dest} (tag {tag}) sent again before the"
                " previous one completed"
            )
        m.buf[...] = values
        m.req.Start()
        return m.req

    def recv(
        self,
        comm: MPI.Comm,
        source: int,
        tag: int,
        shape: Tuple[int, ...],
        dtype=float,
        timer: Optional[Any] = None,
    ) -> np.ndarray:
        """Receive the message from rank ``source`` (blocking); returns its buffer,
        valid until the next receive of the same message. ``timer``: an object whose
        ``wait`` and ``transfer`` (s) get the time blocked until the message had
        arrived and the time receiving it (the receive first probes for the message,
        so the two are told apart)."""
        key = self._key(comm, source, tag, shape, dtype)
        m = self._recvs.get(key)
        if m is None:
            buf = np.empty(shape, dtype=dtype)
            req = parallel.mpi4py_autofree(comm.Recv_init(buf, source, tag))
            m = self._recvs[key] = Message(buf, req)
        if timer is None:
            m.req.Start()
            m.req.Wait()
            return m.buf
        t0 = MPI.Wtime()
        comm.Probe(source=source, tag=tag)
        t1 = MPI.Wtime()
        m.req.Start()
        m.req.Wait()
        timer.wait += t1 - t0
        timer.transfer += MPI.Wtime() - t1
        return m.buf

    def recv_into(
        self,
        comm: MPI.Comm,
        buf: np.ndarray,
        source: int,
        tag: int,
        timer: Optional[Any] = None,
    ) -> None:
        """:meth:`recv` into ``buf``."""
        buf[...] = self.recv(comm, source, tag, buf.shape, buf.dtype, timer)

    def __len__(self) -> int:
        return len(self._sends) + len(self._recvs)

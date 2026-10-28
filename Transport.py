"""Request/response channel over one shared-memory slot per client.

A channel is described only by named arrays (a per-row shape and a dtype each) and the most rows one
request may carry. The slot layout is derived from that description:

    header (64 bytes): int64 [state, count, seq, resp_seq]
    one region per input array:  max_rows rows, 64-byte aligned
    one region per output array: max_rows rows, 64-byte aligned

State machine, each side only acts in its own state:
    IDLE -(client writes rows, count, seq)-> REQUEST -(server writes rows, resp_seq)-> RESPONSE -(client copies)-> IDLE
The state word is written last on every transition, so a reader that sees the new state also sees the data.
"""
import time
from multiprocessing.shared_memory import SharedMemory

import numpy as np

IDLE, REQUEST, RESPONSE = 0, 1, 2
STATE, COUNT, SEQ, RESP_SEQ = 0, 1, 2, 3
ALIGN = 64
HEADER_BYTES = 64
SPIN_LIMIT = 2000


def _align(n):
    return (n + ALIGN - 1) // ALIGN * ALIGN


class ArraySpec:
    __slots__ = "name", "shape", "dtype", "row_nbytes", "offset"

    def __init__(self, name, shape, dtype, offset):
        self.name = name
        self.shape = tuple(int(d) for d in shape)
        self.dtype = np.dtype(dtype)
        self.row_nbytes = int(np.prod(self.shape, dtype=np.int64)) * self.dtype.itemsize
        self.offset = offset

    def __getstate__(self):
        return self.name, self.shape, self.dtype.str, self.offset

    def __setstate__(self, state):
        name, shape, dtype, offset = state
        self.__init__(name, shape, dtype, offset)


class ChannelSpec:
    """inputs / outputs: {name: (row_shape, dtype)}, in a fixed order. max_rows: rows per request."""

    def __init__(self, inputs: dict, outputs: dict, max_rows: int):
        if max_rows < 1:
            raise ValueError("max_rows must be >= 1")
        if not inputs or not outputs:
            raise ValueError("a channel needs at least one input and one output array")
        self.max_rows = int(max_rows)
        offset = HEADER_BYTES
        self.inputs, self.outputs = {}, {}
        for group, arrays in ((self.inputs, inputs), (self.outputs, outputs)):
            for name, (shape, dtype) in arrays.items():
                array = ArraySpec(name, shape, dtype, offset)
                if array.row_nbytes == 0:
                    raise ValueError(f"{name} has zero-size rows")
                group[name] = array
                offset = _align(offset + array.row_nbytes * self.max_rows)
        self.nbytes = offset

    def views(self, buf):
        header = np.ndarray((4,), dtype=np.int64, buffer=buf, offset=0)
        views = []
        for group in (self.inputs, self.outputs):
            views.append({name: np.ndarray((self.max_rows, *a.shape), dtype=a.dtype, buffer=buf, offset=a.offset)
                          for name, a in group.items()})
        return header, views[0], views[1]

    def __eq__(self, other):
        return isinstance(other, ChannelSpec) and self.__getstate__() == other.__getstate__()

    def __getstate__(self):
        return (self.max_rows,
                [(a.name, a.shape, a.dtype.str) for a in self.inputs.values()],
                [(a.name, a.shape, a.dtype.str) for a in self.outputs.values()])

    def __setstate__(self, state):
        max_rows, inputs, outputs = state
        self.__init__({n: (s, d) for n, s, d in inputs}, {n: (s, d) for n, s, d in outputs}, max_rows)

    def __repr__(self):
        fmt = lambda g: ", ".join(f"{a.name}{a.shape}:{a.dtype}" for a in g.values())
        return f"ChannelSpec(inputs=[{fmt(self.inputs)}], outputs=[{fmt(self.outputs)}], max_rows={self.max_rows})"


def create_slots(spec: ChannelSpec, n: int):
    slots = [SharedMemory(create=True, size=spec.nbytes) for _ in range(n)]
    for shm in slots:
        np.ndarray((4,), dtype=np.int64, buffer=shm.buf)[:] = 0
    return slots


def release_slots(slots):
    for shm in slots:
        shm.close()
        shm.unlink()


class Wakeup:
    """Non-blocking sleep/wake hint between processes; the slot state stays the source of truth."""

    def __init__(self, ctx=None):
        import multiprocessing as mp
        self._sem = (ctx or mp).BoundedSemaphore(1)
        self._sem.acquire()

    def signal(self):
        try:
            self._sem.release()
        except ValueError:
            pass

    def wait(self, timeout):
        return self._sem.acquire(timeout=timeout)


def _await(header, wanted, wakeup, spin_limit, deadline):
    while header[STATE] != wanted:
        for _ in range(spin_limit):
            if header[STATE] == wanted:
                return
        if deadline is not None and time.perf_counter() > deadline:
            raise TimeoutError(f"slot never reached state {wanted}")
        if wakeup is not None:
            wakeup.wait(0.05)


class _End:
    def __init__(self, spec: ChannelSpec, shm):
        self.spec = spec
        self.shm = shm
        if shm.size < spec.nbytes:
            raise ValueError(f"slot is {shm.size} bytes, the spec needs {spec.nbytes}")
        self.header, self.in_views, self.out_views = spec.views(shm.buf)

    def close(self):
        # drop the numpy views first, SharedMemory.close refuses while they export the buffer
        self.header = self.in_views = self.out_views = None


class ClientEnd(_End):
    def __init__(self, spec, shm, request_event=None, response_event=None, spin_limit=SPIN_LIMIT, timeout=None):
        super().__init__(spec, shm)
        self.request_event = request_event
        self.response_event = response_event
        self.spin_limit = spin_limit
        self.timeout = timeout
        self.seq = int(self.header[SEQ])

    def _validate(self, inputs: dict) -> int:
        if list(inputs.keys()) != list(self.spec.inputs.keys()):
            raise KeyError(f"inputs must be exactly {list(self.spec.inputs)} in order, got {list(inputs)}")
        n = None
        for name, array in inputs.items():
            a = self.spec.inputs[name]
            if not isinstance(array, np.ndarray):
                raise TypeError(f"{name} must be a numpy array")
            if array.dtype != a.dtype:
                raise TypeError(f"{name} is {array.dtype}, the channel carries {a.dtype}")
            if array.shape[1:] != a.shape or array.ndim != len(a.shape) + 1:
                raise ValueError(f"{name} rows are {array.shape[1:]}, the channel carries {a.shape}")
            if n is None:
                n = array.shape[0]
            elif array.shape[0] != n:
                raise ValueError("every input array must have the same number of rows")
        if not 1 <= n <= self.spec.max_rows:
            raise ValueError(f"{n} rows, the channel carries 1..{self.spec.max_rows}")
        return n

    def request(self, inputs: dict) -> dict:
        n = self._validate(inputs)
        if self.header[STATE] != IDLE:
            raise RuntimeError(f"slot is in state {int(self.header[STATE])}, a request is already outstanding")
        for name, array in inputs.items():
            self.in_views[name][:n] = array
        self.seq += 1
        self.header[COUNT] = n
        self.header[SEQ] = self.seq
        self.header[STATE] = REQUEST
        if self.request_event is not None:
            self.request_event.signal()

        deadline = None if self.timeout is None else time.perf_counter() + self.timeout
        _await(self.header, RESPONSE, self.response_event, self.spin_limit, deadline)
        if self.header[RESP_SEQ] != self.seq:
            raise RuntimeError(f"response is for request {int(self.header[RESP_SEQ])}, expected {self.seq}")
        outputs = {name: view[:n].copy() for name, view in self.out_views.items()}
        self.header[STATE] = IDLE
        return outputs


class ServerEnd(_End):
    def __init__(self, spec, shm, response_event=None):
        super().__init__(spec, shm)
        self.response_event = response_event

    def pending(self) -> int:
        """Rows waiting in this slot, 0 when there is no request."""
        if self.header[STATE] != REQUEST:
            return 0
        n = int(self.header[COUNT])
        if not 1 <= n <= self.spec.max_rows:
            raise RuntimeError(f"corrupt request: {n} rows")
        return n

    def inputs(self, n: int) -> dict:
        return {name: view[:n] for name, view in self.in_views.items()}

    def outputs(self, n: int) -> dict:
        return {name: view[:n] for name, view in self.out_views.items()}

    def respond(self) -> None:
        if self.header[STATE] != REQUEST:
            raise RuntimeError(f"respond() in state {int(self.header[STATE])}, there is no request")
        self.header[RESP_SEQ] = self.header[SEQ]
        self.header[STATE] = RESPONSE
        if self.response_event is not None:
            self.response_event.signal()

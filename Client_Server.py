import os
import time
import numpy as np
from collections import deque

import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory


def preload_gpu_dlls():
    """Make the pip installed CUDA / cuDNN / TensorRT DLLs findable on windows.

    The nvidia-* wheels put their DLLs under site-packages/nvidia/..., which is NOT
    on the windows DLL search path. Without this, onnxruntime_providers_tensorrt.dll
    fails to load ("cublas64_12.dll is missing", error 126) and ORT SILENTLY falls
    back to CPU. Nothing warns you: get_available_providers() still lists TensorRT,
    so Validate_Configs passes and self play then runs on CPU at a fraction of the
    speed. Measured: with the fallback, "TensorRT" connect4 batch 32 did 9,950
    states/s; with this call it does 60,784.

    No-op on linux and on onnxruntime < 1.21, which does not have the function.
    """
    import onnxruntime as rt
    if hasattr(rt, "preload_dlls"):
        rt.preload_dlls()


def _needs_fixed_shapes(providers):
    return any("Dml" in (p[0] if isinstance(p, tuple) else p) for p in providers)


def make_engine(file_path, providers, max_batch, buckets=None, batch_axis=0):
    """Pick the session strategy the execution provider needs.

    This is the ONLY place that branches on the provider, and both engines are drop in
    replacements for rt.InferenceSession, so Server never learns which one it got.
    """
    if _needs_fixed_shapes(providers):
        return BucketedEngine(file_path, providers, max_batch, buckets, batch_axis)
    return DynamicEngine(file_path, providers)


class DynamicEngine:
    """One session, any batch size. Correct for CPU, CUDA and TensorRT."""

    def __init__(self, file_path, providers):
        import onnxruntime as rt
        preload_gpu_dlls()
        self.sess = rt.InferenceSession(file_path, providers=providers)
        self.buckets = None

    def bucket_for(self, batch):
        return batch

    def run(self, output_names, input_feed, run_options=None):
        return self.sess.run(output_names, input_feed, run_options)


class BucketedEngine:
    """One shape-pinned session per bucket size; callers pad up to the nearest bucket.

    DirectML re-optimises whenever it meets a new input shape, so a single free
    dimension session thrashes when the batch varies from call to call, which is
    exactly what this server does. Measured on connect4 at batch 6: 7024us with a
    free dim against 630us pinned, an 11x difference.

    Pinning is done with add_free_dimension_override_by_name, which is a
    SessionOptions setting - the .onnx on disk keeps its dynamic batch dimension, so
    TensorRT, the CPU EP and the quantisation scripts are all unaffected.

    Cost is one session per bucket: measured ~12 MB each for connect4 (417k params)
    and ~31 MB each for gomoku (2.75M params), building in well under 0.1s each.

    Padding and trimming happen INSIDE run(), so this stays a drop in replacement for
    rt.InferenceSession: hand it any batch size and get back exactly that batch size.
    """

    def __init__(self, file_path, providers, max_batch, buckets=None, batch_axis=0):
        import onnxruntime as rt
        preload_gpu_dlls()

        # Which axis holds the batch, in the layout run() is handed. 0 for every model
        # here; pass it explicitly if a model ever puts the batch elsewhere.
        self.batch_axis = batch_axis
        self.buckets = sorted(set(buckets or self._default_buckets(max_batch)))
        self.max_batch = self.buckets[-1]

        # The free dimension's name is needed for the override and cannot be hardcoded:
        # tf2onnx regenerates names like 'unk__130' on every export, and a stale one
        # means the override silently stops applying - no error, just the slow path.
        # onnxruntime reports symbolic dimensions as strings, so a throwaway CPU
        # session gives us the name without parsing the file.
        probe = rt.InferenceSession(file_path, providers=["CPUExecutionProvider"])
        self.dim_name = next((d for d in probe.get_inputs()[0].shape
                              if isinstance(d, str)), None)
        del probe

        self.sessions = {}
        for b in self.buckets:
            sess_options = rt.SessionOptions()
            sess_options.log_severity_level = 3
            # both of these are required by the DirectML EP, not optional tuning
            sess_options.enable_mem_pattern = False
            sess_options.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
            if self.dim_name:
                sess_options.add_free_dimension_override_by_name(self.dim_name, b)
            self.sessions[b] = rt.InferenceSession(file_path,
                                                   sess_options=sess_options,
                                                   providers=providers)

    @staticmethod
    def _default_buckets(max_batch):
        # Every size while the pool is cheap, multiples of 4 beyond that. Padding is
        # nearly free for a small dispatch bound model (connect4 latency is flat from
        # batch 1 to 32) but costs real compute once the gpu saturates (gomoku scales
        # linearly with batch), so buckets stay tight rather than powers of two.
        if max_batch <= 16:
            return range(1, max_batch + 1)
        return [1, 2, 3] + list(range(4, max_batch + 1, 4)) + [max_batch]

    def bucket_for(self, batch):
        for bucket in self.buckets:
            if bucket >= batch:
                return bucket
        return self.max_batch

    def _resize(self, arr, size):
        """Pad with zeros or trim, along the batch axis."""
        current = arr.shape[self.batch_axis]
        if current == size:
            return arr
        if current > size:
            index = [slice(None)] * arr.ndim
            index[self.batch_axis] = slice(0, size)
            return arr[tuple(index)]
        pad = [(0, 0)] * arr.ndim
        pad[self.batch_axis] = (0, size - current)
        return np.pad(arr, pad)

    def run(self, output_names, input_feed, run_options=None):
        # Pad up to the nearest bucket, run, trim back. Callers hand us any batch size
        # and get that same batch size back, so this stays interchangeable with
        # rt.InferenceSession and with DynamicEngine.
        real = next(iter(input_feed.values())).shape[self.batch_axis]
        target = self.bucket_for(real)
        if target != real:
            input_feed = {name: self._resize(arr, target)
                          for name, arr in input_feed.items()}
        outputs = self.sessions[target].run(output_names, input_feed, run_options)
        if target != real:
            outputs = [self._resize(arr, real) for arr in outputs]
        return outputs


SPIN_LIMIT = 500          # ~25-50us of polling. the measured fast path is 2.5-3.6us,
# so this catches an already-arrived answer without paying for a wakeup. Beyond it,
# blocking is far cheaper than burning a core: measured 76us (windows) / 124us (linux)
# to wake on an event, against workers spending 95% / 78% of their life spinning.


def create_events(num_workers):
    """One event the workers share to wake the server, plus one per worker for replies."""
    return mp.Event(), [mp.Event() for _ in range(num_workers)]


class Parallelized_Session:
    def __init__(self,
                 worker_id,
                 shm,
                 inputs_feed_info: dict,  # this is single batch
                 outputs_feed_info: dict,
                 request_ready=None,
                 response_ready=None,
                 spin_limit=SPIN_LIMIT):
        # This is a client class which will use shared memory to send the board state and receive the policy/value
        # from the server class

        self.inputs_feed_info = inputs_feed_info
        self.outputs_feed_info = outputs_feed_info
        self.outputs_feed_info = {name: [arr_shape, np.prod(arr_shape)] for name, arr_shape in
                                  self.outputs_feed_info.items()}

        self.worker_id = worker_id
        self.shm = shm
        self.shared_arr = np.ndarray(shape=(shm.size // 4), dtype=np.float32, buffer=shm.buf)
        # Both optional: leave them None and this spins exactly as it used to, which
        # keeps the old path available for A/B measurement.
        self.request_ready = request_ready
        self.response_ready = response_ready
        self.spin_limit = spin_limit
        self.wait_timeouts = 0    # diagnostics: see _await_free_slot
        self.wait_wakeups = 0

    def _await_free_slot(self):
        """Block until shared_arr[0] == 0.0 without holding a core the whole time.

        shared_arr[0] stays the ONLY source of truth; the event is just a hint that it
        may be worth looking again. That makes a stale set harmless (we re-check and go
        back to waiting) and makes a lost wakeup cost latency rather than a deadlock,
        because every wait has a timeout.
        """
        while self.shared_arr[0] != 0.0:
            for _ in range(self.spin_limit):
                if self.shared_arr[0] == 0.0:
                    return
            if self.response_ready is None:
                continue          # no events supplied: original busy-wait behaviour
            woke = self.response_ready.wait(timeout=0.05)
            self.response_ready.clear()
            if not woke:
                # Timed out rather than being signalled. Rare and self correcting (the
                # flag is re-checked above), but if it is common the 50ms is dominating
                # and SPIN_LIMIT or the timeout needs tuning - so count it.
                self.wait_timeouts += 1
            else:
                self.wait_wakeups += 1

    def run(self, output_names: list, input_feed: dict):
        if not list(input_feed.keys()) == list(self.inputs_feed_info.keys()):
            raise ValueError(
                f"input feed key's doesn't match in content and order to the input_feed_shape, {self.inputs_feed_info.keys()}, {input_feed.keys()}")

        # check if we can send a request (measured at 0.3-0.4% of worker time)
        self._await_free_slot()
        data = np.concatenate([arr.flatten() for arr in input_feed.values()], dtype=np.float32)
        self.shared_arr[1:1 + len(data)] = data
        self.shared_arr[0] = 1.0
        if self.request_ready is not None:
            self.request_ready.set()   # wake the server if it is asleep

        # wait for the response (measured at 94.7% of worker time)
        self._await_free_slot()
        raw_outputs = self.shared_arr[:]
        # Get the outputs from the shared memory

        outputs = [None] * len(self.outputs_feed_info)
        start_index = 1
        for i, (arr_shape, arr_len) in enumerate(self.outputs_feed_info.values()):
            # Only need a reshape because batch dim = 1
            outputs[i] = np.array(raw_outputs[start_index: start_index + arr_len], copy=True).reshape(arr_shape)
            # Must copy as without doing so, any changes to shared memory will reflect in the outputs
            start_index += arr_len
        return outputs


class Server:
    def __init__(self,
                 inputs_feed_info: dict[str: list],  # this is multi-batched
                 outputs_feed_info: dict[str: list],
                 shared_memories: list[SharedMemory, ...],
                 providers,
                 file_path,
                 per_process_wait_time=0.001,
                 request_ready=None,
                 response_events=None):

        # Optional: without them the collection loop spins exactly as it used to.
        self.request_ready = request_ready
        self.response_events = response_events

        # batch dim is denoted with -1
        for input_shape in inputs_feed_info.values():
            if not isinstance(input_shape, list):
                raise TypeError("The input's shape must be a list")
            if input_shape.count(-1) > 1:
                raise ValueError("There can only be 1 dimension for the batch! This means that only 1 (-1) batch dim can be included")

        for output_shape in outputs_feed_info.values():
            if not isinstance(output_shape, list):
                raise TypeError("The output's shape must be a list")
            if output_shape.count(-1) > 1:
                raise ValueError("There can only be 1 dimension for the batch! This means that only 1 (-1) batch dim can be included")

        # inputs_feed_info will be a dict {"input_name": shape (as a numpy array or tuple)}
        self.inputs_feed_info = inputs_feed_info
        self.inputs_memory_length = 0
        self.inputs_feed_config = {}  # this is for getting the inputs from the client
        for input_name, (input_shape, infer_dtype) in inputs_feed_info.items():
            self.inputs_feed_config[input_name] = [-np.prod(input_shape),
                                                   self.compute_transposition_to_standard(input_shape)]
            input_shape.remove(-1)
            self.inputs_feed_config[input_name] += [input_shape, infer_dtype]
            # -np.prod(shape) because the batch dim is -1 and thus mul by -1 back to get a pos number
            self.inputs_memory_length += np.prod(input_shape)

        self.outputs_feed_info = outputs_feed_info
        self.outputs_memory_length = 0
        self.outputs_feed_config = {}  # this is for sending the outputs to the client
        for output_name, output_shape in outputs_feed_info.items():
            self.outputs_feed_config[output_name] = [-np.prod(output_shape),
                                                     self.compute_transposition_from_standard(output_shape)]
            output_shape.remove(-1)
            self.outputs_feed_config[output_name].append(output_shape)
            self.outputs_memory_length += np.prod(output_shape)

        # raise ValueError
        if self.inputs_memory_length > self.outputs_memory_length:
            self.memory_length = self.outputs_memory_length
        else:
            self.memory_length = self.inputs_memory_length

        self.shms = shared_memories
        self.providers = providers
        self.num_workers = len(self.shms)
        self.file_path = file_path
        self.wait_time = per_process_wait_time * self.num_workers if self.num_workers > 1 else 0.0
        # self.wait_time = 1e-2
        if self.num_workers > 1:
            self.past_fills = deque()
            self.past_wait_times = deque()
        # since there if there is only 1 worker, we can always wait for it, thus no need to wait

        self.engine = make_engine(self.file_path, providers, self.num_workers)

    def compute_transposition_to_standard(self, new_shape: list):
        # original being batch dim at index 0
        # locate the place of the -1
        # original will be the shape with the batch_dim at the start at index 0

        batch_dim = new_shape.index(-1)
        transposition = list(range(len(new_shape)))
        if batch_dim == 0:
            return None
        # the original shape is [0, 1, 2] # batch dim is 0
        # propose the new shape is [1, 2, 0] # batch dim is 2

        transposition.insert(batch_dim + 1, 0)
        return transposition[1:]

    def compute_transposition_from_standard(self, new_shape: list):
        # locate the place of the -1
        # original will be the shape with the batch_dim at the start at index 0

        batch_dim = new_shape.index(-1)
        transposition = list(range(len(new_shape)))
        if batch_dim == 0:
            return None
        # the original shape is [0, 1, 2] # batch dim is 0
        # propose the new shape is [1, 2, 0] # batch dim is 2
        transposition.insert(0, batch_dim)
        transposition.pop(batch_dim + 1)  # literally don't care if it is O(n)
        return transposition
    def compute_wait_time(self, alpha, avg_request_rate, min_wait=5e-6):
        if avg_request_rate <= 0:
            return min_wait

        ratio = alpha * self.num_workers / avg_request_rate
        if ratio < 1:
            optimal_t = -np.log(ratio) / avg_request_rate
        else:
            optimal_t = min_wait
        return max(min_wait, min(6e-3, optimal_t))


    def start(self):
        self.shared_arrs = [np.ndarray(shape=(shm.size // 4), dtype=np.float32, buffer=shm.buf) for shm in self.shms]
        while True:
            active_connections = []
            active_indexes = set()
            batched_input_feed = {name: [] for name in self.inputs_feed_info.keys()}

            # perf_counter, NOT monotonic: on windows monotonic is GetTickCount64 with
            # 15.6ms resolution, so every short collection window measured exactly 0.0.
            # That both disabled this controller (it targets 5e-6 to 6e-3) and made the
            # divisor below hit zero once the deque held 1000 consecutive zeros, which
            # crashed the server at 12 workers.
            start_time = time.perf_counter()
            ordered_indexes = []
            while len(active_connections) == 0 or (
                    len(active_connections) < len(self.shared_arrs) and time.perf_counter() - start_time <= self.wait_time):
                for i, shared_array in enumerate(self.shared_arrs):
                    if i in active_indexes:
                        continue
                    if shared_array[0] == 1.0 and len(active_connections) < len(
                            self.shared_arrs):  # wait until the client has sent some data
                        active_connections.append(shared_array)
                        active_indexes.add(i)
                        ordered_indexes.append(i)  # so replies can be matched back to workers
                        start_index = 1
                        for input_name, (arr_len, _, arr_shape, _) in self.inputs_feed_config.items():
                            batched_input_feed[input_name].append(
                                shared_array[start_index: start_index + arr_len].reshape(arr_shape))
                            start_index += arr_len

                if not active_connections and self.request_ready is not None:
                    # Nothing has arrived. Sleep on the event rather than holding a core,
                    # then re-check the flags - they stay the source of truth, so a stale
                    # or spurious set just sends us round again.
                    self.request_ready.wait(timeout=0.05)
                    self.request_ready.clear()
                    # the batching window starts when work appears, not when we began
                    # idling, otherwise idle time poisons the wait time statistics
                    start_time = time.perf_counter()
            elapsed_time = time.perf_counter() - start_time
            if self.num_workers > 1:
                self.past_fills.append(len(active_indexes)) # number of worker's requests collected
                self.past_wait_times.append(elapsed_time)
                if len(self.past_wait_times) > 1000: # 400 SMA
                    self.past_wait_times.popleft()
                    self.past_fills.popleft()
                total_wait = sum(self.past_wait_times)
                if total_wait > 0:  # guard: every retained sample can still be ~0
                    average_request_rate = sum(self.past_fills) / total_wait
                    self.wait_time = self.compute_wait_time(0.05, average_request_rate)
                # print(elapsed_time, len(active_indexes), average_request_rate, self.wait_time)
                if os.environ.get("GZ_SERVER_DEBUG"):
                    # Set GZ_SERVER_DEBUG=1 to see what the batching controller is
                    # actually doing. batch_med collapsing toward 1 while wait_time_med
                    # climbs to 6000us is the signature of the drain problem.
                    self._dbg = getattr(self, "_dbg", [])
                    self._dbg.append((elapsed_time, len(active_indexes), self.wait_time))
                    self._dbg_n = getattr(self, "_dbg_n", 0) + 1
                    if len(self._dbg) >= 2000:
                        import statistics as st
                        w, f, wt = zip(*self._dbg)
                        print(f"[srv] batches={self._dbg_n} "
                              f"collect_med={st.median(w) * 1e6:.0f}us "
                              f"batch_med={st.median(f):.1f} batch_max={max(f)} "
                              f"wait_time_med={st.median(wt) * 1e6:.0f}us", flush=True)
                        self._dbg.clear()

            if len(active_connections) > len(self.shared_arrs):
                raise ValueError("Too many samples in 1 batch sth went wrong")
            # implement sync by checking for inputs again

            for input_name, (_, transposition, shape, infer_dtype) in self.inputs_feed_config.items():
                if transposition is None:
                    batched_input_feed[input_name] = np.array(batched_input_feed[input_name], dtype=infer_dtype)
                else:
                    batched_input_feed[input_name] = np.array(batched_input_feed[input_name],
                                                              dtype=infer_dtype).transpose(transposition)

            batched_outputs = self.engine.run(list(self.outputs_feed_info.keys()),
                                              input_feed=batched_input_feed)
            # Must transpose if possible to (batch, ...) to iterate through it

            for i, (_, transposition, _) in enumerate(self.outputs_feed_config.values()):
                if transposition is not None:
                    batched_outputs[i] = batched_outputs[i].transpose(transposition)

            for i, shared_array in enumerate(active_connections):
                flattened_outputs = np.concatenate([output[i].reshape((-1,)) for output in batched_outputs],
                                                   dtype=np.float32)
                shared_array[1:1 + len(flattened_outputs)] = flattened_outputs
                shared_array[0] = 0.0
                # reset it so that the session can pick it up
                event = (self.response_events[ordered_indexes[i]]
                         if self.response_events else None)
                if event is not None:
                    # flag first, then the nudge: the worker re-checks the flag on wake,
                    # so this ordering can never hand back a stale response
                    event.set()


def start_server(inputs_feed_info, outputs_feed_info, shms, providers, file_path,
                 per_process_wait_time=0.001, request_ready=None, response_events=None):

    server = Server(inputs_feed_info,
                    outputs_feed_info,
                    shms,
                    providers,
                    file_path,
                    per_process_wait_time,
                    request_ready,
                    response_events
                    )
    server.start()


def create_shared_memory(inputs_feed_info, outputs_feed_info, num_workers=os.cpu_count()):
    max_length_inputs = 1
    for (input_shape, _) in inputs_feed_info.values():
        max_length_inputs += -np.prod(input_shape)

    max_length_outputs = 1
    for output_shape in outputs_feed_info.values():
        max_length_outputs += -np.prod(output_shape)

    shared_mem_len = int(max(max_length_inputs, max_length_outputs))
    return [SharedMemory(create=True, size=(4 * (shared_mem_len + 1))) for _ in range(num_workers)]


def convert_to_single_info(batched_info):
    # removes the -1 and replaces it with a 1, which should be what a single instance requires
    unbatched_info = {}
    for name, info in batched_info.items():
        if isinstance(info[0], list):
            arr_shape = np.array(info[0])
        else:
            arr_shape = np.array(info)
        arr_shape[arr_shape == -1] = 1
        unbatched_info[name] = arr_shape.tolist()
    return unbatched_info


if __name__ == "__main__":
    import onnxruntime as rt
    from Gomoku.config import build_config, train_config

    embed_size, num_heads, num_layers = build_config["embed_size"], build_config["num_heads"], build_config[
        "num_layers"]
    min_shape, max_shape, opt_shape = 1, 12, 12
    providers = [
        ('TensorrtExecutionProvider', {
            "trt_engine_cache_enable": True,
            "trt_dump_ep_context_model": True,
            "trt_builder_optimization_level": 5,
            "trt_auxiliary_streams": 0,
            "trt_ep_context_file_path": "Gomoku/Cache/",

            "trt_profile_min_shapes": f"inputs:{min_shape}x15x15,input_state:{num_layers}x2x{min_shape}x{embed_size},input_state_matrix:{num_layers}x{min_shape}x{num_heads}x{embed_size // num_heads}x{embed_size // num_heads}",
            "trt_profile_max_shapes": f"inputs:{max_shape}x15x15,input_state:{num_layers}x2x{max_shape}x{embed_size},input_state_matrix:{num_layers}x{max_shape}x{num_heads}x{embed_size // num_heads}x{embed_size // num_heads}",
            "trt_profile_opt_shapes": f"inputs:{opt_shape}x15x15,input_state:{num_layers}x2x{opt_shape}x{embed_size},input_state_matrix:{num_layers}x{opt_shape}x{num_heads}x{embed_size // num_heads}x{embed_size // num_heads}",
        }),
        'CUDAExecutionProvider',
        'CPUExecutionProvider']

    batched_inputs_feed_info = {"inputs": [[-1, 15, 15, ], np.float32],
                                "input_state": [[num_layers, 2, -1, embed_size], np.float32],
                                "input_state_matrix": [
                                    [num_layers, -1, num_heads, embed_size // num_heads, embed_size // num_heads],
                                    np.float32]}  # required inputs shape for the neural network
    batched_outputs_feed_info = {"policy": [-1, 225],
                                 "value": [-1, 1],
                                 "output_state": [num_layers, 2, -1, embed_size],
                                 "output_state_matrix": [num_layers, -1, num_heads, embed_size // num_heads,
                                                         embed_size // num_heads]}  # neural networks batch dim
    # print(convert_to_single_info(batched_inputs_feed_info))
    # raise ValueError
    num_workers = 12

    shms = create_shared_memory(batched_inputs_feed_info, batched_outputs_feed_info, num_workers)

    dummy_inputs = np.random.randint(-1, 2, (num_workers, 15, 15)).astype(np.float32)
    dummy_state = np.random.uniform(size=[num_layers, 2, num_workers, embed_size]).astype(dtype=np.float32)
    dummy_state_matrix = np.random.uniform(
        size=[num_layers, num_workers, num_heads, embed_size // num_heads, embed_size // num_heads]).astype(
        dtype=np.float32)


    def task(worker_id, shm):
        inputs_feed_shape = convert_to_single_info(batched_inputs_feed_info)
        outputs_feed_shape = convert_to_single_info(batched_outputs_feed_info)

        sess = Parallelized_Session(worker_id, shm, inputs_feed_shape, outputs_feed_shape)
        # print(dummy_state[:, :,worker_id: worker_id + 1].shape)
        sess.run(list(outputs_feed_shape.keys()),
                 {"inputs": dummy_inputs[worker_id],
                  "input_state": np.copy(dummy_state[:, :, worker_id: worker_id + 1]),
                  "input_state_matrix": np.copy(dummy_state_matrix[:, worker_id: worker_id + 1])
                  })
        return sess


    server_process = mp.Process(target=start_server, args=(batched_inputs_feed_info,
                                                           batched_outputs_feed_info,
                                                           shms,
                                                           providers,
                                                           rt.SessionOptions(),
                                                           "Gomoku/Cache/model_ctx.onnx"))
    server_process.start()

    real_sess = rt.InferenceSession("Gomoku/Cache/model_ctx.onnx", providers=providers)
    # real_sess = rt.InferenceSession("Gomoku/model.onnx", providers=providers)

    b_p, b_v, b_state, b_state_matrix = real_sess.run(["policy", "value", "output_state", "output_state_matrix"], {
        "inputs": dummy_inputs,
        "input_state": dummy_state,
        "input_state_matrix": dummy_state_matrix
    })

    # print(state0.shape, b_state.shape)
    # print(np.allclose(state0, b_state))
    # print(state0 == b_state)

    policy, value, state, state_matrix = [], [], [], []
    # sessions = [task(worker_id, shms[worker_id]) for worker_id in range(num_workers)]
    # for sess in sessions:
    #     sess.get_outputs()

    # print(np.sum(dummy_state))
    # print(np.sum(dummy_state_matrix))
    sessions = [task(worker_id, shms[worker_id]) for worker_id in range(num_workers)]

    for session in sessions:
        outputs = session.get_outputs()
        p, v, s, s_m = outputs
        policy.append(p)
        value.append(v)
        state.append(s)
        state_matrix.append(s_m)

    policy = np.concatenate(policy)
    value = np.concatenate(value)
    state = np.concatenate(state, axis=2)
    # print(state.shape)
    state_matrix = np.concatenate(state_matrix, axis=1)

    print(np.allclose(policy, b_p))
    print(np.mean(abs(policy - b_p)))

    print(np.allclose(value, b_v))
    print(np.mean(abs(value - b_v)))

    print(np.allclose(state, b_state, atol=1e-3, rtol=1e-5))
    print(np.mean(abs(state - b_state)))

    print(np.allclose(state_matrix, b_state_matrix, atol=1e-3, rtol=1e-5))
    print(np.mean(abs(state_matrix - b_state_matrix)))
    print(np.max(abs(state_matrix - b_state_matrix)))

    for shm in shms:
        shm.close()
        shm.unlink()
    server_process.terminate()

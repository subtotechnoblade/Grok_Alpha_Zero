import os
import numpy as np
import onnxruntime as rt

import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
class Parallelized_Session:
    def __init__(self,
                 worker_id,
                 shm,
                 inputs_feed_info:dict, # this is single batch
                 outputs_feed_info:dict):
        # This is a client class which will use shared memory to send the board state and receive the policy/value
        # from the server class

        self.inputs_feed_info = inputs_feed_info
        self.outputs_feed_info = outputs_feed_info

        self.worker_id = worker_id
        self.shared_arr = np.ndarray(shape=(shm.size // 4), dtype=np.float32, buffer=shm.buf)

    def run(self, output_names:list, input_feed:dict):
        if not list(input_feed.keys()) == list(self.inputs_feed_info.keys()):
            raise ValueError(f"input feed key's doesn't match in content and order to the input_feed_shape, {self.inputs_feed_info.keys()}, {input_feed.keys()}")

        while True:
            if self.shared_arr[0] == 0.0:

                data = np.concatenate([input_feed[key].reshape((-1,)) for key in input_feed.keys()], dtype=np.float32)

                self.shared_arr[1:1 + len(data)] = data
                self.shared_arr[0] = 1.0
                break
    def get_outputs(self):
        # assume the server puts data into the shared memory buffer
        # and signifies with a 0.0 that the returned p,v is here

        while True:
            if self.shared_arr[0] == 0.0:

                outputs = []
                start_index = 1
                for arr_shape in self.outputs_feed_info.values():
                    arr_len = np.prod(arr_shape)
                    # print(arr_len)
                    outputs.append(np.copy(self.shared_arr[start_index: start_index + arr_len]).reshape(arr_shape))
                    # Only need a reshape because batch dim = 1
                    # Must copy as without doing so, any changes to shared memory will reflect in the outputs
                    start_index += arr_len
                # self.shared_arr[:] = 0.0 # Can do this because the outputs are copied
                # ^ probably not needed
                return outputs

class Server:
    def __init__(self,
                 inputs_feed_info: dict[str: list], # this is multi-batched
                 outputs_feed_info: dict[str: list],
                 shared_memories: list[SharedMemory, ...],
                 providers,
                 file_path):

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
        self.inputs_feed_config = {} # this is for getting the inputs from the client
        for input_name, (input_shape, infer_dtype) in inputs_feed_info.items():

            self.inputs_feed_config[input_name] = [-np.prod(input_shape), self.compute_transposition_to_standard(input_shape)]
            input_shape.remove(-1)
            self.inputs_feed_config[input_name] += [input_shape, infer_dtype]
            # -np.prod(shape) because the batch dim is -1 and thus mul by -1 back to get a pos number
            self.inputs_memory_length += np.prod(input_shape)

        self.outputs_feed_info = outputs_feed_info
        self.outputs_memory_length = 0
        self.outputs_feed_config = {} # this is for sending the outputs to the client
        for output_name, output_shape in outputs_feed_info.items():
            self.outputs_feed_config[output_name] = [-np.prod(output_shape), self.compute_transposition_from_standard(output_shape)]
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


        self.sess = rt.InferenceSession(f"{self.file_path}",
                                        providers=providers)

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
        transposition.pop(batch_dim + 1) # literally don't care if it is O(n)
        return transposition

    def start(self):
        while True:
            active_connections = []
            inactivate_connections = []
            batched_input_feed = {name:[] for name in self.inputs_feed_info.keys()}

            self.shared_arrs = [np.ndarray(shape=(shm.size // 4), dtype=np.float32, buffer=shm.buf) for shm in self.shms]
            while len(active_connections) == 0:
                for shared_array in self.shared_arrs:
                    if shared_array[0] == 1.0: # wait until the client has sent some data
                        active_connections.append(shared_array)
                        start_index = 1
                        for input_name, (arr_len, _, arr_shape, _) in self.inputs_feed_config.items():
                            batched_input_feed[input_name].append(shared_array[start_index: start_index + arr_len].reshape(arr_shape))
                            start_index += arr_len
                    else:
                        inactivate_connections.append(shared_array)

            # for shared_array in inactivate_connections:
            #     if shared_array[0] == 1.0:  # wait until the client has sent some data
            #         active_connections.append(shared_array)
            #         start_index = 1
            #         for input_name, (arr_len, _, shape) in self.inputs_feed_config.items():
            #             batched_input_feed[input_name].append(shared_array[start_index: start_index + arr_len].reshape(shape))
            #             start_index += arr_len

            # implement sync by checking for inputs again

            for input_name, (_, transposition, shape, infer_dtype) in self.inputs_feed_config.items():
                if transposition is None:
                    batched_input_feed[input_name] = np.array(batched_input_feed[input_name], dtype=infer_dtype)
                else:
                    batched_input_feed[input_name] = np.array(batched_input_feed[input_name], dtype=infer_dtype).transpose(transposition)


            batched_outputs = self.sess.run(list(self.outputs_feed_info.keys()), input_feed=batched_input_feed)

            # print(batched_outputs[1])

            # Must transpose if possible to (batch, ...) to iterate through it
            for i, (_, transposition, _) in enumerate(self.outputs_feed_config.values()):
                if transposition is not None:
                    batched_outputs[i] = batched_outputs[i].transpose(transposition)

            # verify that
            for o in batched_outputs:
                if o.shape[0] != len(active_connections):
                    raise ValueError(f"The last batch dim isn't iterable because is it greater or less than the active connections. Meaning that it is not the batch dim. Shape:{o.shape}!")

            # print(active_connections[0][0])
            for i, shared_array in enumerate(active_connections):
                flattened_outputs = np.concatenate([output[i].reshape((-1,)) for output in batched_outputs], dtype=np.float32)
                shared_array[1:1 + len(flattened_outputs)] = flattened_outputs
                shared_array[0] = 0.0 # reset it so that the session can pick it up


        # policies,
if __name__ == "__main__":
    from Gomoku.Gomoku import build_config, train_config
    embed_size, num_heads, num_layers = build_config["embed_size"],  build_config["num_heads"], build_config["num_layers"]
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



    batched_inputs_feed_info = {"inputs": [[-1, 15, 15,], np.float32],
                                 "input_state": [[num_layers, 2, -1, embed_size], np.float32],
                                 "input_state_matrix": [[num_layers, -1, num_heads, embed_size // num_heads, embed_size // num_heads], np.float32]} # required inputs shape for the neural network
    batched_outputs_feed_info = {"policy": [-1, 255],
                                  "value": [-1, 1],
                                  "output_state": [num_layers, 2, -1, embed_size],
                                  "output_state_matrix": [num_layers, -1, num_heads, embed_size // num_heads, embed_size // num_heads]} # neural networks batch dim

    max_length_inputs = 1
    for (input_shape, _) in batched_inputs_feed_info.values():
        max_length_inputs += -np.prod(input_shape)

    max_length_outputs = 1
    for output_shape in batched_outputs_feed_info.values():
        max_length_outputs += -np.prod(output_shape)

    shared_mem_len = max(max_length_inputs, max_length_outputs)

    num_workers = 12
    shms = [SharedMemory(create=True, size=(4 * (shared_mem_len + 1))) for worker_id in range(num_workers)]
    dummy_inputs = np.random.randint(-1, 2, (num_workers, 15, 15)).astype(np.float32)
    dummy_state = np.random.uniform(size=[num_layers, 2, num_workers, embed_size]).astype(dtype=np.float32)
    dummy_state_matrix = np.random.uniform(size=[num_layers, num_workers, num_heads, embed_size // num_heads, embed_size // num_heads]).astype(dtype=np.float32)
    def task(worker_id, shm):
        inputs_feed_shape = {"inputs": [1, 15, 15,],
                             "input_state":[num_layers, 2, 1, embed_size],
                             "input_state_matrix": [num_layers, 1, num_heads, embed_size // num_heads, embed_size // num_heads]}# shouldn't have the batch dimension

        outputs_feed_shape = {"policy": [1, 225],
                              "value": [1, 1],
                              "output_state": [num_layers, 2, 1, embed_size],
                              "output_state_matrix": [num_layers, 1, num_heads, embed_size // num_heads, embed_size // num_heads]}

        sess = Parallelized_Session(worker_id, shm, inputs_feed_shape, outputs_feed_shape)
        # print(dummy_state[:, :,worker_id: worker_id + 1].shape)
        sess.run(list(outputs_feed_shape.keys()),
                 {"inputs": dummy_inputs[worker_id],
                  "input_state": np.copy(dummy_state[:, :, worker_id: worker_id + 1]),
                  "input_state_matrix": np.copy(dummy_state_matrix[:, worker_id: worker_id + 1])
                  })
        return sess

    def start_server(inputs_feed_info, outputs_feed_info, shms, providers):

        server = Server(inputs_feed_info,
                        outputs_feed_info,
                        shms,
                        providers,
                        # "Gomoku/model.onnx"
                        "Gomoku/Cache/model_ctx.onnx"
                        )

        server.start()
    server_process = mp.Process(target=start_server, args=(batched_inputs_feed_info, batched_outputs_feed_info, shms, providers))
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
    sessions = [task(worker_id, shms[worker_id]) for worker_id in range(num_workers)]
    for sess in sessions:
        sess.get_outputs()

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



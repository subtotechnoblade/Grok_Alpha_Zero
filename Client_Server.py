import os
import numpy as np
import onnxruntime as rt

import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
class Parallelized_Session:
    def __init__(self,
                 worker_id,
                 shm,
                 inputs_feed_shape:dict, # this is single batch
                 outputs_feed_shape:dict):
        # This is a client class which will use shared memory to send the board state and receive the policy/value
        # from the server class

        self.inputs_feed_shape = inputs_feed_shape
        self.outputs_feed_shape = outputs_feed_shape

        self.worker_id = worker_id
        self.shared_arr = np.ndarray(shape=(shm.size // 4), dtype=np.float32, buffer=shm.buf)

    def run(self, output_names:list, input_feed:dict):
        if not list(input_feed.keys()) == list(self.inputs_feed_shape.keys()):
            raise ValueError(f"input feed key's doesn't match in content and order to the input_feed_shape, {self.inputs_feed_shape.keys()}, {input_feed.keys()}")

        while True:
            if self.shared_arr[0] == 0.0:

                data = np.concatenate([input_feed[key].reshape((-1,)) for key in input_feed.keys()])

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
                for arr_shape in self.outputs_feed_shape.values():
                    arr_len = np.prod(arr_shape)
                    # print(arr_len)
                    outputs.append(self.shared_arr[start_index: start_index + arr_len].reshape(arr_shape))
                    # Only need a reshape because batch dim = 1
                    start_index += arr_len
                # print(outputs)
                return outputs

class Server:
    def __init__(self,
                 inputs_feed_shape: dict[str: list], # this is multi-batched
                 outputs_feed_shape: dict[str: list],
                 shared_memories: list[SharedMemory, ...],
                 providers,
                 file_path):

        # batch dim is denoted with -1

        for shape in inputs_feed_shape.values():
            if not isinstance(shape, list):
                raise TypeError("The input's shape must be a list")
            if shape.count(-1) > 1:
                raise ValueError("There can only be 1 dimension for the batch! This means that only 1 (-1) batch dim can be included")

        for shape in outputs_feed_shape.values():
            if not isinstance(shape, list):
                raise TypeError("The output's shape must be a list")
            if shape.count(-1) > 1:
                raise ValueError("There can only be 1 dimension for the batch! This means that only 1 (-1) batch dim can be included")

        # inputs_feed_shape will be a dict {"input_name": shape (as a numpy array or tuple)}
        self.inputs_feed_shape = inputs_feed_shape
        self.inputs_memory_length = 0
        self.inputs_feed_config = {} # this is for getting the inputs from the client
        for input_name, input_shape in inputs_feed_shape.items():
            self.inputs_feed_config[input_name] = [-np.prod(input_shape), self.compute_transposition_to_standard(input_shape)]
            input_shape.remove(-1)
            self.inputs_feed_config[input_name].append(input_shape)
            # -np.prod(shape) because the batch dim is -1 and thus mul by -1 back to get a pos number
            self.inputs_memory_length += np.prod(input_shape)

        self.outputs_feed_shape = outputs_feed_shape
        self.outputs_memory_length = 0
        self.outputs_feed_config = {} # this is for sending the outputs to the client
        for output_name, output_shape in outputs_feed_shape.items():
            print("Output trans:", self.compute_transposition_from_standard(output_shape), output_shape)
            self.outputs_feed_config[output_name] = [-np.prod(output_shape), self.compute_transposition_from_standard(output_shape)]
            # if self.outputs_feed_config[output_name][1]  is not None:
            #     # print(np.array(output_shape), (self.outputs_feed_config[output_name][1]))
            #     print(np.array(np.zeros([5, 2, 1, 128]).transpose(self.outputs_feed_config[output_name][1])).shape)
            #
            #     # raise ValueError
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

        active_connections = []
        inactivate_connections = []
        batched_input_feed = {name:[] for name in self.inputs_feed_shape.keys()}

        self.shared_arrs = [np.ndarray(shape=(shm.size // 4), dtype=np.float32, buffer=shm.buf) for shm in self.shms]
        while len(active_connections) == 0:
            for shared_array in self.shared_arrs:
                if shared_array[0] == 1.0: # wait until the client has sent some data
                    active_connections.append(shared_array)
                    start_index = 1
                    for input_name, (arr_len, _, shape) in self.inputs_feed_config.items():
                        batched_input_feed[input_name].append(shared_array[start_index: start_index + arr_len].reshape(shape))
                        start_index += arr_len
                else:
                    inactivate_connections.append(shared_array)

        # for shared_array in inactivate_connections:
        #     if shared_array[0] == 1.0:  # wait until the client has sent some data
        #         active_connections.append(shared_array)
        #         start_index = 1
        #         for input_name, (arr_len, _, shape) in self.inputs_feed_config.items():
        #             input_feeds[input_name].append(shared_array[start_index: start_index + arr_len].reshape(shape))
        #             start_index += arr_len

        # implement sync by checking for inputs again

        for input_name, (_, transposition, shape) in self.inputs_feed_config.items():
            if transposition is None:
                batched_input_feed[input_name] = np.array(batched_input_feed[input_name])
            else:
                batched_input_feed[input_name] = np.array(batched_input_feed[input_name]).transpose(transposition)

        # print(batched_input_feed)
        # outputs = [np.array([np.ones(shape=shape, dtype=np.float32) for _ in range(self.num_workers)]).transpose([1, 0]) for _, _, shape in self.outputs_feed_config.values()]
        # outputs[0] *= 0
        print([s.shape for s in batched_input_feed.values()])
        batched_outputs = self.sess.run(list(batched_outputs_feeds_shape.keys()), input_feed=batched_input_feed)

        # Must transpose if possible to (batch, ...) to iterate through it
        for i, (_, transposition, _) in enumerate(self.outputs_feed_config.values()):
            if transposition is not None:
                # print(batched_outputs[i].shape, transposition)
                batched_outputs[i] = batched_outputs[i].transpose(transposition)
                # print(batched_outputs[i].shape)
                # print(batched_outputs[i].shape.transpose(transposition))

        # verify that
        for o in batched_outputs:
            if o.shape[0] != len(active_connections):
                raise ValueError(f"The last batch dim isn't iterable because is it greater or less than the active connections. Meaning that it is not the batch dim. Shape:{o.shape}!")

        # print(active_connections[0][0])
        for i, shared_array in enumerate(active_connections):
            flattened_outputs = np.concatenate([output[i].flatten() for output in batched_outputs], dtype=np.float32)
            shared_array[1:1 + len(flattened_outputs)] = flattened_outputs

            shared_array[0] = 0.0 # reset it so that the session can pick it up


        # policies,
if __name__ == "__main__":
    from Gomoku.Gomoku import build_config, train_config
    embed_size, num_heads, num_layers = build_config["embed_size"],  build_config["num_heads"], build_config["num_layers"]
    min_shape, max_shape, opt_shape = 1, 12, 12
    providers = [
        # ('TensorrtExecutionProvider', {
        # "trt_engine_cache_enable": True,
        # "trt_dump_ep_context_model": True,
        # "trt_builder_optimization_level": 5,
        # "trt_auxiliary_streams": 0,
        # "trt_ep_context_file_path": "Gomoku/Cache/",
        #
        # "trt_profile_min_shapes": f"inputs:{min_shape}x15x15,input_state:{num_layers}x2x{min_shape}x{embed_size},input_state_matrix:{num_layers}x{min_shape}x{num_heads}x{embed_size // num_heads}x{embed_size // num_heads}",
        # "trt_profile_max_shapes": f"inputs:{max_shape}x15x15,input_state:{num_layers}x2x{max_shape}x{embed_size},input_state_matrix:{num_layers}x{max_shape}x{num_heads}x{embed_size // num_heads}x{embed_size // num_heads}",
        # "trt_profile_opt_shapes": f"inputs:{opt_shape}x15x15,input_state:{num_layers}x2x{opt_shape}x{embed_size},input_state_matrix:{num_layers}x{opt_shape}x{num_heads}x{embed_size // num_heads}x{embed_size // num_heads}",
        # }),
        # 'CUDAExecutionProvider',
        'CPUExecutionProvider']



    batched_inputs_feeds_shape = {"inputs": [-1, 15, 15,],
                                  "input_state":[num_layers, 2, -1, embed_size],
                                  "input_state_matrix": [num_layers, -1, num_heads, embed_size // num_heads, embed_size // num_heads]} # required inputs shape for the neural network
    batched_outputs_feeds_shape = {"policy": [-1, 255],
                                   "value": [-1, 1],
                                   "output_state": [num_layers, 2, -1, embed_size],
                                   "output_state_matrix": [num_layers, -1, num_heads, embed_size // num_heads, embed_size // num_heads]} # neural networks batch dim

    max_length_inputs = 1
    for shape in batched_inputs_feeds_shape.values():
        max_length_inputs += -np.prod(shape)

    max_length_outputs = 1
    for shape in batched_outputs_feeds_shape.values():
        max_length_outputs += -np.prod(shape)

    shared_mem_len = max(max_length_inputs, max_length_outputs)

    num_workers = 2
    shms = [SharedMemory(create=True, size=(4 * (shared_mem_len + 1))) for worker_id in range(num_workers)]
    dummy_inputs = np.random.randint(-1, 2, (2, 15, 15)).astype(np.float32)
    dummy_state = np.zeros([num_layers, 2, 2, embed_size], dtype=np.float32)
    dummy_state_matrix = np.zeros([num_layers, 2, num_heads, embed_size // num_heads, embed_size // num_heads], np.float32)
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
                  "input_state": dummy_state[:, :, worker_id: worker_id + 1],
                  "input_state_matrix": dummy_state[:, worker_id: worker_id + 1]
                  })
        return sess
    sess0 = task(0, shms[0])
    sess1 = task(1, shms[1])
    # print(np.ndarray(shape=(shms[0].size // 4), dtype=np.float32, buffer=shms[0].buf))

    server = Server(batched_inputs_feeds_shape,
                    batched_outputs_feeds_shape,
                    shms,
                    providers,
                    "Gomoku/model.onnx")

    server.start()

    policy0, value0, state0, state_matrix0 =  sess0.get_outputs()
    policy1, value1, state1, state_matrix1 = sess1.get_outputs()
    # print(state0.shape, state1.shape)

    # real_sess = rt.InferenceSession("Gomoku/Cache/model_ctx.onnx", providers=providers)
    real_sess = rt.InferenceSession("Gomoku/model.onnx", providers=providers)

    b_p, b_v, b_state, b_state_matrix = real_sess.run(["policy", "value", "output_state", "output_state_matrix"], {
        "inputs": dummy_inputs,
        "input_state": dummy_state,
        "input_state_matrix": dummy_state_matrix
    })
    # print(state0.shape, b_state.shape)
    # print(np.allclose(state0, b_state))
    # print(state0 == b_state)

    policy = np.concatenate((policy0, policy1))
    value = np.concatenate((value0, value1))
    state = np.concatenate((state0, state1), axis=2)
    print(state.shape)
    state_matrix = np.concatenate((state_matrix0, state_matrix1), axis=1)

    print(np.allclose(policy, b_p))
    print(np.allclose(value, b_v))
    print(np.allclose(state, b_state))
    print(np.allclose(state_matrix, b_state_matrix))

    for shm in shms:
        shm.close()
        shm.unlink()



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
        self.shm: SharedMemory = shm

    def run(self, output_names:list, input_feed:dict):
        if not list(input_feed.keys()) == list(self.inputs_feed_shape.keys()):
            raise ValueError(f"input feed key's doesn't match in content and order to the input_feed_shape, {self.inputs_feed_shape.keys()}, {input_feed.keys()}")

        shared_arr = np.ndarray(shape=(self.shm.size // 4), dtype=np.float32, buffer=self.shm.buf)

        while True:
            if shared_arr[0] == 0.0:

                data = np.concatenate([input_feed[key].reshape((-1,)) for key in input_feed.keys()])

                shared_arr[1:1 + len(data)] = data
                shared_arr[0] = 1.0
                break
        # assume the server puts data into the shared memory buffer
        # and signifies with a 0.0 that the returned p,v is here

        # while True:
        #     if shared_arr[0] == 0.0:
        #         outputs = []
        #         start_index = 1
        #         for arr_shape in self.outputs_feed_shape.values():
        #             arr_len = np.prod(arr_shape)
        #             outputs.append(shared_arr[start_index: start_index + arr_len])
        #             start_index += arr_len
        #         return outputs

class Server:
    def __init__(self,
                 inputs_feed_shape: dict[str: list], # this is multi-batched
                 outputs_feed_shape: dict[str: list],
                 shared_memories: list[SharedMemory, ...],
                 providers,
                 file_path,
                 num_workers=os.cpu_count()):

        # batch dim is denoted with -1

        for shape in inputs_feed_shape.values():
            if not isinstance(shape, list):
                raise TypeError("The input's shape must be a list")
            if shape.count(-1) > 1:
                raise ValueError("There can only be 1 dimension for the batch! This means that only 1 None can eb included")

        for shape in outputs_feed_shape.values():
            if not isinstance(shape, list):
                raise TypeError("The output's shape must be a list")
            if shape.count(-1) > 1:
                raise ValueError("There can only be 1 dimension for the batch! This means that only 1 None can eb included")

        # inputs_feed_shape will be a dict {"input_name": shape (as a numpy array or tuple)}
        self.inputs_feed_shape = inputs_feed_shape
        self.inputs_memory_length = 0
        self.inputs_feed_config = {} # this is for getting the inputs from the client
        for input_name, shape in inputs_feed_shape.items():
            self.inputs_feed_config[input_name] = [-np.prod(shape), self.compute_transposition(shape)]
            shape.remove(-1)
            self.inputs_feed_config[input_name].append(shape)
            # -np.prod(shape) because the batch dim is -1 and thus mul by -1 back to get a pos number
            self.inputs_memory_length += np.prod(shape)

        self.outputs_feed_shape = outputs_feed_shape
        self.outputs_memory_length = 0
        self.outputs_feed_config = {} # this is for sending the outputs to the client
        for output_name, shape in outputs_feed_shape.items():
            self.outputs_feed_config[output_name] = [-np.prod(shape), self.compute_transposition(shape)]
            shape.remove(-1)
            self.outputs_feed_config[output_name].append(shape)
            self.outputs_memory_length += np.prod(shape)

        if self.inputs_memory_length > self.outputs_memory_length:
            self.memory_length = self.outputs_memory_length
        else:
            self.memory_length = self.inputs_memory_length

        self.providers = providers
        self.num_workers = num_workers
        self.file_path = file_path


        # self.sess = rt.InferenceSession(f"{self.file_path}/TRT_cache/model_ctx.onnx",
        #                                 providers=providers)
        self.shms = shared_memories

    def compute_transposition(self, new_shape: list):
        # locate the place of the -1
        # original will be the shape with the batch_dim at the start at index 0

        batch_dim = new_shape.index(-1)
        transposition = list(range(len(new_shape)))
        if batch_dim == 0:
            return None
        # original shape is [0, 1, 2] # batch dim is 0
        # propose the new shape is [1, 2, 0] # batch dim is 2

        transposition.insert(batch_dim + 1, 0)
        return transposition[1:]

    def start(self):

        active_connections = []
        inactivate_connections = []
        input_feeds = {name:[] for name in self.inputs_feed_shape.keys()}

        self.shared_arrs = [np.ndarray(shape=(shm.size // 4), dtype=np.float32, buffer=shm.buf) for shm in self.shms]
        while len(active_connections) == 0:
            for shared_array in self.shared_arrs:
                if shared_array[0] == 1.0: # wait until the client has sent some data
                    active_connections.append(shared_array)
                    start_index = 1
                    for input_name, (arr_len, _, shape) in self.inputs_feed_config.items():
                        input_feeds[input_name].append(shared_array[start_index: start_index + arr_len].reshape(shape))
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
                input_feeds[input_name] = np.array(input_feeds[input_name])
            else:

                input_feeds[input_name] = np.array(input_feeds[input_name]).transpose(transposition)
        # print(input_feeds)
        print(input_feeds["state"].shape)
        print(input_feeds)



        # input_feed = {"inputs": np.array(board_states),
        #                 "input_state": np.array(input_states).transpose([1, 2, 0, 3]),
        #               "input_state_matrix": np.array(input_state_matrices).transpose([1, 0, 2, 3, 4])
        #               }
        # policies,
if __name__ == "__main__":
    batched_inputs_feeds_shape = {"state": [-1, 3]}
    batched_outputs_feeds_shape = {"y": [2, -1]}


    shms = [SharedMemory(create=True, size=(4 * (-np.prod(batched_inputs_feeds_shape["state"]) + 1))) for worker_id in range(2)]

    def task(worker_id, shm):
        inputs_feed_shape = {"state": [1, 3], }# shouldn't have the batch dimension
        outputs_feed_shape = {"y": [1, 2,]}

        sess = Parallelized_Session(worker_id, shm, inputs_feed_shape, outputs_feed_shape)
        sess.run(["y"],
                 {"state": np.ones((1, 3), dtype=np.float32) * 1})
    task(0, shms[0])
    task(1, shms[1])
    print(np.ndarray(shape=(shms[0].size // 4), dtype=np.float32, buffer=shms[0].buf))

    server = Server(batched_inputs_feeds_shape,
                    batched_outputs_feeds_shape,
                    shms,
                    None,
                    "",
                    0,)





    server.start()
    for shm in shms:
        shm.close()
        shm.unlink()



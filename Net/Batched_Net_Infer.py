import tensorflow as tf

# These are to be in replacement of the classes from Batched_Net
# To create a inference model as training is parallelized
# Ask Brian, if you are confused
class Batch(tf.keras.layers.Layer):
    def __init__(self, operation, **kwargs):
        super().__init__(**kwargs)
        self.operation = operation

    def call(self, inputs):
        return self.operation(inputs)
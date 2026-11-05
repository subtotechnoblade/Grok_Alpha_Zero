import tensorflow as tf


def network_input(input_shape, train_config):
    """The model's "inputs" layer in nn_input_dtype, and the float32 tensor the trunk starts from."""
    dtype = train_config.get("nn_input_dtype") or "float32"
    inputs = tf.keras.layers.Input(batch_shape=(None, *input_shape), dtype=dtype, name="inputs")  # the name must be "inputs"
    x = inputs if dtype == "float32" else tf.keras.ops.cast(inputs, "float32")
    return inputs, x

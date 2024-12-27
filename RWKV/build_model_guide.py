import tensorflow as tf
import RWKV_v6, RWKV_v6_infer

# This is just a template
def build_model(input_shape, policy_shape, build_config):
    # Since this is just and example for tictactoe
    # feel free to copy and modify

    embed_size = build_config["embed_size"]
    num_heads = build_config["num_heads"]
    num_layers = build_config["num_layers"]
    token_shift_hidden_dim = build_config["token_shift_hidden_dim"]
    hidden_size = build_config["hidden_size"]

    # input shape should be (3, 3)
    inputs = tf.keras.layers.Input(batch_shape=(None, None, *input_shape))
    # x = inputs
    x = tf.keras.layers.Reshape((-1, input_shape[0] * input_shape[1]))(inputs)
    x = RWKV_v6.Batch_Dense(embed_size)(x)


    for layer_id in range(num_layers):
        x = RWKV_v6.RWKV_Block(layer_id, num_heads, embed_size, token_shift_hidden_dim, hidden_size)(x)

    x = RWKV_v6.Batch_Dense(policy_shape)(x)


    return tf.keras.Model(inputs=inputs, outputs=x)

if __name__ == '__main__':
    from Gomoku.Gomoku import Gomoku, build_config
    print(build_config)
    model = build_model((3, 3), (9,), build_config)
    model.summary()

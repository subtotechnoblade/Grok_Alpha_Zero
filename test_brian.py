import numpy as np
import tensorflow as tf
if __name__ == "__main__":
    def create_model():
        inputs = tf.keras.layers.Input(batch_shape=(None, 1))
        x = tf.keras.layers.Dense(1)(inputs)
        return tf.keras.Model(inputs=inputs, outputs=x)

    model1 = create_model()
    model1.save_weights("model1.weights.h5")

    model2 = create_model()
    model2.load_weights("model1.weights.h5")



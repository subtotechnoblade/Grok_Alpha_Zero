import tensorflow as tf

class SE_Block(tf.keras.layers.Layer):
    def __init__(self, num_filers, broadcast_shape, ratio=2, **kwargs):
        super().__init__(**kwargs)
        self.num_filters = num_filers
        self.broadcast_shape = broadcast_shape
        self.ratio = ratio

        self.squeeze = tf.keras.layers.GlobalAveragePooling3D()
        self.dense1 = tf.keras.layers.Dense(units=self.num_filters / self.ratio)
        self.dense2 = tf.keras.layers.Dense(units=self.num_filters)
    def call(self, inputs):
        squeeze = self.squeeze(inputs)
        excitation = self.dense1(squeeze)
        excitation = tf.keras.layers.Activation("relu")(excitation)
        excitation = self.dense2(excitation)
        excitation = tf.keras.layers.Activation('sigmoid')(excitation)
        excitation = tf.keras.layers.Reshape(self.broadcast_shape)(excitation)
        scale = inputs * excitation
        return scale
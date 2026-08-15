import tensorflow as tf

# in reference to https://zhangtemplar.github.io/resnet/

class ResNet_Block(tf.keras.layers.Layer):
    def __init__(self, filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same", **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides

        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv1 = tf.keras.layers.Conv2D(self.filters, kernel_size=self.kernel_size, strides=self.strides, padding=padding, kernel_initializer='he_normal')

        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(self.filters, kernel_size=self.kernel_size, strides=(1, 1), padding="same", kernel_initializer='he_normal')


        self.use_projection = self.strides != (1, 1)
        self.residual_conv = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), strides=strides, padding=padding, kernel_initializer='he_normal')


    def build(self, input_shape):
        self.use_projection = input_shape[-1] != self.filters
        super().build(input_shape)

    def call(self, inputs):
        residual = inputs
        if self.use_projection:
            residual = self.residual_conv(inputs)

        x = self.bn1(inputs)
        x = tf.keras.layers.Activation("relu")(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = self.conv2(x)

        x += residual
        return x

class Recurrent_ResNet(tf.keras.layers.Layer):
    def __init__(self,  num_layers, filters, kernel_size=(3, 3), recurrence=2, **kwargs):
        super().__init__(**kwargs)
        self.num_layers = num_layers

        self.resnet_layers = [ResNet_Block(filters, kernel_size) for _ in range(self.num_layers)]
        self.recurrence = recurrence
    def singe_call(self, inputs):
        x = inputs
        for i in range(self.num_layers):
            x = self.resnet_layers[i](x)
        return x
    def call(self, inputs):
        x = inputs
        for _ in range(self.recurrence):
            x = self.singe_call(x)
        return x

# class Recurrent_ResNet(tf.keras.layers.Layer):
#     def __init__(self, num_layers, filters, kernel_size=(3, 3), min_recurrence=2, max_recurrence=4, inference=False, **kwargs):
#         super().__init__(**kwargs)
#         self.num_layers = num_layers
#         self.min_recurrence = min_recurrence
#         self.max_recurrence = max_recurrence
#
#         # Create ResNet layers to be reused in each iteration
#         self.resnet_layers = [ResNet_Identity2D(filters, kernel_size) for _ in range(num_layers)]
#
#         self.inference = inference
#
#     def single_call(self, inputs):
#         """Applies all ResNet layers once."""
#         x = inputs
#         for layer in self.resnet_layers:
#             x = layer(x)
#         return x
#
#     def call(self, inputs):
#         """Applies recurrent processing with dynamic iterations during training."""
#         x = inputs
#
#         # Determine number of iterations
#         if not self.inference:
#             # Sample random integer between [min_recurrence, max_recurrence]
#             num_iterations = tf.random.uniform(
#                 shape=[],
#                 minval=self.min_recurrence,
#                 maxval=self.max_recurrence + 1,  # +1 because maxval is exclusive
#                 dtype=tf.int32
#             )
#         else:
#             # Use maximum recurrence during inference
#             num_iterations = self.max_recurrence
#
#         # Dynamic iteration loop
#         def body(i, x):
#             x = self.single_call(x)
#             return i + 1, x
#
#         # Run the loop
#         if not self.inference:
#             _, x = tf.while_loop(
#                 cond=lambda i, _: i < num_iterations,
#                 body=body,
#                 loop_vars=(tf.constant(0, dtype=tf.int32), x)
#             )
#         else:
#             for _ in range(self.max_recurrence):
#                 x = self.single_call(x)
#
#
#         return x

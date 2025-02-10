import tensorflow as tf

# Brian's reimplementation grok fast ema based on https://arxiv.org/pdf/2405.20233
# grok fast github: https://github.com/ironjr/grokfast
class Grok_Fast_EMA_Model(tf.keras.Model):
    def __init__(self, alpha=0.99, lamb=5.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grads = [tf.Variable(tf.zeros_like(var), trainable=False) for var in self.trainable_variables]
        self.grads_updated = [False for _ in range(len(self.grads))]
        self.alpha = alpha
        self.lamb = lamb


    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:

            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compute_loss(y=y, y_pred=y_pred)

            if self.optimizer is not None:
                loss = self.optimizer.scale_loss(loss)

            # gradfilter ema from the grok fast paper and github https://github.com/ironjr/grokfast?tab=readme-ov-file
            trainable_vars = self.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)

            if not all(self.grads_updated):
                for i in range(len(trainable_vars)):
                    if not self.grads_updated[i] and gradients[i] is not None:

                        self.grads[i].assign(tf.convert_to_tensor(gradients[i]))

            updated_gradients = []
            for i in range(len(trainable_vars)):
                if gradients[i] is not None:
                    current_gradients = tf.convert_to_tensor(gradients[i])
                    self.grads[i].assign(self.grads[i].value() * self.alpha + current_gradients * (1 - self.alpha))
                    updated_gradients.append(current_gradients + self.grads[i] * self.lamb)
                else:
                    updated_gradients.append(gradients[i])
        # Update weights
        self.optimizer.apply_gradients(zip(updated_gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)

        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)

        return self.compute_metrics(x, y, y_pred, loss)

class Ortho_Model(tf.keras.Model):
    def __init__(self, alpha=0.99, lamb=5.0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.alpha = alpha
        self.lamb = lamb

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:

            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compute_loss(y=y, y_pred=y_pred)

            if self.optimizer is not None:
                loss = self.optimizer.scale_loss(loss)

            # gradfilter ema from the grok fast paper and github https://github.com/ironjr/grokfast?tab=readme-ov-file
            gradients = tape.gradient(loss, self.trainable_variables)

        updated_gradients = []
        for i in range(len(self.trainable_variables)):
            if gradients[i] is not None:
                current_gradients = tf.convert_to_tensor(gradients[i])
                g = tf.reshape(current_gradients, (-1,))
                w = tf.reshape(self.trainable_variables[i], (-1,))

                w_norm_sq = tf.tensordot(w, w, axes=1) + 1e-30
                proj = tf.tensordot(w, g, axes=1) / w_norm_sq
                g_orth = g - proj * w

                g_norm = tf.norm(g, 2)
                g_orth_norm = tf.norm(g_orth, 2) + 1e-30
                g_orth_scaled = g_orth * (g_norm / g_orth_norm)

                updated_gradients.append(tf.reshape(g_orth_scaled, current_gradients.shape))
            else:
                updated_gradients.append(gradients[i])
        # Update weights
        self.optimizer.apply_gradients(zip(updated_gradients, self.trainable_variables))
        # Update metrics (includes the metric that tracks the loss)

        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)

        return self.compute_metrics(x, y, y_pred, loss)


class Ortho_Grok_Fast_EMA_Model(tf.keras.Model):
    def __init__(self, alpha=0.99, lamb=5.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grads = [tf.Variable(tf.zeros_like(var), trainable=False) for var in self.trainable_variables]
        self.grads_updated = [False for _ in range(len(self.grads))]
        self.alpha = alpha
        self.lamb = lamb

    def compute_ortho_grads(self, gradients):
        for i in range(len(self.trainable_variables)):
            if gradients[i] is not None:
                current_gradients = tf.convert_to_tensor(gradients[i])
                g = tf.reshape(current_gradients, (-1,))
                w = tf.reshape(self.trainable_variables[i], (-1,))

                w_norm_sq = tf.tensordot(w, w, axes=1) + 1e-25
                proj = tf.tensordot(w, g, axes=1) / w_norm_sq
                g_orth = g - proj * w

                g_norm = tf.norm(g, 2)
                g_orth_norm = tf.norm(g_orth, 2) + 1e-25
                g_orth_scaled = g_orth * (g_norm / g_orth_norm)
                gradients[i] = tf.reshape(g_orth_scaled, current_gradients.shape)

        return gradients

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:

            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compute_loss(y=y, y_pred=y_pred)

            if self.optimizer is not None:
                loss = self.optimizer.scale_loss(loss)

            # gradfilter ema from the grok fast paper and github https://github.com/ironjr/grokfast?tab=readme-ov-file
            trainable_vars = self.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)
            gradients = self.compute_ortho_grads(gradients)

            if not all(self.grads_updated):
                for i in range(len(trainable_vars)):
                    if not self.grads_updated[i] and gradients[i] is not None:

                        self.grads[i].assign(tf.convert_to_tensor(gradients[i]))

            # updated_gradients = []
            for i in range(len(trainable_vars)):
                if gradients[i] is not None:
                    current_gradients = tf.convert_to_tensor(gradients[i])
                    self.grads[i].assign(self.grads[i].value() * self.alpha + current_gradients * (1 - self.alpha))
                    gradients[i] = current_gradients + self.grads[i] * self.lamb
                #     updated_gradients.append(current_gradients + self.grads[i] * self.lamb)
                # else:
                #     updated_gradients.append(gradients[i])
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # self.optimizer.apply_gradients(zip(updated_gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)

        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)

        return self.compute_metrics(x, y, y_pred, loss)
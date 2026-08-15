import tensorflow as tf
from keras.src import ops
from keras.src.optimizers import optimizer
import re


# based on https://kellerjordan.github.io/posts/muon/
class Muon(optimizer.Optimizer):
    def __init__(self,
                 learning_rate=1e-3,
                 adam_lr_ratio=1.0,
                 use_nadam=True,
                 weight_decay=0.004,
                 exclude_layers=[],
                 exclude_embeddings=True,
                 nesterov=True,  # uses nesterov momentum in muon
                 adam_beta_1=0.90,
                 adam_beta_2=0.995,
                 muon_beta=0.95,  # muon momentum
                 muon_a=3.4445,
                 muon_b=-4.7750,
                 muon_c=2.0315,
                 ns_steps=5,
                 epsilon=1e-8,
                 name="Muon",
                 **kwargs):
        super().__init__(learning_rate=learning_rate, name=name, **kwargs)
        self.adam_lr_ratio = adam_lr_ratio
        self.use_nadam = use_nadam
        # NOT self.weight_decay. Keras' BaseOptimizer runs its own _apply_weight_decay()
        # inside apply() whenever self.weight_decay is not None, which would decay twice on
        # top of the manual apply_weight_decay() in update_step below. The manual one is the
        # one to keep, because it is the only path that still runs when this optimizer is
        # wrapped by Orthograd/Grokfast (then only the wrapper's apply() fires, never ours).
        self.muon_weight_decay = weight_decay

        self.exclude_layers = exclude_layers
        self.exclude_embeddings = exclude_embeddings

        self.nesterov = nesterov
        self.adam_beta_1 = adam_beta_1
        self.adam_beta_2 = adam_beta_2

        self.muon_beta = muon_beta
        self.muon_a = muon_a
        self.muon_b = muon_b
        self.muon_c = muon_c

        self.ns_steps = ns_steps
        self.epsilon = epsilon

    def _should_use_adamw(self, variable):
        """Determine if AdamW should be used for this variable."""
        # Check dimensionality - any {0,1}-D parameters should use AdamW
        if len(variable.shape) < 2:
            return True

        var_name = variable.path.lower()
        if "bias" in var_name:
            return True

        # Check if variable is from embedding layer (by name pattern)
        if self.exclude_embeddings:
            if 'embedding' in var_name:
                return True

        # Check if variable matches any exclude layer patterns
        for pattern in self.exclude_layers:
            if re.search(pattern, variable.path):
                return True

        # Default to Muon for 2D, 3D, 4D variables that don't match exclusions
        return False

    def build(self, variables):
        super().build(variables)
        self.adam_momentums = []
        self.adam_velocities = []

        self.muon_momentums = []
        for var in variables:
            if self._should_use_adamw(var):
                self.adam_momentums.append(self.add_variable_from_reference(var, "adam_momentum"))
                self.adam_velocities.append(self.add_variable_from_reference(var, "adam_velocity"))
                self.muon_momentums.append(None)

            else:
                self.muon_momentums.append(self.add_variable_from_reference(var, "muon_momentum"))
                self.adam_momentums.append(None)
                self.adam_velocities.append(None)

    def get_scaling_factor(self, shape):
        # Moonshot/Moonlight update scale: gives every layer per-element RMS ~0.2
        d_out = float(shape[-1])
        d_in = 1.0
        for dim in shape[:-1]:
            d_in *= float(dim)
        return 0.2 * max(d_out, d_in) ** 0.5

    def apply_weight_decay(self, variable, learning_rate):
        # Moonshot-exact decoupled decay: w *= (1 - lr*wd) with the NOMINAL branch lr.
        # No moonshot scaling factor on the decay (only the update gets scaled), no
        # dimensionality exemption (their reference decays the adamw-routed 1-D params
        # too), and it is applied BEFORE the update, matching p.data.mul_(1 - lr*wd).
        weight_decay = ops.cast(self.muon_weight_decay, dtype=variable.dtype)
        learning_rate = ops.cast(learning_rate, dtype=variable.dtype)
        variable.assign_sub(variable * (learning_rate * weight_decay))

    def auto_transpose(self, X):  # transposes the last two axis regardless of the dim in front
        if X.ndim == 2:
            return ops.transpose(X, axes=[1, 0])
        ndim = X.ndim
        return ops.transpose(X, axes=[*range(ndim - 2), ndim - 1, ndim - 2])

    def zeropower_via_newtonschulz5(self, G, epsilon=1e-7):  # from KellerJordan's implementation
        tf.debugging.assert_greater_equal(tf.rank(G), 2)
        X = ops.cast(G, tf.bfloat16)

        is_transposed = False
        if G.shape[-2] > G.shape[-1]:
            X = self.auto_transpose(X)
            is_transposed = True

        norm = tf.norm(X, ord="fro", axis=[-2, -1], keepdims=True) + epsilon
        X = X / norm

        for _ in range(self.ns_steps):
            X_t = ops.transpose(X, [1, 0])
            A = ops.matmul(X, X_t)
            B = self.muon_b * A + self.muon_c * (ops.matmul(A, A))
            X = self.muon_a * X + ops.matmul(B, X)

        if is_transposed:
            X = self.auto_transpose(X)

        return ops.cast(X, tf.float32)

    def muon_update(self, variable, gradient, learning_rate):
        muon_beta = ops.cast(self.muon_beta, variable.dtype)

        muon_m = self.muon_momentums[self._get_variable_index(variable)]
        muon_m_update = muon_beta * muon_m + gradient * (1 - muon_beta)
        muon_m.assign(muon_m_update)

        update = (1 - muon_beta) * gradient + muon_beta * muon_m_update if self.nesterov else muon_m_update

        is_reshaped = False
        if update.ndim > 2:
            ndim = len(update.shape)
            update = ops.transpose(update, axes=[ndim - 1, *range(ndim - 1)])
            shape = update.shape  # (out_channels, kh, kw, in_channels)
            update = ops.reshape(update, (shape[0], -1))
            is_reshaped = True

        update = self.zeropower_via_newtonschulz5(update)
        update *= self.get_scaling_factor(gradient.shape)

        if is_reshaped:
            update = ops.reshape(update, shape)
            update = ops.transpose(update, axes=[*range(1, ndim), 0])

        final_u = update * learning_rate

        return final_u

    def adam_update(self, variable, gradient, learning_rate):
        epsilon = ops.cast(self.epsilon, variable.dtype)
        its = ops.cast(self.iterations + 1, variable.dtype)

        adam_beta_1 = ops.cast(self.adam_beta_1, variable.dtype)
        adam_beta_2 = ops.cast(self.adam_beta_2, variable.dtype)
        m = self.adam_momentums[self._get_variable_index(variable)]
        v = self.adam_velocities[self._get_variable_index(variable)]

        m_update = adam_beta_1 * m + (1 - adam_beta_1) * gradient

        v_update = adam_beta_2 * v + (1 - adam_beta_2) * ops.square(gradient)

        # bias correction
        beta_1_t = ops.power(adam_beta_1, its)
        beta_2_t = ops.power(adam_beta_2, its)

        m_hat = m_update / (1.0 - beta_1_t)
        v_hat = v_update / (1.0 - beta_2_t)

        if self.use_nadam:
            nesterov = (m_hat * adam_beta_1) + ((1.0 - adam_beta_1) * gradient) / (1.0 - beta_1_t)
            u = nesterov / (ops.sqrt(v_hat) + epsilon)
        else:
            u = m_hat / (ops.sqrt(v_hat) + epsilon)

        final_u = u * learning_rate

        m.assign(m_update)
        v.assign(v_update)
        return final_u

    def update_step(self, gradient, variable, learning_rate):
        if gradient is None:
            return
        muon_learning_rate = ops.cast(learning_rate, variable.dtype)
        adam_learning_rate = self.adam_lr_ratio * muon_learning_rate

        if self.muon_momentums[self._get_variable_index(variable)] is not None:
            update = self.muon_update(variable, gradient, muon_learning_rate)
            self.apply_weight_decay(variable, muon_learning_rate)
            variable.assign_sub(update)
        else:
            update = self.adam_update(variable, gradient, adam_learning_rate)
            self.apply_weight_decay(variable, adam_learning_rate)
            variable.assign_sub(update)

    def get_config(self):
        config = super().get_config()
        config.update({
            "learning_rate": self.learning_rate,
            "adam_lr_ratio": self.adam_lr_ratio,
            "use_nadam": self.use_nadam,
            "nesterov": self.nesterov,
            "weight_decay": self.muon_weight_decay,
            "exclude_layers": self.exclude_layers,
            "exclude_embeddings": self.exclude_embeddings,

            "adam_beta_1": self.adam_beta_1,
            "adam_beta_2": self.adam_beta_2,

            "muon_beta": self.muon_beta,
            "muon_a": self.muon_a,
            "muon_b": self.muon_b,
            "muon_c": self.muon_c,
            "ns_steps": self.ns_steps,
            "epsilon": self.epsilon,
        })
        return config

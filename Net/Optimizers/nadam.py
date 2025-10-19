from keras.src import ops
from keras.src.optimizers import optimizer
class Nadam(optimizer.Optimizer):
    def __init__(self,
                 learning_rate=1e-3,
                 beta_1=0.9,
                 beta_2=0.995,
                 weight_decay=0.004,
                 caution=True,
                 epsilon=1e-8,
                 name="adam",
                 **kwargs):
        super().__init__(learning_rate=learning_rate,
                         weight_decay=weight_decay,
                         name=name, **kwargs)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        self.caution = caution


    def build(self, variables):
        super().build(variables)

        self._momentums = []
        self._velocities = []
        for var in variables:
            self._momentums.append(self.add_variable_from_reference(var, "momentum"))
            self._velocities.append(self.add_variable_from_reference(var, "velocity"))


    def update_step(self, gradient, variable, learning_rate):
        if gradient is None:
            return

        learning_rate = ops.cast(learning_rate, variable.dtype)
        beta_1 = ops.cast(self.beta_1, variable.dtype)
        beta_2 = ops.cast(self.beta_2, variable.dtype)
        epsilon = ops.cast(self.epsilon, variable.dtype)

        its = ops.cast(self.iterations + 1, variable.dtype)


        m = self._momentums[self._get_variable_index(variable)]
        v = self._velocities[self._get_variable_index(variable)]

        m_update = beta_1 * m + (1 - beta_1) * gradient

        v_update = beta_2 * v + (1 - beta_2) * ops.square(gradient)

        #bias correction
        beta_1_t = ops.power(beta_1, its)
        beta_2_t = ops.power(beta_2, its)

        m_hat = m_update / (1.0 - beta_1_t)
        v_hat = v_update / (1.0 - beta_2_t)

        nesterov = (m_hat * beta_1) + ((1.0 - beta_1) * gradient) / (1.0 - beta_1_t)
        u = nesterov / (ops.sqrt(v_hat) + epsilon)

        if self.caution:
            mask = ops.where(u * gradient > 0, 1.0, 0.0)
            scaled_lr = learning_rate * (mask / (ops.mean(mask) + epsilon))
            final_u = u * mask * scaled_lr
        else:
            final_u = u * learning_rate

        variable.assign_sub(final_u)

        m.assign(m_update)
        v.assign(v_update)

    def get_config(self):
        config = super().get_config()
        config.update({
            "learning_rate": self.learning_rate,
            "beta_1": self.beta_1,
            "beta_2": self.beta_2,
            "epsilon": self.epsilon,
            "caution": self.caution
        })
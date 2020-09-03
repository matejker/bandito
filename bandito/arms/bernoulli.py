import numpy as np

from bandito import Arm
import bandito.arms.exceptions as ex


class Bernoulli(Arm):
    def __int__(self, p: float = 0.5, a: float = 0, b: float = 1, **kwargs):
        if p < 0 or 1 < p:
            raise ex.BernoulliProbabilityBeyondBounds(f"Probability p={p} has to be in [0, 1]")
        if b < a:
            raise ex.BernoulliBounds(f"Bounds have to be a={a} <= b={b}]")

        super().__init__(**kwargs)
        self.mu = p
        self.x_temp = (np.random.uniform(size=self.t_max) > p) * (b - a) + a

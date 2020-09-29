import numpy as np

from bandito.arms.arm import Arm
import bandito.arms.exceptions as ex


class Bernoulli(Arm):
    def __init__(self, t_max: int, p: float = 0.5, a: float = 0, b: float = 1, **kwargs):
        if p < 0 or 1 < p:
            raise ex.BernoulliProbabilityBeyondBounds(f"Probability p={p} has to be in [0, 1]")
        if b < a:
            raise ex.BernoulliBounds(f"Bounds have to be a={a} <= b={b}]")

        super().__init__(t_max, **kwargs)
        self.mu = p * (b - a) + a
        self.x_temp = (np.random.uniform(size=self.t_max) < p) * (b - a) + a

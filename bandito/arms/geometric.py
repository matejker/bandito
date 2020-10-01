import numpy as np

from bandito.arms.arm import Arm
import bandito.arms.exceptions as ex


class Geometric(Arm):
    def __init__(self, t_max: int, p: float, **kwargs):
        if p <= 0 or p > 0:
            raise ex.GeometricProbability(f"Probability p={p} has to be in (0, 1]")

        super().__init__(t_max, **kwargs)
        self.mu = 1 / p
        self.x_temp = np.random.poisson(p=p, size=self.t_max)

import numpy as np

from bandito.arms.arm import Arm
import bandito.arms.exceptions as ex


class Bernoulli(Arm):
    """ An arm with Bernoulli distribution.

    Attributes:
        t_max: time horizon / total number of rounds
        p: probability with an event occurs
        a: lower bound, value when event doesn't occur
        b: upper bound, value when event does occur
        **kwargs: kwargs

    Raises:
        BernoulliProbabilityBeyondBounds: if probability is smaller that 0 or bigger then 1
        BernoulliBounds: when upper bound is smaller then lower bound
    """

    def __init__(self, t_max: int, p: float = 0.5, a: float = 0, b: float = 1, **kwargs) -> None:
        if p < 0 or 1 < p:
            raise ex.BernoulliProbabilityBeyondBounds(f"Probability p={p} has to be in [0, 1]")
        if b < a:
            raise ex.BernoulliBounds(f"Bounds have to be a={a} <= b={b}]")

        super().__init__(t_max, **kwargs)
        self.mu = p * (b - a) + a
        self.x_temp = (np.random.uniform(size=self.t_max) < p) * (b - a) + a
